import os
import json
import argparse
import torch
import deepspeed
import pytorch_lightning as pl

from tokenizers import Tokenizer
from datasets import IterableDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from google.cloud import storage
from google.oauth2 import service_account

from Data.DataCollator import DataCollatorForHFUnigramSpanMLM
from Model.DebertaV3.DebertaV3 import LitDebertaV3ForPretrainingWithDeepSpeedDistributed

def get_args():
    parser = argparse.ArgumentParser(description="DeBERTaV3")
    parser.add_argument('--general_config_path', type=str)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    with open(args.general_config_path, 'r') as f:
        general_config = json.load(f)
        for key, value in general_config.items():
            setattr(args, key, value)

    return args

def train(args):
    pl.seed_everything(args.seed)

    credentials = service_account.Credentials.from_service_account_file(args.gcp_key_path)
    client = storage.Client(credentials = credentials, project = credentials.project_id)
    gcp_bucket = client.get_bucket(args.gcp_bucket_name)
    gcp_bucket.download_blob(args.tokenizer_path).download_to_filename(args.tokenizer_path)
    for data_path in args.data_paths:
        gcp_bucket.download_blob(data_path).download_to_filename(data_path)

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    mask_id = tokenizer.get_vocab()[args.mask_token]
    pad_id = tokenizer.get_vocab()[args.pad_token]

    log_path = os.path.join(args.log_dir, args.log_name)

    if not os.path.exists(args.generator_save_dir):
        os.makedirs(args.generator_save_dir)
    if not os.path.exists(args.discriminator_save_dir):
        os.makedirs(args.discriminator_save_dir)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    def gen():
        for data_path in args.data_paths:
            with open(data_path, encoding=args.encoding) as f:
                for line in f:
                    yield line

    ds = IterableDataset.from_generator(gen)
    if args.shuffle:
        ds.shuffle(seed=args.seed, buffer_size=args.buffer_size)
    ds = ds.skip(max(0,args.current_step-1) * args.batch_size)
    if args.collate_fn == 'DataCollatorForHFUnigramSpanMLM':
        collate_fn = DataCollatorForHFUnigramSpanMLM(tokenizer, truncation_argument={'max_length':args.max_length}, mask_prob=args.mask_prob)
    sampler = DistributedSampler(ds, shuffle=False)
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn, sampler=sampler)

    torch.set_float32_matmul_precision(args.torch_float32_matmul_precision)

    debertav3_pretrainer = LitDebertaV3ForPretrainingWithDeepSpeedDistributed(
        ds_config=args.deepspeed_config,
        model_name=args.model_name,
        mask_id=mask_id, 
        pad_id=pad_id, 
        current_step=args.current_step,
        max_steps=args.max_steps, 
        log_path=log_path,
        log_per_steps=args.log_per_steps,
        save_per_steps=args.save_per_steps,
        gcp_bucket=gcp_bucket,
        gradient_checkpointing=args.gradient_checkpointing,
        generator_save_dir=args.generator_save_dir,
        discriminator_save_dir=args.discriminator_save_dir,
        load_pretrained=args.load_pretrained,
        generator_checkpoint_id=args.generator_checkpoint_id,
        discriminator_checkpoint_id=args.discriminator_checkpoint_id,
    )
                               
    trainer = pl.Trainer(
        accelerator=args.pl_accelerator,
        max_steps=args.max_steps - args.current_step,
    )

    trainer.fit(debertav3_pretrainer,dl)

if __name__ == "__main__":
    args = get_args()
    train(args)