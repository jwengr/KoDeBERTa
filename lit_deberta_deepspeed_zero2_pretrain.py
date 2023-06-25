import os
import json
import argparse
import deepspeed
import pytorch_lightning as pl

from tokenizers import Tokenizer
from datasets import IterableDataset
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

from Data.DataCollator import DataCollatorForHFUnigramSpanMLM
from Model.DebertaV3.DebertaV3 import LitDebertaV3ForPretrainingWithDeepSpeed

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
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    mask_id = tokenizer.get_vocab()[args.mask_token]
    pad_id = tokenizer.get_vocab()[args.pad_token]

    if not os.path.exists(args.generator_save_dir):
        os.makedirs(args.generator_save_dir)
    if not os.path.exists(args.discriminator_save_dir):
        os.makedirs(args.discriminator_save_dir)

    def gen():
        for data_path in args.data_paths:
            with open(data_path, encoding=args.encoding) as f:
                for line in f:
                    yield line

    ds = IterableDataset.from_generator(gen)
    if args.shuffle:
        ds.shuffle(seed=args.seed, buffer_size=args.buffer_size)
    ds = ds.skip(args.current_step * args.batch_size)
    if args.collate_fn == 'DataCollatorForHFUnigramSpanMLM':
        collate_fn = DataCollatorForHFUnigramSpanMLM(tokenizer, truncation_argument={'max_length':args.max_length}, mask_prob=args.mask_prob)
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn)

    debertav3_pretrainer = LitDebertaV3ForPretrainingWithDeepSpeed(
        ds_config=args.deepspeed_config,
        model_name=args.model_name,
        mask_id=mask_id, 
        pad_id=pad_id, 
        current_step=args.current_step,
        max_steps=args.max_steps, 
        save_per_steps=args.save_per_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        generator_save_dir=args.generator_save_dir,
        discriminator_save_dir=args.discriminator_save_dir,
        load_pretrained=args.load_pretrained,
        generator_checkpoint_id=args.generator_checkpoint_id,
        discriminator_checkpoint_id=args.discriminator_checkpoint_id
    )

    logger = TensorBoardLogger(args.log_dir, name=args.log_name, version=args.log_version, purge_step=args.current_step)
                               
    trainer = pl.Trainer(
        accelerator=args.pl_accelerator,
        max_steps=args.max_steps - args.current_step,
        logger=logger,
    )

    trainer.fit(debertav3_pretrainer,dl)

if __name__ == "__main__":
    args = get_args()
    train(args)