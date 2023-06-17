import json
import argparse
import deepspeed
import pytorch_lightning as pl

from tokenizers import Tokenizer
from datasets import IterableDataset
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

from Data.DataCollator import DataCollatorForHFUnigramSpanMLM
from Model.DebertaV3 import LitDebertaV3ForPretrainingWithDeepSpeed

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
    SEED=args.seed
    pl.seed_everything(SEED)
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    mask_id = tokenizer.get_vocab()[args.mask_token]
    pad_id = tokenizer.get_vocab()[args.pad_token]

    batch_size = args.batch_size
    max_steps = args.max_steps
    if max_steps==-1:
        max_steps = 500_000 * (8192//batch_size)

    def gen():
        with open(args.datapath, encoding=args.encoding) as f:
            for line in f:
                yield line

    ds = IterableDataset.from_generator(gen)
    ds.shuffle(seed=SEED, buffer_size=8_800_000)
    if args.collate_fn == 'DataCollatorForHFUnigramSpanMLM':
        collate_fn = DataCollatorForHFUnigramSpanMLM(tokenizer, truncation_argument={'max_length':args.max_length}, mask_prob=args.mask_prob)
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)

    debertav3_pretrainer = LitDebertaV3ForPretrainingWithDeepSpeed(model_name=args.model_name, mask_id=mask_id, pad_id=pad_id, max_steps=max_steps, save_dir=args.save_dir, ds_config=args.deepspeed_config)

    logger = TensorBoardLogger(args.log_dir, name="LitDebertaV3ForPretrainingWithDeepSpeed_DataCollatorForHFUnigramSpanMLM")
                               
    trainer = pl.Trainer(
        accelerator=args.pl_accelerator,
        max_steps=max_steps,
        logger=logger
    )

    trainer.fit(debertav3_pretrainer,dl)

if __name__ == "__main__":
    args = get_args()
    train(args)