import json
import argparse
import deepspeed
import pytorch_lightning as pl

from tokenizers import Tokenizer
from datasets import IterableDataset
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

from Model.DebertaV3 import LitDebertaV3ForPretrainingWithDeepSpeed
from Data.DataCollator import DataCollatorForHFUnigramSpanMLM

def get_args():
    parser = argparse.ArgumentParser(description="DeBERTaV3")
    parser.add_argument('--general_config', type=str)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    with open(args.general_config, 'r') as f:
        general_config = json.load(f)
        for key, value in general_config.items():
            setattr(args, key, value)

    return args

def train(args):
    SEED=args.seed
    pl.seed_everything(SEED)
    tokenizer = Tokenizer.from_file(args.tokenizerpath)
    mask_id = tokenizer.get_vocab()[args.masktoken]
    pad_id = tokenizer.get_vocab()[args.padtoken]

    batch_size = args.batchsize
    max_steps = args.maxsteps
    if max_steps==-1:
        max_steps = 500_000 * (8192//batch_size)

    def gen():
        with open(args.datapath, encoding=args.encoding) as f:
            for line in f:
                yield line

    ds = IterableDataset.from_generator(gen)
    ds.shuffle(seed=SEED, buffer_size=8_800_000)
    if args.collatefn == 'DataCollatorForHFUnigramSpanMLM':
        collate_fn = DataCollatorForHFUnigramSpanMLM(tokenizer, truncation_argument={'max_length':args.maxlength}, mask_prob=args.maskprob)
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=DataCollatorForHFUnigramSpanMLM(tokenizer, collate_fn))

    debertav3_pretrainer = LitDebertaV3ForPretrainingWithDeepSpeed(model_name=args.modelname, mask_id=mask_id, pad_id=pad_id, max_steps=max_steps, save_dir=args.savedir, ds_config=args.deepspeed_config)

    logger = TensorBoardLogger(args.logdir, name="LitDebertaV3ForPretrainingWithDeepSpeed_DataCollatorForHFUnigramSpanMLM")
                               
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        max_steps=max_steps,
        logger=logger
    )

    trainer.fit(debertav3_pretrainer,dl)

if __name__ == "__main__":
    args = get_args()
    train(args)