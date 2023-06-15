import gc
import argparse
import torch
import torch.nn as nn
import deepspeed
import pytorch_lightning as pl

from copy import deepcopy
from tokenizers import Tokenizer
from datasets import IterableDataset
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import DebertaV2Config, DebertaV2ForMaskedLM, DebertaV2ForTokenClassification

from Data.DataCollator import DataCollatorForHFUnigramSpanMLM

class DebertaV3ForPretraining(pl.LightningModule):
    def __init__(self, model_name, mask_id, pad_id, max_steps, save_dir, ds_config):
        super().__init__()
        self.save_hyperparameters()
        
        self.generator_config = DebertaV2Config.from_pretrained(self.hparams.model_name)
        self.generator_config.num_hidden_layers = self.generator_config.num_hidden_layers // 2
        self.generator = DebertaV2ForMaskedLM(config=self.generator_config)
        model_engine, _, _, _ = deepspeed.initialize(model=self.generator, model_parameters=self.generator.parameters(), config=self.hparams.ds_config)
        self.generator_engine = model_engine
        
        self.discriminator_config = DebertaV2Config.from_pretrained(self.hparams.model_name)
        self.discriminator_config.num_labels = 2
        self.discriminator = DebertaV2ForTokenClassification(config=self.discriminator_config)
        model_engine, _, _, _ = deepspeed.initialize(model=self.discriminator, model_parameters=self.discriminator.parameters(), config=self.hparams.ds_config)
        self.discriminator_engine = model_engine
        
        self.automatic_optimization = False

    def discriminator_postprocessing(self):
        self.discriminator.deberta.embeddings.word_embeddings.weight = nn.Parameter(
            self.generator.deberta.embeddings.word_embeddings.weight.detach() + self.discriminator.deberta.embeddings.word_embeddings.weight.detach(),
            requires_grad=True
        )

    def forward_generator(self, masked_ids, attention_mask, label_ids):
        labels = torch.where(masked_ids == self.hparams.mask_id, label_ids, -100).type_as(masked_ids)
        outputs = self.generator(input_ids=masked_ids, attention_mask=attention_mask, labels=labels)
        logits, loss = outputs.logits, outputs.loss
        pred_ids = logits.argmax(dim=-1)
        del labels
        return pred_ids, loss
    
    def forward_discriminator(self, inputs_embeds, attention_mask, masked_ids, label_ids):
        labels = torch.zeros_like(masked_ids, dtype=torch.long, device=masked_ids.device, requires_grad=False)
        labels[masked_ids==self.hparams.mask_id]=1
        labels[label_ids==self.hparams.pad_id]=-100
        loss = self.discriminator(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels).loss
        del labels
        return loss

    def training_step(self, batch, batch_idx):
        label_ids = batch['label_ids']
        masked_ids = batch['masked_ids']
        attention_mask = batch['attention_mask']
        
        self.generator.zero_grad()
        pred_ids, loss_generator = self.forward_generator(masked_ids=masked_ids, attention_mask=attention_mask, label_ids=label_ids)
        pred_ids = pred_ids.detach()
        self.generator_engine.backward(loss_generator)
        self.generator_engine.step()

        self.discriminator.zero_grad()
        inputs_embeds = self.generator.deberta.embeddings.word_embeddings(pred_ids).detach() + self.discriminator.deberta.embeddings.word_embeddings(pred_ids)
        loss_discriminator = self.forward_discriminator(inputs_embeds=inputs_embeds, attention_mask=attention_mask, masked_ids=masked_ids, label_ids=label_ids)
        loss_discriminator = loss_discriminator * 50
        self.discriminator_engine.backward(loss_discriminator)
        self.discriminator_engine.step()

        self.log_dict({"Loss_G": loss_generator, "Loss_D": loss_discriminator}, on_step=True, on_epoch=False, prog_bar=True)

        if self.global_step>0 and self.global_step%(self.hparams.max_steps//20)==0:
            self.generator_engine.save_checkpoint(self.hparams.save_dir, loss_generator.item(), client_sd = self.global_step)
            self.discriminator_engine.save_checkpoint(self.hparams.save_dir, loss_discriminator.item(), client_sd = self.global_step)

        gc.collect()
        torch.cuda.empty_cache()


def get_argument_parser():
    parser = argparse.ArgumentParser(description="DeBERTaV3")
    parser.add_argument('--datapath', type=str)
    parser.add_argument('--encoding', type=str, default='utf-8-sig')
    parser.add_argument('--modelname', type=str)
    parser.add_argument('--maxlength', type=int, default=512)
    parser.add_argument('--batchsize', type=int, default=4)
    parser.add_argument('--maxsteps', type=int, default=-1)
    parser.add_argument('--warmupsteps', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--buffersize', type=int, default=8_800_000)
    parser.add_argument('--tokenizerpath', type=str)
    parser.add_argument('--masktoken', type=str, default='[MASK]')
    parser.add_argument('--padtoken', type=str, default='[PAD]')
    parser.add_argument('--dataloader', type=str)
    parser.add_argument('--mask_prob', type=float, default=0.15)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--savedir', type=str)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

def add_ds_config(args):
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batchsize,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-7
            }
        },
        "fp16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 1,
            "offload_optimizer": {
                "device": "cpu"
            }
        }
    }
    setattr(args,'ds_config',ds_config)
    return args

def train(args):
    SEED=args.seed
    pl.seed_everything(SEED)
    tokenizer = Tokenizer.from_file(args.tokenizerpath)
    mask_id = tokenizer.get_vocab()[args.masktoken]
    pad_id = tokenizer.get_vocab()[args.padtoken]

    batch_size = args.batchsize
    max_steps = args.max_steps
    warmup_steps = args.warmupsteps
    if max_steps==-1:
        max_steps = 500_000 * (8192//batch_size)
    if warmup_steps==-1:
        warmup_steps = int(max_steps*0.08)

    def gen():
        with open(args.datapath, encoding=args.encoding) as f:
            for line in f:
                yield line

    ds = IterableDataset.from_generator(gen)
    ds.shuffle(seed=SEED, buffer_size=8_800_000)
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=DataCollatorForHFUnigramSpanMLM(tokenizer, truncation_argument={'max_length':512}))

    debertav3_pretrainer = DebertaV3ForPretraining('microsoft/deberta-v3-xsmall', mask_id=mask_id, pad_id=pad_id, max_steps=max_steps, save_dir=args.save_dir, ds_config=args.ds_config)

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        max_steps=max_steps,
    )

    trainer.fit(debertav3_pretrainer,dl)


if __name__ == "__main__":
    args = get_argument_parser()
    args = add_ds_config(args)
    train(args)