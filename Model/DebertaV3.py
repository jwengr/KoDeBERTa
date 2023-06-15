import gc
import torch
import torch.nn as nn
import deepspeed
import pytorch_lightning as pl

from transformers import DebertaV2Config, DebertaV2ForMaskedLM, DebertaV2ForTokenClassification


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad_(False)

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad_(True)

class LitDebertaV3ForPretrainingWithDeepSpeed(pl.LightningModule):
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
    
    def forward_discriminator(self, inputs_embeds, attention_mask, masked_ids):
        labels = torch.zeros_like(masked_ids, dtype=torch.long, device=masked_ids.device, requires_grad=False)
        labels[masked_ids==self.hparams.mask_id]=1
        labels[masked_ids==self.hparams.pad_id]=-100
        loss = self.discriminator(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels).loss
        del labels
        return loss

    def training_step(self, batch, batch_idx):
        label_ids = batch['label_ids']
        masked_ids = batch['masked_ids']
        attention_mask = batch['attention_mask']
        
        unfreeze_model(self.generator)
        self.generator.zero_grad()
        pred_ids, loss_generator = self.forward_generator(masked_ids=masked_ids, attention_mask=attention_mask, label_ids=label_ids)
        pred_ids = pred_ids.detach()
        self.generator_engine.backward(loss_generator)
        self.generator_engine.step()
        freeze_model(self.generator)

        unfreeze_model(self.discriminator)
        self.discriminator.zero_grad()
        inputs_embeds = self.generator.deberta.embeddings.word_embeddings(pred_ids).detach() + self.discriminator.deberta.embeddings.word_embeddings(pred_ids)
        loss_discriminator = self.forward_discriminator(inputs_embeds=inputs_embeds, attention_mask=attention_mask, masked_ids=masked_ids)
        loss_discriminator = loss_discriminator * 50
        self.discriminator_engine.backward(loss_discriminator)
        self.discriminator_engine.step()
        freeze_model(self.discriminator)

        self.log_dict({"Loss_G": loss_generator, "Loss_D": loss_discriminator}, on_step=True, on_epoch=False, prog_bar=True)

        if self.global_step>0 and self.global_step%(self.hparams.max_steps//20)==0:
            torch.save(self.generator.state_dict(), f'generator_step={self.global_step}_loss={loss_generator.item()}.pth')
            torch.save(self.discriminator.state_dict(), f'discriminator_step={self.global_step}_loss={loss_discriminator.item()}.pth')
            self.generator_engine.save_checkpoint(self.hparams.save_dir, loss_generator.item(), client_sd = self.global_step)
            self.discriminator_engine.save_checkpoint(self.hparams.save_dir, loss_discriminator.item(), client_sd = self.global_step)

        gc.collect()
        torch.cuda.empty_cache()
        
    def configure_optimizers(self):
        return

