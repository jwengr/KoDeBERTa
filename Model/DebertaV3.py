import gc
import torch
import torch.nn as nn
import pytorch_lightning as pl

from copy import deepcopy
from torch.optim import AdamW
from transformers import DebertaV2Config, DebertaV2ForMaskedLM, DebertaV2ForTokenClassification
from transformers import get_linear_schedule_with_warmup

class DebertaV3ForPretraining(pl.LightningModule):
    def __init__(self, model_name, mask_id, pad_id, lr=1e-6, num_warmup_steps=10_000, num_training_steps=125_000):
        super().__init__()
        self.save_hyperparameters()
        
        self.generator_config = DebertaV2Config.from_pretrained(self.hparams.model_name)
        self.generator_config.num_hidden_layers = self.generator_config.num_hidden_layers // 2
        self.generator = DebertaV2ForMaskedLM(config=self.generator_config)
        
        self.discriminator_config = DebertaV2Config.from_pretrained(self.hparams.model_name)
        self.discriminator_config.num_labels = 2
        self.discriminator = DebertaV2ForTokenClassification(config=self.discriminator_config)
        
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
        
        optimizer_generator, optimizer_discriminator = self.optimizers()
        scheduler_generator, scheduler_discriminator = self.lr_schedulers()
        
        self.toggle_optimizer(optimizer_generator, 0)
        optimizer_generator.zero_grad()
        pred_ids, loss_generator = self.forward_generator(masked_ids=masked_ids, attention_mask=attention_mask, label_ids=label_ids)
        pred_ids = pred_ids.detach()
        self.manual_backward(loss_generator)
        self.clip_gradients(optimizer_generator, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        optimizer_generator.step()
        scheduler_generator.step()
        self.untoggle_optimizer(0)

        self.toggle_optimizer(optimizer_discriminator, 1)
        optimizer_discriminator.zero_grad()
        inputs_embeds = self.generator.deberta.embeddings.word_embeddings(pred_ids).detach() + self.discriminator.deberta.embeddings.word_embeddings(pred_ids)
        loss_discriminator = self.forward_discriminator(inputs_embeds=inputs_embeds, attention_mask=attention_mask, masked_ids=masked_ids, label_ids=label_ids)
        loss_discriminator = loss_discriminator * 50
        self.manual_backward(loss_discriminator)
        self.clip_gradients(optimizer_discriminator, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        optimizer_discriminator.step()
        scheduler_discriminator.step()
        self.untoggle_optimizer(1)

        self.log_dict({"Loss_G": loss_generator, "Loss_D": loss_discriminator}, on_step=True, on_epoch=False, prog_bar=True)

        gc.collect()
        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx):
        label_ids = batch['label_ids']
        masked_ids = batch['masked_ids']
        attention_mask = batch['attention_mask']
                
        pred_ids, loss_generator = self.forward_generator(masked_ids=masked_ids, attention_mask=attention_mask, label_ids=label_ids)
        pred_ids = pred_ids.detach()

        inputs_embeds = self.discriminator.deberta.embeddings.word_embeddings(pred_ids)
        loss_discriminator = self.forward_discriminator(inputs_embeds=inputs_embeds, attention_mask=attention_mask, masked_ids=masked_ids)
        loss_discriminator = loss_discriminator * 50
        
        self.log_dict({"Loss_G": loss_generator, "Loss_D": loss_discriminator}, on_step=True, on_epoch=False)
    
    def configure_optimizers(self):
        optimizer_generator = AdamW(self.generator.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), weight_decay=0.01)
        optimizer_discriminator = AdamW(self.discriminator.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), weight_decay=0.01)
        scheduler_generator = get_linear_schedule_with_warmup(optimizer_generator, self.hparams.num_warmup_steps, self.hparams.num_training_steps)
        scheduler_discriminator = get_linear_schedule_with_warmup(optimizer_discriminator, self.hparams.num_warmup_steps, self.hparams.num_training_steps)
        return [optimizer_generator, optimizer_discriminator], [scheduler_generator, scheduler_discriminator]
