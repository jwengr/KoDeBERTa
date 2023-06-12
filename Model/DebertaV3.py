import torch
import torch.nn as nn
import pytorch_lightning as pl

from copy import deepcopy
from transformers import DebertaV2Config, DebertaV2ForMaskedLM, DebertaV2ForTokenClassification
from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup

class DebertaV3ForPretraining(pl.LightningModule):
    def __init__(self, model_name, mask_id, pad_id, lr=1e-6, num_warmup_steps=10_000, num_training_steps=125_000):
        super(DebertaV3ForPretraining, self).__init__()
        self.model_name = model_name
        self.mask_id = mask_id
        self.pad_id = pad_id
        self.lr = lr
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        
        self.generator_config = DebertaV2Config.from_pretrained(model_name)
        self.generator_config.num_hidden_layers = self.generator_config.num_hidden_layers // 2
        self.generator = DebertaV2ForMaskedLM(config=self.generator_config)
        
        self.discriminator_config = DebertaV2Config.from_pretrained(model_name)
        self.discriminator_config.num_labels = 2
        self.discriminator = DebertaV2ForTokenClassification(config=self.discriminator_config)
        self.discriminator.deberta.embeddings.word_embeddings_diff = deepcopy(self.discriminator.deberta.embeddings.word_embeddings)
        
        self.automatic_optimization = False

    def forward_generator(self, masked_ids, attention_mask, label_ids):
        labels = torch.where(masked_ids == self.mask_id, label_ids, -100)
        outputs = self.generator(input_ids=masked_ids, attention_mask=attention_mask, labels=labels)
        logits, loss = outputs.logits, outputs.loss
        pred_ids = logits.argmax(dim=-1)
        return pred_ids, loss
    
    def forward_discriminator(self, inputs_embeds, attention_mask, masked_ids):
        labels = torch.zeros_like(masked_ids, dtype=torch.long, device=inputs_embeds.device)
        labels[masked_ids==self.mask_id]=1
        labels[masked_ids==self.pad_id]=-100
        loss = self.discriminator(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels).loss
        return loss

    def training_step(self, batch, batch_idx):
        label_ids = batch['label_ids']
        masked_ids = batch['masked_ids']
        attention_mask = batch['attention_mask']
        
        optimizer_generator, optimizer_discriminator = self.optimizers()
        scheduler_generator, scheduler_discriminator = self.lr_schedulers()
        
        pred_ids, loss_generator = self.forward_generator(masked_ids=masked_ids.detach(), attention_mask=attention_mask.detach(), label_ids=label_ids.detach())
        pred_ids = pred_ids.detach()
        optimizer_generator.zero_grad()
        self.manual_backward(loss_generator)
        self.clip_gradients(optimizer_generator, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        optimizer_generator.step()
        scheduler_generator.step()
        
        inputs_embeds_stop_gradient = self.generator.deberta.embeddings.word_embeddings(pred_ids).detach()
        inputs_embeds_diff = self.discriminator.deberta.embeddings.word_embeddings_diff(pred_ids)
        inputs_embeds = inputs_embeds_stop_gradient + inputs_embeds_diff
        loss_discriminator = self.forward_discriminator(inputs_embeds=inputs_embeds.detach(), attention_mask=attention_mask.detach(), masked_ids=masked_ids.detach())
        loss_discriminator = loss_discriminator * 50
        optimizer_discriminator.zero_grad()
        self.manual_backward(loss_discriminator)
        self.clip_gradients(optimizer_discriminator, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        optimizer_discriminator.step()
        scheduler_discriminator.step()
        
        self.discriminator.deberta.embeddings.word_embeddings.weight = nn.Parameter(
            self.generator.deberta.embeddings.word_embeddings.weight.detach() + self.discriminator.deberta.embeddings.word_embeddings_diff.weight.detach(),
            requires_grad=True
        )

        self.log_dict({"Loss_G": loss_generator.item(), "Loss_D": loss_discriminator.item()}, on_step=True, on_epoch=False)
        torch.cuda.empty_cache()
        return
    
    def validation_step(self, batch, batch_idx):
        label_ids = batch['label_ids']
        masked_ids = batch['masked_ids']
        attention_mask = batch['attention_mask']
                
        pred_ids, loss_generator = self.forward_generator(masked_ids=masked_ids, attention_mask=attention_mask, label_ids=label_ids)
        pred_ids = pred_ids.detach()

        inputs_embeds = self.discriminator.deberta.embeddings.word_embeddings(pred_ids)
        loss_discriminator = self.forward_discriminator(inputs_embeds=inputs_embeds, attention_mask=attention_mask, masked_ids=masked_ids)
        loss_discriminator = loss_discriminator * 50
        
        self.log_dict({"Loss_G": loss_generator.item(), "Loss_D": loss_discriminator.item()}, on_step=True)
        return 
    
    def configure_optimizers(self):
        optimizer_generator = AdamW(self.generator.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.01)
        optimizer_discriminator = AdamW(self.discriminator.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.01)
        scheduler_generator = get_linear_schedule_with_warmup(optimizer_generator, self.num_warmup_steps, self.num_training_steps)
        scheduler_discriminator = get_linear_schedule_with_warmup(optimizer_discriminator, self.num_warmup_steps, self.num_training_steps)
        return [optimizer_generator, optimizer_discriminator], [scheduler_generator, scheduler_discriminator]
