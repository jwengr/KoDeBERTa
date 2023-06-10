import torch
import torch.nn as nn
import pytorch_lightning as pl

from copy import deepcopy
from transformers import DebertaV2Config, DebertaV2ForMaskedLM, DebertaV2ForTokenClassification
from transformers.optimization import AdamW

class DebertaV3ForPretraining(pl.LightningModule):
    def __init__(self, model_name):
        super(DebertaV3ForPretraining, self).__init__()
        self.model_name = model_name
        
        self.generator_config = DebertaV2Config.from_pretrained(model_name)
        self.generator_config.num_hidden_layers = self.generator_config.num_hidden_layers // 2
        self.generator = DebertaV2ForMaskedLM(config=self.generator_config)
        
        self.discriminator_config = DebertaV2Config.from_pretrained(model_name)
        self.discriminator_config.num_labels = 2
        self.discriminator = DebertaV2ForTokenClassification(config=self.discriminator_config)
        self.discriminator.deberta.embeddings.word_embeddings_diff = deepcopy(self.discriminator.deberta.embeddings.word_embeddings)
        
        self.automatic_optimization = False

    def forward_generator(self, inputs_ids, attention_mask, labels):
        outputs = self.generator(inputs_ids=inputs_ids, attention_mask=attention_mask)
        logits, loss = outputs.logits, outputs.loss
        ids_pred = logits.argmax(dim=-1)
        return ids_pred, loss
    
    def forward_discriminator(self, inputs_embeds, attention_mask):
        loss = self.discriminator(inputs_embeds=inputs_embeds, attention_mask=attention_mask).loss
        return loss

    def training_step(self, batch, batch_idx):
        inputs_ids = batch['inputs_ids']
        attention_mask = batch['attention_mask']
        
        optimizer_generator, optimizer_discriminator = self.optimizers()
        
        inputs_ids_pred, loss_generator = self.forward_generator(inputs_ids=inputs_ids, attention_mask=attention_mask)
        inputs_ids_pred = inputs_ids_pred.detach()
        optimizer_generator.zero_grad()
        self.manual_backward(loss_generator)
        self.clip_gradients(optimizer_generator, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        optimizer_generator.step()
        
        inputs_embeds_stop_gradient = self.generator.deberta.embeddings.word_embeddings(inputs_ids_pred).detach()
        inputs_embeds_diff = self.discriminator.deberta.embeddings.word_embeddings_diff(inputs_ids_pred)
        inputs_embeds = inputs_embeds_stop_gradient + inputs_embeds_diff
        loss_discriminator = self.forward_discriminator(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        loss_discriminator = loss_discriminator * 50
        optimizer_discriminator.zero_grad()
        self.manual_backward(loss_discriminator)
        self.clip_gradients(optimizer_discriminator, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        optimizer_discriminator.step()
        
        self.discriminator.deberta.embeddings.word_embeddings.weight = nn.Parameter(
            self.generator.deberta.embeddings.word_embeddings.weight.detach() + self.discriminator.deberta.embeddings.word_embeddings_diff.weight.detach(),
            requires_grad=True
        )
        
        self.log_dict({"Loss_G": loss_generator, "Loss_D": loss_discriminator}, prog_bar=True)
        return
    
    def validation_step(self, batch, batch_idx):
        return        
    
    def configure_optimizers(self):
        lr = 1e-6
        betas = (0.9, 0.999)
        weight_decay = 0.01
        optimizer_generator = AdamW(self.generator.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        optimizer_discriminator = AdamW(self.discriminator.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        return [optimizer_generator, optimizer_discriminator]
