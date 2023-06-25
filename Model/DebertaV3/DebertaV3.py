import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad_(False)

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad_(True)

class LitDebertaV3ForPretrainingWithDeepSpeedZero3(pl.LightningModule):
    def __init__(
            self, 
            ds_config, 
            model_name, 
            mask_id, 
            pad_id, 
            current_step,
            max_steps, 
            save_per_steps,
            gradient_checkpointing,
            generator_save_dir,
            discriminator_save_dir,
            load_pretrained=False,
            generator_checkpoint_id=str(),
            discriminator_checkpoint_id=str(),
        ):
        super().__init__()
        import deepspeed

        from .CustomDebertaV2DeepSpeed import DebertaV2Config, DebertaV2ForMaskedLM, DebertaV2ForTokenClassification

        self.save_hyperparameters()
        
        self.generator_config = DebertaV2Config.from_pretrained(self.hparams.model_name)
        self.generator_config.num_hidden_layers = self.generator_config.num_hidden_layers // 2
        with deepspeed.zero.Init():
            self.generator = DebertaV2ForMaskedLM(config=self.generator_config)
        self.generator._set_gradient_checkpointing(self.hparams.gradient_checkpointing)
        self.generator_engine, _, _, _  = deepspeed.initialize(model=self.generator, model_parameters=self.generator.parameters(), config=self.hparams.ds_config)

        self.discriminator_config = DebertaV2Config.from_pretrained(self.hparams.model_name)
        self.discriminator_config.num_labels = 2
        with deepspeed.zero.Init():
            self.discriminator = DebertaV2ForTokenClassification(config=self.discriminator_config)
            self.discriminator.deberta.embeddings.word_embeddings = nn.Linear(self.discriminator_config.hidden_size, self.discriminator_config.hidden_size)
        self.discriminator_engine, _, _, _ = deepspeed.initialize(model=self.discriminator, model_parameters=self.discriminator.parameters(), config=self.hparams.ds_config)

        if self.hparams.load_pretrained:
            self.load_pretrained()

        if self.hparams.gradient_checkpointing:
            self.generator.gradient_checkpointing_enable()
            self.discriminator.gradient_checkpointing_enable()

        self.generator_checkpoint_id = None
        self.discriminator_checkpoint_id = None

        self.automatic_optimization = False

    def load_pretrained(self):
        self.generator.load_state_dict(torch.load(f'{self.hparams.generator_save_dir}/{self.hparams.generator_checkpoint_id}.pth'))
        self.generator_engine.load_checkpoint(self.hparams.generator_save_dir, self.hparams.generator_checkpoint_id)
        self.discriminator.load_state_dict(torch.load(f'{self.hparams.discriminator_save_dir}/{self.hparams.discriminator_checkpoint_id}.pth'))
        self.discriminator._set_gradient_checkpointing(self.hparams.gradient_checkpointing)
        self.discriminator_engine.load_checkpoint(self.hparams.discriminator_save_dir, self.hparams.discriminator_checkpoint_id)

    def discriminator_postprocessing(self):
        self.discriminator.deberta.embeddings.word_embeddings.weight = nn.Parameter(
            F.linear(self.generator.deberta.embeddings.word_embeddings.weight, self.discriminator.deberta.embeddings.word_embeddings.weight, self.discriminator.deberta.embeddings.word_embeddings.bias),
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
        labels = torch.where(masked_ids == self.hparams.mask_id, 1, torch.where(masked_ids == self.hparams.pad_id, -100, 0)).type_as(masked_ids)
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
        inputs_embeds = self.discriminator.deberta.embeddings.word_embeddings(self.generator.deberta.embeddings.word_embeddings(pred_ids).detach())
        loss_discriminator = self.forward_discriminator(inputs_embeds=inputs_embeds, attention_mask=attention_mask, masked_ids=masked_ids)
        loss_discriminator = loss_discriminator * 50
        self.discriminator_engine.backward(loss_discriminator)
        self.discriminator_engine.step()
        freeze_model(self.discriminator)

        self.log_dict({"Loss_G": loss_generator, "Loss_D": loss_discriminator}, on_step=True, on_epoch=False, prog_bar=True)
        self.generator_checkpoint_id = f'current_step={self.hparams.current_step}_loss={loss_generator.item()}'
        self.discriminator_checkpoint_id = f'current_step={self.hparams.current_step}_loss={loss_discriminator.item()}'
        
        if self.hparams.current_step>0 and self.hparams.current_step%self.hparams.save_per_steps==0:
            self.save()

        self.hparams.current_step = self.hparams.current_step+1

        gc.collect()
        torch.cuda.empty_cache()

    def save(self):
        torch.save(self.generator.state_dict(), f'{self.hparams.generator_save_dir}/{self.generator_checkpoint_id}.pth')
        self.generator_engine.save_checkpoint(self.hparams.generator_save_dir, f'{self.generator_checkpoint_id}')
        torch.save(self.discriminator.state_dict(), f'{self.hparams.discriminator_save_dir}/{self.discriminator_checkpoint_id}.pth')
        self.discriminator_engine.save_checkpoint(self.hparams.discriminator_save_dir, f'{self.discriminator_checkpoint_id}')

    def on_train_epoch_end(self):
        self.save()
        raise ValueError('Epoch end. You should reset dataset.')
    
    def on_train_end(self):
        self.save()
        return ValueError('Train end.')
        
    def configure_optimizers(self):
        return


class LitDebertaV3ForPretrainingWithDeepSpeed(pl.LightningModule):
    def __init__(
            self, 
            ds_config, 
            model_name, 
            mask_id, 
            pad_id, 
            current_step,
            max_steps, 
            save_per_steps,
            gradient_checkpointing,
            generator_save_dir,
            discriminator_save_dir,
            load_pretrained=False,
            generator_checkpoint_id=str(),
            discriminator_checkpoint_id=str(),
        ):
        super().__init__()
        import deepspeed

        from .CustomDebertaV2 import DebertaV2Config, DebertaV2ForMaskedLM, DebertaV2ForTokenClassification

        self.save_hyperparameters()
        
        self.generator_config = DebertaV2Config.from_pretrained(self.hparams.model_name)
        self.generator_config.num_hidden_layers = self.generator_config.num_hidden_layers // 2
        self.generator = DebertaV2ForMaskedLM(config=self.generator_config)
        self.generator._set_gradient_checkpointing(self.hparams.gradient_checkpointing)
        self.generator_engine, _, _, _  = deepspeed.initialize(model=self.generator, model_parameters=self.generator.parameters(), config=self.hparams.ds_config)

        self.discriminator_config = DebertaV2Config.from_pretrained(self.hparams.model_name)
        self.discriminator_config.num_labels = 2
        self.discriminator = DebertaV2ForTokenClassification(config=self.discriminator_config)
        self.discriminator.deberta.embeddings.word_embeddings = nn.Linear(self.discriminator_config.hidden_size, self.discriminator_config.hidden_size)
        self.discriminator_engine, _, _, _ = deepspeed.initialize(model=self.discriminator, model_parameters=self.discriminator.parameters(), config=self.hparams.ds_config)

        if self.hparams.load_pretrained:
            self.load_pretrained()

        if self.hparams.gradient_checkpointing:
            self.generator.gradient_checkpointing_enable()
            self.discriminator.gradient_checkpointing_enable()

        self.generator_checkpoint_id = None
        self.discriminator_checkpoint_id = None

        self.automatic_optimization = False

    def load_pretrained(self):
        self.generator.load_state_dict(torch.load(f'{self.hparams.generator_save_dir}/{self.hparams.generator_checkpoint_id}.pth'))
        self.generator_engine.load_checkpoint(self.hparams.generator_save_dir, self.hparams.generator_checkpoint_id)
        self.discriminator.load_state_dict(torch.load(f'{self.hparams.discriminator_save_dir}/{self.hparams.discriminator_checkpoint_id}.pth'))
        self.discriminator._set_gradient_checkpointing(self.hparams.gradient_checkpointing)
        self.discriminator_engine.load_checkpoint(self.hparams.discriminator_save_dir, self.hparams.discriminator_checkpoint_id)

    def discriminator_postprocessing(self):
        self.discriminator.deberta.embeddings.word_embeddings.weight = nn.Parameter(
            F.linear(self.generator.deberta.embeddings.word_embeddings.weight, self.discriminator.deberta.embeddings.word_embeddings.weight, self.discriminator.deberta.embeddings.word_embeddings.bias),
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
        labels = torch.where(masked_ids == self.hparams.mask_id, 1, torch.where(masked_ids == self.hparams.pad_id, -100, 0)).type_as(masked_ids)
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
        inputs_embeds = self.discriminator.deberta.embeddings.word_embeddings(self.generator.deberta.embeddings.word_embeddings(pred_ids).detach())
        loss_discriminator = self.forward_discriminator(inputs_embeds=inputs_embeds, attention_mask=attention_mask, masked_ids=masked_ids)
        loss_discriminator = loss_discriminator * 50
        self.discriminator_engine.backward(loss_discriminator)
        self.discriminator_engine.step()
        freeze_model(self.discriminator)

        self.log_dict({"Loss_G": loss_generator, "Loss_D": loss_discriminator}, on_step=True, on_epoch=False, prog_bar=True)
        self.generator_checkpoint_id = f'current_step={self.hparams.current_step}_loss={loss_generator.item()}'
        self.discriminator_checkpoint_id = f'current_step={self.hparams.current_step}_loss={loss_discriminator.item()}'
        
        if self.hparams.current_step>0 and self.hparams.current_step%self.hparams.save_per_steps==0:
            self.save()

        self.hparams.current_step = self.hparams.current_step+1

        gc.collect()
        torch.cuda.empty_cache()

    def save(self):
        torch.save(self.generator.state_dict(), f'{self.hparams.generator_save_dir}/{self.generator_checkpoint_id}.pth')
        self.generator_engine.save_checkpoint(self.hparams.generator_save_dir, f'{self.generator_checkpoint_id}')
        torch.save(self.discriminator.state_dict(), f'{self.hparams.discriminator_save_dir}/{self.discriminator_checkpoint_id}.pth')
        self.discriminator_engine.save_checkpoint(self.hparams.discriminator_save_dir, f'{self.discriminator_checkpoint_id}')

    def on_train_epoch_end(self):
        self.save()
        raise ValueError('Epoch end. You should reset dataset.')
    
    def on_train_end(self):
        self.save()
        return ValueError('Train end.')
        
    def configure_optimizers(self):
        return


class LitDebertaV3ForPretraining(pl.LightningModule):
    def __init__(
            self, 
            model_name, 
            mask_id, 
            pad_id, 
            lr,
            num_warmup_steps,
            num_training_steps,
            gradient_checkpointing,
            current_step,
        ):
        super().__init__()
        from .CustomDebertaV2 import DebertaV2Config, DebertaV2ForMaskedLM, DebertaV2ForTokenClassification
        
        self.save_hyperparameters()
        
        self.generator_config = DebertaV2Config.from_pretrained(self.hparams.model_name)
        self.generator_config.num_hidden_layers = self.generator_config.num_hidden_layers // 2
        self.generator = DebertaV2ForMaskedLM(config=self.generator_config)

        self.discriminator_config = DebertaV2Config.from_pretrained(self.hparams.model_name)
        self.discriminator_config.num_labels = 2
        self.discriminator = DebertaV2ForTokenClassification(config=self.discriminator_config)

        if self.hparams.gradient_checkpointing:
            self.generator.gradient_checkpointing_enable()
            self.discriminator.gradient_checkpointing_enable()

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
        labels = torch.where(masked_ids == self.hparams.mask_id, 1, torch.where(masked_ids == self.hparams.pad_id, -100, 0)).type_as(masked_ids)
        loss = self.discriminator(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels).loss
        del labels
        return loss

    def training_step(self, batch, batch_idx):
        label_ids = batch['label_ids']
        masked_ids = batch['masked_ids']
        attention_mask = batch['attention_mask']

        generator_optimizer, discriminator_optimizer = self.optimizers()
        generator_scheduler, discriminator_scheduler = self.lr_schedulers()
        
        unfreeze_model(self.generator)
        self.toggle_optimizer(optimizer=generator_optimizer)
        pred_ids, loss_generator = self.forward_generator(masked_ids=masked_ids, attention_mask=attention_mask, label_ids=label_ids)
        pred_ids = pred_ids.detach()
        generator_optimizer.zero_grad()
        self.manual_backward(loss_generator)
        self.clip_gradients(generator_optimizer, gradient_clip_val=1.0)
        generator_optimizer.step()
        generator_scheduler.step()
        self.untoggle_optimizer(optimizer=generator_optimizer)
        freeze_model(self.generator)

        unfreeze_model(self.discriminator)
        self.toggle_optimizer(optimizer=discriminator_optimizer)
        inputs_embeds = self.generator.deberta.embeddings.word_embeddings(pred_ids).detach() + self.discriminator.deberta.embeddings.word_embeddings(pred_ids)
        loss_discriminator = self.forward_discriminator(inputs_embeds=inputs_embeds, attention_mask=attention_mask, masked_ids=masked_ids)
        loss_discriminator = loss_discriminator * 50
        discriminator_optimizer.zero_grad()
        self.manual_backward(loss_discriminator)
        self.clip_gradients(discriminator_optimizer, gradient_clip_val=1.0)
        discriminator_optimizer.step()
        discriminator_scheduler.step()
        self.untoggle_optimizer(optimizer=discriminator_optimizer)
        freeze_model(self.discriminator)

        self.hparams.current_step  = self.hparams.current_step + 1
        self.log_dict({"train_percentage":self.hparams.current_step/self.hparams.num_training_steps, "Loss_G": loss_generator, "Loss_D": loss_discriminator}, on_step=True, on_epoch=False, prog_bar=True)
        
        gc.collect()
        torch.cuda.empty_cache()
                
    def configure_optimizers(self):
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup

        generator_optimizer = AdamW(self.generator.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), weight_decay=0.01)
        generator_scheduler = get_linear_schedule_with_warmup(generator_optimizer, num_warmup_steps=self.hparams.num_warmup_steps, num_training_steps=self.hparams.num_training_steps)
        discriminator_optimizer = AdamW(self.discriminator.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), weight_decay=0.01)
        discriminator_scheduler = get_linear_schedule_with_warmup(discriminator_optimizer, num_warmup_steps=self.hparams.num_warmup_steps, num_training_steps=self.hparams.num_training_steps)

        return [generator_optimizer, discriminator_optimizer], [generator_scheduler, discriminator_scheduler]