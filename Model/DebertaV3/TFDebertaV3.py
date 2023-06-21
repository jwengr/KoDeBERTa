import tensorflow as tf

from transformers import DebertaV2Config, TFDebertaV2ForMaskedLM, TFDebertaV2ForTokenClassification

class LinearSchedulerWithWarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, learning_rate, warmup_steps, total_steps):
        super().__init__()
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        if step < self.warmup_steps:
            return self.learning_rate * (step / self.warmup_steps)
        else:
            return self.learning_rate * (1 - (step - self.warmup_steps) / (self.total_steps - self.warmup_steps))

class TFDebertaV3ForPretraining(tf.keras.Model):
    def __init__(
            self,
            name,
            model_name,
            mask_id,
            pad_id,
            learning_rate,
            warmup_steps,
            total_steps,
        ):
        super().__init__()
        

        self.name = name
        self.model_name = model_name
        self.mask_id = mask_id
        self.pad_id = pad_id
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        self.generator_config = DebertaV2Config.from_pretrained(self.model_name)
        self.generator_config.num_hidden_layers = self.generator_config.num_hidden_layers // 2
        self.generator = TFDebertaV2ForMaskedLM(self.generator_config, name="generator")
        self.generator_scheduler = LinearSchedulerWithWarmUp(self.learning_rate, self.warmup_steps, self.total_steps)
        self.generator_optimizer = tf.keras.optimizers.AdamW(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, weith_decay=0.01, clipnorm=1.0)

        self.discriminator_config = DebertaV2Config.from_pretrained(self.model_name)
        self.discriminator_config.num_labels = 2
        self.discriminator = TFDebertaV2ForTokenClassification(self.discriminator_config, name="discriminator")
        self.discriminator_scheduler = LinearSchedulerWithWarmUp(self.learning_rate, self.warmup_steps, self.total_steps)
        self.discriminator_optimizer = tf.keras.optimizers.AdamW(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, weith_decay=0.01, clipnorm=1.0)
        

    def forward_generator(self, masked_ids, attention_mask, label_ids):
        labels = tf.where(masked_ids == self.mask_id, label_ids, -100)
        outputs = self.generator(input_ids=masked_ids, labels=labels, attention_mask=attention_mask)
        loss, logits = outputs.loss, outputs.logits
        pred_ids = tf.cast(tf.argmax(logits, axis=-1), label_ids.dtype)
        return loss, pred_ids
    
    def forward_discriminator(self, pred_ids, attention_mask, masked_ids):
        pred_ids = tf.stop_gradient(pred_ids)
        inputs_embeds = tf.stop_gradient(self.generator.deberta.embeddings.word_embeddings(pred_ids)) + self.discriminator.deberta.embeddings.word_embeddings(pred_ids)
        labels = tf.where(masked_ids == self.pad_id, -100, tf.where(masked_ids == self.mask_id, 1, 0))
        loss = self.discriminator(inputs_embeds=inputs_embeds, labels=labels, attention_mask=attention_mask).loss
        return loss
    
    def __call__(self, masked_ids, attention_mask, label_ids, current_step):        
        with tf.GradientTape() as generator_tape:
            pred_ids, loss_generator = self.forward_generator(masked_ids=masked_ids, attention_mask=attention_mask, label_ids=label_ids)
            pred_ids = tf.stop_gradient(pred_ids)
        generator_gradients = generator_tape.gradient(loss_generator, self.generator.trainable_variables)
        self.generator_optimizer.learning_rate = self.generator_scheduler(current_step)
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))

        with tf.GradientTape() as discriminator_tape:
            loss_discriminator = self.forward_discriminator(pred_ids=pred_ids, attention_mask=attention_mask, masked_ids=masked_ids)
            loss_discriminator = loss_discriminator * 50
        discriminator_gradients = discriminator_tape.gradient(loss_discriminator, self.discriminator.trainable_variables)
        self.discriminator_optimizer.learning_rate = self.discriminator_scheduler(current_step)
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

        return loss_generator, loss_discriminator
