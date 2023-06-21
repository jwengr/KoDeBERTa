import random
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf

from tokenizers import Tokenizer

class DataCollatorForHFUnigramSpanMLM:
    def __init__(
            self,
            tokenizer:Tokenizer,
            pad_token='[PAD]',
            mask_prob=0.15,
            mask_token='[MASK]',
            metaspace_token='▁',
            padding_argument={},
            truncation_argument={},
            from_hf_datasets=False,
            return_tensors='pt',
        ):
        self.tokenizer = tokenizer
        self.pad_token = pad_token
        self.pad_id = self.tokenizer.get_vocab()[self.pad_token]
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.mask_id = self.tokenizer.get_vocab()[self.mask_token]
        self.metaspace_token = metaspace_token
        self.metaspace_id = self.tokenizer.get_vocab()[self.metaspace_token]
        self.padding_argument = padding_argument
        self.padding_argument['pad_token'] = self.pad_token
        self.padding_argument['pad_id'] = self.pad_id
        self.truncation_argument = truncation_argument
        self.from_hf_datasets=from_hf_datasets
        self.return_tensors=return_tensors

        if self.padding_argument:
            self.tokenizer.enable_padding(**self.padding_argument)
        if self.truncation_argument:
            self.tokenizer.enable_truncation(**self.truncation_argument)

    def __call__(self, batch):
        if self.from_hf_datasets:
            batch = [data['text'] for data in batch]
        encodes = self.tokenizer.encode_batch(batch)
        label_ids = []
        masked_ids = []
        attention_mask = []
        for encode in encodes:
            label_id = encode.ids
            label_ids.append(label_id)
            masked_id = self._span_mlm(encode.ids)
            masked_ids.append(masked_id)
            attention_mask.append(encode.attention_mask)

        if self.return_tensors=='pt':
            label_ids = torch.LongTensor(label_ids)
            masked_ids = torch.LongTensor(masked_ids)
            attention_mask = torch.LongTensor(attention_mask)
        elif self.return_tensors=='np':
            label_ids = np.array(label_ids)
            masked_ids = np.array(masked_ids)
            attention_mask = np.array(attention_mask)
        elif self.return_tensors=='tf':
            label_ids = tf.convert_to_tensor(label_ids)
            masked_ids = tf.convert_to_tensor(masked_ids)
            attention_mask = tf.convert_to_tensor(attention_mask)

        batch = {
            'label_ids' : label_ids,
            'masked_ids' : masked_ids,
            'attention_mask' : masked_ids
        }

        return batch

    def _span_mlm(self, encode_ids):
        encode_ids_mlm=[]
        mask_flag = random.random()<self.mask_prob
        for encode_id in encode_ids:
            if encode_id==self.pad_id:
                encode_ids_mlm.append(self.pad_id)
                continue
            elif encode_id==self.metaspace_id:
                if random.random()<self.mask_prob:
                    encode_ids_mlm.append(self.mask_id)
                else:
                    encode_ids_mlm.append(encode_id)
                mask_flag = random.random()<self.mask_prob
                continue
            else:
                if mask_flag:
                    encode_ids_mlm.append(self.mask_id)
                else:
                    encode_ids_mlm.append(encode_id)
                continue
        return encode_ids_mlm
        
