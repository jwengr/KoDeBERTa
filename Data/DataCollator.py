import random
import numpy as np
import torch 
import torch.nn as nn

from tokenizers import Tokenizer

class DataCollatorForSentencePieceSpanMLM:
    def __init__(
            self,
            tokenizer:Tokenizer,
            pad_token='[PAD]',
            mask_prob=0.15,
            mask_token='[MASK]',
            metaspace_token='▁',
            padding_argument={},
            truncation_argument={}
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

        if self.padding_argument:
            self.tokenizer.enable_padding(**self.padding_argument)
        if self.truncation_argument:
            self.tokenizer.enable_truncation(**self.truncation_argument)

    def __call__(self, texts):
        encodes = self.tokenizer.encode_batch(texts)
        label_ids = []
        masked_ids = []
        attention_mask = []
        for encode in encodes:
            label_id = encode.ids
            label_ids.append(label_id)
            masked_id = self._span_mlm(encode.ids)
            masked_ids.append(masked_id)
            attention_mask.append(encode.attention_mask)

        batch = {
            'label_ids' : torch.LongTensor(label_ids),
            'masked_ids' : torch.LongTensor(masked_ids),
            'attention_mask' : torch.LongTensor(attention_mask)
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
        


class DataCollatorForMLM:
    pass
