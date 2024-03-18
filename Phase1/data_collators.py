from transformers.data.data_collator import DataCollatorMixin, _torch_collate_batch
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import random
import torch
from transformers import EsmTokenizer, DataCollatorForLanguageModeling
import numpy as np


class DataCollatorForMixedMLM(DataCollatorMixin):
    
    def __init__(self, tokenizer: EsmTokenizer, mlm: bool = True, return_tensors: str = 'pt', pad_to_multiple_of: Optional[int] = None):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.return_tensors = return_tensors
        self.pad_to_multiple_of = pad_to_multiple_of


    def torch_call(self, input): # input is list of examples from dataset
        batch = self.tokenizer(input, return_tensors='pt', padding='longest', truncation=False, add_special_tokens=False)
        if self.mlm:
            batch['input_ids'], labels = self.torch_mask_tokens(batch['input_ids'])
            batch['labels'] = labels # the keys here, need to match the keys in the model
        return batch # in here are input_ids, attention_mask, and labels as keys in dictionary

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 90% MASK, 5% random, 5% original.
        """
        labels = inputs.clone()
        #probability_matrix = torch.full(labels.shape, random.uniform(self.min_prob, self.max_prob))
        probability_matrix = torch.normal(mean=0.3, std=0.12, size=labels.shape)
        probability_matrix = torch.clamp(probability_matrix, min=0.0, max=1.0)

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        # 85% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.90)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced # 0.5 because half of remaining are random
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        # The rest of the time (5% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    

class DataCollatorForMLM(DataCollatorForLanguageModeling):

    def __init__(self, tokenizer, mlm_probability=0.15, **kwargs):
        super().__init__(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability, **kwargs)


def get_data_collator(mlm_probability, tokenizer):

    if mlm_probability == 'mixed':
        return DataCollatorForMixedMLM(tokenizer=tokenizer)
    else:
        mlm_probability = float(mlm_probability)
        return DataCollatorForMLM(tokenizer=tokenizer, mlm_probability=mlm_probability)
