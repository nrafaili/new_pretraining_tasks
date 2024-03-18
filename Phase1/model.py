from torch.utils.data import Dataset as TorchDataset
import random
from transformers import EsmConfig, EsmForMaskedLM
import torch
from torch import nn

class EsmForMLM(EsmForMaskedLM):

    def __init__(self, model_path, hidden_size, num_hidden_layers, seed):
        torch.manual_seed(seed)
        self.model_path = model_path
        config = EsmConfig.from_pretrained(model_path)
        config.hidden_size = hidden_size
        config.num_hidden_layers = num_hidden_layers
        config.num_attention_heads = config.hidden_size // 32
        super().__init__(config)

        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
