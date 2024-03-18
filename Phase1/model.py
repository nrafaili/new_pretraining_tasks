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

            
def calculate_batching_parameters(args, train_dataset):
    if args.effective_batch_size > args.batch_size:
        num_devices = torch.cuda.device_count() if torch.cuda.device_count() > 1 else 1  # for CPU
        avg_length = train_dataset.__avg__()  # Assuming this is a method to get the average length
        args.grad_accum = int((args.effective_batch_size / avg_length) / (args.batch_size * num_devices))
        args.grad_accum = max(args.grad_accum, 1)  # Ensure grad_accum is at least 1

        if args.grad_accum == 1:
            args.effective_batch_size = avg_length * args.batch_size * num_devices

        print('\n-----Batching Summary-----\n')
        print(f'Number of devices: {num_devices}')
        print(f'Average sequence length: {avg_length}')
        print(f'Local batch size: {args.batch_size} seqs')
        print(f'Gradient accumulation: {args.grad_accum}')
        print(f'Effective batch size: {int(args.effective_batch_size)} tokens')
    else:
        args.grad_accum = 1

    return args
