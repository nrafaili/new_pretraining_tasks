from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import torch
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.trainer import has_length
from transformers.utils import is_datasets_available
from transformers.trainer_pt_utils import LengthGroupedSampler, RandomSampler
from torch.utils.data import Dataset as TorchDataset
from data import Dataset


class SortedTrainer(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, TorchDataset):
                lengths = self.train_dataset.lengths # this requires your dataset has a self.lengths with the lengths in it
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )
        else:
            return RandomSampler(self.train_dataset)

def HF_trainer(model,
               train_dataset,
               tokenizer,
               compute_metrics=None,
               data_collator=None,
               patience=3,
               *args, **kwargs):
    training_args = TrainingArguments(*args, **kwargs)
    trainer = SortedTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)]
    )
    return trainer