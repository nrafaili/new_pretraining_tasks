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

def setup_and_train(model, tokenizer, data_collator, train_dataset, compute_metrics, args):
    # Define training arguments
    training_args = TrainingArguments(
        report_to='wandb' if args.wandb else None,
        output_dir=args.save_path,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=args.logging_steps,
        evaluation_strategy='steps',
        eval_steps=args.eval_steps,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type='cosine',
        learning_rate=args.lr,
        optim='adamw_torch',
        seed=args.seed,
        data_seed=args.seed,
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        greater_is_better=False,
        fp16=args.fp16,
        group_by_length=True,
    )


    trainer = SortedTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]  # usually 3 - 5
    )

    trainer.train()
