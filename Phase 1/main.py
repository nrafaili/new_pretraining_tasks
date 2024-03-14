import argparse
import yaml
from transformers import EsmTokenizer
from data_collators import DataCollatorForMixedMLM, DataCollatorForMLM, get_data_collator
from model import EsmForMLM
from metrics import compute_metrics, log_metrics
from trainer import SortedTrainer, setup_and_train
from datasets import load_dataset
from data import Dataset


def parse_args():

    parser = argparse.ArgumentParser()

    # Data arguments
    parser.add_argument('--data_path', type=str, required=True, help='Dataset path')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Tokenizer path')
    parser.add_argument('--save_path', type=str, default='./best_model', help='Save Path')
    parser.add_argument('--weight_path', type=str, default='./best_model', help='Weight Save Path')

    # Model arguments
    parser.add_argument('--hidden_size', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--num_hidden_layers', type=int, default=24, help='Number of transformer blocks')
    parser.add_argument('--seed', type=int, default=338, help='Random seed for reproducibility')
    parser.add_argument('--model_path', type=str, required=True, help='Model path')

    # Training arguments
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--mlm_probability', type=str, choices=['0.05', '0.15', '0.3', '0.5', 'mixed'], default='mixed',
                        help='MLM probability')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--group_by_length', action='store_true', default=True, help='Group by length for SortedTrainer')
    parser.add_argument('--grad_accum', type=int, default=1, help='Gradient Accumulation')
    parser.add_argument('--evaluation_strategy', type=str, default='steps', help='Evaluation strategy to use')
    parser.add_argument('--log_path', type=str, default = './logmlm.txt', help='log_path')
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine', help='Type of learning rate scheduler')
    parser.add_argument('--optim', type=str, default='adamw_torch', help='Optimizer')
    parser.add_argument('--logging_steps', type=int, default=10, help='Step interval for logging')
    parser.add_argument('--eval_steps', type=int, default=50, help='Step interval for evaluation')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay rate')
    parser.add_argument('--warmup_steps', type=int, default=10, help='Number of warmup steps')
    parser.add_argument('--effective_batch_size', type=int, default=1000000, help='Effective batch size for training')
    parser.add_argument('--fp16', action='store_true', help='Enable mixed-precision training')
    parser.add_argument('--save_total_limit', type=int, default=5, help='Maximum number of checkpoints to keep')
    parser.add_argument('--load_best_model_at_end', action='store_true', help='Load the best model at the end of training')
    parser.add_argument('--greater_is_better', action='store_true', help='Determines if a greater metric signifies a better model')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')


args = parse_args()

def get_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        args = yaml.safe_load(file)

    return args

dataset = load_dataset(args.data_path)['train']
train_dataset = Dataset(dataset)
model = EsmForMLM(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, seed=args.seed)
tokenizer = EsmTokenizer.from_pretrained(args.tokenizer_path)
data_collator = get_data_collator(args.mlm_probability, tokenizer)
setup_and_train(model, tokenizer, data_collator, train_dataset, compute_metrics, args)