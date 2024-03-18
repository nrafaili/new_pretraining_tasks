import argparse
import yaml
from transformers import EsmTokenizer
from data_collators import DataCollatorForMixedMLM, DataCollatorForMLM, get_data_collator
from model import EsmForMLM, calculate_batching_parameters
from metrics import compute_metrics, log_metrics
from trainer import SortedTrainer, HF_trainer
from datasets import load_dataset
from data import Dataset



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', type=str)
    args = parser.parse_args()
    return args


def get_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        args = yaml.safe_load(file)
    return args


def main():
    parse = parse_args()
    args = get_yaml(parse.yaml_path)
    dataset = load_dataset(args.data_path)['train']
    train_dataset = Dataset(dataset)
    model = EsmForMLM(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, seed=args.seed)
    tokenizer = EsmTokenizer.from_pretrained(args.tokenizer_path)
    data_collator = get_data_collator(args.mlm_probability, tokenizer)
    args = calculate_batching_parameters(args, train_dataset)
    

    trainer = HF_trainer(model, tokenizer, data_collator, train_dataset, compute_metrics, args)
    trainer.train()
    trainer.push_to_hub(args.HF_path)
    model.push_to_hub(args.HF_path)

if __name__ == "__main__":
    main()
