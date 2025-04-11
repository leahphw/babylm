"""
Language Model Training Script

This script trains a language model (GPT2, Llama, or GPTJ) on the BabyLM dataset.
It supports configuration via YAML files and command-line arguments.

Usage:
    python train.py --config ./config/gpt-97M.yaml --lr 2.5e-4 --model_name my_model
"""

import os
import argparse
from pathlib import Path
from random import sample, seed
import yaml

import torch
from torch.utils.data import Subset
from transformers import (
    GPT2Config, GPT2LMHeadModel, 
    LlamaConfig, LlamaForCausalLM, 
    GPTJConfig, GPTJForCausalLM,
    GPT2TokenizerFast,
    Trainer, TrainingArguments, 
    DataCollatorForLanguageModeling
)

from babylm_dataset import BabylmDataset

# Set CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a language model on BabyLM dataset")
    parser.add_argument(
        "--config", 
        type=str, 
        default="./config/gpt-97M.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=None,
        help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default=None,
        help="Model name (overrides config)"
    )
    return parser.parse_args()


def load_config(config_path, args):
    """Load and update configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config parameters if provided as command-line arguments
    if args.lr:
        config['training']['lr'] = args.lr
    if args.model_name:
        config['model']['name'] = args.model_name
    
    return config


def create_model(config, tokenizer):
    """Create model based on configuration."""
    model_type = config['model']['type']
    
    if model_type == "Llama":
        model_config = LlamaConfig(
            vocab_size=tokenizer.vocab_size,
            max_position_embeddings=2*tokenizer.model_max_length,
            hidden_size=config['model']['hidden_size'],
            intermediate_size=config['model']['intermediate_size'],
            num_hidden_layers=config['model']['n_layer'],
            num_attention_heads=config['model']['n_head'],
            tie_word_embeddings=config['model'].get('tie_word_embeddings', False),
            pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
        )
        model = LlamaForCausalLM(model_config)
    
    elif model_type == "GPT2":
        model_config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions=2*tokenizer.model_max_length,
            n_embd=config['model']['hidden_size'],
            n_layer=config['model']['n_layer'],
            n_head=config['model']['n_head'],
            resid_pdrop=config['model']['resid_pdrop'],
            embd_pdrop=config['model']['embd_pdrop'],
            attn_pdrop=config['model']['attn_pdrop'],
            pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
        )
        model = GPT2LMHeadModel(model_config)
    
    elif model_type == "GPTJ":
        model_config = GPTJConfig(
            vocab_size=tokenizer.vocab_size,
            n_positions=2*tokenizer.model_max_length,
            n_embd=config['model']['hidden_size'],
            n_layer=config['model']['n_layer'],
            n_head=config['model']['n_head'],
            resid_pdrop=config['model']['resid_pdrop'],
            embd_pdrop=config['model']['embd_pdrop'],
            attn_pdrop=config['model']['attn_pdrop'],
            tie_word_embeddings=config['model']['tie_word_embeddings'],
            pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
        )
        model = GPTJForCausalLM(model_config)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model


def check_gpu_availability():
    """Check GPU availability and configuration."""
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print("To use all GPUs run: torchrun --nproc_per_node=2 train.py")
    else:
        print("No GPU found, using CPU.")
        print("Exiting")
        exit(1)

    assert torch.cuda.device_count() == 2, "Using too many GPUs, professor will not be happy"


def main():
    """Main training function."""
    # Parse arguments and load config
    args = parse_args()
    config = load_config(args.config, args)
    
    # Set random seed for reproducibility
    seed(2023)
    
    # Initialize tokenizer
    tokenizer = GPT2TokenizerFast(tokenizer_file=str(config['data']['tokenizer_path']))
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "<pad>"
    tokenizer.model_max_length = config['data']['seq_length']
    
    # Prepare datasets
    train_dataset = BabylmDataset(
        config['data']['train_path'],
        config['data']['seq_length'],
        tokenizer=tokenizer,
        random_chunk=True,  # Expected to improve model performance
    )
    
    full_eval_dataset = BabylmDataset(
        config['data']['eval_path'],
        config['data']['seq_length'],
        tokenizer=tokenizer,
        offset=0
    )
    
    # Create evaluation subset
    eval_indices = sample(range(len(full_eval_dataset)), config['data']['eval_samples'])
    eval_dataset = Subset(full_eval_dataset, eval_indices)
    del full_eval_dataset
    
    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create model
    model = create_model(config, tokenizer)
    print(f'Model parameters = {model.num_parameters()}')
    
    # Setup training arguments
    output_dir = Path(config['logging']['output_dir']) / config['model']['name']
    accumulation_steps = config['training']['gradient_accumulation_steps']
    per_device_bsz = config['training']['batch_size'] // accumulation_steps
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        num_train_epochs=config['training']['num_epochs'],
        gradient_accumulation_steps=accumulation_steps,
        per_device_train_batch_size=per_device_bsz,
        save_total_limit=1,
        warmup_steps=config['training']['warmup_steps'],
        lr_scheduler_type="cosine",
        learning_rate=float(config['training']['lr']),
        logging_steps=20,
        fp16=config['training']['fp16'],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        torch_compile=config['training'].get('torch_compile', False),
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Check GPU availability
    check_gpu_availability()
    
    # Train and save
    print(f"Trainer is using device: {trainer.args.device}")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()