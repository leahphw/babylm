import os
import itertools
import json
import argparse
import torch
import numpy as np
import wandb
from datetime import datetime
from torch.utils.data import Subset
from torch.distributed import init_process_group, destroy_process_group

from transformers import (
    Trainer,
    TrainingArguments,
)

from configurable import (
    check_gpu_availability,
    load_config,
    load_dataset,
    load_models_from_config,
)

# Import distillation components
from transformers import Trainer
import consts
from dkds_noaux import (
    DistillationTrainingArguments,
    DeepSupervisionDistillationTrainer,
)

def get_args():
    parser = argparse.ArgumentParser(description="Grid search for distillation weights")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output_dir", type=str, required=True, help="Base output directory for saving results")
    parser.add_argument("--use_wandb", action="store_true", help="Whether to use wandb for logging")
    parser.add_argument("--wandb_project", type=str, default="distillation-grid-search", help="WandB project name")
    parser.add_argument("--logit_weights", type=str, default="0.1,0.3,0.5,0.7,0.9", help="Comma-separated logit distillation weights")
    parser.add_argument("--hidden_weights", type=str, default="0.1,0.3,0.5,0.7,0.9", help="Comma-separated hidden distillation weights")
    parser.add_argument("--num_epochs", type=int, default=None, help="Override number of epochs in config")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size in config")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--train_fraction", type=float, default=1.0, help="Fraction of the training dataset for speed (0,1 = 10%)")
    return parser.parse_args()

def setup_distributed():
    """Initialize the distributed environment."""
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def get_rank():
    """Get the rank of the current process."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0

def is_main_process():
    """Check if this is the main process."""
    return get_rank() == 0

def run_grid_search(args):
    # Load base configuration
    config = load_config(args.config)
    
    # Parse grid search parameters
    logit_weights = [0.1, 0.3, 0.5, 0.7, 0.9]
    hidden_weights = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    # Create product of all parameter combinations
    param_grid = [(l,h) for l in logit_weights for h in hidden_weights]

    half = len(param_grid) // 2
    second_half = param_grid[half:]

    
    # Create timestamp for unique run ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup base output directory
    base_output_dir = os.path.join(
        args.output_dir, 
        f"grid_search_{timestamp}"
    )
    
    # Only create directories on the main process
    if is_main_process():
        os.makedirs(base_output_dir, exist_ok=True)
        # Save grid search parameters
        with open(os.path.join(base_output_dir, "grid_search_params.json"), "w") as f:
            json.dump({
                "logit_weights": logit_weights,
                "hidden_weights": hidden_weights,
                "timestamp": timestamp,
                "config_path": args.config
            }, f, indent=2)
    
    # Load tokenizer and models once
    tokenizer, student, teachers = load_models_from_config(config)
    
    # Load dataset once
    train_dataset, eval_dataset, data_collator = load_dataset(config, tokenizer)
    
    # To increase the speed of grid search 
    train_fraction = max(min(args.train_fraction, 1.0), 0.0)   # safety clamp
    if train_fraction < 1.0:
        subset_size  = int(len(train_dataset) * train_fraction)
        rng          = np.random.default_rng(seed=consts.RANDOM_SEED)
        subset_idx   = rng.choice(len(train_dataset), size=subset_size, replace=False)
        train_dataset = Subset(train_dataset, subset_idx.tolist())
    
    # Setup layer mappings
    layer_mappings = {}
    for i, teacher in enumerate(config["teachers"]):
        layer_mappings[f'teacher{i+1}'] = teacher["layer_mappings"]
    
    # Results dictionary to store metrics
    results = {}
    
    # Loop through parameter combinations
    for idx, (logit_weight, hidden_weight) in enumerate(second_half):
        # Skip combinations that would make weights sum > 1.0
        # Assuming hard_target_loss_weight is fixed at some value
        hard_target_weight = config["training"]["hard_target_loss_weight"]
        
        # Run ID for this specific parameter combination
        run_id = f"logit{logit_weight}_hidden{hidden_weight}"
        run_output_dir = os.path.join(base_output_dir, run_id)
        
        if is_main_process():
            print(f"\n{'='*50}")
            print(f"Running combination {idx+1}/{len(second_half)}: logit_weight={logit_weight}, hidden_weight={hidden_weight}")
            print(f"{'='*50}\n")
        
        
        # Update config with the current parameters
        config["training"]["logit_distillation_loss_weight"] = logit_weight
        config["training"]["hidden_distillation_loss_weight"] = hidden_weight
        
        # Override epochs and batch size if specified
        if args.num_epochs is not None:
            config["training"]["num_epochs"] = args.num_epochs
        if args.batch_size is not None:
            config["training"]["batch_size"] = args.batch_size
        
        # Create copy of student model for this run
        run_student = type(student)(student.config)
        run_student.load_state_dict(student.state_dict())
        
        # Set up training arguments
        training_args = DistillationTrainingArguments(
            output_dir=run_output_dir,
            overwrite_output_dir=True,
            save_strategy="epoch",
            eval_strategy="epoch",
            num_train_epochs=config["training"]["num_epochs"],
            gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
            per_device_train_batch_size=config["training"]["batch_size"],
            save_total_limit=1,
            report_to=[],
            warmup_steps=config["training"]["warmup_steps"],
            lr_scheduler_type="cosine",
            learning_rate=config["training"]["lr"],
            logging_steps=50,
            fp16=config["training"]["fp16"],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            weight_decay=config["training"]["weight_decay"],
            hard_target_loss_weight=hard_target_weight,
            logit_distillation_loss_weight=logit_weight,
            hidden_distillation_loss_weight=hidden_weight,
            temperature=config["training"]["temperature"],
            # Distributed training settings
            local_rank=args.local_rank,
            ddp_find_unused_parameters=False
        )
        
        # Set up trainer
        trainer = DeepSupervisionDistillationTrainer(
            run_student,
            training_args,
            teacher_models=teachers,
            layer_mappings=layer_mappings,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        # Train the model
        train_result = trainer.train()
        
        # Evaluate the model
        eval_result = trainer.evaluate()
        
        trainer.save_model(run_output_dir)
        tokenizer.save_pretrained(run_output_dir)

        # Save the model and results only on main process
        if is_main_process():
            tokenizer.save_pretrained(run_output_dir)
            
            # Log results
            results[run_id] = {
                "params": {
                    "logit_weight": logit_weight, 
                    "hidden_weight": hidden_weight,
                    "hard_target_weight": hard_target_weight
                },
                "train_results": train_result.metrics,
                "eval_results": eval_result
            }
            
            # Save results for this run
            with open(os.path.join(run_output_dir, "results.json"), "w") as f:
                json.dump(results[run_id], f, indent=2)
    
    # Save overall results on main process
    if is_main_process():
        # Find best configuration based on eval loss
        best_run_id = min(results, key=lambda x: results[x]["eval_results"]["eval_loss"])
        best_params = results[best_run_id]["params"]
        best_eval_loss = results[best_run_id]["eval_results"]["eval_loss"]
        
        summary = {
            "best_run": best_run_id,
            "best_params": best_params,
            "best_eval_loss": best_eval_loss,
            "all_results": results
        }
        
        with open(os.path.join(base_output_dir, "grid_search_results.json"), "w") as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*80)
        print(f"Grid search completed. Best configuration:")
        print(f"logit_weight: {best_params['logit_weight']}, hidden_weight: {best_params['hidden_weight']}")
        print(f"Eval loss: {best_eval_loss}")
        print("="*80 + "\n")
    
    # Cleanup distributed processes
    if torch.distributed.is_initialized():
        destroy_process_group()

def main():
    args = get_args()
    print("GRID IS RUNNING")
    # Setup distributed training
    if args.local_rank != -1:
        setup_distributed()
    
    # Check GPU availability
    check_gpu_availability()
    
    # Set random seed for reproducibility
    torch.manual_seed(consts.RANDOM_SEED)
    np.random.seed(consts.RANDOM_SEED)
    
    # Run the grid search
    run_grid_search(args)

if __name__ == "__main__":
    main()