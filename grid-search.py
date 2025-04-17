import torch
import os
from datetime import datetime
from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
import wandb
import json

# Import your existing classes and modules
from transformers import (
    GPT2TokenizerFast,
    LlamaForCausalLM,
    LlamaConfig,
    GPT2LMHeadModel,
)
# Import your custom modules
from babylm_dataset import BabylmDataset
import consts
from dkds_noaux import DeepSupervisionDistillationTrainer, DistillationTrainingArguments

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("grid_search.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_gpu_availability():
    if torch.cuda.is_available():
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        return True
    else:
        logger.error("No GPU found, using CPU.")
        return False

def run_grid_search():
    # Check GPU availability
    if not check_gpu_availability():
        logger.error("Exiting due to no GPU availability")
        return
    
    # Grid search parameters
    param_grid = {
        'alpha': [0.3, 0.5, 0.7],  # Weight for traditional KD loss
        'beta': [0.2, 0.4, 0.6],   # Weight for hidden layer distillation loss
        'temperature': [1.5, 2.0, 2.5]  # Optional: add temperature to grid search
    }
    
    # Fixed hyperparameters
    BATCH_SIZE = 16  # Reduced batch size for grid search
    SEQ_LENGTH = 128
    LR = 2.5e-4
    EPOCHS = 2  # Reduced epochs for grid search
    
    # Layer mappings 
    LAYER_MAPPINGS = {
        'teacher1': {0: 0, 3: 1, 6: 3, 9: 5, 12: 7, 15: 9, 18: 11, 21: 13, 23: 15},
        'teacher2': {0: 0, 3: 2, 6: 4, 9: 6, 12: 8, 15: 10, 18: 12, 21: 14, 23: 15}
    }
    
    # Load teacher models
    teacher_dir1 = consts.TEACHER_DIR / "BabyLlama1-Teacher-Llama-360M-strict"
    teacher_dir2 = consts.TEACHER_DIR / "BabyLlama1-Teacher-GPT2-705M-strict"
    
    # Load tokenizer
    tokenizer_path = consts.TOKENIZER_PATH
    tokenizer = GPT2TokenizerFast(tokenizer_file=str(tokenizer_path))
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "<pad>"
    tokenizer.model_max_length = SEQ_LENGTH
    
    # Load datasets
    train_dataset = BabylmDataset(consts.TRAIN_DATASET_STRICT_PATH, SEQ_LENGTH, tokenizer=tokenizer, random_chunk=True)
    full_eval_dataset = BabylmDataset(consts.DEV_DATASET_STRICT_PATH, SEQ_LENGTH, tokenizer=tokenizer, offset=0)
    
    # Use a smaller subset for grid search
    GRID_SEARCH_TRAIN_SIZE = min(10000, len(train_dataset))
    GRID_SEARCH_EVAL_SIZE = min(2000, len(full_eval_dataset))
    
    train_indices = np.random.choice(len(train_dataset), GRID_SEARCH_TRAIN_SIZE, replace=False)
    eval_indices = np.random.choice(len(full_eval_dataset), GRID_SEARCH_EVAL_SIZE, replace=False)
    
    grid_train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    grid_eval_dataset = torch.utils.data.Subset(full_eval_dataset, eval_indices)
    
    # Create student config
    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=512,
        num_hidden_layers=16,
        intermediate_size=1024,
        num_attention_heads=8,
        bos_token_id=tokenizer.convert_tokens_to_ids("<s>"),
        eos_token_id=tokenizer.convert_tokens_to_ids("</s>"),
        pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
        max_position_embeddings=2*SEQ_LENGTH,
        output_hidden_states=True,
    )
    
    # Initialize data collector
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Setup for tracking results
    results = []
    
    # Optional: Setup WandB for experiment tracking
    wandb.init(project="knowledge-distillation-grid-search")
    
    # Run grid search
    for params in ParameterGrid(param_grid):
        logger.info(f"Starting run with params: {params}")
        
        # Create unique model name for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f'Baby-Llama-58M-gridsearch-a{params["alpha"]}-b{params["beta"]}-t{params["temperature"]}'
        model_output = consts.GRID_SEARCH_DIR / f"{model_name}_{timestamp}"
        
        # Initialize student model
        student = LlamaForCausalLM(config)
        
        # Load teacher models with output_hidden_states=True
        teacher1 = LlamaForCausalLM.from_pretrained(teacher_dir1, output_hidden_states=True)
        teacher2 = GPT2LMHeadModel.from_pretrained(teacher_dir2, output_hidden_states=True)
        teachers = [teacher1, teacher2]
        
        # Print model parameters
        logger.info(f'Model parameters: student = {student.num_parameters()}')
        logger.info(f'Model parameters: teacher1 = {teacher1.num_parameters()}')
        logger.info(f'Model parameters: teacher2 = {teacher2.num_parameters()}')
        
        # Setup training arguments
        training_args = DistillationTrainingArguments(
            output_dir=model_output,
            overwrite_output_dir=True,
            save_strategy="epoch",
            eval_strategy="epoch",
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            save_total_limit=1,
            report_to=[],
            warmup_steps=100,
            lr_scheduler_type="cosine",
            learning_rate=LR,
            logging_steps=20,
            fp16=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            weight_decay=0.1,
            alpha=params["alpha"],
            beta=params["beta"],
            temperature=params["temperature"],
        )
        
        # Initialize trainer
        trainer = DeepSupervisionDistillationTrainer(
            student,
            training_args,
            teacher_models=teachers,
            layer_mappings=LAYER_MAPPINGS,
            data_collator=data_collator,
            train_dataset=grid_train_dataset,
            eval_dataset=grid_eval_dataset,
        )
        
        # Train model
        train_result = trainer.train()
        
        # Get evaluation metrics
        eval_result = trainer.evaluate()
        
        # Log results
        run_results = {
            "alpha": params["alpha"],
            "beta": params["beta"],
            "temperature": params["temperature"],
            "eval_loss": eval_result["eval_loss"],
            "perplexity": np.exp(eval_result["eval_loss"]),
            "train_runtime": train_result.metrics["train_runtime"],
        }
        
        results.append(run_results)
        logger.info(f"Run completed. Results: {run_results}")
        
        # Log to WandB
        wandb.log(run_results)
        
        # Save this run's specific results
        with open(model_output / "grid_search_results.json", "w") as f:
            json.dump(run_results, f)
    
    # Save all results
    results_dir = Path("grid_search_results")
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        json.dump(results, f)
    
    # Find best hyperparameters
    best_result = min(results, key=lambda x: x["eval_loss"])
    logger.info(f"Best hyperparameters: {best_result}")
    
    # Visualize results
    visualize_results(results)
    
    return best_result

def visualize_results(results):
    """Create visualizations of grid search results"""
    # Extract unique parameter values
    alphas = sorted(set(r["alpha"] for r in results))
    betas = sorted(set(r["beta"] for r in results))
    temperatures = sorted(set(r["temperature"] for r in results))
    
    # Create heatmap of eval_loss for each temperature
    for temp in temperatures:
        temp_results = [r for r in results if r["temperature"] == temp]
        loss_matrix = np.zeros((len(alphas), len(betas)))
        
        for r in temp_results:
            i = alphas.index(r["alpha"])
            j = betas.index(r["beta"])
            loss_matrix[i, j] = r["eval_loss"]
        
        plt.figure(figsize=(10, 8))
        heatmap = plt.pcolor(loss_matrix, cmap='viridis_r')
        plt.colorbar(heatmap, label='Eval Loss')
        
        plt.xticks(np.arange(len(betas)) + 0.5, betas)
        plt.yticks(np.arange(len(alphas)) + 0.5, alphas)
        plt.xlabel('Beta (Hidden State Loss Weight)')
        plt.ylabel('Alpha (KD Loss Weight)')
        plt.title(f'Eval Loss Heatmap (Temperature = {temp})')
        
        # Save figure
        plt.savefig(f'heatmap_temp_{temp}.png')
        plt.close()
    
    # Create summary plot
    plt.figure(figsize=(12, 8))
    for temp in temperatures:
        temp_results = [r for r in results if r["temperature"] == temp]
        # Sort by alpha, then beta for line plot
        temp_results.sort(key=lambda x: (x["alpha"], x["beta"]))
        
        x_labels = [f"α={r['alpha']},β={r['beta']}" for r in temp_results]
        losses = [r["eval_loss"] for r in temp_results]
        
        plt.plot(x_labels, losses, marker='o', label=f'Temp={temp}')
    
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Hyperparameter Combination')
    plt.ylabel('Evaluation Loss')
    plt.title('Comparison of Hyperparameter Combinations')
    plt.legend()
    plt.tight_layout()
    plt.savefig('grid_search_summary.png')
    plt.close()

if __name__ == "__main__":
    best_params = run_grid_search()
    print(f"Best hyperparameters found: alpha={best_params['alpha']}, beta={best_params['beta']}, temperature={best_params['temperature']}")
    print(f"Best eval loss: {best_params['eval_loss']}, perplexity: {best_params['perplexity']}")