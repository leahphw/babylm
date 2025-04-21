import os
os.environ["WANDB_DISABLED"] = "true"

from transformers import (
    GPT2TokenizerFast,
    LlamaForCausalLM,
    LlamaConfig,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
import random
from pathlib import Path
from datetime import datetime

from babylm_dataset import BabylmDataset
import consts

"""
AuxiliaryClassifier: A simple classifier for shallow layers that consists of:
    - Linear layer
    - GELU activation
    - Dropout
    - Final linear layer
"""
class AuxiliaryClassifier(nn.Module):
    """Auxiliary classifier for shallow layers in DSKD."""
    def __init__(self, hidden_size, num_labels, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )
        
    def forward(self, hidden_states):
        return self.classifier(hidden_states)

"""
Extended training arguments with:
    - alpha: Weight for class prediction loss
    - beta: Weight for feature map loss
    - temperature: Temperature for softmax in KL divergence
"""
class DSKDTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, beta=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha  # Weight for class prediction loss
        self.beta = beta    # Weight for feature map loss
        self.temperature = temperature

"""
A trainer for DSKD that consists of:
    - Teacher model
    - Auxiliary classifiers for shallow layers
    - Feature projection layer
    - Compute class prediction loss
    - Compute feature map loss
    - Compute adaptive weights
"""
class DSKDTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self._move_model_to_device(self.teacher, self.model.device)
        self.teacher.eval()
        
        # Add auxiliary classifiers for shallow layers
        self.aux_classifiers = nn.ModuleList([
            AuxiliaryClassifier(self.model.config.hidden_size, self.model.config.vocab_size)
            for _ in range(self.model.config.num_hidden_layers - 1)
        ]).to(self.model.device)
        
        # Projection layer for feature map alignment
        self.feature_projection = nn.Linear(
            self.model.config.hidden_size,
            self.teacher.config.hidden_size
        ).to(self.model.device)
        
    """
    Loss Components:
        - compute_class_prediction_loss: KL divergence between teacher and student logits
        - compute_feature_map_loss: MSE between teacher and student features
        - compute_adaptive_weights: Dynamic weighting based on current performance
    """
    def compute_class_prediction_loss(self, student_logits, teacher_logits, weights=None):
        """Compute KL divergence loss for class predictions."""
        if weights is None:
            weights = torch.ones(len(student_logits)) / len(student_logits)
            
        losses = []
        for logits in student_logits:
            loss = F.kl_div(
                F.log_softmax(logits / self.args.temperature, dim=-1),
                F.softmax(teacher_logits / self.args.temperature, dim=-1),
                reduction='batchmean'
            ) * (self.args.temperature ** 2)
            losses.append(loss)
            
        return sum(w * l for w, l in zip(weights, losses))
    
    def compute_feature_map_loss(self, student_features, teacher_features, weights=None):
        """Compute MSE loss for feature maps."""
        if weights is None:
            weights = torch.ones(len(student_features)) / len(student_features)
            
        losses = []
        for features in student_features:
            # Project student features to teacher dimension
            projected = self.feature_projection(features)
            loss = F.mse_loss(projected, teacher_features)
            losses.append(loss)
            
        return sum(w * l for w, l in zip(weights, losses))
    
    def compute_adaptive_weights(self, student_logits, teacher_logits, student_features, teacher_features):
        """Compute adaptive weights for auxiliary classifiers."""
        # Compute weights for class prediction loss
        kld_losses = [
            F.kl_div(
                F.log_softmax(logits / self.args.temperature, dim=-1),
                F.softmax(teacher_logits / self.args.temperature, dim=-1),
                reduction='batchmean'
            ) for logits in student_logits
        ]
        kld_weights = torch.softmax(torch.tensor(kld_losses), dim=0)
        
        # Compute weights for feature map loss
        mse_losses = [
            F.mse_loss(self.feature_projection(features), teacher_features)
            for features in student_features
        ]
        mse_weights = torch.softmax(torch.tensor(mse_losses), dim=0)
        
        return kld_weights, mse_weights
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Get student outputs with hidden states
        outputs_student = model(**inputs, output_hidden_states=True)
        student_loss = outputs_student.loss
        
        # Get teacher outputs
        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs, output_hidden_states=True)
            teacher_logits = outputs_teacher.logits
            teacher_features = outputs_teacher.hidden_states[-1]
        
        # Get logits from auxiliary classifiers
        student_logits = [outputs_student.logits]
        student_features = [outputs_student.hidden_states[-1]]
        
        for i, classifier in enumerate(self.aux_classifiers):
            # Get hidden states from intermediate layers
            hidden_states = outputs_student.hidden_states[i+1]
            # Get logits from auxiliary classifier
            aux_logits = classifier(hidden_states)
            student_logits.append(aux_logits)
            student_features.append(hidden_states)
        
        # Compute adaptive weights
        kld_weights, mse_weights = self.compute_adaptive_weights(
            student_logits, teacher_logits,
            student_features, teacher_features
        )
        
        # Compute losses
        class_pred_loss = self.compute_class_prediction_loss(
            student_logits, teacher_logits, kld_weights
        )
        
        feature_map_loss = self.compute_feature_map_loss(
            student_features, teacher_features, mse_weights
        )
        
        # Combined loss
        loss = (
            student_loss + 
            self.args.alpha * class_pred_loss +
            self.args.beta * feature_map_loss
        )
        
        return (loss, outputs_student) if return_outputs else loss

def create_student(tokenizer, seq_length):
    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=512,
        num_hidden_layers=16,
        intermediate_size=1024,
        num_attention_heads=8,
        bos_token_id=tokenizer.convert_tokens_to_ids("<s>"),
        eos_token_id=tokenizer.convert_tokens_to_ids("</s>"),
        pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
        max_position_embeddings=2 * seq_length,
    )

    student = LlamaForCausalLM(config)
    return student

def check_gpu_availability():
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPU found, using CPU.")
        print("Exiting")
        exit(1)

def main():
    check_gpu_availability()
    random.seed(consts.RANDOM_SEED)

    # Training parameters
    LR = 2.5e-4
    BATCH_SIZE = 32
    SEQ_LENGTH = 128
    TEMPERATURE = 2.0
    ALPHA = 0.5  # Weight for class prediction loss
    BETA = 0.5   # Weight for feature map loss
    EVAL_SAMPLES = 8192

    # Model paths
    TEACHER_DIR = consts.TEACHER_DIR / "Llama-16M"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    STUDENT_NAME = "Baby-Llama-58M-DSKD-1-Teacher"
    STUDENT_OUTPUT = consts.STUDENT_DIR / f"{STUDENT_NAME}_{timestamp}"

    # Setup tokenizer
    tokenizer = GPT2TokenizerFast(tokenizer_file=str(consts.TOKENIZER_PATH))
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "<pad>"
    tokenizer.model_max_length = SEQ_LENGTH

    # Load datasets
    train_dataset = BabylmDataset(
        consts.TRAIN_DATASET_STRICT_PATH,
        SEQ_LENGTH,
        tokenizer=tokenizer,
        random_chunk=True,
    )
    full_eval_dataset = BabylmDataset(
        consts.DEV_DATASET_STRICT_PATH, SEQ_LENGTH, tokenizer=tokenizer, offset=0
    )
    eval_indices = random.sample(range(len(full_eval_dataset)), EVAL_SAMPLES)
    eval_dataset = Subset(full_eval_dataset, eval_indices)
    del full_eval_dataset

    # Load teacher and create student
    teacher = LlamaForCausalLM.from_pretrained(TEACHER_DIR)
    student = create_student(tokenizer, SEQ_LENGTH)

    print(f"model num parameters: student = {student.num_parameters()}")
    print(f"model num parameters: teacher = {teacher.num_parameters()}")

    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments
    training_args = DSKDTrainingArguments(
        output_dir=STUDENT_OUTPUT,
        overwrite_output_dir=True,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        num_train_epochs=6,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=BATCH_SIZE,
        save_total_limit=1,
        report_to=[],
        warmup_steps=200,
        lr_scheduler_type="cosine",
        learning_rate=LR,
        logging_steps=20,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        weight_decay=0.1,
        alpha=ALPHA,
        beta=BETA,
        temperature=TEMPERATURE,
    )

    # Create trainer
    trainer = DSKDTrainer(
        student,
        training_args,
        teacher_model=teacher,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print(f"Trainer is using device: {trainer.args.device}")
    trainer.train()
    trainer.save_model(STUDENT_OUTPUT)
    tokenizer.save_pretrained(STUDENT_OUTPUT)

if __name__ == "__main__":
    main() 