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
from random import sample
import os
from datetime import datetime
os.environ["WANDB_DISABLED"] = "true"
from pathlib import Path
import random

from babylm_dataset import BabylmDataset
import consts

def check_gpu_availability():
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPU found, using CPU.")
        print("Exiting")
        exit(1)

check_gpu_availability()
random.seed(consts.RANDOM_SEED)

#############
LR = 2.5e-4
BATCH_SIZE = 32
SEQ_LENGTH = 128

TEMPERATURE = 2.0
ALPHA = 0.5  # Weight for traditional KD loss
BETA = 0.4   # Weight for hidden layer distillation loss
LAYER_MAPPINGS = {
    'teacher1': {0: 0, 1: 2, 2: 4, 3: 6, 4: 8, 5: 10, 6: 12, 7: 14},  # Llama-16M to Student
    'teacher2': {0: 0, 1: 1, 2: 3, 3: 4, 4: 6, 5: 7, 6: 9, 7: 10, 8: 12, 9: 13, 10: 14, 11: 15}  # GPT2-97M to Student
}
#############


teacher_dir1 = consts.TEACHER_DIR / "Llama-16M"
teacher_dir2 = consts.TEACHER_DIR / "GPT2-97M"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_NAME = f'Baby-Llama-58M'
MODEL_OUTPUT = consts.STUDENT_DIR / f"{MODEL_NAME}_{timestamp}"
EVAL_SAMPLES = 8192


tokenizer_path = consts.TOKENIZER_PATH
tokenizer = GPT2TokenizerFast(tokenizer_file=str(tokenizer_path))
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.pad_token = "<pad>"

train_dataset = BabylmDataset(consts.TRAIN_DATASET_STRICT_PATH, SEQ_LENGTH, tokenizer=tokenizer, random_chunk=True)
full_eval_dataset = BabylmDataset(consts.DEV_DATASET_STRICT_PATH, SEQ_LENGTH, tokenizer=tokenizer, offset=0)

eval_indices = sample(range(len(full_eval_dataset)), EVAL_SAMPLES)
eval_dataset = Subset(full_eval_dataset, eval_indices)

tokenizer.model_max_length = SEQ_LENGTH

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
    output_hidden_states=True,  # Enable output of hidden states
)

# Initialize student model with output_hidden_states=True
student = LlamaForCausalLM(config)

# Load teacher models with output_hidden_states=True
teacher1 = LlamaForCausalLM.from_pretrained(teacher_dir1, output_hidden_states=True)
teacher2 = GPT2LMHeadModel.from_pretrained(teacher_dir2, output_hidden_states=True)
teachers = [teacher1, teacher2]

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

print(f'model num parameters: student = {student.num_parameters()}')
print(f'model num parameters: teacher1 = {teacher1.num_parameters()}')
print(f'model num parameters: teacher2 = {teacher2.num_parameters()}')


# Projection layers for feature alignment between different architectures
class FeatureProjection(nn.Module):
    def __init__(self, teacher_dim, student_dim):
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim)
        
    def forward(self, x):
        return self.proj(x)


class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, beta=0.4, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature


class DeepSupervisionDistillationTrainer(Trainer):
    def __init__(self, *args, teacher_models=None, layer_mappings=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teachers = teacher_models
        self.layer_mappings = layer_mappings
        
        # Create feature projections for each mapped layer to align dimensions
        self.projections = {}
        self.student_dim = self.model.config.hidden_size
        
        for teacher_idx, teacher in enumerate(self.teachers):
            teacher_key = f'teacher{teacher_idx+1}'
            teacher_dim = teacher.config.hidden_size
            
            if teacher_key in self.layer_mappings:
                self.projections[teacher_key] = {}
                for _, student_layer in self.layer_mappings[teacher_key].items():
                    # Store projection direction for each teacher
                    projection_direction = 'student_to_teacher' if teacher_dim > self.student_dim else 'teacher_to_student'
                    
                    if projection_direction == 'student_to_teacher':
                        projection = FeatureProjection(teacher_dim, self.student_dim)
                    else:
                        projection = FeatureProjection(self.student_dim, teacher_dim)
                    
                    # Store both the projection and direction
                    self.projections[teacher_key][student_layer] = {
                        'proj': projection.to(self.model.device),
                        'direction': projection_direction
                    }
        
        # Make sure each teacher is in eval mode and on the right device
        for teacher in self.teachers:
            self._move_model_to_device(teacher, self.model.device)
            teacher.eval()  # Set teacher to evaluation mode
    
    def compute_hidden_state_loss(self, teacher_hidden_states, student_hidden_states, teacher_key):
        """Compute the MSE loss between teacher and student hidden states."""
        total_hidden_loss = 0
        valid_layer_pairs = 0
        
        for teacher_layer, student_layer in self.layer_mappings[teacher_key].items():
            # Check if the layer indices are valid
            if teacher_layer >= len(teacher_hidden_states) or student_layer >= len(student_hidden_states):
                print(f"Warning: Layer mapping {teacher_layer}->{student_layer} is out of range. Skipping.")
                continue
                
            # Get hidden states from the specified layers
            teacher_state = teacher_hidden_states[teacher_layer]
            student_state = student_hidden_states[student_layer]
            
            # Project student hidden state to match teacher dimensions
            proj_info = self.projections[teacher_key][student_layer]
            projection = proj_info['proj']
            direction = proj_info['direction']
            # Compute MSE loss
            if direction == 'student_to_teacher':
                projected_student = projection(student_state)
                layer_loss = F.mse_loss(projected_student, teacher_state)
            else:
                projected_teacher = projection(teacher_state)
                layer_loss = F.mse_loss(student_state, projected_teacher)
            total_hidden_loss += layer_loss
            valid_layer_pairs += 1
            
        if valid_layer_pairs == 0:
            return torch.tensor(0.0).to(student_hidden_states[0].device)
            
        # Average the loss across all valid layer pairs
        avg_hidden_loss = total_hidden_loss / valid_layer_pairs
        return avg_hidden_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Forward pass with student model, getting hidden states
        outputs_student = model(**inputs, output_hidden_states=True)
        student_loss = outputs_student.loss
        student_hidden_states = outputs_student.hidden_states
        
        # Compute teacher outputs and hidden states
        with torch.no_grad():
            all_teacher_logits = []
            all_hidden_losses = []
            
            for teacher_idx, teacher in enumerate(self.teachers):
                teacher_key = f'teacher{teacher_idx+1}'
                outputs_teacher = teacher(**inputs, output_hidden_states=True)
                all_teacher_logits.append(outputs_teacher.logits)
                
                # If this teacher has layer mappings, compute hidden state loss
                if teacher_key in self.layer_mappings:
                    hidden_loss = self.compute_hidden_state_loss(
                        outputs_teacher.hidden_states,
                        student_hidden_states,
                        teacher_key
                    )
                    all_hidden_losses.append(hidden_loss)
            
            avg_teacher_logits = torch.stack(all_teacher_logits).mean(dim=0)
            
        # Calculate average hidden state loss if any
        hidden_loss = torch.stack(all_hidden_losses).mean() if all_hidden_losses else torch.tensor(0.0).to(model.device)
        
        # Traditional KD loss from logits
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (
            loss_function(
                F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
                F.softmax(avg_teacher_logits / self.args.temperature, dim=-1),
            )
            * (self.args.temperature ** 2)
        )
        
        # Combine losses: original CE loss + KD loss + hidden state loss
        loss = (
            student_loss + 
            self.args.alpha * loss_logits + 
            self.args.beta * hidden_loss
        )
        
        if return_outputs:
            return loss, outputs_student
        return loss



training_args = DistillationTrainingArguments(
    output_dir=MODEL_OUTPUT,
    overwrite_output_dir=True,
    save_strategy="epoch",
    eval_strategy="epoch",
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


trainer = DeepSupervisionDistillationTrainer(
    student,
    training_args,
    teacher_models=teachers,
    layer_mappings=LAYER_MAPPINGS,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)


trainer.train()


trainer.save_model(MODEL_OUTPUT)
tokenizer.save_pretrained(MODEL_OUTPUT)