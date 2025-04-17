from transformers import (
    Trainer,
    TrainingArguments,
)
import torch
import torch.nn as nn
import torch.nn.functional as F

from distill_ensemble_pretraining_configurable import (
    check_gpu_availability,
    load_config,
    load_dataset,
    load_models_from_config,
    load_training_args_from_config,
)

import random

import consts


# Projection layers for feature alignment between different architectures
class FeatureProjection(nn.Module):
    def __init__(self, teacher_dim, student_dim):
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim)

    def forward(self, x):
        return self.proj(x)


class DistillationTrainingArguments(TrainingArguments):
    def __init__(
        self,
        *args,
        hard_target_loss_weight=1.0,
        logit_distillation_loss_weight=0.5,
        hidden_distillation_loss_weight=0.4,
        temperature=2.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hard_target_loss_weight = hard_target_loss_weight
        self.logit_distillation_loss_weight = logit_distillation_loss_weight
        self.hidden_distillation_loss_weight = hidden_distillation_loss_weight
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
            self.args.hard_target_loss_weight * student_loss +
            self.args.logit_distillation_loss_weight *  loss_logits + 
            self.args.hidden_distillation_loss_weight *  hidden_loss
        )
        
        if return_outputs:
            return loss, outputs_student
        return loss

def load_training_args_from_config(config:dict):

    training_config = config["training"]

    return DistillationTrainingArguments(
        output_dir=config["student"]["output_path"],
        overwrite_output_dir=True,
        save_strategy="epoch",
        eval_strategy="epoch",
        num_train_epochs=training_config["num_epochs"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        per_device_train_batch_size=training_config["batch_size"],
        save_total_limit=1,
        report_to=[],
        warmup_steps=training_config["warmup_steps"],
        lr_scheduler_type="cosine",
        learning_rate=training_config["lr"],
        logging_steps=20,
        fp16=training_config["fp16"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        weight_decay=training_config["weight_decay"],
        hard_target_loss_weight=training_config["hard_target_loss_weight"],
        logit_distillation_loss_weight=training_config["logit_distillation_loss_weight"],
        hidden_distillation_loss_weight=training_config["hidden_distillation_loss_weight"],
        temperature=training_config["temperature"],
    )


#############
# LR = 2.5e-4
# BATCH_SIZE = 32
# SEQ_LENGTH = 128

# TEMPERATURE = 2.0
# ALPHA = 0.5  # Weight for traditional KD loss
# BETA = 0.4   # Weight for hidden layer distillation loss
# LAYER_MAPPINGS = {
#     'teacher1': {0: 0, 1: 2, 2: 4, 3: 6, 4: 8, 5: 10, 6: 12, 7: 14},  # Llama-16M to Student
#     'teacher2': {0: 0, 1: 1, 2: 3, 3: 4, 4: 6, 5: 7, 6: 9, 7: 10, 8: 12, 9: 13, 10: 14, 11: 15}  # GPT2-97M to Student
# }
#############


def main():

    config = load_config()
    check_gpu_availability()
    random.seed(consts.RANDOM_SEED)
    wandb_log = False
    if wandb_log:
        wandb.login()
        wandb.init(project="babylm", name=config["student"]["name"])

    tokenizer, student, teachers = load_models_from_config(config)

    train_dataset, eval_dataset, data_collator = load_dataset(config, tokenizer)

    training_args = load_training_args_from_config(config)

    layer_mappings = {}
    for i,teacher in enumerate(config["teachers"]):
        layer_mappings[f'teacher{i+1}'] = teacher["layer_mappings"]


    trainer = DeepSupervisionDistillationTrainer(
        student,
        training_args,
        teacher_models=teachers,
        layer_mappings=layer_mappings,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    trainer.save_model(config["student"]["output_path"])
    tokenizer.save_pretrained(config["student"]["output_path"])


if __name__ == "__main__":
    main()
