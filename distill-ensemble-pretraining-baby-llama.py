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

# import wandb


from babylm_dataset import BabylmDataset
import consts
from datetime import datetime

#  Distillation Trainer
#  We modified the Trainer from this repo https://github.com/philschmid/knowledge-distillation-transformers-pytorch-sagemaker
# to work with an ensemble of teachers


class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, contrastive_weight=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight


class ContrastiveDistillationTrainer(Trainer):
    def __init__(self, *args, teacher_models=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teachers = teacher_models
        for teacher in self.teachers:
            self._move_model_to_device(teacher, self.model.device)
            teacher.eval()
        
        # Define projection heads for contrastive learning
        self.student_projection = nn.Linear(512, 256).to(self.model.device)  # Project to lower dimension
        self.teacher_projection = nn.Linear(512, 256).to(self.model.device)
        
    def compute_contrastive_loss(self, student_hidden, teacher_hidden):
        # Project hidden states to lower dimension
        student_proj = self.student_projection(student_hidden)
        teacher_proj = self.teacher_projection(teacher_hidden)
        
        # Normalize projections
        student_proj = F.normalize(student_proj, dim=-1)
        teacher_proj = F.normalize(teacher_proj, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(student_proj, teacher_proj.transpose(-2, -1))
        
        # Temperature scaling
        similarity = similarity / self.args.temperature
        
        # Create labels (diagonal matrix)
        labels = torch.arange(similarity.size(0), device=similarity.device)
        
        # Compute contrastive loss
        loss = F.cross_entropy(similarity, labels)
        return loss

    def compute_loss(self, model, inputs, return_outputs=False):
        # Get student outputs with hidden states
        outputs_student = model(**inputs, output_hidden_states=True)
        student_loss = outputs_student.loss
        
        # Get teacher outputs with hidden states
        with torch.no_grad():
            all_teacher_logits = []
            all_teacher_hidden = []
            for teacher in self.teachers:
                outputs_teacher = teacher(**inputs, output_hidden_states=True)
                all_teacher_logits.append(outputs_teacher.logits)
                all_teacher_hidden.append(outputs_teacher.hidden_states[-1])  # Last layer hidden states
            
            avg_teacher_logits = torch.stack(all_teacher_logits).mean(dim=0)
            avg_teacher_hidden = torch.stack(all_teacher_hidden).mean(dim=0)

        # Standard distillation loss
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = loss_function(
            F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
            F.softmax(avg_teacher_logits / self.args.temperature, dim=-1),
        ) * (self.args.temperature**2)
        
        # Contrastive loss on hidden states
        contrastive_loss = self.compute_contrastive_loss(
            outputs_student.hidden_states[-1],  # Last layer hidden states
            avg_teacher_hidden
        )
        
        # Combined loss
        loss = (
            self.args.alpha * student_loss + 
            (1.0 - self.args.alpha) * loss_logits +
            self.args.contrastive_weight * contrastive_loss
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

        print("To use all GPUs run: torchrun --nproc_per_node=2 train.py")
    else:
        print("No GPU found, using CPU.")
        print("Exiting")
        exit(1)

    assert (
        torch.cuda.device_count() == 2
    ), "Using too many GPUs, professor will not be happy"


def main():
    check_gpu_availability()
    random.seed(consts.RANDOM_SEED)

    #############
    LR = 2.5e-4
    BATCH_SIZE = 32
    SEQ_LENGTH = 128

    TEMPERATURE = 2.0
    ALPHA = 0.5
    CONTRASTIVE_WEIGHT = 0.1  # Weight for contrastive loss

    EVAL_SAMPLES = 8192
    #############

    TEACHER_DIR1 = consts.TEACHER_DIR / "Llama-16M"
    TEACHER_DIR2 = Path("/scratch/nlp_G1/teachers/GPT2-97M")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    STUDENT_NAME = "Baby-Llama-58M"
    STUDENT_OUTPUT = consts.TEACHER_DIR / f"{STUDENT_NAME}_{timestamp}"

    wandb_log = False

    tokenizer = GPT2TokenizerFast(tokenizer_file=str(consts.TOKENIZER_PATH))
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "<pad>"
    tokenizer.model_max_length = SEQ_LENGTH

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

    # Load teachers with error handling
    teachers = []
    try:
        teacher1 = LlamaForCausalLM.from_pretrained(TEACHER_DIR1)
        teachers.append(teacher1)
        print(f"Successfully loaded teacher1 from {TEACHER_DIR1}")
    except Exception as e:
        print(f"Failed to load teacher1: {e}")
    
    try:
        teacher2 = GPT2LMHeadModel.from_pretrained(TEACHER_DIR2)
        teachers.append(teacher2)
        print(f"Successfully loaded teacher2 from {TEACHER_DIR2}")
    except Exception as e:
        print(f"Failed to load teacher2: {e}")

    if not teachers:
        raise RuntimeError("No teacher models could be loaded. Please check the model directories.")

    student = create_student(tokenizer, SEQ_LENGTH)

    print(f"model num parameters: student = {student.num_parameters()}")
    for i, teacher in enumerate(teachers):
        print(f"model num parameters: teacher{i+1} = {teacher.num_parameters()}")

    if wandb_log:
        wandb.login()
        wandb.init(project="babylm", name=STUDENT_NAME)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = DistillationTrainingArguments(
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
        temperature=TEMPERATURE,
        contrastive_weight=CONTRASTIVE_WEIGHT,
    )

    trainer = ContrastiveDistillationTrainer(
        student,
        training_args,
        teacher_models=teachers,
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
