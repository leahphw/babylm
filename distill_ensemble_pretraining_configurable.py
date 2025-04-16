import argparse
from transformers import (
    GPT2TokenizerFast,
    LlamaForCausalLM,
    LlamaConfig,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    GPTJConfig,
    GPTJForCausalLM,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
import random
from random import sample

from pathlib import Path

import yaml
import consts
import wandb


from babylm_dataset import BabylmDataset


#  Distillation Trainer
#  We modified the Trainer from this repo https://github.com/philschmid/knowledge-distillation-transformers-pytorch-sagemaker
# to work with an ensemble of teachers


class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_models=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teachers = teacher_models
        for teacher in self.teachers:
            # place each teacher on same device as student
            self._move_model_to_device(teacher, self.model.device)
            teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False):
        # compute student output
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss

        # compute teacher output
        with torch.no_grad():
            all_teacher_logits = []
            for teacher in self.teachers:
                outputs_teacher = teacher(**inputs)
                all_teacher_logits.append(outputs_teacher.logits)
            avg_teacher_logits = torch.stack(all_teacher_logits).mean(dim=0)

        # assert size
        assert outputs_student.logits.size() == avg_teacher_logits.size()

        # Soften probabilities and compute distillation loss
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = loss_function(
            F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
            F.softmax(avg_teacher_logits / self.args.temperature, dim=-1),
        ) * (self.args.temperature**2)
        # Return weighted student loss
        loss = self.args.alpha * student_loss + (1.0 - self.args.alpha) * loss_logits
        return (loss, outputs_student) if return_outputs else loss


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


def load_config() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Configuration file path. Example: ./config/gpt-97M.yaml",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print(f"Loaded config: {args.config}")

    # Update output path
    output_dir = Path(config["student"]["output_dir"])
    config["student"]["output_path"] = output_dir / config["student"]["name"]
    del config["student"]["output_dir"]
    return config


def dynamic_model_creation(model_config: dict, tokenizer: GPT2TokenizerFast):

    model_type = model_config["type"]
    if model_type == "Llama":
        model_config = LlamaConfig(
            vocab_size=tokenizer.vocab_size,
            max_position_embeddings=2 * tokenizer.model_max_length,
            hidden_size=model_config["hidden_size"],
            intermediate_size=model_config["intermediate_size"],
            num_hidden_layers=model_config["n_layer"],
            num_attention_heads=model_config["n_head"],
            tie_word_embeddings=model_config.get("tie_word_embeddings", False),
            pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
        )
        model = LlamaForCausalLM(model_config)
    elif model_type == "GPT2":
        model_config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions=2 * tokenizer.model_max_length,
            n_embd=model_config["hidden_size"],
            n_layer=model_config["n_layer"],
            n_head=model_config["n_head"],
            resid_pdrop=model_config["resid_pdrop"],
            embd_pdrop=model_config["embd_pdrop"],
            attn_pdrop=model_config["attn_pdrop"],
            pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
        )
        model = GPT2LMHeadModel(model_config)
    elif model_type == "GPTJ":
        model_config = GPTJConfig(
            vocab_size=tokenizer.vocab_size,
            n_positions=2 * tokenizer.model_max_length,
            n_embd=model_config["hidden_size"],
            n_layer=model_config["n_layer"],
            n_head=model_config["n_head"],
            resid_pdrop=model_config["resid_pdrop"],
            embd_pdrop=model_config["embd_pdrop"],
            attn_pdrop=model_config["attn_pdrop"],
            tie_word_embeddings=model_config["tie_word_embeddings"],
            pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
        )
        model = GPTJForCausalLM(model_config)
    else:
        raise ValueError(f"Model type '{model_type}' not supported")

    return model


def load_models_from_config(config: dict):
    tokenizer = GPT2TokenizerFast(tokenizer_file=config["data"]["tokenizer_path"])
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "<pad>"
    tokenizer.model_max_length = config["data"]["seq_length"]

    student = dynamic_model_creation(config["student"], tokenizer)

    teachers = []
    for teacher in config["teachers"]:
        if teacher["type"] == "Llama":
            teacher_model = LlamaForCausalLM.from_pretrained(teacher["path"])
        elif teacher["type"] == "GPT2":
            teacher_model = GPT2LMHeadModel.from_pretrained(teacher["path"])
        else:
            raise ValueError("Model type '{model_type}' not supported")

        teachers.append(teacher_model)

    return tokenizer, student, teachers


def load_training_args_from_config(config: dict):

    training_config = config["training"]

    args = DistillationTrainingArguments(
        output_dir=config["student"]["output_path"],
        overwrite_output_dir=True,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        num_train_epochs=training_config["num_epochs"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        per_device_train_batch_size=training_config["batch_size"],
        save_total_limit=1,  # Set to zero to avoid saving
        report_to="wandb",
        warmup_steps=training_config["warmup_steps"],
        lr_scheduler_type="cosine",
        learning_rate=training_config["lr"],
        logging_steps=20,
        fp16=training_config["fp16"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        weight_decay=training_config["weight_decay"],
        alpha=training_config["hard_target_loss_weight"],
        temperature=training_config["temperature"],
    )
    return args


def main():
    config = load_config()
    check_gpu_availability()
    random.seed(consts.RANDOM_SEED)
    wandb_log = False
    if wandb_log:
        wandb.login()
        wandb.init(project="babylm", name=config["student"]["name"])

    tokenizer, student, teachers = load_models_from_config(config)

    print(f"model num parameters: student = {student.num_parameters()}")
    for i, teacher in enumerate(teachers):
        teacher_name = config["teachers"][i]["path"].split("/")[-1]
        print(f"model num parameters: teacher {i} {teacher_name} = {teacher.num_parameters()//(10**6)}M")

    # in the original code I had random_chunk = False
    # random_chunk=True is expected to improve the model performance a bit
    train_dataset = BabylmDataset(
        config["data"]["train_path"],
        config["data"]["seq_length"],
        tokenizer=tokenizer,
        random_chunk=True,
    )
    full_eval_dataset = BabylmDataset(
        config["data"]["eval_path"],
        config["data"]["seq_length"],
        tokenizer=tokenizer,
        offset=0,
    )

    eval_indices = sample(range(len(full_eval_dataset)), config["data"]["eval_samples"])
    eval_dataset = Subset(full_eval_dataset, eval_indices)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = load_training_args_from_config(config)

    trainer = DistillationTrainer(
        student,
        training_args,
        teacher_models=teachers,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    trainer.save_model(config["student"]["output_path"])
    tokenizer.save_pretrained(config["student"]["output_path"])


if __name__ == "__main__":
    main()
