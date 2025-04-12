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
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
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
    def __init__(
        self,
        *args,
        layers: tuple[int, ...],
        layer_loss_weight=1.0,
        logit_loss_weight=1.0,
        hard_target_loss_weight=1.0,
        temperature=2.0,
        **kwargs,
    ):
        """Creates a modified training arguments class for distillation training.

        Args:
            *args: Variable length argument list for base TrainingArguments.
            alpha (float): Weight for balancing student loss vs distillation loss. Default 0.5.
            temperature (float): Temperature parameter for softening probability distributions. Default 2.0.
            layers (tuple[int, ...]): Tuple specifying which layers to use for distillation.
            **kwargs: Additional keyword arguments for base TrainingArguments.
        """
        super().__init__(*args, **kwargs)
        self.logit_loss_weight = logit_loss_weight
        self.temperature = temperature
        self.layers = layers
        self.layer_loss_weight = layer_loss_weight
        self.hard_target_loss_weight = hard_target_loss_weight


class LayerDistillationTrainer(Trainer):
    def __init__(self, *args, teacher_models=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teachers: list[torch.nn.Module] = teacher_models
        for teacher in self.teachers:
            # place each teacher on same device as student
            self._move_model_to_device(teacher, self.model.device)
            teacher.eval()

    def compute_loss(self, model: torch.nn.Module, inputs, return_outputs=False):
        # compute student output
        outputs_student = model(
            **inputs,
            output_hidden_states=True,  # Request hidden states from all layers
        )
        loss_student = outputs_student.loss

        # compute teacher output
        teacher_outputs: list[CausalLMOutputWithCrossAttentions] = []
        with torch.no_grad():
            for teacher in self.teachers:
                outputs_teacher = teacher(
                    **inputs,
                    output_hidden_states=True,  # Request hidden states from all layers
                    # output_attentions=True,  # Request self-attention weights (and cross-attentions if applicable)
                )
                teacher_outputs.append(outputs_teacher)

        # Distillate layers
        # TODO: Can we use KLDivLoss if it is not logits?
        # How to add temperature?
        loss_layers = 0.0
        for layer in self.args.layers:
            layer_hidden_states = []
            for output in teacher_outputs:
                hidden_state = output.hidden_states[layer]
                layer_hidden_states.append(hidden_state)

            avg_teacher_hidden_states = torch.stack(layer_hidden_states).mean(dim=0)
            student_hidden_state = outputs_student.hidden_states[layer]

            # Ensure shapes match
            assert (
                student_hidden_state.size() == avg_teacher_hidden_states.size()
            ), f"Shape mismatch: student {student_hidden_state.size()} vs teacher {avg_teacher_hidden_states.size()}"

            # Compute MSE loss for hidden states (alternative to KL for representations)
            mse_loss = F.mse_loss(student_hidden_state, avg_teacher_hidden_states)
            loss_layers += mse_loss

        # Distillate logits
        all_teacher_logits = []
        for output in teacher_outputs:
            all_teacher_logits.append(outputs_teacher.logits)
        avg_teacher_logits = torch.stack(all_teacher_logits).mean(dim=0)

        assert outputs_student.logits.size() == avg_teacher_logits.size()
        # Soften probabilities and compute distillation loss
        loss_logits = nn.KLDivLoss(reduction="batchmean")(
            F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
            F.softmax(avg_teacher_logits / self.args.temperature, dim=-1),
        ) * (self.args.temperature**2)

        # Return weighted student loss
        loss = (
            self.args.hard_target_loss_weight * loss_student
            + self.args.logit_loss_weight * loss_logits
            + self.args.layer_loss_weight * loss_layers
        )
        return (loss, outputs_student) if return_outputs else loss


def create_student(tokenizer, seq_length):
    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256, # Match teacher
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

    EVAL_SAMPLES = 8192
    #############

    TEACHER_DIR1 = consts.TEACHER_DIR / "Llama-16M"
    TEACHER_DIR2 = consts.TEACHER_DIR / "GPT2-97M"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    STUDENT_NAME = "Baby-Llama-58M"
    STUDENT_OUTPUT = consts.STUDENT_DIR / f"{STUDENT_NAME}_{timestamp}"

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

    teacher1 = LlamaForCausalLM.from_pretrained(TEACHER_DIR1)
    # teacher2 = GPT2LMHeadModel.from_pretrained(TEACHER_DIR2)
    # teachers = [teacher1, teacher2]
    teachers = [teacher1]

    student = create_student(tokenizer, SEQ_LENGTH)

    print(f"model num parameters: student = {student.num_parameters()}")
    print(f"model num parameters: teacher1 = {teacher1.num_parameters()}")
    # print(f"model num parameters: teacher2 = {teacher2.num_parameters()}")

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
        save_total_limit=1,  # Set to zero to avoid saving
        report_to=[],  # "wandb",
        warmup_steps=200,
        lr_scheduler_type="cosine",
        learning_rate=LR,
        logging_steps=20,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        weight_decay=0.1,
        logit_loss_weight=ALPHA,
        temperature=TEMPERATURE,
        layers=(-1,),
    )

    trainer = LayerDistillationTrainer(
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
