#!/bin/bash
torchrun --nproc_per_node=2 distill_ensemble_pretraining_configurable.py --config config/distillation/BabyLlama1-58M-strict.yaml
