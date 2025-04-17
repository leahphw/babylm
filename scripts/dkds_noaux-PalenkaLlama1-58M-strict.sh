#!/bin/bash
config_path=config/distillation/dkds_noaux/PalenkaLlama1-58M-strict.yaml
torchrun --nproc_per_node=2 --master_port=29501 dkds-noaux.py --config $config_path

# python dkds-noaux.py --config $config_path
 