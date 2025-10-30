#!/bin/bash

composer train.py \
    --config-path ./configs \
    --config-name res_512_finetune_lcm.yaml \
    exp_name=MicroDiTXL_mask_0_res_512_finetune_lcm_negone_nocfg_heun_40steps_longer_cfg \
    model.dit_ckpt_path=ckpts/dit_4_channel_37M_real_and_synthetic_data.pt
