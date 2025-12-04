torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir /grp01/ids_xh/zhekai/data/embodied/ \
  --dataset_name bridge_orig \
  --run_root_dir runs \
  --adapter_tmp_dir attempt_1_base \
  --lora_rank 32 \
  --batch_size 8 \
  --grad_accumulation_steps 2 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project moto_openvla \
  --wandb_entity azily \
  --max_steps 200000 \
  --save_steps 5000 \
  --shuffle_buffer_size 10000 \
  --use_motion_token True \
  --latent_motion_tokenizer_path /grp01/ids_xh/zhekai/ckpt/moto_oxe_tokenizer \
  --future_observation_k 3


