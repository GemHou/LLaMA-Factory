# Install

```bash
pip install transformers==4.36.2
pip install peft>=0.7.0
pip install trl>=0.7.6
```

# Local 5900X

## Infer(4.2GB)

```bash
CUDA_VISIBLE_DEVICES=0 python src/cli_demo.py \
    --model_name_or_path /home/houjinghp/data/llm/phi-1_5/ \
    --template default \
    --finetuning_type lora
```

## Train SFT(8.5 GB)

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path /home/houjinghp/data/llm/phi-1_5/ \
    --do_train \
    --dataset alpaca_gpt4_en \
    --template default \
    --finetuning_type lora \
    --lora_target Wqkv \
    --lora_rank 8 \
    --output_dir ./data/sft_checkpoint/drive_1109 \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --overwrite_output_dir \
    --val_size 0.1 \
    --do_eval True \
    --evaluation_strategy epoch
```

## Train DPO(XX GB)

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage dpo \
    --do_train \
    --model_name_or_path path_to_llama_model \
    --adapter_name_or_path path_to_sft_checkpoint \
    --create_new_adapter \
    --dataset comparison_gpt4_en \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir path_to_dpo_checkpoint \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 1e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --fp16
```

## Infer(4.2GB)

```bash
CUDA_VISIBLE_DEVICES=0 python src/cli_demo.py \
    --model_name_or_path /home/houjinghp/data/llm/phi-1_5/ \
    --template default \
    --finetuning_type lora \
    --checkpoint_dir ./data/sft_checkpoint/drive_1109
```

# Local 9700K

## Infer(4.2GB)

```bash
CUDA_VISIBLE_DEVICES=0 python src/cli_demo.py \
    --model_name_or_path /home/gemhou/Study/data/phi-1_5_copy/ \
    --template default \
    --finetuning_type lora
```

## Infer(4.2GB)

```bash
CUDA_VISIBLE_DEVICES=0 python src/cli_demo.py \
    --model_name_or_path /home/gemhou/Study/data/phi-1_5_copy/ \
    --template default \
    --finetuning_type lora \
    --checkpoint_dir ./data/sft_checkpoint/drive_1109
```

# Server

## Infer(4.2GB)

```bash
CUDA_VISIBLE_DEVICES=0 python src/cli_demo.py \
    --model_name_or_path /xxx \
    --template default \
    --finetuning_type lora
```

## Train(more than 5 GB)

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path /xxx \
    --do_train \
    --dataset alpaca_gpt4_zh \
    --template default \
    --finetuning_type lora \
    --lora_target Wqkv \
    --lora_rank 2 \
    --output_dir path_to_sft_checkpoint \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16
```