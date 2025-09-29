python v1.py \
    --output_dir "./output/" \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --num_train_epochs 10 \
    --warmup_steps 512 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-4 \
    --evaluation_strategy  steps \
    --save_strategy steps \
    --eval_steps 20 \
    --save_steps 20 \
    --save_total_limit 5 \
    --bf16 
    # --report_to none

## 请多调调这些参数