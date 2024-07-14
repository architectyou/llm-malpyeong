CUDA_VISIBLE_DEVICES=0 python -m train \
    --model_id MLP-KTLim/llama-3-Korean-Bllossom-8B \
    --batch_size 1 \
    --gradient_accumulation_steps 64 \
    --epoch 1 \
    --lr 2e-5 \
    --warmup_steps 20