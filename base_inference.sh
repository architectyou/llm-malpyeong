export CUDA_VISIBLE_DEVICES="MIG-78aecc0e-1524-5b56-a0c4-3b6ce40f2d44, MIG-1a613912-6833-5520-8236-ae3cc8e9a736"

python -m test \
    --output 10_epoch.json \
    --model_id /mnt/models/llama-3-Korean-Bllossom-8B \
    --device cuda:0