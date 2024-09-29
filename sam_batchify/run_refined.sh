CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore batchify_processing.py\
    --sam_checkpoint /home/xiongbutian/workspace/Foundation_Models/SAM/sam_vit_h_4b8939.pth \
    --clip_checkpoint /home/xiongbutian/workspace/Foundation_Models/CLIP/open_clip_pytorch_model.bin \
    --clip_version ViT-B-16 \
    --image_dir /data/simulation/RESULT/RGB \
    --output_dir /home/xiongbutian/workspace/batchfy_sam/test \
    --debugging False \
    --gpu_number 4