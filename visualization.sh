CUDA_VISIBLE_DEVICES=4 python gradio_test.py\
    --clip_checkpoint /home/xiongbutian/workspace/Foundation_Models/CLIP/open_clip_pytorch_model.bin \
    --clip_version ViT-B-16 \
    --image_location /data/simulation/RESULT/RGB \
    --npz_location /home/xiongbutian/workspace/batchfy_sam/test