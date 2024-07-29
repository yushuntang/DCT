seed=2023
CUDA_VISIBLE_DEVICES=1 python main.py --dataset imagenet-c --data_corruption ./datasets/ImageNet-C \
--exp_type normal --method dct --model vitbase_timm --output ./output/debug  \
--seed $seed --test_batch_size 64 --dct_lr 0.01