
# Recommended train steps
```
# (1) SHA pretrain

python train_baseline.py --dataset ShanghaiA --crop_size 256 --save_path ./work_dirs/SHA_pretrain --gpu_id 0 --workers 4 --lr 5e-5 --preload_data --seed 89 --loss_type MSE_SSIM --normalize_mse  --warmup_epochs 100 --epoch 3000 --lr_scheduler warmup_cosine --norm_eval --transforms_type strong --ssim_coefficient 16 --fp16 --model_type vl_timm-regnetx_064 --head_type flower

# (2) NWPU finetune-pretrain
python train_baseline.py --dataset NWPU --crop_size 512 --save_path ./work_dirs/NWPU_pretrain --gpu_id 0 --workers 4 --lr 5e-5 --preload_data --seed 89 --loss_type MSE_SSIM --normalize_mse  --warmup_epochs 100 --epoch 3000 --lr_scheduler warmup_cosine --norm_eval  --ssim_coefficient 16  --model_type vl_timm-regnetx_064 --fp16 --head_type flower --load_checkpoint ./work_dirs/SHA_pretrain/checkpoint.pth --transforms_type medium

# (3) Finetune on your own dataset
```


# Reference
If you find this project is useful for your research, please cite:
```
@article{liang2022focal,
  title={Focal inverse distance transform maps for crowd localization},
  author={Liang, Dingkang and Xu, Wei and Zhu, Yingying and Zhou, Yu},
  journal={IEEE Transactions on Multimedia},
  year={2022},
  publisher={IEEE}
}
```





