# MISS

### Fourier Prompt Tuning for Modality-Incomplete Scene Segmentation [[PDF]](https://arxiv.org/pdf/2401.16923)

In this work, we establish a task called Modality-Incomplete Scene Segmentation (MISS), which encompasses both system-level modality absence and sensor-level modality errors. 
<p align="center">
  <img src="figs/MISS.png" width="400">
</p>
We introduce a Missing-aware Modal Switch (MMS) strategy to proactively manage missing modalities during training, utilizing bit-level batch-wise sampling to enhance the models's performance in both complete and incomplete testing scenarios. Furthermore, we introduce the Fourier Prompt Tuning (FPT) method to incorporate representative spectral information into a limited number of learnable prompts that maintain robustness against all MISS scenarios. 
<div style="display:inline-block" align="center">
  <img src="figs/MMS.png" alt="image1" width="300">
  <img src="figs/FPT.png" alt="image2" width="300">
</div>

## Preparation
Refer to [DeLiVER](https://github.com/jamycheung/DELIVER) for environment installation and downloading DeLiVER dataset.

Calculate the depth maps for the Cityscapes dataset according to the instructions in [RGBD_Semantic_Segmentation_PyTorch](https://github.com/charlesCXK/RGBD_Semantic_Segmentation_PyTorch/issues/1#issuecomment-684875832) and store them as `.npy` files.

`dataset` folder should be structured as:
```
DELIVER
├── img
├── depth
├── missing
└── semantic

cityscapes
├── leftImg8bit
├── depth
├── missing
└── gtFine
```
The `missing` folders contain black images with the same resolution and filenames as those in other modalities.
## Training
Please download the [MultiMAE](https://drive.google.com/file/d/1reL9dvGr_kGPk73HeFdzziS7lIbi2PDg/view?usp=sharing) pretrained weights to the folder `checkpoints/pretrained/`.

When training with MMS, change `MISS` in configuration files from False to True.
```
cd path/to/MISS
conda activate cmnext
export PYTHONPATH="path/to/MISS"
python -m torch.distributed.launch --nproc_per_node=4 --use_env tools/train_prompt.py --cfg configs/config_fpt_deliver.yaml
python -m torch.distributed.launch --nproc_per_node=4 --use_env tools/train_prompt.py --cfg configs/config_fpt_cityscapes.yaml
```

## Evaluation
Please download the following weights to the folders `checkpoints/fpt/` and `checkpoints/fpt_mms/`

DeLIVER:
|Model|mIoU(%)|weight|
|:----|:------|:-----|
|FPT|57.81|[Google Drive](https://drive.google.com/file/d/1zP8tYiHzzYcNT3BQ2d_fk2WXL13EI_SZ/view?usp=sharing)|
|FPT(MMS)|57.38|[Google Drive](https://drive.google.com/file/d/1OKhTrUp7siz1MNSckRSzVAfBdKlUp_-w/view?usp=sharing)|

Cityscapes:
|Model|mIoU(%)|weight|
|:----|:------|:-----|
|FPT|75.16|[Google Drive](https://drive.google.com/file/d/1Un5xpcvILNYF4vBMLptJuDOQnX3uueKa/view?usp=sharing)|
|FPT(MMS)|75.47|[Google Drive](https://drive.google.com/file/d/1BG3C6qA1hrutwCppQURfxXfst2qaMC5X/view?usp=sharing)|

```
cd path/to/DELIVER
conda activate cmnext
export PYTHONPATH="path/to/DELIVER"
CUDA_VISIBLE_DEVICES=0 python tools/val_mm.py --cfg configs/config_fpt_deliver.yaml
CUDA_VISIBLE_DEVICES=0 python tools/val_mm.py --cfg configs/config_fpt_cityscapes.yaml
```

## Citation
If you use our method in your project, please consider referencing
```
@article{liu2024fourier,
  title={Fourier Prompt Tuning for Modality-Incomplete Scene Segmentation},
  author={Liu, Ruiping and Zhang, Jiaming and Peng, Kunyu and Chen, Yufan and Cao, Ke and Zheng, Junwei and Sarfraz, M Saquib and Yang, Kailun and Stiefelhagen, Rainer},
  journal={arXiv preprint arXiv:2401.16923},
  year={2024}
}

```
