# MISS
In this work, we establish a task called Modality-Incomplete Scene Segmentation (MISS), which encompasses both system-level modality absence and sensor-level modality errors. 
<p align="center">
  <img src="figs/MISS.png" width="400">
</p>
We introduce a Missing-aware Modal Switch (MMS) strategy to proactively manage missing modalities during training, utilizing bit-level batch-wise sampling to enhance the models's performance in both complete and incomplete testing scenarios. Furthermore, we introduce the Fourier Prompt Tuning (FPT) method to incorporate representative spectral information into a limited number of learnable prompts that maintain robustness against all MISS scenarios. 
<div style="display:inline-block" align="center">
  <img src="figs/MMS.png" alt="image1" width="400">
  <img src="figs/FPT.png" alt="image2" width="400">
</div>
