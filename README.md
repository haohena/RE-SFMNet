# ENHANCING A SPATIAL-FREQUENCY MUTUAL NETWORK BY RESIDUAL LEARNING FOR FACE SUPER-RESOLUTION(RE-SFMNet)

Our experimental code will be published later！

# Visual quality comparison
## Visual quality comparison on CelebA dataset by the scale of ×8
![celeba compare](https://github.com/haohena/RE-SFMNet/assets/64673962/e35e3a60-8e24-4217-9483-8716831526a7)
## Visual quality comparison on Helen dataset by the scale of ×8
![helen compare](https://github.com/haohena/RE-SFMNet/assets/64673962/7415aa51-4e74-4691-8878-00ea3b2b9d8c)
# Quantitative comparisons for ×8 SR on the CelebA and Helen test sets.
![9d77aabb1e20b271c908b1a84b9cfa00](https://github.com/haohena/RE-SFMNet/assets/64673962/612d82ce-1ef0-445e-9bc7-df8aee526f96)
# Requirement
pytorch 1.12.1 Cuda 11.4
# Train
# Test
# Pretrain Model

| Method | CelebA PSNR | CelebA SSIM | Helen PSNR | Helen SSIM |
|--------------|-------------|-------------|------------|------------|
| Bicubic | 23.58 | 0.6285 | 23.88 | 0.6628 |
| EDSR | 26.84 | 0.7787 | 26.60 | 0.7851 |
| FSRNet | 26.66 | 0.7714 | 26.43 | 0.7799 |
| DIC | 27.37 | 0.8022 | 26.94 | 0.8026 |
| SPARNet | 27.42 | 0.8036 | 26.95 | 0.8029 |
| SISN | 27.31 | 0.7978 | 27.08 | 0.8083 |
| SFMNet | 27.56 | 0.8047 | 27.22 | 0.8141 |
| RE-SFMNet（ours） | 27.70 | 0.8126 | 27.266 | 0.8163 |
