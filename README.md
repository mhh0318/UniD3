# Link for UniD3 ckpt based on CUB-200

[Google Drive Link for CUB model](
https://drive.google.com/file/d/1dVRp3lPrWS0EWFViYG3Bj_tHmD3riVZP/view?usp=sharing)

# Usage

1. Install the required packages according to ``requirements.txt``
2. Download the provided checkpoint and put it anywhere.
3. Download the released VQ-GAN model [GumbelVQGAN on OpenImages](https://facevcstandard.blob.core.windows.net/t-shuygu/release_model/VQ-Diffusion/pretrained_model/taming_dvae/taming_f8_8192_openimages_last.pth?sv=2019-12-12&st=2022-03-09T01%3A59%3A19Z&se=2028-03-10T01%3A59%3A00Z&sr=b&sp=r&sig=T9d9A3bZVuSgGXYCYesEq9egLvMS0Gr7A4h6MCkiDcw%3D) and put them under ``./misc/taming_dvae/``
4. Run commands ``python ./UniDiff/dist_eval_sample.py --model CKPT_PATH  --condition unconditional --log pair_samples``


Suggested 8G+ VRAM.

Have Fun.