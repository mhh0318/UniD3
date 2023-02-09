# *UniD3*: Unified Discrete Diffusion for Simultaneous Vision-Language Generation


<a href="https://arxiv.org/abs/2211.14842"><img src="https://img.shields.io/badge/arXiv-2203.10821-b31b1b.svg" height=22.5></a> 
<a href="https://mhh0318.github.io/unid3/"><img src="https://img.shields.io/badge/Web-Project Page-brightgreen.svg" height=22.5></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" height=22.5></a> 




![unid3](tasks.png)


## Abstract 

The recently developed discrete diffusion models perform extraordinarily well in the text-to-image task, showing significant promise for handling the multi-modality signals. In this work, we harness these traits and present a unified multimodal generation model that can conduct both the "modality translation" and "multi-modality generation" tasks using a single model, performing text-based, image-based, and even vision-language simultaneous generation. Specifically, we unify the discrete diffusion process for multimodal signals by proposing a unified transition matrix. Moreover, we design a mutual attention module with fused embedding layer and a unified objective function to emphasise the inter-modal linkages, which are vital for multi-modality generation. Extensive experiments indicate that our proposed method can perform comparably to the state-of-the-art solutions in various generation tasks.

## Setup

### Installation Requirmenets

The code is compatible with python 3.8 and pytorch 1.9. 

You can create an anaconda environment called `UniD3` with the required dependencies by running:

```
git clone https://github.com/mhh0318/UniD3.git
cd UniD3
conda create -n unid3 python=3.8
pip install -r requirements.txt
```

### Download Pretrained Weights

Download the pretrained models from [here](https://drive.google.com/file/d/1dVRp3lPrWS0EWFViYG3Bj_tHmD3riVZP/view?usp=sharing), and save them to ``pretrained_models/``. 

Download the released VQ-GAN model [GumbelVQGAN on OpenImages](https://facevcstandard.blob.core.windows.net/t-shuygu/release_model/VQ-Diffusion/pretrained_model/taming_dvae/taming_f8_8192_openimages_last.pth?sv=2019-12-12&st=2022-03-09T01%3A59%3A19Z&se=2028-03-10T01%3A59%3A00Z&sr=b&sp=r&sig=T9d9A3bZVuSgGXYCYesEq9egLvMS0Gr7A4h6MCkiDcw%3D) and put them under ``./misc/taming_dvae/``

### Quick Inference

For the simultaneous vision-language generation, please ru:

```bash
python ./UniDiff/dist_eval_sample.py --model CKPT_PATH  --condition unconditional --log pair_samples
```

If the environment is setup correctly, this command should function properly and generate some results in the folder `/pair_samples`.


## Comments 

- Our codebase for the diffusion models builds heavily on [https://github.com/lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch), [VQ-Diffusion](https://github.com/microsoft/VQ-Diffusion) and [Multi-nomial Diffusion](https://github.com/ehoogeboom/multinomial_diffusion)
Thanks for open-sourcing!

- The implementation of the transformer encoder is from [x-transformers](https://github.com/lucidrains/x-transformers) by [lucidrains](https://github.com/lucidrains?tab=repositories). 


## BibTeX

```
@article{hu2022unified,
  title = {Unified Discrete Diffusion for Simultaneous Vision-Language Generation},
  author = {Hu, Minghui and Zheng, Chuanxia and Zheng, Heliang and Cham, Tat-Jen and Wang, Chaoyue and Yang, Zuopeng and Tao, Dacheng and Suganthan, Ponnuthurai N},
  journal = {arXiv},
  year = {2022},
}
```
