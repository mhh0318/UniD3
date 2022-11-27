import os
import math
from tabnanny import check
import time
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import imageio
import numpy as np
import torch
import pickle
import argparse
from model.utils import add_parent_path
from solver import Experiment, add_exp_args
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

# Data
add_parent_path(level=1)
from modules.datasets.data import get_data, get_data_id, add_data_args
from model.wdecay import get_optim, get_optim_id, add_optim_args
# Model
from models import get_model, get_model_id, add_model_args

from torchvision.utils import save_image, make_grid

from modules.vq.taming_gumbel_vqvae import TamingGumbelVQVAE, TamingVQVAE


def get_text(tokens):
    texts = []
    for i in tokens:
        text = train_loader.dataset.tokenizer.decode(i-974)
        texts.append(text)
    return texts

def get_image(tokens):
    # vq_model = TamingGumbelVQVAE()
    vq_model = TamingVQVAE()
    imgs = vq_model.decode(tokens)
    return imgs


parser = argparse.ArgumentParser()
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--model', type=str, default='/home/hu/UniDm/UniDm-UniDiff/2022-06-13T17-40-54_UNIDIFFUSION_cub200_expdecay/checkpoints/epoch=299-step=55499.ckpt')
add_exp_args(parser)
add_data_args(parser)
add_model_args(parser)
add_optim_args(parser)
args = parser.parse_args()
args.batch_size = 25
args.attention_type = 'selfcross'
args.seed = 24
##################
## Specify data ##
##################

train_loader, eval_loader = get_data(args)
data_id = get_data_id(args)

# txt = iter(eval_loader).__next__()['text'].cuda()


###################
## Specify model ##
###################

model = get_model(args)
# if args.parallel == 'dp':
#     model = DataParallelDistribution(model)
checkpoint = torch.load(args.model,map_location='cpu')
ckpt = checkpoint['state_dict']
ckpt = {k[6:]: checkpoint['state_dict'][k] for k in checkpoint['state_dict'].keys()}
model.load_state_dict(ckpt)
print('Loaded weights for model at {}/{} epochs'.format(checkpoint['epoch'], args.epochs))

############
## Sample ##
############


path_samples = os.path.join('/home/hu/UniDm/UniDiff/', 'samples/sample_ep{}_s{}.txt'.format(checkpoint['epoch'], 0))
if not os.path.exists(os.path.dirname(path_samples)):
    os.mkdir(os.path.dirname(path_samples))

# device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
model = model.cuda(2)
model.eval()
# lengths = torch.ones(2, device=device, dtype=torch.long) * (1024+256)
# mask = length_mask(lengths, maxlen=data_shape[0])

j = 0
for _ in range(2):
    for i in eval_loader:
        txt = i['text'].cuda(2)
        # img = i['image'].cuda()
        # samples_chain = model.sample_chain_mask(args.batch_size) 
        samples_chain = model.sample_chain_mask_cond(args.batch_size,txt) 
        img_token = samples_chain[0][:,:256]
        imgs = get_image(img_token)
        for k in range(len(imgs)):
            save_image(imgs[k], f'/home/hu/UniDm/CUB_FID/sample_{j}_{k}_parallel.png')
        j+=1
# samples_chain = model.sample_chain_mask_cond(args.batch_size,txt) 

# sc = samples_chain

# img_token = sc[0][:,:256]
# text_token = sc[0][:,256:]


# sample_text = get_text(text_token)

# with open(path_samples, 'w') as f:
#     f.write('\n\n\n'.join(sample_text))

# imgs = get_image(img_token)

# save_image(make_grid(imgs, nrow=4, padding=0), os.path.join('/home/hu/UniDm/UniDiff/', 'samples/sample_ep{}_s{}.png'.format(checkpoint['epoch'], 0)))


# samples_chain = model.sample_chain_mask(eval_args.samples)  #sample mask
# samples = samples_chain[0]
# # samples = model.sample(eval_args.samples)
# # samples_text = train_loader.dataset.vocab.decode(samples.cpu(), lengths.cpu())
# samples_text = get_text(samples.cpu())
# print([len(s) for s in samples_text])
# with open(path_samples, 'w') as f:
#     f.write('\n\n\n'.join(samples_text))

# T, B, L = samples_chain.size()
# # samples_chain_text = train_loader.dataset.vocab.decode(
# #     samples_chain.view(T * B, L).cpu(), lengths.repeat(T).cpu())
# samples_chain_text = get_text(samples_chain.view(T*B,L).cpu())
# # print('before reshape', samples_chain_text)
# samples_chain_text = np.array(samples_chain_text)
# samples_chain_text = samples_chain_text.reshape((T, B))


# def chain_linspace(samples_chain_text, num_steps=150, repeat_last=10):
#     out = []
#     for i in np.linspace(0, len(samples_chain_text)-1, num_steps):
#         idx = int(i)
#         if idx >= len(samples_chain_text):
#             print('index too big')
#             idx = idx - 1
#         out.append(samples_chain_text[idx])

#     for i in range(repeat_last):
#         out.append(samples_chain_text[-1])
#     return out


# def format_text(batch_text):
#     # print('batch_text', batch_text)
#     out = []
#     for text in batch_text:
#         linesize = 90
#         reformat = text[0:linesize]
#         for i in range(linesize, len(text), linesize):
#             reformat = reformat + '\n' + text[i:i+linesize]

#         out.append(reformat)

#         # print('reformat', reformat)

#     return '\n\n'.join(out)


# def draw_text_to_image(text, invert_color=False):
#     font = ImageFont.truetype("/home/hu/UniDm/misc/CascadiaCode-Regular.otf", 24)

#     black = (0, 0, 0)
#     white = (255, 255, 255)
#     if invert_color:
#         background_color = white
#         textcolor = black
#     else:
#         background_color = black
#         textcolor = white

#     img = Image.new('RGB', (1290, 200), color=background_color)

#     draw = ImageDraw.Draw(img)
#     draw.multiline_text(
#         (10, 10), text, textcolor, font=font)

#     img_np = np.array(img)
#     return img_np


# images = []
# text_chain = []

# for samples_i in chain_linspace(list(reversed(samples_chain_text))):
#     # print('in1', samples_i)
#     samples_i = format_text(samples_i)
#     text_chain.append(samples_i)
#     # print('in2', samples_i)
#     images.append(draw_text_to_image(samples_i))
# with open(path_samples[:-4] + '_chain.txt', 'w') as f:
#     f.write('\n\n\n'.join(text_chain))
# imageio.mimsave(path_samples[:-4] + '_chain.gif', images)


