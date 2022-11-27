from cmath import e
import os
import math
import time
import os, sys
from pathlib import Path
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
import random

import pytorch_fid

# Data
add_parent_path(level=1)
from modules.datasets.data import get_data, get_data_id, add_data_args, get_pl_datamodule
from model.wdecay import get_optim, get_optim_id, add_optim_args
# Model
from models import get_model, get_model_id, add_model_args

from torchvision.utils import save_image, make_grid

from modules.vq.taming_gumbel_vqvae import TamingGumbelVQVAE, TamingVQVAE
from modules.ldm.taming_ldm import LDM
ites = 'text'

def read_single_image(path):
    img = Image.open(path)
    img = img.resize((256, 256))
    img = torch.tensor(np.array(img))
    return img


def drop_tokens(embeddings, word_dropout):
    
    mask = torch.ones_like(embeddings)
    mask = mask.bernoulli_(1-word_dropout)
    embeddings = embeddings * mask
    embeddings[embeddings==0] = 999999
    return embeddings, mask

def drop_text_tokens(text, drop_text):
    # 0 is mask and 1 is unmask or cond
    tttks = get_tokens([text])[0]
    mask = torch.ones_like(tttks)
    to_be_masked = drop_text
    tttks_masked = get_tokens([to_be_masked])[0]
    len_tttks_masked = (tttks_masked!=0).sum()

    for i in range(len(tttks[0])-len_tttks_masked):
        if (tttks[0,i:i+len_tttks_masked] == tttks_masked[0,:len_tttks_masked]).all():
            mask[0,i:i+len_tttks_masked] = 0
            tttks[0,i:i+len_tttks_masked] = 999999
    return tttks, mask

def drop_center_tokens(embeddings):
    # 0 is mask and 1 is unmask or cond
    embeddings = embeddings.reshape(-1,32,32)
    mask = torch.ones_like(embeddings) 
    mask[:, 4:18, 8:24] = 0
    embeddings = embeddings * mask
    embeddings[embeddings==0] = 999999
    embeddings = embeddings.reshape(-1,1024)
    mask = mask.reshape(-1,1024)
    return embeddings, mask

def get_text(tokens):
    texts = []
    for i in tokens:
        try:
            text = train_loader.dataset.tokenizer.decode(i-2887) #974
        except:
            text = data_m.tokenizer.decode(i-974)
        texts.append(text)
    return texts

def get_tokens(text):
    texts = []
    for i in text:
        try:
            text = train_loader.dataset.tokenizer.tokenize(i) #974
        except:
            text = data_m.tokenizer.tokenize(i)
        texts.append(text)
    return texts


def get_image(tokens):
    # vq_model = TamingGumbelVQVAE().to(device)
    vq_model = TamingVQVAE().to(device)
    # vq_model = LDM(
    #     token_shape=[32,32],
    #     config_path='/home/hu/UniDm/misc/taming_dvae/f8_256.yaml',
    #     ckpt_path='/home/hu/UniDm/f8n256model.ckpt',
    #     num_tokens=256,
    # )
    imgs = vq_model.decode(tokens.to(device))
    return imgs

def check_vq(batch):
    # vq_model = TamingGumbelVQVAE()
    vq_model = TamingVQVAE()
    # vq_model = LDM(
    #     token_shape=[32,32],
    #     config_path='/home/hu/UniDm/misc/taming_dvae/f8_256.yaml',
    #     ckpt_path='/home/hu/UniDm/f8n256model.ckpt',
    #     num_tokens=256,
    # )
    tokens = vq_model.get_tokens(batch.to(device))
    imgs = vq_model.decode(tokens.to(device))
    save_image(make_grid(imgs), 'test.png')
    return imgs, tokens


parser = argparse.ArgumentParser()
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--gpus', type=int, default=1)
# parser.add_argument('--model', type=str, default='/home/hu/UniDm/i2t_epoch=10-step=406835.ckpt')
# parser.add_argument('--model', type=str, default='/home/hu/UniDm/cub_t2i_epoch=999-step=554000.ckpt')
parser.add_argument('--model', type=str, default='cub_epoch999_new.ckpt')
# parser.add_argument('--model', type=str, default='/home/hu/UniDm/UniDiff/epoch=90-step=3365635.ckpt')
add_exp_args(parser)
add_data_args(parser)
add_model_args(parser)
add_optim_args(parser)
args = parser.parse_args()
args.batch_size = 4
args.attention_type = 'selfcross'

# device = f'cuda:{args.gpus}'
device = 'cuda'

##################
## Specify data ##
##################

train_loader, eval_loader = get_data(args)
# data_id = get_data_id(args)
# data_m = get_pl_datamodule(args)

it_eval = iter(eval_loader)
next(it_eval)
# next(it_eval)
# next(it_eval)
# # next(it_eval)
# # next(it_eval)
##################
# Specify VQ    ##
##################

# _, igtks = check_vq(next(it_eval)['image'])
txt = next(it_eval)['text'].to(device)

# igtks = igtks.to(device)
# tttks = txt+2887
# with open(os.path.join('/home/hu/UniDm/', 'test.txt'.format()), 'w') as f:
#     f.write('\n\n\n'.join(get_text(tttks)))
# save_image(make_grid(_, nrow=4, padding=0),'test_vq.png')

# caption = ['a table filled with different types of vegetables', 'a kitchen area with toilet and various cleaning appliances', 'the plane is flying high in the sky', 'a lot of zebras standing in the field on a hot summer day']
# txt = data_m.tokenizer.tokenize(
#     caption,
#     128,
#     ).squeeze(0).cuda()

# Image manipulation
# ex_image = read_single_image('/home/hu/database/CUB_200_2011/images/033.Yellow_billed_Cuckoo/Yellow_Billed_Cuckoo_0024_26832.jpg').unsqueeze(0)
# ex_image = read_single_image('/home/hu/database/CUB_200_2011/images/011.Rusty_Blackbird/Rusty_Blackbird_0094_6582.jpg').unsqueeze(0)
# ex_image = read_single_image('/home/hu/database/CUB_200_2011/images/014.Indigo_Bunting/Indigo_Bunting_0061_13259.jpg').unsqueeze(0)
# img,igtks = check_vq(ex_image)
# igtks_mask, mask_i = drop_center_tokens(igtks)
# masksaver = igtks_mask.clone()
# masksaver[masksaver==999999]=0
# msk_img = get_image(masksaver)
# save_image(msk_img,'test_vq_mask.png')
# # ex_text_list = open('/home/hu/database/CUB_200_2011/text/051.Horned_Grebe/Horned_Grebe_0066_34738.txt').readlines()
# # ex_text = random.choice(ex_text_list).strip('\n')
# ex_text = 'a brilliant blue colored bird with black coverts and beak. .' 
# to_be_masked = ''
# tttks_mask, mask_t = drop_text_tokens(ex_text, to_be_masked)
# tttks_mask = get_tokens(ex_text)


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
## Sample ##############


path_samples = "samples"
Path(path_samples).mkdir(parents=True, exist_ok=True)


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
model = model.to(device)
model = model.eval()
# lengths = torch.ones(2, device=device, dtype=torch.long) * (1024+256)
# mask = length_mask(lengths, maxlen=da]ta_shape[0])



# igtks_mask, mask_i = drop_tokens(igtks, 0.9)
# tttks_mask, mask_t = drop_tokens(tttks, 0.5)

# sample_text = get_text(txt+2887)
samples_chain = model.sample_chain_mask(args.batch_size) 
# samples_chain = model.sample_mask_fused(args.batch_size, igtks_mask.to(device), tttks_mask.to(device), mask_i, mask_t) 
# samples_chain = model.ddim_sample_loop(args.batch_size) 
# samples_chain = model.ddim_reserve_sample_loop(igtks, tttks, 10) 
# samples_chain = model.ddim_cycle(igtks, tttks, 10) 
# samples_chain = model.sample_chain_mask_clf(args.batch_size,txt,delta=10) 
# samples_chain = model.sample_chain_mask_cond(args.batch_size,txt) 
# samples_chain = model.sample_chain_mask_fast(args.batch_size) 

# samples_chain = model.sample_chain_image_cond(args.batch_size,igtks.to(device)) 
 
sc = samples_chain


for i in range(0, 1):

    # img_token = sc[:,:32*32]
    # text_token = sc[:,32*32:]
    img_token = sc[0][:,:32*32]
    text_token = sc[0][:,32*32:]
    # img_token = sc[i][:,:16*16]
    # text_token = sc[i][:,16*16:]


    sample_text = get_text(text_token)
    # sample_text = get_text(txt)


    with open(os.path.join('samples', 'sample_ep{}_s{}_{}.txt'.format(checkpoint['epoch'], i, args.gpus)), 'w') as f:
        f.write('\n\n\n'.join(sample_text))

    imgs = get_image(img_token)

    save_image(make_grid(imgs, nrow=4, padding=0), os.path.join('samples', 'sample_ep{}_s{}_{}.png'.format(checkpoint['epoch'], i, args.gpus)))
    for kk in range(len(imgs)):
        save_image(imgs[kk],f'samples/samples_{kk}.png')


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


