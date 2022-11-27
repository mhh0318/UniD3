"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
from pathlib import Path
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch as th
# from model.utils import dist_utils as dist_util
from model.utils import add_parent_path
import random
# from modules.datasets.data import add_data_args,get_data
from modules.preprocessing.tokenizer import SimpleTokenizer
from models import get_model, add_model_args
from modules.datasets.dataset_cub import CubDataset
from modules.datasets.dataset_coco import CocoDataset
from torchvision.utils import save_image, make_grid
from torchvision.transforms import ToTensor, Resize, Compose
from modules.preprocessing.dalle_transform import DalleTransformerPreprocessor
import time
from PIL import Image

def load_img(filepath, root):
    true_filepath = os.path.join(os.path.join(root, 'val2014'), filepath)
    img = Image.open(true_filepath).convert('RGB')
    return img

def get_text(tokens,tkizer):
    texts = []
    for i in tokens:
        text = tkizer.decode(i)
        texts.append(text)
    return texts


def get_img_tks(batch, model):
    vq_model = model.image_tokenizer
    tokens = vq_model.get_tokens(batch)
    # imgs = vq_model.decode(tokens)
    return tokens

def get_token(list,tkizer):
    tokens = tkizer.tokenize(
        list,
        128,
        ).squeeze(0)
    return tokens

def sample_step(model,args,condition=None):
    if args.condition == 'unconditional':
        sample = model.sample_mask(args.batch_size) 
        # sample = model.ddim_sample_loop(args.batch_size) 
    elif args.condition == 'text':
        if args.fast != 0:
            sample = model.sample_chain_mask_clf(args.batch_size, condition, args.fast) 
        else:
            sample = model.sample_chain_mask_cond(args.batch_size,condition)[0]
    elif args.condition == 'image':
        sample = model.sample_chain_image_cond(args.batch_size,condition)[0]
        samples_text_tokens = sample[:,32*32:]
        return None, samples_text_tokens
    samples_image_tokens = sample[:,:32*32]
    samples_image = model.image_tokenizer.decode(samples_image_tokens.to("cuda:{}".format(args.gpus)))
    samples_text_tokens = sample[:,32*32:]
    return samples_image, samples_text_tokens


def main():
    # add_parent_path(level=1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    # parser.add_argument('--iters', type=int, default=1000)
    # parser.add_argument('--num_samples', type=int, default=50000)
    parser.add_argument('--cuiter', type=int, default=1)
    parser.add_argument('--model', type=str)
    parser.add_argument('--fast', default=0, type=int)
    parser.add_argument('--condition', type=str, default='text')
    parser.add_argument('--save_iter', type=int, default=0)
    parser.add_argument('--log', type=str, default='pairs')
    parser.add_argument('--txt_emb', type=int, default=7936) #8192-256
    parser.add_argument('--length_size', type=int, default=128)
    parser.add_argument('--tokenizer', type=str, default='SimpleTokenizer')
    # add_data_args(parser)
    add_model_args(parser)
    args = parser.parse_args()
    # args.root = "/home/hu/database/CUB_200_2011/"
    # args.root = "/home/hu/database/coco/"
    # logger.log("creating model and diffusion...")
    print("creating model and diffusion...")
    a = time.time()
    model = get_model(args)


    checkpoint = th.load(args.model,map_location='cpu')
    ckpt = checkpoint['state_dict']
    ckpt = {k[6:]: checkpoint['state_dict'][k] for k in checkpoint['state_dict'].keys()}
    model.load_state_dict(ckpt)
    print('Loaded weights for model at {} epochs with {:.2f}s'.format(checkpoint['epoch'], time.time()-a))
    model.to("cuda:{}".format(args.gpus))
    model.eval()

    if args.condition != 'unconditional':
        tokenizer = SimpleTokenizer(bpe_path='/home/hu/UniDm/misc/bpe_simple_vocab_16e6.txt.gz',token_length = 7936)
        dataset = CubDataset(data_root=args.root, split='test', tokenizer=tokenizer, downstream=args.downstream)
        # dataset = CocoDataset(data_root=args.root, split='karpathy', tokenizer=tokenizer, downstream=args.downstream)
        if args.condition == 'text':
            captions = [*dataset.caption_dict.values()]
            current_text = captions[args.cuiter*args.batch_size: (args.cuiter+1)*args.batch_size]
            current_text = [random.choice(i).replace('\n', '').lower() for i in current_text]
            assert len(current_text) !=0, 'current iter is {}'.format(args.cuiter) 
            current_token = get_token(current_text, tokenizer)

            if len(current_text) < args.batch_size:
                args.batch_size = len(current_token)
            condition = current_token.to("cuda:{}".format(args.gpus))
        if args.condition == 'image':
            images_ =  dataset.json_file['annotations']
            current_files = images_[args.cuiter*args.batch_size: (args.cuiter+1)*args.batch_size]
            filenames = [i['filename'] for i in current_files]
            trans = DalleTransformerPreprocessor(size=256, phase='test')
            imgs = np.stack([trans(np.array(load_img(i,args.root)).astype(np.uint8))['image'] for i in filenames],0)
            imgs = th.tensor(imgs).to("cuda:{}".format(args.gpus))
            condition = get_img_tks(imgs,model).to("cuda:{}".format(args.gpus))

    else:
        condition = None

    # logger.log("sampling...")
    print("sampling without fast strategy...")

    # for i in tqdm(range(args.curiter,args.iters)):
    path = args.log
    Path(path).mkdir(exist_ok=True)
    # a = time.time()
    samples_image, samples_text_tokens=sample_step(model, args, condition=condition)
    # print('sampled {} images with {}s'.format(args.batch_size, time.time()-a))
    with open(os.path.join(path, 'sample_text_cuda{}.txt'.format(args.gpus)), 'a') as f:
        generate_text = get_text(samples_text_tokens-2887, model.text_tokenizer)
        try:
            tt = [filenames[i]+'\n'+generate_text[i] for i in range(len(generate_text))]
        except:
            tt = generate_text
        f.write('\n\n'.join(tt))
        f.write('\n\n')
    if samples_image is not None:
        for id,j in enumerate(samples_image):
            save_image(j.cpu() , os.path.join(path, 'sample_image_cuda{}_batch{}_{}.png'.format(args.gpus,args.save_iter,id)))
    print(f"created {args.cuiter * args.batch_size} samples")

    # img_arr = np.concatenate(all_images, axis=0)
    # # arr = arr[: args.num_samples]

    # txt_arr = np.concatenate(all_labels, axis=0)

    # shape_str = "x".join([str(x) for x in img_arr.shape])
    # out_path = os.path.join( "/home/hu/UniDm/UniDiff", f"samples_{shape_str}_cuda{args.gpu}.npz")
    # # logger.log(f"saving to {out_path}")
    # print(f"saving to {out_path}")
    # np.savez(out_path, img_arr, txt_arr)

    print("sampling complete")




if __name__ == "__main__":
    main()