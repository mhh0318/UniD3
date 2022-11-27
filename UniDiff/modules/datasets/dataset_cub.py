
from torch.utils.data import Dataset
import numpy as np
import io
from PIL import Image
import os
import json
import random
from tqdm import tqdm
import pickle
import sys, os
sys.path.append("/home/hu/UniDm/UniDiff/")
from modules.preprocessing.dalle_transform import DalleTransformerPreprocessor
def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

class CubDataset(Dataset):
    def __init__(self, data_root, split = 'train', size=256, tokenizer=None, downstream='diffusion+vqvae'):
        self.transform = DalleTransformerPreprocessor(size=size, phase=split)
        self.image_folder = os.path.join(data_root, 'images')
        self.root = os.path.join(data_root, split)
        pickle_path = os.path.join(self.root, "filenames.pickle")
        self.name_list = pickle.load(open(pickle_path, 'rb'), encoding="bytes")

        self.num = len(self.name_list)
        self.tokenizer = tokenizer

        # load all caption file to dict in memory
        self.caption_dict = {}
        
        for index in tqdm(range(self.num)):
            name = self.name_list[index]
            this_text_path = os.path.join(data_root, 'text',  name+'.txt')
            with open(this_text_path, 'r') as f:
                caption = f.readlines()
            self.caption_dict[name] = caption
        self.downstream = downstream
        print("load caption file done")


    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        name = self.name_list[index]
        image_path = os.path.join(self.image_folder, name+'.jpg')
        image = load_img(image_path)
        image = np.array(image).astype(np.uint8)
        image = self.transform(image = image)['image']

        if self.downstream == 'pure_image':
            data = image.transpose(2,0,1)/255.0
            return data

        caption_list = self.caption_dict[name]
        caption = random.choice(caption_list).replace('\n', '').lower()
        if self.downstream == 'pure_text':
            data = caption
            return data
        try:
            tokenized_text = self.tokenizer.tokenize(
                caption,
                128,
                ).squeeze(0)
        except:
            tokenized_text = np.array(self.tokenizer.encode(caption).ids)

        if self.downstream == 'vqvae':
            data =  {'image':image,'path':image_path}

        elif self.downstream == 'diffusion+vqvae':

            data = {'image':image,'text':tokenized_text}
        

        return data
    
if __name__ == '__main__':
    from modules.preprocessing.tokenizer import SimpleTokenizer
    tokenizer = SimpleTokenizer(bpe_path='./misc/bpe_simple_vocab_16e6.txt.gz',token_length = 7936)
    test = CubDataset(data_root='/home/hu/database/CUB_200_2011/', split='test', tokenizer=tokenizer, downstream='diffusion+vqvae')
    from pytorch_fid.fid_score import calculate_fid_given_paths
    fid_value = calculate_fid_given_paths([test,'/home/hu/UniDm/CUB_FID'],batch_size=50,device=0,dims=2048)
    print('ok')