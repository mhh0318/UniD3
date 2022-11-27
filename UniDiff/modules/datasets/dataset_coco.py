from torch.utils.data import Dataset
import numpy as np
import io
from PIL import Image
import os
import json
import pickle
import sys, os
sys.path.append("/home/hu/UniDm/UniDiff/")
from modules.preprocessing.dalle_transform import DalleTransformerPreprocessor

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

def load_pkl(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


'''
COCO Captions contains over one and a half million captions 
describing over 330,000 images.
'''
class CocoDataset(Dataset):
    def __init__(self, data_root, split = 'train', size=256, tokenizer=None, downstream='diffusion+vqvae'):
        self.transform = DalleTransformerPreprocessor(size=size, phase=split)
        self.root = os.path.join(data_root, split)
        # input_file = os.path.join(data_root, input_file)
        caption_file = "captions_"+split+"2017.json"
        caption_file = os.path.join(data_root, "annotations", caption_file)

        self.json_file = json.load(open(caption_file, 'r'))
        print("length of the dataset is ")
        print(len(self.json_file['annotations']))

        self.num = len(self.json_file['annotations'])
        self.image_prename = "COCO_" + split + "2017"
        self.folder_path = os.path.join(data_root, split+'2017')

        if tokenizer is not None:
            self.tokenizer = tokenizer

        self.downstream = downstream
 
    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        
        this_item = self.json_file['annotations'][index]

        if self.downstream == 'pure_image':
            image_name = str(this_item['image_id']).zfill(12)
            # image_path = os.path.join(self.folder_path, self.image_prename+image_name+'.jpg')
            image_path = os.path.join(self.folder_path, image_name+'.jpg')
            image = load_img(image_path)
            image = np.array(image).astype(np.uint8)
            image = self.transform(image = image)['image']
            data = image.transpose(2,0,1)/255.0
            return data
        if self.downstream == 'pure_text':
            data = self.json_file['annotations'][index]
            return data
        

        if 'diffusion' in self.downstream:
            caption = this_item['caption'].lower()

            tokenized_text = self.tokenizer.tokenize(
                caption,
                128,
                ).squeeze(0)

        if 'vqvae' in self.downstream:
            image_name = str(this_item['image_id']).zfill(12)
            # image_path = os.path.join(self.folder_path, self.image_prename+image_name+'.jpg')
            image_path = os.path.join(self.folder_path, image_name+'.jpg')
            image = load_img(image_path)
            image = np.array(image).astype(np.uint8)
            image = self.transform(image = image)['image']

        if self.downstream == 'vqvae':
            data =  {'image':image,'path':image_path}

        elif self.downstream == 'diffusion+vqvae':

            data = {'image':image,'text':tokenized_text}
            
        elif self.downstream == 'diffusion':
            image_name = str(this_item['image_id']).zfill(12)
            # image_path = os.path.join(self.folder_path, self.image_prename+image_name+'.jpg')
            image_path = os.path.join(self.folder_path, image_name+'.pkl')
            image = load_pkl(image_path)
            # image = torch.load(image_path)
            image = np.array(image).astype(np.uint8)
            data = {'image':image,'text':tokenized_text}

        return data

if __name__ == '__main__':
    from modules.preprocessing.tokenizer import SimpleTokenizer
    tokenizer = SimpleTokenizer(bpe_path='./misc/bpe_simple_vocab_16e6.txt.gz',token_length = 7936)
    test = CocoDataset(data_root='/home/hu/database/coco/', split='val', tokenizer=tokenizer, downstream='diffusion+vqvae')
    print('ok')