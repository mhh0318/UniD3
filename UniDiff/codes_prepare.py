import torch
import pickle
from torch.utils.data import DataLoader, ConcatDataset
from modules.datasets.dataset_coco import CocoDataset
from modules.preprocessing.tokenizer import SimpleTokenizer
from modules.vq.taming_gumbel_vqvae import TamingGumbelVQVAE

from torchvision.utils import save_image

from tqdm import tqdm

# def dfs_showdir(path, depth):
#     if depth == 0:
#         print("root:[" + path + "]")
#     for item in os.listdir(path):
#         if '.jpg' not in item:
#             print("| " * depth + "+--" + item)
#             newitem = path +'/'+ item
#             if os.path.isdir(newitem):
#                 dfs_showdir(newitem, depth +1)

# dfs_showdir(root, 0)

"""
root:[/home/hu/database/coco/]
+--val2017
| +--00000001.jpg
| +--00000002.jpg
| +-- ......
+--annotations
| +--instances_val2017.json
| +--instances_train2017.json
| +--person_keypoints_train2017.json
| +--person_keypoints_val2017.json
| +--captions_val2017.json
| +--captions_train2017.json
+--train2017
| +--00000001.jpg
| +--00000002.jpg
| +-- ......
+--test2017
| +--00000001.jpg
| +--00000002.jpg
| +-- ......
"""
def save_pkl(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print('save pkl file:', filepath)

def get_pkl_path(filepath):
    return '.'.join([filepath.split('.')[0],'pkl'])

root = '/home/hu/database/coco/'

tokenizer = SimpleTokenizer(bpe_path='/home/hu/database/bpe_simple_vocab_16e6.txt.gz')
train = CocoDataset(data_root=root, split='train', tokenizer=tokenizer, downstream='vqvae')
valid = CocoDataset(data_root=root, split='val', tokenizer=tokenizer, downstream='vqvae')

loader = DataLoader(ConcatDataset([train,valid]), batch_size=64, shuffle=False)

vq_model = TamingGumbelVQVAE().cuda()

for i, data in enumerate(tqdm(loader)):
    with torch.no_grad():
        tokens = vq_model.get_tokens(data['image'].cuda())
        # imgs = vq_model.decode(tokens)

    # save_image(imgs[3],'recons.png')
    # save_image(vq_model.preprocess(data['image'])[0],'raw.png')
    
    assert tokens.shape[1] == 1024
    list(map(save_pkl,tokens.cpu(),[get_pkl_path(data['path'][j]) for j in range(len(tokens))]))
    # for k in range(tokens):
    #     pkl_path = '.'.join([data['path'][k].split('.')[0],'pkl'])
    #     save_pkl(tokens[k,:], pkl_path)

# '.'.join([a['path'][0].split('.')[0],'pkl'])