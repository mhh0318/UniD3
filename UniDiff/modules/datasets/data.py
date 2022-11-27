
import torch
from torch.utils.data import DataLoader, ConcatDataset
from .dataset_coco import CocoDataset
from .dataset_cub import CubDataset
from modules.preprocessing.tokenizer import SimpleTokenizer
from torch.utils.data.dataloader import default_collate
import pytorch_lightning as pl
from tokenizers import Tokenizer

def add_data_args(parser):

    # Data params
    parser.add_argument('--dataset', type=str, default='coco_caption')
    # parser.add_argument('--validation', type=eval, default=True)
    parser.add_argument('--root', type=str, default='/home/hu/database/coco/')
    parser.add_argument('--split', type=str, default='normal')

    # Train params
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=eval, default=False)
    parser.add_argument('--downstream', type=str, default='diffusion+vqvae')
    parser.add_argument('--txt_emb', type=int, default=7936) #8192-256
    parser.add_argument('--length_size', type=int, default=128)
    parser.add_argument('--tokenizer', type=str, default='SimpleTokenizer')
    


def get_data_id(args):
    return args.dataset


def get_data(args):

    # Dataset

    
    if args.dataset == 'coco_caption':
        tokenizer = SimpleTokenizer(bpe_path='/home/hu/UniDm/misc/bpe_simple_vocab_16e6.txt.gz',token_length = 7936)
        # train = CocoDataset(data_root=args.root, split='train', tokenizer=tokenizer,downstream='diffusion')
        # valid = CocoDataset(data_root=args.root, split='val', tokenizer=tokenizer,downstream='diffusion')
        train = CocoDataset(data_root=args.root, split='train', tokenizer=tokenizer,downstream=args.downstream)
        if args.split == 'normal':
            valid = CocoDataset(data_root=args.root, split='val', tokenizer=tokenizer, downstream=args.downstream)
        else:
            valid = CocoDataset(data_root=args.root, split='karpathy', tokenizer=tokenizer, downstream=args.downstream)
        # test = CocoDataset(data_root=args.root, split='test',tokenizer=tokenizer)
        # data_shape = (128,)
        # num_classes = tokenizer.vocab_size
    elif args.dataset == 'cub200':
        if args.tokenizer == 'SimpleTokenizer':
            tokenizer = SimpleTokenizer(bpe_path='/home/hu/UniDm/misc/bpe_simple_vocab_16e6.txt.gz',token_length = 7936)
        else:
            tokenizer = Tokenizer.from_file("misc/cub_tokenizer.json")
            tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=args.length_size)
        # train = CocoDataset(data_root=args.root, split='train', tokenizer=tokenizer,downstream='diffusion')
        # valid = CocoDataset(data_root=args.root, split='val', tokenizer=tokenizer,downstream='diffusion')
        train = CubDataset(data_root=args.root, split='train', tokenizer=tokenizer,downstream=args.downstream)
        valid = CubDataset(data_root=args.root, split='test', tokenizer=tokenizer,downstream=args.downstream)


    # Data Loader
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=default_collate)
    eval_loader = DataLoader(valid, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=default_collate)
    # else:
    #     dataset_train = ConcatDataset([train, valid])
    #     train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
    #     eval_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    return train_loader, eval_loader
    # return train_loader, eval_loader, data_shape, num_classes

class DataModuleClass(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        if args.tokenizer == 'SimpleTokenizer':
            self.tokenizer = SimpleTokenizer(bpe_path='/home/hu/UniDm/misc/bpe_simple_vocab_16e6.txt.gz',token_length = 7936)
        else:
            self.tokenizer = Tokenizer.from_file("misc/cub_tokenizer.json")
            self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=args.length_size)
        self.args =args
    def setup(self, stage=None):
        args = self.args
        if args.dataset == 'coco_caption':
            self.train_data = CocoDataset(data_root=args.root, split='train', tokenizer=self.tokenizer, downstream='diffusion+vqvae')
            if args.split == 'normal':
                self.valid_data = CocoDataset(data_root=args.root, split='val', tokenizer=self.tokenizer, downstream='diffusion+vqvae')
            else:
                self.valid_data = CocoDataset(data_root=args.root, split='karpathy', tokenizer=self.tokenizer, downstream='diffusion+vqvae')
            
        elif args.dataset == 'cub200':
            self.train_data = CubDataset(data_root=args.root, split='train', tokenizer=self.tokenizer, downstream='diffusion+vqvae')
            self.valid_data = CubDataset(data_root=args.root, split='test', tokenizer=self.tokenizer, downstream='diffusion+vqvae')

    def train_dataloader(self):
        args = self.args
        return DataLoader(self.train_data,batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=default_collate)

    def val_dataloader(self):
        args = self.args
        return DataLoader(self.valid_data,batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=default_collate)

def get_pl_datamodule(args):
    return DataModuleClass(args)
