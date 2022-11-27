import torch
import torch.nn as nn
from omegaconf import OmegaConf

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UniDiff.modules.ldm.autoencoder import VQModelInterface


import math

class LDM(nn.Module):
    def __init__(
            self, 
            trainable=False,
            token_shape=[64,64],
            config_path='misc/taming_dvae/f4_8192.yaml',
            ckpt_path='/home/hu/UniDm/f4model.ckpt',
            num_tokens=8192,
        ):
        super().__init__()
        
        model = self.LoadModel(config_path, ckpt_path)
        self.model = model
        self.num_tokens = num_tokens

        self.trainable = trainable
        self.token_shape = token_shape
        # self._set_trainable()

    def LoadModel(self, config_path, ckpt_path):
        config = OmegaConf.load(config_path)
        model = VQModelInterface(**config.model.params)
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        # new_sd = {}
        # for k in sd.keys():
        #     if 'first_stage_model' in k:
        #         new_sd[k.replace('first_stage_model.', '')] = sd[k]

        model.load_state_dict(sd, strict=False)
        return model

    @property
    def device(self):
        # import pdb; pdb.set_trace()
        return self.model.quant_conv.weight.device

    def preprocess(self, imgs):
        """
        imgs: B x C x H x W, in the range 0-255
        """
        # imgs = rearrange(imgs, 'b h w c -> b c h w')
        imgs = torch.permute(imgs, (0, 3, 1, 2))
        imgs = imgs.div(255) # map to 0 - 1
        return imgs*2-1
    
    def postprocess(self, imgs):
        """
        imgs: B x C x H x W, in the range 0-1
        """
        imgs = imgs * 255
        return imgs

    def get_tokens(self, imgs, **kwargs):
        imgs = self.preprocess(imgs)
        code = self.model.encode(imgs)
        # output = {'token': code}
        # output = {'token': rearrange(code, 'b h w -> b (h w)')}
        return code

    def decode(self, img_seq):

        b, n = img_seq.shape
        # img_seq = rearrange(img_seq, 'b (h w) -> b h w', h = int(math.sqrt(n)))
        img_seq = img_seq.view(b, int(math.sqrt(n)), int(math.sqrt(n)))

        x_rec = self.model.decode(img_seq)
        # x_rec = self.postprocess(x_rec)
        return x_rec


if __name__ == '__main__':
    model = LDM()
    print(model)