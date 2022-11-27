import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedEmbedding(nn.Embedding):
    def __init__(self, linear, start_index, end_index, **kwargs):
        super().__init__(end_index - start_index, linear.weight.shape[1], **kwargs)
        del self.weight

        self.linear = linear
        self.start_index = start_index
        self.end_index = end_index

    def forward(self, input):
        return F.embedding(
            input, torch.cat((self.linear.weight[self.start_index:self.end_index,:],self.linear.weight[-1:,:]),0), self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

class FuseEmbedding(nn.Module):
    def __init__(self,
                 img_num_embed=2887,
                 txt_num_embed=16640,
                 spatial_size=[32, 32], # height and with 
                 length_size = 128,
                 embed_dim=1024, 
                 trainable=True,
                 pos_emb_type='embedding'
        ):
        super().__init__()
        
        if isinstance(spatial_size, int):
            spatial_size = [spatial_size, spatial_size]

        num_embed = img_num_embed + txt_num_embed + 1  #share embedding. 1 for mask and 256 for unique embedding
        self.text_num_embed = txt_num_embed
        self.img_num_embed = img_num_embed
        self.spatial_size = spatial_size
        self.length_size = length_size
        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.trainable = trainable
        self.pos_emb_type = pos_emb_type

        self.to_logits = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self.num_embed),
        )

        assert self.pos_emb_type in ['embedding', 'parameter']
        
        # self.emb = nn.Embedding(self.num_embed, embed_dim)

        # self.image_emb = SharedEmbedding(self.to_logits[1], 0, img_num_embed)
        # self.text_emb = SharedEmbedding(self.to_logits[1], img_num_embed, num_embed-1)

        self.embedding = nn.Embedding(self.num_embed, embed_dim)

        if self.pos_emb_type == 'embedding':
            self.height_emb = nn.Embedding(self.spatial_size[0], embed_dim) # height   
            self.width_emb = nn.Embedding(self.spatial_size[1], embed_dim) # width
            self.length_emb = nn.Embedding(length_size, embed_dim) # length 1 for SOS
        else:
            self.height_emb = nn.Parameter(torch.zeros(1, self.spatial_size[0], embed_dim)) # height #32,1024
            self.width_emb = nn.Parameter(torch.zeros(1, self.spatial_size[1], embed_dim)) # width   #32,1024
            self.length_emb = nn.Parameter(torch.zeros(1, length_size, embed_dim)) # length 1 for SOS

    def forward(self, x, **kwargs):

        # assert self.spatial_size[0] * self.spatial_size[1] + self.length_size == x.size(1)

        img_index = x[:,:self.spatial_size[0] * self.spatial_size[1]]
        text_index = x[:,-self.length_size:]



        # text_index =  text_index - self.img_num_embed
        # txt_emb = self.text_emb(text_index)
        txt_emb = self.embedding(text_index)
        lg_emb = self.length_emb(torch.arange(text_index.shape[1], device = text_index.device))
        txt_emb = txt_emb + lg_emb

        assert img_index.dim() == 2 # B x L
        # try:
        # img_index[img_index < 0] = 0  
        # mask = img_index == self.num_embed-1
        # img_index[mask] = self.img_num_embed
        img_emb = self.embedding(img_index)

        # add col and row embedding
        if img_emb.shape[1] > 0:
        # if False:
            if self.pos_emb_type == 'embedding':
                height_emb = self.height_emb(torch.arange(self.spatial_size[0], device=img_index.device).view(1, self.spatial_size[0])).unsqueeze(2) # 1 x H x D -> 1 x H x 1 x D
                width_emb = self.width_emb(torch.arange(self.spatial_size[1], device=img_index.device).view(1, self.spatial_size[1])).unsqueeze(1) # 1 x W x D -> 1 x 1 x W x D
            else:
                height_emb = self.height_emb.unsqueeze(2) # 1 x H x D -> 1 x H x 1 x D
                width_emb = self.width_emb.unsqueeze(1) # 1 x W x D -> 1 x 1 x W x D
            pos_emb = (height_emb + width_emb).view(1, self.spatial_size[0] * self.spatial_size[1], -1) # 1 x H x W x D -> 1 x L xD
            img_emb = img_emb + pos_emb[:, :img_emb.shape[1], :]
        


        emb = torch.cat([img_emb, txt_emb], dim=1)
        return emb