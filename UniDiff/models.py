from ast import arg
import math
from sklearn.feature_extraction import img_to_graph
import torch
import torch.nn as nn
from model.diffusion import MultinomialDiffusion, AbsorbDiffusion, UniDiffusion
from modules.transformers import UnifiedTransformer
from modules.simple_transformer import LinearAttentionTransformerEmbedding
# from tokenizers import Tokenizer
from modules.preprocessing.tokenizer import SimpleTokenizer
from modules.vq.taming_gumbel_vqvae import TamingGumbelVQVAE, TamingVQVAE
from modules.ldm.taming_ldm import LDM

from modules.embedding import FuseEmbedding
from prettytable import PrettyTable

class Rezero(torch.nn.Module):
    def __init__(self):
        super(Rezero, self).__init__()
        self.alpha = torch.nn.Parameter(torch.zeros(size=(1,)))

    def forward(self, x):
        return self.alpha * x



def add_model_args(parser):

    parser.add_argument('--input_dp_rate', type=float, default=0.0)

    # Transformer params.
    parser.add_argument('--transformer_dim', type=int, default=1024)
    parser.add_argument('--transformer_heads', type=int, default=16)
    parser.add_argument('--transformer_depth', type=int, default=20)
    parser.add_argument('--attention_type', type=str, default='selfcross')

    parser.add_argument('--diffusion_steps', type=int, default=500)
    parser.add_argument('--diffusion_sharing', type=eval, default=True)
    parser.add_argument('--diffusion_loss', type=str, default='vb_stochastic')
    parser.add_argument('--diffusion_parametrization', type=str, default='x0')

    parser.add_argument('--img_emb', type=int, default=2887)
    # parser.add_argument('--txt_emb', type=int, default=8192)
    parser.add_argument('--spatial_size', type=int, default=32)
    
    parser.add_argument('--VQ_Type', type=str, default='TamingGumbelVQVAE')
    parser.add_argument('--VQpath', type=str)
    parser.add_argument('--VQckpt', type=str)


def get_model_id(args):
    return 'UNIDIFFUSION'


def get_model(args):
    transformer_dim = args.transformer_dim
    transformer_heads = args.transformer_heads
    transformer_depth = args.transformer_depth

    diffusion_steps = args.diffusion_steps
    diffusion_loss = args.diffusion_loss

    img_emb = args.img_emb
    txt_emb = args.txt_emb+256
    spatial_size = args.spatial_size
    length_size = args.length_size
    
    VQ_Type = args.VQ_Type

    attention_type = args.attention_type


    embedding_layer = FuseEmbedding(
                 img_num_embed=img_emb,
                 txt_num_embed=txt_emb,
                #  img_num_embed=2887,
                #  txt_num_embed=16384+256,
                 spatial_size=[spatial_size, spatial_size], # height and with 
                #  spatial_size=[32, 32], # height and with 
                 length_size = length_size,
                 embed_dim=transformer_dim, 
                 trainable=True,
                 pos_emb_type='embedding'
    )

    # Transformer.
    class DynamicsTransformer(nn.Module):
        def __init__(self):
            super(DynamicsTransformer, self).__init__()
            self.transformer = UnifiedTransformer(
                n_layer=transformer_depth,
                n_embd=transformer_dim,
                n_head=transformer_heads,
                block_activate='GELU',
                # condition_dim = transformer_dim,
                attn_pdrop=0,
                resid_pdrop=0,
                mlp_hidden_times=4,
                attn_type=attention_type, # self as default
                diffusion_step=diffusion_steps,
                timestep_type='adalayernorm',
                mlp_type="fc",
                text_cls = txt_emb,
                image_cls = img_emb,
                text_length = length_size,
                image_length = spatial_size**2,
                embedding = embedding_layer
            )
            self.rezero = Rezero()

        def forward(self, x, t, cond):
            x_img, x_txt = self.transformer(x, t, cond)
            if x_img is not None:
                # x_img = x_img.permute(0, 2, 1)
                x_img = self.rezero(x_img)
            if x_txt is not None:
                # x_txt = x_txt.permute(0, 2, 1)
                x_txt = self.rezero(x_txt)
            return x_img, x_txt

    # class DynamicsTransformer(nn.Module):
    #     def __init__(self):
    #         super(DynamicsTransformer, self).__init__()
    #         self.transformer = LinearAttentionTransformerEmbedding(
    #             dim=transformer_dim,
    #             heads=transformer_heads,
    #             depth=transformer_depth,
    #             n_blocks=1,
    #             num_timesteps=diffusion_steps,
    #             causal=False,  # auto-regressive or not
    #             ff_dropout=0,  # dropout for feedforward
    #             # dropout right after self-attention layer
    #             attn_dropout=0,  # dropout post-attention
    #             n_local_attn_heads=4,
    #             # number of local attention heads for (qk)v attention.
    #             # this can be a tuple specifying the exact number of local
    #             # attention heads at that depth
    #             local_attn_window_size=128,
    #             # receptive field of the local attention
    #             reversible=False,
    #             # use reversible nets, from Reformer paper
    #             embedding = embedding_layer,
    #             text_cls = 8192,
    #             image_cls = 974,
    #             # text_cls = 16384+256,
    #             # image_cls = 2887,
    #             text_length = 128,
    #             image_length= 256
    #             # image_length = 1024,
    #         )

    #         self.rezero = Rezero()

    #     def forward(self, x, t):
    #         x = self.transformer(x, t)
    #         x = x.permute(0, 2, 1)
    #         x = self.rezero(x)
    #         return x

    dynamics = DynamicsTransformer()

    if args.tokenizer == 'SimpleTokenizer':
        tokenizer = SimpleTokenizer(bpe_path='misc/bpe_simple_vocab_16e6.txt.gz',token_length = 7936)
    else:
        tokenizer = Tokenizer.from_file("misc/cub_tokenizer.json")
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=length_size)
    # tokenizer = SimpleTokenizer(bpe_path='./misc/bpe_simple_vocab_16e6.txt.gz',token_length = 16384)
    # vq_model = TamingGumbelVQVAE().quantize_number
    if VQ_Type == 'TamingVQVAE':
    # vq_model = TamingGumbelVQVAE()
        vq_model = TamingVQVAE(
            trainable=False,
            token_shape=[16,16],
            config_path='misc/taming_dvae/vqgan_imagenet_f16_16384.yaml',
            ckpt_path='misc/taming_dvae/vqgan_imagenet_f16_16384.pth',
            num_tokens=16384,
            quantize_number=974,
            mapping_path='misc/statistics/taming_vqvae_974.pt',
        )
    elif VQ_Type == 'TamingGumbelVQVAE':
        vq_model = TamingGumbelVQVAE()
    elif VQ_Type == 'LDMF4':
        vq_model = LDM()
    elif VQ_Type == 'LDMF8':
        vq_model = LDM(
            token_shape=[32,32],
            config_path=args.VQpath,
            ckpt_path=args.VQckpt,
            num_tokens=args.img_emb,
        )
    # vq_model = 2887

    base_dist = UniDiffusion(
        dynamics,
        timesteps=diffusion_steps,
        loss_type=diffusion_loss,
        text_tokenizer=tokenizer, # or tokenizer,
        image_tokenizer=vq_model)
    pts = PrettyTable()
    pts.add_rows(
        [
            ["Diffusion Step", diffusion_steps],
            ["Image Embedding", img_emb],
            ["Text Embedding", txt_emb],
            ["Spatial Size", spatial_size],
            ["Length Size", length_size],
            ["VQ Type", VQ_Type],
        ]
    )
    print(pts)
    return base_dist


