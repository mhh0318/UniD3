
from cgitb import text
from cmath import log
import torch
import torch.nn.functional as F
import numpy as np
from inspect import isfunction


"""
Based in part on: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L281
"""
eps = 1e-8


def sum_except_batch(x, num_dims=1):
    '''
    Sums all dimensions except the first.

    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)

    Returns:
        x_sum: Tensor, shape (batch_size,)
    '''
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def exists(x):
    return x is not None


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def index_to_log_onehot(x, num_classes):
    # assert x.max().item() < num_classes, \
    #     f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)

    permute_order = (0, -1) + tuple(range(1, len(x.size())))

    x_onehot = x_onehot.permute(permute_order)

    log_x = torch.log(x_onehot.float().clamp(min=1e-30))

    return log_x


def log_onehot_to_index(log_x):
    return log_x.argmax(1)

def alpha_schedule(time_step, N_txt=100, N_img=100, att_1 = 0.99999, att_T = 0.000009, ctt_1 = 0.000009, ctt_T = 0.99999):
    att = np.arange(0, time_step)/(time_step-1)*(att_T - att_1) + att_1
    att = np.concatenate(([1], att))
    at = att[1:]/att[:-1]
    ctt = np.arange(0, time_step)/(time_step-1)*(ctt_T - ctt_1) + ctt_1
    ctt = np.concatenate(([0], ctt))
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1-one_minus_ct
    bt_img = (1-at-ct)/N_img
    bt_txt = (1-at-ct)/N_txt
    att = np.concatenate((att[1:],[1]))
    ctt = np.concatenate((ctt[1:],[0]))
    btt_img = (1-att-ctt)/N_img
    btt_txt = (1-att-ctt)/N_txt
    return at, bt_img, bt_txt, ct, att, btt_img, btt_txt, ctt


class UniDiffusion(torch.nn.Module):
    def __init__(self, denoise_fn, timesteps=1000, auxiliary_loss_weight=0.01, mask_weight=None,
                 loss_type='vb_stochastic', parametrization='x0', text_tokenizer = None, image_tokenizer = None):
        super(UniDiffusion, self).__init__()
        assert loss_type in ('vb_stochastic', 'vb_all')
        assert parametrization in ('x0', 'direct')

        if loss_type == 'vb_all':
            print('Computing the loss using the bound on _all_ timesteps.'
                  ' This is expensive both in terms of memory and computation.')
        try:
            txt_classes = text_tokenizer.vocab_size
        except:
            txt_classes = text_tokenizer.get_vocab_size()
        if type(image_tokenizer) == int :
            img_classes = image_tokenizer
        else:
            try:

                img_classes = image_tokenizer.quantize_number if image_tokenizer.quantize_number != 0 else image_tokenizer.num_tokens
            except:
                img_classes = image_tokenizer.num_tokens

        self.img_classes = img_classes
        self.txt_classes = txt_classes

        self.num_classes = img_classes+txt_classes+1 #mask

        self.text_tokenizer = text_tokenizer
        self.image_tokenizer = image_tokenizer
        
        self.mask_weight = mask_weight

        self._denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.num_timesteps = timesteps
        self.parametrization = parametrization
        self.auxiliary_loss_weight=auxiliary_loss_weight

        at, bt_img, bt_txt, ct, att, btt_img, btt_txt, ctt = alpha_schedule(self.num_timesteps, N_txt=txt_classes, N_img=img_classes)
        at = torch.tensor(at.astype('float64'))
        bt_img = torch.tensor(bt_img.astype('float64'))
        bt_txt = torch.tensor(bt_txt.astype('float64'))
        ct = torch.tensor(ct.astype('float64'))
        log_at = torch.log(at)
        log_bt_img = torch.log(bt_img)
        log_bt_txt = torch.log(bt_txt)
        log_ct = torch.log(ct)
        att = torch.tensor(att.astype('float64'))
        btt_img = torch.tensor(btt_img.astype('float64'))
        btt_txt = torch.tensor(btt_txt.astype('float64'))
        ctt = torch.tensor(ctt.astype('float64'))
        log_cumprod_at = torch.log(att)
        log_cumprod_bt_img = torch.log(btt_img)
        log_cumprod_bt_txt = torch.log(btt_txt)
        log_cumprod_ct = torch.log(ctt)

        log_1_min_ct = log_1_min_a(log_ct)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)

        assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1.e-5 
        assert log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item() < 1.e-5

        self.diffusion_acc_list = [0] * self.num_timesteps
        self.diffusion_keep_list = [0] * self.num_timesteps
        self.register_buffer('log_at', log_at.float())
        self.register_buffer('log_bt_img', log_bt_img.float())
        self.register_buffer('log_bt_txt', log_bt_txt.float())
        self.register_buffer('log_ct', log_ct.float())
        self.register_buffer('log_cumprod_at', log_cumprod_at.float())
        self.register_buffer('log_cumprod_bt_img', log_cumprod_bt_img.float())
        self.register_buffer('log_cumprod_bt_txt', log_cumprod_bt_txt.float())
        self.register_buffer('log_cumprod_ct', log_cumprod_ct.float())
        self.register_buffer('log_1_min_ct', log_1_min_ct.float())
        self.register_buffer('log_1_min_cumprod_ct', log_1_min_cumprod_ct.float())

        self.register_buffer('Lt_history', torch.zeros(timesteps))
        self.register_buffer('Lt_count', torch.zeros(timesteps))

    def multinomial_kl(self, log_prob1, log_prob2):
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t, t):
        log_at = extract(self.log_at, t, log_x_t.shape)             # at
        log_bt_img = extract(self.log_bt_img, t, log_x_t.shape)             # bt
        log_bt_txt = extract(self.log_bt_txt, t, log_x_t.shape)             # bt
        log_ct = extract(self.log_ct, t, log_x_t.shape)             # ct
        log_1_min_ct = extract(self.log_1_min_ct, t, log_x_t.shape)          # 1-ct

        log_probs = torch.cat(
            [
                log_add_exp(log_x_t[:,:self.img_classes,:]+log_at, log_bt_img),
                log_add_exp(log_x_t[:,self.img_classes:self.img_classes+self.txt_classes,:]+log_at, log_bt_txt),
                log_add_exp(log_x_t[:, -1:, :] + log_1_min_ct, log_ct)
            ],
            dim=1
        )
        return log_probs

    def q_pred(self, log_x_start, t):
        t = (t + (self.num_timesteps + 1))%(self.num_timesteps + 1)
        log_cumprod_at = extract(self.log_cumprod_at, t, log_x_start.shape)         # at~
        log_cumprod_bt_txt = extract(self.log_cumprod_bt_txt, t, log_x_start.shape)         # bt~
        log_cumprod_bt_img = extract(self.log_cumprod_bt_img, t, log_x_start.shape)         # bt~
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)         # ct~
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, t, log_x_start.shape)       # 1-ct~
        

        log_probs = torch.cat(
            [
                log_add_exp(log_x_start[:,:self.img_classes,:]+log_cumprod_at, log_cumprod_bt_img),
                log_add_exp(log_x_start[:,self.img_classes:self.img_classes+self.txt_classes,:]+log_cumprod_at, log_cumprod_bt_txt),
                log_add_exp(log_x_start[:,-1:,:]+log_1_min_cumprod_ct, log_cumprod_ct), # 
                # torch.log(torch.exp(log_x_start[:,-1:,:]+log_1_min_cumprod_ct) + torch.exp(log_cumprod_ct))
            ],
            dim=1
        )

        return log_probs
    
    def q_pred_cond(self, log_x_start, t, cond):
        t = (t + (self.num_timesteps + 1))%(self.num_timesteps + 1)
        log_cumprod_at = extract(self.log_cumprod_at, t, log_x_start.shape)         # at~
        log_cumprod_bt_txt = extract(self.log_cumprod_bt_txt, t, log_x_start.shape)         # bt~
        log_cumprod_bt_img = extract(self.log_cumprod_bt_img, t, log_x_start.shape)         # bt~
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)         # ct~
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, t, log_x_start.shape)       # 1-ct~
        
        if cond == 'txt':
            log_probs = torch.cat(
                [
                    log_add_exp(log_x_start[:,:-1,:]+log_cumprod_at, log_cumprod_bt_txt),
   
                    log_add_exp(log_x_start[:,-1:,:]+log_1_min_cumprod_ct, log_cumprod_ct), # 

                ],
                dim=1
            )
        elif cond == 'img':
            log_probs = torch.cat(
                [
                    log_add_exp(log_x_start[:,:-1,:]+log_cumprod_at, log_cumprod_bt_img),

                    log_add_exp(log_x_start[:,-1:,:]+log_1_min_cumprod_ct, log_cumprod_ct), # 
 
                ],
                dim=1
            )
        elif cond is None:
            log_probs = torch.cat(
                [
                    log_add_exp(log_x_start[:,:self.img_classes,:]+log_cumprod_at, log_cumprod_bt_img),
                    log_add_exp(log_x_start[:,self.img_classes:self.img_classes+self.txt_classes,:]+log_cumprod_at, log_cumprod_bt_txt),
                    # log_add_exp(log_x_start[:,-1:,:]+log_1_min_cumprod_ct, log_cumprod_ct), # 
                    torch.log(torch.exp(log_x_start[:,-1:,:]+log_1_min_cumprod_ct) + torch.exp(log_cumprod_ct))
                ],
                dim=1
            )

        return log_probs

    def predict_start(self, log_x_t, t, cond=None):
        x_t = log_onehot_to_index(log_x_t)
        batch_size = log_x_t.size()[0]
        logits_img, logits_txt = self._denoise_fn(x_t, t, cond=cond)

        max_neg_value = -1e4

        if cond == 'img':
            logits_txt = F.pad(logits_txt,[self.img_classes,0],value=max_neg_value)
            logits = logits_txt.permute(0,2,1)
            zero_vector = torch.zeros(batch_size, 1, self.txt_shape).type_as(log_x_t)- 70
        elif cond == 'txt':
            logits_img = F.pad(logits_img,[0,self.txt_classes],value=max_neg_value)
            logits = logits_img.permute(0,2,1)
            zero_vector = torch.zeros(batch_size, 1, self.img_shape).type_as(log_x_t)- 70
        else:
            logits_img = F.pad(logits_img,[0,self.txt_classes],value=max_neg_value)
            logits_txt = F.pad(logits_txt,[self.img_classes,0],value=max_neg_value)
            logits = torch.cat([logits_img, logits_txt], dim=1).permute(0,2,1)
            zero_vector = torch.zeros(batch_size, 1, self.shape[1]).type_as(log_x_t)- 70

        assert logits.size(0) == x_t.size(0)
        assert logits.size(1) == self.num_classes-1

        log_pred = F.log_softmax(logits.double(), dim=1).float()

        log_pred = torch.cat((log_pred, zero_vector), dim=1)
        log_pred = torch.clamp(log_pred, -70, 0)
        return log_pred

    def q_posterior(self, log_x_start, log_x_t, t, cond=None):
        # notice that log_x_t is onehot
        assert t.min().item() >= 0 and t.max().item() < self.num_timesteps
        batch_size = log_x_start.size()[0]
        onehot_x_t = log_onehot_to_index(log_x_t)
        mask = (onehot_x_t == self.num_classes-1).unsqueeze(1) 
        log_one_vector = torch.zeros(batch_size, 1, 1).type_as(log_x_t)
    

        if cond == 'img':
            log_qt = self.q_pred_cond(log_x_t, t , cond=cond)                                  # q(xt|x0)
            log_zero_vector = torch.log(log_one_vector+1.0e-30).expand(-1, -1, self.txt_shape)
        elif cond == 'txt':
            log_qt = self.q_pred_cond(log_x_t, t , cond=cond)      
            log_zero_vector = torch.log(log_one_vector+1.0e-30).expand(-1, -1, self.img_shape)
        else:
            log_qt = self.q_pred(log_x_t, t)                                  # q(xt|x0)
            log_zero_vector = torch.log(log_one_vector+1.0e-30).expand(-1, -1, self.shape[1])

        log_qt = log_qt[:,:-1,:]
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)         
        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.num_classes-1, -1)

        log_qt = (~mask)*log_qt + mask*ct_cumprod_vector

        # https://github.com/ehoogeboom/multinomial_diffusion/blob/66f17340e4cd200059bff228cf98a597bf084c26/diffusion_utils/diffusion_multinomial.py#L191

        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t)        # q(xt|xt_1)
        log_qt_one_timestep = torch.cat((log_qt_one_timestep[:,:-1,:], log_zero_vector), dim=1)
        log_ct = extract(self.log_ct, t, log_x_start.shape)         # ct
        ct_vector = log_ct.expand(-1, self.num_classes-1, -1)
        ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        log_qt_one_timestep = (~mask)*log_qt_one_timestep + mask*ct_vector
        
        # ADDITIONAL LOSS TRICK NEED 
        q = log_x_start[:,:-1,:] - log_qt
        q = torch.cat((q, log_zero_vector), dim=1)
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp
        log_EV_xtmin_given_xt_given_xstart = self.q_pred(q, t-1) + log_qt_one_timestep + q_log_sum_exp
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)

    def p_pred(self, log_x, t, delta=None, cond=None, cond_prob=None):
        if self.parametrization == 'x0': 
            if cond is not None:
                log_x_recon = self.predict_start(torch.cat([log_x, cond_prob], dim=2), t=t, cond=cond)  #only text condition
            else:
                log_x_recon = self.predict_start(log_x, t=t, cond=cond)
            if delta is None:
                delta = 0
    
            log_x0_recon = log_x_recon
            if t[0].item() >= delta:
                log_model_pred = self.q_posterior(
                    log_x_start=log_x0_recon, log_x_t=log_x, t=t-delta, cond=cond)
            else:
                log_model_pred = self.q_posterior(
                    log_x_start=log_x0_recon, log_x_t=log_x, t=t, cond=cond)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(log_x, t=t)
        else:
            raise ValueError
        return log_model_pred

    @torch.no_grad()
    def p_sample(self, log_x, t, delta=None, cond=None, cond_prob=None):
        model_log_prob = self.p_pred(log_x=log_x, t=t, delta=delta, cond=cond, cond_prob=cond_prob)
        out = self.log_sample_categorical(model_log_prob)
        return out

    def log_sample_categorical(self, logits): # sample onehot vector from log probability
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t):
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)

        log_sample = self.log_sample_categorical(log_EV_qxt_x0)

        return log_sample

    def nll(self, log_x_start):
        b = log_x_start.size(0)
        device = log_x_start.device
        loss = 0
        for t in range(0, self.num_timesteps):
            t_array = (torch.ones(b, device=device) * t).long()

            kl = self.compute_Lt(
                log_x_start=log_x_start,
                log_x_t=self.q_sample(log_x_start=log_x_start, t=t_array),
                t=t_array)

            loss  = loss+ kl

        loss = loss + self.kl_prior(log_x_start)

        return loss

    def kl_prior(self, log_x_start):
        b = log_x_start.size(0)
        device = log_x_start.device
        ones = torch.ones(b, device=device).long()

        log_qxT_prob = self.q_pred(log_x_start, t=(self.num_timesteps - 1) * ones)
        log_half_prob = -torch.log(self.num_classes * torch.ones_like(log_qxT_prob))

        kl_prior = self.multinomial_kl(log_qxT_prob, log_half_prob)
        return sum_except_batch(kl_prior)

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def _train_loss(self, x, cond):
        b, device = x.size(0), x.device

        assert self.loss_type == 'vb_stochastic'
        x_start = x
        t, pt = self.sample_time(b, device, 'importance')


        log_x_start = index_to_log_onehot(x_start, self.num_classes)
        log_xt = self.q_sample(log_x_start=log_x_start, t=t)
        # if cond == 'txt':
            
        # xt = log_onehot_to_index(log_xt)
            # log_x_start = log_x_start[:,:,:1024]

        ############### go to p_theta function ###############


        ###UniDM cond as condition
        if cond == 'txt':
            x_start = x_start[:,:self.img_shape]
            log_xt = log_xt[:,:,:self.img_shape]
            cond_prob = log_x_start[:,:,self.img_shape:]
            log_x_start = log_x_start[:,:,:self.img_shape]
            log_x0_recon_cond = self.predict_start(torch.cat([log_xt,cond_prob],dim=-1), t=t, cond=cond)            # P_theta(x0|xt)
            log_x0_recon = log_x0_recon_cond
            # log_x0_recon = torch.cat((log_x0_recon_cond,log_x_start[:,:,self._denoise_fn.transformer.image_length:]),-1)
            log_model_prob = self.q_posterior(log_x_start=log_x0_recon_cond, log_x_t=log_xt, t=t, cond=cond) 
        elif cond == 'img':
            x_start = x_start[:,self.img_shape:]
            log_xt = log_xt[:,:,self.img_shape:]
            cond_prob = log_x_start[:,:,:self.img_shape]
            log_x_start = log_x_start[:,:,self.img_shape:]
            log_x0_recon_cond = self.predict_start(torch.cat([cond_prob,log_xt],dim=-1), t=t, cond=cond)            # P_theta(x0|xt)
            log_x0_recon = log_x0_recon_cond
            # log_x0_recon = torch.cat((log_x_start[:,:,:self._denoise_fn.transformer.image_length],log_x0_recon_cond),-1)
            log_model_prob = self.q_posterior(log_x_start=log_x0_recon_cond, log_x_t=log_xt, t=t,cond=cond) 
        elif cond == 'txt_full':
            log_xt[:,:,self.img_shape:] = log_x_start[:,:,self.img_shape:]
            log_x0_recon_cond = self.predict_start(log_xt, t=t, cond=cond)    
            log_x0_recon = log_x0_recon_cond
            log_x0_recon_cond_txt = torch.cat((log_x0_recon_cond[:,:,:self.img_shape],log_x_start[:,:,self.img_shape:]),-1)
            # log_x0_recon_cond_img = torch.cat((log_x_start[:,:,:self.img_shape],log_x0_recon_cond[:,:,self.img_shape:]),-1)

            log_model_prob_cond_txt = self.q_posterior(log_x_start=log_x0_recon_cond_txt, log_x_t=log_xt, t=t)      # go through q(xt_1_img|xt,x0,xt_1_txt)
            # log_model_prob_cond_img = self.q_posterior(log_x_start=log_x0_recon_cond_img, log_x_t=log_xt, t=t)      # go through q(xt_1_txt|xt,x0,xt_1_img)

            log_model_prob = torch.cat([log_model_prob_cond_txt[:,:,:self.img_shape],log_x_start[:,:,self.img_shape:]],-1)
        elif cond == 'img_full':
            log_xt[:,:,:self.img_shape] = log_x_start[:,:,:self.img_shape]
            log_x0_recon_cond = self.predict_start(log_xt, t=t, cond=cond)    
            log_x0_recon = log_x0_recon_cond
            log_x0_recon_cond_img = torch.cat((log_x_start[:,:,:self.img_shape],log_x0_recon_cond[:,:,self.img_shape:]),-1)
            # log_x0_recon_cond_img = torch.cat((log_x_start[:,:,:self.img_shape],log_x0_recon_cond[:,:,self.img_shape:]),-1)

            log_model_prob_cond_img = self.q_posterior(log_x_start=log_x0_recon_cond_img, log_x_t=log_xt, t=t)      # go through q(xt_1_img|xt,x0,xt_1_txt)
            # log_model_prob_cond_img = self.q_posterior(log_x_start=log_x0_recon_cond_img, log_x_t=log_xt, t=t)      # go through q(xt_1_txt|xt,x0,xt_1_img)

            log_model_prob = torch.cat([log_x_start[:,:,:self.img_shape],log_model_prob_cond_img[:,:,self.img_shape:]],-1)

        elif cond is None:
            log_x0_recon_cond = self.predict_start(log_xt, t=t, cond=cond)    
            log_x0_recon = log_x0_recon_cond
            log_x0_recon_cond_txt = torch.cat((log_x0_recon_cond[:,:,:self.img_shape],log_x_start[:,:,self.img_shape:]),-1)
            log_x0_recon_cond_img = torch.cat((log_x_start[:,:,:self.img_shape],log_x0_recon_cond[:,:,self.img_shape:]),-1)

            log_model_prob_cond_txt = self.q_posterior(log_x_start=log_x0_recon_cond_txt, log_x_t=log_xt, t=t)      # go through q(xt_1_img|xt,x0,xt_1_txt)
            log_model_prob_cond_img = self.q_posterior(log_x_start=log_x0_recon_cond_img, log_x_t=log_xt, t=t)      # go through q(xt_1_txt|xt,x0,xt_1_img)

            log_model_prob = torch.cat([log_model_prob_cond_txt[:,:,:self.img_shape],log_model_prob_cond_img[:,:,self.img_shape:]],-1)

        ################## compute acc list ################
        x0_recon = log_onehot_to_index(log_x0_recon)
        x0_real = x_start
        xt_1_recon = log_onehot_to_index(log_model_prob)

        xt_recon = log_onehot_to_index(log_xt)
        for index in range(t.size()[0]):
            this_t = t[index].item()
            same_rate = (x0_recon[index] == x0_real[index]).sum().cpu()/x0_real.size()[1]
            self.diffusion_acc_list[this_t] = same_rate.item()*0.1 + self.diffusion_acc_list[this_t]*0.9
            same_rate = (xt_1_recon[index] == xt_recon[index]).sum().cpu()/xt_recon.size()[1]
            self.diffusion_keep_list[this_t] = same_rate.item()*0.1 + self.diffusion_keep_list[this_t]*0.9

        # Compute Lt
        # compute log_true_prob now 
        if cond == 'txt_full':
            log_true_prob_ = self.q_posterior(log_x_start=log_x_start, log_x_t=log_xt, t=t, cond=cond)
            log_true_prob = torch.cat([log_true_prob_[:,:,:self.img_shape],log_x_start[:,:,self.img_shape:]],-1)
        elif cond == 'img_full':
            log_true_prob_ = self.q_posterior(log_x_start=log_x_start, log_x_t=log_xt, t=t, cond=cond)
            log_true_prob = torch.cat([log_x_start[:,:,:self.img_shape],log_true_prob_[:,:,self.img_shape:]],-1)
        else:
            log_true_prob = self.q_posterior(log_x_start=log_x_start, log_x_t=log_xt, t=t, cond=cond)

        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        # modal_mask = torch.tensor([1]*self._denoise_fn.transformer.image_length + [0]*self._denoise_fn.transformer.text_length).cuda()
        # mask_weight = modal_mask * self.mask_weight[0] + (1. - modal_mask) * self.mask_weight[1]
        # kl = kl * mask_weight
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        kl_loss = mask * decoder_nll + (1. - mask) * kl
        

        Lt2 = kl_loss.pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

        # Upweigh loss term of the kl
        # vb_loss = kl_loss / pt + kl_prior
        loss1 = kl_loss / pt 
        vb_loss = loss1
        if self.auxiliary_loss_weight != 0:
            kl_aux = self.multinomial_kl(log_x_start[:,:-1,:], log_x0_recon[:,:-1,:])
            # kl_aux = kl_aux * mask_weight
            kl_aux = sum_except_batch(kl_aux)
            kl_aux_loss = mask * decoder_nll + (1. - mask) * kl_aux

            loss2 = self.auxiliary_loss_weight * kl_aux_loss / pt
            vb_loss = vb_loss + loss2

        return - vb_loss

    def prepare_input(self,x):
        device = self.log_at.device
        image =  x['image']
        caption = x['text']

        if torch.is_tensor(caption):
            txt_tokens = caption
        else:
            txt_tokens = self.text_tokenizer.tokenize(caption)

        if len(image.shape) ==4:
            img_tokens = self.image_tokenizer.get_tokens(image.to(device))
        else:
            img_tokens = image

        txt_tokens = txt_tokens.to(device)

        # print('text token max:', txt_tokens.max())

        txt_tokens = txt_tokens + self.img_classes
        img_tokens = img_tokens.to(device)

        # print('image token max:', img_tokens.max())


        assert img_tokens.shape[0] == txt_tokens.shape[0]
        assert img_tokens.max() <= self.img_classes
        assert txt_tokens.max() <= self.num_classes
        self.shape = (img_tokens.shape[0], img_tokens.shape[1]+txt_tokens.shape[1])
        self.txt_shape = txt_tokens.shape[1]
        self.img_shape = img_tokens.shape[1]

        return img_tokens, txt_tokens

    def forward(self, x, train=False, cond='txt'):
        assert x['image'].shape[0] == len(x['text'])
        img, txt = self.prepare_input(x)
        x = torch.cat([img, txt], dim=1)
        if cond=='txt':
            self.mask_weight = [1,0]
        elif cond == 'img':
            self.mask_weight = [0,1]
        else:
            self.mask_weight = [1,1]
        return self.log_prob(x, train, cond)

    def log_prob(self, x, train=False, cond=None):
        b= x.size(0)
        device = self.log_at.device
        if train==True:
            return self._train_loss(x, cond=cond)

        else:
            x_start = x
            t, pt = self.sample_time(b, device, 'importance')


            log_x_start = index_to_log_onehot(x_start, self.num_classes)
            log_xt = self.q_sample(log_x_start=log_x_start, t=t) # one hot logit
            # xt = log_onehot_to_index(log_xt)


            # log_x0_recon = self.predict_start(log_xt, t=t, cond=cond)            # P_theta(x0|xt)

            # log_x0_recon_cond = self.predict_start(log_xt, t=t, cond=cond)            # P_theta(x0|xt)

            ###UniDM cond as condition
            if cond == 'txt':
                x_start = x_start[:,:self.img_shape]
                log_xt = log_xt[:,:,:self.img_shape]
                cond_prob = log_x_start[:,:,self.img_shape:]
                log_x_start = log_x_start[:,:,:self.img_shape]
                log_x0_recon_cond = self.predict_start(torch.cat([log_xt,cond_prob],dim=-1), t=t, cond=cond)   
                log_x0_recon = log_x0_recon_cond
                # log_x0_recon = torch.cat((log_x0_recon_cond,log_x_start[:,:,self._denoise_fn.transformer.image_length:]),-1)
                log_model_prob = self.q_posterior(log_x_start=log_x0_recon_cond, log_x_t=log_xt, t=t, cond=cond) 
            elif cond == 'img':
                x_start = x_start[:,self.img_shape:]
                log_xt = log_xt[:,:,self.img_shape:]
                cond_prob = log_x_start[:,:,:self.img_shape]
                log_x_start = log_x_start[:,:,self.img_shape:]
                log_x0_recon_cond = self.predict_start(torch.cat([cond_prob,log_xt],dim=-1), t=t, cond=cond)            # P_theta(x0|xt)
                log_x0_recon = log_x0_recon_cond
                # log_x0_recon = torch.cat((log_x_start[:,:,:self._denoise_fn.transformer.image_length],log_x0_recon_cond),-1)
                log_model_prob = self.q_posterior(log_x_start=log_x0_recon_cond, log_x_t=log_xt, t=t,cond=cond) 
            elif cond == 'txt_full':
                log_xt[:,:,self.img_shape:] = log_x_start[:,:,self.img_shape:]
                log_x0_recon_cond = self.predict_start(log_xt, t=t, cond=cond)    
                log_x0_recon = log_x0_recon_cond
                log_x0_recon_cond_txt = torch.cat((log_x0_recon_cond[:,:,:self.img_shape],log_x_start[:,:,self.img_shape:]),-1)
                # log_x0_recon_cond_img = torch.cat((log_x_start[:,:,:self.img_shape],log_x0_recon_cond[:,:,self.img_shape:]),-1)

                log_model_prob_cond_txt = self.q_posterior(log_x_start=log_x0_recon_cond_txt, log_x_t=log_xt, t=t)      # go through q(xt_1_img|xt,x0,xt_1_txt)
                # log_model_prob_cond_img = self.q_posterior(log_x_start=log_x0_recon_cond_img, log_x_t=log_xt, t=t)      # go through q(xt_1_txt|xt,x0,xt_1_img)

                log_model_prob = torch.cat([log_model_prob_cond_txt[:,:,:self.img_shape],log_x_start[:,:,self.img_shape:]],-1)
        
            elif cond == 'img_full':
                log_xt[:,:,:self.img_shape] = log_x_start[:,:,:self.img_shape]
                log_x0_recon_cond = self.predict_start(log_xt, t=t, cond=cond)    
                log_x0_recon = log_x0_recon_cond
                log_x0_recon_cond_img = torch.cat((log_x_start[:,:,:self.img_shape],log_x0_recon_cond[:,:,self.img_shape:]),-1)
                # log_x0_recon_cond_img = torch.cat((log_x_start[:,:,:self.img_shape],log_x0_recon_cond[:,:,self.img_shape:]),-1)

                log_model_prob_cond_img = self.q_posterior(log_x_start=log_x0_recon_cond_img, log_x_t=log_xt, t=t)      # go through q(xt_1_img|xt,x0,xt_1_txt)
                # log_model_prob_cond_img = self.q_posterior(log_x_start=log_x0_recon_cond_img, log_x_t=log_xt, t=t)      # go through q(xt_1_txt|xt,x0,xt_1_img)

                log_model_prob = torch.cat([log_x_start[:,:,:self.img_shape],log_model_prob_cond_img[:,:,self.img_shape:]],-1)


            elif cond is None:
                log_x0_recon_cond = self.predict_start(log_xt, t=t, cond=cond)    
                log_x0_recon = log_x0_recon_cond
                log_x0_recon_cond_txt = torch.cat((log_x0_recon_cond[:,:,:self.img_shape],log_x_start[:,:,self.img_shape:]),-1)
                log_x0_recon_cond_img = torch.cat((log_x_start[:,:,:self.img_shape],log_x0_recon_cond[:,:,self.img_shape:]),-1)

                log_model_prob_cond_txt = self.q_posterior(log_x_start=log_x0_recon_cond_txt, log_x_t=log_xt, t=t)      # go through q(xt_1_img|xt,x0,xt_1_txt)
                log_model_prob_cond_img = self.q_posterior(log_x_start=log_x0_recon_cond_img, log_x_t=log_xt, t=t)      # go through q(xt_1_txt|xt,x0,xt_1_img)

                log_model_prob = torch.cat([log_model_prob_cond_txt[:,:,:self.img_shape],log_model_prob_cond_img[:,:,self.img_shape:]],-1)

                # log_model_prob = self.q_posterior(log_x_start=log_x0_recon, log_x_t=log_xt, t=t)      # go through q(xt_1|xt,x0)

            if cond == 'txt_full':
                log_true_prob_ = self.q_posterior(log_x_start=log_x_start, log_x_t=log_xt, t=t, cond=cond)
                log_true_prob = torch.cat([log_true_prob_[:,:,:self.img_shape],log_x_start[:,:,self.img_shape:]],-1)
            elif cond == 'img_full':
                log_true_prob_ = self.q_posterior(log_x_start=log_x_start, log_x_t=log_xt, t=t, cond=cond)
                log_true_prob = torch.cat([log_x_start[:,:,:self.img_shape],log_true_prob_[:,:,self.img_shape:]],-1)
            else:
                log_true_prob = self.q_posterior(log_x_start=log_x_start, log_x_t=log_xt, t=t, cond=cond)
            kl = self.multinomial_kl(log_true_prob, log_model_prob)
            # mask_region = (xt == self.num_classes-1).float()
            # modal_mask = torch.tensor([1]*self._denoise_fn.transformer.image_length + [0]*self._denoise_fn.transformer.text_length).cuda()
            # mask_weight = modal_mask * self.mask_weight[0] + (1. - modal_mask) * self.mask_weight[1]
            # kl = kl * mask_weight
            kl = sum_except_batch(kl)

            decoder_nll = -log_categorical(log_x_start, log_model_prob)
            decoder_nll = sum_except_batch(decoder_nll)

            mask = (t == torch.zeros_like(t)).float()
            kl_loss = mask * decoder_nll + (1. - mask) * kl

            # kl_prior = self.kl_prior(log_x_start)

            # Upweigh loss term of the kl
            # loss = kl_loss / pt + kl_prior
            loss = kl_loss / pt

            return - loss

    @torch.no_grad()
    def sample(self, num_samples):
        b = num_samples
        device = self.log_at.device
        uniform_logits = torch.zeros((b, self.num_classes) + self.shape, device=device)
        log_z = self.log_sample_categorical(uniform_logits)
        for i in reversed(range(0, self.num_timesteps)):
            print(f'Sample timestep {i:4d}', end='\r')

            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z = self.p_sample(log_z, t)
        print()
        return log_onehot_to_index(log_z)

    @torch.no_grad()
    def sample_chain(self, num_samples):
        b = num_samples
        device = self.log_at.device
        uniform_logits = torch.zeros(
            (b, self.num_classes) + self.shape, device=device)

        zs = torch.zeros((self.num_timesteps, b) + self.shape).long()

        log_z = self.log_sample_categorical(uniform_logits)
        for i in reversed(range(0, self.num_timesteps)):
            print(f'Chain timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z = self.p_sample(log_z, t)

            zs[i] = log_onehot_to_index(log_z)
        print()
        return zs

    @torch.no_grad()
    def sample_chain_mask(self, num_samples):
        b = num_samples
        device = self.log_at.device
        self.shape = (b,self._denoise_fn.transformer.image_length+self._denoise_fn.transformer.text_length)
        zero_logits = torch.zeros((b, self.num_classes-1, self._denoise_fn.transformer.image_length+self._denoise_fn.transformer.text_length),device=device)
        one_logits = torch.ones((b, 1, self._denoise_fn.transformer.image_length+self._denoise_fn.transformer.text_length),device=device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        log_z = torch.log(mask_logits)

        zs = torch.zeros((self.num_timesteps, b) + (self._denoise_fn.transformer.image_length+self._denoise_fn.transformer.text_length,)).long().to(device)

        log_z = self.log_sample_categorical(log_z)

        sample_type="top0.84r"
        # sample_wrap = self.p_sample_with_truncation(self.p_sample,sample_type.split(',')[1])
        self.predict_start = self.predict_start_with_truncation(self.predict_start, sample_type.split(',')[0])
        for i in reversed(range(0, self.num_timesteps)):
            print(f'Chain timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z = self.p_sample(log_z, t, cond=None)

            zs[i] = log_onehot_to_index(log_z)
        print()
        return zs

    @torch.no_grad()
    def sample_mask(self, num_samples):
        b = num_samples
        device = self.log_at.device
        self.shape = (b,self._denoise_fn.transformer.image_length+self._denoise_fn.transformer.text_length)
        zero_logits = torch.zeros((b, self.num_classes-1, self._denoise_fn.transformer.image_length+self._denoise_fn.transformer.text_length),device=device)
        one_logits = torch.ones((b, 1, self._denoise_fn.transformer.image_length+self._denoise_fn.transformer.text_length),device=device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        log_z = torch.log(mask_logits)

        # zs = torch.zeros((self.num_timesteps, b) + (self._denoise_fn.transformer.image_length+self._denoise_fn.transformer.text_length,)).long().to(device)

        log_z = self.log_sample_categorical(log_z)

        sample_type="top0.88r"
        # sample_wrap = self.p_sample_with_truncation(self.p_sample,sample_type.split(',')[1])
        self.predict_start = self.predict_start_with_truncation(self.predict_start, sample_type.split(',')[0])
        from tqdm import tqdm
        for i in tqdm(reversed(range(0, self.num_timesteps)),desc="Chain timestep ",total=self.num_timesteps):
            # print(f'\nChain timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z = self.p_sample(log_z, t)

            # zs[i] = log_onehot_to_index(log_z)

        zs=log_onehot_to_index(log_z)
        return zs
    

    @torch.no_grad()
    def sample_mask_fused(self, num_samples, igtks, tttks, mask_i, mask_t):
        b = num_samples
        device = self.log_at.device
        igtks[igtks==999999] = self.num_classes-1
        tttks[tttks==999999] = self.num_classes-1-self.img_classes # trick. imgclass  is the index of the last token in the text.
        ig_logits = index_to_log_onehot(igtks, self.num_classes)
        tt_logits = index_to_log_onehot(tttks+self.img_classes, self.num_classes)
        masks = torch.cat([mask_i, mask_t], dim=1).unsqueeze(1).bool()
        masks = masks.repeat(b, self.num_classes, 1) # 0 is mask and 1 is unmask or cond
        self.shape = (b,self._denoise_fn.transformer.image_length+self._denoise_fn.transformer.text_length)
        # mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        log_z = torch.cat((ig_logits, tt_logits), dim=2)
        log_z = log_z.repeat(b,1,1)
        log_z_raw = log_z.clone()
        # zs = torch.zeros((self.num_timesteps, b) + (self._denoise_fn.transformer.image_length+self._denoise_fn.transformer.text_length,)).long().to(device)

        log_z = self.log_sample_categorical(log_z)
        

        sample_type="top0.7r"
        self.predict_start = self.predict_start_with_truncation(self.predict_start, sample_type.split(',')[0])
        for i in reversed(range(0, self.num_timesteps)):
            print(f'Chain timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z = self.p_sample(log_z, t)
            log_z[masks] = log_z_raw[masks]
            # zs[i] = log_onehot_to_index(log_z)

        zs=log_onehot_to_index(log_z)
        return zs

    @torch.no_grad()
    def sample_mask_fast(self, num_samples, delta=10):
        b = num_samples
        device = self.log_at.device
        self.shape = (b,self._denoise_fn.transformer.image_length+self._denoise_fn.transformer.text_length)
        zero_logits = torch.zeros((b, self.num_classes-1, self._denoise_fn.transformer.image_length+self._denoise_fn.transformer.text_length),device=device)
        one_logits = torch.ones((b, 1, self._denoise_fn.transformer.image_length+self._denoise_fn.transformer.text_length),device=device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        log_z = torch.log(mask_logits)

        log_z = self.log_sample_categorical(log_z)

        sample_type="top0.85r"

        self.predict_start = self.predict_start_with_truncation(self.predict_start, sample_type.split(',')[0])
        for i in reversed(range(0, self.num_timesteps, delta)):
            print(f'Chain timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)

            log_z = self.p_sample(log_z, t, delta=delta)

            log_z = self.log_sample_categorical(log_z)
        zs=log_onehot_to_index(log_z)
        return zs



    @torch.no_grad()
    def sample_chain_mask_fast(self, num_samples,delta=10):
        b = num_samples
        device = self.log_at.device
        self.shape = (b,self._denoise_fn.transformer.image_length+self._denoise_fn.transformer.text_length)
        zero_logits = torch.zeros((b, self.num_classes-1, self._denoise_fn.transformer.image_length+self._denoise_fn.transformer.text_length),device=device)
        one_logits = torch.ones((b, 1, self._denoise_fn.transformer.image_length+self._denoise_fn.transformer.text_length),device=device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        log_z = torch.log(mask_logits)

        zs = torch.zeros((self.num_timesteps, b) + (self._denoise_fn.transformer.image_length+self._denoise_fn.transformer.text_length,)).long().to(device)

        log_z = self.log_sample_categorical(log_z)

        sample_type="top0.85r"

        self.predict_start = self.predict_start_with_truncation(self.predict_start, sample_type.split(',')[0])
        for i in reversed(range(0, self.num_timesteps, delta)):
            print(f'Chain timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_x_recon = self.predict_start(log_z, t, cond=None)
            if i > delta:
                model_log_prob = self.q_posterior(log_x_start=log_x_recon, log_x_t=log_z, t=t-delta)
            else:
                model_log_prob = self.q_posterior(log_x_start=log_x_recon, log_x_t=log_z, t=t)

            log_z = self.log_sample_categorical(model_log_prob)
            zs[i] = log_onehot_to_index(log_z)
        print()
        return zs

    def sample_chain_mask_cond(self, num_samples, cond):
        b = num_samples
        device = self.log_at.device
        self.shape = (b,self._denoise_fn.transformer.image_length+self._denoise_fn.transformer.text_length)
        self.txt_shape = self._denoise_fn.transformer.text_length
        self.img_shape = self._denoise_fn.transformer.image_length
        zero_logits = torch.zeros((b, self.num_classes-1, self.img_shape),device=device)
        one_logits = torch.ones((b, 1, self.img_shape),device=device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        log_img = torch.log(mask_logits)


        cond_prob = index_to_log_onehot(cond+self.img_classes, self.num_classes)

        cond_prob = self.log_sample_categorical(cond_prob) 
        log_img = self.log_sample_categorical(log_img)

        zs = torch.zeros((self.num_timesteps, b) + (self.img_shape,)).long()

        log_z = log_img
        sample_type="top0.86r"
        self.predict_start = self.predict_start_with_truncation(self.predict_start, sample_type.split(',')[0])

        for i in reversed(range(0, self.num_timesteps)):
            print(f'Chain timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z = self.p_sample(log_z, t, cond='txt', cond_prob=cond_prob)

            zs[i] = log_onehot_to_index(log_z)
            
        print()
        return zs

    def sample_chain_image_cond(self, num_samples, cond):
        b = num_samples
        device = self.log_at.device
        self.shape = (b,self._denoise_fn.transformer.image_length+self._denoise_fn.transformer.text_length)
        self.txt_shape = self._denoise_fn.transformer.text_length
        self.img_shape = self._denoise_fn.transformer.image_length
        zero_logits = torch.zeros((b, self.num_classes-1, self.txt_shape),device=device)
        one_logits = torch.ones((b, 1, self.txt_shape),device=device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        log_txt = torch.log(mask_logits)


        cond_prob = index_to_log_onehot(cond, self.num_classes)

        log_txt = self.log_sample_categorical(log_txt)

        zs = torch.zeros((self.num_timesteps, b) + (self.img_shape+self.txt_shape,)).long()

        log_z = torch.cat((cond_prob, log_txt), dim=2)
        # log_z = log_img
        
        sample_type="top0.6r"
        self.predict_start = self.predict_start_with_truncation(self.predict_start, sample_type.split(',')[0])

        for i in reversed(range(0, self.num_timesteps)):
            print(f'Chain timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z = self.p_sample(log_z, t, cond=None)

            log_z[:,:,:1024] = cond_prob
            zs[i] = log_onehot_to_index(log_z)
            
        print()
        return zs


    def sample_txt_chain_mask_cond(self, num_samples, cond):
        b = num_samples
        device = self.log_at.device
        self.shape = (b,self._denoise_fn.transformer.image_length+self._denoise_fn.transformer.text_length)
        self.txt_shape = self._denoise_fn.transformer.text_length
        self.img_shape = self._denoise_fn.transformer.image_length
        zero_logits = torch.zeros((b, self.num_classes-1, self.txt_shape),device=device)
        one_logits = torch.ones((b, 1, self.txt_shape),device=device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        log_txt = torch.log(mask_logits)


        cond_prob = index_to_log_onehot(cond, self.num_classes)

        log_txt = self.log_sample_categorical(log_txt)

        zs = torch.zeros((self.num_timesteps, b) + (self.img_shape+self.txt_shape,)).long()

        log_z = torch.cat((cond_prob, log_txt), dim=2)
        sample_type="top0.85r"
        self.predict_start = self.predict_start_with_truncation(self.predict_start, sample_type.split(',')[0])

        for i in reversed(range(0, self.num_timesteps)):
            print(f'Chain timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z = self.p_sample(log_z, t, cond='img',cond_start=cond_prob)
            zs[i] = log_onehot_to_index(log_z)
        print()
        return zs


    def p_sample_with_truncation(self, func, sample_type):
        truncation_rate = float(sample_type.replace('q', ''))
        def wrapper(*args, **kwards):
            out = func(*args, **kwards)
            import random
            if random.random() < truncation_rate:
                out = func(out, args[1], args[2], **kwards)
            return out
        return wrapper
    
    def predict_start_with_truncation(self, func, sample_type):

        truncation_r = float(sample_type[:-1].replace('top', ''))
        def wrapper(*args, **kwards):
            out = func(*args, **kwards)
            # notice for different batches, out are same, we do it on out[0]
            temp, indices = torch.sort(out, 1, descending=True) 
            temp1 = torch.exp(temp)
            temp2 = temp1.cumsum(dim=1)
            temp3 = temp2 < truncation_r
            new_temp = torch.full_like(temp3[:,0:1,:], True)
            temp6 = torch.cat((new_temp, temp3), dim=1)
            temp3 = temp6[:,:-1,:]
            temp4 = temp3.gather(1, indices.argsort(1))
            temp5 = temp4.float()*out+(1-temp4.float())*(-70)
            probs = temp5
            return probs
        return wrapper
    
    def ddim_sample(
        self,
        log_x,
        t,
        cond = None,
        eta = 0.
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        assert eta == 0.
 
        log_x_recon = self.predict_start(log_x, t=t, cond=cond)

        log_cumprod_at = extract(self.log_cumprod_at, t, log_x.shape)

        log_cumprod_bt_txt = extract(self.log_cumprod_bt_txt, t, log_x.shape)         # bt~

        log_cumprod_bt_img = extract(self.log_cumprod_bt_img, t, log_x.shape)         # bt~

        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x.shape)   

        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, t, log_x.shape)

        
        log_probs = torch.cat(
            [
                log_add_exp(log_x_recon[:,:self.img_classes,:]+log_cumprod_at, log_cumprod_bt_img),
                log_add_exp(log_x_recon[:,self.img_classes:self.img_classes+self.txt_classes,:]+log_cumprod_at, log_cumprod_bt_txt),
                log_add_exp(log_x_recon[:,-1:,:]+log_1_min_cumprod_ct, log_cumprod_ct), # 
                # torch.log(torch.exp(log_x_start[:,-1:,:]+log_1_min_cumprod_ct) + torch.exp(log_cumprod_ct))
            ],
            dim=1
        )
        out = self.log_sample_categorical(log_probs)
        return out
    
    def ddim_reverse_sample(
        self,
        log_x,
        t,
        stride=1,
        cond = None,

        eta = 0.
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        assert eta == 0.

        log_x_recon = self.predict_start(log_x, t=t, cond=cond)


        log_cumprod_at_next = extract(self.log_cumprod_at, t+stride, log_x.shape)

        log_cumprod_bt_txt_next  = extract(self.log_cumprod_bt_txt, t+stride, log_x.shape)      # bt~

        log_cumprod_bt_img_next  = extract(self.log_cumprod_bt_img, t+stride, log_x.shape)    # bt~

        log_cumprod_ct_next  = extract(self.log_cumprod_ct, t+stride, log_x.shape)   
        log_1_min_cumprod_ct_next = extract(self.log_1_min_cumprod_ct, t+stride, log_x.shape)  


        
        log_probs = torch.cat(
            [
                log_add_exp(log_x_recon[:,:self.img_classes,:]+log_cumprod_at_next, log_cumprod_bt_img_next),
                log_add_exp(log_x_recon[:,self.img_classes:self.img_classes+self.txt_classes,:]+log_cumprod_at_next, log_cumprod_bt_txt_next),
                log_add_exp(log_x_recon[:,-1:,:]+log_1_min_cumprod_ct_next, log_cumprod_ct_next), # 

            ],
            dim=1
        )

        return log_probs
    
    @torch.no_grad()
    def ddim_sample_loop(self, num_samples, stride=10, noise=None):
        b = num_samples
        device = self.log_at.device
        self.shape = (b,self._denoise_fn.transformer.image_length+self._denoise_fn.transformer.text_length)
        zero_logits = torch.zeros((b, self.num_classes-1, self._denoise_fn.transformer.image_length+self._denoise_fn.transformer.text_length),device=device)
        one_logits = torch.ones((b, 1, self._denoise_fn.transformer.image_length+self._denoise_fn.transformer.text_length),device=device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        log_z = torch.log(mask_logits)


        log_z = self.log_sample_categorical(log_z)

        sample_type="top0.85r"

        self.predict_start = self.predict_start_with_truncation(self.predict_start, sample_type.split(',')[0])
        for i in reversed(range(0, self.num_timesteps, stride)):
            print(f'Chain timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)

            log_z = self.ddim_sample(log_z, t, cond=None)

        zs = log_onehot_to_index(log_z)
        print()
        return zs
    
    @torch.no_grad()
    def ddim_reserve_sample_loop(self, image, text, stride=10):
        b = image.shape[0]
        device = self.log_at.device
        image = image.to(device)
        self.shape = (b,self._denoise_fn.transformer.image_length+self._denoise_fn.transformer.text_length)

        zs = torch.zeros((self.num_timesteps, b) + (self._denoise_fn.transformer.image_length+self._denoise_fn.transformer.text_length,)).long().to(device)
        temp_logz = []

        log_z = index_to_log_onehot(torch.cat([image,text],dim=1), self.num_classes)


        for i in range(0, self.num_timesteps, 10):
            print(f'Chain timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z = self.ddim_reverse_sample(log_z, t, cond=None)
            temp_logz.append(log_z.detach().cpu().numpy())
            zs[i] = log_onehot_to_index(log_z)
        print()
        return zs
    

    def ddim_cycle(self, image, text, stride=10):
        b = image.shape[0]
        device = self.log_at.device
        image = image.to(device)
        self.shape = (b,self._denoise_fn.transformer.image_length+self._denoise_fn.transformer.text_length)

        zs_re = torch.zeros((self.num_timesteps, b) + (self._denoise_fn.transformer.image_length+self._denoise_fn.transformer.text_length,)).long().to(device)
        zs_sa = torch.zeros((self.num_timesteps, b) + (self._denoise_fn.transformer.image_length+self._denoise_fn.transformer.text_length,)).long().to(device)

        log_z = index_to_log_onehot(torch.cat([image,text],dim=1), self.num_classes)

        sample_type="top0.85r"
        self.predict_start = self.predict_start_with_truncation(self.predict_start, sample_type.split(',')[0])
        for i in range(0, self.num_timesteps, stride):
            print(f'Chain timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z = self.ddim_reverse_sample(log_z, t, stride, cond=None)

            zs_re[i] = log_onehot_to_index(log_z)
        print()
        for i in reversed(range(0, self.num_timesteps, stride)):
            print(f'Chain timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z = self.ddim_sample(log_z, t, cond=None)

            zs_sa[i] = log_onehot_to_index(log_z)
        return zs_sa