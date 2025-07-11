import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.sl_module.transformer import Transformer, Transformer_with_ib, Transformer_with_ib_post, Transformer_with_ib_after_ff
from networks.sl_module.score_net import ScoreFCN
import math
from torch.nn.parameter import Parameter
import numpy as np
from torch.autograd import Variable

# ------------------------使用multi-modality via information bottleneck 的IB layer 設計---------------------
class InformationBottleneck(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super(InformationBottleneck, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, input_dim),
                                   #  nn.ReLU(inplace=True),
                                    # nn.Linear(1024, 1024),
                                     nn.ReLU(inplace=True) ) 
        self.fc_mu = nn.Linear(input_dim, bottleneck_dim)  # 均值映射
        self.fc_std = nn.Linear(input_dim, bottleneck_dim)  # 方差映射
        self.decoder = nn.Linear(bottleneck_dim, input_dim)  # 还原

    def reparameterize(self, mu, std):
        #变分重参数技巧 (Reparameterization Trick) 
        #logvar = torch.clamp(logvar, min=-10, max=10)
        #std = torch.exp(0.5 * logvar)  # 计算标准差
        eps = torch.randn_like(std)  # 采样标准正态分布
        return mu + eps * std  # 生成随机样本
        
    def encode(self, x):
        #获取瓶颈层的均值和方差 
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = F.softplus(self.fc_std(x) - 5, beta=1)
        return mu, logvar
        
    def forward(self, x):
        mu, std = self.encode(x)
        z = self.reparameterize(mu, std)  # 采样瓶颈特征
        x_reconstructed = self.decoder(z)  # 还原输入
        return z, x_reconstructed, mu, std  # 额外返回 mu 和 logvar 用于计算 KL 损失

def kl_divergence_a(mu, std):
    #计算 KL 散度，使分布逼近标准正态 
    #KL = 0.5 * torch.mean(mu.pow(2) + std.pow(2) - 2 * std.log() - 1)
    #KL = -0.5 * torch.sum(1+std - mu.pow(2) - std.exp(), dim=-1).mean()
    # 标准KL散度（与正态分布）
    KL = 0.5 * torch.mean(mu.pow(2) + std.pow(2) - torch.log(std.pow(2)) - 1)
    return KL
def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

class SL_module_IB_tran(nn.Module): #IB->transformer->score

    def __init__(self, input_dim, depth, heads, mlp_dim, dropout_ratio, bottleneck_dim):
        super(SL_module_IB_tran, self).__init__()

        # 加入 IB 层
        self.ib = InformationBottleneck(input_dim, bottleneck_dim)

        # Transformer 处理瓶颈后的特征
        self.transformer = Transformer(dim=bottleneck_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout_ratio)

        # Score 计算
        self.score_model = ScoreFCN(emb_dim=bottleneck_dim)
        
    def load_state_dict(self, state_dict, strict=True):
        if 'transformer' in state_dict.keys(): 
            self.transformer.load_state_dict(state_dict['transformer'])
            self.score_model.load_state_dict(state_dict['score_model'])
        else:
            super(SL_module_IB_tran, self).load_state_dict(state_dict)

    def forward(self, x, mask):

        # **Transformer 提取特征**\
        z, x_reconstructed, mu, logvar = self.ib(x)
        transformed_emb = self.transformer(z)
        # **信息瓶颈处理**
        
        # **最终评分**
        score = self.score_model(transformed_emb).squeeze(-1)
        score = torch.sigmoid(score)
        kl_loss = kl_divergence(mu, logvar)

        return score, kl_loss  # 返回 KL 损失所需的 mu & logvar


class SL_module_tran_IB(nn.Module): #transformer->IB->score

    def __init__(self, input_dim, depth, heads, mlp_dim, dropout_ratio, bottleneck_dim):
        super(SL_module_tran_IB, self).__init__()

        # 加入 IB 层
        self.ib = InformationBottleneck(input_dim, bottleneck_dim)

        # Transformer 处理瓶颈后的特征
        self.transformer = Transformer(dim=bottleneck_dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
                                       dropout=dropout_ratio)

        # Score 计算
        self.score_model = ScoreFCN(emb_dim=bottleneck_dim)

    def load_state_dict(self, state_dict, strict=True):
        if 'transformer' in state_dict.keys():
            self.transformer.load_state_dict(state_dict['transformer'])
            self.score_model.load_state_dict(state_dict['score_model'])
        else:
            super(SL_module_tran_IB, self).load_state_dict(state_dict)

    def forward(self, x, mask):

        # **Transformer 提取特征**\
        transformed_emb = self.transformer(x)
        # **信息瓶颈处理**
        z, x_reconstructed, mu, logvar = self.ib(transformed_emb)
        # **最终评分**
        score = self.score_model(z).squeeze(-1)
        score = torch.sigmoid(score)
        kl_loss = kl_divergence(mu, logvar)

        return score, kl_loss  # 返回 KL 损失所需的 mu & logvar


class SL_module_CIB(nn.Module): #v->IB->score->v_z, a->IB->score->a_z, a_z+v_z->IB->score(concat)

    def __init__(self, visual_input_dim, audio_input_dim, depth, heads, mlp_dim, dropout_ratio, visual_bottleneck_dim, audio_bottleneck_dim):
        super(SL_module_CIB, self).__init__()

        # 加入 IB 层
        self.ib_v = InformationBottleneck(visual_input_dim, visual_bottleneck_dim)
        self.ib_a = InformationBottleneck(audio_input_dim, audio_bottleneck_dim)
        self.ib_m = InformationBottleneck(visual_bottleneck_dim+audio_bottleneck_dim, visual_bottleneck_dim+audio_bottleneck_dim)
        # Transformer 处理瓶颈后的特征
        self.transformer_v = Transformer(dim=visual_bottleneck_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout_ratio)
        self.transformer_a = Transformer(dim=audio_bottleneck_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout_ratio)
        self.transformer_m = Transformer(dim=visual_bottleneck_dim+audio_bottleneck_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout_ratio)
        # Score 计算
        self.score_model_v = ScoreFCN(emb_dim=visual_bottleneck_dim)
        self.score_model_a = ScoreFCN(emb_dim=audio_bottleneck_dim)
        self.score_model_m = ScoreFCN(emb_dim=visual_bottleneck_dim+audio_bottleneck_dim)

    def load_state_dict(self, state_dict, strict=True):
        if 'transformer' in state_dict.keys(): 
            self.transformer.load_state_dict(state_dict['transformer'])
            self.score_model.load_state_dict(state_dict['score_model'])
        else:
            super(SL_module_CIB, self).load_state_dict(state_dict)

    def forward(self, v, a, mask):
        # **信息瓶颈处理**
        z_v, x_reconstructed_v, mu_v, logvar_v = self.ib_v(v)
        kl_loss_v = kl_divergence_a(mu_v, logvar_v)
        transformed_emb_v = self.transformer_v(z_v)
        score_v = self.score_model_v(transformed_emb_v).squeeze(-1)
        score_v = torch.sigmoid(score_v)

        
        z_a, x_reconstructed_a, mu_a, logvar_a = self.ib_a(a)
        kl_loss_a = kl_divergence_a(mu_a, logvar_a)
        transformed_emb_a = self.transformer_a(z_a)
        score_a = self.score_model_a(transformed_emb_a).squeeze(-1)
        score_a = torch.sigmoid(score_a)

        m=torch.cat([z_v,z_a],dim=-1)
        #print(m.shape)
        z_m, x_reconstructed_m, mu_m, logvar_m = self.ib_m(m)
        kl_loss_m = kl_divergence_a(mu_m, logvar_m)
        transformed_emb_m = self.transformer_m(z_m)
        score_m = self.score_model_m(transformed_emb_m).squeeze(-1)
        score_m = torch.sigmoid(score_m)
        #print(f"v.shape={v.shape}, z_v.shape={z_v.shape}")
        try:
            self.visual_feature = z_v  # shape: (B, D) or (B, T, D)
            self.audio_feature = z_a
            self.fused_feature = z_m
        except:
            print("not success in self.fused_feature")
        kl_loss = kl_loss_v + kl_loss_a + kl_loss_m
        return score_v, score_a, score_m,  kl_loss_v, kl_loss_a, kl_loss_m, kl_loss  # 返回 KL 损失所需的 mu & logvar
        
class SL_module_EIB(nn.Module): #a+v->IB->score(concat)

    def __init__(self, visual_input_dim, audio_input_dim, depth, heads, mlp_dim, dropout_ratio, visual_bottleneck_dim, audio_bottleneck_dim):
        super(SL_module_EIB, self).__init__()

        # 加入 IB 层
        self.ib_m = InformationBottleneck(visual_input_dim+audio_input_dim, visual_bottleneck_dim+audio_bottleneck_dim)
        # Transformer 处理瓶颈后的特征
        self.transformer_m = Transformer(dim=visual_bottleneck_dim+audio_bottleneck_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout_ratio)
        # Score 计算
        self.score_model_m = ScoreFCN(emb_dim=visual_bottleneck_dim+audio_bottleneck_dim)
        
    def load_state_dict(self, state_dict, strict=True):
        if 'transformer' in state_dict.keys(): 
            self.transformer.load_state_dict(state_dict['transformer'])
            self.score_model.load_state_dict(state_dict['score_model'])
        else:
            super(SL_module_EIB, self).load_state_dict(state_dict)

    def forward(self, v, a, mask):
        # **信息瓶颈处理**
        
        m=torch.cat([v, a],dim=-1)
        z_m, x_reconstructed_m, mu_m, logvar_m = self.ib_m(m)
        kl_loss_m = kl_divergence(mu_m, logvar_m)
        transformed_emb_m = self.transformer_m(z_m)
        score_m = self.score_model_m(transformed_emb_m).squeeze(-1)
        score_m = torch.sigmoid(score_m)
        
        return score_m,  kl_loss_m # 返回 KL 损失所需的 mu & logvar
        
class SL_module_LIB(nn.Module):#v->IB->score->v_z, a->IB->score->a_z, a_z+v_z-> score(concat)

    def __init__(self, visual_input_dim, audio_input_dim, depth, heads, mlp_dim, dropout_ratio, visual_bottleneck_dim, audio_bottleneck_dim):
        super(SL_module_LIB, self).__init__()

        # 加入 IB 层
        self.ib_v = InformationBottleneck(visual_input_dim, visual_bottleneck_dim)
        self.ib_a = InformationBottleneck(audio_input_dim, audio_bottleneck_dim)
        # Transformer 处理瓶颈后的特征
        self.transformer_v = Transformer(dim=visual_bottleneck_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout_ratio)
        self.transformer_a = Transformer(dim=audio_bottleneck_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout_ratio)
        #self.transformer_a = Transformer(dim=audio_inputDim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout_ratio)
        self.transformer_m = Transformer(dim=visual_bottleneck_dim+audio_bottleneck_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout_ratio)
        #self.transformer_m = Transformer(dim=visual_bottleneck_dim+audio_inputDim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout_ratio)
        # Score 计算
        self.score_model_v = ScoreFCN(emb_dim=visual_bottleneck_dim)
        self.score_model_a = ScoreFCN(emb_dim=audio_bottleneck_dim)
        #self.score_model_a = ScoreFCN(emb_dim=audio_inputDim)
        self.score_model_m = ScoreFCN(emb_dim=visual_bottleneck_dim+audio_bottleneck_dim)
        #self.score_model_m = ScoreFCN(emb_dim=visual_bottleneck_dim+audio_inputDim)

    def load_state_dict(self, state_dict, strict=True):
        if 'transformer' in state_dict.keys(): 
            self.transformer.load_state_dict(state_dict['transformer'])
            self.score_model.load_state_dict(state_dict['score_model'])
        else:
            super(SL_module_LIB, self).load_state_dict(state_dict)

    def forward(self, v, a, mask):
        # **信息瓶颈处理**
        z_v, x_reconstructed_v, mu_v, logvar_v = self.ib_v(v)
        kl_loss_v = kl_divergence(mu_v, logvar_v)
        transformed_emb_v = self.transformer_v(z_v)
        score_v = self.score_model_v(transformed_emb_v).squeeze(-1)
        score_v = torch.sigmoid(score_v)

        
        z_a, x_reconstructed_a, mu_a, logvar_a = self.ib_a(a)
        kl_loss_a = kl_divergence(mu_a, logvar_a)
        transformed_emb_a = self.transformer_a(z_a)
        #transformed_emb_a = self.transformer_a(a)
        score_a = self.score_model_a(transformed_emb_a).squeeze(-1)
        score_a = torch.sigmoid(score_a)

        z_m=torch.cat([z_v,z_a],dim=-1)
        #z_m=torch.cat([z_v,a],dim=-1)
        transformed_emb_m = self.transformer_m(z_m)
        score_m = self.score_model_m(transformed_emb_m).squeeze(-1)
        score_m = torch.sigmoid(score_m)
        
        return score_v, score_a, score_m,  kl_loss_v, kl_loss_a, kl_loss_m, kl_loss  # 返回 KL 损失所需的 mu & logvar

#baseline-----------
class SL_module(nn.Module):

    def __init__(self, input_dim, depth, heads, mlp_dim, dropout_ratio):
        super(SL_module, self).__init__()
        
        self.transformer = Transformer(dim=input_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout_ratio)
        self.score_model = ScoreFCN(emb_dim=input_dim) #audio
        #self.score_model = ScoreFCN(emb_dim=1024) #visual
        #self.score_model = ScoreFCN(emb_dim=1152) #multi2
        #self.score_model = ScoreFCN(emb_dim=1152)  #crossattention
        
    def forward(self, x, mask):

        transformed_emb = self.transformer(x)
        score = self.score_model(transformed_emb).squeeze(-1)
        #print(transformed_emb)
        score = torch.sigmoid(score)

        return score, 0.0

    def load_state_dict(self, state_dict, strict=True):
        if 'transformer' in state_dict.keys(): 
            self.transformer.load_state_dict(state_dict['transformer'])
            self.score_model.load_state_dict(state_dict['score_model'])
        else:
            super(SL_module, self).load_state_dict(state_dict)

#-----------------------------沒有加入IB layer---------------------
class SL_module_visual(nn.Module):

    def __init__(self, input_dim, depth, heads, mlp_dim, dropout_ratio):
        super(SL_module_visual, self).__init__()
        
        self.transformer = Transformer(dim=input_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout_ratio)
        #self.score_model = ScoreFCN(emb_dim=128) #audio
        #self.score_model = ScoreFCN(emb_dim=1024) #visual
        #self.score_model = ScoreFCN(emb_dim=1152) #multi2
        self.score_model = ScoreFCN(emb_dim=1024)  #crossattention
        
    def forward(self, x, mask):

        transformed_emb = self.transformer(x)
        score = self.score_model(transformed_emb).squeeze(-1)
        #print(transformed_emb)
        score = torch.sigmoid(score)

        return score, 0.0

    def load_state_dict(self, state_dict, strict=True):
        if 'transformer' in state_dict.keys(): 
            self.transformer.load_state_dict(state_dict['transformer'])
            self.score_model.load_state_dict(state_dict['score_model'])
        else:
            super(SL_module_visual, self).load_state_dict(state_dict)
class SL_module_audio(nn.Module):

    def __init__(self, input_dim, depth, heads, mlp_dim, dropout_ratio):
        super(SL_module_audio, self).__init__()
        
        self.transformer = Transformer(dim=input_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout_ratio)
        #self.score_model = ScoreFCN(emb_dim=128) #audio
        #self.score_model = ScoreFCN(emb_dim=1024) #visual
        #self.score_model = ScoreFCN(emb_dim=1152) #multi2
        self.score_model = ScoreFCN(emb_dim=128)  #crossattention
        
    def forward(self, x, mask):

        transformed_emb = self.transformer(x)
        score = self.score_model(transformed_emb).squeeze(-1)
        #print(transformed_emb)
        score = torch.sigmoid(score)

        return score, 0.0

    def load_state_dict(self, state_dict, strict=True):
        if 'transformer' in state_dict.keys(): 
            self.transformer.load_state_dict(state_dict['transformer'])
            self.score_model.load_state_dict(state_dict['score_model'])
        else:
            super(SL_module_audio, self).load_state_dict(state_dict)
class SL_module_multi(nn.Module):

    def __init__(self, input_dim, depth, heads, mlp_dim, dropout_ratio):
        super(SL_module_multi, self).__init__()
        
        self.transformer = Transformer(dim=input_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout_ratio)
        #self.score_model = ScoreFCN(emb_dim=128) #audio
        #self.score_model = ScoreFCN(emb_dim=1024) #visual
        self.score_model = ScoreFCN(emb_dim=1152) #multi2
        #self.score_model = ScoreFCN(emb_dim=1024)  #crossattention
        
    def forward(self, x, mask):

        transformed_emb = self.transformer(x)
        score = self.score_model(transformed_emb).squeeze(-1)
        #print(transformed_emb)
        score = torch.sigmoid(score)


        return score, 0.0

    def load_state_dict(self, state_dict, strict=True):
        if 'transformer' in state_dict.keys(): 
            self.transformer.load_state_dict(state_dict['transformer'])
            self.score_model.load_state_dict(state_dict['score_model'])
        else:
            super(SL_module_multi, self).load_state_dict(state_dict)


#------------------------------使用VIB相同的IB layer設計---------------------------
#VIB type-----------------------------------------------------------------------------------------------
# def reparameterize_VIB(mu, logvar, batch_size, cuda=False, sampling=True):
#     # output dim: batch_size * dim
#     if sampling:
#         std = logvar.mul(0.5).exp_()
#         eps = torch.FloatTensor(batch_size, std.size(0)).to(mu).normal_()
#         eps = Variable(eps)
#         return mu.view(1, -1) + eps * std.view(1, -1)
#         #return mu + eps * std
#     else:
#         return mu.view(1, -1)
#         #return mu
def reparameterize_VIB(mu, logD, batch_size=1, cuda=True, sampling=True):
    if sampling:
        std = torch.exp(0.5 * logD)
        # 确保eps和返回值维度正确
        eps = torch.FloatTensor(batch_size, std.size(0)).to(mu.device).normal_()
        # 确保mu被正确扩展到batch_size
        mu_expanded = mu.expand(batch_size, -1)
        return mu_expanded + eps * std.expand(batch_size, -1)
    else:
        return mu.expand(batch_size, -1)
class InformationBottleneck_VIB(nn.Module):
    def __init__(self, dim, mask_thresh=0, init_mag=9, init_var=0.01,
                kl_mult=1, divide_w=False, sample_in_training=True, sample_in_testing=False, masking=False):
        super(InformationBottleneck_VIB, self).__init__()
        self.prior_z_logD = Parameter(torch.Tensor(dim))
        self.post_z_mu = Parameter(torch.Tensor(dim))
        self.post_z_logD = Parameter(torch.Tensor(dim))

        self.epsilon = 1e-8
        self.dim = dim
        self.sample_in_training = sample_in_training
        self.sample_in_testing = sample_in_testing
        # if masking=True, apply mask directly
        self.masking = masking

        # initialization
        stdv = 1. / math.sqrt(dim)
        self.post_z_mu.data.normal_(1, init_var)
        self.prior_z_logD.data.normal_(-init_mag, init_var)
        self.post_z_logD.data.normal_(-init_mag, init_var)

        self.need_update_z = True # flag for updating z during testing
        self.mask_thresh = mask_thresh
        self.kl_mult=kl_mult
        self.divide_w=divide_w


    def adapt_shape(self, src_shape, x_shape):
        # to distinguish conv layers and fc layers
        # see if we need to expand the dimension of x
        new_shape = src_shape if len(src_shape)==2 else (1, src_shape[0])
        if len(x_shape)>2:
            new_shape = list(new_shape)
            new_shape += [1 for i in range(len(x_shape)-2)]
        return new_shape

    def get_logalpha(self):
        return self.post_z_logD.data - torch.log(self.post_z_mu.data.pow(2) + self.epsilon)

    def get_dp(self):
        logalpha = self.get_logalpha()
        alpha = torch.exp(logalpha)
        return alpha / (1+alpha)

    def get_mask_hard(self, threshold=0):
        logalpha = self.get_logalpha()
        hard_mask = (logalpha < threshold).float()
        return hard_mask

    def get_mask_weighted(self, threshold=0):
        logalpha = self.get_logalpha()
        mask = (logalpha < threshold).float()*self.post_z_mu.data
        return mask

    # """def forward(self, x):
    #     # 4 modes: sampling, hard mask, weighted mask, use mean value
    #     if self.masking:
    #         mask = self.get_mask_hard(self.mask_thresh)
    #         new_shape = self.adapt_shape(mask.size(), x.size())
    #         return x * Variable(mask.view(new_shape))

    #     bsize = x.size(0)
    #     if (self.training and self.sample_in_training) or (not self.training and self.sample_in_testing):
    #         z_scale = reparameterize_VIB(self.post_z_mu, self.post_z_logD, bsize, cuda=True, sampling=True)
    #         if not self.training:
    #             z_scale *= Variable(self.get_mask_hard(self.mask_thresh))
    #     else:
    #         z_scale = Variable(self.get_mask_weighted(self.mask_thresh))
    #     self.kld = self.kl_closed_form(x)
    #     # 检查是否处理3D序列数据
    #     print("x.shape=",x.shape)
    #     print("z_scale.shape=",z_scale.shape)
    #     if len(x.size()) == 3 and len(z_scale.size()) == 2:
    #         # 对于序列数据，将z_scale形状从[batch, feature]变为[batch, 1, feature]
    #         # 这样可以在序列维度上进行广播
    #         z_scale = z_scale.unsqueeze(1)
    #         print("x.shape=",x.shape)
    #         print("z_scale.shape=",z_scale.shape)
    #         return x * z_scale
    #     else:
    #         # 原始处理方式
    #         new_shape = self.adapt_shape(z_scale.size(), x.size())
    #         print("x.shape=",x.shape)
    #         print("z_scale.view(new_shape).shape=",z_scale.view(new_shape).shape)
    #         return x * z_scale.view(new_shape)"""
    def forward(self, x):
        # 4 modes: sampling, hard mask, weighted mask, use mean value
        if self.masking:
            mask = self.get_mask_hard(self.mask_thresh)
            new_shape = self.adapt_shape(mask.size(), x.size())
            return x * Variable(mask.view(new_shape))

        bsize = x.size(0)
        # 确保z_scale有批次维度
        if (self.training and self.sample_in_training) or (not self.training and self.sample_in_testing):
            # 修改reparameterize_VIB确保返回[batch, features]形状的张量
            z_scale = reparameterize_VIB(self.post_z_mu, self.post_z_logD, bsize, cuda=True, sampling=True)
            if not self.training:
                z_scale *= Variable(self.get_mask_hard(self.mask_thresh))
        else:
            # 确保加权掩码也有批次维度
            z_scale = Variable(self.get_mask_weighted(self.mask_thresh)).expand(bsize, -1)
        
        self.kld = self.kl_closed_form(x)
        # 适当处理3D输入
        if len(x.size()) == 3:
            # 将z_scale从[batch, features]变为[batch, 1, features]以便序列广播
            z_scale = z_scale.unsqueeze(1)
            return x * z_scale
        else:
            return x * z_scale 

    # def kl_closed_form(self, x):
    #     new_shape = self.adapt_shape(self.post_z_mu.size(), x.size())
    #     h_D = torch.exp(self.post_z_logD.view(new_shape))
    #     h_mu = self.post_z_mu.view(new_shape)

    #     KLD = torch.sum(torch.log(1 + h_mu.pow(2)/(h_D + self.epsilon) )) * x.size(2) / h_D.size(2)

    #     if x.dim() > 2:
    #         if self.divide_w:
    #             # divide it by the width
    #             KLD *= x.size()[2]
    #         else:
    #             KLD *= np.prod(x.size()[2:])
    #     return KLD * 0.5 * self.kl_mult
    def kl_closed_form(self, x):
        # 使用模型的变分参数，而非输入x的统计特性
        mu = self.post_z_mu
        logvar = self.post_z_logD
        
        # 计算标准KL散度
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 根据需要应用缩放因子
        return kl_div * self.kl_mult
    

class SL_module_VIB(nn.Module):
    def __init__(self, input_dim, depth, heads, mlp_dim, dropout_ratio):
        super(SL_module_VIB, self).__init__()

        # 加入 IB 层
        self.ib = InformationBottleneck_VIB(input_dim)

        # Transformer 处理瓶颈后的特征
        self.transformer = Transformer(dim=input_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout_ratio)

        # Score 计算
        self.score_model = ScoreFCN(emb_dim=input_dim)
        
    def load_state_dict(self, state_dict, strict=True):
        if 'transformer' in state_dict.keys(): 
            self.transformer.load_state_dict(state_dict['transformer'])
            self.score_model.load_state_dict(state_dict['score_model'])
        else:
            super(SL_module_VIB, self).load_state_dict(state_dict)

    def forward(self, x, mask):

        # **Transformer 提取特征**\
        z= self.ib(x)
        transformed_emb = self.transformer(z)
        # **信息瓶颈处理**
        
        # **最终评分**
        score = self.score_model(transformed_emb).squeeze(-1)
        score = torch.sigmoid(score)
        kl_loss = self.ib.kl_closed_form(x)

        return score, kl_loss  # 返回 KL 损失所需的 mu & logvar

class SL_module_VIB_in_transformer(nn.Module):
    def __init__(self, input_dim, depth, heads, mlp_dim, dropout_ratio):
        super(SL_module_VIB_in_transformer, self).__init__()
        
        self.transformer = Transformer_with_ib(dim=input_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout_ratio)
        self.score_model = ScoreFCN(emb_dim=input_dim)

    def load_state_dict(self, state_dict, strict=True):
        if 'transformer' in state_dict.keys(): 
            self.transformer.load_state_dict(state_dict['transformer'])
            self.score_model.load_state_dict(state_dict['score_model'])
        else:
            super(SL_module_VIB_in_transformer, self).load_state_dict(state_dict)

    def forward(self, x, mask):

        # **Transformer 提取特征**\
        transformed_emb, kl_loss = self.transformer(x)
        # **信息瓶颈处理**
        
        # **最终评分**
        score = self.score_model(transformed_emb).squeeze(-1)
        score = torch.sigmoid(score)

        return score, kl_loss  # 返回 KL 损失所需的 mu & logvar

class SL_module_VIB_in_transformer_post(nn.Module):
    def __init__(self, input_dim, depth, heads, mlp_dim, dropout_ratio):
        super(SL_module_VIB_in_transformer_post, self).__init__()
        
        self.transformer = Transformer_with_ib_post(dim=input_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout_ratio)
        self.score_model = ScoreFCN(emb_dim=input_dim)

    def load_state_dict(self, state_dict, strict=True):
        if 'transformer' in state_dict.keys(): 
            self.transformer.load_state_dict(state_dict['transformer'])
            self.score_model.load_state_dict(state_dict['score_model'])
        else:
            super(SL_module_VIB_in_transformer_post, self).load_state_dict(state_dict)

    def forward(self, x, mask):

        # **Transformer 提取特征**\
        transformed_emb, kl_loss = self.transformer(x)
        # **信息瓶颈处理**

        # **最终评分**
        score = self.score_model(transformed_emb).squeeze(-1)
        score = torch.sigmoid(score)

        return score, kl_loss  # 返回 KL 损失所需的 mu & logvar


class SL_module_VIB_in_transformer_after_ff(nn.Module):
    def __init__(self, input_dim, depth, heads, mlp_dim, dropout_ratio):
        super(SL_module_VIB_in_transformer_after_ff, self).__init__()

        self.transformer = Transformer_with_ib_after_ff(dim=input_dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
                                                    dropout=dropout_ratio)
        self.score_model = ScoreFCN(emb_dim=input_dim)

    def load_state_dict(self, state_dict, strict=True):
        if 'transformer' in state_dict.keys():
            self.transformer.load_state_dict(state_dict['transformer'])
            self.score_model.load_state_dict(state_dict['score_model'])
        else:
            super(SL_module_VIB_in_transformer_after_ff, self).load_state_dict(state_dict)

    def forward(self, x, mask):

        # **Transformer 提取特征**\
        transformed_emb, kl_loss = self.transformer(x)
        # **信息瓶颈处理**

        # **最终评分**
        score = self.score_model(transformed_emb).squeeze(-1)
        score = torch.sigmoid(score)

        return score, kl_loss  # 返回 KL 损失所需的 mu & logvar

class SL_module_VIB_postib(nn.Module):
    def __init__(self, input_dim, depth, heads, mlp_dim, dropout_ratio):
        super(SL_module_VIB_postib, self).__init__()

        # Transformer 处理瓶颈后的特征
        self.transformer = Transformer(dim=input_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout_ratio)

        self.ib = InformationBottleneck_VIB(input_dim)

        # Score 计算
        self.score_model = ScoreFCN(emb_dim=input_dim)
        
    def load_state_dict(self, state_dict, strict=True):
        if 'transformer' in state_dict.keys(): 
            self.transformer.load_state_dict(state_dict['transformer'])
            self.score_model.load_state_dict(state_dict['score_model'])
        else:
            super(SL_module_VIB_postib, self).load_state_dict(state_dict)

    def forward(self, x, mask):

        # **Transformer 提取特征**\
        
        transformed_emb = self.transformer(x)

        z= self.ib(transformed_emb)
        # **信息瓶颈处理**
        
        # **最终评分**
        score = self.score_model(z).squeeze(-1)
        score = torch.sigmoid(score)
        kl_loss = self.ib.kl_closed_form(x)

        return score, kl_loss  # 返回 KL 损失所需的 mu & logvar


#--------------------------使用wasserstein loss取代IB中kl loss--------------------------------
class TimeAwareCost(nn.Module):
    """时间感知成本函数：量化序列间的几何差异"""
    def __init__(self, w1=0.4, w2=0.4, w3=0.2):
        super().__init__()
        self.w = [w1, w2, w3]

    def soft_dtw(self, x, y, gamma=0.1):
        """改进版软DTW实现，确保正值输出（支持自动微分）"""
        n, m = x.shape[0], y.shape[0]

        x_norm = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
        y_norm = y / (torch.norm(y, dim=1, keepdim=True) + 1e-8)

        if n == 1 and m == 1:
            D = torch.norm(x_norm - y_norm, p=2).unsqueeze(0).unsqueeze(0)
        elif n == 1:
            D = torch.norm(x_norm.unsqueeze(0) - y_norm, p=2, dim=1).unsqueeze(0)
        elif m == 1:
            D = torch.norm(x_norm - y_norm.unsqueeze(0), p=2, dim=1).unsqueeze(1)
        else:
            D = torch.cdist(x_norm, y_norm, p=2)

        D = D / (D.max() + 1e-8)

        R = torch.zeros(n+1, m+1, device=x.device) + float('inf')
        R[0, 0] = 0

        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = D[i-1, j-1]
                min_val = torch.min(torch.min(R[i-1, j], R[i, j-1]), R[i-1, j-1])
                R[i, j] = cost + min_val

        return torch.abs(R[-1, -1])

    def autocorr(self, z):
        """计算序列的自相关函数（ACF）"""
        z_centered = z - torch.mean(z, dim=0)
        acf = torch.stack([
            torch.sum(z_centered[:-lag] * z_centered[lag:], dim=0)
            for lag in range(1, 6)
        ])
        return acf / torch.max(torch.abs(acf), dim=0)[0]

    def forward(self, x_i, x_j, z_i, z_j):
        """计算两个序列对的Wasserstein传输成本"""
        is_same_sample = torch.all(x_i == x_j) and torch.all(z_i == z_j)
        if is_same_sample:
            return torch.tensor(0.0, device=x_i.device)

        d_time = self.soft_dtw(x_i, x_j)
        d_latent = torch.norm(z_i - z_j, p=2) / z_i.shape[0]
        acf_i = self.autocorr(z_i)
        acf_j = self.autocorr(z_j)
        d_acf = torch.norm(acf_i - acf_j, p=1)

        return (self.w[0] * d_time +
                self.w[1] * d_latent +
                self.w[2] * d_acf)

def sinkhorn_cost(X, Z, cost_fn, epsilon=0.1, n_iters=50):
    """Sinkhorn 算法计算 Wasserstein 距离"""
    batch_size = X.shape[0]
    device = X.device

    C = torch.zeros(batch_size, batch_size, device=device)
    for i in range(batch_size):
        for j in range(batch_size):
            C[i, j] = cost_fn(X[i], X[j], Z[i], Z[j])

    u = torch.zeros(batch_size, device=device)
    v = torch.zeros(batch_size, device=device)

    K = torch.exp(-C / epsilon)
    for _ in range(n_iters):
        u = epsilon * (torch.log(torch.ones(batch_size, device=device) / batch_size) -
                      torch.log(torch.sum(K * v.exp().unsqueeze(0), dim=1)))
        v = epsilon * (torch.log(torch.ones(batch_size, device=device) / batch_size) -
                      torch.log(torch.sum(K * u.exp().unsqueeze(1), dim=0)))

    P = torch.exp((u.unsqueeze(1) + v.unsqueeze(0) - C) / epsilon)
    W = torch.sum(P * C)

    return W

class SL_module_SWD(nn.Module):
    def __init__(self, input_dim, depth, heads, mlp_dim, dropout_ratio):
        super(SL_module_SWD, self).__init__()

        self.transformer = Transformer(dim=input_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout_ratio)
        self.score_model = ScoreFCN(emb_dim=input_dim)

        # 编码器网络
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(mlp_dim, input_dim)
        )

    def compute_swd(self, x, z, num_projections=50):
        """计算 Sliced Wasserstein Distance"""
        # 获取维度信息
        batch_size, seq_len, dim = x.shape
        device = x.device

        # 生成随机投影向量
        theta = torch.randn(num_projections, dim, device=device)
        theta = theta / torch.norm(theta, dim=1, keepdim=True)

        # 将输入展平为 2D 张量以便投影
        x_flat = x.reshape(-1, dim)  # (batch_size * seq_len, dim)
        z_flat = z.reshape(-1, dim)  # (batch_size * seq_len, dim)

        # 计算投影
        x_proj = torch.mm(x_flat, theta.t())  # (batch_size * seq_len, num_projections)
        z_proj = torch.mm(z_flat, theta.t())  # (batch_size * seq_len, num_projections)

        # 分别为每个批次计算
        total_distance = 0
        for i in range(batch_size):
            # 获取当前批次的投影
            x_proj_batch = x_proj[i * seq_len:(i + 1) * seq_len]  # (seq_len, num_projections)
            z_proj_batch = z_proj[i * seq_len:(i + 1) * seq_len]  # (seq_len, num_projections)

            # 在序列维度上排序
            x_sort = torch.sort(x_proj_batch, dim=0)[0]  # (seq_len, num_projections)
            z_sort = torch.sort(z_proj_batch, dim=0)[0]  # (seq_len, num_projections)

            # 计算每个批次的距离
            batch_distance = torch.mean((x_sort - z_sort) ** 2)
            total_distance += batch_distance

        # 计算平均距离
        wasserstein_dist = total_distance / batch_size

        return wasserstein_dist

    def forward(self, x, mask=None):
        # 获取原始特征的编码
        z = self.encoder(x)

        # 使用 transformer 处理编码后的特征
        transformed_emb = self.transformer(z)

        # 计算最终评分
        score = self.score_model(transformed_emb).squeeze(-1)
        score = torch.sigmoid(score)

        # 计算优化后的 Sliced Wasserstein Distance 作为正则化项
        wasserstein_loss = self.compute_swd(x, z)

        return score, wasserstein_loss

    def load_state_dict(self, state_dict, strict=True):
        if 'transformer' in state_dict.keys():
            self.transformer.load_state_dict(state_dict['transformer'])
            self.score_model.load_state_dict(state_dict['score_model'])
        else:
            super(SL_module_SWD, self).load_state_dict(state_dict)

class InformationBottleneck_VIB_SWD(nn.Module):
    def __init__(self, dim, mask_thresh=0, init_mag=9, init_var=0.01,
                w_mult=1, divide_w=False, sample_in_training=True, sample_in_testing=False, masking=False):
        super(InformationBottleneck_VIB_SWD, self).__init__()
        self.post_z_mu = Parameter(torch.Tensor(dim))
        self.post_z_logD = Parameter(torch.Tensor(dim))

        self.epsilon = 1e-8
        self.dim = dim
        self.sample_in_training = sample_in_training
        self.sample_in_testing = sample_in_testing
        self.masking = masking

        # initialization
        stdv = 1. / math.sqrt(dim)
        self.post_z_mu.data.normal_(1, init_var)
        self.post_z_logD.data.normal_(-init_mag, init_var)

        self.need_update_z = True
        self.mask_thresh = mask_thresh
        self.w_mult = w_mult
        self.divide_w = divide_w
        self.num_projections = 50  # 随机投影数量

        # 初始化 w_loss
        self.register_buffer('w_loss', torch.tensor(0.0))

    def compute_wasserstein_distance(self, x, z):
        """计算序列数据的 Sliced Wasserstein Distance"""
        # 获取维度信息
        batch_size, seq_len, dim = x.shape
        device = x.device

        # 生成随机投影向量
        theta = torch.randn(self.num_projections, dim, device=device)
        theta = theta / torch.norm(theta, dim=1, keepdim=True)

        # 重塑输入以进行批量矩阵乘法
        x_flat = x.reshape(batch_size * seq_len, dim)
        z_flat = z.reshape(batch_size * seq_len, dim)

        # 计算投影
        x_proj = torch.mm(x_flat, theta.t())  # (batch_size * seq_len, num_projections)
        z_proj = torch.mm(z_flat, theta.t())  # (batch_size * seq_len, num_projections)

        # 分别为每个批次计算
        total_distance = 0
        for i in range(batch_size):
            # 获取当前批次的投影
            x_proj_batch = x_proj[i * seq_len:(i + 1) * seq_len]  # (seq_len, num_projections)
            z_proj_batch = z_proj[i * seq_len:(i + 1) * seq_len]  # (seq_len, num_projections)

            # 在序列维度上排序
            x_sort = torch.sort(x_proj_batch, dim=0)[0]  # (seq_len, num_projections)
            z_sort = torch.sort(z_proj_batch, dim=0)[0]  # (seq_len, num_projections)

            # 计算每个批次的距离
            batch_distance = torch.mean((x_sort - z_sort) ** 2)
            total_distance += batch_distance

        # 计算平均距离
        wasserstein_dist = total_distance / batch_size

        return wasserstein_dist * self.w_mult

    def get_logalpha(self):
        return self.post_z_logD.data - torch.log(self.post_z_mu.data.pow(2) + self.epsilon)

    def get_dp(self):
        logalpha = self.get_logalpha()
        alpha = torch.exp(logalpha)
        return alpha / (1+alpha)

    def get_mask_hard(self, threshold=0):
        logalpha = self.get_logalpha()
        hard_mask = (logalpha < threshold).float()
        return hard_mask

    def get_mask_weighted(self, threshold=0):
        logalpha = self.get_logalpha()
        mask = (logalpha < threshold).float() * self.post_z_mu.data
        return mask

    def forward(self, x):
        # 处理掩码
        if self.masking:
            mask = self.get_mask_hard(self.mask_thresh)
            if len(x.shape) > 2:  # 序列数据
                mask = mask.unsqueeze(1)  # 添加序列维度
            return x * Variable(mask)

        bsize = x.size(0)
        # 生成编码
        if (self.training and self.sample_in_training) or (not self.training and self.sample_in_testing):
            std = torch.exp(0.5 * self.post_z_logD)
            eps = torch.randn(bsize, *std.shape, device=x.device)
            z_scale = self.post_z_mu.expand(bsize, -1) + eps * std.expand(bsize, -1)
            if not self.training:
                z_scale *= Variable(self.get_mask_hard(self.mask_thresh))
        else:
            z_scale = Variable(self.get_mask_weighted(self.mask_thresh)).expand(bsize, -1)

        # 将 z_scale 扩展到与输入相同的序列长度
        if len(x.size()) == 3:
            z_scale_expanded = z_scale.unsqueeze(1).expand(-1, x.size(1), -1)
            # 计算 Wasserstein loss
            self.w_loss = self.compute_wasserstein_distance(x, z_scale_expanded)
            return x * z_scale_expanded
        else:
            # 计算 Wasserstein loss
            self.w_loss = self.compute_wasserstein_distance(x.unsqueeze(1), z_scale.unsqueeze(1))
            return x * z_scale

    def get_loss(self):
        """获取最近计算的 Wasserstein loss"""
        return self.w_loss


class SL_module_VIBPRE_SWD(nn.Module):  #--------IB->transformer->score---------------
    def __init__(self, input_dim, depth, heads, mlp_dim, dropout_ratio):
        super(SL_module_VIBPRE_SWD, self).__init__()

        # 加入 IB 层
        self.ib = InformationBottleneck_VIB_SWD(input_dim)

        # Transformer 处理瓶颈后的特征
        self.transformer = Transformer(dim=input_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout_ratio)

        # Score 计算
        self.score_model = ScoreFCN(emb_dim=input_dim)

    def load_state_dict(self, state_dict, strict=True):
        if 'transformer' in state_dict.keys():
            self.transformer.load_state_dict(state_dict['transformer'])
            self.score_model.load_state_dict(state_dict['score_model'])
        else:
            super(SL_module_VIBPRE_SWD, self).load_state_dict(state_dict)
    def forward(self, x, mask=None):
        # 获取原始特征的编码
        z = self.ib(x)

        # 使用 transformer 处理编码后的特征
        transformed_emb = self.transformer(z)

        # 计算最终评分
        score = self.score_model(transformed_emb).squeeze(-1)
        score = torch.sigmoid(score)

        WL_loss = self.ib.get_loss()

        return score, WL_loss  # 返回 WL 损失所需的 mu & logvar

class SL_module_VIBPOST_SWD(nn.Module):  #--------transformer->IB->score---------------
    def __init__(self, input_dim, depth, heads, mlp_dim, dropout_ratio):
        super(SL_module_VIBPOST_SWD, self).__init__()

        # 加入 IB 层
        self.ib = InformationBottleneck_VIB_SWD(input_dim)

        # Transformer 处理瓶颈后的特征
        self.transformer = Transformer(dim=input_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout_ratio)

        # Score 计算
        self.score_model = ScoreFCN(emb_dim=input_dim)

    def load_state_dict(self, state_dict, strict=True):
        if 'transformer' in state_dict.keys():
            self.transformer.load_state_dict(state_dict['transformer'])
            self.score_model.load_state_dict(state_dict['score_model'])
        else:
            super(SL_module_VIBPRE_SWD, self).load_state_dict(state_dict)

    def forward(self, x, mask=None):
        # 获取原始特征的编码
        # 使用 transformer 处理编码后的特征
        transformed_emb = self.transformer(x)
        z = self.ib(transformed_emb)

        # 计算最终评分
        score = self.score_model(z).squeeze(-1)
        score = torch.sigmoid(score)

        WL_loss = self.ib.get_loss()

        return score, WL_loss  # 返回 WL 损失所需的 mu & logvar