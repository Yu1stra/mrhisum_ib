import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.sl_module.transformer import Transformer
from networks.sl_module.score_net import ScoreFCN
from networks.sl_module.transformer import Transformer
from networks.sl_module.score_net import ScoreFCN
#ib+KL-----------------------------------------------------
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

def kl_divergence(mu, std):
    #计算 KL 散度，使分布逼近标准正态 
    KL = 0.5 * torch.mean(mu.pow(2) + std.pow(2) - 2 * std.log() - 1)
    #KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - std.exp(), dim=-1).mean()
    return KL
class SL_module_IB(nn.Module):

    def __init__(self, input_dim, depth, heads, mlp_dim, dropout_ratio, bottleneck_dim):
        super(SL_module_IB, self).__init__()

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
            super(SL_module_IB, self).load_state_dict(state_dict)

    def forward(self, x, mask):
        # **信息瓶颈处理**
        z, x_reconstructed, mu, logvar = self.ib(x)

        # **Transformer 提取特征**
        transformed_emb = self.transformer(z)

        # **最终评分**
        score = self.score_model(transformed_emb).squeeze(-1)
        score = torch.sigmoid(score)
        kl_loss = kl_divergence(mu, logvar)

        return score, kl_loss  # 返回 KL 损失所需的 mu & logvar
        
class SL_module_CIB(nn.Module):

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
        kl_loss_v = kl_divergence(mu_v, logvar_v)
        transformed_emb_v = self.transformer_v(z_v)
        score_v = self.score_model_v(transformed_emb_v).squeeze(-1)
        score_v = torch.sigmoid(score_v)

        
        z_a, x_reconstructed_a, mu_a, logvar_a = self.ib_a(a)
        kl_loss_a = kl_divergence(mu_a, logvar_a)
        transformed_emb_a = self.transformer_a(z_a)
        score_a = self.score_model_a(transformed_emb_a).squeeze(-1)
        score_a = torch.sigmoid(score_a)

        m=torch.cat([z_v,z_a],dim=-1)
        #print(m.shape)
        z_m, x_reconstructed_m, mu_m, logvar_m = self.ib_m(m)
        kl_loss_m = kl_divergence(mu_m, logvar_m)
        transformed_emb_m = self.transformer_m(z_m)
        score_m = self.score_model_m(transformed_emb_m).squeeze(-1)
        score_m = torch.sigmoid(score_m)
        
        return score_v, score_a, score_m,  kl_loss_v+kl_loss_a+kl_loss_m  # 返回 KL 损失所需的 mu & logvar
        
class SL_module_EIB(nn.Module):

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
        
class SL_module_LIB(nn.Module):

    def __init__(self, visual_input_dim, audio_input_dim, depth, heads, mlp_dim, dropout_ratio, visual_bottleneck_dim, audio_bottleneck_dim):
        super(SL_module_LIB, self).__init__()

        # 加入 IB 层
        self.ib_v = InformationBottleneck(visual_input_dim, visual_bottleneck_dim)
        self.ib_a = InformationBottleneck(audio_input_dim, audio_bottleneck_dim)
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
        score_a = self.score_model_a(transformed_emb_a).squeeze(-1)
        score_a = torch.sigmoid(score_a)

        z_m=torch.cat([z_v,z_a],dim=-1)
        transformed_emb_m = self.transformer_m(z_m)
        score_m = self.score_model_m(transformed_emb_m).squeeze(-1)
        score_m = torch.sigmoid(score_m)
        
        return score_v, score_a, score_m,  kl_loss_v+kl_loss_a # 返回 KL 损失所需的 mu & logvar
        
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