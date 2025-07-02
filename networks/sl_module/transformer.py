import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import math

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

class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PreNormWithIB(nn.Module):
    def __init__(self, dim, fn):
        super(PreNormWithIB, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.ib = InformationBottleneck_VIB(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        # 先应用归一化
        normalized = self.norm(x)
        # 然后应用IB层
        ib_output = self.ib(normalized)
        # 最后应用函数（Attention或FeedForward）
        self.kld = self.ib.kld  # 存储KL散度以便在外部访问
        return self.fn(ib_output, **kwargs)

class PreNormWithPostIB(nn.Module):
    def __init__(self, dim, fn):
        super(PreNormWithPostIB, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        self.ib = InformationBottleneck_VIB(dim)

    def forward(self, x, **kwargs):
        # 先应用归一化
        normalized = self.norm(x)
        # 然后应用函数（Attention或FeedForward）
        fn_output = self.fn(normalized, **kwargs)
        # 最后应用IB层
        ib_output = self.ib(fn_output)
        self.kld = self.ib.kld  # 存储KL散度以便在外部访问
        return ib_output

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super(Attention, self).__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        #print(f"x.shape: {x.shape}")
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = h)
        
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        #print(x.shape)
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask
        
        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))

    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)

        return x

class Transformer_with_ib(nn.Module): #LayerNorm → IB → Attention → LayerNorm → IB → FeedForward xN
    def __init__(self, dim, depth, heads, mlp_dim, dropout, bottleneck_dim=None):
        super(Transformer_with_ib, self).__init__()
        # 如果没有指定瓶颈维度，则使用与输入相同的维度
        if bottleneck_dim is None:
            bottleneck_dim = dim
            
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # 使用PreNormWithIB代替PreNorm，在norm层后立即应用IB层
                Residual(PreNormWithIB(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNormWithIB(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        kl_losses = []
        
        for attn, ff in self.layers:
            # 应用注意力层（包含norm和IB）
            x = attn(x, mask=mask)
            kl_losses.append(attn.fn.kld)  # 获取注意力块中的KL损失
            
            # 应用前馈网络（包含norm和IB）
            x = ff(x)
            kl_losses.append(ff.fn.kld)  # 获取前馈块中的KL损失
            
        return x, sum(kl_losses)  # 返回输出和累计的KL损失


class Transformer_with_ib_post(nn.Module): #LayerNorm → Attention → IB → LayerNorm → FeedForward → IB xN
    def __init__(self, dim, depth, heads, mlp_dim, dropout, bottleneck_dim=None):
        super(Transformer_with_ib_post, self).__init__()
        # 如果没有指定瓶颈维度，则使用与输入相同的维度
        if bottleneck_dim is None:
            bottleneck_dim = dim
            
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # 使用PreNormWithPostIB代替PreNormWithIB，在Attention之后应用IB层
                Residual(PreNormWithPostIB(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNormWithPostIB(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        kl_losses = []
        
        for attn, ff in self.layers:
            # 应用注意力层和IB
            x = attn(x, mask=mask)
            kl_losses.append(attn.fn.kld)  # 获取注意力块中的KL损失
            
            # 应用前馈网络和IB
            x = ff(x)
            kl_losses.append(ff.fn.kld)  # 获取前馈块中的KL损失
            
        return x, sum(kl_losses)  # 返回输出和累计的KL损失

class Transformer_with_ib_after_ff(nn.Module): #LayerNorm → Attention → LayerNorm → FeedForward → IB xN 
    def __init__(self, dim, depth, heads, mlp_dim, dropout, bottleneck_dim=None):
        super(Transformer_with_ib_after_ff, self).__init__()
        # 如果没有指定瓶颈维度，则使用与输入相同的维度
        if bottleneck_dim is None:
            bottleneck_dim = dim
            
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            # 每个块包含: LayerNorm→Attention→LayerNorm→FeedForward→IB
            self.layers.append(nn.ModuleList([
                # 标准的LayerNorm→Attention
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                # 标准的LayerNorm→FeedForward
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))),
                # IB层
                InformationBottleneck_VIB(dim)
            ]))

    def forward(self, x, mask=None):
        kl_losses = []
        
        for attn, ff_norm, ib in self.layers:
            # 应用注意力层（包含归一化和残差连接）
            x = attn(x, mask=mask)
            
            # 应用前馈网络（包含归一化）
            ff_out = ff_norm(x)
            
            # 应用IB层
            ib_out = ib(ff_out)
            kl_losses.append(ib.kld)
            
            # 添加残差连接
            x = x + ib_out
            
        return x, sum(kl_losses)  # 返回输出和累计的KL损失

if __name__ == "__main__":
    inputs = torch.rand((10, 4096))
    transformer = Transformer(dim=4096, depth=5, heads=8, mlp_dim=8192, dropout=0)

    inputs_ = inputs.unsqueeze(0)
    outputs_ = transformer(inputs_)
    outputs = outputs_.squeeze(0)
    print(outputs.shape)