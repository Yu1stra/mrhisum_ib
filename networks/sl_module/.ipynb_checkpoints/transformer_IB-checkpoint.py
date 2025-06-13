import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

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

class Transformer_each_IB(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super(Transformer_each_IB, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))),
                InformationBottleneck(dim, int(dim/4))
            ]))

    def forward(self, x, mask = None):
        all_mu, all_logvar = [], []

        for attn, ff, ib in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
            x, mu, logvar = ib(x)
            all_mu.append(mu)
            all_logvar.append(logvar)
        #計算出depth數KL loss 在座mean
        #return x, kl loss mean
        return x, all_mu, all_logvar

class Transformer_IB(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super(Transformer_IB, self).__init__()
        self.layers = nn.ModuleList([])
        self.ib_layer = InformationBottleneck(dim, dim)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))

    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        #return z, x_reconstructed, mu, std
        z, x_reconstructed, mu, logvar = self.ib_layer(x)

        return z, x_reconstructed, mu, logvar

class Transformer_IB_visual(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super(Transformer_IB_visual, self).__init__()
        self.layers = nn.ModuleList([])
        self.ib_layer = InformationBottleneck(dim, dim)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))

    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        #return z, x_reconstructed, mu, std
        z, x_reconstructed, mu, logvar = self.ib_layer(x)

        return z, x_reconstructed, mu, logvar
if __name__ == "__main__":
    inputs = torch.rand((10, 4096))
    transformer_ib = Transformer_IB(dim=4096, depth=5, heads=8, mlp_dim=8192, dropout=0)
    transformer_each_ib = Transformer_each_IB(dim=4096, depth=5, heads=8, mlp_dim=8192, dropout=0)
    
    inputs_ = inputs.unsqueeze(0)
    outputs_ = transformer_ib(inputs_)
    outputs = outputs_.squeeze(0)
    print(f"transformer_ib: {outputs.shape}")

    inputs_ = inputs.unsqueeze(0)
    outputs_ = transformer_each_ib(inputs_)
    outputs = outputs_.squeeze(0)
    print(f"transformer_each_ib: {outputs.shape}")