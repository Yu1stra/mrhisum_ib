import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.sl_module.transformer import Transformer
from networks.sl_module.score_net import ScoreFCN
from networks.sl_module.transformer_IB import *
from networks.sl_module.score_net import ScoreFCN

# class SL_module(nn.Module):

#     def __init__(self, input_dim, depth, heads, mlp_dim, dropout_ratio):
#         super(SL_module, self).__init__()
        
#         self.transformer = Transformer(dim=input_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout_ratio)
#         self.score_model = ScoreFCN(emb_dim=input_dim) #audio
#         #self.score_model = ScoreFCN(emb_dim=1024) #visual
#         #self.score_model = ScoreFCN(emb_dim=1152) #multi2
#         #self.score_model = ScoreFCN(emb_dim=1152)  #crossattention
        
#     def forward(self, x, mask):

#         transformed_emb = self.transformer(x)
#         score = self.score_model(transformed_emb).squeeze(-1)
#         #print(transformed_emb)
#         score = torch.sigmoid(score)

#         return score, 0.0

#     def load_state_dict(self, state_dict, strict=True):
#         if 'transformer' in state_dict.keys(): 
#             self.transformer.load_state_dict(state_dict['transformer'])
#             self.score_model.load_state_dict(state_dict['score_model'])
#         else:
#             super(SL_module, self).load_state_dict(state_dict)     
# #baseline-----------
class SL_module_eachib(nn.Module):

    def __init__(self, input_dim, depth, heads, mlp_dim, dropout_ratio):
        super(SL_module_eachib, self).__init__()
        
        self.transformer = Transformer_each_IB(dim=input_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout_ratio)
        self.score_model = ScoreFCN(emb_dim=input_dim) #audio
        #self.score_model = ScoreFCN(emb_dim=1024) #visual
        #self.score_model = ScoreFCN(emb_dim=1152) #multi2
        #self.score_model = ScoreFCN(emb_dim=1152)  #crossattention
        
    def forward(self, x, mask):
        #return x, all_mu, all_logvar
        transformed_emb, mu, logvar = self.transformer(x)
        score = self.score_model(transformed_emb).squeeze(-1)
        #print(transformed_emb)
        score = torch.sigmoid(score)
        kl_loss = kl_divergence(mu, logvar)

        return score, kl_loss

    def load_state_dict(self, state_dict, strict=True):
        if 'transformer' in state_dict.keys(): 
            self.transformer.load_state_dict(state_dict['transformer'])
            self.score_model.load_state_dict(state_dict['score_model'])
        else:
            super(SL_module_eachib, self).load_state_dict(state_dict)

class SL_module_ib(nn.Module):

    def __init__(self, input_dim, depth, heads, mlp_dim, dropout_ratio):
        super(SL_module_ib, self).__init__()
        
        self.transformer = Transformer_IB(dim=input_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout_ratio)
        self.score_model = ScoreFCN(emb_dim=input_dim) #audio
        #self.score_model = ScoreFCN(emb_dim=1024) #visual
        #self.score_model = ScoreFCN(emb_dim=1152) #multi2
        #self.score_model = ScoreFCN(emb_dim=1152)  #crossattention
        
    def forward(self, x, mask):

        transformed_emb, _, mu, logvar = self.transformer(x)
        score = self.score_model(transformed_emb).squeeze(-1)
        #print(transformed_emb)
        score = torch.sigmoid(score)
        kl_loss = kl_divergence(mu, logvar)
        
        return score, kl_loss

    def load_state_dict(self, state_dict, strict=True):
        if 'transformer' in state_dict.keys(): 
            self.transformer.load_state_dict(state_dict['transformer'])
            self.score_model.load_state_dict(state_dict['score_model'])
        else:
            super(SL_module_ib, self).load_state_dict(state_dict)

class SL_module_ibv(nn.Module):

    def __init__(self, input_dim, depth, heads, mlp_dim, dropout_ratio):
        super(SL_module_ibv, self).__init__()
        
        self.transformer = Transformer_IB_visual(dim=input_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout_ratio)
        self.score_model = ScoreFCN(emb_dim=int(input_dim)) #audio
        #self.score_model = ScoreFCN(emb_dim=1024) #visual
        #self.score_model = ScoreFCN(emb_dim=1152) #multi2
        #self.score_model = ScoreFCN(emb_dim=1152)  #crossattention
        
    def forward(self, x, mask):

        transformed_emb, _, mu, logvar = self.transformer(x)
        score = self.score_model(transformed_emb).squeeze(-1)
        #print(transformed_emb)
        score = torch.sigmoid(score)
        kl_loss = kl_divergence(mu, logvar)
        
        return score, kl_loss

    def load_state_dict(self, state_dict, strict=True):
        if 'transformer' in state_dict.keys(): 
            self.transformer.load_state_dict(state_dict['transformer'])
            self.score_model.load_state_dict(state_dict['score_model'])
        else:
            super(SL_module_ibv, self).load_state_dict(state_dict)


