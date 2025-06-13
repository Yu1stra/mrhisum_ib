from __future__ import absolute_import
from __future__ import division

import torch
import math
from torch import nn
from torch.nn import functional as F
import sys

class DynamicLinear(nn.Module):
    def __init__(self, out_features):
        super(DynamicLinear, self).__init__()
        self.out_features = out_features

    def forward(self, x):
        in_features = x.shape[-1]
        linear_layer = nn.Linear(in_features, in_features, bias=False).to(x.device)
        return linear_layer(x)
        
class DynamicLinear2(nn.Module):
    def __init__(self, out_features):
        super(DynamicLinear2, self).__init__()
        self.out_features = out_features

    def forward(self, x):
        in_features = x.shape[-1]
        linear_layer = nn.Linear(in_features, self.out_features, bias=False).to(x.device)
        return linear_layer(x)


class CAM(nn.Module):
    def __init__(self):
        super(CAM, self).__init__()
        #self.corr_weights = torch.nn.Parameter(torch.empty(
        #        1024, 1024, requires_grad=True).type(torch.cuda.FloatTensor))

        #self.encoder1 = nn.Linear(128, 512)
        #self.encoder2 = nn.Linear(1024, 512)
        self.encoder1 = nn.Identity()  # 保留原始音訊特徵
        self.encoder2 = nn.Identity()  # 保留原始視覺特徵

        #self.affine_a = nn.Linear(256, 8, bias=False)
        #self.affine_v = nn.Linear(256, 8, bias=False)
        self.affine_a = DynamicLinear(1)
        self.affine_v = DynamicLinear(1)
        #self.W_a = nn.Linear(8, 32, bias=False)
        #self.W_v = nn.Linear(8, 32, bias=False)
        self.W_a = DynamicLinear(32)
        self.W_v = DynamicLinear(32)
        #self.W_ca = nn.Linear(256, 32, bias=False)
        #self.W_cv = nn.Linear(256, 32, bias=False)
        self.W_ca = None  # 動態創建
        self.W_cv = None

        self.W_ha = nn.Linear(128, 128, bias=False)
        self.W_hv = nn.Linear(1024, 1024, bias=False)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.regressor = nn.Sequential(nn.Linear(640, 128),
                                     nn.Dropout(0.6),
                                 nn.Linear(128, 1))

    #def first_init(self):
    #    nn.init.xavier_normal_(self.corr_weights)

    def forward(self, f1_norm, f2_norm):
        #f1 = f1.squeeze(1)
        #f2 = f2.squeeze(1)

        #f1_norm = F.normalize(f1_norm, p=2, dim=2, eps=1e-12)
        #f2_norm = F.normalize(f2_norm, p=2, dim=2, eps=1e-12)

        fin_audio_features = []
        fin_visual_features = []
        sequence_outs = []

        #for i in range(f1_norm.shape[0]):
        audfts = f1_norm#[i,:,:]#.transpose(0,1)
        visfts = f2_norm#[i,:,:]#.transpose(0,1)
        input_shape = f1_norm.shape[0]
        #print(input_shape)
        if self.W_ca is None:
            self.W_ca = nn.Linear(1152, input_shape).to(f1_norm.device)
            self.W_cv = nn.Linear(1152, input_shape).to(f2_norm.device)

        
        aud_fts = self.encoder1(audfts)#frame,128
        vis_fts = self.encoder2(visfts)#frame,128
        #print(f"aud_fts={aud_fts.shape}")
        #print(f"vis_fts={vis_fts.shape}")
        aud_vis_fts = torch.cat((aud_fts, vis_fts), 1)#frame,256
        #print(f"aud_vis_fts={aud_vis_fts.shape}")
        #ca
        #print("ca")
        a_t = self.affine_a(aud_vis_fts.transpose(0,1))#[256,frame]*[frame,8]
        #print(a_t.shape)
        #print(f"aud_fts.transpose(0,1), a_t.transpose(0,1)={aud_fts.transpose(0,1).shape, a_t.transpose(0,1).shape}")
        att_aud = torch.mm(aud_fts.transpose(0,1), a_t.transpose(0,1))#[128,frame]*[8,256]
        audio_att = self.tanh(torch.div(att_aud, math.sqrt(aud_vis_fts.shape[1])))

        aud_vis_fts = torch.cat((aud_fts, vis_fts), 1)#frame,256
        #cv
        #print("cv")
        v_t = self.affine_v(aud_vis_fts.transpose(0,1))
        #print(f"vis_fts.transpose(0,1), v_t.transpose(0,1)={vis_fts.transpose(0,1).shape, v_t.transpose(0,1).shape}")
        att_vis = torch.mm(vis_fts.transpose(0,1), v_t.transpose(0,1))
        vis_att = self.tanh(torch.div(att_vis, math.sqrt(aud_vis_fts.shape[1])))
        #print(f"(aud_fts.transpose(0,1))={aud_fts.transpose(0,1).shape}")
        #print(f"self.W_ca(audio_att) + self.W_a(aud_fts)={self.W_ca(audio_att).shape , self.W_a(aud_fts.transpose(0,1)).shape}")
        #print(f"self.W_ca(audio_att.transpose(0,1)) + self.W_a(aud_fts.transpose(0,1))={self.W_ca(audio_att).shape , self.W_a(aud_fts.transpose(0,1)).shape}")
        #print(audio_att.shape,vis_att.shape)
        
        H_a = self.relu(self.W_ca(audio_att) + self.W_a(aud_fts.transpose(0,1)))
        H_v = self.relu(self.W_cv(vis_att) + self.W_v(vis_fts.transpose(0,1)))
        #print(f"H_a.transpose(0,1) + aud_fts={H_a.transpose(0,1).shape ,aud_fts.shape}")
        att_audio_features = self.W_ha(H_a.transpose(0,1)) + aud_fts
        att_visual_features = self.W_hv(H_v.transpose(0,1)) + vis_fts


        #a1 = torch.matmul(aud_fts.transpose(0,1), self.corr_weights)
        #cc_mat = torch.matmul(a1, vis_fts)

        #audio_att = F.softmax(cc_mat, dim=1)
        #visual_att = F.softmax(cc_mat.transpose(0,1), dim=1)

        #atten_audiofeatures = torch.matmul(aud_fts, audio_att)
        #atten_visualfeatures = torch.matmul(vis_fts, visual_att)

        #added_atten_audiofeatures = aud_fts.add(atten_audiofeatures)
        #added_atten_visualfeatures = vis_fts.add(atten_visualfeatures)

        #### apply tanh on features

        ### apply same dimensions
        #att_audio_features = self.tanh(atten_audiofeatures)
        #att_visual_features = self.tanh(atten_visualfeatures)
        #print(f"att_audio_features, att_visual_features={att_audio_features.shape, att_visual_features.shape}")
        audiovisualfeatures = torch.cat((att_audio_features, att_visual_features), 1)
        #print(audiovisualfeatures.shape)
        #outs = self.regressor(audiovisualfeatures) #.transpose(0,1))
        #seq_outs, _ = torch.max(outs,0)
        #print(seq_outs)
        sequence_outs.append(audiovisualfeatures)
        fin_audio_features.append(att_audio_features)
        fin_visual_features.append(att_visual_features)
        
        final_aud_feat = torch.stack(fin_audio_features)
        final_vis_feat = torch.stack(fin_visual_features)
        final_outs = torch.stack(sequence_outs)
        final_outs=final_outs.squeeze(0)
        #print(final_outs.shape)
        return final_outs #final_aud_feat.transpose(1,2), final_vis_feat.transpose(1,2)