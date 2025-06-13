import torch
import torch.nn as nn
import torch.nn.functional as F

class graph_fusion(nn.Module):
    def __init__(self, in_size, output_dim, hidden=50, dropout=0.5):
        super(graph_fusion, self).__init__()
        self.audio_proj = nn.Linear(128, in_size)
        self.norm = nn.BatchNorm1d(in_size)
        self.norm2 = nn.BatchNorm1d(in_size * 2)
        self.drop = nn.Dropout(p=dropout)

        # 定義 graph fusion 層 (用於學習 Audio-Visual 交互)
        self.graph_fusion = nn.Sequential(
            nn.Linear(in_size * 2, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, in_size),
            nn.Tanh()
        )

        self.attention = nn.Linear(in_size, 1)
        self.linear_1 = nn.Linear(in_size * 2, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)
        self.linear_3 = nn.Linear(hidden, output_dim)

        self.hidden = hidden
        self.in_size = in_size

    def forward(self, a1, v1):
        # ---------------------------
        # **Unimodal Layers (單模態處理)**
        # ---------------------------
        a1 = self.audio_proj(a1)
        sa = torch.tanh(self.attention(a1))  # Audio 注意力
        sv = torch.tanh(self.attention(v1))  # Visual 注意力
        #unimodal_a = sa.expand(a1.size(0), self.in_size) * a1
        #unimodal_v = sv.expand(v1.size(0), self.in_size) * v1
        unimodal_a = sa.expand_as(a1) * a1
        unimodal_v = sv.expand_as(v1) * v1
        unimodal = (unimodal_a + unimodal_v) / 2  # 平均視覺 & 音訊特徵

        # ---------------------------
        # **Bimodal Fusion (雙模態融合)**
        # ---------------------------
        a = F.softmax(a1, dim=1)
        v = F.softmax(v1, dim=1)

        # 計算音訊-視覺交互注意力
        sav = (1 / (torch.matmul(a.unsqueeze(-1), v.unsqueeze(-2)).squeeze() + 0.5)) * (sa + sv)

        a_v = torch.tanh(
            (sav.expand_as(a1)) * self.graph_fusion(torch.cat([a1, v1], dim=-1))
        )

        bimodal = a_v  # 只有 Audio-Visual 互動，沒有三模態 (trimodal)

        # ---------------------------
        # **最終融合**
        # ---------------------------
        fusion = torch.cat([unimodal, bimodal], dim=1)  # 結合單模態 + 雙模態資訊
        
        #y_1 = torch.tanh(self.linear_1(fusion))
        #y_1 = torch.tanh(self.linear_2(y_1))
        #y_2 = torch.tanh(self.linear_3(y_1))  # 最終輸出
        print(fusion.shape)
        return fusion
