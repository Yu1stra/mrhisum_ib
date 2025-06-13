import numpy as np
from collections import namedtuple
from scipy import stats
import torch
import math
try:
    from model.entropy_estimators import mi as KNN_MI
except:
    from entropy_estimators import mi as KNN_MI
from tqdm import tqdm
device = torch.device("cuda:1")

# Index of test example
n = 6
# Number of runs
n_runs = 1
psi_mean_all_run = []
psi_max_all_run = []

def sample_from_sphere(d):
    vec = torch.randn( (d, 1) )
    vec /= torch.norm(vec, dim=0)
    return vec

def knn_mi_gpu_exact(x, y, k=3, base=2, device=None):
    """精确复制KNN_MI函数的GPU版本"""
    if device is None:
        device = x.device
    
    # 模拟CPU实现的数据预处理
    if len(x.shape) > 2:
        x = x.view(x.shape[0], -1)
    if len(y.shape) > 2:
        y = y.view(y.shape[0], -1)
    
    # 转换为双精度以匹配NumPy精度
    x = x.double()
    y = y.double()
    
    # 添加随机噪声（与原始函数相同）
    def add_noise_torch(x, noise_level=1e-10):
        return x + noise_level * torch.randn_like(x)
    
    x = add_noise_torch(x)
    y = add_noise_torch(y)
    
    # 合并数据点
    points = torch.cat([x, y], dim=1)
    
    # 最近邻计算 - 使用max-norm (p=inf)
    # 注意：PyTorch的cdist支持p=float('inf')
    p_dist = torch.cdist(points, points, p=float('inf'))
    
    # 将自己到自己的距离设为大值
    diag_mask = torch.eye(len(x), device=device, dtype=torch.bool)
    p_dist.masked_fill_(diag_mask, float('inf'))
    
    # 获取前k个最近邻
    _, indices = torch.topk(p_dist, k=k, dim=1, largest=False)
    
    # 计算kth近邻距离向量
    dvec = torch.gather(p_dist, 1, indices[:, k-1].unsqueeze(1)).squeeze()
    
    # 实现avgdigamma函数
    def avgdigamma(data, dvec):
        """计算平均digamma值"""
        # 计算每一点在其它特征空间中的近邻数
        n_samples = data.shape[0]
        dd = torch.zeros(n_samples, device=device)
        
        for i in range(n_samples):
            # 计算max-norm距离
            if data.shape[1] == 1:
                # 一维数据使用绝对值差
                dist = torch.abs(data - data[i].unsqueeze(0))
            else:
                # 多维使用max-norm
                dist = torch.max(torch.abs(data - data[i].unsqueeze(0)), dim=1)[0]
            
            # 计算小于等于dvec[i]的距离数
            dd[i] = torch.sum(dist <= dvec[i])
        
        # 应用digamma函数
        return torch.mean(torch.polygamma(0, dd))
    
    # 计算各部分
    a = avgdigamma(x, dvec)
    b = avgdigamma(y, dvec)
    c = torch.polygamma(0, torch.tensor(k, dtype=torch.double, device=device))
    d = torch.polygamma(0, torch.tensor(len(x), dtype=torch.double, device=device))
    
    # 计算最终结果
    result = (-a - b + c + d) / math.log(base)
    
    return result.item()

def sliceMI(X, Y, M = 1000, n = None, DX = False, DY = False, method = "KNN", ifYLabel = False, info = False):
    '''
    X : (batchSize, dims) 
    Y : label
    n ：sample num
    DX：if X is discrete
    DY：if Y is discrete

    根據論文：Using Sliced Mutual Information to Study Memorization and Generalization in Deep Neural Networks
    降維之後我們使用 NPEET 提供的方法，NPEET 是基於 KNN 的方法，並且設定 k = 3
    '''
    if len(X.shape) > 2:
        X = X.view(X.shape[0],-1)

    if ifYLabel and len(Y.shape) > 1:
        Y = torch.argmax(Y, dim=1)
    elif len(Y.shape) > 2:
        Y = Y.view(Y.shape[0],-1)
    if n != None:
        X = X[:n]
        Y = Y[:n]
    SI_Ms = []
    if info:
        iterator = tqdm(range(M))
    else:
        iterator = range(M)
    X = X.to(device)
    Y = Y.to(device)
    for m in iterator:
        if not DX:
            theta = sample_from_sphere(X.shape[1]).to(device)
            thetaX = torch.mm(X, theta)
        else:
            thetaX = X
        if not DY:
            theta = sample_from_sphere(Y.shape[1]).to(device)
            thetaY = torch.mm(Y, theta)
        else:
            thetaY = Y
        if method == "KNN":
            SI_Ms.append(KNN_MI(thetaX, thetaY, k=3))
        if method == "KNN_gpu":
            SI_Ms.append(knn_mi_gpu_exact(thetaX, thetaY, k=3, device=device))

    return sum(SI_Ms) / len(SI_Ms)



def labelGatch(Y):
    if len(Y.shape) == 1:
        Y = Y.view(-1, 1)
    if Y.shape[1] != 1:
        labelList = [i for i in range(Y.shape[1])]
        label_indices = [torch.nonzero(Y[:, i]).view(-1) for i in range(len(labelList))]
        pass
    else:
        labelList = np.unique(Y[:, 0])
        label_indices = [torch.nonzero(Y[:, 0] == i).squeeze() for i in range(len(labelList))]
    return labelList, label_indices

def psi(X, Y):
    '''
    X : (batchSize, dim) 還沒寫 , (batchSize, otherDim(channel), targetDim..)
    Y : label
    '''
    labelList, label_indices = labelGatch(Y)
    for fiber in range(X.shape[2]):
        for m in range(100):
            theta = sample_from_sphere(X.shape[1])
            thetaX = torch.mm(X[:,:,fiber], theta)
            for i in range(len(labelList)):
                thetaX_class = thetaX[label_indices[i]].view(-1)
                mu, std = stats.norm.fit(thetaX_class)

                pass


if __name__ == "__main__":
    def generate_random_vector(N):
        random_indices = [np.random.randint(0, 10) for _ in range(N)]
        random_vectors = torch.zeros( (N, 10) )

        for i in range(N):
            random_vectors[i][random_indices[i]] = 1


        return random_vectors, torch.tensor(random_indices)
    
    X = torch.randn( (48000, 1, 28, 28) )
    Y, random_indices = generate_random_vector(48000)
    print("method is use cpu")
    print(sliceMI(X, Y, M=1000, n = 10000, method = "KNN",info = True))
    print("#--------------------------------------------------------------")
    print("method is use gpu")
    print(sliceMI(X, Y, M=1000, n = 10000, method="KNN_gpu",info = True))