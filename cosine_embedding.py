import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

batchsize = 512
feature_dim = 32
center_num = 7
margin = .2
torch.manual_seed(1)
np.random.seed(1)
x1 = torch.randn(batchsize, feature_dim)
x2 = torch.randn(center_num, feature_dim)
y = np.random.randint(0, center_num, (batchsize))
print(y)
label = np.ones((batchsize, center_num), dtype=np.int32)*-1
label[np.arange(batchsize), y] = 1
label = torch.Tensor(label)
print(label)
l = torch.nn.CosineEmbeddingLoss(margin)

print(l(
    x1.unsqueeze(2).expand(batchsize,feature_dim,center_num),
    x2.transpose(0,1).unsqueeze(0).expand(batchsize,feature_dim,center_num),
    label))


a = np.random.random((8,10))
b = np.random.random((6,10))
print(cosine_similarity(a,b))

a = torch.Tensor(a)
a = a.unsqueeze(2).expand(8,10,6)
b = torch.Tensor(b)
b = b.transpose(0,1).unsqueeze(0).expand(8,10,6)
print(torch.nn.functional.cosine_similarity(a,b))