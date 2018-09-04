import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

x1 = torch.randn(8, 10)
x2 = torch.randn(6, 10)
y = torch.empty(8, 6).bernoulli_().mul_(2).sub_(1)
l = torch.nn.CosineEmbeddingLoss(.2)

print(l(
    x1.unsqueeze(2).expand(8,10,6),
    x2.transpose(0,1).unsqueeze(0).expand(8,10,6),
    y))


a = np.random.random((8,10))
b = np.random.random((6,10))
print(cosine_similarity(a,b))

a = torch.Tensor(a)
a = a.unsqueeze(2).expand(8,10,6)
b = torch.Tensor(b)
b = b.transpose(0,1).unsqueeze(0).expand(8,10,6)
print(torch.nn.functional.cosine_similarity(a,b))