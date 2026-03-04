import torch
import torch.nn as nn

# dropout layer
print("dropout layer")
dropout_layer = nn.Dropout(p=0.5)
t1 = torch.tensor([1, 2, 3, 4, 5])
t2 = dropout_layer(t1)
print(t2)
# 这里dropout_layer会随机将50%的元素置为0。 同时为了保持期望不变，会将其余元素乘以1/(1-p)
# 举一个例子： 假设第一个和第三个元素置为0， 其他元素需要乘2， 那么t2 = [0, 4, 0, 8, 10]

#linear layer
# 线性层会进行一个线性变换： y = Wx + b
# 其中W是权重矩阵， b是偏置向量， x是输入向量， y是输出向量
# 权重矩阵的形状是(out_features, in_features)， 偏置向量的形状是(out_features)
# 输入向量的形状是(in_features)， 输出向量的形状是(out_features)
# 权重矩阵的形状是(out_features, in_features)， 偏置向量的形状是(out_features)
print("linear layer")
linear_layer = nn.Linear(in_features=3, out_features=5, bias=True)
t1 = torch.tensor([1, 2, 3]) #shape: (3)
t2 = torch.tensor([[4, 5, 6]]) #shape: (1, 3)
print(linear_layer.weight)       # W，shape: (5, 3)
print(linear_layer.bias)         # b，shape: (5,)
# 这里会随机初始化权重矩阵和偏置向量
output = linear_layer(t1) #shape: (5)
print(output)

# view
# view会改变张量的形状， 但是不会改变张量的数据
print("view")
t4 = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]) #[2, 5]
t4_view = t4.view(5, 2) #[5, 2]
print(t4_view)
# 这里t4_view的形状是(5, 2)， 但是数据还是[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# 所以， view不会改变张量的数据， 只是改变张量的形状




