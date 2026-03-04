import torch

# where用法: 根据条件选择x或y
print("where用法:")
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
y = torch.tensor([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])

# 创建条件张量：判断x中每个元素是否大于3，大于为True，小于等于为False
condition = x > 3
# 使用torch.where根据condition选择x或y，相当于numpy的where函数
# condition为True的位置，取x；为False的位置，取y
result = torch.where(condition, x, y)
print(result)

#torch.arange用法: 创建一个从0到n-1的等差数列
t = torch.arange(0, 10, 2)
# 0, 2, 4, 6, 8
print(t)


#torch.outer用法: 计算两个向量的外积
print("outer用法:")
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = torch.outer(a, b)
# [[4, 5, 6], [8, 10, 12], [12, 15, 18]]
# 外积: 两个向量a和b的外积是一个矩阵，矩阵的每个元素是a的每个元素与b的每个元素的乘积
print(c)


#torch.cat用法: 拼接两个张量
print("cat用法:")
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = torch.cat((a, b))
# [1, 2, 3, 4, 5, 6]
# 拼接: 将两个张量a和b按指定维度拼接在一起
print(c)
t1 = torch.tensor([[1, 2, 3], [4, 5, 6]], [7, 8, 9], [10, 11, 12])
t2 = torch.tensor([[13, 14, 15], [16, 17, 18], [19, 20, 21]])
c = torch.cat((t1, t2), dim=0)
# [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18], [19, 20, 21]]
# 拼接: 将两个张量t1和t2按指定维度拼接在一起
print(c)

#unsqueeze用法: 在指定维度上增加一个维度
print("unsqueeze用法:")
a = torch.tensor([1, 2, 3])
b = a.unsqueeze(0)
# [[1, 2, 3]]
# 增加一个维度: 在指定维度上增加一个维度
print(b)
print(a.shape)
print(b.shape)



