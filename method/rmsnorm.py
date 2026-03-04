import torch

# RMSNorm
# -> torch.Tensor 表示返回的是tensor类型
# weight: 权重 
"""归一化会把输入向量的 “尺度” 压缩到均方根为 1 的范围，虽然稳定了训练，但会丢失原始数据的尺度信息（比如某些维度本应更重要，数值更大）。乘以可学习的权重 γ，
可以让模型自主调整每个维度的尺度，恢复甚至优化原始的表达能力。"""
def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # 对最后一维做 RMS 归一化
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    # sqrt 平方根
    # mean 均值, 求平方和
    x_norm = x / rms
    return x_norm * weight


if __name__ == "__main__":
    x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    weight = torch.ones(3)
    y = rmsnorm(x, weight)

    print("input:", x)
    print("output:", y)