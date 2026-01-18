from transformers import PretrainedConfig


class YzMiniMindConfig(PretrainedConfig):
    model_type = "yz_minimind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_base: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        
        ############ MoE ############
        use_moe:bool=False,
        num_experts_per_tok:int=2,
        n_routed_experts:int=4,
        n_shared_experts:int=1,
        scoring_func:str='softmax',
        aux_loss_alpha:float=0.1,
        seq_aux:bool=True,
        norm_topk_prob:bool=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_base = rope_base
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe=use_moe
        self.num_experts_per_tok=num_experts_per_tok
        self.n_routed_experts=n_routed_experts
        self.n_shared_experts=n_shared_experts
        self.seq_aux=seq_aux
        self.norm_topk_prob=norm_topk_prob
        self.aux_loss_alpha=aux_loss_alpha
        self.scoring_func=scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 4,
                "beta_slow": 1,
                "factor": 4,
                "original_max_position_embeddings": 2048,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


import torch
from torch import nn

# 继承nn.Module
# __init__
# norm
# forward
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # 创建张量：torch.ones(hidden_size) 创建一个形状为 (hidden_size,) 的全1张量
        # 包装为参数：nn.Parameter() 将这个张量包装成一个可学习参数
        # 注册到模型：当这个参数被赋值给 self.weight 时，PyTorch 会自动将其注册为模型的参数

    def _norm(self, x):
        # return x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        # 等同于下方的计算
        return x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        # 计算每个位置的归一化因子：torch.mean(x**2, dim=-1, keepdim=True) 计算每个位置的平方和，然后取平方根
        # 添加小常数 self.eps 防止除零错误
        # rsqrt 计算平方根的倒数，避免了额外的平方根操作
        # keepdim=True 保持维度，确保输出形状与输入相同
        # dim=-1 对最后一个维度进行操作


    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)
        # type_as(x) 确保输出与输入的 dtype 相同


# 写出最初rope的实现
def precompute_rope_freqs_cis(
    head_dim: int,
    max_seq_len: int = 32768, # 序列最大长度
    rope_base: float = 1e6,  # rope的theta参数, 可以理解为rope_theta
    rope_scaling: Optional[dict] = None, # rope缩放参数, yarn需要
    # device: torch.device = None,
):  
    '''
    计算rope频率
    Args:
        head_dim: 头维度
        max_seq_len: 最大序列长度
        rope_base: rope的theta参数, 可以理解为rope_theta
        rope_scaling: rope缩放参数, yarn需要
    Returns:
        freq_cos: cosine频率
        freq_sin: sine频率
        shape为(max_seq_len, head_dim)
    '''
    # 计算rope频率
    freqs = 1.0 / (rope_base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)) # 计算rope频率, 单位时间内转动角度
    # torch.arange(start, end, step, dtype)  这个函数创建一个等差数列，从0开始，到head_dim-1结束，步长为2

    if rope_scaling is not None:
        # orig_max, factor, beta_fast, beta_slow = (
        #     rope_scaling["original_max_position_embeddings"],
        #     rope_scaling["factor"],
        #     rope_scaling["beta_fast"],
        #     rope_scaling["beta_slow"],
        # )
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048), # 原始最大序列长度
            rope_scaling.get("factor", 4), # 缩放因子
            rope_scaling.get("beta_fast", 4.0), # 快速缩放参数
            rope_scaling.get("beta_slow", 1.0), # 慢速缩放参数
        )
        #使用get更安全，同时也更方便处理缺失键的情况

        if end / orig_max > 1:
            # 计算corr_dim
            corr_dim = next((i for i in range(head_dim//2) if 2 * math.pi / freqs[i] > orig_max), head_dim//2)
            # // 表示整数除法, 取不大于结果的最大整数
            # range(head_dim//2) 生成一个从0到head_dim//2-1的整数序列
            # 2 * math.pi / freqs[i] > orig_max 检查每个频率是否大于 orig_max
            # next() 函数用于获取迭代器的下一个元素, 这里用于获取第一个频率大于 orig_max 的位置
            # corr_dim 是第一个频率大于 orig_max 的位置, 也即第一个需要被截断的位置
            '''
            这行代码意思是：
            首先从0到head_dim//2-1生成一个等差数列，然后检查每个频率造成的周期长度是否大于 orig_max。
            如果大于 orig_max, 则返回当前位置 i, 也即第一个需要被截断的位置。
            如果遍历完所有位置都没有大于 orig_max 的频率, 则返回 head_dim//2
            '''
            '''
            这行代码实际上是YaRN（Yet another RoPE extensioN）方法中的高频截断策略
            具体来说，它计算每个频率对应的周期长度（2 * math.pi / freq），并检查是否大于 orig_max。
            如果大于 orig_max, 则返回当前位置 i, 也即第一个需要被截断的位置。
            如果遍历完所有位置都没有大于 orig_max 的频率, 则返回 head_dim//2

            freqs[i]是指转动的角度, 2 * math.pi / freqs[i] 是指周期长度， 所以 max_seq_len * freqs[i] 是指总转动角度, 应当小于等于 2 * math.pi(2π，这里2π其实是360度的意思)
            '''

            power = torch.arange(0, dim // 2, device=freqs.device).float() / max(
                dim // 2 - 1, 1
            )

            beta = beta_slow + (beta_fast - beta_slow) * power

            scale = torch.where(
                torch.arange(dim // 2, device=freqs.device) < corr_dim,
                (beta * factor - beta + 1) / (beta * factor),
                1.0 / factor,
            )

            freqs = freqs * scale

        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()

        freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
        freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)

        return freqs_cos, freqs_sin


# 计算corr_dim

# 计算power

# 计算beta

# 计算scale

# 应用scale

# 返回cos 和 sin