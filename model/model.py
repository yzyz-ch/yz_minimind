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



def precompute_freqs(
    dim: int,    # 隐藏层维度（特征数），通常等于模型的hidden_size。
    end: int = int(32 * 1024),  # 目标序列长度，即预计算的最大token位置，默认32768。
    rope_base: float = 1e6,  # 旋转位置编码的基数base，对应RoPE的θ，默认1e6。   
    rope_scaling: Optional[dict] = None,  # 用于扩展RoPE到更长长度的配置字典，例如YARN算法的超参数。
):
    """
    计算旋转位置编码（RoPE, Rotary Positional Embedding）所需的频率参数。

    该函数用于根据模型的隐藏维度、目标序列长度、RoPE基数等参数，
    预先生成RoPE的频率向量。如果启用rope_scaling（如YARN方法），
    则会根据扩展长度做频率缩放和混合处理，以提升模型在长序列上的表现。

    返回:
        (freqs_cos, freqs_sin): 
            freqs_cos (torch.Tensor): 余弦部分的频率向量，形状为[end, dim]。
            freqs_sin (torch.Tensor): 正弦部分的频率向量，形状为[end, dim]。
    """
    # 1. 初始化标准 RoPE 频率。
    # torch.arange(0, dim, 2) 生成 [0, 2, 4, ... dim-2]
    # [: (dim // 2)] 取前dim//2个， 为了保险起见
    # 计算出的 freqs 就是标准的 1 / (base ** (2i / d))
    freqs, attn_factor = (
        # 生成RoPE的频率向量freqs：
        # torch.arange(0, dim, 2)[: (dim // 2)] 生成 [0, 2, 4, ..., dim-2] 的索引（一共dim//2个）
        # 每个索引都取float后除以dim，再作为RoPE的分母指数： freq_idx/dim
        # 计算方式等价于公式： 1 / (base ** (2i/d))。其中i为偶数索引。
        1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)),
        1.0,
    )

    if rope_scaling is not None:
        # 2. 从配置字典中提取 YaRN 的超参数
        # orig_max: 模型预训练时的原始最大长度（例如 Llama-2 是 2048 或 4096）
        # factor: 要扩展的倍数 s (比如从 2k 扩展到 32k，factor 就是 16)
        # beta_fast (对应论文中的 α): 高频边界，波长比例大于此值的维度不缩放
        # beta_slow (对应论文中的 β): 低频边界，波长比例小于此值的维度全量缩放
        # attn_factor: 注意力温度补偿，由于距离拉长导致注意力分布发散（变平缓），需要乘上一个系数让注意力重新“聚焦”
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )

        # 只有当要推断的长度大于原始训练长度时，才应用缩放
        if end / orig_max > 1.0:
            # 3. 使用前文推导的公式，定义波长比例 b 到维度索引 i 的映射函数
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (
                2 * math.log(rope_base)
            )

            # 4. 计算高频区和低频区的维度切分点
            # low: 不需要缩放的高频部分的最高索引
            # high: 需要完全缩放的低频部分的最低索引
            low, high = (
                max(math.floor(inv_dim(beta_fast)), 0),
                min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1),
            )

            # 5. 计算混合因子 γ (Ramp)
            # 论文中的公式：γ = (b - low) / (high - low)
            # ramp 是一个 0→1 的线性平滑系数
            # 在 low 之前，ramp 为 0；在 high 之后，ramp 为 1；在 low 和 high 之间，线性过渡。
            # clamp 函数限制了数值只能在 [0, 1] 之间。
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )

            # 6. 频率融合公式：f'(i) = f(i) * ((1-γ) + γ/s)
            # 当 ramp=0 时（高频）：系数为 1，保持原频率不变。
            # 当 ramp=1 时（低频）：系数为 1/factor，即对频率进行线性插值缩放。
            # ramp在0-1之间时：平滑过渡。
            freqs = freqs * (1 - ramp + ramp / factor)

    # 7. 根据目标长度 end，生成位置索引向量 t
    t = torch.arange(end, device=freqs.device)

    # 8. 计算外积：将位置 t 与处理好的频率 freqs 相乘，得到每个位置的旋转角度 θ
    freqs = torch.outer(t, freqs).float()

    # 9. 计算 Cos 和 Sin，并应用注意力补偿系数 (attn_factor)
    # cat: 将两个频率向量拼接在一起，dim=-1表示在最后一个维度拼接
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # rotate_half: 将输入张量 x 的最后一个维度分成两半，前一半取反，然后拼接在一起
    # //: 整数除法，返回商的整数部分
    # [..., x.shape[-1] // 2:]: 取后一半
    # [..., : x.shape[-1] // 2]: 取前一半
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    # unsqueeze: 在指定维度上增加一个维度，维度大小为1
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed    


