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
        rope_theta: int = 1000000,
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
        self.rope_theta = rope_theta
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

