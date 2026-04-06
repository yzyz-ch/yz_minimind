from transformers import PretrainedConfig


class YzMiniMindConfig(PretrainedConfig):
    model_type = "yz_minimind"
    # 这个配置类的作用：
    # 1. 统一保存模型结构超参数与训练/推理相关开关。
    # 2. 在构建 Attention、RoPE、MoE、Embedding、LM Head 等模块时传入。
    # 3. 训练时用于实例化模型；推理/保存/加载模型时也会随配置一起持久化。

    def __init__(
        self,
        # dropout: dropout 概率。
        # 使用位置：Attention 中的 attn_dropout / resid_dropout，
        # 训练时用于正则化；推理时通常不会生效。
        dropout: float = 0.0,

        # bos_token_id: 句子开始 token 的 id。
        # 使用位置：分词/生成任务开始时可作为起始标记；保存到配置中供生成流程读取。
        bos_token_id: int = 1,

        # eos_token_id: 句子结束 token 的 id。
        # 使用位置：文本生成遇到该 token 时通常停止解码。
        eos_token_id: int = 2,

        # hidden_act: 前馈网络/专家网络里使用的激活函数名称。
        # 常见值如 silu、gelu。当前片段里只保存配置，通常在 MLP 或 MoE 层构建时使用。
        hidden_act: str = "silu",

        # hidden_size: 模型隐藏维度 d_model。
        # 使用位置：Embedding 输出维度、Attention 输入输出维度、RMSNorm 维度、MLP 输入维度等。
        hidden_size: int = 512,

        # intermediate_size: 前馈网络中间层维度。
        # 使用位置：MLP / MoE expert 的上投影维度。通常会大于 hidden_size。
        # 当前代码片段里先保存，具体在 FFN 或专家层实例化时使用。
        intermediate_size: int = None,

        # max_position_embeddings: 模型支持的最大位置长度。
        # 使用位置：控制训练/推理最大序列长度，也影响 RoPE 预计算范围和长上下文设置。
        max_position_embeddings: int = 32768,

        # num_attention_heads: Query 头数，也就是总 attention head 数。
        # 使用位置：Attention 中计算 q_proj 输出维度、head_dim，以及多头拆分逻辑。
        num_attention_heads: int = 8,

        # num_hidden_layers: Transformer block 层数。
        # 使用位置：构建模型时决定堆叠多少层 Attention + FFN/MoE block。
        num_hidden_layers: int = 8,

        # num_key_value_heads: Key/Value 头数。
        # 使用位置：Attention 中决定是否使用 GQA/MQA。
        # 如果小于 num_attention_heads，就表示多个 Q 头共享同一组 KV 头。
        num_key_value_heads: int = 2,

        # vocab_size: 词表大小。
        # 使用位置：Token Embedding 层和最终 LM Head 输出维度。
        vocab_size: int = 6400,

        # rms_norm_eps: RMSNorm 的数值稳定项 epsilon。
        # 使用位置：RMSNorm 前向计算时避免除零或数值过小导致不稳定。
        rms_norm_eps: float = 1e-05,

        # rope_base: RoPE 的频率基数 theta/base。
        # 使用位置：precompute_pos_cis 中生成 cos/sin 位置编码频率。
        rope_base: int = 1000000,

        # inference_rope_scaling: 推理时是否启用长上下文 RoPE 缩放。
        # 使用位置：如果为 True，会生成 rope_scaling 配置并在 RoPE 预计算中启用 YaRN 类缩放。
        inference_rope_scaling: bool = False,

        # flash_attn: 是否允许优先走 PyTorch 的 Flash Attention 路径。
        # 使用位置：Attention.forward 中决定调用 scaled_dot_product_attention 还是手写 attention。
        flash_attn: bool = True,
        
        ############ MoE ############
        # use_moe: 是否启用 Mixture of Experts。
        # 使用位置：构建 block 时决定采用普通 MLP 还是 MoE 前馈层。
        use_moe:bool=False,

        # num_experts_per_tok: 每个 token 路由到多少个专家（top-k）。
        # 使用位置：MoE gate 选择专家时的 top-k 路由数量。
        num_experts_per_tok:int=2,

        # n_routed_experts: 可被 gate 动态路由的专家数量。
        # 使用位置：MoE 路由层构建专家池时使用。
        n_routed_experts:int=4,

        # n_shared_experts: 所有 token 都共享使用的专家数量。
        # 使用位置：某些 MoE 结构会同时保留共享专家与路由专家。
        n_shared_experts:int=1,

        # scoring_func: gate 打分函数类型。
        # 使用位置：MoE router 中决定用 softmax 等方式把 router logits 转成分数/概率。
        scoring_func:str='softmax',

        # aux_loss_alpha: MoE 辅助负载均衡损失的权重。
        # 使用位置：训练时给 router 增加辅助损失，避免专家使用过于不均衡。
        aux_loss_alpha:float=0.1,

        # seq_aux: 是否按序列粒度计算辅助损失。
        # 使用位置：MoE 训练时决定 auxiliary loss 的统计方式。
        seq_aux:bool=True,

        # norm_topk_prob: 是否对 top-k 专家的概率重新归一化。
        # 使用位置：MoE 路由后融合多个专家输出时控制权重归一化方式。
        norm_topk_prob:bool=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # dropout 配置会在 Attention 的 dropout 层中直接读取。
        self.dropout = dropout
        # 生成起始 token id，主要供文本生成流程读取。
        self.bos_token_id = bos_token_id
        # 生成终止 token id，主要供解码停止条件读取。
        self.eos_token_id = eos_token_id
        # 前馈/专家层的激活函数类型；当前先存配置，构建 MLP/MoE 时再消费。
        self.hidden_act = hidden_act
        # 整个模型的主隐藏维度，几乎所有核心层都会依赖它。
        self.hidden_size = hidden_size
        # FFN / Expert 的中间维度，通常在 MLP/MoE 构造阶段使用。
        self.intermediate_size = intermediate_size
        # 最大位置长度限制，影响训练/推理支持的最大序列长度。
        self.max_position_embeddings = max_position_embeddings
        # Q 头数，Attention 中会据此拆分多头。
        self.num_attention_heads = num_attention_heads
        # Transformer 层数，模型主体堆叠时使用。
        self.num_hidden_layers = num_hidden_layers
        # KV 头数，Attention 中用于实现 GQA/MQA。
        self.num_key_value_heads = num_key_value_heads
        # 词表大小，Embedding 和输出层都会用到。
        self.vocab_size = vocab_size
        # RMSNorm 数值稳定项，RMSNorm.forward 中会使用。
        self.rms_norm_eps = rms_norm_eps
        # RoPE 基数，位置编码频率预计算时使用。
        self.rope_base = rope_base
        # 是否启用推理时的 RoPE 长上下文缩放。
        self.inference_rope_scaling = inference_rope_scaling
        # 是否允许 Attention 使用 flash 路径。
        self.flash_attention = flash_attention
        # 是否启用 MoE 结构。
        self.use_moe=use_moe
        # 每个 token 选几个专家。
        self.num_experts_per_tok=num_experts_per_tok
        # 路由专家数量。
        self.n_routed_experts=n_routed_experts
        # 共享专家数量。
        self.n_shared_experts=n_shared_experts
        # MoE 辅助损失统计方式开关。
        self.seq_aux=seq_aux
        # top-k 专家概率是否重新归一化。
        self.norm_topk_prob=norm_topk_prob
        # MoE 辅助损失权重。
        self.aux_loss_alpha=aux_loss_alpha
        # router 打分函数类型。
        self.scoring_func=scoring_func

        # rope_scaling 是给 RoPE 预计算函数使用的字典配置。
        # 当 inference_rope_scaling=True 时启用，用于长上下文外推。
        # 后续会传入 precompute_pos_cis / 频率缩放逻辑中。
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

class Attention(nn.Module):
    def __init__(self, args: YzMiniMindConfig):
        super().__init__()

        # 如果没有单独指定 num_key_value_heads，
        # 就默认 KV 头数 = Q 头数，也就是普通 MHA
        # 如果指定了更少的 KV 头数，那就是 GQA / MQA 的做法
        self.num_key_value_heads = (
            args.num_attention_heads
            if args.num_key_value_heads is None
            else args.num_key_value_heads
        )

        # 要保证 Q 头数能被 KV 头数整除，
        # 因为后面要把每个 KV 头复制 n_rep 次去匹配 Q 头
        assert args.num_attention_heads % self.num_key_value_heads == 0

        # Q 头数
        self.n_local_heads = args.num_attention_heads #q_heads

        # KV 头数
        self.n_local_kv_heads = self.num_key_value_heads #kv_heads

        # 每个 KV 头需要复制多少次，才能和 Q 头数量对齐
        # 例如 Q=8, KV=2，则 n_rep=4
        self.n_rep = self.n_local_heads // self.n_local_kv_heads

        # 每个头的维度
        # hidden_size = num_heads * head_dim
        self.head_dim = args.hidden_size // args.num_attention_heads

        # Q 投影：输入 hidden_size，输出 num_attention_heads * head_dim
        # 其实这个输出大小通常就等于 hidden_size
        self.q_proj = nn.Linear(
            args.hidden_size,
            args.num_attention_heads * self.head_dim,
            bias=False
        )

        # K 投影：输出 KV 头数 * head_dim
        # 如果 KV 头数少于 Q 头数，这里参数量会更少
        self.k_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False
        )

        # V 投影：和 K 一样，也是 KV 头数 * head_dim
        self.v_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False
        )

        # 输出投影：把所有 head 拼回去后，再投影回 hidden_size
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=False
        )

        # 注意力权重上的 dropout
        self.attn_dropout = nn.Dropout(args.dropout)

        # attention 输出之后的 dropout
        self.resid_dropout = nn.Dropout(args.dropout)

        self.dropout = args.dropout

        # 是否启用 PyTorch 自带的 Flash Attention
        # 前提：
        # 1. 当前 PyTorch 支持 scaled_dot_product_attention
        # 2. 配置里允许 flash attention
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and args.flash_attn
        )

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # x: [bsz, seq_len, hidden_size]
        bsz, seq_len, _ = x.shape

        # 先把输入 x 通过三个线性层，得到 Q / K / V
        # 这时它们仍然是 3 维张量
        # xq: [bsz, seq_len, q_heads * head_dim]
        # xk: [bsz, seq_len, kv_heads * head_dim]
        # xv: [bsz, seq_len, kv_heads * head_dim]
        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)

        # 把最后一维拆成 “头数 x 每头维度”
        # xq: [bsz, seq_len, q_heads, head_dim]
        # xk: [bsz, seq_len, kv_heads, head_dim]
        # xv: [bsz, seq_len, kv_heads, head_dim]
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # position_embeddings 是外部提前算好的 RoPE cos/sin
        # cos: [max_seq_len, head_dim]
        # sin: [max_seq_len, head_dim]
        cos, sin = position_embeddings

        # 只取当前序列长度对应的位置编码
        # RoPE 只作用在 Q 和 K 上，不作用在 V 上
        # 输出形状不变：
        # xq: [bsz, seq_len, q_heads, head_dim]
        # xk: [bsz, seq_len, kv_heads, head_dim]
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # -----------------------------
        # KV Cache：只在推理阶段常用
        # -----------------------------
        # 如果传入 past_key_value，说明前面已经缓存了历史 token 的 K/V
        # 当前新算出的 K/V 要拼接到历史缓存后面
        if past_key_value is not None:
            # past_key_value[0]: 历史 K，形状大致 [bsz, past_len, kv_heads, head_dim]
            # xk: 当前步 K，形状 [bsz, seq_len, kv_heads, head_dim]
            # 在 seq_len 这个维度拼接，得到总长度 past_len + seq_len
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)

        # 如果 use_cache=True，就把更新后的 K/V 返回出去，
        # 供下一次 forward 继续复用
        past_kv = (xk, xv) if use_cache else None

        # -----------------------------
        # 调整维度顺序，准备 attention 计算
        # -----------------------------
        # attention 常见计算格式是 [bsz, heads, seq_len, head_dim]
        # 所以把 head 维度换到前面
        #
        # xq 原来: [bsz, seq_len, q_heads, head_dim]
        # 变成:    [bsz, q_heads, seq_len, head_dim]
        #
        # xk/xv 在 transpose 之前先 repeat_kv
        # 因为 KV 头数可能少于 Q 头数，要先复制到一样多
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )

        # 经过 repeat_kv 后：
        # xk: [bsz, q_heads, total_kv_len, head_dim]
        # xv: [bsz, q_heads, total_kv_len, head_dim]
        # 这样就能和 xq 的 q_heads 对齐了

        # -----------------------------
        # 路径1：Flash Attention
        # -----------------------------
        # 条件：
        # 1. 当前环境支持 flash
        # 2. 序列长度 > 1
        # 3. 没有复杂 mask，或者 mask 全 1
        if self.flash and seq_len > 1 and (
            attention_mask is None or torch.all(attention_mask == 1)
        ):
            output = F.scaled_dot_product_attention(
                xq,                      # [bsz, heads, q_len, head_dim]
                xk,                      # [bsz, heads, kv_len, head_dim]
                xv,                      # [bsz, heads, kv_len, head_dim]
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True           # 因果注意力：当前位置不能看未来位置
            )

        # -----------------------------
        # 路径2：手写普通 Attention
        # -----------------------------
        else:
            # 先算注意力分数 QK^T / sqrt(d)
            # xq: [bsz, heads, q_len, head_dim]
            # xk.transpose(-2, -1): [bsz, heads, head_dim, kv_len]
            # scores: [bsz, heads, q_len, kv_len]
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # 构造上三角 mask，屏蔽未来 token
            # 对角线以上位置填 -inf
            # softmax 后这些位置概率就接近 0
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)

            # 如果外部又传入了 attention_mask（比如 padding mask）
            # 就继续叠加到 scores 上
            if attention_mask is not None:
                # [bsz, seq_len] -> [bsz, 1, 1, seq_len]
                # 这样可以广播到所有 head 和所有 query 位置
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

                # 原始 mask 通常是 1 表示可见，0 表示不可见
                # 转换成：可见=0，不可见=-1e9
                # 这样加到 scores 上后，被 mask 的位置 softmax 后接近 0
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            # softmax 得到注意力概率
            # 先转 float 是为了数值更稳定
            # 再 type_as(xq) 转回与 xq 一致的数据类型
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)

            # 对注意力概率做 dropout
            scores = self.attn_dropout(scores)

            # 用注意力权重加权 V
            # scores: [bsz, heads, q_len, kv_len]
            # xv:     [bsz, heads, kv_len, head_dim]
            # output: [bsz, heads, q_len, head_dim]
            output = scores @ xv

        # 把多头结果再拼回去
        # [bsz, heads, seq_len, head_dim]
        # -> [bsz, seq_len, heads, head_dim]
        # -> [bsz, seq_len, heads * head_dim]
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)

        # 再过一个输出线性层，映射回 hidden_size
        output = self.resid_dropout(self.o_proj(output))

        # 返回：
        # 1. 当前 attention 层输出
        # 2. 更新后的缓存 past_kv（如果 use_cache=True）
        return output, past_kv

class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        # Transformer 中的前馈网络通常先把 hidden_size 扩展到更高维，
        # 在更大的特征空间里做非线性变换后，再投影回 hidden_size。
        # 这里如果没有显式指定 intermediate_size，就采用 LLaMA 风格的经验值:
        #   intermediate_size ~= 8 / 3 * hidden_size
        # 然后向上对齐到 64 的倍数，便于张量核心/硬件高效计算。
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        # gate_proj 和 up_proj 都把输入从 hidden_size 映射到 intermediate_size，
        # 这是 SwiGLU/GLU 风格 FFN 的典型结构：
        # 1. gate_proj(x) 经过激活函数，生成“门控权重”
        # 2. up_proj(x) 生成待调制的内容分量
        # 3. 两者逐元素相乘，保留有用信息并抑制无关信息
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        # down_proj 再把高维特征压回 hidden_size，保证输出可以继续走残差连接。
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        # ACT2FN 是“激活函数名字 -> 具体函数实现”的映射表，例如 silu / gelu。
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # 计算流程:
        # x
        # -> gate_proj(x) -> act_fn(...)       产生门控分支
        # -> up_proj(x)                        产生内容分支
        # -> 两分支逐元素相乘                  完成门控
        # -> down_proj(...)                    投影回 hidden_size
        # -> dropout                           训练时做正则化
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))        


class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        # 每个注意力头处理的隐藏维度。
        # hidden_size 必须能被 num_attention_heads 整除。
        self.head_dim = config.hidden_size // config.num_attention_heads
        # 单层 Transformer 的注意力子层。
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        # 这里采用 Pre-Norm 结构：
        # 先做 LayerNorm/RMSNorm，再送入 Attention 或 MLP。
        # 相比 Post-Norm，Pre-Norm 在深层网络里通常更稳定。
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 根据配置决定使用普通 FFN 还是 MoE FFN。
        # use_moe=False 时是标准前馈层；
        # use_moe=True  时由多个 expert 共同完成前馈计算。
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # 保存残差分支输入。Attention 子层计算结束后会与它相加。
        residual = hidden_states
        # 注意力子层的数据流:
        # 1. 先对输入做 RMSNorm
        # 2. 再进入 self_attn，内部会完成 QKV 投影、RoPE 位置编码、mask、softmax 等
        # 3. 返回新的 hidden_states，以及当前层可缓存的 K/V
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        # Attention 残差连接: x + Attention(Norm(x))
        hidden_states += residual
        # MLP 子层同样采用 Pre-Norm + 残差:
        # hidden_states = hidden_states + MLP(Norm(hidden_states))
        # 这样每个 Block 的主干结构就是:
        #   x -> Attention -> residual -> MLP -> residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        # token embedding: 把离散 token id 映射成连续向量表示。
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        # 堆叠 num_hidden_layers 个 Transformer Block，构成主干网络。
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        # 最后一层归一化，和 LLaMA 系列结构保持一致。
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 预先计算好 RoPE 所需的 cos/sin 表。
        # dim 使用的是单头维度 head_dim，因为 RoPE 是按每个 attention head 独立施加的。
        # end 是最多支持的位置长度 max_position_embeddings。
        # 这样前向时只需要按 [start_pos : start_pos + seq_length] 切片即可，避免重复计算。
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, rope_base=config.rope_theta,
                                                    rope_scaling=config.rope_scaling)
        # register_buffer 表示这两个张量属于模型状态，会随着 .to(device) 一起迁移，
        # 但它们不是可训练参数，不参与梯度更新。
        # persistent=False 表示保存 checkpoint 时不强制写入 state_dict，
        # 因为它们可以根据配置重新计算出来。
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        # input_ids: [batch_size, seq_length]
        # 在训练阶段通常是整段序列；
        # 在自回归推理阶段，往往是“当前新输入的几个 token”。
        batch_size, seq_length = input_ids.shape
        # 某些上层框架可能传入自定义 cache 对象而不是本文件期望的 list[tuple]，
        # 这里做一个兼容兜底：如果发现它像一个带 layers 属性的 cache 容器，就先忽略掉。
        if hasattr(past_key_values, 'layers'): past_key_values = None
        # 如果没有 cache，就为每一层补一个 None，后面统一按“逐层 zip”处理。
        past_key_values = past_key_values or [None] * len(self.layers)
        # start_pos 表示“当前这批 token 在整条上下文中的起始位置”。
        # 例如:
        # - 首次前向时没有 cache，start_pos = 0
        # - 增量解码时，若历史缓存长度为 128，则新 token 的位置从 128 开始
        # 这里从第 0 层缓存的 key 长度推断历史序列长度。
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # [B, T] -> [B, T, hidden_size]
        # 先把 token id 查表变成 embedding，再做 dropout。
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 为当前这段 token 切出对应位置范围的 RoPE 旋转参数。
        # 假设 start_pos=128, seq_length=4，则取 [128:132] 这 4 个位置的 cos/sin。
        # 后续每一层的 attention 都复用同一份位置编码切片。
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        # 收集每一层新的 K/V cache，供下一次增量解码继续使用。
        presents = []
        # 逐层执行 Transformer Block。
        # hidden_states 会在每一层被更新，形成深层语义表示。
        # 每层也会根据 use_cache 返回自己的 present_key_value。
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        # 所有层结束后，再做一次最终归一化，输出给 LM Head 或其他上层任务头。
        hidden_states = self.norm(hidden_states)

        # 如果某些层使用的是 MoE FFN，它们通常会维护一个辅助损失 aux_loss，
        # 用于鼓励 expert 路由更加均衡，防止所有 token 都挤到少数 expert 上。
        # 普通 FFN 层没有这个属性，因此只对 MOEFeedForward 层求和。
        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )

        # 返回:
        # 1. hidden_states: 最终隐状态 [B, T, hidden_size]
        # 2. presents: 每层新的 KV cache，供推理阶段复用
        # 3. aux_loss: MoE 辅助损失；若未使用 MoE，通常为 0
        return hidden_states, presents, aux_loss
