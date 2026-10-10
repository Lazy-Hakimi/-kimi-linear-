# Kimi Linear musicGPT — 实现（PyTorch）

模型核心解码部分按论文 **《Kimi Linear: An Expressive, Efficient Attention Architecture》**（arXiv:2510.26692）
实现的完整模型参考代码。所有超参数集中在**单一配置文档**中，模型代码不含任何硬编码维度。

## 目录结构

```
configs/                              # 配置文档（1.5B 主配置 + tiny 测试配置）
kimi_linear/                          # Kimi Linear 主干（论文参考实现）
├── configuration.py                  #   配置加载/校验 + 解析参数量计算
├── precision.py                      #   硬件精度适配层（V100=fp16 / Ampere+=bf16）
├── fp16.py                           #   纯 FP16 训练核心（Kahan + 8-bit Adam + FP16GradScaler）
├── kda.py                            #   KDA：chunkwise + recurrent 双内核、ShortConv、门控
├── mla.py                            #   MLA（NoPE，吸收式 latent attention）
├── ffn.py                            #   FFN 分支工厂：Dense / MoE（由配置切换）
├── modeling.py                       #   Block / Backbone / CausalLM / 推理缓存
└── graph_decode.py                   #   GraphedDecoder：CUDA Graph 捕获式增量解码
kimi_musicgpt/                        # MusicGPT 完整管线（codec + 分词器 + 训练/生成）
scripts/                              # 生产训练入口
└── train_musicgpt_fp16.py            #   完整 musicGPT 两阶段生产训练（codec GAN + 1.5B GPT，纯 FP16）
```

## 1.5B 稠密配置（按论文结构重新计算，非原文 48B）

| 项 | 值 | 论文依据 |
|---|---|---|
| hidden_size | 2048 | — |
| 层数 | 24 = `[kda,kda,kda,mla] × 6`（**3:1 混合比**） | §4 / Table 1（3:1 最优） |
| KDA | 16 heads × head_dim 128（dk=dv=128），ShortConv k=4，门控/输出门低秩 rank=128 | §4 Neural Parameterization |
| MLA | 16 heads，q_lora 1024，kv_lora 512，qk_nope 128，v_head 128，**NoPE**（qk_rope=0） | §4（MLA 层用 NoPE） |
| FFN | Dense SwiGLU intermediate 5760（`ffn.mode="dense"`） | 论文 FFN 为 Moonlight 式；此处按需求支持 Dense/MoE 切换 |
| 词表 | 131072（tied embedding） | — |

**总参数量：1,509,215,232（≈1.509B）**，解析计算与 meta 设备实例化逐参数精确一致。

参数构成：

```
embedding            268,435,456
KDA 层  ×18          322,117,632   (每层 17,895,424)
MLA 层  ×6            69,215,232   (每层 11,535,872)
Dense FFN ×24        849,346,560   (每层 35,389,440)
final norm                 2,048
─────────────────────────────────
总计              1,509,215,232
```

## 与论文公式的对应

### KDA（Kimi Delta Attention）

递推式（论文 Eq. 1，`kda.kda_recurrent`，用于逐步解码与正确性基准）：

```
S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t−1} + β_t k_t v_tᵀ ,   o_t = S_tᵀ q_t
```

- 与 GDN 的关键差异：α_t 是**逐通道**细粒度门控（GDN 为每头标量）；
- β_t = Sigmoid(W_β x)，每头标量；q 缩放 K^{-0.5}。

chunkwise 内核（论文 Eq. 6–9 + 附录 C Listing 1，`kda.kda_chunkwise`）：

```
M = (I + L)⁻¹,  L[c,i] = β_c ⟨k_c, exp(gc_c − gc_i) k_i⟩  (i < c)，再右乘 Diag(β)
W = M (exp(gc)⊙K),  U = M V                                    (UT transform, Eq. 6/7)
O = (exp(gc)⊙Q) S₀ + Tril[(Q⊙exp(gc))(K⊙exp(−gc))ᵀ] (U − W S₀)   (Eq. 9)
S₁ = exp(gc_C)·S₀ + (exp(gc_C−gc)⊙K)ᵀ (U − W S₀)                 (Eq. 8)
```

其中 gc 为块内累计对数衰减。该实现已与 FLA 官方 `fla/ops/kda/naive.py`
逐元素对齐（误差 ~1e-7，fp32）。

模块级参数化（论文 §4，全部由配置驱动）：

```
q,h/k,h = L2Norm(Swish(ShortConv(W_q/k x)))     每头 128 维，因果深度卷积 k=4
v,h     = Swish(ShortConv(W_v x))
α_t     = exp(−exp(A_log) ⊙ softplus(W↑α W↓α x + dt_bias))   逐通道 ∈ (0,1]，f(·) 同 GDN/Mamba
β_t     = Sigmoid(W_β x)
输出     = W_o( Sigmoid(W↑g W↓g x) ⊙ HeadwiseRMSNorm(core) )   (Eq. 10，Sigmoid 输出门)
```

### 混合架构与 MLA

- 3 层 KDA + 1 层 MLA 周期堆叠（配置 `layer_pattern`，可任意改比例）；
- MLA 为 DeepSeek 式 latent attention，**NoPE**（`qk_rope_head_dim=0`），
  位置信息完全由 KDA 承担（论文 §4、§6.1）；
- 推理时使用**吸收式**计算（q 用 W_kv_bᵀ 投影进 latent 空间，直接对压缩缓存
  `c_KV` 做注意力），这是 MLA KV 缓存缩减的来源；`qk_rope_head_dim>0` 时支持 RoPE。

### FFN 分支（Dense / MoE 可切换）

配置文档中 `ffn.mode` 即开关，仅此一处决定分支：

```jsonc
"ffn": {
  "mode": "dense",                    // "dense" | "moe"  ← 切换分支
  "hidden_act": "silu",
  "intermediate_size": 5760,          // Dense 分支（SwiGLU：gate/up/down 三个矩阵）
  "moe": {                            // MoE 分支（Moonlight 式，仅 mode="moe" 时使用）
    "n_routed_experts": 64,
    "num_experts_per_tok": 6,
    "n_shared_experts": 1,
    "moe_intermediate_size": 1408,
    "routed_scaling_factor": 1.0,
    "norm_topk_prob": true,
    "scoring_func": "sigmoid",        // sigmoid 打分 + top-k（论文/Moonlight 采用）
    "first_k_dense_replace": 1,       // 首层固定稠密（论文：首层 dense 保训练稳定）
    "aux_loss_coeff": 0.0             // >0 时启用 Switch 式负载均衡辅助损失
  }
}
```

把 `mode` 改为 `"moe"` 即得 MoE 稀疏版（其余 MoE 参数也在同一文档中），无需改任何代码。
按上述示例 MoE 参数切换后的总参数量为 **13,631,147,008（≈13.6B）**（每 MoE 层 562,429,952）；
如需控制 MoE 版总规模，调整 `n_routed_experts` / `moe_intermediate_size` 即可。

## 使用

```python
import torch, sys
sys.path.insert(0, "kimi_linear")                    # 仓库根目录下
from kimi_linear import KimiLinearConfig, KimiLinearForCausalLM

cfg  = KimiLinearConfig.from_json("kimi_linear/configs/kimi_linear_1p5b_dense.json")
model = KimiLinearForCausalLM(cfg).cuda().to(torch.bfloat16)

ids = torch.randint(0, cfg.vocab_size, (1, 128), device="cuda")
out  = model(ids, labels=ids)          # logits / loss / (aux_loss 当 MoE 且 aux_loss_coeff>0)
# MoE 训练时请自行把 out["aux_loss"] 加入总损失（实现中单独返回，不自动相加）

# 增量解码（KDA 循环态 + MLA latent 缓存）
out  = model(ids[:, :32], use_cache=True)
out  = model(ids[:, 32:33], cache=out["cache"], use_cache=True)
# 或直接 model.generate(ids, max_new_tokens=64)
```

脚本：

```bash
# 完整 musicGPT 生产训练（两阶段：codec GAN + 1.5B GPT，纯 FP16，V100-16GB 实测）
python scripts/train_musicgpt_fp16.py --stage both \
    --gpt-config kimi_musicgpt/configs/musicgpt_1p5b.json \
    --codec-config kimi_musicgpt/configs/codec_tiny_16k.json \
    --songs 64 --steps 3000 --batch-size 1 --max-frames 100 --out runs/musicgpt_1p5b
```


## 混合精度（V100 / sm_70 适配）

主配置 `torch_dtype` 已改为 **fp16**：Tesla V100（sm_70）只有 FP16 Tensor Core，
没有 bf16 / TF32 硬件（实测 bf16 GEMM 比 fp32 还慢 20%，`allow_tf32` 是空操作）。
`kimi_linear/precision.py` 按计算能力自动选型（`resolve_precision("auto")`）；
后续生产布局已升级为**纯 FP16**（权重 fp16 存储 + Kahan 补偿 + 8-bit Adam，KDA 循环状态 S 在半精度运行时
自动保存在 fp32（fp32 master state，防跨块累积误差）。

## 测试结果

- KDA chunkwise vs 逐步递推：输出 3.4e-8 / 状态 4.2e-7（fp32）；与 FLA 官方 naive 实现差 0；
- 前缀不变性（因果性）：整模型 3.0e-7，KDA 内核精确 0；
- 缓存增量解码 == 整段前向：3.6e-7（dense 与 moe 均通过）；
- MoE：首层稠密、路由权重归一、辅助损失、8/8 专家被利用；
- 反向传播：164 个梯度张量，KDA 的 A_log/dt_bias/conv、MLA 的 kv_b、FFN 均有梯度；
- 参数量：解析 1,509,215,232 == meta 实例化（tiny 与 1.5B 两份配置均一致）；
- 可学习性：tiny 模型在固定批次上训练 60 步，dense 5.55→0.03、MoE 5.84→0.03（AdamW, lr=3e-3）；
- GPU 推理训练测试：1.5B fp16 前向 T=256 loss=11.783 ≈ ln(131072)=11.78（随机初始化理论值）。

## 已知边界（参考实现范围）

- 纯 PyTorch 内核（未用 Triton/FLA CUDA 内核）。KDA chunkwise 已做分块 log 空间
  向量化 + `solve_triangular`（不再有逐列 Python 循环），q/k/v 保持 bf16 存储、
  逐块 fp32 上采样以省 ~25-30% 算子峰值显存（内部数学与全 fp32 上采样逐位一致），
  MLA 预填充已走 `scaled_dot_product_attention` 融合路径；但仍是参考实现，不含
  bf16 张量核二级分块补偿与 vLLM 级推理栈（paged attention / MoE 并行），长序列
  吞吐不及 FLA/FlashKDA 官方内核；
- 未实现 padding mask（批量变长输入）；增量推理按批内相同步数假设；
- MLA 缓存保存压缩 latent `c_KV`（论文的高效形态）；
- 权重初始化为参考实现约定（Normal(0, initializer_range)，A_log/dt_bias 由配置的
  初始值；因原论文未公布其精确初始化方案）。

## 独立复核

实现完成后由独立智能体对照论文全文与 FLA 官方 `fla/ops/kda/naive.py` 逐项复核
（KDA 递推式/chunkwise/参数化、MLA 吸收式代数、3:1 混合、FFN 分支、参数量算术），
9 项检查全部通过；其提出的缓存路径与配置化建议均已修复并复测
（复核发现 A 在报告返回前已被修复，16+32 分块扩展测试 2.7e-7 证实生效）。
