---
layout:     post
title:      "从参数量到 FLOPs：手算 Transformer 的规模与开销"
subtitle:   "用矩阵形状连接模型权重、训练计算量和推理耗时"
date:       2026-09-23 12:00:00 +0800
author:     "Liu Mengxuan"
mathjax:    true
header-img: "img/post-bg-miui6.jpg"
categories: [技术]
tags:       [技术, AI Infra, Transformer, FLOPs, 推理, 性能分析]
---

模型名称里的“7B”描述了参数规模，却没有直接告诉我们处理一段文本要花多少计算，更不能直接换算成生成一个 token 的时间。把这几个量连接起来，需要从矩阵形状出发：矩阵里存了多少个数，决定参数量；矩阵与多少行输入相乘，决定计算量；计算和数据搬运在硬件上的执行效率，才决定时间。

下面用 MHA、SwiGLU 和 RMSNorm 组成的 Decoder-only 模型贯穿计算，再将同一套方法延伸到 GQA、训练和带 KV Cache 的推理。

## 1. 一个权重矩阵，两种计数

设线性层的输入、权重和输出分别为：

<p align="center">$X_{T\times d_{in}}W_{d_{in}\times d_{out}}=Y_{T\times d_{out}}$</p>

权重矩阵包含 `d_in × d_out` 个可学习的数，因此忽略偏置时：

<p align="center">$P=d_{in}d_{out}$</p>

`T` 表示本次处理的 token 数。无论输入一行还是一百行，所有行使用同一份权重，参数量不变。

计算量则随输入行数增长。一个输出元素是输入的一行与权重的一列做点积，需要 `d_in` 次乘法和 `d_in−1` 次加法。共有 `T×d_out` 个输出，因而：

<p align="center">$F=T d_{out}(2d_{in}-1)\approx2T d_{in}d_{out}=2TP$</p>

<p align="center">
  <img src="/img/in-post/ai-infra-cost-matmul.svg" alt="用二行三列输入、三行四列权重和二行四列输出展示矩阵乘法，权重十二个参数，八个输出各做三乘两加" style="max-width: 100%;">
</p>

图中小矩阵精确需要 40 次运算，`2MKN` 公式估算为 48 次。实际线性层的内维通常有几千，省略每个点积少一次加法的差别就很小。

这里按常见口径把乘法和加法各计一次，即一次乘加计 2 FLOPs，即使硬件用一条融合乘加指令执行。**FLOPs 是运算总量，FLOP/s 是每秒运算能力**；312 TFLOPS 表示每秒约 `312×10¹²` 次浮点运算。

## 2. 沿模型结构累加参数量

记隐藏维度为 `d=d_model`，FFN 中间维度为 `d_ff`，词表大小为 `V`，Block 数为 `L`。先忽略线性层偏置。

### 2.1 Embedding 与输出头

Embedding 是一张 `[V,d]` 的表，每个 token ID 选择其中一行，所以有 `Vd` 个参数。最后的 LM Head 将隐藏向量投影成词表 logits，其数学权重形状为 `[d,V]`，也有 `Vd` 个参数。

如果使用 Weight Tying，两处引用同一份权重，只计一次。共享时，Embedding 查表使用 `E`，输出打分使用 `hEᵀ`；这两个运算并非数学上的互逆操作。

### 2.2 Attention 投影

标准 MHA 满足 `h×d_k=d`。把所有 head 的输出合在一起，Q、K、V 和输出投影的四个矩阵都是 `[d,d]`：

<p align="center">$P_{attn}=4d^2$</p>

GQA 保持 Query 头数 `h_q`，减少 KV 头数为 `h_kv`。在 `h_q d_k=d` 的常见配置下：

<p align="center">$P_{attn}=2d^2+2d h_{kv}d_k$</p>

前一项来自 Q 和输出投影，后一项来自 K、V 投影。固定总 Query 宽度时，减少 KV 头会缩小 K/V 权重和 KV Cache。

### 2.3 FFN 与 Norm

SwiGLU 的 gate、up 矩阵都是 `[d,d_ff]`，down 矩阵是 `[d_ff,d]`，所以：

<p align="center">$P_{ffn}=3dd_{ff}$</p>

每个 RMSNorm 通常只有长度为 `d` 的可学习缩放向量，一个 Block 中两个 RMSNorm 共 `2d` 个参数。均方根是从输入算出来的统计量，不是需要额外学习的参数。

对于上述 MHA Block：

<p align="center">$P_{block}=4d^2+3dd_{ff}+2d$</p>

若 Embedding 和 LM Head 不共享，整个模型为：

<p align="center">$P_{total}=2Vd+LP_{block}+d$</p>

最后一个 `d` 来自所有 Block 之后的最终 RMSNorm。

## 3. 用 LLaMA-2-7B 完成一次手算

采用配置 `d=4096、d_ff=11008、L=32、V=32000`，32 个 Query 头，每头 128 维，使用 MHA。

| 单 Block 组件 | 计算 | 参数量 |
| --- | --- | ---: |
| Attention 投影 | `4 × 4096²` | 67,108,864 |
| SwiGLU | `3 × 4096 × 11008` | 135,266,304 |
| 两个 RMSNorm | `2 × 4096` | 8,192 |
| 合计 | | **202,383,360** |

单个 Block 中，FFN 占约 66.8%，Attention 投影占约 33.2%。Norm 参数量很小，几乎不影响总量。

| 整体组件 | 参数量 |
| --- | ---: |
| Token Embedding | 131,072,000 |
| 32 个 Block | 6,476,267,520 |
| 最终 RMSNorm | 4,096 |
| LM Head | 131,072,000 |
| 合计 | **6,738,415,616** |

<p align="center">
  <img src="/img/in-post/ai-infra-cost-parameters.svg" alt="三十二个 Decoder Block 的平铺图及全模型参数比例条，展示 Block 占约百分之九十六点一，单 Block 中 FFN 约占三分之二" style="max-width: 100%;">
</p>

总量约 6.74B，通常称为 7B。LLaMA-2 使用独立的 Embedding 和 LM Head；如果某个模型选择共享这两者，才会少计一份 `Vd`。

只存一份 BF16/FP16 权重，每参数 2 字节，以上参数对应约 13.48 GB，即 12.55 GiB。实际推理还需要 KV Cache、临时激活和运行时空间；训练还会保存梯度、优化器状态和反向所需的激活。这些内容不属于参数量，却同样消耗显存。

## 4. 从参数量推到前向计算量

将开头的 `2TP` 应用到各个线性层，就能得到常见的“每 token 约 2P FLOPs”。为明确统计口径，下面用 `P_linear` 表示每个 token 都会执行的投影矩阵的参数量。

对上述模型，如果所有位置都计算词表 logits：

<p align="center">$P_{linear}=L(4d^2+3dd_{ff})+Vd$</p>

Embedding 是查表，不执行整张表的矩阵乘，因此不计入这一项。Norm、激活和 softmax 的操作也暂时省略。若 batch 为 `B`、长度为 `S`，则 `T=BS`：

<p align="center">$F_{linear}\approx2BS P_{linear}$</p>

对示例模型，`P_linear=6,607,077,376`，每个 token 的主要线性层计算约为 13.21 GFLOPs。直接用总参数量估算会得到约 13.48 GFLOPs，两者接近，但计数含义不同。

推理 Prefill 如果只需要最后一个位置的 logits，LM Head 也可以只计算最后一个位置，此时不必为所有 prompt token 都承担这一项。

## 5. Attention 的另一部分随序列长度平方增长

Attention 除了投影权重，还要计算 token 之间的匹配与加权汇总。一个样本、一个 head 中：

<p align="center">$QK^T:[S,d_k]\times[d_k,S]\rightarrow[S,S]$</p>

分数经过 softmax 得到 `A`，再与 Value 相乘：

<p align="center">$AV:[S,S]\times[S,d_k]\rightarrow[S,d_k]$</p>

两次矩阵乘各约 `2S²d_k` FLOPs。乘以 batch、头数和层数，在 `h d_k=d` 时得到：

<p align="center">$F_{attention}\approx4BLS^2d$</p>

这部分没有新的可学习矩阵，却需要实际运算，因此不能只靠参数量计数。整段前向的主要计算量是：

<p align="center">$F_{forward}\approx2BS P_{linear}+4BLS^2d$</p>

这里按完整 Attention 方阵估算。因果 Attention 的有效位置约占一半，能跳过未来位置的 kernel 会减少这部分算术量；softmax 等操作仍未计入。

<p align="center">
  <img src="/img/in-post/ai-infra-cost-context.svg" alt="序列长度翻倍时，线性层输入从四行变八行，Attention 配对矩阵从四乘四变八乘八；Decode 只计算一行 Query 与全部缓存位置的匹配" style="max-width: 100%;">
</p>

序列翻倍时，线性层多处理一倍 token，计算量约翻倍；Attention 的 Query 和 Key 都变多，配对数量变成四倍。FlashAttention 可以减少中间结果的显存往返，但完整注意力仍需覆盖允许的 Query-Key 配对。

## 6. 训练中的 6P 与推理中的 2P

一个线性层 `Y=XW`，训练时主要执行三次矩阵乘：

```text
前向：       Y = XW
输入梯度：  dX = dY Wᵀ
权重梯度：  dW = Xᵀ dY
```

三者的主要计算量相近，所以每 token 的线性层前向约 `2P`，反向约 `4P`，合计约 `6P`。在稠密模型的粗略估算中，将 `P` 取模型参数规模，训练总共处理 `T_train` 个 token：

<p align="center">$F_{train}\approx6P T_{train}$</p>

每 token 是 `6P`，长度为 `S` 的单条序列是 `6PS`。长上下文 Attention、激活重计算以及其他算子需要按实际方案另行考虑。

Prefill 批量处理 prompt，可以采用上一节的序列公式。带 KV Cache 的单请求 Decode，每次只为一个新 token 执行投影和 FFN，但当前 Query 仍需要访问长度为 `C` 的缓存：

```text
QKᵀ：[1, d_k] × [d_k, C] → [1, C]
AV： [1, C]   × [C, d_k] → [1, d_k]
```

因此，在这个 MHA 模型中，一步 Decode 的主要计算量为：

<p align="center">$F_{decode}\approx2P_{linear}+4LCd$</p>

`C` 包含当前可见的全部位置。历史 token 的投影不必重复计算，但访问历史 K/V 并做注意力汇总仍有成本。这个公式描述单步；生成许多 token 时，需要把各步不断增长的 `C` 对应的开销累加。

## 7. 从 FLOPs 走到实际时间

计算量除以吞吐率，可以估算算术时间，但要同时考虑数据搬运。用约 6.7B 的参数规模做粗略演示，单 token 的 `2P` 约为 13.4 GFLOPs。假设硬件能达到 312 TFLOP/s：

<p align="center">$t_{compute}=\frac{13.4\times10^9}{312\times10^{12}}\approx0.043\text{ ms}$</p>

如果约 13.4 GB 的半精度权重需要从 HBM 加载，按 2 TB/s 的带宽计算：

<p align="center">$t_{memory}=\frac{13.4\text{ GB}}{2000\text{ GB/s}}=6.7\text{ ms}$</p>

这是一组简化的硬件上限假设。两个数不是实测时间，也不应直接相加；计算与搬运可能重叠，理想下界要看两者中更大的约束。这里仅权重搬运的时间估计，就已经远高于峰值算术时间。

<p align="center">
  <img src="/img/in-post/ai-infra-cost-time.svg" alt="同一时间刻度下，峰值算术时间估计零点零四三毫秒与权重搬运时间估计六点七毫秒的对比，均为简化假设而非实测" style="max-width: 100%;">
</p>

低 batch Decode 中，权重难以被少量 token 充分复用，因此常受到显存带宽限制；批量增大后，同一份权重可以服务更多输入行，计算利用率可能改善。KV Cache、通信、kernel 启动和实际访存效率还会进一步影响延迟。

手算时，可以先用矩阵形状得到参数量，再乘 token 数估算线性层 FLOPs，单独补上 Attention 的序列项，最后对照算力和带宽。这几步把模型规模、上下文长度和硬件代价放在了同一套可检查的计算中。

## 参考资料

- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- [Meta Llama 模型实现](https://github.com/meta-llama/llama/blob/main/llama/model.py)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
