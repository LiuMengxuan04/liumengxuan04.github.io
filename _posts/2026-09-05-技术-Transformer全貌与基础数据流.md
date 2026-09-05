---
layout:     post
title:      "Transformer 全貌与基础数据流"
subtitle:   "从 Encoder-Decoder、Pre/Post-Norm 到 Prefill 与 Decode"
date:       2026-09-05 20:00:00 +0800
author:     "Liu Mengxuan"
mathjax:    true
header-img: "img/post-bg-miui6.jpg"
categories: [技术]
tags:       [技术, AI Infra, Transformer, Encoder, Decoder, LayerNorm, RMSNorm, Prefill, Decode]
---

理解 Transformer，不能只记住 Attention 的公式。对 AI Infra 来说，更重要的是知道一段输入怎样依次经过 Embedding、Attention、FFN、残差和归一化，每一步产生什么形状，以及这些计算为什么会对应到 GEMM、Kernel 融合、KV Cache 和并行策略。

本文从原始 Transformer 的 Encoder-Decoder 结构出发，重点厘清几个容易混淆的问题：Encoder 和 Decoder 分别做什么；Pre-Norm 与 Post-Norm 如何判断；LayerNorm、RMSNorm 和 BatchNorm 有什么区别；激活函数藏在 Block 的什么位置；Prefill 所说的“并行处理整个 Prompt”又究竟并行了什么。

## 1. 为什么 AI Infra 必须先看懂 Transformer？

AI Infra 的任务，是把模型里的计算高效地映射到 GPU、集群和推理服务上。Transformer 的每个组件都对应一组具体的系统问题：

| Transformer 组件 | 主要计算或数据 | 对应的 AI Infra 问题 |
| --- | --- | --- |
| QKV 与输出投影 | 大型 GEMM | Tensor Core、量化、张量并行 |
| Attention | $QK^T$、Softmax、$PV$ | FlashAttention、分块与融合 |
| FFN | 两个或三个大型权重矩阵 | GEMM、量化、MoE、张量并行 |
| Residual + Norm | 归约和逐元素运算 | Kernel 融合、HBM 读写 |
| Block 堆叠 | 多层顺序执行 | 流水线并行、激活重计算 |
| K、V | 随上下文增长的数据 | KV Cache、PagedAttention、KV 量化 |
| 自回归生成 | 每步产生一个 token | Continuous Batching、推测解码 |

Transformer 取代 RNN 的一个关键原因，也是它适合现代硬件的原因。RNN 处理长度为 $S$ 的序列时，第 $s$ 步依赖第 $s-1$ 步，序列方向难以并行。Self-Attention 把 token 间的交互组织成矩阵乘法，同一层中的多个 token 位置可以一起计算。

这不等于 Transformer 的所有计算都能同时完成。第 2 层仍然必须等待第 1 层的输出，自回归 Decode 的下一 token 仍然必须等待当前 token 生成。它消除的是很多**层内的序列依赖**，并没有消除层与层、生成步与生成步之间的依赖。

## 2. 先统一形状

本文使用以下符号：

| 符号 | 含义 |
| --- | --- |
| $B$ | batch size，一次处理的序列数量 |
| $S_{src}$ | Encoder 源序列的 token 数量 |
| $S_{tgt}$ | Decoder 目标序列的 token 数量 |
| $H$ | hidden size，也写作 $d_{model}$ |
| $N_h$ | Attention head 数量 |
| $D_h$ | 每个 head 的维度，通常 $D_h=H/N_h$ |
| $V$ | 词表大小 |
| $L$ | Transformer Block 的层数 |

一段自然语言首先经过 Tokenizer：

```text
“我正在学习 Transformer”
        ↓ Tokenizer
[token_id₀, token_id₁, ..., token_idₛ₋₁]
```

Token ID 张量的形状是：

<p align="center">$[B,S]$</p>

Embedding 表的形状是：

<p align="center">$E\in\mathbb R^{V\times H}$</p>

每个 token ID 从表中查出一行 $H$ 维向量，于是得到：

<p align="center">$X\in\mathbb R^{B\times S\times H}$</p>

这里的 $H$ 不是网络层数，而是**每个 token 用多少个数表示**。$L$ 才表示 Block 堆叠的层数。

原始 Transformer 还会把位置编码加到 Token Embedding 上：

<p align="center">$X=\operatorname{TokenEmbedding}(ids)+\operatorname{PositionEncoding}(positions)$</p>

位置编码不会改变形状，输入仍是 `[B,S,H]`。

## 3. 原始 Transformer 的全貌

原始 Transformer 是为机器翻译一类 Seq2Seq 任务设计的。它由两部分组成：

```text
源序列 → Encoder → 一组上下文化表示
                         ↓
目标序列 → Decoder ──────┘ → 下一个目标 token
```

Encoder 读取完整源序列，让每个源 token 与其他源 token 交换信息。Decoder 一边查看已经出现的目标 token，一边通过 Cross-Attention 查询 Encoder 的输出，然后预测下一个目标 token。

<p align="center">
  <img src="/img/in-post/ai-infra-transformer-encoder-decoder-overview.svg" alt="原始 Transformer Encoder-Decoder 架构及三种 Attention 的数据来源" style="max-width: 100%;">
</p>

原始架构中有三种 Attention。它们底层都使用 Scaled Dot-Product Attention，区别在于 Q、K、V 从哪里来，以及允许看哪些位置。

| 类型 | Q 来源 | K、V 来源 | 可见范围 |
| --- | --- | --- | --- |
| Encoder Self-Attention | Encoder 当前表示 | Encoder 当前表示 | 完整源序列，双向可见 |
| Masked Decoder Self-Attention | Decoder 当前表示 | Decoder 当前表示 | 当前及更早的目标位置 |
| Cross-Attention | Decoder 当前表示 | Encoder 最终输出 | 完整源序列 |

Cross-Attention 的分数矩阵通常不是方阵。若目标长度为 $S_{tgt}$，源长度为 $S_{src}$：

<p align="center">$Q_{dec}\in\mathbb R^{S_{tgt}\times D_h},\quad K_{enc}\in\mathbb R^{S_{src}\times D_h}$</p>

因此：

<p align="center">$Q_{dec}K_{enc}^T\in\mathbb R^{S_{tgt}\times S_{src}}$</p>

每一行表示一个目标位置正在查询哪些源位置。

## 4. 一个 Encoder 层究竟做了什么？

一个 Encoder 层包含两个主要子层：

```text
Multi-Head Self-Attention：不同 token 之间交换信息
Position-wise FFN：        每个 token 独立加工自己的特征
```

除此之外，每个子层周围还有一条残差连接和一个归一化操作。

### 4.1 Encoder Self-Attention 是双向的

假设源序列有 5 个 token。Encoder 不需要预测未来，因此所有位置可以互相查看：

```text
     t0  t1  t2  t3  t4
t0 [  1   1   1   1   1 ]
t1 [  1   1   1   1   1 ]
t2 [  1   1   1   1   1 ]
t3 [  1   1   1   1   1 ]
t4 [  1   1   1   1   1 ]
```

因此，即使只经过一层，开头 token 的新表示也可以直接吸收末尾 token 的信息。输出仍保留每一个 token，并不是把整句话压成一个向量：

<p align="center">$[B,S_{src},H]\rightarrow[B,S_{src},H]$</p>

### 4.2 残差连接为什么要求形状不变？

残差连接写作：

<p align="center">$y=x+F(x)$</p>

逐元素相加要求两个分支形状一致：

```text
x：    [B, S, H]
F(x)： [B, S, H]
结果： [B, S, H]
```

所以 Attention 最后的输出投影会把多头结果重新映射回 $H$ 维，FFN 在中间升维后也必须降回 $H$。每个 Block 的输入输出形状相同，Block 才能连续堆叠。

## 5. 激活函数藏在 FFN 里面

很多 Transformer 结构图只画一个 `Feed-Forward Network` 方框，激活函数因此看起来像消失了。原始 Transformer 的 FFN 实际是：

<p align="center">$\operatorname{FFN}(x)=W_2\operatorname{ReLU}(W_1x+b_1)+b_2$</p>

形状通常经历：

```text
[B,S,H]
   ↓ Linear 1
[B,S,H_ff]
   ↓ ReLU
[B,S,H_ff]
   ↓ Linear 2
[B,S,H]
```

原始论文取 $H=512$、$H_{ff}=2048$，即中间宽度是隐藏维的 4 倍。

FFN 被称为 position-wise，是因为每个 token 都独立执行同一套参数：

```text
token 0 的 H 维向量 → FFN → 新的 H 维向量
token 1 的 H 维向量 → FFN → 新的 H 维向量
token 2 的 H 维向量 → FFN → 新的 H 维向量
```

FFN 内部不混合 token；token 间的信息交换已经由 Attention 完成。

激活函数不可省略。如果没有 ReLU、GELU 或 SiLU：

<p align="center">$W_2(W_1x)=(W_2W_1)x$</p>

两个线性层就可以合并成一个线性层，FFN 无法表达更复杂的非线性关系。

不同代际的模型常用不同 FFN：

| 模型或架构 | 常见 FFN 激活 |
| --- | --- |
| 原始 Transformer | ReLU |
| BERT、GPT-2 等 | GELU |
| LLaMA 等现代 LLM | SwiGLU，其中包含 SiLU |

SwiGLU 可以写成：

<p align="center">$\operatorname{SwiGLU}(x)=\left[\operatorname{SiLU}(xW_g)\odot(xW_u)\right]W_d$</p>

它用 gate 和 up 两个投影产生中间表示，逐元素相乘后再由 down 投影降回 $H$ 维。

Attention 中的 Softmax 也是非线性操作，但讨论 Transformer 的“激活函数”时，通常特指 FFN 中的 ReLU、GELU、SiLU 或门控激活。

## 6. LayerNorm、RMSNorm 与 BatchNorm

LayerNorm 的名字容易造成误会：它不是沿网络的多层 $L$ 计算，而是对**当前层中每个 token 的隐藏维 $H$**做归一化。

给定：

<p align="center">$X\in\mathbb R^{B\times S\times H}$</p>

LayerNorm 对每个 `(b,s)` 位置分别计算：

<p align="center">$\mu_{b,s}=\frac1H\sum_{h=1}^{H}X_{b,s,h}$</p>

<p align="center">$\sigma_{b,s}^2=\frac1H\sum_{h=1}^{H}(X_{b,s,h}-\mu_{b,s})^2$</p>

再归一化并使用可学习的 $\gamma$、$\beta$ 调整：

<p align="center">$\operatorname{LayerNorm}(x_h)=\gamma_h\frac{x_h-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_h$</p>

RMSNorm 使用同样的归约方向，但不减均值：

<p align="center">$\operatorname{RMSNorm}(x_h)=\gamma_h\frac{x_h}{\sqrt{\frac1H\sum_i x_i^2+\epsilon}}$</p>

例如 $x=[3,4]$：

<p align="center">$\operatorname{RMS}(x)=\sqrt{\frac{3^2+4^2}{2}}\approx3.536$</p>

忽略 $\gamma$，输出约为 `[0.849, 1.131]`。它控制整体幅度，但不会强制输出均值为 0。

<p align="center">
  <img src="/img/in-post/ai-infra-transformer-layernorm-vs-batchnorm.svg" alt="LayerNorm、RMSNorm 与 BatchNorm 在 B、S、H 三个维度上的归约方向" style="max-width: 100%;">
</p>

### 6.1 为什么 Transformer 不常用 BatchNorm？

BatchNorm 的典型思想，是对同一个特征跨样本统计。应用到序列张量时，具体实现可能跨 $B$，也可能同时把有效序列位置纳入统计。它会带来几个不适合 Transformer 的性质：

1. 训练时统计量依赖当前 batch，单卡 micro-batch 很小时容易不稳定；
2. 不同序列长度和 padding 会让统计与 mask 处理更复杂；
3. 多卡训练若需要全局统计量，还会引入同步通信；
4. 训练通常使用当前 batch 统计量，推理则使用 running statistics，存在两套行为；
5. 自回归推理经常出现 `B=1、S=1`，逐 token 的 LayerNorm/RMSNorm 更自然。

严格来说，BatchNorm 在推理模式下使用固定的 running statistics，因此同批次的其他请求不会再改变当前请求的输出。问题主要发生在训练阶段的 batch 依赖，以及训练和推理统计方式的差异。

LayerNorm 和 RMSNorm 都只读取当前 token 的 $H$ 个值，不依赖其他句子或 token，也不需要跨卡同步 batch 统计量。这正符合 Transformer 的执行方式。

### 6.2 LayerNorm 和 RMSNorm 的区别

LayerNorm 和 RMSNorm 都对每个 token 独立沿隐藏维 $H$ 归一化，也都保持输入输出形状不变：

```text
输入： [B, S, H]
输出： [B, S, H]
```

两者的区别在于如何定义“当前向量的尺度”。LayerNorm 先减去均值，再除以标准差；RMSNorm 不减均值，直接除以均方根。

取一个 token 的隐藏向量：

<p align="center">$x=[1,2,3]$</p>

LayerNorm 先得到均值 $\mu=2$ 和标准差 $\sigma=\sqrt{2/3}$。忽略 $\epsilon$、$\gamma$ 和 $\beta$ 后：

<p align="center">$\operatorname{LayerNorm}(x)\approx[-1.225,0,1.225]$</p>

输出均值为 0，表示向量的整体偏移被去除了。

RMSNorm 计算：

<p align="center">$\operatorname{RMS}(x)=\sqrt{\frac{1^2+2^2+3^2}{3}}=\sqrt{\frac{14}{3}}\approx2.160$</p>

所以忽略 $\epsilon$ 和 $\gamma$ 后：

<p align="center">$\operatorname{RMSNorm}(x)\approx[0.463,0.926,1.389]$</p>

它只把整体幅度拉回稳定范围，输出均值不必为 0。

| 对比项 | LayerNorm | RMSNorm |
| --- | --- | --- |
| 归约方向 | 每个 token 沿 $H$ | 每个 token 沿 $H$ |
| 是否减均值 | 是 | 否 |
| 缩放依据 | 标准差 | 均方根 |
| 常见可学习参数 | $\gamma$、$\beta$ | 通常只有 $\gamma$ |
| 典型模型 | 原始 Transformer、BERT、GPT-2 | LLaMA、Qwen 等现代 LLM |

RMSNorm 少了求均值和减均值的步骤，计算形式更简单；实际速度收益仍取决于 Kernel 融合和显存访问。模型通常在一个 Norm 位置选择其中一种，两者不会按顺序连续执行。

## 7. Post-Norm 和 Pre-Norm 到底怎么判断？

判断标准只有一个：**对同一个子层，Norm 位于子层之前还是残差相加之后。**

原始 Transformer 使用 Post-Norm。对 Attention 和 FFN 两个子层分别写为：

<p align="center">$h=\operatorname{LayerNorm}\left(x+\operatorname{Attention}(x)\right)$</p>

<p align="center">$y=\operatorname{LayerNorm}\left(h+\operatorname{FFN}(h)\right)$</p>

现代大模型常用 Pre-Norm：

<p align="center">$h=x+\operatorname{Attention}(\operatorname{Norm}(x))$</p>

<p align="center">$y=h+\operatorname{FFN}(\operatorname{Norm}(h))$</p>

<p align="center">
  <img src="/img/in-post/ai-infra-transformer-pre-post-norm.svg" alt="原始 Transformer Post-Norm 与现代大模型 Pre-Norm 的完整对比" style="max-width: 100%;">
</p>

### 7.1 为什么 Norm 在 FFN 前仍可能是 Post-Norm？

考虑下面这条 Encoder 数据流：

```text
x → Attention → +x → LayerNorm → FFN → +残差 → LayerNorm
```

第一个 LayerNorm 虽然位于 FFN 前面，但它处理的是 `x + Attention(x)`，因此属于 **Attention 子层的 Post-Norm**。它的输出同时成为下一个 FFN 子层的输入，不能只根据图上的上下位置把它判断为 FFN 的 Pre-Norm。

要逐个子层判断：

```text
Attention 子层：Attention → Add → Norm，所以是 Post-Norm
FFN 子层：      FFN       → Add → Norm，所以也是 Post-Norm
```

所以这条完整数据流是 Post-Norm。判断时应该把每个 Norm 与它前面的残差加法或后面的子层配对，而不是只看 Norm 是否恰好画在某个方框上方。

### 7.2 Pre-Norm 为什么常用于深层模型？

Pre-Norm 中，残差主路径可以直接从 $x$ 走到加法节点，不需要穿过 Norm：

<p align="center">$y=x+F(\operatorname{Norm}(x))$</p>

反向传播时存在一条更直接的梯度路径，因此深层网络通常更容易优化。采用 Pre-Norm 的模型通常还会在所有 Block 之后增加一次 Final Norm，再进入输出头。

“Residual Add + RMSNorm 融合”也不与 Pre-Norm 冲突。一个子层结束时先形成：

<p align="center">$h=x+F(\operatorname{Norm}(x))$</p>

紧接着，下一个子层要计算 `RMSNorm(h)`。CUDA Kernel 可以在产生 $h$ 时同时完成下一次 Norm，减少把 $h$ 写回 HBM 后再读一遍的开销。

## 8. 从原始架构到现代 Decoder-only

原始 Transformer 后来形成三条常见路线：

| 架构 | 保留的主体 | Attention 可见性 | 常见用途 |
| --- | --- | --- | --- |
| Encoder-only | Encoder | 双向 | 分类、表示学习、信息抽取 |
| Encoder-Decoder | Encoder + Decoder + Cross-Attention | 源端双向，目标端因果 | 翻译、摘要、Seq2Seq |
| Decoder-only | 因果 Decoder Block，不含 Cross-Attention | 只能看当前和历史 token | GPT、LLaMA、Qwen 等生成模型 |

“Decoder-only”并不是把原始 Decoder 原封不动留下来：没有 Encoder 后，连接 Encoder 的 Cross-Attention 也随之删除。一个现代 Pre-Norm Decoder Block 可以写成：

<p align="center">$u=x+\operatorname{CausalAttention}(\operatorname{RMSNorm}(x))$</p>

<p align="center">$y=u+\operatorname{SwiGLU}(\operatorname{RMSNorm}(u))$</p>

它只保留两大子层：因果 Self-Attention 和 FFN。

## 9. Prefill：并行处理整个 Prompt

假设模型使用：

```text
B = 1
P = 10 个 prompt token
H = 4096
L = 32 层 Decoder Block
```

完整张量形状应写为：

```text
[1, 10, 4096]
```

有些实现会把这个张量简写成 `(10,4096)`：它可能省略了值为 1 的 batch 维，也可能已经把 `B×S` 展平。这里的 `10` 是序列长度 $S$，不是 batch size。在线性层内部常把前两维展平：

<p align="center">$[B,S,H]\rightarrow[BS,H]$</p>

因此当 `B=1、S=10` 时，GEMM 的行数 $M=BS=10$。

在 Prefill 阶段，同一层内 10 个 token 的 QKV 投影、FFN 和各 Query 位置的 Attention 可以批量并行计算。对于 Decoder-only 模型，Attention 仍然带因果 Mask：位置 3 虽然与其他位置同时计算，但只能使用位置 0 到 3 的信息。

一层中的主要形状是：

```text
输入：          [1, 10, 4096]
RMSNorm：       [1, 10, 4096]
Q/K/V：         [1, Nh, 10, Dh]
Attention 分数：[1, Nh, 10, 10]
Attention 输出：[1, 10, 4096]
FFN 输出：      [1, 10, 4096]
```

每一层还会把这 10 个 token 的 K、V 写入 KV Cache，供后续 Decode 使用。32 层之间仍然顺序执行：第 2 层必须拿到第 1 层输出，第 32 层必须等待前 31 层完成。

全部层完成后，模型根据最后一个 prompt 位置的 logits 选出第一个输出 token。用户从请求发出到看到第一个 token 的延迟通常称为 TTFT（Time To First Token）。

## 10. Decode：每次只处理一个新 token

产生第一个输出 token 后，模型进入 Decode。每一步的新输入形状是：

```text
[B, 1, H]
```

对于每一层，只需要为这个新 token 计算新的 Q、K、V：

```text
新 Q：查询历史所有 K
历史 K/V：直接从 KV Cache 读取
新 K/V：计算后追加到 KV Cache
历史 Q：未来不会再次作为当前 Query，因此无需缓存
```

如果当前上下文长度为 $T$：

```text
Q 长度：1
K/V 长度：T
Attention 分数：[B, Nh, 1, T]
```

<p align="center">
  <img src="/img/in-post/ai-infra-transformer-prefill-vs-decode.svg" alt="LLM 推理中 Prefill 与 Decode 的输入形状、KV Cache 与执行差异" style="max-width: 100%;">
</p>

Prefill 和 Decode 使用同一组模型权重，却表现出不同的硬件特征：

| 阶段 | 一次处理的 Query 数量 | 典型计算形态 | 关键指标 |
| --- | --- | --- | --- |
| Prefill | 整个 prompt | 较大的 GEMM，通常有更高算术强度 | TTFT |
| Decode | 每个请求每步 1 个新 token | 瘦矩阵或矩阵向量计算，并反复读取权重与 KV Cache | TPOT、吞吐 |

Prefill 经常更容易利用 Tensor Core；Decode 在低 batch 下经常受 HBM 带宽限制。这里不能机械地给二者永久贴上 Compute Bound 或 Memory Bound 标签，因为 prompt 长度、并发 batch、量化方式、硬件和 Kernel 实现都会改变瓶颈。

## 11. 把数据流映射回 AI Infra

现在可以把一次 Decoder-only 推理请求串起来：

```text
文本
  ↓ CPU Tokenizer
Token IDs [B,S]
  ↓ Embedding
Hidden States [B,S,H]
  ↓
重复 L 次：
  RMSNorm
  → QKV GEMM
  → RoPE
  → Causal Attention
  → 输出投影
  → Residual Add
  → RMSNorm
  → SwiGLU FFN
  → Residual Add
  ↓
Final RMSNorm
  ↓ LM Head
Logits [B,S,V]
  ↓ 采样
下一个 token
```

其中每个位置都能找到对应的工程优化：

```text
QKV / FFN GEMM       → Tensor Core、量化、张量并行
RoPE + Attention     → Kernel 融合、FlashAttention
Residual + RMSNorm  → Fused Kernel，减少 HBM 往返
多层 Block           → 流水线并行、激活重计算
KV Cache             → PagedAttention、GQA/MQA、KV 量化
逐 token Decode      → Continuous Batching、推测解码
```

学习后续技术时，可以固定问三个问题：它作用于 Transformer 的哪个位置；这个位置正在处理什么形状；当前瓶颈来自算力、显存容量、显存带宽，还是跨设备通信。只要这三点能回答，AI Infra 里的优化名词就会重新落回一条具体的数据流。
