---
layout:     post
title:      "Transformer 架构快速入门"
subtitle:   "从 Self-Attention、多头机制与 RoPE 到 Decoder Block"
date:       2026-09-03 20:00:00 +0800
author:     "Liu Mengxuan"
mathjax:    true
header-img: "img/post-bg-miui6.jpg"
categories: [技术]
tags:       [技术, AI Infra, Transformer, Self-Attention, RoPE, GQA, Decoder Block]
---

学习 Transformer 时，我最开始能记住 Attention 的公式，却说不清公式里的每个张量究竟来自哪里，也不清楚“多头”为什么在代码里经常只是一次大矩阵乘法。继续往下看 RoPE 和 Decoder Block 后，新的问题又出现了：一个 token 为什么要旋转多组特征？相对位置到底发生在哪两个对象之间？多个 head 拼接以后为什么还要经过输出投影？

后来我发现，这些问题其实属于同一条数据流：一段文本先变成 token 序列，每个 token 被表示成隐藏向量；Attention 负责让 token 之间交换信息，FFN 负责让每个 token 独立加工信息，RoPE、Norm 和残差连接则保证顺序信息与训练稳定性。

```text
文本
  ↓ Tokenizer 与 Embedding
[B, S, H]
  ↓ RMSNorm
Masked Multi-Head Self-Attention
  ↓ 残差连接
[B, S, H]
  ↓ RMSNorm
SwiGLU FFN
  ↓ 残差连接
[B, S, H]
```

本文从张量形状出发，把 Self-Attention、多头工程实现、MHA/GQA/MQA、RoPE 和现代 Decoder Block 串成一条完整链路。

## 1. 先统一符号

Transformer 中同一个字母有时会被不同资料赋予不同含义。为了避免把序列长度、层数、Value 和词表混在一起，本文统一使用下面的符号：

| 符号 | 含义 |
| --- | --- |
| $B$ | batch size，一次处理的序列数量 |
| $S$ | sequence length，一条序列中的 token 数量 |
| $H$ | hidden size，也写作 $d_{model}$，每个 token 的隐藏向量维度 |
| $N_h$ | Query head 的数量 |
| $N_{kv}$ | Key/Value head 的数量 |
| $D_h$ | 每个 Attention head 的维度，通常 $D_h=H/N_h$ |
| $H_{ff}$ | FFN 的中间维度 |
| $L$ | Decoder Block 的层数 |
| $|\mathcal V|$ | tokenizer 的词表大小 |
| $V_{val}$ | Attention 中的 Value 张量，避免与词表混淆 |

这里最容易混淆的是 $S$、$H$ 和 $|\mathcal V|$：

```text
S：当前这段输入实际有多少个 token
H：模型用多少个浮点数表示一个 token
|V|：整个词表中一共有多少种 token
```

它们描述的是三个不同的问题。

## 2. 从文本到 Transformer 输入

模型的输入最初是一段文本，例如：

```text
我喜欢学习 AI Infra
```

Tokenizer 会把它转换成一串 token ID：

```text
文本
  ↓ Tokenizer
[id₀, id₁, id₂, ..., idₛ₋₁]
```

如果一共得到 $S$ 个 token，那么 token ID 张量的形状是：

<p align="center">$\text{token\_ids}\in\mathbb{N}^{B\times S}$</p>

模型内部维护一个 Embedding 参数矩阵：

<p align="center">$E\in\mathbb{R}^{|\mathcal V|\times H}$</p>

其中每一行保存一个词表 token 的 $H$ 维表示。对于第 $s$ 个输入 token，查表得到：

<p align="center">$x_s=E[\text{id}_s]\in\mathbb{R}^{H}$</p>

把当前输入中的 $S$ 个 token 向量按顺序排列起来，得到：

<p align="center">$X\in\mathbb{R}^{B\times S\times H}$</p>

所以 Transformer 的输入是 $[B,S,H]$，而不是 $[B,|\mathcal V|,H]$。$|\mathcal V|$ 只表示整个 Embedding 表有多少行；当前句子只从中查出了 $S$ 行。

例如：

```text
词表大小 |V| = 32000
当前序列长度 S = 10
隐藏维度 H = 4096
batch size B = 1

Embedding 参数： [32000, 4096]
Token IDs：       [1, 10]
查表后的 X：      [1, 10, 4096]
```

<p align="center">
  <img src="/img/in-post/transformer-raw-text-to-embedding.png" alt="文本经过 Tokenizer 与 Embedding 变成隐藏向量" style="max-width: 100%;">
</p>

## 3. Transformer 的三种常见架构

原始 Transformer 由 Encoder 和 Decoder 两部分组成：

```text
源序列 → Encoder → 上下文表示
                       ↓
目标序列 → Decoder + Cross-Attention → 输出
```

后来逐渐形成三类常见架构：

| 架构 | 注意力方式 | 代表模型 | 常见任务 |
| --- | --- | --- | --- |
| Encoder-only | 双向 Self-Attention | BERT | 分类、表示学习、信息抽取 |
| Encoder-Decoder | Encoder 双向，Decoder 因果注意力并包含 Cross-Attention | T5、原始 Transformer | 翻译、摘要、Seq2Seq |
| Decoder-only | 因果 Self-Attention | GPT、LLaMA、Qwen | 自回归文本生成 |

当前大语言模型通常采用 Decoder-only：删除原始 Decoder 中连接 Encoder 的 Cross-Attention，只保留因果 Self-Attention 和 FFN，并堆叠很多层相同形状的 Block。

需要注意，Decoder-only 是否使用 Pre-Norm，是另一个独立的架构选择。删除 Cross-Attention 和调整 Norm 位置不是同一件事。

本文后面主要讨论现代大模型常见的 Decoder-only、Pre-Norm 结构。

## 4. Self-Attention 在做什么？

Self-Attention 的目标是：让序列中的每个 token 根据相关程度，从其他可见 token 中汇总信息。

可以把 Q、K、V 理解成一次检索：

- Query：当前 token 想查什么；
- Key：每个 token 用什么特征接受匹配；
- Value：匹配后真正取回的内容。

这只是帮助理解的比喻。从计算上看，Q、K、V 是同一个输入经过三组可学习线性投影得到的结果。

### 4.1 线性投影生成 Q、K、V

暂时忽略多头细节，输入为：

<p align="center">$X\in\mathbb{R}^{B\times S\times H}$</p>

经过三组参数矩阵：

<p align="center">$Q=XW_Q,\qquad K=XW_K,\qquad V_{val}=XW_V$</p>

在标准多头注意力中，三个大投影的输出宽度通常都是 $H$：

<p align="center">$W_Q,W_K,W_V\in\mathbb{R}^{H\times H}$</p>

于是：

<p align="center">$Q,K,V_{val}\in\mathbb{R}^{B\times S\times H}$</p>

这里并没有让 token 数量发生变化。每个 token 仍然对应一行，只是从原来的隐藏表示变成了用于查询、匹配和传递内容的三种表示。

### 4.2 计算匹配分数

对于单个 head，Q 和 K 的形状是：

<p align="center">$Q_i,K_i\in\mathbb{R}^{B\times S\times D_h}$</p>

计算：

<p align="center">$S_i=Q_iK_i^\top$</p>

在 batch 中，每个样本独立计算，形状变化是：

```text
[B, S, Dh] @ [B, Dh, S] -> [B, S, S]
```

分数矩阵中的元素 $S_i[b,m,n]$ 表示：

> 第 $b$ 条序列中，位置 $m$ 的 Query 与位置 $n$ 的 Key 在第 $i$ 个 head 中有多匹配。

因此每一行对应一个 Query token，每一列对应一个 Key token。

### 4.3 为什么要除以 $\sqrt{D_h}$？

缩放后的分数是：

<p align="center">$\widetilde S_i=\dfrac{Q_iK_i^\top}{\sqrt{D_h}}$</p>

一个 Q 和 K 的点积包含 $D_h$ 个乘积的累加：

<p align="center">$q^\top k=\sum_{r=1}^{D_h}q_rk_r$</p>

如果各分量大致独立、均值为 0、方差约为 1，那么这个和的方差大约随 $D_h$ 增长，标准差大约是 $\sqrt{D_h}$。维度越大，未经缩放的分数越容易出现较大的绝对值。

Softmax 中包含指数运算。分数差距过大时，概率会过早接近 one-hot，很多位置的梯度变得很小。除以 $\sqrt{D_h}$ 后，不同 head dimension 下的分数尺度更稳定。

这里除的是每个 head 的维度 $D_h$，而不是整个隐藏维度 $H$。

### 4.4 因果 Mask：不能看到未来答案

Decoder-only 模型按顺序预测下一个 token，所以位置 $m$ 不能访问位置 $m$ 之后的 token。

例如：

```text
输入：我 喜欢 学习 AI

“我”   只能看：我
“喜欢” 可以看：我、喜欢
“学习” 可以看：我、喜欢、学习
“AI”   可以看：我、喜欢、学习、AI
```

将未来位置的分数设为负无穷：

<p align="center">$M_{m,n}=-\infty,\qquad n>m$</p>

再做 Softmax：

<p align="center">$A_i=\operatorname{softmax}\left(\dfrac{Q_iK_i^\top}{\sqrt{D_h}}+M\right)$</p>

Softmax 沿 Key 的位置维，也就是最后一维执行。由于 $e^{-\infty}=0$，未来位置的注意力权重变成 0。

### 4.5 用权重汇总 Value

得到注意力权重后：

<p align="center">$O_i=A_iV_i$</p>

形状变化是：

```text
[B, S, S] @ [B, S, Dh] -> [B, S, Dh]
```

对于第 $m$ 个 Query token：

<p align="center">$o_m=\sum_{n=1}^{S}A_{m,n}v_n$</p>

它得到的不是某一个 Value，而是所有可见 Value 的加权和。这就是“每个 token 聚合上下文信息”的具体含义。

一个 head 的完整公式是：

<p align="center">$\operatorname{head}_i=\operatorname{softmax}\left(\dfrac{Q_iK_i^\top}{\sqrt{D_h}}+M\right)V_i$</p>

## 5. 多头注意力到底“多”在哪里？

假设：

```text
H  = 4096
Nh = 32
Dh = 128
```

它们满足：

<p align="center">$H=N_hD_h=32\times128=4096$</p>

同一个 token 在 32 个 head 中各有一份 128 维的 Q、K、V 表示。所有 head 都处理完整序列，拆分的是特征维度，不是把 token 分成 32 组。

### 5.1 每个 head 都有独立的投影参数吗？

从数学上看，第 $i$ 个 Query head 对应一块参数：

<p align="center">$W_Q^{(i)}\in\mathbb{R}^{H\times D_h}$</p>

它计算：

<p align="center">$Q_i=XW_Q^{(i)}$</p>

32 个 head 可以写成 32 次独立投影。但工程上不会为了这个表达真的启动 32 次小矩阵乘法，而是把这些参数沿输出列方向拼起来：

<p align="center">$W_Q=[W_Q^{(1)},W_Q^{(2)},\ldots,W_Q^{(32)}]\in\mathbb{R}^{4096\times4096}$</p>

一次大矩阵乘法即可得到：

<p align="center">$Q=XW_Q=[Q_1,Q_2,\ldots,Q_{32}]$</p>

然后 reshape：

```text
[B, S, 4096]
    ↓ reshape
[B, S, 32, 128]
    ↓ 调整逻辑维度顺序
[B, 32, S, 128]
```

所以“每个 head 有独立参数”和“实现中维护一个大矩阵”并不冲突。大矩阵的不同列块就是不同 head 的参数。

<p align="center">
  <img src="/img/in-post/ai-infra-qkv-gemm-and-multi-heads.png" alt="一次大 GEMM 同时计算多个独立 Attention head 的 Query 投影" style="max-width: 100%;">
</p>

### 5.2 为什么这仍然是真正的多头？

如果只有一个 4096 维的大 head，那么每对 token 只产生一套注意力分布：

<p align="center">$A=\operatorname{softmax}\left(\dfrac{QK^\top}{\sqrt{4096}}\right)$</p>

而 32 个 head 会分别产生 32 套注意力分布：

<p align="center">$A_1,A_2,\ldots,A_{32}\in\mathbb{R}^{B\times S\times S}$</p>

每个 head 的 Q、K 参数不同，Softmax 也独立执行，因此同一个 Query token 可以在不同特征子空间中采用不同的关注模式。

多头的本质不在于参数是否保存成 32 个 Python 对象，而在于：

> 特征被分成多个子空间，每个子空间独立计算 $Q_iK_i^\top$ 和 Softmax，产生独立的注意力分布。

<p align="center">
  <img src="/img/in-post/multi-head-attention-split-features-not-tokens.png" alt="多头注意力拆分特征子空间而不是拆分 token" style="max-width: 100%;">
</p>

### 5.3 Q、K、V 还可以融合成一次 GEMM

GEMM 是 General Matrix-Matrix Multiplication，即通用矩阵乘法。Transformer 中的线性层主要由 GEMM 实现。

逻辑上，Q、K、V 是三次投影：

<p align="center">$Q=XW_Q,\qquad K=XW_K,\qquad V_{val}=XW_V$</p>

在标准 MHA 中，还可以把三组权重继续拼接：

<p align="center">$W_{QKV}=[W_Q,W_K,W_V]\in\mathbb{R}^{H\times3H}$</p>

然后执行一次更大的 GEMM：

<p align="center">$[Q,K,V_{val}]=XW_{QKV}$</p>

形状是：

```text
[B×S, H] @ [H, 3H] -> [B×S, 3H]
```

之后再 split 出 Q、K、V，并 reshape 成多个 head。不同框架也可能保留三个独立 Linear，因此更准确的说法是：

> 数学上有三组线性投影，工程上可以根据模型和 Kernel 实现为一次或多次 GEMM。

### 5.4 多头拼接后为什么还要输出投影？

每个 head 的输出是：

<p align="center">$O_i\in\mathbb{R}^{B\times S\times D_h}$</p>

32 个 head 沿特征维拼接：

<p align="center">$O_{cat}=\operatorname{Concat}(O_1,\ldots,O_{32})\in\mathbb{R}^{B\times S\times4096}$</p>

此时每个 token 的向量可以看成：

```text
[head 1 的 128 维 | head 2 的 128 维 | ... | head 32 的 128 维]
```

Concat 只是把各个 head 放在一起，并没有主动混合它们。接下来使用输出投影：

<p align="center">$Y=O_{cat}W_O$</p>

其中：

<p align="center">$W_O\in\mathbb{R}^{(N_hD_h)\times H}$</p>

常见情况下 $N_hD_h=H$，因此 $W_O$ 是 $H\times H$ 方阵。它有两个主要作用：

1. 对不同 head 的输出通道进行可学习的混合；
2. 把结果映射回残差流要求的隐藏维度 $H$。

它不会在不同 token 之间再次交换信息，只在每个 token 自己的特征维上进行线性变换。token 之间的信息聚合已经发生在前面的 Attention 中。

这里的 $W_O$ 也不是模型最后把 hidden state 映射到词表 logits 的 LM Head。二者形状和职责不同：

```text
Attention 输出投影： [B, S, H]   -> [B, S, H]
LM Head：             [B, S, H]   -> [B, S, |V|]
```

## 6. MHA、GQA 和 MQA

推理时，每一层都需要缓存历史 token 的 K 和 V。标准 MHA 为每个 Query head 保存一组 K/V，KV Cache 很快会占用大量显存。

MQA 和 GQA 的思路是：保留多个 Query head，但减少 K/V head 数量，让多个 Query head 共享 K/V。

### 6.1 MHA：一对一

标准 MHA 中：

<p align="center">$N_{kv}=N_h$</p>

假设有 32 个 Query head：

```text
Q head 1  ↔ K/V head 1
Q head 2  ↔ K/V head 2
...
Q head 32 ↔ K/V head 32
```

张量形状为：

<p align="center">$Q,K,V_{val}\in\mathbb{R}^{B\times S\times32\times128}$</p>

### 6.2 MQA：全部 Query 共享一组 K/V

Multi-Query Attention 保留 32 个 Query head，但只有 1 个 KV head：

<p align="center">$N_{kv}=1$</p>

```text
Q head 1  ─┐
Q head 2  ─┤
...        ├─ 共享 K/V head 1
Q head 32 ─┘
```

形状变为：

<p align="center">$Q\in\mathbb{R}^{B\times S\times32\times128}$</p>

<p align="center">$K,V_{val}\in\mathbb{R}^{B\times S\times1\times128}$</p>

共享 K/V 不表示 32 个 head 的输出相同。每个 $Q_i$ 仍然不同，因此 $Q_iK^\top$ 得到的注意力分布也可以不同。

### 6.3 GQA：若干 Query 共享一组 K/V

Grouped-Query Attention 位于 MHA 和 MQA 之间。假设有 32 个 Query head、8 个 KV head，那么每 4 个 Query head 共享一组 K/V：

```text
Q head  1 ~  4 -> K/V head 1
Q head  5 ~  8 -> K/V head 2
...
Q head 29 ~ 32 -> K/V head 8
```

形状为：

<p align="center">$Q\in\mathbb{R}^{B\times S\times32\times128}$</p>

<p align="center">$K,V_{val}\in\mathbb{R}^{B\times S\times8\times128}$</p>

每组包含的 Query head 数量是：

<p align="center">$G=\dfrac{N_h}{N_{kv}}=\dfrac{32}{8}=4$</p>

三种机制可以统一理解为：

| 机制 | Query head 数 | KV head 数 | 共享方式 |
| --- | ---: | ---: | --- |
| MHA | 32 | 32 | 一个 Q 对应一组 K/V |
| GQA | 32 | 8 | 每 4 个 Q 共享一组 K/V |
| MQA | 32 | 1 | 所有 Q 共享一组 K/V |

GQA 不会把 K/V 在显存中真的复制成 32 份。高效 Attention Kernel 会在逻辑上把同一个 KV head 提供给对应的多个 Query head，否则会抵消缓存和带宽收益。

<p align="center">
  <img src="/img/in-post/ai-infra-mha-gqa-mqa-kv-cache.png" alt="MHA、GQA 与 MQA 的 Query head 和 KV head 共享关系及缓存差异" style="max-width: 100%;">
</p>

### 6.4 为什么它能减少 KV Cache？

KV Cache 的大小与 KV head 数 $N_{kv}$ 成正比。忽略额外管理开销，其字节数约为：

<p align="center">$M_{KV}=2LBSN_{kv}D_h\cdot s$</p>

其中：

- $2$ 表示 K 和 V；
- $L$ 是层数；
- $B$ 是 batch size；
- $S$ 是缓存的 token 数；
- $s$ 是每个元素占用的字节数。

当 Query head 数为 32 时：

```text
MHA：Nkv = 32
GQA：Nkv = 8  -> KV Cache 约为 MHA 的 1/4
MQA：Nkv = 1  -> KV Cache 约为 MHA 的 1/32
```

GQA 在模型表达能力和推理效率之间提供了折中，因此被很多现代大模型采用。

## 7. 为什么 Attention 需要位置信息？

先考虑没有位置编码、也没有因果 Mask 的 Self-Attention。

输入：

```text
[猫, 追, 狗]
```

如果重新排列成：

```text
[狗, 猫, 追]
```

Q、K、V 的行会跟着 token 一起重新排列。对于“猫”来说，它仍然用相同的 Query 去匹配同一组“猫、追、狗”的 Key；对应权重和 Value 同步换了顺序，加权和的内容不会因此改变。

原来：

<p align="center">$o_{猫}=0.2v_{猫}+0.3v_{追}+0.5v_{狗}$</p>

重新排列后可能写成：

<p align="center">$o'_{猫}=0.5v_{狗}+0.2v_{猫}+0.3v_{追}$</p>

加法项只是换了顺序，因此结果仍然相同。整体表现为：

```text
[猫, 追, 狗] -> [o猫, o追, o狗]
[狗, 猫, 追] -> [o狗, o猫, o追]
```

输入怎样排列，输出就跟着怎样排列，这叫**排列等变**，不是排列不变。

如果用排列矩阵 $P$ 表示换序，在无位置编码、无位置相关 Mask 的条件下：

<p align="center">$\operatorname{Attn}(PX)=P\operatorname{Attn}(X)$</p>

这说明 Attention 自身不会知道一个 token 原来处于第几个位置。

需要补充：Decoder 的因果 Mask 本身已经规定了“过去”和“未来”，所以加入固定因果 Mask 后，不再对任意排列满足上面的等变关系。但 Mask 主要限制可见范围，模型仍需要位置编码来表达更丰富的绝对位置、相对方向和距离关系。

## 8. RoPE：在 Q、K 内部旋转位置

原始 Transformer 把位置向量直接加到 token embedding 上。现代 Decoder-only 大模型更常使用 RoPE，也就是 Rotary Position Embedding。

它不增加新的 token，也不改变 Q、K 的形状，而是在每个 head 内部，把 Q 和 K 的特征两两组成二维坐标，再按 token 位置进行旋转。

### 8.1 一个 token 到底旋转了什么？

假设某个 token 在一个 head 中的 Q 是 8 维：

<p align="center">$q=(q_0,q_1,q_2,q_3,q_4,q_5,q_6,q_7)$</p>

为了直观说明，可以把它分成四个二维组：

```text
第 0 组：(q₀, q₁)
第 1 组：(q₂, q₃)
第 2 组：(q₄, q₅)
第 3 组：(q₆, q₇)
```

每个二维组旋转一次，旋转后再组合回一个 8 维向量：

```text
一个 8 维 Q
  ↓ 分成 4 个二维组
每组按自己的角度旋转一次
  ↓ 重新组合
一个新的 8 维 Q
```

因此，一个 token 不会因为 RoPE 变成多份 token 或多个完整 Q。它仍然只有一个 Q 向量，只是向量内部的二维特征组分别发生了数值变换。

真实模型中，如果 $D_h=128$，就可以形成 64 个二维组。不同实现可能采用相邻维度配对，也可能采用前后半区配对；只要训练和推理采用一致约定，数学本质都是二维子空间中的旋转。

### 8.2 每组的旋转速度不同

第 $i$ 个二维组使用频率：

<p align="center">$\theta_i=\dfrac{1}{10000^{2i/D_r}}$</p>

其中 $D_r$ 是参与 RoPE 的 rotary dimension，很多模型中等于 $D_h$，也有模型只旋转 head dimension 的一部分。

位置为 $m$ 的 token，第 $i$ 组旋转角度是：

<p align="center">$\alpha_{m,i}=m\theta_i$</p>

同一组中的两个元素使用同一个角度，不同组的 $	heta_i$ 不同。所以：

- 前面的高频组每移动一个位置转得更多，对局部位置变化更敏感；
- 后面的低频组转得更慢，周期更长，能在更长范围内保留位置差异。

可以把它类比成钟表中的秒针、分针和时针：单个指针都存在周期重复，但多个速度联合起来，可以表示多个尺度的位置关系。

### 8.3 二维旋转公式

对第 $i$ 个二维组，旋转角度记为 $\alpha=m\theta_i$：

<p align="center">$q'_{2i}=q_{2i}\cos\alpha-q_{2i+1}\sin\alpha$</p>

<p align="center">$q'_{2i+1}=q_{2i}\sin\alpha+q_{2i+1}\cos\alpha$</p>

也可以使用旋转矩阵表示：

<p align="center">$R(\alpha)=\begin{bmatrix}\cos\alpha&-\sin\alpha\\\sin\alpha&\cos\alpha\end{bmatrix}$</p>

位置 $m$ 上的 Query 和位置 $n$ 上的 Key 分别是：

<p align="center">$q'_{m,i}=R(m\theta_i)q_{m,i},\qquad k'_{n,i}=R(n\theta_i)k_{n,i}$</p>

### 8.4 相对位置为什么会出现在点积中？

旋转后的 Q、K 做点积：

<p align="center">$(q'_{m,i})^\top k'_{n,i}=q_{m,i}^\top R(m\theta_i)^\top R(n\theta_i)k_{n,i}$</p>

旋转矩阵满足：

<p align="center">$R(m\theta_i)^\top R(n\theta_i)=R((n-m)\theta_i)$</p>

所以：

<p align="center">$(q'_{m,i})^\top k'_{n,i}=q_{m,i}^\top R((n-m)\theta_i)k_{n,i}$</p>

这说明位置相关部分只通过相对位置 $n-m$ 进入点积。

例如：

```text
Query 在位置 3，Key 在位置 8：相对距离 8 - 3 = 5
Query 在位置 13，Key 在位置 18：相对距离 18 - 13 = 5
```

两组 token 的绝对位置不同，但 RoPE 引入的相对旋转相同。

更准确地说，最终注意力分数仍然同时依赖：

1. Q、K 的内容；
2. 两个 token 的相对位置。

不能把它理解成“距离相同，注意力分数就一定相同”。只有位置造成的那一部分变化依赖相对距离。

<p align="center">
  <img src="/img/in-post/ai-infra-rope-relative-position.png" alt="RoPE 将 Q 和 K 的特征组成二维组并通过旋转编码相对位置" style="max-width: 100%;">
</p>

### 8.5 为什么只旋转 Q 和 K？

Q、K 的点积决定“应该关注谁”：

<p align="center">$A=\operatorname{softmax}\left(\dfrac{QK^\top}{\sqrt{D_h}}+M\right)$</p>

因此对 Q、K 注入位置，可以直接影响注意力权重。V 主要承载权重确定后要取回的内容：

<p align="center">$O=AV_{val}$</p>

所以 RoPE 通常旋转 Q 和 K，而不旋转 V。

## 9. FFN：每个 token 独立加工信息

Attention 负责不同 token 之间的信息交换，FFN 则对每个 token 独立执行同一套非线性变换。

标准 FFN 可以写成：

<p align="center">$\operatorname{FFN}(x)=\operatorname{Activation}(xW_1)W_2$</p>

它先升维，再降维：

```text
[B, S, H]
    ↓ W₁
[B, S, Hff]
    ↓ 激活函数
[B, S, Hff]
    ↓ W₂
[B, S, H]
```

如果 $H=4096$，标准 FFN 的 $H_{ff}$ 常取接近 $4H$ 的量级。

### 9.1 从 ReLU、GELU 到 SwiGLU

ReLU：

<p align="center">$\operatorname{ReLU}(x)=\max(0,x)$</p>

GELU：

<p align="center">$\operatorname{GELU}(x)=x\Phi(x)$</p>

现代 LLaMA 风格模型常使用 SwiGLU：

<p align="center">$\operatorname{SwiGLU}(x)=\left[\operatorname{SiLU}(xW_{gate})\odot(xW_{up})\right]W_{down}$</p>

其中：

```text
x @ Wgate：生成门控分支
x @ Wup：  生成候选信息
SiLU(gate) ⊙ up：逐元素决定信息通过多少
结果 @ Wdown：映射回 H 维
```

以 $H=4096$、$H_{ff}=11008$ 为例：

```text
x：       [B, S, 4096]
gate：    [B, S, 11008]
up：      [B, S, 11008]
mid：     [B, S, 11008]
down：    [B, S, 4096]
```

FFN 不改变 token 数量。每个 token 独立经过同一组参数，所以它只改变每个 token 内部的特征表示。

## 10. Norm 与残差连接

现代大模型通常使用 Pre-Norm：先归一化，再进入 Attention 或 FFN，最后与残差分支相加。

### 10.1 LayerNorm 与 RMSNorm

LayerNorm 会减均值、除以标准差，再进行可学习的缩放和平移：

<p align="center">$\operatorname{LayerNorm}(x)=\gamma\odot\dfrac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$</p>

LLaMA 使用 RMSNorm，它不减均值，而是按均方根缩放：

<p align="center">$\operatorname{RMSNorm}(x)=\gamma\odot\dfrac{x}{\sqrt{\operatorname{mean}(x^2)+\epsilon}}$</p>

两者通常都沿每个 token 的隐藏维度进行计算：

```text
每个 token 的 H 个特征 -> 单独归一化
```

形状始终不变：

<p align="center">$[B,S,H]\rightarrow[B,S,H]$</p>

### 10.2 残差连接

残差连接的形式是：

<p align="center">$y=x+F(x)$</p>

它让子层主要学习“在当前表示上增加什么”，并提供一条更直接的信息和梯度通路。

对 Transformer 来说，残差相加要求两个分支形状一致：

```text
原始输入：   [B, S, H]
子层输出：   [B, S, H]
逐元素相加： [B, S, H]
```

这也是每个 Decoder Block 都保持隐藏维度 $H$ 的重要原因。

### 10.3 Pre-Norm 与 Post-Norm

原始 Transformer 常写成 Post-Norm：

<p align="center">$y=\operatorname{Norm}(x+\operatorname{Sublayer}(x))$</p>

现代大模型常使用 Pre-Norm：

<p align="center">$y=x+\operatorname{Sublayer}(\operatorname{Norm}(x))$</p>

Pre-Norm 中，残差主路径不穿过 Norm，通常更容易稳定训练很深的网络。但这不是说它在所有模型和训练配方中一定更好，而是它已经成为大模型中的常见工程选择。

## 11. 组装完整的 Transformer Decoder Block

把前面的模块拼起来，现代 Pre-Norm Decoder Block 可以写成两个公式：

<p align="center">$U_l=X_l+\operatorname{Attention}(\operatorname{RMSNorm}(X_l))$</p>

<p align="center">$X_{l+1}=U_l+\operatorname{SwiGLU}(\operatorname{RMSNorm}(U_l))$</p>

完整数据流是：

```text
X_l [B, S, H]
  │
  ├──────────────── 残差 ────────────────┐
  ↓                                      │
RMSNorm                                  │
  ↓                                      │
QKV 投影                                 │
  ↓                                      │
Q/K 应用 RoPE                            │
  ↓                                      │
Causal Multi-Head Attention              │
  ↓                                      │
输出投影 W_O                              │
  ↓                                      │
加法 ←───────────────────────────────────┘
  ↓
U_l [B, S, H]
  │
  ├──────────────── 残差 ────────────────┐
  ↓                                      │
RMSNorm                                  │
  ↓                                      │
SwiGLU FFN                               │
  ↓                                      │
加法 ←───────────────────────────────────┘
  ↓
X_{l+1} [B, S, H]
```

### 11.1 用 LLaMA-2-7B 的形状走一遍

取下面的配置：

```text
B = 1
S = 2048
H = 4096
Nh = 32
Nkv = 32
Dh = 128
Hff = 11008
```

#### Attention 子层

```text
Block 输入：                    [1, 2048, 4096]
RMSNorm：                       [1, 2048, 4096]

融合 QKV 投影：
[1×2048, 4096] @ [4096, 12288]
                            -> [1×2048, 12288]

拆分并 reshape：
Q、K、V                     -> [1, 32, 2048, 128]

Q、K 应用 RoPE：                形状不变

每个 head 计算 QKᵀ：
[2048, 128] @ [128, 2048]   -> [2048, 2048]

32 个 head 的分数：             [1, 32, 2048, 2048]
Mask + Softmax：                 [1, 32, 2048, 2048]

权重乘 V：
[2048, 2048] @ [2048, 128]  -> [2048, 128]

拼接 32 个 head：               [1, 2048, 4096]
输出投影 W_O：                  [1, 2048, 4096]
第一次残差相加：                [1, 2048, 4096]
```

#### FFN 子层

```text
RMSNorm：                       [1, 2048, 4096]

gate 投影：
[1×2048, 4096] @ [4096, 11008]
                            -> [1×2048, 11008]

up 投影：                       [1, 2048, 11008]
SiLU(gate) ⊙ up：               [1, 2048, 11008]

down 投影：
[1×2048, 11008] @ [11008, 4096]
                            -> [1×2048, 4096]

第二次残差相加：                [1, 2048, 4096]
Block 输出：                    [1, 2048, 4096]
```

从头到尾，残差流都保持 $[B,S,H]$。因此多个 Block 可以连续堆叠：

```text
Embedding 输出 [B, S, H]
        ↓
Decoder Block 1 [B, S, H]
        ↓
Decoder Block 2 [B, S, H]
        ↓
...
        ↓
Decoder Block L [B, S, H]
```

每一层形状相同，但参数并不共享，token 表示会在层与层之间不断变化。

<p align="center">
  <img src="/img/in-post/ai-infra-transformer-pre-norm-decoder-block.png" alt="现代 Pre-Norm Transformer Decoder Block 的完整数据流与残差路径" style="max-width: 100%;">
</p>

### 11.2 一个 Block 的参数主要在哪里？

仍以 $H=4096$、$H_{ff}=11008$ 的标准 MHA 为例。

Attention 的四个矩阵为：

<p align="center">$W_Q,W_K,W_V,W_O\in\mathbb{R}^{4096\times4096}$</p>

参数量：

<p align="center">$4\times4096^2=67,108,864$</p>

SwiGLU 的三个矩阵为：

<p align="center">$W_{gate},W_{up}\in\mathbb{R}^{4096\times11008},\qquad W_{down}\in\mathbb{R}^{11008\times4096}$</p>

参数量：

<p align="center">$3\times4096\times11008=135,266,304$</p>

两个 RMSNorm 各有一个长度为 4096 的缩放参数，因此单个 Block 总计约：

<p align="center">$67,108,864+135,266,304+8192=202,383,360$</p>

也就是约 202M 参数。可以看出，FFN 约占单个 Block 参数量的三分之二，Attention 约占三分之一。

这直接影响后续的 AI Infra 优化：

- Attention 是长序列二次复杂度和 KV Cache 的核心；
- FFN 拥有大量权重和 GEMM 计算，是张量并行与量化的重要对象。

## 12. 从最后一个 Block 到词表 logits

经过 $L$ 个 Decoder Block 后，模型通常还会执行一次 Final RMSNorm，得到：

<p align="center">$X_{final}\in\mathbb{R}^{B\times S\times H}$</p>

然后通过 LM Head 映射到整个词表：

<p align="center">$W_{lm}\in\mathbb{R}^{H\times|\mathcal V|}$</p>

<p align="center">$\text{logits}=X_{final}W_{lm}$</p>

形状变化是：

```text
[B, S, H] @ [H, |V|] -> [B, S, |V|]
```

每个位置都得到对词表中所有候选 token 的打分。训练时可以用这些 logits 计算下一个 token 的交叉熵；生成时通常只取当前最后一个位置的 logits，经过采样得到下一个 token。

<p align="center">
  <img src="/img/in-post/transformer-final-projection-logits.png" alt="Transformer 最终隐藏状态经过 LM Head 得到词表 logits" style="max-width: 100%;">
</p>

## 13. 自回归生成、Prefill 与 Decode

Decoder-only 模型一次预测一个新 token：

```text
输入：“人工智能的”
  ↓
预测：“核”
  ↓ 把“核”加入输入
预测：“心”
  ↓
继续生成……
```

推理可以分成 Prefill 和 Decode 两个阶段。

### 13.1 Prefill

Prefill 一次处理完整 prompt。假设 prompt 长度为 $P$：

```text
Query 长度：P
Key/Value 长度：P
```

所有 prompt token 可以并行完成 QKV 投影和 Attention，并把每一层产生的 K、V 写入 KV Cache。

在常见配置中，Prefill 有较大的矩阵乘法，通常更容易利用 GPU 算力；但是否严格属于 Compute Bound，仍取决于 prompt 长度、batch、模型、数据类型和 Kernel 实现。

### 13.2 Decode

进入 Decode 后，每一步只有一个新 token：

```text
新 Query 长度：1
历史 Key/Value 长度：P + 已生成 token 数
```

新 token 只需要生成自己的 Q、K、V：

1. 新 Q 与所有历史 K 计算注意力分数；
2. 根据权重汇总历史 V；
3. 把新的 K、V 追加到 KV Cache；
4. 经过其余层后预测下一个 token。

历史 Q 不需要缓存，因为未来步骤不会再次用历史 token 充当 Query；未来每一步只需要当前新 token 的 Query。

Decode 的矩阵在单请求下很“瘦”，同时每一步都要读取模型权重和历史 KV Cache，因此常常受到显存带宽限制。但连续批处理增大 batch 后，它也可能重新形成更适合 GPU 的矩阵计算。

### 13.3 为什么 KV Cache 很重要？

如果没有 KV Cache，每生成一个 token 都要为整个历史序列重复计算 K、V。缓存后，历史 K、V 可以直接复用，每一步只新增当前 token 的 K、V。

代价是显存随以下因素线性增长：

<p align="center">$M_{KV}=2LBSN_{kv}D_h\cdot s$</p>

这也解释了为什么 GQA、MQA、PagedAttention 和 KV Cache 量化是推理优化中的核心技术。

## 14. 从 AI Infra 角度重新看 Transformer

理解完整 Decoder Block 后，后续技术不再是孤立名词：

| Transformer 组件 | 主要计算或资源 | AI Infra 关联 |
| --- | --- | --- |
| QKV 投影 | 大型 GEMM | Tensor Core、融合 QKV、张量并行 |
| $QK^\top$ 与 $AV$ | Batched GEMM | FlashAttention、长序列优化 |
| Softmax | 行归约、指数、归一化 | Stable/Online Softmax、算子融合 |
| RoPE | 高频逐元素运算 | 融合进 Attention Kernel |
| 输出投影 $W_O$ | GEMM、跨 head 混合 | 张量并行后的通信与归并 |
| RMSNorm | 归约、逐元素缩放 | Welford、Residual + Norm 融合 |
| SwiGLU FFN | 三个大型权重矩阵 | GEMM、量化、张量并行、MoE |
| 多层 Block | 层级堆叠 | 流水线并行、激活重计算 |
| K/V | 随序列增长的缓存 | GQA/MQA、PagedAttention、KV 量化 |
| 自回归生成 | 逐 token 串行 | Continuous Batching、Speculative Decoding |

FlashAttention 不会改变精确 Attention 的算术复杂度。它的核心是通过分块和在线维护 Softmax 状态，避免把完整的 $S\times S$ 分数或概率矩阵写回 HBM，从而减少中间结果物化和显存读写。

## 15. 把完整链路串起来

现在可以把 Transformer Decoder-only 模型压缩成下面这条数据流：

```text
文本
  ↓ Tokenizer
Token IDs [B, S]
  ↓ Embedding 查表
Hidden States [B, S, H]
  ↓
重复 L 次 Decoder Block：
  RMSNorm
    ↓
  QKV 投影（逻辑上三组投影，工程上可融合为大 GEMM）
    ↓
  拆成多个 Query/KV head
    ↓
  对 Q、K 应用 RoPE
    ↓
  QKᵀ / sqrt(Dh) + causal mask
    ↓
  Softmax
    ↓
  权重乘 V
    ↓
  拼接 Query heads + 输出投影 W_O
    ↓
  第一次残差连接
    ↓
  RMSNorm
    ↓
  SwiGLU FFN
    ↓
  第二次残差连接
  ↓
Final RMSNorm
  ↓ LM Head
Logits [B, S, |V|]
  ↓
预测下一个 token
```

这次最重要的几个理解是：

1. $S$ 是当前输入的 token 数，$H$ 是每个 token 的隐藏维度，$|\mathcal V|$ 是整个词表大小；
2. Q、K、V 是同一输入经过不同参数投影得到的三种表示；
3. 多头的投影可以合并成一次大 GEMM，但每个 head 仍会独立计算注意力分数和 Softmax；
4. 输出投影 $W_O$ 用来混合多个 head，并让结果回到残差流的 $H$ 维；
5. MHA、GQA、MQA 的主要差异是 KV head 数量，以及每组 K/V 被多少个 Query head 共享；
6. RoPE 旋转的是每个 token 的 Q、K 向量内部的二维特征组，token 之间的相对位置在 Q、K 点积时出现；
7. Attention 负责 token 之间交换信息，FFN 负责每个 token 独立加工信息；
8. 一个现代 Decoder Block 可以概括为两次 `Norm → 子层 → 残差相加`。

> Transformer 架构不是后续 AI Infra 学习之外的一章，而是所有算子优化、并行训练与推理系统共同作用的对象。只有先把张量怎样流过 Decoder Block 看清楚，后续的 FlashAttention、张量并行、KV Cache 和量化才有明确的落点。
