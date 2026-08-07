---
layout:     post
title:      "从自然语言到 logits：我终于把 Transformer 的形状串起来了"
subtitle:   "一次 AI Infra 数学基础学习记录"
date:       2026-08-07 20:00:00 +0800
author:     "Liu Mengxuan"
mathjax:    true
header-img: "img/post-bg-miui6.jpg"
categories: [技术]
tags:       [技术, AI Infra, Transformer, 数学基础, 线性代数, 深度学习]
---

最近开始系统学习 AI Infra。我先从仓库里的前置知识开始看，到了数学基础这一章，遇到了一条看起来很短、但实际上包含了很多信息的公式：

$$
Y=XW,
\qquad X\in\mathbb{R}^{(BS)\times H},
\qquad W\in\mathbb{R}^{H\times V}
$$

一开始我对里面的 `B`、`S`、`H`、`V` 只有模糊印象，尤其不明白：为什么 batch size 会影响输出形状？`H` 是某一层的维度，还是整个网络的维度？

后来我把问题从“记住公式”改成了“沿着数据流走一遍”，才慢慢把它串起来：一段自然语言是怎样变成 token，token 又怎样变成 hidden state，最后怎样变成词表上的预测分数。

## 1. 先把符号认全

在 Transformer 和大语言模型中，常见的符号是：

| 符号 | 含义 |
| --- | --- |
| $B$ | batch size，一次处理多少条样本 |
| $S$ | sequence length，每条样本包含多少个 token |
| $H$ | hidden size，每个 token 的隐藏向量维度 |
| $V$ | vocabulary size，词表大小 |
| $N_h$ | attention head 的数量 |
| $D_h$ | 每个 attention head 的维度，$D_h=H/N_h$ |
| $L$ | Transformer 层数，也就是网络深度 |

这里最容易混淆的是 $L$ 和 $H$：

> $L$ 是深度，表示有多少层；$H$ 是宽度，表示每个 token 的向量有多长。

可以把 Transformer 想成一栋楼：$L$ 是楼层数，$H$ 是每一层房间的宽度。它们描述的是完全不同的东西。

## 2. 一切从自然语言开始

模型最原始的输入不是矩阵，而是自然语言，例如：

```text
我喜欢机器学习
```

但是神经网络不能直接处理汉字、单词或句子，它需要数字。因此通常要经过下面几步：

```text
自然语言
    ↓
Tokenizer 分词
    ↓
Token IDs
    ↓
Embedding 查表
    ↓
Hidden states
```

下面这张图把完整流程画了出来：

<p align="center">
  <img src="/img/in-post/ai-infra-transformer-token-to-logits.png" alt="从自然语言到 Transformer 输出 logits 的完整流程" style="max-width: 100%;">
</p>

### 2.1 Tokenizer：把文字变成 token

Tokenizer 会把文本切成模型词表中的 token。对于中文，token 不一定严格按照单个汉字切分；不同 tokenizer 可能使用汉字、词、子词甚至字节片段。

例如，为了说明流程，可以抽象成：

```text
我喜欢机器学习
    ↓
[我] [喜欢] [机器] [学习]
```

实际模型可能得到完全不同的切分结果，这取决于 tokenizer 的词表和算法。这里重要的不是具体怎么切，而是：

> Tokenizer 把连续的自然语言转换成离散的 token 序列。

### 2.2 Token IDs：给每个 token 一个整数编号

词表中的每个 token 都有一个整数 ID：

```text
[我] [喜欢] [机器] [学习]
  ↓    ↓     ↓     ↓
 42   187   936   512
```

这些数字本身没有大小关系。`936` 并不比 `42` 更“重要”，它们只是词表中的索引。

如果词表大小是 $V$，那么 token ID 通常落在：

$$
0,1,2,\ldots,V-1
$$

## 3. Embedding：把 token ID 变成 H 维向量

模型有一个 embedding 矩阵：

$$
E\in\mathbb{R}^{V\times H}
$$

它有 $V$ 行，每一行是一个 token 的 $H$ 维向量。

所以，token ID 并不是直接拿去做矩阵乘法，而是用来查表：

```text
token ID = 42   → 取 E 的第 42 行
token ID = 187  → 取 E 的第 187 行
token ID = 936  → 取 E 的第 936 行
token ID = 512  → 取 E 的第 512 行
```

每取出一行，就得到一个 $H$ 维向量：

$$
x_{b,s}\in\mathbb{R}^{H}
$$

其中 $x_{b,s}$ 表示第 $b$ 条样本中第 $s$ 个 token 的向量。

如果有 $B$ 条样本、每条样本有 $S$ 个 token，那么所有 token 的 embedding 组成：

$$
X\in\mathbb{R}^{B\times S\times H}
$$

这就是 Transformer 的输入 hidden states。

## 4. 为什么需要 batch size？

这里的 $B$ 表示：

> 一次送进模型的样本数量。

例如：

```text
B = 4
```

表示一次前向计算处理 4 条样本，而不是每次只处理一条。

### 4.1 训练时的 batch

训练集可能有几十万条样本，但不会一次全部送入模型，而是切成很多小批次：

```text
batch 1：样本 1 ~ 32
batch 2：样本 33 ~ 64
batch 3：样本 65 ~ 96
```

如果 batch size 是 32，那么每次前向传播、计算 loss、反向传播和参数更新，通常都基于 32 条样本。

使用 batch 的重要原因是 GPU 擅长并行处理大量相似的矩阵运算。相比逐条处理，把样本组成一个 batch 能够：

- 提高 GPU 利用率；
- 把很多小计算合并成更大的矩阵计算；
- 减少频繁的 Kernel 启动和调度开销；
- 提高整体吞吐量。

### 4.2 推理时的 batch

推理时 $B$ 也可能表示多个请求，但不一定严格等于请求数量。

例如：

- 交互式聊天一次处理一个请求，通常 $B=1$；
- 服务端把多个用户请求合并计算，可能 $B=8$；
- 一个请求要求生成多个候选答案，也可能出现 $B>1$。

所以更准确的说法是：

> $B$ 表示一次计算中包含多少条样本或序列，而不是专门表示用户请求数。

### 4.3 B 为什么影响输出形状？

假设：

```text
B = 2
S = 3
H = 4
```

那么输入形状是：

```text
[2, 3, 4]
```

含义是：

```text
2 条样本
每条 3 个 token
每个 token 4 维
```

总共有：

$$
B\times S=2\times3=6
$$

个 token。

如果把 $B$ 从 2 改成 4，总 token 数就从 6 变成 12。每个 token 的向量仍然是 $H$ 维，但 token 的总数量翻倍，所以矩阵的行数也翻倍。

因此：

> $B$ 不会改变每个 token 输出向量的长度，它改变的是一次计算中需要产生多少组输出。

## 5. 经过 L 层 Transformer

输入 hidden states 的形状是：

$$
[B,S,H]
$$

经过第 1 层、第 2 层，直到第 $L$ 层后，通常仍然是：

$$
[B,S,H]
$$

例如：

```text
第 1 层：[B, S, H]
第 2 层：[B, S, H]
...
第 L 层：[B, S, H]
```

为什么每层通常保持相同的 $H$？因为 Transformer 中有残差连接：

$$
X_{l+1}=X_l+\operatorname{Block}(X_l)
$$

要做加法，两边的形状必须一致。

如果某一层要把 $H$ 从 4096 改成 8192，就必须额外加入投影，把残差分支也变成 8192。理论上可以这样设计，但会增加计算、内存搬运和实现复杂度。因此标准 Transformer 通常在各层保持相同的 hidden size。

## 6. Attention 中的 H、Nh 和 Dh

Attention 内部会把 $H$ 拆成多个 head：

$$
[B,S,H]
\rightarrow
[B,S,N_h,D_h]
\rightarrow
[B,S,H]
$$

其中：

$$
D_h=\frac{H}{N_h}
$$

例如：

```text
H = 4096
Nh = 32
Dh = 4096 / 32 = 128
```

也就是把每个 token 的 4096 维表示拆成 32 个 head，每个 head 128 维。

各个 head 分别进行注意力计算，最后再合并回 4096 维。

这里要注意：Attention 内部可能会出现不同的临时形状，但经过合并后，残差流仍然回到 $H$ 维。

## 7. 最后的输出投影：从 H 维映射到 V 维

经过最后一个 Transformer Block 后，得到最终 hidden states：

$$
X\in\mathbb{R}^{B\times S\times H}
$$

为了方便做矩阵乘法，通常把前两维展平：

$$
[B,S,H]\rightarrow[BS,H]
$$

此时，每一行代表一个 token 的 hidden vector。

输出投影矩阵为：

$$
W\in\mathbb{R}^{H\times V}
$$

于是：

$$
Y=XW
$$

维度计算是：

$$
[BS,H]\times[H,V]=[BS,V]
$$

这里中间的两个 $H$ 必须匹配，输出保留两边的外部维度：

```text
[BS, H] × [H, V] = [BS, V]
```

### 7.1 每个输出元素怎么算？

对于第 $i$ 个 token 和词表中第 $j$ 个 token：

$$
Y_{i,j}=\sum_{h=1}^{H}X_{i,h}W_{h,j}
$$

也就是说，一个 token 的 $H$ 维 hidden vector，会和 $W$ 的每一列做点积，最终得到 $V$ 个分数。

这 $V$ 个分数分别对应词表中的 $V$ 个候选 token，叫作 logits。它们还不是概率，后面通常还要经过 Softmax。

### 7.2 为什么最后恢复成 `[B, S, V]`？

矩阵乘法结束后，形状是：

$$
[BS,V]
$$

但从语义上，我们仍然希望知道：

- 这是第几条样本？
- 是样本中的第几个位置？
- 对词表中的哪个 token 给出了分数？

所以会把它 reshape 回：

$$
[BS,V]\rightarrow[B,S,V]
$$

最终：

$$
Y_{b,s,:}\in\mathbb{R}^{V}
$$

表示第 $b$ 条样本、第 $s$ 个位置，对整个词表的 $V$ 维 logits。

## 8. 用一个小数字完整走一遍

假设：

```text
B = 2
S = 3
H = 4
V = 10
```

那么：

```text
输入 hidden states： [2, 3, 4]
展平：              [6, 4]
输出权重 W：        [4, 10]
矩阵乘法：          [6, 4] × [4, 10] = [6, 10]
恢复形状：          [2, 3, 10]
```

最后的 `[2, 3, 10]` 可以读成：

```text
2 条样本
每条样本 3 个 token 位置
每个位置对 10 个词表 token 给出一个分数
```

## 9. 读 AI 公式时的三个问题

以后看到类似公式，我希望自己先问三个问题。

### 第一，形状是否匹配？

$$
[BS,H]\times[H,V]=[BS,V]
$$

中间维度 $H$ 相同，因此可以做矩阵乘法。

### 第二，计算量有多大？

输出一共有 $BSV$ 个元素，每个元素需要大约 $H$ 次乘加，所以计算量约为：

$$
2BSHV\quad\text{FLOPs}
$$

这也说明为什么 $B$、$S$、$H$、$V$ 任意一个变大，计算压力都可能明显增加。

### 第三，工程风险在哪里？

最后的输出投影权重大小是：

$$
H\times V
$$

当 hidden size 和词表都很大时，$W$ 本身就可能占用大量显存；输出 logits 的形状是 $[B,S,V]$，同样可能很大。

因此工程实现还要继续考虑：

- 数据类型和数值稳定性；
- 矩阵的内存布局；
- GEMM 的分块和并行策略；
- logits 是否需要完整物化；
- Softmax 和交叉熵如何高效计算。

## 10. 最后的理解

这次学习让我意识到，AI Infra 里的数学公式，不能只看成抽象符号。更好的读法是把它还原成数据流：

```text
自然语言
→ Tokenizer
→ Token IDs
→ Embedding
→ [B, S, H]
→ L 层 Transformer
→ Attention 拆分与合并
→ [BS, H]
→ [H, V] 输出投影
→ [B, S, V] logits
```

其中：

- $B$ 决定一次处理多少条样本；
- $S$ 决定每条样本有多少个 token；
- $H$ 决定每个 token 的表示宽度；
- $L$ 决定 Transformer 有多深；
- $V$ 决定每个位置要对多少个候选 token 打分。

真正重要的不是把这些字母背下来，而是看到一个 shape 时，能够立刻说清楚：**每一维在语义上代表什么，矩阵乘法为什么能做，以及它会带来多少计算和内存成本。**
