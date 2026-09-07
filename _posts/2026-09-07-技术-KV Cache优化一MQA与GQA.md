---
layout:     post
title:      "KV Cache 优化（一）：从 MHA 到 MQA 与 GQA"
subtitle:   "少存几组 K/V，为什么就能明显改善 Decode？"
date:       2026-09-07 21:00:00 +0800
author:     "Liu Mengxuan"
mathjax:    true
header-img: "img/post-bg-miui6.jpg"
categories: [技术]
tags:       [技术, AI Infra, Transformer, KV Cache, MHA, MQA, GQA, Attention]
---

标准 KV Cache 避免了自回归生成时对历史 token 的重复计算，但它也带来了新的问题：上下文越长、并发请求越多，缓存就越大。

对于每一层、每一个 token，模型都需要保存一组历史 Key 和 Value：

<p align="center">$M_{KV}=2LBTN_{KV}D_h\times\text{bytes}$</p>

这里的 `2` 表示 K 和 V，`L` 是层数，`B` 是并发序列数，`T` 是上下文长度，`N_KV` 是 KV head 数，`D_h` 是每个 head 的维度。

前一篇文章已经解释了 `L`、`B` 和 `T` 为什么存在。这一篇只盯住其中一个变量：

> 能不能减少 $N_{KV}$，让多个 Query head 共享 K/V？

Multi-Query Attention（MQA）和 Grouped-Query Attention（GQA）正是沿着这条思路发展出来的。

<p align="center">
  <img src="/img/in-post/ai-infra-mqa-gqa-head-sharing.svg" alt="MHA、GQA 与 MQA 的 Query head 和 KV head 共享关系" style="max-width: 100%;">
</p>

## 1. 回顾 KV Cache 的基本过程

假设模型已经处理了：

```text
我   是   你
```

现在需要处理新 token“的”。在模型的第 `l` 层中，当前隐藏状态会产生新的 Q/K/V：

<p align="center">$q_t^{(l)}=h_t^{(l-1)}W_Q^{(l)},\quad k_t^{(l)}=h_t^{(l-1)}W_K^{(l)},\quad v_t^{(l)}=h_t^{(l-1)}W_V^{(l)}$</p>

当前的 K/V 会追加到这一层的缓存中，当前 Query 再读取全部历史 K/V：

<p align="center">$o_t^{(l)}=\operatorname{softmax}\left(\dfrac{q_t^{(l)}(K_{1:t}^{(l)})^T}{\sqrt{D_h}}\right)V_{1:t}^{(l)}$</p>

这里还没有体现多头。真正的模型通常有很多个 Query head，每个 head 可以学习不同的匹配方式，例如：

```text
head 1：更关注主语
head 2：更关注局部搭配
head 3：更关注位置关系
head 4：更关注指代关系
```

这些分工只是帮助理解的类比，并不是人工给每个 head 指定的任务。多头的关键不是把一个 head 机械复制多份，而是每个 Query head 有自己的投影和注意力分布。为突出缓存关系，上面的投影公式省略了 Norm 和位置编码；Pre-Norm 模型会先归一化，再投影。

问题在于：这些 Query head 是否一定都需要独立的 K/V head？

## 2. MHA：每个 Query head 都有自己的 K/V

Multi-Head Attention（MHA）是一对一结构。假设有 8 个 Query heads：

```text
Q₁ ↔ K₁、V₁
Q₂ ↔ K₂、V₂
...
Q₈ ↔ K₈、V₈
```

第 `i` 个 head 的输出是：

<p align="center">$o_i=\operatorname{softmax}\left(\dfrac{Q_iK_i^T}{\sqrt{D_h}}\right)V_i$</p>

最后把所有 head 的结果拼接起来，再做输出投影：

<p align="center">$O=\operatorname{Concat}(o_1,\ldots,o_{N_h})W_O$</p>

MHA 给每个 head 最大的独立性，但也意味着每个 token、每一层都要缓存 `N_h` 份 K 和 `N_h` 份 V。

如果：

```text
Query head 数 Nh = 8
KV head 数 NKV = 8
每个 head 维度 Dh = 128
缓存精度 = FP16，即每个元素 2 bytes
```

那么每层每个 token 的 KV Cache 是：

<p align="center">$2\times8\times128\times2=4096\ \text{bytes}=4\ \text{KiB}$</p>

这还只是一个 token、一个层。乘上几十层、几万个 token 和多个并发请求后，缓存会迅速放大。

## 3. MQA：所有 Query heads 共享一组 K/V

Multi-Query Attention（MQA）的做法非常直接：保留多个 Query heads，但所有 Query heads 共享同一个 K head 和 V head。

```text
Q₁ ─┐
Q₂ ─┤
... ├──→ 同一组 K、V
Q₈ ─┘
```

第 `i` 个 Query head 仍然有自己的 `Q_i`，但它们读取相同的 K/V：

<p align="center">$o_i=\operatorname{softmax}\left(\dfrac{Q_iK^T}{\sqrt{D_h}}\right)V$</p>

所以“共享 K/V”不意味着多个 head 得到相同结果。即使 K/V 相同，不同的 Query 也会产生不同的注意力分数：

```text
Q₁：更像在问“主语是谁？”
Q₂：更像在问“它位于哪里？”

同一份资料库 K/V
Q₁ 得到一组注意力权重
Q₂ 得到另一组注意力权重
```

用一个两维例子验证：设两个历史位置的 Key 分别是 `[1,0]`、`[0,1]`，Value 分别是 `[10,0]`、`[0,10]`。为简化数字，暂时省略缩放因子。

```text
Q1 = [2,0] → 分数 [2,0] → 权重约 [0.88,0.12] → 输出 [8.8,1.2]
Q2 = [0,2] → 分数 [0,2] → 权重约 [0.12,0.88] → 输出 [1.2,8.8]
```

同样的 K/V，通过不同 Q 得到了不同的汇总结果。共享的是可查阅的信息，不是注意力权重或最终输出。

如果仍有 8 个 Query heads，MQA 只保存 1 个 KV head：

<p align="center">$2\times1\times128\times2=512\ \text{bytes/token/layer}$</p>

与上面的 MHA 相比，KV Cache 缩小到原来的 `1/8`。

### 3.1 为什么 Decode 会更快？

Decode 时，当前 Query 每一步都要从显存读取历史 K/V。长上下文下，这部分经常受 HBM 带宽限制。

MQA 同时带来两种收益：

1. KV Cache 占用更小，可以容纳更长上下文或更多并发请求；
2. 每一步需要从 HBM 读取的历史 K/V 更少，Attention Decode 更容易加速。

它并没有减少 Query head 数，也没有把所有注意力 head 合并成一个。减少的是会随序列长度不断增长的 K/V 状态。

### 3.2 代价是什么？

MHA 中，每个 head 都有独立的 K/V 表达空间；MQA 中，所有 Query heads 只能查询同一份 K/V 资料库。

这会减少 K/V 的多样性。模型仍可通过不同 Query 和输出投影学习不同功能，但共享程度过高时，模型质量或训练行为可能受到影响。具体影响取决于模型规模、训练方式和任务，不能简单断言 MQA 一定损失多少质量。

## 4. GQA：在 MHA 和 MQA 之间分组

Grouped-Query Attention（GQA）采用折中方式：把 Query heads 分成若干组，每一组共享一对 K/V head。

假设有 8 个 Query heads、2 个 KV heads：

```text
第 1 组：Q₁、Q₂、Q₃、Q₄ → K₁、V₁
第 2 组：Q₅、Q₆、Q₇、Q₈ → K₂、V₂
```

令 Query head 总数为 `N_h`，KV head 数为 `N_KV`，一般要求：

<p align="center">$N_h\bmod N_{KV}=0$</p>

每个 KV head 服务的 Query head 数是：

<p align="center">$G=\dfrac{N_h}{N_{KV}}$</p>

于是三种结构可以统一起来：

```text
NKV = Nh：MHA，每个 Query head 独享一组 K/V
1 < NKV < Nh：GQA，同组 Query heads 共享一组 K/V
NKV = 1：MQA，所有 Query heads 共享一组 K/V
```

GQA 保留了多组 K/V，因此通常比 MQA 有更强的表达自由度；同时它又比 MHA 少存很多 KV Cache。

## 5. 用一个完整 shape 例子计算

假设一个模型采用：

```text
L = 32 层
Nh = 32 个 Query heads
Dh = 128
T = 4096 tokens
B = 1
FP16 = 2 bytes
```

不同结构只改变 `N_KV`：

| 结构 | Query heads | KV heads | 每层每 token KV | 全模型 4096 tokens |
| --- | ---: | ---: | ---: | ---: |
| MHA | 32 | 32 | 16 KiB | 2048 MiB |
| GQA | 32 | 8 | 4 KiB | 512 MiB |
| MQA | 32 | 1 | 0.5 KiB | 64 MiB |

以 GQA 为例：

<p align="center">$M_{KV}=2\times32\times1\times4096\times8\times128\times2=512\ \text{MiB}$</p>

这里的 GQA 是 MHA 的 `1/4`，因为它把 KV head 数从 32 减到了 8。

<p align="center">
  <img src="/img/in-post/ai-infra-mqa-gqa-memory.svg" alt="MHA、GQA 与 MQA 的 KV Cache 容量和读取量对比" style="max-width: 100%;">
</p>

## 6. Prefill 和 Decode 分别发生了什么？

### 6.1 Prefill

Prompt 中的所有 token 一起进入模型。以 GQA 为例：

```text
Q：[B, Nh, T, Dh]
K：[B, NKV, T, Dh]
V：[B, NKV, T, Dh]
```

多个 Query heads 在计算 Attention 时，通过分组映射到对应的 KV head。逻辑上可以把某个 KV head 广播给组内多个 Query heads，但高效实现不一定真的复制数据。

### 6.2 Decode

每个请求只有一个新 token：

```text
当前 Q：[B, Nh, 1, Dh]
KV Cache：[B, NKV, T, Dh]
```

当前 token 只新增 `N_KV` 组 K/V，而不是 `N_h` 组。随着 `T` 增长，这个差异会持续累积。

需要强调：GQA/MQA 不会让标准 Attention 完全摆脱对长度 `T` 的读取。当前 Query 仍要查询全部历史位置，只是每个位置保存和读取的 KV heads 更少。

## 7. 它们改变了模型结构吗？

改变了。`W_K` 和 `W_V` 的输出维度会随 `N_KV` 改变：

<p align="center">$W_K,W_V\in\mathbb{R}^{H\times(N_{KV}D_h)}$</p>

因此，一个已经训练好的普通 MHA 模型不能仅靠推理参数开关无损变成 MQA 或 GQA。通常有三种情况：

1. 模型从头就按 MQA/GQA 结构训练；
2. 将 MHA checkpoint 转换为 GQA，再继续训练或微调；
3. 使用专门的蒸馏、合并或适配方法降低转换损失。

这和 PagedAttention 不同。PagedAttention 主要改变 KV Cache 的物理管理方式，不改变模型学习到的 Attention 数学结构；MQA/GQA 则直接改变了 K/V head 的参数化方式。

## 8. 如何理解三者的取舍？

可以把每个 KV head 想成一套可检索的资料库：

```text
MHA：每位检索员都有自己的资料库
MQA：所有检索员共用一个资料库
GQA：检索员分组，每组共用一个资料库
```

| 结构 | KV Cache | K/V 多样性 | 典型取舍 |
| --- | --- | --- | --- |
| MHA | 最大 | 最高 | 表达自由度优先 |
| MQA | 最小 | 最低 | Decode 带宽与容量优先 |
| GQA | 居中 | 居中 | 质量和推理效率折中 |

工程实践中，GQA 常见的原因并不神秘：它保留多个 Query heads 的同时，用较少的 KV heads 显著降低缓存和读取压力，又没有像 MQA 那样把所有 K/V 完全共享。

## 9. 总结

这三种结构的区别可以压缩成一个变量：

<p align="center">$N_{KV}:\quad N_h\rightarrow g\rightarrow1$</p>

对应：

```text
MHA → GQA → MQA
独立 K/V 最多 → 分组共享 → 全部共享
KV Cache 最大 → 折中 → 最小
```

最重要的是不要把它理解为“减少 Attention head”：

> MQA 和 GQA 通常保留 Query head 数，减少的是 K/V head 数；因此它们直接降低了 KV Cache 容量和 Decode 阶段的历史 K/V 读取量。

下一篇将继续沿着“每个 token 到底要缓存多少数据”这个问题，进入 DeepSeek-V2 提出的 Multi-head Latent Attention。它不再只是共享若干 K/V heads，而是尝试把 K/V 压进一个更小的潜在向量中。

## 系列导航

- [前篇：KV Cache 基础]({% post_url 2026-09-07-技术-KV Cache与Transformer推理 %})
- 本篇：MQA 与 GQA
- [第二篇：MLA]({% post_url 2026-09-07-技术-KV Cache优化二MLA %})
- [第三篇：局部注意力与缓存淘汰]({% post_url 2026-09-07-技术-KV Cache优化三局部注意力与缓存淘汰 %})
- [第四篇：投机解码]({% post_url 2026-09-08-技术-KV Cache优化四投机解码 %})

## 参考资料

- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150), 2019.
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245), 2023.
