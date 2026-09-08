---
layout:     post
title:      "FlashAttention：从 Safe Softmax 到分块计算"
subtitle:   "用在线归一化把 Attention 的中间结果留在片上"
date:       2026-09-08 22:00:00 +0800
author:     "Liu Mengxuan"
mathjax:    true
header-img: "img/post-bg-miui6.jpg"
categories: [技术]
tags:       [技术, AI Infra, Transformer, Attention, FlashAttention, CUDA, Softmax, HBM]
---

FlashAttention 的名字很容易让人以为它提出了一个新的 Attention 公式。实际上，它并没有改变 Attention 的数学结果：给定相同的 `Q`、`K`、`V`，输出仍然是普通的精确 Attention。

它真正改变的是计算过程：不把完整的 `N×N` 注意力分数矩阵和概率矩阵写回 HBM，而是把 `Q`、`K`、`V` 切成 tile，在 GPU 片上的 SRAM 和 registers 中完成局部计算，并用在线 Softmax 的状态把不同 tile 合并起来。

如果用一句话概括：

> FlashAttention 不是少算了 Attention，而是减少了中间结果在 HBM 和片上存储之间的往返。

本文从普通 Softmax 的数值稳定性开始，依次走到 Safe Softmax、Online Softmax，再把它们放进 Attention 的分块计算中，最后简单介绍 FlashAttention-2 和 FlashAttention-3 的优化重点。

<p align="center">
  <img src="/img/in-post/ai-infra-flashattention-hbm.svg" alt="朴素 Attention 与 FlashAttention 的 HBM 访问对比" style="max-width: 100%;">
</p>

## 1. 先回到 Attention 的完整公式

为了聚焦 IO，先只看一个 attention head，并把 batch 和 head 维度省略。设序列长度为 `N`，每个 head 的维度为 `d`：

<p align="center">$Q,K,V\in\mathbb{R}^{N\times d}$</p>

Attention 可以拆成三步：

<p align="center">$S=\dfrac{QK^T}{\sqrt d}$</p>

<p align="center">$P=\operatorname{softmax}(S)$</p>

<p align="center">$O=PV$</p>

合起来就是：

<p align="center">$O=\operatorname{softmax}\left(\dfrac{QK^T}{\sqrt d}\right)V$</p>

这里的形状是：

```text
Q       [N, d]
K       [N, d]
QKᵀ     [N, N]       每个 Query 对每个 Key 的分数
P       [N, N]       每个 Query 对所有 Key 的概率
V       [N, d]
O       [N, d]
```

`S` 和 `P` 都是 `N×N`。序列从 4K 增长到 32K 时，它们的元素数量会按平方增长。这些矩阵未必比模型权重更大，却经常成为 Attention 的显存和带宽负担。

### 1.1 单独看一行，就更容易理解

固定第 `i` 个 Query，令：

<p align="center">$s_i=q_iK^T=[s_{i1},s_{i2},\ldots,s_{iN}]$</p>

它是一条长度为 `N` 的分数向量。经过 Softmax 后得到：

<p align="center">$p_{ij}=\dfrac{e^{s_{ij}}}{\sum_{t=1}^{N}e^{s_{it}}}$</p>

最后用这条概率向量加权所有 Value：

<p align="center">$o_i=\sum_{j=1}^{N}p_{ij}v_j$</p>

FlashAttention 的核心，就是把这条长度为 `N` 的计算切成若干小段，却仍然得到同一个 `o_i`。

## 2. 为什么普通 Softmax 需要 Safe Softmax？

Softmax 的直接形式是：

<p align="center">$y_i=\dfrac{e^{x_i}}{\sum_j e^{x_j}}$</p>

指数运算很容易造成数值问题。例如在 FP16 中，`exp(11)` 已经接近可表示范围的上限；如果 `x` 中有更大的数，分子或分母可能变成 `Inf`。相反，如果所有输入都很负，指数又可能下溢成 0。

Safe Softmax 使用一个简单的等价变形。分子和分母同时乘以 `e^{-c}`：

<p align="center">$\dfrac{e^{x_i}}{\sum_j e^{x_j}}=\dfrac{e^{x_i-c}}{\sum_j e^{x_j-c}}$</p>

取：

<p align="center">$m=\max_jx_j$</p>

于是：

<p align="center">$y_i=\dfrac{e^{x_i-m}}{\sum_j e^{x_j-m}}$</p>

因为所有 `x_i-m≤0`，所以：

<p align="center">$0<e^{x_i-m}\le1$</p>

这就是 Safe Softmax 的核心：**把最大值移到指数的零点，避免指数向正方向爆炸**。

### 2.1 一个小例子

假设分数是：

```text
x = [1, 2, 3]
m = 3
```

直接计算会出现 `e¹、e²、e³`；Safe Softmax 改成：

```text
x - m = [-2, -1, 0]
exp(x - m) = [e⁻², e⁻¹, 1]
```

概率仍然完全相同，因为分子和分母被同时除以 `e³`。这里的“完全相同”指数学等价；实际浮点计算仍可能因为舍入而出现极小差异。

<p align="center">
  <img src="/img/in-post/ai-infra-safe-online-softmax.svg" alt="Safe Softmax 与 Online Softmax 的参考值和状态合并" style="max-width: 100%;">
</p>

## 3. 从 Safe Softmax 到 Online Softmax

Safe Softmax 已经解决了指数溢出，但它通常需要先知道整条向量的最大值 `m`，再计算指数和，最后归一化：

```text
第一遍：扫描 x，找全局最大值 m
第二遍：扫描 x，计算 l = Σ exp(xᵢ - m)
第三遍：扫描 x，输出 exp(xᵢ - m) / l
```

如果只计算 Softmax 向量，这种多遍扫描已经可以接受。但在 Attention 中，我们最终真正需要的是：

<p align="center">$o=\sum_i\operatorname{softmax}(s)_iv_i$</p>

我们并不一定需要把整条概率向量永久保存下来。于是可以在扫描分数的同时，直接维护加权 Value 的分子。

### 3.1 用两个状态表示一整段分数

对一段分数 `s`，维护三个量：

<p align="center">$m=\max_i s_i$</p>

<p align="center">$\ell=\sum_i e^{s_i-m}$</p>

<p align="center">$u=\sum_i e^{s_i-m}v_i$</p>

最终输出是：

<p align="center">$o=\dfrac{u}{\ell}$</p>

其中：

- `m` 是当前看到的最大分数；
- `ℓ` 是以 `m` 为参考值的指数和；
- `u` 是同一参考系下的加权 Value 和。

### 3.2 新来一块，怎样和旧状态合并？

假设已经处理完旧块，手里有：

<p align="center">$m_{old},\quad \ell_{old},\quad u_{old}$</p>

现在读入新块，它自己的局部统计量是：

<p align="center">$m_b=\max(s_b)$</p>

<p align="center">$\ell_b=\sum_{j\in b}e^{s_j-m_b}$</p>

<p align="center">$u_b=\sum_{j\in b}e^{s_j-m_b}v_j$</p>

新旧两块必须先换到同一个参考值：

<p align="center">$m_{new}=\max(m_{old},m_b)$</p>

旧状态和新状态分别重缩放：

<p align="center">$\ell_{new}=\ell_{old}e^{m_{old}-m_{new}}+\ell_be^{m_b-m_{new}}$</p>

<p align="center">$u_{new}=u_{old}e^{m_{old}-m_{new}}+u_be^{m_b-m_{new}}$</p>

最后继续使用：

<p align="center">$o_{new}=\dfrac{u_{new}}{\ell_{new}}$</p>

这就是 Online Softmax 的合并规则。

### 3.3 “换基”这个直觉应该怎样理解？

把它理解为“换基”很有帮助，但需要加一个严格限定：这里并不是线性代数中把向量从一组正交基换到另一组基，而是**给指数运算更换参考零点**。

例如第一块分数 `[1,2]` 的局部最大值是 `2`，它使用 `2` 作为零点：

<p align="center">$e^{1-2}+e^{2-2}=e^{-1}+1$</p>

第二块分数 `[3]` 的局部最大值是 `3`，它使用 `3` 作为零点：

<p align="center">$e^{3-3}=1$</p>

两块的“1”并不处在同一个尺度上。第二块的真实分数整体比第一块高，所以合并时要把第一块的结果乘上：

<p align="center">$e^{2-3}=e^{-1}$</p>

于是：

<p align="center">$\ell=e^{-1}(e^{-1}+1)+1=e^{-2}+e^{-1}+1$</p>

这正是对完整向量 `[1,2,3]` 使用全局最大值 `3` 后得到的分母。

如果令 `v=[10,20,30]`，则第一块的未归一化加权和为：

<p align="center">$u_1=10e^{-1}+20$</p>

将它换到全局参考值 `3` 后，再加上第二块：

<p align="center">$u=e^{-1}u_1+30$</p>

最后 `u/ℓ` 与一次性对 `[1,2,3]` 做完整 Softmax 后再加权 `v` 的结果相同。Online Softmax 只需要不断更新 `m、ℓ、u`，不需要把所有中间概率都保存下来。

## 4. Attention 的真正瓶颈：不是只算得多，而是搬得多

GPU 的 HBM 容量大，但访问延迟和带宽成本都高；SRAM、shared memory 和 registers 容量小，却更适合保存当前正在使用的数据。

可以先用这个简化层次建立直觉：

```text
HBM / Global Memory：容量大，模型和长序列数据主要在这里
        ↓
L2 Cache
        ↓
Shared Memory / L1：每个 SM 上的片上空间
        ↓
Registers：线程私有，最快但容量最小
```

朴素 Attention 的数据流可以写成：

```text
Q、K 从 HBM 读入
        ↓
计算 S = QKᵀ
        ↓
把 N×N 的 S 写回 HBM
        ↓
再从 HBM 读 S，计算 Softmax
        ↓
把 N×N 的 P 写回 HBM
        ↓
再从 HBM 读 P，与 V 计算 PV
        ↓
把 O 写回 HBM
```

这里最浪费的是 `S` 和 `P`：它们只是后续计算的中间结果，却拥有 `N×N` 的规模，还要经历写回和重新读取。即使 HBM 能容纳它们，搬运这些数据也会占用大量带宽。

FlashAttention 的优化目标可以说得更准确一些：

> 不追求让 `QKᵀ` 少做数学乘法，而是避免让 `N×N` 的中间分数和概率矩阵落到 HBM。

这也是为什么 FlashAttention 被称为 IO-aware Attention。

## 5. FlashAttention：把 Attention 切成 tile

设 `Q` 按行切成 Query blocks，`K` 和 `V` 按序列维切成 Key/Value blocks：

```text
Q = [Q₁, Q₂, ...]
K = [K₁, K₂, ...]
V = [V₁, V₂, ...]
```

对某个固定的 Query block `Qᵢ`，依次处理所有 `Kⱼ、Vⱼ`：

```text
1. 从 HBM 读取 Qᵢ，放到 SRAM / registers
2. 读取一个 Kⱼ、Vⱼ tile
3. 在片上计算 Sᵢⱼ = QᵢKⱼᵀ
4. scale、应用 causal mask
5. 对 Sᵢⱼ 做局部 Safe Softmax
6. 用 Online Softmax 合并 m、ℓ、O 状态
7. 释放当前 Kⱼ、Vⱼ tile，读取下一个 tile
8. 所有 K/V tile 完成后，把最终 Oᵢ 写回 HBM
```

单个 tile 内的计算闭环是：

<p align="center">$Q_iK_j^T\rightarrow\text{scale/mask}\rightarrow\text{局部 Softmax}\rightarrow P_{ij}V_j$</p>

但局部的 `Sᵢⱼ` 和 `Pᵢⱼ` 只在片上短暂存在。它们用完就被下一个 tile 覆盖，不会组成完整的 `N×N` 矩阵。

<p align="center">
  <img src="/img/in-post/ai-infra-flashattention-tiling.svg" alt="FlashAttention 使用 Query block 和 Key Value block 进行分块计算" style="max-width: 100%;">
</p>

### 5.1 为什么分块仍然得到完整结果？

以第 `i` 个 Query 行为例，完整结果是：

<p align="center">$o_i=\dfrac{\sum_{j=1}^{N}e^{s_{ij}-m_i}v_j}{\sum_{j=1}^{N}e^{s_{ij}-m_i}}$</p>

FlashAttention 将 `j=1...N` 切成多个块。每个块只计算一部分分子和分母，然后通过 Online Softmax 的重缩放规则，把旧块和新块搬到同一个最大值参考系中。

所以它改变了：

```text
先生成完整 S、完整 P，再做 PV
```

变成：

```text
读取一个 K/V tile
    → 计算局部分数
    → 局部归一化
    → 更新输出状态
    → 释放 tile
```

它没有丢掉任何历史位置，也没有近似截断注意力范围。只要 tile 内的 mask、重缩放和累加实现正确，数学上仍然覆盖全部 `K/V`。

### 5.2 Causal Mask 放在哪里？

对于 Decoder 的因果 Attention，Query 位置不能读取未来 Key。分块后仍然要遵守这个规则：

<p align="center">$s_{ij}=-\infty\quad\text{if }j>i$</p>

在 tile 内，未来位置的分数被设为负无穷，于是：

<p align="center">$e^{-\infty}=0$</p>

这些位置既不会影响局部最大值，也不会进入指数和或 Value 汇总。FlashAttention 只是把 Mask 融进 tile 计算，因果语义没有改变。

## 6. FlashAttention 到底减少了哪些 HBM 访问？

最容易出现的误解是：“FlashAttention 把所有数据都只从 HBM 读一次。”这并不准确。为了让不同 Query blocks 都能使用 K/V，某些 K/V tile 可能会被不同的 block 重复加载，具体次数取决于 tile 形状、循环顺序、缓存命中和 kernel 实现。

它减少的重点是 **中间结果的 HBM 读写**：

| 中间数据 | 朴素 Attention | FlashAttention |
| --- | --- | --- |
| 分数 `S=QKᵀ` | 通常物化为 `N×N` 并写回 | tile 内短暂计算，不落地完整矩阵 |
| 概率 `P=softmax(S)` | 通常物化为 `N×N` 并再次读写 | tile 内使用，直接参与 `PV` |
| 输出 `O` | 最终写回 | 最终写回，必要时保留片上累加 |
| Softmax 状态 | 可能隐含在中间矩阵中 | 每个 Query 行只维护 `m、ℓ、u/O` |

因此，FlashAttention 并不是让所有 HBM 流量变成 `O(N)`，而是把最昂贵的 `N×N` 中间矩阵读写去掉，并通过片上数据复用降低 IO。对于训练前向，额外的 Attention 中间存储从平方级降到与输出和少量状态相关的线性级；完整的 IO 复杂度还会随 tile 大小、`d` 和片上存储容量变化。

可以用一个更贴近硬件的数据流总结：

```text
HBM：保存 Q、K、V 和最终 O
片上 SRAM：保存当前 Q/K/V tile 与局部计算
registers：保存每个 Query 行的 m、ℓ、u
HBM：不保存完整 N×N 的 S 和 P
```

这就是它减少 HBM 访问的根本原因：**中间结果不再跨存储层级往返，而是在产生它的地方直接被消费。**

## 7. FlashAttention 的前向伪代码

下面的伪代码省略了 batch、head、边界和 kernel 并行细节，只保留数学数据流：

```python
def flash_attention_forward(Q, K, V, scale):
    O = zeros_like(Q)

    for q_block in split_rows(Q):
        # 每个 Query 行都有自己的在线状态
        m = full(q_block.rows, -inf)
        l = zeros(q_block.rows)
        u = zeros(q_block.rows, V.width)

        for k_block, v_block in zip(split_rows(K), split_rows(V)):
            scores = q_block @ k_block.T * scale
            scores = apply_causal_mask(scores)

            valid_rows = any_finite_in_each_row(scores)
            if not any(valid_rows):
                continue

            # 对当前 tile 中至少有一个可见位置的行进行在线更新。
            scores_v = scores[valid_rows]
            m_block = row_max(scores_v)
            weights = exp(scores_v - m_block[:, None])
            l_block = row_sum(weights)
            u_block = weights @ v_block

            m_old = m[valid_rows]
            m_new = maximum(m_old, m_block)
            old_scale = exp(m_old - m_new)
            block_scale = exp(m_block - m_new)

            l[valid_rows] = (
                l[valid_rows] * old_scale + l_block * block_scale
            )
            u[valid_rows] = (
                u[valid_rows] * old_scale[:, None]
                + u_block * block_scale[:, None]
            )
            m[valid_rows] = m_new

        O[q_block.rows] = u / l[:, None]

    return O
```

这里的 `weights` 只表示当前 tile 的局部权重，并不意味着要把整条 `N` 长度的概率向量保存下来。真实 kernel 还会把 QK、Softmax、PV 以及数据搬运进一步融合，使用 shared memory、registers 和 Tensor Core。

## 8. 训练时为什么还可以省显存？

训练需要反向传播，直觉上似乎必须保存前向的 `S` 和 `P`。FlashAttention 的做法是保存较小的辅助信息，例如每个 Query 行的 LogSumExp 或等价的 Softmax 统计量；反向时重新读取 `Q/K/V`，分块重算需要的局部分数和概率。

这是一种典型的取舍：

```text
多保存 N×N 中间矩阵：反向重算少，但显存占用大
少保存统计量：显存占用小，但反向需要重算部分中间结果
```

在大模型训练中，显存容量和 HBM 带宽往往比少做一部分重算更宝贵，因此这种重计算换显存的策略很有价值。它不改变训练目标，只改变中间状态的保存策略。

## 9. FlashAttention-2 和 FlashAttention-3 简单看什么？

FlashAttention-1 建立了主要数学和 IO 思路：分块、片上计算、Online Softmax，以及不物化完整注意力矩阵。

后续版本的重点更多是让这条算法更充分地使用 GPU，而不是重新发明一套 Attention 公式。

<p align="center">
  <img src="/img/in-post/ai-infra-flashattention-evolution.svg" alt="FlashAttention 1、2、3 的优化重点概览" style="max-width: 100%;">
</p>

### 9.1 FlashAttention-2：改善并行与工作划分

FlashAttention-2 主要关注 V1 中的并行效率和线程协作：

- 更好地沿 Query / 序列维度并行，让更多线程块同时工作；
- 重新安排一个 thread block 内 warp 的职责，减少不必要的 shared memory 通信和同步；
- 降低非矩阵乘部分的开销，让 Tensor Core 承担更高比例的计算；
- 针对不同 batch、head 和序列长度改善负载均衡。

可以把 V1 理解为“先把正确的分块算法跑起来”，把 V2 理解为“重新安排谁处理哪一块、什么时候同步”，使相同的数学流程更接近 GPU 的并行结构。

### 9.2 FlashAttention-3：利用 Hopper 的异步硬件

FlashAttention-3 面向 NVIDIA Hopper 等更新的 GPU 架构，重点利用新硬件提供的异步能力：

- 使用异步数据搬运，让 HBM 到片上的传输与计算重叠；
- 通过 warp specialization，让不同 warp 分别承担数据搬运、矩阵计算或 Softmax 等职责；
- 利用 TMA、WGMMA 等 Hopper 相关机制提高 Tensor Core 的供给效率；
- 针对 FP8 等低精度路径处理缩放、累加和误差控制。

因此 V3 的核心可以概括为：**把“搬数据”和“做矩阵乘”安排成流水线，尽量不要让任何一方空等另一方。** 这些优化依赖具体硬件，不应简单移植成所有 GPU 都通用的规则。

<p align="center">$\text{FlashAttention-1：减少中间 IO}\rightarrow\text{FlashAttention-2：改善并行}\rightarrow\text{FlashAttention-3：重叠搬运与计算}$</p>

## 10. FlashAttention 没有做什么？

为了准确理解它，也要明确几个边界：

1. 它没有把完整 Attention 变成稀疏 Attention，默认仍然计算所有允许的 Query-Key 配对；
2. 它没有消除 `QKᵀ` 的 `O(N²)` 数学计算量，主要优化的是 IO 和中间存储；
3. 它不是近似算法，目标是得到与标准 Attention 数学等价的结果，但浮点运算顺序改变后不保证逐位相同；
4. 它不是只适用于训练，Prefill 的长序列计算也非常适合使用分块 Attention；
5. Decode 阶段只有一个或少量 Query 时，问题形态不同，通常还需要 Flash-Decoding 等针对长 KV Cache 的策略。

## 11. 总结：从公式到硬件的一条链

FlashAttention 的完整逻辑可以压缩成下面这条链：

```text
普通 Softmax
    ↓ 减去最大值
Safe Softmax：数值稳定
    ↓ 分块后每块有自己的最大值
Online Softmax：维护 m、ℓ、u 并重缩放合并
    ↓ 把 K/V 切成 tile
FlashAttention：片上完成 QKᵀ → Softmax → PV
    ↓ 不物化完整 N×N 中间矩阵
减少 HBM 往返，提高数据复用和 GPU 利用率
```

最值得记住的不是某一条复杂公式，而是两个问题：

- **数学上**：不同 tile 的局部 Softmax，怎样通过统一参考值和重缩放得到全局结果？
- **工程上**：哪些数据必须跨步骤保存，哪些中间结果可以在 SRAM 中产生后立即消费？

Online Softmax 回答了第一个问题，FlashAttention 则把这个答案嵌入了 Attention 的分块数据流中。后续的 FlashAttention-2 和 FlashAttention-3，主要是在并行组织、线程协作、异步流水线和硬件特性利用上继续推进。

## 参考资料

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08691)
- [CUDA Online Softmax 实现优化](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/52-cuda-online-softmax实现/)
