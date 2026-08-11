---
layout:     post
title:      "AI Infra 线性代数基础：从 Shape 到 Attention、SVD 与 LoRA"
subtitle:   "把抽象的 2.1—2.13 重新排成一条能走通的学习路线"
date:       2026-08-11 20:00:00 +0800
author:     "Liu Mengxuan"
mathjax:    true
header-img: "img/post-bg-miui6.jpg"
categories: [技术]
tags:       [技术, AI Infra, 线性代数, Transformer, Attention, SVD, LoRA]
---

最近在学习 AI Infra 的数学基础时，我读到了从“标量、向量、矩阵和张量”一直讲到 LoRA 的一组内容。每个概念单独看都不算复杂，但原来的顺序更像一份概念索引：定义出现得很快，例子很少，张量 shape、矩阵运算、数值误差和模型压缩又混在同一层级里，读完很难建立一条连贯的主线。

经过一轮逐项讨论，我发现更自然的学习方式不是机械地按照编号走，而是围绕几个逐渐深入的问题展开：

```text
数据装在哪里？
    ↓
不同 shape 的数据怎样计算？
    ↓
向量怎样比较大小和相似度？
    ↓
矩阵乘法怎样改变 shape？
    ↓
Attention 为什么是批量矩阵乘法？
    ↓
大矩阵为什么可以低秩压缩？
    ↓
SVD、LoRA 和特征值分别解决什么问题？
```

本文按照这条路线，重新整理 AI Infra 中最常用的线性代数基础。贯穿全文的例子统一采用：

<p align="center">$B=2,\qquad S=3,\qquad H=4$</p>

也就是一次处理 2 条序列，每条序列有 3 个 token，每个 token 用一个 4 维隐藏向量表示。

## 1. 从数据容器开始：标量、向量、矩阵和张量

线性代数里的这些名字，本质上描述的是数据有多少个轴。

| 名称 | 示例 | shape | 直觉 |
| --- | --- | --- | --- |
| 标量 | `3.14` | `[]` | 一个数字 |
| 向量 | `[1, 2, 3, 4]` | `[4]` | 一排数字 |
| 矩阵 | 2 行 3 列 | `[2,3]` | 一张表 |
| 张量 | 多个矩阵继续堆叠 | `[B,S,H]` | 多轴数组 |

Transformer 的 hidden states 通常写作：

<p align="center">$X\in\mathbb{R}^{B\times S\times H}$</p>

当 `X.shape = [2,3,4]` 时，它表达的是：

```text
2 个 batch
每个 batch 有 3 个 token
每个 token 是一个 4 维向量
```

因此，一个具体 token 的隐藏表示可以写成：

<p align="center">$X_{b,s,:}\in\mathbb{R}^{H}$</p>

这里的冒号表示取出该 token 在隐藏维上的全部分量。

理解高阶张量时，不需要试图在脑中画出四维、五维空间。更实用的方法是不断问：

> 固定前面的索引以后，最后还剩下一个什么对象？

对于 `[B,S,H]`，固定 `b` 和 `s` 后，剩下的就是长度为 `H` 的 token 向量。

## 2. Shape、索引和 stride：逻辑结构与物理布局

`shape` 说明每个维度有多长，但它不一定说明数据在内存中怎样排列。`stride` 表示沿某个轴移动一步，需要跨过多少个底层元素。

在常见的连续布局 `[B,S,H]` 中，`H` 是最后一维，同一个 token 的隐藏分量通常连续存放。因此：

```text
X[b, s, 0], X[b, s, 1], ..., X[b, s, H-1]
```

在内存中彼此相邻。

这件事对 AI Infra 很重要，因为同样的数学运算，如果访存模式不同，实际性能可能完全不同。`transpose`、`permute` 等操作有时只改变 shape 和 stride，并不立即搬运数据；但后续 Kernel 如果要求连续内存，就可能触发额外复制。

所以看一个张量时，需要区分两个问题：

```text
逻辑上：每一维表示什么？
物理上：这些元素在内存中怎样排列？
```

## 3. 逐元素运算与广播：让 shape 对齐

逐元素运算表示相同位置的元素分别计算，例如：

<p align="center">$C_{i,j}=A_{i,j}+B_{i,j}$</p>

当两个张量 shape 不完全相同时，框架可能使用广播。典型例子是给 hidden states 加 bias：

<p align="center">$X\in\mathbb{R}^{B\times S\times H},\qquad b\in\mathbb{R}^{H}$</p>

为了理解广播，可以把 `b` 的 shape 对齐为：

<p align="center">$[H]\rightarrow[1,1,H]\rightarrow[B,S,H]$</p>

这里的 `1` 表示“这个维度的长度是 1”，不是数据里的数值 1。

对于 `B=2、S=3、H=4`：

```text
b.shape = [4]
        ↓ 前面补两个维度
        [1,1,4]
        ↓ 沿 B 和 S 复用
        [2,3,4]
```

同一个四维偏置向量会加到 6 个 token 位置上。广播发生在 `B` 和 `S`，`H` 中的四个分量并没有互相复制或混合。

<p align="center">
  <img src="/img/in-post/ai-infra-linear-algebra-broadcasting.png" alt="B、S、H 三个维度上的广播过程" style="max-width: 100%;">
</p>

广播一般是逻辑扩展，不一定真的预先复制出一个完整张量。但错误的 shape 也可能“恰好可以广播”，导致代码不报错、语义却错了。因此每次广播都应该明确：

1. 哪些维度长度相等？
2. 哪些维度长度为 1？
3. 到底准备沿哪些维度复用？

## 4. 点积：相似度与加权求和

两个长度相同的向量可以做点积：

<p align="center">$x^\top y=\sum_{i=1}^{H}x_i y_i$</p>

它有两种常用理解。

第一种是加权求和：用一个向量中的数字给另一个向量的各个分量分配权重，然后全部加起来。

第二种是方向匹配：

<p align="center">$x^\top y=\lVert x\rVert_2\lVert y\rVert_2\cos\theta$</p>

点积越大，通常说明两个向量越同向；点积为负，通常说明方向相反。但点积同时受到方向和向量长度影响，因此只想比较方向时，需要归一化：

<p align="center">$\cos\theta=\dfrac{x^\top y}{\lVert x\rVert_2\lVert y\rVert_2}$</p>

Attention 中用 Query 和 Key 的点积衡量匹配程度，Embedding 检索中也经常使用归一化后的点积。

## 5. 范数：把高维对象变成一个“规模”

范数可以理解为一把衡量“有多大”的尺子。常见向量范数包括：

<p align="center">$\lVert x\rVert_1=\sum_i|x_i|$</p>

<p align="center">$\lVert x\rVert_2=\sqrt{\sum_i x_i^2}$</p>

<p align="center">$\lVert x\rVert_\infty=\max_i|x_i|$</p>

例如 `x=[3,4]`：

```text
L1  范数 = 7
L2  范数 = 5
L∞  范数 = 4
```

它们不是在争论哪个才是真正的长度，而是在关注不同的规模：总量、欧氏距离和最大分量。

### 5.1 什么叫沿 H 维归约？

对于：

<p align="center">$X\in\mathbb{R}^{B\times S\times H}$</p>

沿隐藏维 `H` 求 L2 范数，表示对每个 token 的 `H` 个分量分别求长度：

<p align="center">$n_{b,s}=\sqrt{\sum_{h=1}^{H}X_{b,s,h}^2}$</p>

原来每个 token 有 `H` 个数，归约后只剩一个数，因此：

<p align="center">$[B,S,H]\rightarrow[B,S]$</p>

“沿某个维度归约”的含义就是：沿着那个轴把一组数字通过求和、平均、最大值或范数等操作合并成更少的数字。默认情况下，被归约的维度会消失；如果保留维度，则会变成长度为 1 的轴。

### 5.2 在模型中有什么用？

范数常用于：

- 观察参数、激活和梯度是否过大；
- 梯度裁剪；
- 向量归一化；
- 比较参考实现和优化 Kernel 的误差；
- 对权重施加正则化约束。

Transformer 中的 RMSNorm 也会沿 `H` 处理每个 token：

<p align="center">$\operatorname{RMS}(x)=\sqrt{\dfrac{1}{H}\sum_{i=1}^{H}x_i^2}$</p>

它与 L2 范数关系密切，只是多了平均操作。目的不是丢掉 hidden states，而是用这个统计量把向量的数值规模控制在稳定范围内。

### 5.3 绝对误差与相对误差

比较浮点结果时，通常同时考虑：

<p align="center">$e_{abs}=|\hat{x}-x|,\qquad e_{rel}=\dfrac{|\hat{x}-x|}{|x|}$</p>

当参考值接近 0 时，相对误差可能被无限放大，因此测试通常组合两种容差：

<p align="center">$|\hat{x}-x|\le\text{atol}+\text{rtol}\cdot|x|$</p>

这对验证低精度计算、融合 Kernel 和算子优化是否正确非常重要。

## 6. 矩阵乘法：一组点积

设：

<p align="center">$A\in\mathbb{R}^{M\times K},\qquad B\in\mathbb{R}^{K\times N}$</p>

那么：

<p align="center">$C=AB\in\mathbb{R}^{M\times N}$</p>

shape 规则可以写成：

```text
(M, K) @ (K, N) -> (M, N)
```

中间的 `K` 被归约，外侧的 `M`、`N` 保留下来。结果中的每一个元素，都是第一矩阵的一行和第二矩阵的一列做点积：

<p align="center">$C_{i,j}=\sum_{k=1}^{K}A_{i,k}B_{k,j}$</p>

这就是为什么矩阵乘法不是逐元素相乘，也通常不能交换顺序。

### 6.1 转置为什么重要？

转置会交换矩阵的行和列：

<p align="center">$A\in\mathbb{R}^{M\times N}\quad\Rightarrow\quad A^T\in\mathbb{R}^{N\times M}$</p>

Attention 中的 `K` 原本和 `Q` 一样，最后一维都是 `D_h`。为了让 `D_h` 成为矩阵乘法中间被归约的维度，需要把 `K` 的最后两维转置。

单位矩阵 `I` 表示“不改变向量”的变换，逆矩阵表示“撤销原来的变换”。不过在深度学习工程中，通常不会为了求解问题而显式计算巨大矩阵的逆，更常见的是使用数值稳定的分解或求解算法。

## 7. Batched Matrix Multiplication：Attention 的核心形状

多头 Attention 中：

<p align="center">$Q\in\mathbb{R}^{B\times N_h\times S_q\times D_h}$</p>

<p align="center">$K\in\mathbb{R}^{B\times N_h\times S_k\times D_h}$</p>

转置 `K` 的最后两维：

<p align="center">$K^\top\in\mathbb{R}^{B\times N_h\times D_h\times S_k}$</p>

于是：

<p align="center">$QK^\top\in\mathbb{R}^{B\times N_h\times S_q\times S_k}$</p>

这里真正执行矩阵乘法的是最后两维：

```text
(S_q, D_h) @ (D_h, S_k) -> (S_q, S_k)
```

前面的 `(B,N_h)` 表示有多少组独立计算。换句话说，每一个 `(batch, head)` 都执行一次二维矩阵乘法，然后把所有结果堆叠起来。

例如：

```text
B=2, Nh=4, Sq=3, Sk=5, Dh=8

Q   : [2,4,3,8]
K^T : [2,4,8,5]
结果: [2,4,3,5]
```

总共有 `2×4=8` 组：

```text
[3,8] @ [8,5] -> [3,5]
```

最终的 `[S_q,S_k]` 表示每个 Query token 对所有 Key token 的注意力分数。

多头注意力也不是把 token 分给不同 head。所有 head 都会看到全部 token；真正被拆分的是每个 token 的隐藏特征：

<p align="center">$H=N_hD_h$</p>

<p align="center">
  <img src="/img/in-post/multi-head-attention-split-features-not-tokens.png" alt="多头注意力拆分的是隐藏特征而不是 token" style="max-width: 100%;">
</p>

## 8. 线性变换与仿射变换：神经网络的 Linear 层

严格的线性变换写作：

<p align="center">$y=Wx$</p>

神经网络中的 Linear 层通常还包含 bias：

<p align="center">$y=Wx+b$</p>

因为多了平移项 `b`，数学上更准确的名字是仿射变换。

从 shape 看，语言模型最后的词表投影可以写成：

<p align="center">$X\in\mathbb{R}^{(BS)\times H},\qquad W\in\mathbb{R}^{H\times V}$</p>

<p align="center">$Y=XW\in\mathbb{R}^{(BS)\times V}$</p>

它把每个 token 的 `H` 维 hidden state 投影成 `V` 个词表分数。矩阵乘法消掉 `H`，保留 token 数量 `BS`，并产生新的输出维度 `V`。

<p align="center">
  <img src="/img/in-post/transformer-final-projection-logits.png" alt="hidden states 经过输出投影得到词表 logits" style="max-width: 100%;">
</p>

连续堆叠多个没有激活函数的线性层，最终仍然可以合并成一个线性或仿射变换。因此深层网络真正获得复杂表达能力，还依赖激活函数、Attention、归一化等非线性结构。

## 9. 线性相关与秩：矩阵里到底有多少独立信息？

如果一个向量可以由其他向量线性组合出来，它就没有提供新的独立方向。

矩阵的秩 `rank` 可以理解成：

> 这个矩阵里真正独立的方向有多少个。

对于：

<p align="center">$A\in\mathbb{R}^{M\times N}$</p>

有：

<p align="center">$\operatorname{rank}(A)\le\min(M,N)$</p>

如果一个很大的矩阵只有少量独立方向，就可以尝试用两个更小的矩阵近似：

<p align="center">$A\approx UV$</p>

其中：

<p align="center">$U\in\mathbb{R}^{M\times r},\qquad V\in\mathbb{R}^{r\times N}$</p>

只要 `r` 很小，参数量就会从 `MN` 降为：

<p align="center">$r(M+N)$</p>

这就是低秩压缩和 LoRA 的数学起点。

## 10. SVD：找出矩阵最重要的方向

任意实矩阵都可以进行奇异值分解：

<p align="center">$A=U\Sigma V^\top$</p>

这个公式更容易通过“一个向量怎样经过 A”来理解：

<p align="center">$Ax=U\bigl(\Sigma(V^\top x)\bigr)$</p>

矩阵乘法从右向左执行：

```text
x
↓ V^T：把输入改写到一组特殊方向上
↓ Σ  ：沿这些方向分别缩放
↓ U  ：把结果映射到输出方向
Ax
```

这里的“调整方向”不是为了多做两次无意义的旋转，而是因为矩阵最强、最弱的作用方向通常不是原坐标轴。`V` 找到输入空间中的重要方向，`Σ` 记录各方向的缩放倍数，`U` 给出这些方向在输出空间中的对应方向。

`Σ` 对角线上的数字叫奇异值：

<p align="center">$\sigma_1\ge\sigma_2\ge\cdots\ge0$</p>

奇异值越大，说明矩阵沿对应方向的作用越强。如果后面的奇异值很小，就可以只保留前 `r` 个：

<p align="center">$A\approx U_r\Sigma_rV_r^\top$</p>

这相当于保留矩阵最重要的 `r` 个方向，舍弃影响较弱的方向，从而得到低秩近似。

SVD 可以用于矩阵压缩、降噪和结构分析，但对巨大模型权重做完整 SVD 本身也很昂贵。工程中常用截断、随机化算法，或者直接训练低秩因子。

## 11. LoRA：直接训练低秩更新

LoRA 不直接修改完整预训练权重 `W_0`，而是冻结它，只训练一个低秩更新：

<p align="center">$W=W_0+\Delta W$</p>

<p align="center">$\Delta W=\dfrac{\alpha}{r}BA$</p>

其中：

<p align="center">$A\in\mathbb{R}^{r\times d_{in}},\qquad B\in\mathbb{R}^{d_{out}\times r}$</p>

于是：

```text
[d_out, r] @ [r, d_in] -> [d_out, d_in]
```

更新矩阵的最终 shape 与原权重一致，但可训练参数量从：

<p align="center">$d_{out}d_{in}$</p>

变成：

<p align="center">$r(d_{out}+d_{in})$</p>

LoRA 背后的假设是：模型适配新任务时，真正需要新增的更新方向可能远少于完整权重空间的维度。因此可以用较小的 `r` 表达主要变化。

## 12. 特征值与特征向量：放到最后理解

特征值并不是理解 Transformer shape 的前置条件，更适合在掌握矩阵变换以后，用来分析系统的长期行为。

对于方阵 `A`，如果存在非零向量 `v` 满足：

<p align="center">$Av=\lambda v$</p>

那么：

- `v` 是特征向量，表示经过 `A` 后方向不变的特殊方向；
- `λ` 是特征值，表示该方向被缩放了多少倍。

如果反复应用矩阵：

<p align="center">$A^k v=\lambda^k v$</p>

那么：

```text
|λ| > 1：这个方向不断放大
|λ| < 1：这个方向不断衰减
|λ| = 1：这个方向规模保持稳定
λ < 0 ：缩放的同时发生方向翻转
```

这可以帮助理解动态系统稳定性、Hessian 的不同曲率方向，以及梯度经过多层 Jacobian 连乘后为什么可能爆炸或消失。

特征值主要针对方阵；神经网络中的权重矩阵经常是非方阵，所以分析一般矩阵时，奇异值往往更直接。

## 13. 把整条路线重新串起来

现在可以把 2.1—2.13 的内容重新组织为四层。

第一层是张量表示：

```text
标量、向量、矩阵、张量
→ shape、索引、stride
→ 逐元素运算与广播
```

第二层是神经网络的基本计算：

```text
点积与范数
→ 矩阵乘法与转置
→ Batched Matrix Multiplication
→ Attention
→ 线性层与输出投影
```

第三层是矩阵中的信息结构：

```text
线性相关
→ 秩
→ SVD
→ 低秩近似
→ LoRA
```

第四层是数值和稳定性分析：

```text
范数与误差
→ 激活/梯度规模
→ 特征值与系统稳定性
```

## 14. AI Infra 中读公式的统一方法

以后看到一个公式，可以固定问五个问题：

1. 每个张量的 shape 是什么？
2. 每一维在业务或模型中表示什么？
3. 哪些维度被归约，哪些维度保留？
4. 是否发生广播、转置、reshape 或数据复制？
5. 计算量、访存量和浮点误差可能在哪里出现？

例如：

<p align="center">$Y=XW,\qquad X\in\mathbb{R}^{(BS)\times H},\qquad W\in\mathbb{R}^{H\times V}$</p>

先不急着算数值，只看 shape：

```text
(BS, H) @ (H, V) -> (BS, V)
```

马上就能读出：`H` 被归约，每个 token 得到 `V` 个词表分数；计算规模大约与 `BSHV` 成正比；输出 logits 的存储规模大约与 `BSV` 成正比。

对 AI Infra 来说，线性代数不只是为了推导公式。它同时决定了张量 shape、Kernel 选择、访存模式、计算量、显存占用和数值稳定性。真正值得掌握的不是孤立地背诵定义，而是能从公式一路读到实现。
