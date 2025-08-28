---
layout:     post
title:      "FlashAttention 中的 Safe Softmax"
subtitle:   "从数值稳定性到分块重缩放"
date:       2025-08-28 12:00:00 +0800
author:     "Liu Mengxuan"
header-img: "img/post-bg-miui6.jpg"
categories: [技术]
tags: [技术, FlashAttention, Softmax, 数值稳定性, CUDA]
---

> 本文面向**仅熟悉标准 Softmax**的读者，用逐行推导的方式拆解 FlashAttention 为解决数值溢出与显存爆炸而引入的 Safe Softmax 及其分块实现。附可直接落地的 PyTorch 伪码。

---

## 1. 标准 Softmax 的数值陷阱

给定向量 \( x = [x_1, x_2, \dots, x_N] \)，标准 Softmax 为  

\[
\operatorname{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}
\]

**问题 1：上溢**  
FP16 最大可表示 ≈ 6.55×10⁴，而 \(e^{11} \approx 6×10^4\) 已逼近极限；FP32 临界值约 88。一旦分量先达到 `Inf`，后续计算全部变 `NaN`。

**问题 2：下溢**  
若所有 \(x_j\) 负且绝对值很大，和分母先变成 0，出现除零错误。

---

## 2. Safe Softmax：减最大值的三步公式

在分子、分母同乘常数 \(e^{-c}\) 不改变结果：

\[
\frac{e^{x_i}}{\sum_j e^{x_j}}
= \frac{e^{x_i - c}}{\sum_j e^{x_j - c}}
\]

选 **\(c = \max_j x_j\)** 后：

- 最大指数项 \(e^0 = 1\)，远离上溢；  
- 所有项非负，远离下溢。

由此得到 **Safe Softmax** 的三步流程：

1. 全局最大值  
   \[
   m = \max_{1\le j\le N} x_j
   \tag{1}
   \]

2. 指数和  
   \[
   \ell = \sum_{j=1}^{N} e^{x_j - m}
   \tag{2}
   \]

3. 归一化  
   \[
   y_i = \frac{e^{x_i - m}}{\ell}
   \tag{3}
   \]

---

## 3. 分块动机：SRAM 容不下整条向量

GPU 的两级存储：

| 存储 | 容量 | 带宽 |
|---|---|---|
| HBM（Global Memory） | 数十 GB | 低 |
| SRAM（Shared Memory） | 每 SM < 200 KB | 高 |

当序列长度 \(N\) 达到 8 k、16 k 乃至更高时，即使向量本身也放不进 SRAM，更不用说 Attention 的 \(N\times N\) 矩阵。FlashAttention 将输入切成 **tile**，每块大小 \(B \ll N\)，在 SRAM 中完成计算。

---

## 4. 分块 Safe Softmax：重缩放公式的逐行推导

### 4.1 符号约定

- 原向量 \(x \in \mathbb{R}^{N}\)  
- 切成两块（可推广）：  
  \[
  x = [x^{(1)}, x^{(2)}],\quad x^{(1)} \in \mathbb{R}^{B},\ x^{(2)} \in \mathbb{R}^{N-B}
  \]

### 4.2 子块先算局部安全值

| 子块 | 局部最大值 | 局部指数和（已减局部最大值） |
|---|---|---|
| 1 | \(m^{(1)} = \max x^{(1)}\) | \(\displaystyle \ell^{(1)} = \sum_{i=1}^{B} e^{x^{(1)}_i - m^{(1)}}\) |
| 2 | \(m^{(2)} = \max x^{(2)}\) | \(\displaystyle \ell^{(2)} = \sum_{i=1}^{N-B} e^{x^{(2)}_i - m^{(2)}}\) |

### 4.3 全局最大值

\[
m = \max\!\bigl(m^{(1)}, m^{(2)}\bigr)
\tag{4}
\]

### 4.4 显微镜：重缩放公式（第三步）

我们希望得到 **全局尺度** 下的指数和  
\[
\sum_{\text{子块 1}} e^{x_i - m}
\]

但手上只有 **局部尺度** 的  
\[
\ell^{(1)} = \sum_{\text{子块 1}} e^{x_i - m^{(1)}}
\]

两者差一个 **整体平移量** \(m^{(1)} - m\)：

\[
\sum_{\text{子块 1}} e^{x_i - m}
&= \sum_{\text{子块 1}} e^{\bigl(x_i - m^{(1)}\bigr) + (m^{(1)} - m)} \\
&= e^{m^{(1)} - m}\sum_{\text{子块 1}} e^{x_i - m^{(1)}} \\
&= e^{m^{(1)} - m}\cdot \ell^{(1)}
\tag{5}
\]

同理，子块 2：  
\[
\sum_{\text{子块 2}} e^{x_i - m}
= e^{m^{(2)} - m}\cdot \ell^{(2)}
\tag{6}
\]

于是 **全局分母**  
\[
\ell = e^{m^{(1)} - m}\,\ell^{(1)} + e^{m^{(2)} - m}\,\ell^{(2)}
\tag{7}
\]

---

## 5. 复杂度与内存访问对比

| 指标 | 朴素实现 | 分块 Safe Softmax |
|---|---|---|
| 显存峰值 | \(O(N)\) 向量 + \(O(N^2)\) 矩阵 | \(O(B)\) tile |
| HBM 读写 | \(O(N^2)\) | \(O(N)\) |
| 数值稳定 | ❌ 易溢出 | ✅ 远离溢出 |
| 数学精度 | — | bit-wise 一致 |
