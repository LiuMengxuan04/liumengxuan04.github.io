---
layout:     post
title:      "FlashAttention 中的 Safe Softmax"
subtitle:   "从数值稳定性到分块重缩放"
date:       2025-08-28 12:00:00 +0800
author:     "Liu Mengxuan"
mathjax: true
header-img: "img/post-bg-miui6.jpg"
categories: [技术]
tags: [技术, FlashAttention, Softmax, 数值稳定性, CUDA]
---

> 本文面向**仅熟悉标准 Softmax**的读者，用逐行推导的方式拆解 FlashAttention 为解决数值溢出与显存爆炸而引入的 Safe Softmax 及其分块实现。



## 1. 标准 Softmax 的数值陷阱
给定向量 $x=[x_1,x_2,\dots,x_N]$，标准 Softmax 为  
<p align="center">$\displaystyle \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}$</p>

**问题 1：上溢**  
FP16 最大 ≈ 6.55×10⁴，而 $e^{11}\approx 6\times10^4$ 已逼近极限；FP32 临界值约 88。一旦分量先达到 `Inf`，后续全部变 `NaN`。

**问题 2：下溢**  
若所有 $x_j$ 负且绝对值很大，分母先变成 0，出现除零错误。



## 2. Safe Softmax：减最大值的三步公式
在分子、分母同乘 $e^{-c}$ 不改变结果：

<p align="center">
$\displaystyle \frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{e^{x_i - c}}{\sum_j e^{x_j - c}}$
</p>

取 $c = \max_j x_j$ 后：

1. 全局最大值  
   <p align="center">$m = \max_{1\le j\le N} x_j$</p>
2. 指数和  
   <p align="center">$\ell = \sum_{j=1}^{N} e^{x_j - m}$</p>
3. 归一化  
   <p align="center">$y_i = \frac{e^{x_i - m}}{\ell}$</p>

## 3. 分块动机：SRAM 容不下整条向量
GPU 存储对比：

| 存储 | 容量 | 带宽 |
|---|---|---|
| HBM | 数十 GB | 低 |
| SRAM | 每 SM < 200 KB | 高 |

当 $N$ 达到 8 k、16 k 时，向量本身也放不进 SRAM。FlashAttention 将输入切成 **tile**，每块大小 $B \ll N$，在 SRAM 中完成计算。



## 4. 分块 Safe Softmax：重缩放公式逐行推导

### 4.1 符号约定
- 原向量 $x \in \mathbb{R}^{N}$  
- 切成两块：  
  $x = [x^{(1)}, x^{(2)}]$, 其中 $x^{(1)} \in \mathbb{R}^{B},\ x^{(2)} \in \mathbb{R}^{N-B}$

### 4.2 子块先算局部安全值
<table>
  <thead>
    <tr>
      <th>子块</th>
      <th>局部最大值</th>
      <th>局部指数和</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>$m^{(1)} = \max x^{(1)}$</td>
      <td>$\ell^{(1)} = \sum_{i=1}^{B} e^{x^{(1)}_i - m^{(1)}}$</td>
    </tr>
    <tr>
      <td>2</td>
      <td>$m^{(2)} = \max x^{(2)}$</td>
      <td>$\ell^{(2)} = \sum_{i=1}^{N-B} e^{x^{(2)}_i - m^{(2)}}$</td>
    </tr>
  </tbody>
</table>

### 4.3 全局最大值
<p align="center">$m = \max(m^{(1)}, m^{(2)})$</p>

下面给出 **“第三步”重缩放公式** 的 **完整、逐步、细节级** 整合版。  
阅读节奏：先场景 → 再推导 → 再数值直觉 → 一句话总结。  
所有公式统一用 `$...$`，可直接贴到 GitHub Markdown。

---

### 4.4 显微镜：重缩放公式（第三步）——完整拆解

#### 【场景】我们手里有什么？
把整条向量  
$x = [a^{(1)}, a^{(2)}]$  
切成两块后，**在 SRAM 里只能先算局部信息**：

<table>
  <thead>
    <tr>
      <th>子块</th>
      <th>已知量</th>
      <th>数学形式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><code>a<sup>(1)</sup></code></td>
      <td>局部最大值&nbsp;<em>m</em><sup>(1)</sup></td>
      <td><em>m</em><sup>(1)</sup>&nbsp;=&nbsp;max<sub>i</sub>&nbsp;<em>a</em><sub>i</sub><sup>(1)</sup></td>
    </tr>
    <tr>
      <td>局部指数和&nbsp;ℓ<sup>(1)</sup></td>
      <td>ℓ<sup>(1)</sup>&nbsp;=&nbsp;∑<sub>i</sub>&nbsp;e<sup>a<sub>i</sub><sup>(1)</sup>&nbsp;-&nbsp;m<sup>(1)</sup></sup></td>
    </tr>
    <tr>
      <td rowspan="2"><code>a<sup>(2)</sup></code></td>
      <td>局部最大值&nbsp;<em>m</em><sup>(2)</sup></td>
      <td><em>m</em><sup>(2)</sup>&nbsp;=&nbsp;max<sub>j</sub>&nbsp;<em>a</em><sub>j</sub><sup>(2)</sup></td>
    </tr>
    <tr>
      <td>局部指数和&nbsp;ℓ<sup>(2)</sup></td>
      <td>ℓ<sup>(2)</sup>&nbsp;=&nbsp;∑<sub>j</sub>&nbsp;e<sup>a<sub>j</sub><sup>(2)</sup>&nbsp;-&nbsp;m<sup>(2)</sup></sup></td>
    </tr>
  </tbody>
</table>

注意： $\ell^{(1)}$ 和 $\ell^{(2)}$ 已经减去了**各自的局部最大值**，因此它们的“零点”并不一致，不能直接把 $\ell^{(1)}+\ell^{(2)}$ 当成全局分母。

---

#### 【推导】如何把局部和搬到同一个“零点”？
1. 先确定**全局零点**  
   $m = \max(m^{(1)}, m^{(2)})$

2. **逐块平移**  
   对子块 $a^{(1)}$：  
   我们希望计算  
   $\sum_i e^{\,a^{(1)}_i - m}$  
   但现有的是  
   $\ell^{(1)} = \sum_i e^{\,a^{(1)}_i - m^{(1)}}$  

   把指数拆开：  
   $a^{(1)}_i - m = \bigl(a^{(1)}_i - m^{(1)}\bigr) + \bigl(m^{(1)} - m\bigr)$  

   于是  
   $\sum_i e^{\,a^{(1)}_i - m}
   = e^{\,m^{(1)} - m} \sum_i e^{\,a^{(1)}_i - m^{(1)}}
   = e^{\,m^{(1)} - m} \cdot \ell^{(1)}$

   同理子块 $a^{(2)}$：  
   $\sum_j e^{\,a^{(2)}_j - m}
   = e^{\,m^{(2)} - m} \cdot \ell^{(2)}$

3. **合并成全局分母**  
   $\ell = e^{\,m^{(1)} - m}\,\ell^{(1)} + e^{\,m^{(2)} - m}\,\ell^{(2)}$

---

#### 【数值直觉】
- 若 $m^{(1)} = m$，则 $e^{0}=1$，该子块无需再缩；  
- 若 $m^{(1)} < m$，则 $e^{m^{(1)}-m}<1$，所有指数项再向下压一次，防止溢出。

---

#### 【总结】
拿到全局最大值后，把每个子块先前按 **局部最大值** 算出的指数和 **整体平移** 到统一尺度，再相加，从而得到 **全局、数值稳定的 Softmax 分母**。


## 5. 复杂度与内存访问对比

| 指标 | 朴素实现 | 分块 Safe Softmax |
|---|---|---|
| 显存峰值 | $O(N)$ 向量 + $O(N^2)$ 矩阵 | $O(B)$ tile |
| HBM 读写 | $O(N^2)$ | $O(N)$ |
| 数值稳定 | ❌ 易溢出 | ✅ 远离溢出 |
| 数学精度 | — | bit-wise 一致 |


## 6. 参考实现（PyTorch 伪码）

```python
import torch

def safe_softmax_block(x, block_size=1024):
    N = x.numel()
    m_global = x.max()
    l_global = 0.0

    for i in range(0, N, block_size):
        xb = x[i : i + block_size]
        mb = xb.max()
        lb = (xb - mb).exp().sum()
        l_global += (mb - m_global).exp() * lb

    return (x - m_global).exp() / l_global
```


## 7. 结语

- Safe Softmax = 减最大值，根治数值溢出。  
- 重缩放公式把局部尺度一次性搬到全局尺度，既防溢出又保证数学等价。  
- FlashAttention 借助这两点，把显存复杂度从 $O(N^2)$ 降到 $O(N)$，同时保持数值稳定与结果精确，为超长上下文模型铺平道路。
