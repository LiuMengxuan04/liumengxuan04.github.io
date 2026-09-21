---
layout:     post
title:      "FFN 中的 CUDA Kernel 融合"
subtitle:   "减少中间张量的显存往返，让逐元素操作留在片上"
date:       2026-09-21 12:00:00 +0800
author:     "Liu Mengxuan"
mathjax:    true
header-img: "img/post-bg-miui6.jpg"
categories: [技术]
tags:       [技术, AI Infra, CUDA, GPU, FFN, Kernel Fusion]
---

SwiGLU 的数学表达式很短：

<p align="center">$Y=\left[\operatorname{SiLU}(XW_{gate})\odot(XW_{up})\right]W_{down}$</p>

但如果每个步骤都启动一个独立 CUDA kernel，中间结果就需要反复写回和读出 GPU 显存。Kernel fusion（内核融合）要做的是：**把适合连续执行的操作放进同一个 kernel，让中间值留在寄存器或共享内存中。**

## 1. 先看未融合的计算

可以把 SwiGLU 拆成五步：

```python
G = X @ W_gate       # 矩阵乘法
U = X @ W_up         # 矩阵乘法
A = silu(G)          # 逐元素激活
H = A * U            # 逐元素乘法
Y = H @ W_down       # 矩阵乘法
```

一个常见的朴素执行方式是五个 kernel：

```text
Kernel 1：GEMM，计算 G，写回显存
Kernel 2：GEMM，计算 U，写回显存
Kernel 3：SiLU，读 G，写 A
Kernel 4：逐元素乘，读 A 和 U，写 H
Kernel 5：GEMM，计算 Y，写回显存
```

`A=SiLU(G)` 只服务于下一步乘法，却被完整写回显存，又立即读回来。这类中间张量就是融合的重点。

## 2. 融合 SiLU 与门控乘法

不融合时：

```text
读 G → 计算 A=SiLU(G) → 写 A
读 A、U → 计算 H=A*U → 写 H
```

融合后，一个线程可以对自己负责的元素连续执行：

```python
g = G[i]
u = U[i]
a = g / (1 + exp(-g))
H[i] = a * u
```

`a` 暂存在寄存器里，不再形成完整的 `A` 张量。若 `G、U、A、H` 都有 `E` 个元素，按逻辑上的全局内存访问计算：

| 方式 | 读取 | 写入 | 合计 |
| --- | --- | --- | --- |
| 分开 | `G、A、U` | `A、H` | `5E` |
| 融合 | `G、U` | `H` | `3E` |

因此省掉 `A` 的一次写入和一次读取，也省掉一次 kernel launch。

<p align="center">
  <img src="/img/in-post/ai-infra-ffn-fusion.svg" alt="未融合与融合的 SiLU 加门控乘法数据流对比，中间激活留在寄存器中" style="max-width: 100%;">
</p>

## 3. GEMM 的 Epilogue Fusion

门控投影可以写成：

```python
G = X @ W_gate
A = SiLU(G)
```

分开执行时，矩阵乘法先把 `G` 写回显存，另一个 kernel 再读 `G` 并计算 SiLU。Epilogue fusion 把激活放进 GEMM 的收尾阶段：

```text
矩阵乘法完成累加
        ↓
在寄存器中计算 SiLU
        ↓
只把激活后的 A 写回显存
```

这里必须等一个输出元素的矩阵乘累加完成后再激活，因为正确的数学顺序是：

<p align="center">$\operatorname{SiLU}\left(\sum_k X_kW_k\right)$</p>

而不是对每个乘积先激活再求和。cuBLAS、CUTLASS 等库提供了不同形式的 epilogue 扩展，方便在 GEMM 输出阶段加入逐元素操作。

## 4. 合并 Gate 和 Up 两次 GEMM

Gate 和 Up 使用同一个输入 `X`，可以把权重沿输出维拼接：

<p align="center">$W_{packed}=[W_{gate}\;W_{up}]$</p>

于是：

<p align="center">$XW_{packed}=[XW_{gate}\;XW_{up}]=[G\;U]$</p>

代码表示为：

```python
# 两次 GEMM
G = X @ W_gate
U = X @ W_up

# 一次更宽的 GEMM
W_packed = concat([W_gate, W_up], dim=1)
GU = X @ W_packed       # [N, 2*d_ff]
G, U = split(GU, d_ff, dim=1)
```

这不会减少乘加总量，但可以减少一次 kernel launch，并让一次更大的矩阵乘更充分地利用 GPU。实际实现通常在权重布局阶段就准备好 packed 权重，不会在每次前向时重复拼接。

## 5. 融合残差与归一化

FFN 输出经常还要和残差相加，再进行 LayerNorm 或 RMSNorm：

```python
Z = Y + residual
out = norm(Z)
```

如果拆成多个 kernel，`Z` 也会被写回后再读出。融合后，可以把 `Y`、`residual` 和归一化参数一次读入，在片上完成：

```text
读 Y、residual
    ↓
片上计算 Z = Y + residual
    ↓
片上统计均值/平方和
    ↓
完成归一化并写出 out
```

归一化需要同一 token 的多个特征共同参与统计，因此比单纯的逐元素融合需要更多线程协作。但它仍然遵循相同原则：中间结果产生后立即消费。

## 6. 为什么 Decode 阶段通常更容易受益？

Decode 一次通常只处理一个或少量新 token，矩阵乘规模较小。此时 kernel launch 延迟和显存带宽在总耗时中的比例更高，省掉中间张量的写读往返往往更有价值。

Prefill 处理整段 prompt，GEMM 规模更大，矩阵乘本身占据更多时间，逐元素操作和 kernel launch 的占比相对下降。因此融合仍有收益，但通常要通过 benchmark 判断，不能只看算子数量估计加速比。

## 7. 融合不是越多越好

寄存器和共享内存容量有限。如果把过多操作塞进同一个 kernel，可能增加寄存器压力、降低 occupancy，甚至让代码失去灵活性。矩阵乘、逐元素运算和归一化的并行模式也不同，强行全部融合未必更快。

判断一次融合是否值得，主要看三件事：

1. 它是否消除了大块中间张量的显存读写？
2. 它是否减少了 kernel launch 或提高了矩阵乘利用率？
3. 它是否带来过大的寄存器、共享内存和线程协作开销？

Kernel fusion 的目标不是让数学公式变少，而是让数据在 GPU 片上多停留一会儿，少经过几次 HBM 往返。

## 参考资料

- [CUTLASS Collective Epilogue](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)
- [cuBLASLt Matrix Multiplication](https://docs.nvidia.com/cuda/cublas/#cublasltmatmul)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
