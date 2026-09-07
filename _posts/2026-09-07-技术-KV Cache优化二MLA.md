---
layout:     post
title:      "KV Cache 优化（二）：深入理解 Multi-head Latent Attention"
subtitle:   "DeepSeek 如何用低秩潜在表示压缩每个 token 的 K/V"
date:       2026-09-07 22:00:00 +0800
author:     "Liu Mengxuan"
mathjax:    true
header-img: "img/post-bg-miui6.jpg"
categories: [技术]
tags:       [技术, AI Infra, Transformer, KV Cache, MLA, DeepSeek, Attention, RoPE]
---

MQA 和 GQA 通过共享 K/V heads 减少 KV Cache。DeepSeek-V2 提出的 Multi-head Latent Attention（MLA）换了一个角度：

> 不直接缓存展开后的多头 K/V，而是先把它们压缩成一个低维潜在向量，只缓存这个更短的表示。

当后续 Query 需要读取历史信息时，再通过上投影恢复所需的 K/V；在经过代数变换的高效实现中，部分上投影还可以被吸收到 Query 或输出投影中，避免真的展开整份多头 K/V。

MLA 的难点不只在“压缩再解压”。RoPE 会让某些投影无法被简单吸收，因此需要把与位置有关的部分单独处理。理解这一步，才算真正理解 MLA。

<p align="center">
  <img src="/img/in-post/ai-infra-mla-compress.svg" alt="MHA 直接缓存完整多头 K/V，而 MLA 缓存低维潜在向量" style="max-width: 100%;">
</p>

## 1. MHA/GQA 仍然在保存展开后的 K/V

本文统一使用行向量；为突出低秩关系，省略投影前后的 Norm 和偏置。设当前某一层进入 Attention 投影的 token 表示为：

<p align="center">$h_t\in\mathbb{R}^{H}$</p>

标准 MHA 直接投影出每个 head 的 K/V：

<p align="center">$k_t=h_tW_K,\qquad v_t=h_tW_V$</p>

如果有 `N_h` 个 heads，每个维度为 `D_h`，那么单个 token、单层需要缓存：

<p align="center">$2N_hD_h$</p>

个元素。

GQA 把 `N_h` 降成较小的 `N_KV`，但保存的仍然是已经展开的多个 K/V heads：

<p align="center">$2N_{KV}D_h$</p>

MLA 进一步追问：这些高维 K/V 是否存在可以共享的低维结构？

## 2. 先用“压缩文件”理解 MLA

假设某个 token 展开后的多头 K/V 一共有 4096 个数字。MLA 不把这 4096 个数字逐项保存，而是学习一个下投影矩阵，将隐藏状态压缩成 512 维潜在表示：

<p align="center">$c_t^{KV}=h_tW_{DKV},\qquad c_t^{KV}\in\mathbb{R}^{d_c},\quad d_c\ll N_hD_h$</p>

需要 K/V 时，再用上投影生成：

<p align="center">$k_t^C=c_t^{KV}W_{UK},\qquad v_t^C=c_t^{KV}W_{UV}$</p>

其中：

```text
WDKV：下投影，把隐藏状态压到 latent space
cKV：真正跨 Decode 步骤缓存的潜在向量
WUK、WUV：上投影，把 latent 表示映射到多头 K/V 空间
```

可以类比为：

```text
标准 MHA：仓库里保存每个部门展开后的整套资料
MLA：仓库只保存一份压缩档案，各部门读取时按自己的方式解释
```

这里的“解压”不是无损压缩算法。它是训练过程中学习出的低秩表示，模型会从一开始就适应这条信息通路。

## 3. 为什么低秩能够节省 KV Cache？

如果暂时忽略位置编码，标准 MHA 单 token、单层缓存元素数约为：

<p align="center">$M_{MHA}=2N_hD_h$</p>

MLA 主要缓存一个长度为 `d_c` 的联合 KV latent：

<p align="center">$M_{MLA}\approx d_c$</p>

只要：

<p align="center">$d_c\ll2N_hD_h$</p>

缓存就会明显缩小。

例如，假设 `N_h=32`、`D_h=128`：

```text
MHA：2 × 32 × 128 = 8192 个元素/token/layer
MLA latent：dc = 512 个元素/token/layer
```

暂不考虑 RoPE 附加状态时，元素数是原来的 `1/16`。

这里要区分两个概念：

- `D_h` 是单个 attention head 的维度；
- `d_c` 是所有 heads 共享的压缩潜在维度。

MLA 不是把每个 head 分别压成 `d_c`，否则缓存仍会乘上 head 数。关键正是多个 heads 共享同一个潜在表示。

## 4. Query 也可以低秩投影，但不需要缓存

DeepSeek-V2 的 MLA 也对 Query 使用低秩投影：

<p align="center">$c_t^Q=h_tW_{DQ},\qquad q_t^C=c_t^QW_{UQ}$</p>

这可以减少 Q 投影的参数和计算结构，但 `c_t^Q` 不需要进入 KV Cache，因为历史 Query 以后不会再被读取。

所以要把两件事分开：

```text
Query 低秩：主要影响参数化和当前步计算
KV 联合低秩：直接减少跨步骤保存的 KV Cache
```

## 5. 为什么不能简单地每次都“完整解压”？

最直观的实现是：从缓存读取所有历史 `c^{KV}`，上投影成完整 K/V，再做 Attention。

```text
历史 latent cache
    ↓ 每一步解压
完整多头 K/V
    ↓
Attention
```

这样虽然省了持久缓存容量，但每个 Decode 步骤都要重新展开全部历史 K/V，可能引入大量计算和临时访存。

MLA 的关键工程价值来自矩阵结合律：某些上投影可以被吸收到其他投影中。

先看不带 RoPE 的注意力分数。若：

<p align="center">$k_t^C=c_t^{KV}W_{UK}$</p>

那么：

<p align="center">$q_i^C(k_t^C)^T=q_i^C(c_t^{KV}W_{UK})^T=(q_i^CW_{UK}^T)(c_t^{KV})^T$</p>

可以先把当前 Query 变换为：

<p align="center">$\tilde q_i=q_i^CW_{UK}^T$</p>

再直接与缓存的 `c_t^{KV}` 点积，而不必为所有历史 token 显式恢复完整 Key。

这里的 Query 和上投影应按 head 配对理解：对于 head `i`，使用该 head 的 `q_i` 和 `W_UK,i`，不是让单头 Query 乘上不匹配的全头矩阵。

Value 路径也可以利用线性运算的结合关系。设 head `i` 对位置 `j` 的权重为 `α_ij`：

<p align="center">$\sum_j\alpha_{ij}(c_j^{KV}W_{UV,i})=\left(\sum_j\alpha_{ij}c_j^{KV}\right)W_{UV,i}$</p>

即先在 latent 空间加权，再上投影一次。例如 `c1=[1,2]`、`c2=[3,0]`，权重为 `0.25、0.75`，先得到 `0.25c1+0.75c2=[2.5,0.5]`，再乘 `W_UV,i`，与先展开两个 Value 再汇总相同。每个 head 的权重不同，latent 汇总结果也不同；随后还可将线性映射与输出投影组合。具体 kernel 是否物化中间量取决于实现，但核心思想是：

> 缓存低维 latent，同时尽量让 Attention 直接在压缩表示附近完成，而不是每一步制造庞大的临时 K/V。

## 6. RoPE 为什么让事情变复杂？

RoPE 对 Q/K 的不同二维分量施加与位置有关的旋转。把位置 `t` 的旋转记为 `R_t`：

<p align="center">$q_t^R=q_tR_t^T,\qquad k_t^R=k_tR_t^T$</p>

如果完整 Key 是由潜在向量上投影得到：

<p align="center">$k_t=c_t^{KV}W_{UK}$</p>

对当前位置 `i`、历史位置 `j`，分数包含 `q_i R_i^T R_j W_UK^T c_j^T`。由于每个历史位置的 `R_j` 不同，通常不能把 `W_UK` 像前面那样统一、简单地吸收到当前 Query 投影中。

直觉上看：

```text
内容投影：所有 token 使用同一个 WUK，容易合并
位置旋转：token 1、token 2、token 3 使用不同 Rt，难以统一合并
```

## 7. 解耦 RoPE：内容部分压缩，位置部分单独保存

MLA 将 Q/K 拆成两部分：

```text
内容部分（NoPE）：参与低秩压缩，可做投影吸收
位置部分（RoPE）：维度较小，单独应用旋转并缓存
```

对某个 head，可以写成：

<p align="center">$q_{t,i}=[q_{t,i}^C;q_{t,i}^R],\qquad k_{t,i}=[k_{t,i}^C;k_t^R]$</p>

对应的点积自然分成两项：

<p align="center">$q_{t,i}k_{j,i}^T=q_{t,i}^C(k_{j,i}^C)^T+q_{t,i}^R(k_j^R)^T$</p>

于是每个历史 token 主要保存：

```text
cjKV：压缩后的内容 latent
kjR：较小的 RoPE Key
```

而不是完整的多头 K/V。

<p align="center">
  <img src="/img/in-post/ai-infra-mla-rope.svg" alt="MLA 将内容通路和 RoPE 位置通路解耦" style="max-width: 100%;">
</p>

在 DeepSeek-V2 的典型配置中，KV latent 维度为 512，解耦后的 RoPE Key 维度为 64，因此每层每个 token 主要缓存 `512+64=576` 个元素。这个数字来自该模型的具体配置，不是 MLA 必须采用的固定维度。

## 8. 用“我是你的人”走一遍 Decode

假设 Prompt 是“我是你”，Prefill 在某一层完成后，缓存中不是三组展开的多头 K/V，而是：

```text
token “我”： [c我KV, k我R]
token “是”： [c是KV, k是R]
token “你”： [c你KV, k你R]
```

模型生成“的”后，在该层执行：

```text
h的
 ├─→ 下投影 → c的KV ─→ 追加到 latent cache
 └─→ RoPE 路径 → k的R ─→ 追加到 position cache

当前 q的：
 ├─→ 内容 Query，与历史 cKV 匹配
 └─→ 位置 Query，与历史 kR 匹配
```

两部分分数相加后，按完整 Query/Key 维度进行缩放，再做 Softmax，最后汇总 Value 信息，得到“的”在当前层的 Attention 输出。当前“的”的 latent 和位置 Key 同样参与本步 Attention；例子假定每个汉字是一个 token，实际切分由 tokenizer 决定。

下一轮处理“人”时，缓存增长为：

```text
[c我KV, c是KV, c你KV, c的KV]
[k我R,  k是R,  k你R,  k的R ]
```

增长规律仍是“每个 token、每一层追加一份状态”，只是追加的数据比完整多头 K/V 更短。

## 9. MLA、GQA 和 MQA 到底差在哪里？

| 方法 | 核心动作 | 每个 token 保存什么 | 是否改变 Attention 结构 |
| --- | --- | --- | --- |
| MQA | 所有 Q heads 共享一组 K/V | 1 个 K head + 1 个 V head | 是 |
| GQA | 同组 Q heads 共享一组 K/V | `N_KV` 组展开 K/V | 是 |
| MLA | 多头共享低维潜在表示 | KV latent + RoPE Key | 是 |

可以把它们理解为：

```text
MQA/GQA：减少资料库的份数
MLA：改变资料库的保存格式，只存低维底稿
```

它们都需要模型结构与权重配合，不是对任意 MHA 模型打开一个推理开关就能无损使用。

## 10. MLA 没有消除哪些成本？

MLA 显著降低的是 KV Cache 的容量和读取压力，但它没有让 Attention 与上下文长度无关：

- 每个新 Query 仍需与历史位置建立匹配；
- 上下文越长，需要扫描的历史 latent 和 RoPE Key 越多；
- 低秩投影、投影吸收和专用 kernel 带来了实现复杂度；
- 缓存压缩能力来自训练后的模型结构，不是完全无代价的数据压缩。

因此，MLA 解决的是“每个历史 token 太宽”，而不是“历史 token 太多”。后一个问题要交给 Sliding Window、StreamingLLM 或 KV Cache pruning 一类方法。

## 11. 总结

MLA 的主线可以压缩成：

```text
隐藏状态 h
    ↓ 下投影
共享 KV latent cKV       小型 RoPE Key kR
    ↓ 缓存                    ↓ 缓存
内容匹配与 Value 汇总     位置匹配
             ↓
        Attention 输出
```

最重要的三点是：

1. MLA 不缓存展开后的完整多头 K/V，而是缓存共享的低维 KV latent；
2. 线性投影可以通过矩阵结合律被吸收，避免简单解压方案的巨大中间开销；
3. RoPE 依赖 token 位置，因此 MLA 将较小的位置 Key 与可压缩的内容通路解耦。

## 系列导航

- [第一篇：MQA 与 GQA]({% post_url 2026-09-07-技术-KV Cache优化一MQA与GQA %})
- 本篇：MLA
- [第三篇：局部注意力与缓存淘汰]({% post_url 2026-09-07-技术-KV Cache优化三局部注意力与缓存淘汰 %})
- [第四篇：投机解码]({% post_url 2026-09-08-技术-KV Cache优化四投机解码 %})

## 参考资料

- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434), 2024.
