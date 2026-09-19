---
layout:     post
title:      "Transformer 前馈网络：从 FFN 到 SwiGLU"
subtitle:   "从逐 token 加工、激活函数到 4 倍与 8/3 倍中间维度"
date:       2026-09-19 12:00:00 +0800
author:     "Liu Mengxuan"
mathjax:    true
header-img: "img/post-bg-miui6.jpg"
categories: [技术]
tags:       [技术, AI Infra, Transformer, FFN, SwiGLU, 激活函数]
---

前面梳理 Transformer 时，Attention、KV Cache 和 FlashAttention 占据了很多篇幅。但在一个 Transformer Block 中，Attention 后面的 FFN 同样值得拆开来看：它不直接混合不同 token，却往往占据 Block 中很大一部分参数和矩阵乘计算。

这篇笔记对应 AIInfraGuide「3.4 Transformer 前馈网络 FFN 深入理解」的前六节，围绕三个问题展开：**FFN 在加工什么？激活函数和门控改变了什么？为什么中间维度常见 4 倍和 8/3 倍？**

## 1. FFN 在 Transformer 中负责什么？

### 1.1 Attention 汇聚上下文，FFN 加工每个位置的特征

用现代 Decoder-only 模型常见的 Pre-Norm 结构表示一个 Block：

<p align="center">$H=X+\operatorname{Attention}(\operatorname{Norm}(X))$</p>

<p align="center">$Y=H+\operatorname{FFN}(\operatorname{Norm}(H))$</p>

这里省略了 dropout 等细节。两条残差路径要求子模块的输入输出形状保持一致。

如果输入是 `[B, S, D]`，其中 `B` 是 batch size，`S` 是序列长度，`D` 是隐藏维度，那么：

- Attention 沿序列维汇聚允许访问的位置；在因果 Attention 中，当前位置只能使用自己和历史位置。
- FFN 沿隐藏维加工每个位置的向量，输入和输出仍然是 `[B, S, D]`。

**“每个 token 独立”描述的是 FFN 这一步的计算方式，不代表它的输入没有上下文。** Attention 已经把上下文信息汇入当前向量，FFN 接着处理这个包含上下文的表示。

<p align="center">
  <img src="/img/in-post/ai-infra-ffn-role.svg" alt="Attention 在允许的位置间汇聚信息，FFN 对每个 token 使用相同权重独立加工" style="max-width: 100%;">
</p>

### 1.2 独立计算，不等于每个 token 有独立参数

同一层 FFN 的权重在所有位置和 batch 样本之间共享。设输入的第 `b` 个样本、第 `s` 个位置为一个行向量：

<p align="center">$Y_{b,s,:}=f(X_{b,s,:};\theta_{\mathrm{FFN}})$</p>

每个位置输入不同，但使用的是同一个函数和同一套参数。不同 Transformer 层通常拥有各自的 FFN 参数。

实现时，可以把前两维合并成 `T=B×S`，让 `[T, D]` 一次参与矩阵乘。GPU 同时处理很多 token，并不会因此让 FFN 在 token 之间混合信息：矩阵乘改变的是每行的特征维。

### 1.3 两个容易过度简化的说法

第一，Attention 的 `PV` 在固定 `P` 时是对 Value 的线性加权，但完整 Attention 中的 `P=softmax(QKᵀ/√d)` 依赖输入。因此，**不能把整个 Attention 说成线性运算**。FFN 的价值是提供额外的、逐位置的可学习非线性特征变换。

第二，研究确实发现 FFN 与事实性知识存储有关，可以借助“特征匹配后组合输出”的直觉理解它。但知识和推理能力来自模型各组件的共同作用，不能把 FFN 当作唯一的知识库，也不能把 Attention 的职责限定为查表。

## 2. FFN 的“展开—压缩”结构

### 2.1 先沿隐藏维展开，再回到残差流的宽度

设中间维度为 `M`，标准 FFN 写成：

<p align="center">$\operatorname{FFN}(X)=\phi(XW_{up}+b_{up})W_{down}+b_{down}$</p>

其中 `φ` 是激活函数。本文采用行向量右乘权重的记法：

| 对象 | 形状 | 作用 |
| --- | --- | --- |
| 输入 `X` | `[B, S, D]` | 每个 token 有 `D` 个特征 |
| `W_up` | `[D, M]` | 生成 `M` 个中间特征 |
| 升维结果与激活结果 | `[B, S, M]` | 对中间特征施加非线性 |
| `W_down` | `[M, D]` | 将中间特征重新组合为 `D` 维 |
| 输出 | `[B, S, D]` | 与残差分支形状一致 |

以 `D=4096、M=16384` 为例：

```text
[B, S, 4096]
    → up projection
[B, S, 16384]
    → ReLU / GELU 等逐元素激活
[B, S, 16384]
    → down projection
[B, S, 4096]
```

这里没有增加 token 数量，也没有把一个 token 拆成四个 token，只是扩展了每个 token 的特征通道。

### 2.2 为什么需要中间的非线性？

暂时忽略偏置。如果没有激活函数：

<p align="center">$(XW_{up})W_{down}=X(W_{up}W_{down})$</p>

两次线性变换可以合并成一次。即使带偏置，合并后仍然只是一个仿射变换。

加入 `φ` 后，不同输入会产生不同的中间响应，无法普遍把它们折叠成一个固定矩阵。把中间维度做宽，相当于提供更多可以学习和组合的非线性特征。

但**线性升维本身不会凭空创造信息，也不保证原本不可线性分割的数据变得可分**；关键是学习到的投影与非线性的组合。降维也不是自动挑出“最重要的特征”，而是学习如何把中间响应组合回模型需要的表示。

### 2.3 一个足够小的实现

```python
import torch
from torch import nn
from torch.nn import functional as F


class StandardFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))
```

注意 PyTorch 的 `nn.Linear(D, M).weight` 实际存储形状是 `[M, D]`，前向对应 `x @ weight.T`。它与本文公式中 `[D, M]` 的数学权重记法相差一次转置，参数量不变。

## 3. 激活函数：ReLU、GELU 与 Swish/SiLU

先把三条曲线放在一起。下图使用精确 GELU，Swish 固定 `β=1`，即 SiLU；右侧放大负半轴和零点附近。

<p align="center">
  <img src="/img/in-post/ai-infra-ffn-activation-curves.svg" alt="ReLU、精确 GELU 和 SiLU 函数曲线，以及负半轴局部放大图" style="max-width: 100%;">
</p>

### 3.1 ReLU：正值通过，负值归零

<p align="center">$\operatorname{ReLU}(x)=\max(0,x)$</p>

ReLU 的规则很直接：正半轴输出等于输入，负半轴输出为零。它的计算简单，输出可能包含大量零值。

在 `x>0` 时导数为 1，在 `x<0` 时导数为 0；`x=0` 处不可微，实现通常选择一个约定的反向值，例如 PyTorch 取 0。

负区间的零梯度也带来了 dying ReLU 风险：如果一个通道对大量甚至所有训练样本都持续落在负区间，它可能长期收不到来自该激活路径的有效梯度。**某一次输入为负，不等于这个通道永久死亡**，后续输入和上游参数仍可能改变。

另外，输出含零不代表普通 dense GEMM 会自动跳过计算；能否利用稀疏性还取决于布局和 kernel。

### 3.2 GELU：用正态分布 CDF 平滑地调节输入

<p align="center">$\operatorname{GELU}(x)=x\Phi(x)=\frac{x}{2}\left[1+\operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$</p>

`Φ(x)` 是标准正态分布的累积分布函数，取值在 0 和 1 之间。可以把它理解成一个随输入变化的软系数：大正值基本原样通过，大负值被压到接近零的位置。

“以概率 `Φ(x)` 保留输入的期望”是一种数学解释，**实际 GELU 前向是确定性计算，不会随机丢弃元素**。

它与 ReLU 的明显区别出现在零点附近：负数不会全部被截断，而是形成一个小的负值凹陷。精确 GELU 的最小值约为 `−0.170`，出现在 `x≈−0.752`。

常见的 tanh 近似为：

<p align="center">$\operatorname{GELU}(x)\approx\frac{x}{2}\left[1+\tanh\left(\sqrt{\frac{2}{\pi}}(x+0.044715x^3)\right)\right]$</p>

使用 PyTorch 时，`F.gelu(x)` 默认采用精确形式，`F.gelu(x, approximate="tanh")` 使用该近似。做实现对齐时，需要确认两边选择一致。

### 3.3 Swish 与 SiLU：用 sigmoid 调节自身

<p align="center">$\operatorname{Swish}_{\beta}(x)=x\operatorname{sigmoid}(\beta x)$</p>

其中 `β` 可以固定或学习。当 `β=1` 时：

<p align="center">$\operatorname{SiLU}(x)=x\operatorname{sigmoid}(x)=\frac{x}{1+e^{-x}}$</p>

SiLU 与 GELU 都是平滑函数，都在负半轴保留一个小凹陷；SiLU 的最小值约为 `−0.278`，出现在 `x≈−1.278`，负值尾部也更明显。

对一般 Swish，`β=0` 时结果是 `x/2`；当 `β` 趋向正无穷时，函数逐点趋向 ReLU。下图展示这个变化过程。

<p align="center">
  <img src="/img/in-post/ai-infra-ffn-swish-beta.svg" alt="Swish 在 beta 等于 0、0.5、1、5 时的曲线，与 ReLU 的对比" style="max-width: 100%;">
</p>

### 3.4 从梯度曲线看差异

精确 GELU 和 SiLU 的导数分别是：

<p align="center">$\operatorname{GELU}'(x)=\Phi(x)+x\frac{e^{-x^2/2}}{\sqrt{2\pi}}$</p>

<p align="center">$\operatorname{SiLU}'(x)=s+x\,s(1-s),\quad s=\operatorname{sigmoid}(x)$</p>

<p align="center">
  <img src="/img/in-post/ai-infra-ffn-activation-gradients.svg" alt="ReLU、GELU 和 SiLU 的导数曲线，展示 ReLU 的零点跳变与平滑函数的负梯度区间" style="max-width: 100%;">
</p>

GELU 与 SiLU 在零点的导数都是 `0.5`，没有 ReLU 的跳变。但平滑不代表梯度总是正数：两者在部分负区间的导数为负，在各自最低点处导数为零，大负值处梯度也趋近零。因此，不应说它们“处处梯度非零”或“保证不会出现梯度问题”。

| 特性 | ReLU | GELU（精确形式） | SiLU（Swish，β=1） |
| --- | --- | --- | --- |
| 核心规则 | `max(0,x)` | `x·Φ(x)` | `x·sigmoid(x)` |
| 输出范围 | `[0,+∞)` | 约 `[−0.170,+∞)` | 约 `[−0.278,+∞)` |
| 单调性 | 单调不减 | 非单调 | 非单调 |
| 零点可微 | 否 | 是 | 是 |
| 负半轴 | 全部置零 | 小幅负值后趋近零 | 小幅负值后趋近零 |
| 常见应用 | 原始 Transformer FFN | BERT、GPT-2 | LLaMA 等模型的 SwiGLU 门控分支 |

激活函数的实际效果需要结合模型和训练方案比较。ReLU 运算最简单；GELU、SiLU 涉及更复杂的逐元素运算，但端到端开销还取决于 GEMM 大小、访存和融合方式，不能仅从函数公式判断速度。

## 4. SwiGLU：多一条可学习的门控分支

### 4.1 从单分支激活到双分支相乘

标准 FFN 的中间状态是 `φ(XW_up)`。GLU 引入两套投影，一套产生内容，一套调节内容：

<p align="center">$U=XW_{up},\quad G=XW_{gate}$</p>

<p align="center">$H_{\mathrm{GLU}}=U\odot\operatorname{sigmoid}(G)$</p>

`⊙` 表示逐元素相乘。两条分支都依赖当前输入，且都使用训练得到的权重；sigmoid 门控是确定性缩放系数，不需要随机采样。

SwiGLU 将这里的 sigmoid 换成 Swish。在本文讨论的常见 `β=1` 实现中，完整 FFN 是：

<p align="center">$\operatorname{FFN}_{\mathrm{SwiGLU}}(X)=\left[\operatorname{SiLU}(XW_{gate})\odot(XW_{up})\right]W_{down}$</p>

<p align="center">
  <img src="/img/in-post/ai-infra-ffn-swiglu.svg" alt="标准 FFN 的两矩阵结构，与 SwiGLU 的门控投影、内容投影及下投影三矩阵结构对比" style="max-width: 100%;">
</p>

### 4.2 SwiGLU 的门控值不是概率

虽然 sigmoid 的输出在 0 到 1 之间，但 SiLU 的输出是 `g·sigmoid(g)`，可以为负，也可以大于 1。

例如令两条分支在某个 token 上的投影结果为：

```text
G = [-1,    0,    2]
U = [ 2,    3,    4]

SiLU(G) ≈ [-0.269, 0, 1.762]
H       ≈ [-0.538, 0, 7.046]
```

因此，SwiGLU 的“门”既可以抑制，也可以放大或改变符号，不能完全理解成一个只决定通过多少的 0～1 开关。

标准 FFN 本来就能通过不同的投影列学习不同特征。SwiGLU 增加的是**两组可学习特征之间的乘性交互**：一组内容的贡献可以由另一组输入相关的响应来调节。它提供了不同的表达方式，但不会保证每种任务都优于标准 FFN。

### 4.3 对应的 PyTorch 实现

```python
class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate(x))
        content = self.up(x)
        return self.down(gate * content)


# 用小张量验证形状，避免入门示例直接分配上亿参数。
x = torch.randn(2, 5, 12)
standard = StandardFFN(d_model=12, d_ff=48)
swiglu = SwiGLUFFN(d_model=12, d_ff=32)

print(standard(x).shape)  # torch.Size([2, 5, 12])
print(swiglu(x).shape)    # torch.Size([2, 5, 12])
print(sum(p.numel() for p in standard.parameters()))  # 1152
print(sum(p.numel() for p in swiglu.parameters()))    # 1152
```

这里内容分支没有额外激活，激活位于门控分支；相乘之后才执行下投影。两种结构的参数量为什么能相同，可以直接从矩阵形状推出来。

## 5. 参数量：两个矩阵与三个矩阵

### 5.1 先写通式

忽略偏置，标准 FFN 有两个矩阵：

<p align="center">$P_{\mathrm{standard}}=DM+MD=2DM$</p>

SwiGLU 有三个矩阵：

<p align="center">$P_{\mathrm{SwiGLU}}=DM+DM+MD=3DM$</p>

如果采用带偏置的版本，标准 FFN 还要加上 `M+D` 个参数；三个 Linear 都带偏置的 SwiGLU 还要加上 `2M+D`。本文示例都使用无偏置版本。

### 5.2 用 D=4096 算一次

| 结构 | 中间维度 M | 每个矩阵的参数量 | 矩阵数 | FFN 总参数量 |
| --- | --- | --- | --- | --- |
| 标准 FFN | 16384 | 67,108,864 | 2 | 134,217,728 |
| SwiGLU | 11008 | 45,088,768 | 3 | 135,266,304 |

两者约为 134M 和 135M 参数。这里 `M` 在数字后表示百万，在维度公式里则表示中间宽度，需要结合上下文区分。

只保存 BF16/FP16 权重、每参数 2 字节时，这两组 FFN 权重分别占 **256 MiB** 和 **258 MiB**。这不是训练总显存，还没有包含梯度、优化器状态和激活。

### 5.3 “FFN 占 Block 的 2/3”有什么前提？

对于 Q、K、V 和输出投影都是 `[D,D]` 的标准 MHA：

<p align="center">$P_{\mathrm{Attention}}\approx4D^2$</p>

采用 `M=4D` 的标准 FFN：

<p align="center">$P_{\mathrm{FFN}}=8D^2$</p>

忽略 Norm 和偏置后，FFN 在这个 Block 中的参数占比约为：

<p align="center">$\frac{8D^2}{4D^2+8D^2}=\frac{2}{3}$</p>

这个结论有条件：GQA/MQA 的 K/V 投影更小，FFN 比例会改变；中间维度和 MoE 结构也会改变结果。它描述的是 Block 参数占比，不是包含 embedding、输出层后的整个模型占比，更不是运行时间占比。长序列 Attention 仍然可能承担大量计算。

### 5.4 三个矩阵是否意味着多 50% 计算？

只有在**中间宽度相同**时，这个比较才成立。对 `T=B×S` 个 token，按一次乘加计 2 FLOPs，主要前向矩阵乘计算量是：

<p align="center">$F_{\mathrm{standard}}\approx4TDM$</p>

<p align="center">$F_{\mathrm{SwiGLU}}\approx6TDM$</p>

如果缩小 SwiGLU 的中间维度以匹配参数预算，两者的主要 GEMM FLOPs 也可以接近相同。SwiGLU 额外包含门控激活与逐元素乘法，实际耗时还受 kernel 和内存访问影响。

此外，gate 与 up 两个投影可以沿输出维拼接为一次更宽的 GEMM，结果再拆成两份。因此，“有三个权重矩阵”不代表工程实现必须执行三次独立的 GEMM 调用。

## 6. 中间维度为什么常见 4 倍和 8/3 倍？

### 6.1 4 倍是常见设计，不是数学定理

原始 Transformer 使用 `D=512、M=2048`，也就是 `M=4D`。后续许多标准 FFN 延续了这个比例。

更宽的中间层提供更多特征通道，同时增加参数量、计算量和中间激活开销。4 倍是一种常见的工程折中，最佳比例取决于模型、训练预算和硬件，不存在所有模型都必须遵守的最优值。

### 6.2 8/3 倍来自配平参数预算

标准 FFN 采用 4 倍中间宽度时：

<p align="center">$P_{\mathrm{standard}}=2D(4D)=8D^2$</p>

如果 SwiGLU 仍然使用 `4D`：

<p align="center">$P_{\mathrm{SwiGLU}}=3D(4D)=12D^2$</p>

参数量会增加 50%。为了保持同样的预算，令：

<p align="center">$3DM=8D^2\quad\Longrightarrow\quad M=\frac{8}{3}D$</p>

所以 **8/3 不是 SiLU 的特殊数学性质，而是把两个矩阵换成三个矩阵后，为匹配参数规模计算出的宽度**。

<p align="center">
  <img src="/img/in-post/ai-infra-ffn-budget.svg" alt="标准 FFN 的两份 4D 平方参数与 SwiGLU 的三份 8/3 D 平方参数配平示意" style="max-width: 100%;">
</p>

### 6.3 为什么 LLaMA-2-7B 使用 11008？

对于 `D=4096`：

<p align="center">$\frac{8}{3}\times4096=10922.\overline{6}$</p>

把这个理想宽度向上对齐到 256 的倍数：

<p align="center">$M=256\left\lceil\frac{(8/3)\times4096}{256}\right\rceil=256\times43=11008$</p>

对应的参数量比 `8D²` 多约 **0.78%**，因此表格中的两种结构只是近似等参数，而非完全相等。

对齐有利于匹配矩阵乘 tile、并行切分和硬件执行方式，但 256 不是所有 Tensor Core 运算的统一硬性要求，也不是所有 SwiGLU 模型都采用的宽度规则。实际模型可能使用不同的中间维度或额外倍率，应以配置为准。

## 7. 把这一段知识串起来

```text
Attention：把允许访问的上下文汇入当前 token
    ↓
FFN：每个 token 独立计算，同一层共享权重
    ↓
标准 FFN：D → M → 非线性 → D
    ↓
SwiGLU：两条 D → M 投影，SiLU 门控 × 内容，再 M → D
    ↓
矩阵数从 2 变成 3
    ↓
为了匹配参数预算，中间宽度从 4D 调整到约 8D/3
```

再遇到 FFN 配置时，可以依次检查：输入输出隐藏维度是多少，中间维度是多少，激活位于哪条路径，有几个投影矩阵。把这些形状写清楚，参数量、主要矩阵乘计算量和中间激活规模就都有了计算起点。

## 参考资料

- [AIInfraGuide：3.4 Transformer 前馈网络 FFN 深入理解](https://caomaolufei.github.io/AIInfraGuide/guides/模块一-前置知识/transformer/34-transformer前馈网络ffn深入理解/)：本文整理范围为第 1～6 节。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)
- [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913)
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
