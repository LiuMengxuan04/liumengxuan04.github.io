---
layout:     post
title:      "KV Cache 补充：从最后一行到下一个 Token"
subtitle:   "为什么只取最后位置的隐藏向量，以及 LM Head 怎样预测词表中的下一个 token"
date:       2026-09-24 18:00:00 +0800
author:     "Liu Mengxuan"
mathjax:    true
header-img: "img/post-bg-miui6.jpg"
categories: [技术]
tags:       [技术, AI Infra, Transformer, KV Cache, 推理, Attention, LM Head]
---

前面的 [KV Cache 基础篇]({% post_url 2026-09-07-技术-KV Cache与Transformer推理 %}) 介绍了 Prefill、Decode 和历史 K/V 的复用。这篇作为系列补充，继续沿着“中华人 → 民”的例子，把 Attention 输出、最后位置的隐藏向量和词表预测接起来。

示意图经常把完整 Attention 方阵保留下来，又把 Attention 输出直接连到新 token，容易让人误以为旧分数也被缓存，或者一个 Attention 向量就能直接变成文字。实际推理中，这里包含几个不同的计算阶段。

从当前 Query 读取历史 K/V 开始，Attention 先得到一个隐藏向量；它继续经过后续网络计算，最后由 LM Head 给整个词表打分，解码策略再选出新 token。下面沿着这条数据流展开。

## 1. KV Cache 阶段：只重新计算当前 Query 的一行

假设 Prompt 是“中华”，Prefill 已经预测出“人”。现在将“人”作为新输入做一次 Decode，处理后的上下文是下面三个 token。为便于说明，假设一个汉字就是一个 token；实际分词边界由 tokenizer 决定。

```text
位置：  1    2    3
内容：  中   华    人
```

现在模型要根据“中华人”预测下一个 token。以某一层、某一个 Attention head 为例，“中”“华”的 K/V 从缓存读取，“人”的 Q/K/V 本轮计算。把当前 K/V 追加后，参与 Attention 的完整 K/V 为：

<p align="center">$K_{1:3}=\begin{bmatrix}k_{\text{中}}\\k_{\text{华}}\\k_{\text{人}}\end{bmatrix},\qquad V_{1:3}=\begin{bmatrix}v_{\text{中}}\\v_{\text{华}}\\v_{\text{人}}\end{bmatrix}$</p>

当前 Query 是“人”这个位置新算出的 $q_{\text{人}}$。它要做的是：

<p align="center">$s_{\text{人}}=q_{\text{人}}K_{1:3}^{T}=\begin{bmatrix}q_{\text{人}}k_{\text{中}}^{T}&q_{\text{人}}k_{\text{华}}^{T}&q_{\text{人}}k_{\text{人}}^{T}\end{bmatrix}$</p>

这就是完整 Attention 分数矩阵中最下面的那一行。完整方阵中“中/中”“华/中”等旧行描述的是历史位置当时如何关注上下文；它们已经完成了自己的计算，本轮不再需要，也通常不会作为缓存保存。方阵右上角对应旧位置关注未来位置，受因果 Mask 限制，同样不需要补算。当前 Query 虽然读取了缓存的 Key，但 Query 已经改变，因此最下面这一行的点积要新算。

因此，带 KV Cache 的 Decode 单步不会重新计算一个 `3×3` 的分数矩阵，而是计算：

```text
当前 Q：[1, d_k]
完整 Kᵀ：[d_k, 3]
分数：  [1, 3]
```

对这一行做缩放、Softmax，再读取缓存中的 Value：

<p align="center">$o_{\text{人}}=\operatorname{softmax}\left(\frac{s_{\text{人}}}{\sqrt{d_k}}\right)V_{1:3}$</p>

这里缓存的是历史 K/V，不是 $QK^T$ 的整张分数矩阵。当前的 $q_{\text{人}}$、$k_{\text{人}}$、$v_{\text{人}}$ 本轮计算；新的 K/V 还会追加到缓存，供下一步使用。

<p align="center">
  <img src="/img/in-post/ai-infra-kv-cache-last-row.svg" alt="带 KV Cache 的 Decode 只用当前 Query 计算最后一行 Attention 分数，历史 Key 和 Value 从缓存读取" style="max-width: 100%;">
</p>

## 2. Attention 输出还要经过后续模型计算

上面的 $o_{\text{人}}$ 只是**某一层、某一个 head 的 Attention 输出**。多个 head 的结果需要拼接并经过输出投影，再结合本层的残差、归一化和 FFN 计算，才能传到下一层。这里按常见的 Decoder-only 模型描述，具体归一化位置取决于架构。

如果有 `L` 个 Transformer Block，当前“人”这个位置的表示会沿着网络向前传播：

<p align="center">$h_{\text{人}}^{(0)}\rightarrow\text{Block}_1\rightarrow h_{\text{人}}^{(1)}\rightarrow\cdots\rightarrow\text{Block}_L\rightarrow h_{\text{人}}^{(L)}$</p>

最后再经过模型末端的归一化，得到用于词表预测的最终隐藏向量。记它为：

<p align="center">$h_{\text{人}}\in\mathbb{R}^{H}$</p>

这里的下标“人”表示它位于“人”这个位置，并不表示它只包含这个字的信息。由于因果 Attention，这个向量已经融合了它能看到的前文“中、华、人”。

## 3. Prefill 取最后一个有效位置预测新 token

换一个输入边界：如果一开始就把“中华人”作为完整 Prompt 做 Prefill，经过全部 Transformer Block 和最终归一化后，会得到所有位置的隐藏状态：

<p align="center">$H_{\text{all}}=\begin{bmatrix}h_{\text{中}}\\h_{\text{华}}\\h_{\text{人}}\end{bmatrix}\in\mathbb{R}^{3\times H}$</p>

每一行负责一个位置的“下一 token”预测：

| 当前隐藏向量 | 能看到的内容 | 它对应的预测目标 |
| --- | --- | --- |
| $h_{\text{中}}$ | 中 | “中”之后的 token |
| $h_{\text{华}}$ | 中华 | “中华”之后的 token |
| $h_{\text{人}}$ | 中华人 | Prompt 之后的第一个新 token |

现在我们要接着整个 Prompt 继续生成，所以需要的是最后一个有效位置：

<p align="center">$h_{\text{last}}=H_{\text{all}}[-1,:]=h_{\text{人}}$</p>

“最后一行”描述的是**序列位置维度**，不是 Transformer 的最后一层。它通常已经经过全部 Transformer Block；只是从最后一层输出的 `[batch, sequence, hidden]` 矩阵中，选择每条序列最后一个有效 token 的那一行。

<p align="center">
  <img src="/img/in-post/ai-infra-kv-cache-last-hidden.svg" alt="完整 Prompt 经过所有 Transformer Block 后得到隐藏状态矩阵，取最后一个有效位置的最后一行送入 LM Head" style="max-width: 100%;">
</p>

代码中经常看到：

```python
# hidden_states: [batch_size, sequence_length, hidden_size]
last_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_size]
```

另一种常见接口先返回所有位置的词表分数，再选最后位置：

```python
# outputs.logits: [B, T, V]
next_token_logits = outputs.logits[:, -1, :]  # [B, V]
```

这里取的是最后位置的 logits；前一个例子取的是最后位置的 hidden，再交给 LM Head。两者选择的是同一个序列位置。

如果 batch 中有 padding，不能无条件取最后一列；应该取每条序列最后一个有效 token 的位置。左 padding、右 padding 或 packed sequence 的实现方式不同，但原则相同：**取最后一个有效位置，而不是盲目取数组最后一列。**

## 4. LM Head：用一个隐藏向量给整个词表打分

得到 $h_{\text{last}}$ 后，LM Head 通常是一个线性层。假设隐藏维度为 `H`，词表大小为 `V`：

<p align="center">$h_{\text{last}}\in\mathbb{R}^{1\times H},\qquad W_{\text{LM}}\in\mathbb{R}^{H\times V}$</p>

两者相乘得到词表中每个 token 的分数：

<p align="center">$z=h_{\text{last}}W_{\text{LM}}\in\mathbb{R}^{1\times V}$</p>

第 `j` 个 logit 是：

<p align="center">$z_j=\sum_{i=1}^{H}h_iW_{ij}$</p>

也就是说，LM Head 的每一列对应一个候选 token。隐藏向量会分别与这些列做点积，得到每个候选的匹配分数。

### 4.1 用一个小词表手算

为了看清楚矩阵乘法，暂时把隐藏维度缩小为 3，把词表缩小为四个 token：`民、工、们、生`。

设最终隐藏向量是：

<p align="center">$h_{\text{last}}=\begin{bmatrix}2&1&-1\end{bmatrix}$</p>

LM Head 的权重矩阵为：

<p align="center">$W_{\text{LM}}=\begin{bmatrix}1&0&1&-1\\1&2&0&0\\-1&0&1&1\end{bmatrix}$</p>

四列的顺序对应 `民、工、们、生`。矩阵乘法得到：

<p align="center">$z=h_{\text{last}}W_{\text{LM}}=\begin{bmatrix}4&2&1&-3\end{bmatrix}$</p>

展开第一列，“民”的分数是：

<p align="center">$z_{\text{民}}=2\times1+1\times1+(-1)\times(-1)=4$</p>

四个 logits 可以写成：

```text
民： 4
工： 2
们： 1
生：-3
```

上面的数字只是演示，真实权重由训练得到。例如训练文本里“中华人”的下一 token 是“民”，交叉熵损失就会推动模型提高“民”的预测概率，梯度同时更新 LM Head 和前面的 Transformer。

这些是原始分数，不是概率，也不要求加起来等于 1。它们会交给后续的解码策略：

- **贪心解码**：直接取最大 logit，这个例子会选“民”；
- **Temperature**：先缩放 logits，改变分布的尖锐程度；
- **Top-K / Top-P**：过滤候选，再从剩余分布采样。

如果需要概率，可以计算：

<p align="center">$p_j=\frac{e^{z_j}}{\sum_{k=1}^{V}e^{z_k}}$</p>

这里的 Softmax 沿词表维度归一化，得到“下一个 token 选谁”的分布；Attention 里的 Softmax 沿上下文位置归一化，决定“当前 Query 从哪里读取信息”。两个分布的含义不同。

实际实现中通常先做数值稳定的处理，再进行采样。`argmax` 场景甚至不需要真的计算 Softmax，因为指数函数保持大小关系。

## 5. 真实模型中的张量形状

假设 batch 大小为 `B`，Prompt 长度为 `T`，隐藏维度为 `H`，词表大小为 `V`：

```text
最后一层隐藏状态： [B, T, H]
取最后一个有效位置：[B, H]
LM Head：           [B, H] × [H, V]
得到 logits：        [B, V]
```

在 Prefill 阶段，如果只需要每条 Prompt 的第一个新 token，推理引擎可以只对最后一个有效位置执行 LM Head。前面的 Transformer 仍需计算各位置以建立缓存，省掉的是其他位置的词表投影：

```text
完整做法： [B, T, H] × [H, V] → [B, T, V]
实际需要： [B, H]    × [H, V] → [B, V]
```

完整的 `[B, T, V]` logits 在训练或需要每个位置预测时很有用；普通自回归生成下一 token 时只需要每条序列最后位置的 `[B, V]` logits。两种做法在最后一个位置上的数学结果相同，区别是是否计算和保存其他位置的词表投影。

PyTorch 中，`nn.Linear` 的权重通常保存为 `[out_features, in_features]`，所以代码写成：

```python
import torch

lm_head = torch.nn.Linear(H, V, bias=False)

# h_last: [B, H]
logits = lm_head(h_last)          # [B, V]
# 等价于 h_last @ lm_head.weight.T
```

有些模型会让输入 Embedding 和 LM Head 共享权重。若 Embedding 表为 $E\in\mathbb{R}^{V\times H}$，则输出可以写成：

<p align="center">$z=h_{\text{last}}E^T$</p>

输入时，token ID 用来从 `E` 中查一行；输出时，隐藏向量与 `E` 的每一行做点积。这两种用途共享参数，但并不是数学上的互逆运算。

## 6. Prefill 和 Decode 汇合到同一条输出路径

两者都会经过“最后位置隐藏向量 → LM Head → 下一个 token”这条路径，但前面的计算方式不同。

### 6.1 Prefill

Prompt 的所有 token 同时进入模型。每个位置都能得到隐藏状态和相应的 K/V；因果 Mask 保证位置不会看到未来。建立缓存后，只取最后一个有效位置的 logits，预测第一个新 token：

```text
Prompt：[中 华 人]
Prefill 最后一个位置 → logits → 预测“民”
```

### 6.2 Decode

新 token 作为下一轮输入，每一层只新增一行 hidden state。当前 Query 查询完整的历史 K/V，得到当前 token 的输出；经过所有层后，当前这一个位置本身就是最后位置，因此直接进入 LM Head：

```text
上一轮预测“民”
    ↓ 作为本轮输入
只计算“民”的一行
    ↓
LM Head → 预测下一个 token
```

还有一个时间上的先后关系：**刚预测出“民”时，缓存里通常还没有“民”的 K/V。只有下一轮把“民”送入模型，才会逐层计算并缓存它的 K/V。** 如果“民”就是停止 token、请求已经结束，也就不需要再为它做下一轮 Decode。

所以，Decode 中“只算一行 Attention”和“最终取最后一行”并不是同一件事：前者是 KV Cache 带来的增量计算；后者是从最终隐藏状态中选择要用于下一 token 预测的序列位置。

## 7. 把整条链路放回 KV Cache

一次普通 Decode 的数据流可以写成：

```text
当前输入 token + 各层已有 KV Cache
        ↓
每一层计算当前 token 的 q、k、v
        ↓
把当前 k、v 追加到该层 KV Cache
        ↓
当前 q 查询该层完整 K/V
        ↓
得到当前层输出，继续经过 FFN 和后续层
        ↓
最终 hidden 的最后一个有效位置 h_last
        ↓
LM Head 产生 [词表大小] 个 logits
        ↓
采样或取最大值，得到下一个 token
```

其中有三类容易混在一起的对象：

| 对象 | 作用 | 是否作为标准 KV Cache 长期保存 |
| --- | --- | --- |
| 历史 K/V | 供未来 Query 反复检索和读取 | 是 |
| $QK^T$ Attention 分数 | 当前 Query 这一次计算的匹配结果 | 通常不是 |
| 最终 hidden / logits | 产生下一个 token 的当前结果 | 通常只保留完成当前步骤所需的状态 |

KV Cache 省掉的是历史 token 的重复投影和重复逐层计算；LM Head 则负责把当前最终隐藏向量翻译成对整个词表的选择分数。把这两段连接起来，就不会再把“缓存历史 K/V”“计算当前 Attention 一行”和“取最后一行预测 token”混成同一个操作。

## KV Cache 系列

- [基础：Transformer 推理中的增量计算]({% post_url 2026-09-07-技术-KV Cache与Transformer推理 %})
- [优化（一）：MQA 与 GQA]({% post_url 2026-09-07-技术-KV Cache优化一MQA与GQA %})
- [优化（二）：MLA]({% post_url 2026-09-07-技术-KV Cache优化二MLA %})
- [优化（三）：局部注意力与缓存淘汰]({% post_url 2026-09-07-技术-KV Cache优化三局部注意力与缓存淘汰 %})
- [优化（四）：投机解码]({% post_url 2026-09-08-技术-KV Cache优化四投机解码 %})
- **补充：从最后一行到下一个 Token（本文）**

## 参考资料

- [Continuous Batching：从请求排队到逐轮调度](https://liumengxuan04.github.io/技术/2026/09/24/技术-Continuous-Batching从请求排队到逐轮调度/)
- [AIInfraGuide：从 Transformer 到 LLM 自回归生成深入理解](https://caomaolufei.github.io/AIInfraGuide/guides/模块一-前置知识/transformer/38-从transformer到llm自回归生成深入理解/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
