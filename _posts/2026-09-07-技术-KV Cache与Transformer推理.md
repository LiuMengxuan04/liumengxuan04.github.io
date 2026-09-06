---
layout:     post
title:      "KV Cache：Transformer 推理中的增量计算"
subtitle:   "从 Prefill、Decode 到 K/V 的追加与显存带宽"
date:       2026-09-07 20:00:00 +0800
author:     "Liu Mengxuan"
mathjax:    true
header-img: "img/post-bg-miui6.jpg"
categories: [技术]
tags:       [技术, AI Infra, Transformer, KV Cache, 推理, Attention, GQA, MQA]
---

大语言模型生成文本时，并不是每次都把已经处理过的整段文本重新计算一遍。它会把历史 token 在每一层产生的 Key 和 Value 保存下来，下一次生成时直接复用。这块保存下来的状态就是 KV Cache。

KV Cache 的核心可以先浓缩成一句话：

> Prefill 一次处理完整 Prompt 并建立缓存；Decode 每次只处理一个新 token，用它的 Query 查询历史 KV，再把新的 Key 和 Value 追加进去。

这句话同时解释了几个常见现象：为什么推理分成 Prefill 和 Decode，为什么 Decode 的矩阵乘法更像矩阵乘向量，为什么缓存 K/V 而不是 Q，以及为什么上下文变长后显存和带宽压力会越来越大。

<p align="center">
  <img src="/img/in-post/ai-infra-kv-cache-prefill-decode.svg" alt="Transformer 推理中的 Prefill、Decode 与 KV Cache" style="max-width: 100%;">
</p>

## 1. 先看 Attention 到底需要什么

对单个 attention head，缩放点积注意力可以写成：

<p align="center">$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\dfrac{QK^T}{\sqrt{D_h}}\right)V$</p>

如果序列中有 `T` 个 token，那么在一层、一个 head 内可以把它们写成：

<p align="center">$Q,K,V\in\mathbb{R}^{T\times D_h}$</p>

其中每一行对应一个 token：

```text
Q 的一行：这个 token 想从上下文中寻找什么
K 的一行：这个 token 提供什么可匹配的索引
V 的一行：这个 token 真正携带的内容
```

对第 `t` 个 token 来说，因果注意力只允许它看到自己以及左侧的历史位置，因此它的输出是：

<p align="center">$o_t=\operatorname{softmax}\left(\dfrac{q_tK_{1:t}^{T}}{\sqrt{D_h}}\right)V_{1:t}$</p>

这个公式非常关键：

- 当前输出需要当前的 `q_t`；
- 它需要所有历史位置的 `K_{1:t}`，用来计算匹配分数；
- 它需要所有历史位置的 `V_{1:t}`，用来按照分数取回内容；
- 公式中没有 `q_1` 到 `q_{t-1}`。

最后一点正是“为什么没有 Q Cache”的数学起点。

## 2. Prefill：先把 Prompt 的缓存建好

假设输入 Prompt 被分成：

```text
x₁   x₂   x₃   ...   xₜ
```

这些 token 在开始推理时全部已知。Prefill 阶段会把整段输入送入模型，在每一层中并行计算：

<p align="center">$X\in\mathbb{R}^{T\times H}\xrightarrow{W_Q,W_K,W_V}Q,K,V$</p>

对于线性投影，序列维度是 `T`，所以更接近：

<p align="center">$[T,H]@[H,D_h]\rightarrow[T,D_h]$</p>

这类计算可以组织成较大的矩阵乘法，更容易利用 GPU 的 Tensor Core 和并行带宽。

### 2.1 为什么 Prefill 要计算整个序列？

最终从 Prompt 预测下一个 token 时，通常只会取最后一个位置的 logits。但这不代表前面的位置可以完全不计算。

原因有两个：

1. 每一层都需要为所有历史位置生成 K 和 V，供后续位置查询；
2. Transformer 是多层堆叠的，上一层所有位置的输出会成为下一层所有位置的输入。

因此，Prefill 的目标不是只算最后一个 token，而是逐层计算整段 Prompt 的中间状态，并把各层的 KV Cache 初始化完整。中间状态会继续流向下一层，真正需要跨生成步骤长期保存的是每层的 K/V。

### 2.2 因果 Mask 如何保留自回归语义？

如果一次性计算：

<p align="center">$Q_{1:T}K_{1:T}^{T}$</p>

会得到一个 `T×T` 的矩阵。以 `T=3` 为例：

<p align="center">$\begin{bmatrix}q_1k_1^T&q_1k_2^T&q_1k_3^T\\q_2k_1^T&q_2k_2^T&q_2k_3^T\\q_3k_1^T&q_3k_2^T&q_3k_3^T\end{bmatrix}$</p>

右上角代表“当前位置读取未来 token”，必须被 Mask 成负无穷：

<p align="center">$\begin{bmatrix}a_{11}&-\infty&-\infty\\a_{21}&a_{22}&-\infty\\a_{31}&a_{32}&a_{33}\end{bmatrix}$</p>

逐行 Softmax 后，负无穷位置的权重为 0。于是：

<p align="center">$\text{第 1 行}=q_1K_1^T,\qquad\text{第 2 行}=q_2K_{1:2}^T,\qquad\text{第 3 行}=q_3K_{1:3}^T$</p>

这在数学上等价于依次计算每个位置的有效历史范围。区别在于，Prefill 可以把大量位置组织成并行矩阵运算。实际的 FlashAttention 等高效实现还会通过分块避免显式物化完整的 `T×T` 矩阵，但因果 Mask 的逻辑不变。

<p align="center">
  <img src="/img/in-post/ai-infra-kv-cache-causal-mask.svg" alt="因果 Mask 让 Prefill 的并行计算等价于逐步计算" style="max-width: 100%;">
</p>

### 2.3 Prefill 结束时保存什么？

对每一层 `l`，Prefill 会得到：

<p align="center">$K^{(l)}_{1:T},\qquad V^{(l)}_{1:T}$</p>

并将它们放入该层自己的缓存中：

```text
第 1 层：K Cache、V Cache
第 2 层：K Cache、V Cache
...
第 L 层：K Cache、V Cache
```

缓存保存的是 K/V 投影的结果，不是原始 token，也不是完整的 Attention 分数矩阵。Q 在 Prefill 中参与了各位置的注意力计算，但没有必要作为未来步骤的持久状态保存下来。

## 3. Decode：每次只处理一个新 token

Prefill 根据最后一个 Prompt 位置的 logits 采样出第一个新 token，例如：

```text
Prompt：x₁ x₂ x₃
Prefill 预测：x₄
```

当 `x₄` 被送回模型、用于预测 `x₅` 时，就进入 Decode。此时每层只接收当前这个新 token 的 hidden state：

<p align="center">$h^{(l-1)}_{t+1}\in\mathbb{R}^{1\times H}$</p>

经过当前层的 Q/K/V 投影：

<p align="center">$q^{(l)}_{t+1}=h^{(l-1)}_{t+1}W_Q^{(l)},\qquad k^{(l)}_{t+1}=h^{(l-1)}_{t+1}W_K^{(l)},\qquad v^{(l)}_{t+1}=h^{(l-1)}_{t+1}W_V^{(l)}$</p>

接下来必须先把当前的 `k` 和 `v` 追加进去，再计算当前 token 的注意力：

<p align="center">$K^{(l)}_{1:t+1}=\operatorname{Concat}\left(K^{(l)}_{1:t},k^{(l)}_{t+1}\right)$</p>

<p align="center">$V^{(l)}_{1:t+1}=\operatorname{Concat}\left(V^{(l)}_{1:t},v^{(l)}_{t+1}\right)$</p>

<p align="center">$o^{(l)}_{t+1}=\operatorname{softmax}\left(\dfrac{q^{(l)}_{t+1}\left(K^{(l)}_{1:t+1}\right)^T}{\sqrt{D_h}}\right)V^{(l)}_{1:t+1}$</p>

这里包含一个容易遗漏的细节：当前 token 通常允许关注自己，所以当前步的 `k_{t+1}` 和 `v_{t+1}` 也要参与当前注意力。它们不是等到下一轮才第一次使用。

整个 Decode 单步可以概括成：

```text
当前 token
    ↓
当前层的 q、k、v
    ↓
追加 k、v 到当前层缓存
    ↓
当前 q 查询完整 K/V
    ↓
残差、归一化、FFN，进入下一层
    ↓
最终 hidden → LM Head → 下一个 token
```

<p align="center">
  <img src="/img/in-post/ai-infra-kv-cache-qkv-roles.svg" alt="Q、K、V 在 KV Cache 中的分工" style="max-width: 100%;">
</p>

### 3.1 如果没有 KV Cache，会发生什么？

假设当前上下文已经有 `T` 个 token。没有 KV Cache 时，为了得到下一步所需的历史 K/V，只能把长度为 `T` 的前缀重新送过所有 Transformer 层：

```text
没有 KV Cache：每生成一步，都重新计算整个前缀
[T, H] → 每层的 Q/K/V、Attention、FFN → 下一个 token
```

有了 KV Cache，历史 K/V 已经存在，每层只需要处理新 token 的一行：

```text
使用 KV Cache：只计算新增的一行
[1, H] → 当前 q/k/v → 查询 K/V Cache → 下一个 token
```

它省掉的是历史 token 的重复投影、历史位置的重复 Attention 输出以及逐层 FFN 计算。不过，当前 Query 仍然要读取长度为 `T` 的 K/V：

<p align="center">$[1,D_h]@[D_h,T]\rightarrow[1,T]$</p>

所以 KV Cache 并没有让单步 Attention 变成 $O(1)$；标准注意力的单步读取和计算量仍会随上下文长度近似线性增长。它本质上是用显存保存可复用结果，换掉更昂贵的整段前缀重算。

## 4. 为什么只需要当前 token 的 Q？

### 4.1 因果性让历史输出保持不变

在因果 Transformer 中，位置 `t` 的表示只能依赖 `1` 到 `t` 的 token，不能依赖未来位置。因此，当新 token `t+1` 出现时：

- 位置 `1` 到 `t` 的隐藏状态不会因为未来 token 出现而改变；
- 它们之前计算出的注意力输出不需要重算；
- 历史 Query 已经完成了自己的注意力计算，不再参与生成 `t+1` 之后的输出。

未来真正需要的是一个新的查询：

<p align="center">$o_{t+1}=\operatorname{Attention}\left(q_{t+1},K_{1:t+1},V_{1:t+1}\right)$</p>

公式里只有 `q_{t+1}`，而没有 `q_1` 到 `q_t`。

### 4.2 K/V 为什么必须保留？

未来的每个 Query 都要和所有历史 Key 做匹配，并根据匹配结果读取所有历史 Value：

<p align="center">$q_{t+2}\rightarrow K_{1:t+2}\rightarrow V_{1:t+2}$</p>

因此，K/V 是会被反复访问的历史记忆：

```text
Q：当前这一次要问的问题，用完即释放
K：历史位置的检索索引，未来每一步都会匹配
V：历史位置的内容，未来每一步都可能读取
```

这就是 KV Cache 的名字来源，也是它与 Q 的根本区别。

### 4.3 为什么不把 Q 也拼起来？

把历史 Q 保存下来并不会让未来的注意力多得到任何信息，因为未来的公式不读取它们。强行保存 Q 只会增加显存占用；如果把历史 Q 再送进计算，还会重新计算已经确定的历史输出。

需要注意，“只保留当前 Q”描述的是标准自回归 Decode 的核心依赖，并不意味着所有推理系统都只能保存 K/V。一些系统可能为了调度、量化、投机解码或其他优化保存额外状态，但那属于额外工程状态，不是标准 KV Cache 的必要组成。

## 5. `Oₜ`、下一步 token 和下一层 QKV 的关系

“上一步的输出变成下一步的 QKV”这句话需要分层理解。

### 5.1 在同一层内

当前层先接收一个 hidden state，再通过三组不同权重生成 Q、K、V：

<p align="center">$h^{(l-1)}_{t}=\text{上一层输出},\qquad(q^{(l)}_t,k^{(l)}_t,v^{(l)}_t)=h^{(l-1)}_t(W_Q^{(l)},W_K^{(l)},W_V^{(l)})$</p>

Q/K/V 是同一个当前 hidden state 的三种投影，不是由某个注意力输出直接复制出来的。

### 5.2 在生成时间上

最后一层 hidden state 经过词表投影得到 logits，再根据 logits 选择下一个 token：

<p align="center">$z_t=h^{(L)}_tW_{out}$</p>

<p align="center">$z_t\rightarrow x_{t+1}\rightarrow\text{Embedding}(x_{t+1})\rightarrow h^{(0)}_{t+1}$</p>

随后，新的 token 从第 1 层开始向前传播，在每一层产生新的 Q/K/V：

<p align="center">$h^{(0)}_{t+1}\rightarrow(q^{(1)}_{t+1},k^{(1)}_{t+1},v^{(1)}_{t+1})\rightarrow h^{(1)}_{t+1}\rightarrow\cdots\rightarrow(q^{(L)}_{t+1},k^{(L)}_{t+1},v^{(L)}_{t+1})$</p>

因此，更准确的依赖链是：

```text
最终 hidden
    ↓ LM Head
logits
    ↓ 采样
新 token
    ↓ Embedding
新 token 的各层 hidden
    ↓
每一层产生新的 q、k、v
```

某一层的注意力输出是下一层输入的一部分，但不是“直接变成下一时间步所有层 QKV”。每一层都有自己的权重和自己的 KV Cache。

## 6. 用一个具体例子走一遍

假设 Prompt 是：

```text
我   是   你
```

### 6.1 Prefill 结束

模型一次处理这 3 个 token。在某一层得到：

```text
Q：q我、q是、q你       参与本轮注意力，随后可以释放
K：k我、k是、k你       写入 K Cache
V：v我、v是、v你       写入 V Cache
```

最后一个位置的输出用于预测下一个 token，假设结果是“的”。

### 6.2 Decode “的”

“的”作为新输入进入每一层。在当前层中：

```text
计算 q的、k的、v的
    ↓
K Cache = [k我, k是, k你, k的]
V Cache = [v我, v是, v你, v的]
    ↓
q的 查询完整的 K/V
    ↓
得到当前输出，并预测“人”
```

这里并没有重新计算“我 是 你”的 Q/K/V。历史 K/V 直接从缓存读取，只有“的”的投影是新增计算。

### 6.3 Decode “人”

生成“人”时，重复同样的步骤：

```text
计算 q人、k人、v人
    ↓
把 k人、v人 追加到已有缓存
    ↓
q人 查询 [我, 是, 你, 的, 人] 的 K/V
```

于是缓存只增长，不需要从头重建：

<p align="center">$K_{1:3}\rightarrow K_{1:4}\rightarrow K_{1:5}$</p>

<p align="center">$V_{1:3}\rightarrow V_{1:4}\rightarrow V_{1:5}$</p>

## 7. KV Cache 占用多少显存？

设：

- `L`：Transformer 层数；
- `B`：同时服务的序列数；
- `T`：当前上下文长度，包括 Prompt 和已经生成的 token；
- `N_KV`：保存 K/V 的 head 数；
- `D_h`：每个 head 的维度；
- `bytes`：每个元素占用的字节数。

那么 KV Cache 的元素数量约为：

<p align="center">$2\times L\times B\times T\times N_{KV}\times D_h$</p>

乘上数据类型的字节数，就得到显存占用：

<p align="center">$M_{KV}=2\times L\times B\times T\times N_{KV}\times D_h\times\text{bytes}$</p>

最前面的 `2` 来自 K 和 V 两份缓存。

### 7.1 一个具体数字

假设：

```text
L = 32，B = 1，T = 4096
NKV = 8，Dh = 128，FP16 = 2 bytes
```

则：

<p align="center">$M_{KV}=2\times32\times1\times4096\times8\times128\times2\text{ bytes}=512\text{ MiB}$</p>

如果使用标准 MHA，并且 `NKV=Nh=32`，同样条件下就是约 `2048 MiB`。这也是 GQA/MQA 对长上下文推理很有价值的原因：它们减少了需要保存的 KV head 数。

<p align="center">
  <img src="/img/in-post/ai-infra-kv-cache-memory.svg" alt="KV Cache 的显存占用与 MHA、GQA、MQA 对比" style="max-width: 100%;">
</p>

### 7.2 MHA、GQA、MQA 的区别

```text
MHA：每个 Query head 都有自己的一对 K/V head
GQA：Query heads 被分组，同一组内共享一对 K/V head
MQA：所有 Query heads 共享同一对 K/V head
```

它们主要通过减少 `N_KV` 来减少 KV Cache：

<p align="center">$N_{KV}^{MQA}=1\le N_{KV}^{GQA}<N_{KV}^{MHA}=N_h$</p>

代价是 K/V 的表达自由度减少，需要在显存、带宽和模型质量之间做取舍。

## 8. 为什么 Decode 更像矩阵乘向量？

Prefill 的线性层输入通常是：

<p align="center">$[T,H]@[H,H]\rightarrow[T,H]$</p>

Decode 的单个请求每次只有一个 token：

<p align="center">$[1,H]@[H,H]\rightarrow[1,H]$</p>

后者在数学形态上接近矩阵乘向量。注意力分数也类似：

<p align="center">$[1,D_h]@[D_h,T]\rightarrow[1,T]$</p>

注意这里的“接近 GEMV”描述的是单请求、单 token 的计算形态，不意味着底层一定调用一个名为 GEMV 的 kernel：

- 单请求时，确实很像 GEMV；
- 同时服务 `B` 个请求时，可以把它们组成 `[B,H]`，使用小规模 GEMM 或 grouped GEMM；
- 推理框架还会使用融合 Kernel、PagedAttention 等专用实现。

Decode 的问题不在于乘加次数突然变多，而在于每次只做很少的计算，却需要读取大块模型权重和不断增长的 KV Cache。因此它通常比 Prefill 更容易受显存带宽和访存效率限制。

可以用下面的因果链理解：

```text
每步只有一个新 token
    ↓
矩阵的有效行数很小
    ↓
单次计算的算术强度下降
    ↓
权重和 KV Cache 的读取占比上升
    ↓
Decode 更容易成为 memory-bound
```

这也是推理系统需要关注连续 batching、KV Cache 布局、PagedAttention、KV Cache 量化和 GQA/MQA 的原因。

## 9. 几个容易混淆的地方

### 9.1 KV Cache 不是“把整段隐藏状态都缓存起来”

标准 KV Cache 主要保存每一层的 K 和 V。注意力分数、Softmax 权重、Attention 输出以及 FFN 中间结果通常不会作为跨 Decode 步骤的永久缓存保存，因为它们依赖当前 Query 或当前计算路径。

### 9.2 当前 token 的 K/V 不是下一轮才使用

当前 token 通常可以关注自己，所以正确顺序是：

```text
先计算当前 k/v
    ↓
追加到缓存
    ↓
当前 q 查询包含自己的完整缓存
```

如果把当前 K/V 排除在外，就改变了标准因果注意力的定义。

### 9.3 LayerNorm 不会跨 token 统计

LayerNorm/RMSNorm 通常沿每个 token 的隐藏维 `H` 计算，不是把整个序列的 token 混在一起计算均值和方差。Prefill 处理整段序列的主要原因，是需要并行生成所有位置的中间状态和 K/V，而不是因为 LayerNorm 必须跨序列统计。

### 9.4 “只计算当前 Q”不等于“只存储一个向量”

当前 Q 确实只有一个新 token 的 Query，但它要读取长度为 `T` 的 K/V Cache。Decode 的输入计算量小，不代表访问的数据量永远小；上下文越长，读取历史 KV 的成本越高。

## 10. 总结

把整个过程压缩成一条流水线：

```text
自然语言
    ↓ tokenizer
token IDs
    ↓ Embedding
Prompt hidden states
    ↓ Prefill：整段并行计算
每层建立 K₁:ₜ、V₁:ₜ
    ↓ 预测并采样新 token
Decode：一次处理一个新 token
    ↓
产生当前 q、k、v
    ↓
追加 k/v，当前 q 查询完整 KV Cache
    ↓
预测下一个 token，循环继续
```

最重要的三句话是：

1. Prefill 用完整 Prompt 建立每一层的 KV Cache；
2. Decode 只需要当前 token 的 Q，以及历史和当前的 K/V；
3. Q 是一次性的查询，K/V 是会被未来反复访问的历史记忆。

理解这三点之后，KV Cache、Prefill/Decode、GEMM/GEMV、GQA/MQA 以及长上下文显存压力就会落在同一条因果链上。
