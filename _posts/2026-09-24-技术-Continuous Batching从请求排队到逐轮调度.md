---
layout:     post
title:      "Continuous Batching：从请求排队到逐轮调度"
subtitle:   "借助操作系统调度，理解 LLM 怎样把不同进度的请求放进同一批计算"
date:       2026-09-24 12:00:00 +0800
author:     "Liu Mengxuan"
mathjax:    true
header-img: "img/post-bg-miui6.jpg"
categories: [技术]
tags:       [技术, AI Infra, Transformer, 推理, Continuous Batching, KV Cache, 调度]
---

一个用户让模型回答“好的”，另一个用户让模型写一篇长文章。如果把它们放进同一个 batch，前一个请求很快结束，后一个却还要继续生成很久。此时即使有新用户在排队，固定成员的 batch 也不能马上把空出来的容量交给他。

Continuous Batching（连续批处理）改变的是这个调度方式：**每一轮模型计算结束后，完成的请求退出，等待中的请求在资源允许时加入，下一轮可以换一组成员继续算。**

它很像操作系统安排多个带状态的任务运行。不过，LLM 推理还有一个重要特点：调度器选出来的一组请求，可以合成一个 batch，共同使用模型权重完成计算。

## 1. 从一个请求的生成，走到多个请求合批

带 KV Cache 的自回归生成分为 Prefill 和 Decode。Prefill 处理 prompt，建立各层的 KV Cache，并通过最后一个位置的 logits 采样出第一个输出 token。随后每一步 Decode 处理上一步得到的 token，追加它的 K/V，再预测下一个 token。

对同一个请求来说，后一个 token 依赖前面的生成结果，通常要一步一步推进。不同请求之间却没有这种依赖：A 的下一句话不需要 B 的生成结果，所以可以一起计算。

```text
一次 Decode 前向：
    A 的当前 token + A 的 KV Cache → A 的下一个 token
    B 的当前 token + B 的 KV Cache → B 的下一个 token
    C 的当前 token + C 的 KV Cache → C 的下一个 token
```

这里讨论普通自回归 Decode，暂不引入投机解码。一次前向通常让每个参与的请求各推进一个 token，而不是让 A 生成完整回答以后才轮到 B。

如果 batch 成员始终固定，这一批就要等最长的请求结束才能接纳新成员。短请求可以先结束、先把结果返回给用户，但释放出来的容量不能用来接待排队中的新请求。即使实现能够跳过已完成序列的计算，剩余的小 batch 也可能无法充分利用 GPU。

因此，“静态批处理要同时结束”更准确的含义是：**整批结束前不补充新成员，而不是所有用户必须同时收到答案。**

## 2. 在每轮结束后补进新请求

假设最多同时处理两个请求。A 还需要 6 步 Decode，B、C、D 各需要 2 步。先处理 A、B，C、D 等待。

为了单独观察调度，先假设四个请求都已完成 Prefill，只看剩余的 Decode 工作。这里的步数是事后给出的长度，真实服务通常事先不知道一个请求会在哪一步结束。

<p align="center">
  <img src="/img/in-post/ai-infra-continuous-batching-timeline.svg" alt="静态批处理前六轮只处理 A 和 B，B 在第二轮结束后不能补入新请求，第七八轮才处理 C 和 D；连续批处理在第三轮补入 C，第五轮补入 D，第六轮全部完成" style="max-width: 100%;">
</p>

静态批处理中，B 在第 2 轮结束后，A 还要独自执行 4 轮。直到 A 完成，C 和 D 才能组成下一批。

连续批处理则在第 2 轮结束时移除 B，第 3 轮让 C 和 A 一起执行；C 在第 4 轮结束，第 5 轮再补入 D。A 仍然走完自己的 6 步，但这段期间系统也完成了 B、C、D。

这就是 **Iteration-level Scheduling，迭代级调度**。batch 的成员可以在相邻轮次之间变化；“连续”描述的是这种持续接纳请求的能力，不是往正在执行的 GPU kernel 中途塞入任务。

图中的两行只是逻辑上的并发名额，不是两个固定的 GPU 核心，也不是两个独占的执行窗口。每一列里的请求合批完成本轮计算。横轴表示迭代轮次，各轮的实际耗时会随 batch 大小、上下文长度和工作类型变化，因此不能直接把 8 轮与 6 轮换算成真实加速比。

## 3. 用操作系统调度理解请求的进退

可以把每个生成请求看成一个带状态的任务。服务接收请求，将它放入等待队列；调度器根据剩余显存、每轮 token 预算和调度策略，选择本轮执行的工作；计算结束后更新状态，再决定下一轮的安排。

| 操作系统中的概念 | LLM 推理服务中的对应物 |
| --- | --- |
| 进程或线程 | 一个生成请求 |
| 就绪队列 | 等待执行的请求队列 |
| 任务的执行状态 | 已有 token、生成位置、各层 KV Cache 等 |
| 调度器选择运行任务 | 调度器选择本轮参与计算的请求 |
| 任务结束并释放资源 | 触发停止条件，回收不再需要的资源 |

这个类比最有用的地方，是把“保存任务进度”和“安排下一次执行”分开：请求没有在本轮执行，并不意味着它已经丢失之前的计算结果。

不过，CPU 抢占式调度可以借助时钟中断切换线程；Continuous Batching 通常在明确的模型迭代边界调整成员。从服务调度层面看，它更接近在工作片段结束时交回控制权。一个 Decode 步有点像时间片，但它是一段工作量，不是固定的几毫秒。

另一个区别是，单个 CPU 核心通常在不同线程间切换，推理调度器则会选出一组请求合批计算。它不仅要考虑谁等得久，还要考虑这组工作放在一起是否高效、是否放得进显存，以及是否会拖慢正在输出的用户。

## 4. 生成进度不同，也可以一起计算

A 已经生成了 100 个 token，C 才生成了 2 个，它们仍然可以一起 Decode。这一轮两者各自处理一个新 token，进入线性层的隐藏向量长度都是 `H`。

将这两行拼起来，某个线性投影可以写成：

<p align="center">$X=\begin{bmatrix}x_A\\x_C\end{bmatrix}\in\mathbb{R}^{2\times H},\qquad Y=XW,\qquad W\in\mathbb{R}^{H\times D}$</p>

两行使用同一份权重，各自得到长度为 `D` 的输出。对 QKV 投影、Attention 输出投影和 FFN 中的线性层，都可以利用这种合批方式。

Attention 则需要保留请求之间的边界。以一层、一个 head 为例，A 的当前 Query 查询 A 自己的缓存，C 的 Query 查询 C 自己的缓存：

<p align="center">$o_A=\operatorname{softmax}\left(\frac{q_AK_A^T}{\sqrt{d_k}}\right)V_A,\qquad o_C=\operatorname{softmax}\left(\frac{q_CK_C^T}{\sqrt{d_k}}\right)V_C$</p>

<p align="center">
  <img src="/img/in-post/ai-infra-continuous-batching-state.svg" alt="A 和 C 的当前隐藏向量合批，使用共享投影权重，随后 A 的 Query 只查询 A 的 KV Cache，C 的 Query 只查询 C 的 KV Cache，两者缓存长度可以不同" style="max-width: 100%;">
</p>

两份缓存的长度不必相等。推理引擎通过每条序列的长度、位置索引、缓存地址或块表，以及相应的 Attention 实现来处理这些差别。各请求的位置编码也使用自己的位置，不能用它在 batch 中的行号代替。

因此，请求可以从上一轮 batch 的第 3 行移动到下一轮的第 1 行，只要引擎始终把这一行关联到正确的请求状态。**合批共享的是模型计算，不是把不同用户的上下文连成一条序列。**

暂时未被选中的请求，其 KV Cache 通常可以继续留在显存中。普通的 batch 成员调整不需要把整份缓存搬到 CPU 再搬回来；只有显存紧张时，某些引擎才可能采用抢占、换出或之后重计算等策略。

## 5. 合批提高吞吐，仍然需要权衡延迟

连续批处理首先减少了等待补位造成的容量浪费。其次，维持足够的有效 batch 大小，可以让多个 token 更充分地复用模型权重。

对一个线性层，单请求 Decode 与 `B` 个请求合批的形状分别是：

```text
单请求：[1, H] × [H, D] → [1, D]
合批：  [B, H] × [H, D] → [B, D]
```

权重矩阵没有因为请求变多而扩大。GPU 在计算中加载的权重块，可以服务多行输入，减少平均到每个 token 上的权重读取成本。对于低 batch 下受显存带宽限制的 Decode，这种复用很有价值。

但计算量会随 batch 增大，各请求的 KV Cache 通常也要分别读取。上下文越长，Attention 的计算和访存负担越大。所以 batch 不能无限加大，单轮耗时也不会保持不变。

可以用一组假设数据区分吞吐和延迟：一个请求每轮耗时 10 ms，每轮产生 1 个 token，整体约为 100 token/s；8 个请求合批后，如果每轮耗时 20 ms，每轮产生 8 个 token，整体约为 400 token/s，但每个请求约 20 ms 才得到下一个 token。这只是计算示例，不是硬件实测。

**系统每秒输出更多 token，和单个用户看到 token 的间隔更短，是两个指标。** 连续批处理可能缩短新请求的排队时间，但不保证每个请求的出字速度都变快。调度器需要在总吞吐、首 token 延迟、后续 token 间隔和公平性之间做取舍。

## 6. 新请求要先经过 Prefill

前面的时间线省略了 Prefill。真实的新请求到来时，通常还没有自己的 KV Cache，不能直接当成一个 Decode token 塞进下一轮。

调度器需要先安排它的 prompt 计算。Prefill 完成后，请求才具备后续增量 Decode 所需的状态。如果一次处理一个特别长的 prompt，占用 GPU 的时间很长，已有请求的下一次 Decode 就可能被推迟，用户会感觉输出停顿。

Chunked Prefill 将长 prompt 分成多个块，让调度器能够把这些块和正在进行的 Decode 工作交错安排。支持混合批次的引擎，还可以在同一轮中处理已有请求的 Decode token 与新请求的一段 prompt。

<p align="center">
  <img src="/img/in-post/ai-infra-continuous-batching-prefill.svg" alt="A 和 B 持续 Decode，C 在第二轮和第三轮分块完成 Prefill，第四轮加入 Decode；同一轮可包含 Decode token 与 Prefill 块，轮次不表示等长时间" style="max-width: 100%;">
</p>

图中 C 在完成最后一块 Prefill 后，可以从最后一个 prompt 位置的 logits 得到首个输出 token；下一轮再处理这个 token，进入 Decode。具体怎样混合、每次切多大的块，取决于引擎和调度策略。

这也解释了为什么实际调度不能只数“有几个请求”：一个普通 Decode 请求本轮通常处理 1 个 token，而一个 Prefill 块可能包含很多 token。服务通常还会限制每轮处理的 token 总数，并检查 KV Cache 的可用空间。即使 token 数相同，上下文长度不同也会带来不同开销。

Chunked Prefill 提供了更细的调度机会，但没有消除 prompt 本身的计算，也不保证新请求的首 token 一定更快。

## 7. 调度与缓存管理配合起来

请求不断进入、缓存不断增长、请求完成后又释放空间，要求推理引擎能灵活管理 KV Cache。PagedAttention 用分页方式组织缓存，让各请求不必预占一整块足够容纳最长序列的连续空间，也便于按块分配和回收。

| 技术 | 负责的事情 |
| --- | --- |
| Continuous Batching | 决定这一轮让哪些请求参与计算 |
| PagedAttention | 组织和访问 KV Cache，支持按块分配与回收 |
| Chunked Prefill | 将长 prompt 的计算切成更小的可调度工作 |

它们经常一起出现在现代推理引擎中，但不是同一个概念。Continuous Batching 并不以 PagedAttention 为必要条件，也可以搭配其他缓存管理方式。分页管理只是让高并发、频繁进出的请求更容易高效运行。

沿着操作系统调度的思路看，一个生成请求始终带着自己的执行状态，调度器在每轮边界决定让谁继续推进；沿着 GPU 计算的思路看，被选中的请求又能组成一个 batch，共同复用模型权重。把这两部分接起来，就能理解连续批处理的收益来自哪里，也能理解它为什么同时受计算、显存和延迟目标约束。

## 参考资料

- [AIInfraGuide：从 Transformer 到 LLM 自回归生成深入理解](https://caomaolufei.github.io/AIInfraGuide/guides/模块一-前置知识/transformer/38-从transformer到llm自回归生成深入理解/#54-continuous-batching)
- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [vLLM：Chunked Prefill](https://docs.vllm.ai/en/latest/configuration/optimization/#chunked-prefill)
