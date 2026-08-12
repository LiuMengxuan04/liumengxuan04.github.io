---
layout:     post
title:      "Softmax、交叉熵与 KL 散度"
subtitle:   "从 Logits、LogSumExp 到语言模型训练与采样"
date:       2026-08-12 20:00:00 +0800
author:     "Liu Mengxuan"
mathjax:    true
header-img: "img/post-bg-miui6.jpg"
categories: [技术]
tags:       [技术, AI Infra, Softmax, 交叉熵, KL散度, LogSumExp, LLM]
---

Softmax、交叉熵和 KL 散度并不是三组彼此独立的公式，而是一条概率建模链路：模型先输出 logits，Softmax 将其归一化为概率分布，交叉熵衡量模型预测与目标之间的差异，KL 散度则进一步把这种差异解释为两个完整概率分布之间的不匹配。

在语言模型中，这条链路同时连接训练和推理：

```text
训练：
logits + 正确 token
        ↓
Cross Entropy
        ↓
标量 loss
        ↓
反向传播

推理：
logits
  ↓ temperature / Top-k / Top-p
候选 logits
  ↓ Softmax
概率分布
  ↓ sampling
下一个 token
```

## 1. Logits 为什么不是概率？

假设词表中暂时只有三个候选 token：

```text
苹果
香蕉
汽车
```

模型最后一层输出：

<p align="center">$z=[2,1,0]$</p>

这三个数字叫 logits，表示模型给不同 token 的原始分数：

```text
苹果：2
香蕉：1
汽车：0
```

logits 可以是任意实数：可以为负，不要求落在 `[0,1]`，总和也不需要等于 1。因此 logits 不能直接解释成概率。

对于完整语言模型，输出 logits 的 shape 通常是：

<p align="center">$Z\in\mathbb{R}^{B\times S\times V}$</p>

也就是每个 batch、每个 token 位置，都有 `V` 个词表分数。

## 2. Softmax：把相对分数变成概率分布

Softmax 定义为：

<p align="center">$p_i=\operatorname{softmax}(z)_i=\dfrac{e^{z_i}}{\sum_j e^{z_j}}$</p>

对：

<p align="center">$z=[2,1,0]$</p>

先计算指数：

<p align="center">$[e^2,e^1,e^0]\approx[7.39,2.72,1]$</p>

再除以总和：

<p align="center">$7.39+2.72+1=11.11$</p>

得到：

<p align="center">$p\approx[0.665,0.245,0.090]$</p>

因此：

```text
苹果：66.5%
香蕉：24.5%
汽车： 9.0%
```

Softmax 用指数把所有项变成正数，再用总和进行归一化，所以：

<p align="center">$p_i>0,\qquad\sum_i p_i=1$</p>

Softmax 一般沿词表维 `V` 执行，shape 不变：

<p align="center">$[B,S,V]\xrightarrow{\text{Softmax over }V}[B,S,V]$</p>

每个 `(batch, token)` 位置分别得到一套词表概率分布。

<p align="center">
  <img src="/img/in-post/ai-infra-softmax-cross-entropy-flow.png" alt="从 logits 经 Softmax 到训练交叉熵和推理解码" style="max-width: 100%;">
</p>

## 3. 为什么 Softmax 要减去最大值？

如果 logits 是：

<p align="center">$z=[1002,1001,1000]$</p>

直接计算 `exp(1002)` 很可能超过浮点数的表示范围，产生 `inf`。

Softmax 对所有 logits 同时平移不敏感：

<p align="center">$\operatorname{softmax}(z+c)=\operatorname{softmax}(z)$</p>

原因是：

<p align="center">$\dfrac{e^{z_i+c}}{\sum_j e^{z_j+c}}=\dfrac{e^ce^{z_i}}{e^c\sum_j e^{z_j}}=\dfrac{e^{z_i}}{\sum_j e^{z_j}}$</p>

因此令：

<p align="center">$m=\max_j z_j$</p>

并计算：

<p align="center">$p_i=\dfrac{e^{z_i-m}}{\sum_j e^{z_j-m}}$</p>

上面的 logits 减去 `m=1002` 后变成：

<p align="center">$[0,-1,-2]$</p>

现在最大的指数值只有 `exp(0)=1`，其他项都小于 1，可以避免正向溢出。

很小的项仍可能下溢成 0。如果后续需要对概率取对数，就不应该先显式计算 Softmax，而应直接使用稳定的 `log_softmax`。

## 4. LogSumExp 为什么会出现？

LogSumExp 定义为：

<p align="center">$\operatorname{LSE}(z)=\log\sum_j e^{z_j}$</p>

它不是突然引入的新概念，而是 Softmax 的归一化分母取对数后的结果。

从 Softmax 开始：

<p align="center">$\operatorname{softmax}(z)_i=\dfrac{e^{z_i}}{\sum_j e^{z_j}}$</p>

两边取对数：

<p align="center">$\log\operatorname{softmax}(z)_i=\log\left(\dfrac{e^{z_i}}{\sum_j e^{z_j}}\right)$</p>

使用：

<p align="center">$\log(a/b)=\log a-\log b$</p>

得到：

<p align="center">$\log\operatorname{softmax}(z)_i=\log e^{z_i}-\log\sum_j e^{z_j}$</p>

因为：

<p align="center">$\log e^{z_i}=z_i$</p>

所以：

<p align="center">$\boxed{\log\operatorname{softmax}(z)_i=z_i-\operatorname{LSE}(z)}$</p>

## 5. LogSumExp 的稳定形式是怎么推出来的？

直接计算：

<p align="center">$\operatorname{LSE}(z)=\log\sum_j e^{z_j}$</p>

仍然可能在指数步骤溢出。令：

<p align="center">$m=\max_j z_j$</p>

将每个 `z_j` 写成：

<p align="center">$z_j=m+(z_j-m)$</p>

于是：

<p align="center">$\sum_j e^{z_j}=\sum_j e^{m+(z_j-m)}$</p>

根据 `exp(a+b)=exp(a)exp(b)`：

<p align="center">$\sum_j e^{z_j}=e^m\sum_j e^{z_j-m}$</p>

两边取对数：

<p align="center">$\log\sum_j e^{z_j}=\log\left(e^m\sum_j e^{z_j-m}\right)$</p>

根据 `log(ab)=log a+log b`：

<p align="center">$\log\sum_j e^{z_j}=\log e^m+\log\sum_j e^{z_j-m}$</p>

最终得到：

<p align="center">$\boxed{\operatorname{LSE}(z)=m+\log\sum_j e^{z_j-m}}$</p>

这不是近似，而是完全相等的代数变形。

对于：

<p align="center">$z=[1002,1001,1000]$</p>

有：

<p align="center">$z-m=[0,-1,-2]$</p>

因此：

<p align="center">$\operatorname{LSE}(z)=1002+\log(1+e^{-1}+e^{-2})\approx1002.408$</p>

所有真正参与指数计算的输入都不大于 0，从而避免巨大中间值。

## 6. LogSumExp 怎样直接得到交叉熵？

由：

<p align="center">$\log\operatorname{softmax}(z)_i=z_i-\operatorname{LSE}(z)$</p>

若正确类别为 `y`，one-hot 标签下的交叉熵为：

<p align="center">$L=-\log p_y$</p>

代入 `log softmax`：

<p align="center">$L=-\left(z_y-\operatorname{LSE}(z)\right)$</p>

即：

<p align="center">$\boxed{L=\operatorname{LSE}(z)-z_y}$</p>

对 `z=[1002,1001,1000]`，如果正确类别是第二类：

<p align="center">$L=1002.408-1001=1.408$</p>

因此训练时不必先计算一个可能下溢为 0 的概率，再执行 `log(0)`。框架通常融合 `log_softmax + NLLLoss`，直接从 logits 稳定地计算交叉熵，同时减少中间张量和访存。

在 PyTorch 中通常直接使用：

```python
loss = torch.nn.functional.cross_entropy(logits, target)
```

而不是手工编写：

```python
probabilities = softmax(logits)
loss = -log(probabilities[target])
```

## 7. 交叉熵到底在评价什么？

设真实分布为 `q`，模型预测分布为 `p`，交叉熵定义为：

<p align="center">$H(q,p)=-\sum_i q_i\log p_i$</p>

它可以理解为：

> 如果数据实际遵循 q，而我们使用模型分布 p 来描述或编码这些结果，平均需要付出多大代价？

普通单标签分类中，真实标签通常被表示为 one-hot。假设正确类别是“香蕉”：

<p align="center">$q=[0,1,0]$</p>

模型预测为：

<p align="center">$p=[0.665,0.245,0.090]$</p>

那么：

<p align="center">$H(q,p)=-\left(0\log0.665+1\log0.245+0\log0.090\right)$</p>

只剩：

<p align="center">$H(q,p)=-\log0.245\approx1.41$</p>

因此 one-hot 标签下，交叉熵就是正确类别概率的负对数：

```text
正确类别概率越高 → loss 越小
正确类别概率越低 → loss 越大
自信地预测错误    → 惩罚非常大
```

## 8. 为什么用负对数？

负对数对低概率有越来越强的惩罚：

| 正确类别概率 | 负对数损失 |
| ---: | ---: |
| 0.99 | 0.01 |
| 0.90 | 0.11 |
| 0.50 | 0.69 |
| 0.10 | 2.30 |
| 0.01 | 4.61 |
| 0.001 | 6.91 |

此外，序列概率通常是多个条件概率的乘积：

<p align="center">$p(y_1,\ldots,y_T)=\prod_{t=1}^{T}p(y_t\mid y_{<t})$</p>

取负对数后，乘积变成求和：

<p align="center">$-\log p(y_1,\ldots,y_T)=-\sum_{t=1}^{T}\log p(y_t\mid y_{<t})$</p>

这既方便优化，也避免大量小概率相乘造成下溢。

## 9. 信息熵：真实分布本身有多不确定？

一个分布自身的信息熵是：

<p align="center">$H(q)=-\sum_i q_i\log q_i$</p>

如果：

<p align="center">$q=[1,0,0]$</p>

结果完全确定，熵为 0。

如果三个类别均匀分布：

<p align="center">$q=[1/3,1/3,1/3]$</p>

不确定性最大，熵为：

<p align="center">$H(q)=\log3$</p>

所以：

```text
分布越集中 → 熵越低 → 越确定
分布越均匀 → 熵越高 → 越不确定
```

## 10. KL 散度是干什么的？

KL 散度比较两个完整概率分布：

<p align="center">$D_{KL}(q\lVert p)=\sum_i q_i\log\dfrac{q_i}{p_i}$</p>

这里统一使用：

- `q`：真实分布或目标分布；
- `p`：模型预测分布。

考虑一个三分类问题。真实数据生成过程中的分布为：

<p align="center">$q=[0.7,0.2,0.1]$</p>

模型预测为：

<p align="center">$p=[0.4,0.4,0.2]$</p>

模型低估了第一个类别，同时高估了后两个类别。KL 散度衡量的是：

> 模型使用 p 近似真实分布 q 时，由于分布不匹配而多付出的平均信息代价。

代入公式：

<p align="center">$D_{KL}(q\lVert p)=0.7\log\dfrac{0.7}{0.4}+0.2\log\dfrac{0.2}{0.4}+0.1\log\dfrac{0.1}{0.2}$</p>

使用自然对数，约为：

<p align="center">$D_{KL}(q\lVert p)\approx0.184$</p>

KL 散度满足：

<p align="center">$D_{KL}(q\lVert p)\ge0$</p>

当 `p=q` 时，KL 为 0；模型分布与真实分布越不一致，KL 通常越大。

<p align="center">
  <img src="/img/in-post/ai-infra-cross-entropy-kl-real-vs-model.png" alt="真实分布与模型预测分布之间的交叉熵和 KL 散度" style="max-width: 100%;">
</p>

### 10.1 “真实分布”在实际训练中从哪里来？

严格来说，真实分布 `q` 是数据生成过程的条件分布。对同一个上下文，理论上可能存在多个合理的下一个 token，并且各自有不同概率。

但普通监督数据通常只观察到一个正确 token。因此训练时常用 one-hot 标签作为这一次观测的经验目标：

<p align="center">$q_y=1,\qquad q_{i\ne y}=0$</p>

这不表示现实世界真的只有一个可能答案，而是表示当前训练样本只提供了一个观测结果。

在 label smoothing、软标签或能够获得目标概率的任务中，`q` 可以是非 one-hot 的完整分布。

## 11. 为什么 KL 散度不是距离？

KL 散度通常不对称：

<p align="center">$D_{KL}(q\lVert p)\ne D_{KL}(p\lVert q)$</p>

`D_KL(q || p)` 以真实分布 `q` 为权重，重点询问：

> q 认为重要的区域，p 是否给了足够概率？

如果某个事件满足 `q_i>0`，但模型给出 `p_i=0`，那么：

<p align="center">$q_i\log\dfrac{q_i}{p_i}\rightarrow+\infty$</p>

也就是说，真实可能发生的事件被模型判为绝不可能，会受到无限大的理论惩罚。

反过来的 `D_KL(p || q)` 使用 `p` 作为权重，关注点不同。因此 KL 没有对称性，也不满足严格距离所需的全部性质。

## 12. 交叉熵与 KL 的关系

交叉熵可以拆解为：

<p align="center">$H(q,p)=H(q)+D_{KL}(q\lVert p)$</p>

推导很直接：

<p align="center">$H(q)+D_{KL}(q\lVert p)$</p>

<p align="center">$=-\sum_iq_i\log q_i+\sum_iq_i\log\dfrac{q_i}{p_i}$</p>

把比值的对数展开：

<p align="center">$\log\dfrac{q_i}{p_i}=\log q_i-\log p_i$</p>

因此：

<p align="center">$H(q)+D_{KL}(q\lVert p)$</p>

<p align="center">$=-\sum_iq_i\log q_i+\sum_iq_i\log q_i-\sum_iq_i\log p_i$</p>

前两项抵消，得到：

<p align="center">$H(q)+D_{KL}(q\lVert p)=-\sum_iq_i\log p_i=H(q,p)$</p>

训练时真实分布 `q` 固定，因此 `H(q)` 不随模型参数变化。所以：

> 最小化交叉熵，等价于最小化模型分布 p 与真实分布 q 之间的 `D_KL(q || p)`。

对于 one-hot 标签，`H(q)=0`，交叉熵和对应的 KL 在数值上相同。

KL 散度还会用于分布匹配、知识蒸馏、RLHF/PPO 约束和变分推断。无论在哪种场景，都必须确认 API 接收的是 logits、概率还是 log 概率，以及 KL 的方向。

## 13. 温度参数控制什么？

带温度的 Softmax 是：

<p align="center">$p_i(T)=\dfrac{e^{z_i/T}}{\sum_j e^{z_j/T}},\qquad T>0$</p>

```text
T < 1：放大 logit 差异，分布更尖锐
T = 1：普通 Softmax
T > 1：缩小 logit 差异，分布更平坦
```

温度只改变当前 logits 对应的采样分布，不会修改模型权重。实现时应先缩放 logits，再应用稳定 Softmax 或执行过滤采样。

## 14. Perplexity 是什么？

如果平均 token 负对数似然为 `L_bar`，困惑度定义为：

<p align="center">$\operatorname{PPL}=e^{\bar L}$</p>

例如：

<p align="center">$\bar L=\log10\quad\Rightarrow\quad\operatorname{PPL}=10$</p>

可以粗略理解成模型每一步仿佛在 10 个有效候选之间做选择。

PPL 越低通常表示模型对评测文本越有把握，但 tokenizer、测试数据、上下文长度、特殊 token 和 loss 归约方式都会影响结果，因此不能脱离评测设置直接横向比较。

## 15. Top-k 与 Top-p 怎样选择下一个 token？

模型得到词表分布后，还需要通过解码策略选择下一个 token。

### Greedy

直接选择概率最大的 token。

### Top-k

只保留概率最高的 `k` 个 token，在这些候选中重新归一化并采样。

### Top-p

选择累计概率至少达到 `p` 的最小候选集合。

例如 `p=0.9`：

```text
token A：0.50  累计 0.50
token B：0.25  累计 0.75
token C：0.12  累计 0.87
token D：0.05  累计 0.92  ← 到这里停止
```

然后只在 A、B、C、D 中重新归一化并采样。

Top-k 的候选数量固定，Top-p 的候选数量会随着分布的尖锐程度动态变化。工程实现通常先在 logits 上过滤，把排除项设为负无穷，再执行稳定 Softmax。

## 16. 把整条链路串起来

训练阶段：

```text
hidden states [B,S,H]
        ↓ 输出投影
logits [B,S,V]
        ↓ 融合 log_softmax + NLL
token losses [B,S]
        ↓ 对有效 token 归约
标量 loss
        ↓
反向传播
```

推理阶段：

```text
最后位置 logits [B,V]
        ↓ temperature
缩放 logits
        ↓ Top-k / Top-p
过滤候选
        ↓ stable Softmax
概率分布
        ↓ sampling
下一个 token
```

比较分布时：

```text
真实分布 q
模型分布 p
      ↓
D_KL(q || p)
      ↓
衡量模型没有复现真实分布所造成的额外代价
```

最后可以把三个核心概念压缩成三句话：

> Softmax 把模型的任意实数打分转换成概率分布。

> 交叉熵衡量使用模型分布描述真实目标需要付出的总代价。

> KL 散度衡量模型预测分布相对于真实分布额外造成了多少分布偏差，而且方向不能随意交换。
