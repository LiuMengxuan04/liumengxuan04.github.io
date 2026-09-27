---
layout:     post
title:      "vLLM 学习笔记 02：nano-vllm 源码逐段精读"
subtitle:   "用 1400 行 Python 看清一个推理引擎的全部细节"
date:       2026-09-27 23:00:00 +0800
author:     "Liu Mengxuan"
mathjax:    true
header-img: "img/post-bg-miui6.jpg"
categories: [技术]
tags:       [技术, vLLM, nano-vllm, 大模型推理, KV Cache, PagedAttention, 源码阅读, 学习笔记]
---

> **关于本文**：[上一篇笔记](/技术/2026/09/27/技术-vLLM学习笔记01-一个请求在vLLM里的一生/)从概念上走了一遍 vLLM，这一篇换个方式：找一个足够小、但该有的都有的实现，把源码从头到尾读一遍。选的是 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)（作者 Xingkai Yu，MIT 协议），全部核心代码不到 1400 行，却实现了分页 KV 缓存、前缀缓存、分块预填充、抢占、张量并行、CUDA Graph 和 torch.compile。
>
> 本文对照的版本是 main 分支 commit `bb823b3`（“Merge PR #218 chunked-prefill-refactor”，包版本 0.2.0）。文中的源码片段都摘自这个版本，版权归原作者，按 MIT 协议引用；解析文字、配图和实验都是我自己写的、画的、跑的。实验环境是一张 RTX 3090 24G，模型 Qwen3-0.6B。

## 0. 先看全貌

### 0.1 nano-vllm 做了什么、没做什么

nano-vllm 是一个**离线推理**引擎：你给它一批 prompt，它把它们全部生成完再一起返回。它没有 HTTP 服务、没有流式输出、没有 top-p/top-k，也只支持 Qwen3 这一种模型结构。但推理引擎里真正难、真正值得学的那部分，它一个没少：

- **分页 KV 缓存**：KV 按固定大小的块存放，一个请求的块在显存里可以不连续；
- **前缀缓存**：内容相同的块用哈希识别出来，多个请求直接共享，不重复计算；
- **分块预填充**：一个特别长的 prompt 可以分几步喂进模型；
- **抢占**：显存不够时把某个请求踢回等待队列，释放它的块；
- **张量并行**：最多 8 张卡切分同一个模型；
- **CUDA Graph + torch.compile**：压掉解码阶段的 Python 和 kernel 启动开销。

### 0.2 代码地图

![图 1](/img/in-post/vllm-notes-02/fig1-code-map.svg)

从上往下三层：

1. **接口层**：`LLM` 就是 `LLMEngine` 的别名，配合 `Config` 和 `SamplingParams` 两个数据类；
2. **调度层**：`Scheduler` 决定每一步算谁、算多少，它手下的 `BlockManager` 管 KV 块，`Sequence` 记录每个请求的状态；
3. **执行层**：`ModelRunner` 把调度结果变成张量，跑 `Qwen3ForCausalLM`，模型又由 `layers/` 下的各个算子拼起来。

调度层和执行层之间还有一条“暗线”：`utils/context.py` 里的全局 `Context`。`ModelRunner` 在前向之前把本批次的元数据（每个序列多长、KV 写到哪儿、块表是什么）塞进去，注意力层在最深处再把它取出来。这样模型的 `forward` 签名就能保持 `(input_ids, positions)` 这样干净的形式。

### 0.3 跑起来是什么样子

仓库里的 `example.py` 就是最小用法：

```python
path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
tokenizer = AutoTokenizer.from_pretrained(path)
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = [
    "introduce yourself",
    "list all prime numbers within 100",
]
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    for prompt in prompts
]
outputs = llm.generate(prompts, sampling_params)
```

几个要注意的地方：

- 构造 `LLM` 时传的是**本地目录**，不是 Hugging Face 上的模型名，`Config` 里会断言它是个目录；
- `enforce_eager=True` 表示不用 CUDA Graph，启动快一些，适合调试；
- 聊天模型要自己套 chat template，`generate` 只认字符串或者 token id 列表；
- 返回值是一个列表，每项是 `{"text": ..., "token_ids": ...}`，只包含**生成出来**的部分，不含 prompt。

下面按“请求从进来到出去”的顺序读代码：先读数据结构（Config、Sequence），再读调度（BlockManager、Scheduler），然后是把它们串起来的 LLMEngine，最后深入执行层。

## 1. 配置：Config 与 SamplingParams

### 1.1 Config

```python
@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
```

`slots=True` 让这个数据类用 `__slots__` 存字段，实例不能再随手加新属性，访问也稍快。各字段的含义：

| 字段 | 默认值 | 作用 |
| --- | --- | --- |
| `max_num_batched_tokens` | 16384 | 一步预填充最多处理多少个 token，是分块预填充的“预算” |
| `max_num_seqs` | 512 | 一步最多处理多少个序列，也决定 CUDA Graph 最大录多大的批 |
| `max_model_len` | 4096 | 单个序列允许的最大长度，会和模型的 `max_position_embeddings` 取较小值 |
| `gpu_memory_utilization` | 0.9 | 整张卡最多用到 90% 显存，剩下的全部给 KV 缓存 |
| `tensor_parallel_size` | 1 | 张量并行卡数，断言在 1 到 8 之间 |
| `enforce_eager` | False | True 时不录 CUDA Graph |
| `hf_config` | None | 在 `__post_init__` 里从模型目录读出的 Hugging Face 配置 |
| `eos` | -1 | 结束符 id，引擎初始化时从 tokenizer 填进来 |
| `kvcache_block_size` | 256 | 每个 KV 块装多少个 token |
| `num_kvcache_blocks` | -1 | KV 块总数，由 `ModelRunner` 测完显存后**回填** |

有两个“先占位、后回填”的字段：`eos` 和 `num_kvcache_blocks`。它们在构造时不知道，等模型加载、显存测完之后才写进同一个 `config` 对象，再交给 `Scheduler` 使用。所以 `LLMEngine` 里 `ModelRunner` 必须先于 `Scheduler` 构造，这个顺序是有意为之的。

`kvcache_block_size % 256 == 0` 这条断言来自 flash-attn：它的分页 KV 接口要求页大小（`page_block_size`）是 256 的倍数。vLLM 默认块大小是 16，nano-vllm 因为直接复用 flash-attn 的分页注意力，只能用 256 起步的大块。块越大，前缀缓存的粒度越粗（不满 256 个 token 的前缀没法共享），最后一个块的浪费也越多，这是简化带来的代价。

Qwen3-0.6B 的 `max_position_embeddings` 是 40960，所以 `max_model_len` 最后还是 4096。

### 1.2 SamplingParams

```python
@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
```

采样参数只有三个：温度、最多生成多少 token、是否无视结束符（压测时常用，保证每个请求都生成满 `max_tokens`）。

温度必须大于 0，不支持贪心解码。原因在采样器里：logits 要除以温度，温度为 0 就是除零。想要接近贪心的效果，可以把温度设得很小，比如 `1e-5`，这时 softmax 后几乎所有概率都集中在最大的那个 token 上。

## 2. Sequence：一个请求的全部状态

`Sequence` 是整个引擎里被传来传去最多的对象。调度器靠它判断状态，块管理器往它身上挂块表，模型执行器从它身上取 token。

### 2.1 字段

```python
class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
```

逐个看：

- `block_size` 和 `counter` 是**类属性**，所有序列共享。`counter = count()` 是一个从 0 开始的无限计数器，每 `next` 一次得到下一个整数，所以 `seq_id` 在整个进程里全局递增、不会重复。一个小细节：`ModelRunner` 预热时也会创建假序列，它们会先吃掉几个 id。默认配置下预热创建 4 个假序列，所以你的第一个真实请求 id 是 4，不是 0。
- `status`：三种状态，新请求都是 `WAITING`。
- `token_ids = copy(token_ids)`：浅拷贝一份，之后往里追加生成的 token 不会改动调用方传进来的列表。
- `last_token`：最后一个 token。解码时模型只需要这一个作为输入，单独存一份方便跨进程传输（后面讲序列化时会看到）。
- `num_tokens`：当前总长度 = prompt 长度 + 已生成长度。
- `num_prompt_tokens`：prompt 长度，固定不变，用来切分 prompt 和生成部分。
- `num_cached_tokens`：**已经算过、KV 已经在缓存里的 token 数**。这是整个调度逻辑里最关键的计数器，它既包括前缀缓存命中的部分，也包括之前几步已经算完的部分。
- `num_scheduled_tokens`：**这一步要算的 token 数**。调度器填，后处理时清零。
- `is_prefill`：这个序列是否还处在预填充阶段。一开始是 True，第一次被安排解码时改成 False；被抢占后又改回 True。
- `block_table`：块表，第 i 项是第 i 个逻辑块对应的物理块号。
- 最后三个是从 `SamplingParams` 里拆出来的，直接存在序列上，省得每次去找。

默认参数 `sampling_params = SamplingParams()` 是 Python 里经典的“可变默认参数”写法：这个对象只在定义函数时创建一次，所有不传参数的调用共享它。这里无害，因为代码只读取它的字段，从不修改。

两个计数器 `num_cached_tokens` 和 `num_scheduled_tokens` 配合起来描述“进度”：

- `[0, num_cached_tokens)`：KV 已经在缓存里；
- `[num_cached_tokens, num_cached_tokens + num_scheduled_tokens)`：这一步要算的；
- 其余：还没轮到。

只要记住这一条，后面调度、准备输入、后处理的代码都能看懂。

### 2.2 属性和辅助方法

```python
    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
```

- `len(seq)` 返回总长度，`seq[a:b]` 直接切 `token_ids`，让序列用起来像个列表；
- `num_blocks` 是向上取整：`(n + bs - 1) // bs`。1500 个 token、块大小 256，需要 6 个块；
- `last_block_num_tokens` 是最后一个块里装了几个 token。1500 个 token 时是 1500 - 5×256 = 220。注意它的值在 1 到 256 之间，块刚好装满时是 256，而不是 0；
- `block(i)` 返回第 i 个块对应的那段 token，算哈希时要用；
- `append_token` 追加一个新生成的 token，同时更新 `last_token` 和 `num_tokens`。注意它**不**动 `num_cached_tokens`：新 token 的 KV 还没算，要等下一步解码才会写进缓存。

### 2.3 自定义序列化

```python
    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
```

`__getstate__` 和 `__setstate__` 控制 pickle 怎么存取这个对象。只有多卡时才会用到：主进程要把每一步的序列列表 pickle 后通过共享内存发给其他卡的进程（见 6.2 节）。

它只挑了子进程**真正需要**的字段：四个计数器、块表，再加上 token。token 部分分两种情况：

- 预填充阶段：发完整的 `token_ids`，因为子进程要从中切出这一步要算的那段；
- 解码阶段：只发 `last_token` 一个整数。解码时模型的输入就只有最后一个 token，没必要把几千个历史 token 每步都发一遍。

反序列化时用 `isinstance(last_state, list)` 区分两种情况。`seq_id`、`status`、温度等字段都没发，子进程也用不到：采样只在主进程做。

还有个坑要提前说：类属性 `block_size` 不在序列化内容里。子进程里的 `Sequence.block_size` 是模块导入时的默认值 256，而 `LLMEngine` 只在主进程里把它改成配置值。第 10 节会用实验说明这个问题。

## 3. BlockManager：块分配、引用计数与前缀缓存

`BlockManager` 只做簿记：哪些物理块空闲、哪些在用、每个块被几个序列引用、每个块里装的是什么内容。它不碰显存，真正的 KV 张量在 `ModelRunner` 里。

### 3.1 Block

```python
class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
```

- `ref_count`：有几个序列的块表里有这个块。大于 1 就说明它被前缀共享了；
- `hash`：块装满之后才会算的内容哈希，-1 表示还没算（块没满，或者刚被重新分配）；
- `token_ids`：块里的 token 内容，和 `hash` 一起存，用来在哈希命中时二次确认，防止哈希碰撞；
- `update`：块装满时调用，记下哈希和内容；
- `reset`：块被重新分配给某个序列时调用。注意 `ref_count` 直接设成 1，因为 reset 的时刻就是“被一个序列领走”的时刻。

### 3.2 BlockManager 的四个容器

```python
class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
```

- `blocks`：所有块对象，下标就是物理块号；
- `hash_to_block_id`：内容哈希到块号的映射，前缀缓存靠它查找；
- `free_block_ids`：空闲块的**双端队列**，初始是 0, 1, 2, …；
- `used_block_ids`：在用块的集合。

这里有个很重要的设计：**空闲不等于内容作废**。一个块被释放后回到 `free_block_ids`，但它的 `hash`、`token_ids` 以及显存里的 KV 数据都原封不动，`hash_to_block_id` 里的映射也还在。只要它还没被别人领走并覆盖，后来的请求依然可以命中它。这就是“懒回收”：只有真正需要一个空块时，才把旧内容作废。

### 3.3 计算哈希

```python
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()
```

用 xxHash 的 64 位版本，快、分布好，不是加密哈希，但这里也不需要。

关键在 `prefix`：第 i 个块的哈希 = H(第 i-1 个块的哈希, 第 i 个块的 token)。于是一个块的哈希值不只由它自己的内容决定，而是**由从开头到它为止的全部内容决定**。两个请求第 3 块内容一样、但前面不一样，它们的哈希也不一样，不会被错误地共享。这很必要，因为 KV 不只取决于 token 本身，还取决于它前面的所有上下文。

实现细节：前一个哈希转成 8 字节小端序；token 列表转成 numpy 数组再取原始字节（默认 int64，每个 token 8 字节）。`intdigest()` 返回一个 Python 整数。

### 3.4 分配和释放单个块

```python
    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)
```

分配：

1. 从空闲队列**头部**取一个块；
2. 断言它没人引用；
3. 如果它带着旧哈希，并且哈希表里这个哈希**仍然指向它自己**，才删掉这条映射。为什么要多判断一次“仍然指向自己”？因为同样内容的块可能不止一个：两个请求同时预填充同一段内容时，它们各自领了块，谁后完成谁就把 `hash_to_block_id[h]` 覆盖成自己。这时旧块的 `hash` 字段还是 h，但映射已经指向别的块了，如果无条件删除，就会误删别人的映射；
4. `reset()` 清空内容、引用计数设为 1；
5. 放进在用集合。

释放：从在用集合移除，追加到空闲队列**尾部**。

头部取、尾部放，空闲队列自然形成了一个近似 LRU 的顺序：最近释放的块排在最后，最晚被覆盖，它们的内容也就最有机会被后来的请求命中。

### 3.5 can_allocate：先看能不能分

```python
    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks
```

这个函数同时回答两个问题：前缀能命中几个块？空闲块够不够？返回值 -1 表示不够，否则返回命中的块数。

逐行看：

- `num_new_blocks` 初始为序列需要的总块数，表示“需要从空闲队列里拿几个块”；
- 循环只到 `num_blocks - 1`，**最后一个块永远不查**。哪怕最后一块刚好装满、而且缓存里有，也不用。原因是模型至少要算一个 token 才能产生 logits、采样出下一个 token；如果整个 prompt 都命中缓存，这一步就没东西可算了。代价是 prompt 长度恰好是 256 整数倍时，最后整整 256 个 token 都要重算；
- 每块用链式哈希查表，查不到，或者查到了但内容对不上（哈希碰撞），立刻 `break`。前缀缓存只认**从头开始连续**命中的块，中间断了后面就不看了；
- 命中后 `num_cached_blocks += 1`。但 `num_new_blocks` 只在命中块**正在被使用**时才减 1。这是个很细的点：如果命中的块此刻在空闲队列里（原主人已经结束了），把它拿回来同样要占用一个空闲名额，所以不能少算；只有命中在用块时才是真正的“白嫖”，只需加个引用计数；
- 最后比较空闲块数和需要的新块数。

### 3.6 allocate：真正分配

```python
    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size
```

- 先断言序列还没有块表，只有从未分配过、或者被抢占清空过的序列才会走到这里；
- 前 `num_cached_blocks` 块：再算一遍链式哈希（和 `can_allocate` 里算的一样，这里没有缓存上次的结果），找到命中的块：
  - 在用：引用计数加一，共享；
  - 空闲：引用计数设为 1，从空闲队列里**摘出来**，放进在用集合。`deque.remove` 是 O(n) 的线性查找，块数上千时有点开销，但相对一次前向微不足道；
  - 注意这里没有调用 `reset()`，块的 `hash` 和 `token_ids` 保留，因为内容本来就是对的。
- 剩下的块全部新分配；
- 最后把 `num_cached_tokens` 设为命中块数乘块大小。这一行就是前缀缓存生效的地方：调度器看到 `num_cached_tokens` 大于 0，就只安排剩余部分去计算。

### 3.7 deallocate：释放整个序列

```python
    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()
```

**倒序**释放，引用计数减到 0 才真正还给空闲队列。倒序是有讲究的：块按“后面的先入队”的顺序追加到空闲队列尾部，于是序列的**开头**块排在队列最末尾，最晚被覆盖。开头的块正是最有可能被别的请求共享的公共前缀（比如系统提示词），保留得越久越好。

最后把 `num_cached_tokens` 清零、块表清空。被抢占的序列之后重新调度时，会从头再查一遍前缀缓存。

### 3.8 解码时追加块

```python
    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())
```

解码时每步给序列加一个 token。什么时候需要新块？看 `len(seq) % block_size == 1`：

`len(seq)` 已经**包含**了这一步要算的那个 token（上一步采样出来后已经 `append_token` 了）。如果长度除以 256 余 1，说明这个 token 是某个新块的第一个，需要一个新块；其他时候它都能放进最后一个块的空位。

`can_append` 的写法很紧凑：右边是个布尔值，在 Python 里 True 就是 1，False 就是 0。于是它的意思是：“需要新块时空闲块数 ≥ 1，不需要时 ≥ 0（永远成立）”。

### 3.9 hash_blocks：块装满后登记哈希

```python
    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
```

这个函数在每步计算完成后由调度器调用（见 4.4 节），把**这一步刚刚装满**的块登记进哈希表。

- `start`：这一步开始前，已经装满的块数（整除，向下取整）；
- `end`：这一步结束后，已经装满的块数；
- 两者相等说明这一步没有填满任何新块，直接返回；
- 前缀哈希从第 `start - 1` 块上取，它在之前某一步已经算好存在块上了，不用从头重算整条链；
- 对 `[start, end)` 的每个块算哈希、存进块里、登记映射。登记时直接覆盖，这就是 3.4 节说的“同一个哈希后来者覆盖”。

举个例子。1500 个 token 的 prompt 分两步预填充，第一步算 1024 个：`start = 0`、`end = 1024 // 256 = 4`，登记 0 到 3 号块。第二步算 476 个：`start = 4`、`end = 1500 // 256 = 5`，登记 4 号块。第 5 块只装了 220 个，不登记。之后解码，每步 `num_scheduled_tokens = 1`，当 `num_cached_tokens + 1` 刚好到 1536 时，`end` 变成 6，第 5 块才被登记。

也就是说，**只有装满的块才会进入前缀缓存**，没满的块内容还在变，不能拿来共享。

图 4 是我实测的一个前缀缓存命中的例子：请求 B 和 C 有 600 个 token 的公共前缀。

![图 4](/img/in-post/vllm-notes-02/fig4-prefix-cache-trace.svg)

B 结束后它的块都回到了空闲队列，但块 6 和块 7 的哈希映射还在。C 来的时候，`can_allocate` 用链式哈希查到了这两块，并且内容比对一致；第 3 块从 token 512 开始就和 B 不同了，查不到，于是新领了块 9。最终 C 只需要为 620 - 512 = 108 个 token 跑模型。

## 4. Scheduler：每一步算谁、算多少

### 4.1 两个队列

```python
class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)
```

- `waiting`：等待预填充的序列，包括新来的、预填充到一半的（分块）、被抢占回来的；
- `running`：预填充已经完成、进入解码的序列；
- 两个队列都空了，整个 `generate` 就结束了；
- `BlockManager` 在这里用 `config.num_kvcache_blocks` 构造，此时这个值已经被 `ModelRunner` 回填好了。

注意 `waiting` 里的序列**已经可能持有块**：一个分块预填充进行到一半的序列，块已经全部分好，只是还没算完，所以它仍然待在 `waiting` 的队首。这就是下面 `schedule` 里 `if not seq.block_table` 这条分支的由来。

### 4.2 schedule：预填充优先

![图 3](/img/in-post/vllm-notes-02/fig3-schedule-flow.svg)

`schedule` 返回 `(本步要算的序列列表, 是否是预填充)`。它分两个阶段：先尽量安排预填充，只要安排上了哪怕一个，就直接返回；一个都安排不上，才去安排解码。

```python
    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True
```

一行一行拆开：

1. **循环条件**：等待队列不空，且本批序列数没到上限。
2. **只看队首** `self.waiting[0]`，不 pop。严格先来先服务，队首安排不上，后面的也不看（不会“插队”）。
3. **剩余预算** `remaining`：这一步还能塞多少 token。用完了就停。
4. **算这个序列这一步需要多少 token** `num_tokens`，分两种情况：
   - 没有块表（新序列或被抢占过的）：先问 `can_allocate`。返回 -1 说明显存不够，停止安排预填充。否则需要的 token 数 = 总长 - 命中块数 × 256。这时还没真正分配，`seq.num_cached_tokens` 还是 0，所以用 `num_cached_blocks * block_size` 算；
   - 已有块表（分块预填充的后续块）：需要的 = 总长 - 已缓存。
5. **分块的限制**：`remaining < num_tokens and scheduled_seqs`。如果剩余预算装不下整个序列，而本批**已经有别的序列了**，就停。换句话说，**只有本批的第一个序列允许被切块**。这样一批里最多只有一个“半截”序列，逻辑简单。
6. **真正分配块**：只有新序列才需要。注意分配发生在第 5 步的检查之后，避免分了块又不安排。
7. **这一步算多少** `num_scheduled_tokens = min(num_tokens, remaining)`。第一个序列放不下时就只算 `remaining` 个，这就是分块预填充。
8. **是否已经“追上”**：如果已缓存 + 本步要算的 = 总长，说明这一步做完预填充就完成了，把它状态改成 `RUNNING`，从 `waiting` 弹出，放到 `running` 尾部。没追上的（分块中）继续留在 `waiting` 队首，下一步接着算。
9. 加入本批。

有个容易忽略的点：序列在**计算之前**就被移进 `running` 了，但此时它的 `num_cached_tokens` 还没更新。更新发生在计算之后的 `postprocess` 里。

还有一个设计选择要注意：**预填充和解码从不混在一批里**。只要等待队列里有能安排的，这一步就全是预填充，正在解码的序列全部暂停一步。vLLM V1 会把预填充块和解码 token 拼进同一批，nano-vllm 为了简单没有这么做。代价是有新请求源源不断进来时，解码会被反复打断。

### 4.3 schedule：解码与抢占

```python
        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
```

这段用了 Python 里不太常见的 **`while ... else`**：`else` 分支只在 `while` 条件变成假、**正常退出**时执行；如果是 `break` 跳出来的，就不执行。

逐句拆解：

1. 从 `running` 队首取一个序列。
2. 内层 `while`：只要它需要新块但没有空闲块，就抢占：
   - 运行队列里还有别人：踢掉**队尾**那个（`running.pop()`）。队尾是最晚进入运行队列的，优先牺牲新来的，保护已经跑了很久的；
   - 已经没有别人可踢了：只能把**自己**踢回去，然后 `break`。这时 `else` 不执行，它不进本批。
3. 内层 `while` 正常结束（有块可用）时进 `else`：本步算 1 个 token，标记为解码阶段，按需领一个新块，加入本批。
4. `assert scheduled_seqs`：走到这里一定至少安排了一个。只要有序列在跑，最坏情况也会把其他序列都抢占掉，腾出块给当前这个。
5. `self.running.extendleft(reversed(scheduled_seqs))`：把本批序列**按原来的顺序**放回运行队列的**头部**。`extendleft` 会把元素一个个插到左边，顺序会被反过来，所以先 `reversed` 一次再插，负负得正。这样下一步解码时的顺序和这一步一样。

`preempt` 做四件事：状态改回等待、标记为预填充、释放所有块、插到等待队列**头部**（`appendleft`）。放头部是为了让它下一次最先被安排。它的 `token_ids` 里已经包含了 prompt 和生成到一半的内容，重新预填充时整段一起算。

被抢占的序列并不会从零开始：它释放的块内容还在空闲队列里，满块的哈希也还登记着。重新调度时 `can_allocate` 会命中这些块，只需重算最后那个不满的块。第 10 节有实测。

### 4.4 postprocess：计算完成之后

```python
    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
```

每个序列依次：

1. **登记装满的块**：`hash_blocks` 要用到“更新前”的 `num_cached_tokens` 和本步的 `num_scheduled_tokens`，所以必须在更新计数器之前调用；
2. **推进进度**：已缓存 += 本步算的，本步计数清零；
3. **分块未完成就跳过**：预填充阶段，如果算完这一块还没追上总长，模型对这一块最后一个位置输出的 token 是没有意义的：它预测的是 prompt 里下一个本来就已知的 token。这里直接丢弃采样结果，`continue`；
4. **追加新 token**；
5. **判断结束**：两个条件满足一个就结束：
   - 没设 `ignore_eos`，且采样到了 eos；
   - 生成数量达到 `max_tokens`；
6. 结束时：状态改为完成，释放块（内容留在空闲队列里供后来者复用），从运行队列移除。`deque.remove` 同样是线性查找。

关于 eos 有个细节：`self.eos` 来自 `tokenizer.eos_token_id`，只有一个 id。对 Qwen3-0.6B 来说是 151645（`<|im_end|>`）。而模型目录里的 `generation_config.json` 写的是 `[151645, 151643]` 两个结束符，nano-vllm 没读这个文件，所以采样到 151643（`<|endoftext|>`）时不会停。用 chat template 时模型基本都以 `<|im_end|>` 结尾，影响不大；直接喂裸文本续写时可能会多生成一些。

## 5. LLMEngine：把调度和执行串起来

### 5.1 构造

```python
class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)
```

1. **过滤参数**：用 `dataclasses.fields` 拿到 `Config` 的所有字段名，只把 kwargs 里名字对得上的传进去，其余的**静默丢弃**。好处是兼容 vLLM 风格的调用，传了不认识的参数也不报错；坏处是参数名拼错了你也不会知道。
2. **同步块大小**：把 `Sequence` 的类属性改成配置里的块大小。
3. **启动其他卡的进程**：rank 1 到 tp-1 各起一个进程，进程的入口直接就是 `ModelRunner` 类本身，构造函数就是它的主函数（6.1 节会看到，子进程的构造函数最后会进入一个死循环，直到收到 exit）。每个子进程配一个 `Event`，主进程用它通知子进程“有新任务了”。用 `spawn` 而不是 `fork`，是因为 CUDA 在 fork 出来的子进程里不能正常初始化。
4. **主进程自己当 rank 0**：直接在本进程构造 `ModelRunner`，把所有 Event 的列表传给它。构造过程中完成模型加载、预热、KV 分配、CUDA Graph 录制，并回填 `config.num_kvcache_blocks`。
5. **tokenizer 和 eos**：加载快速 tokenizer，回填 `config.eos`。
6. **最后才建调度器**：前面说过，它需要回填后的块数和 eos。
7. **注册退出钩子**：Python 进程退出时自动调用 `exit`，保证子进程和 NCCL 被正确清理。

```python
    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()
```

`call("exit")` 会同时通知所有子进程退出（见 6.2 节），然后等它们结束。

### 5.2 add_request 与 step

```python
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens
```

![图 2](/img/in-post/vllm-notes-02/fig2-step-sequence.svg)

`add_request` 把字符串编码成 token id（已经是 id 列表就直接用），包成 `Sequence` 放进等待队列。

`step` 就是引擎的一次“心跳”，三步：调度 → 执行 → 后处理。

`num_tokens` 用了一个小技巧：**用正负号区分阶段**。预填充时是本步处理的 token 总数（正数），解码时是 **负的**序列数（每个序列 1 个 token）。`generate` 靠这个符号决定把耗时算进预填充吞吐还是解码吞吐，省得多返回一个值。

`model_runner.call("run", ...)` 在单卡时就是直接调用 `run`；多卡时会先把调用广播给其他进程。

最后从本批里挑出已经结束的序列，返回 `(seq_id, 生成的 token 列表)`。

### 5.3 generate

```python
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
```

- 采样参数可以给一个（所有 prompt 共用），也可以给列表（一一对应）。给一个时用 `[sp] * n` 复制引用，同一个对象被共享，因为只读所以没问题；
- 一次性把所有请求加入等待队列，然后不停 `step` 直到两个队列都空；
- 每步计时，按 `num_tokens` 的符号更新两个吞吐量显示在进度条上。这是**单步**的瞬时吞吐，不是平均值；
- 请求结束的顺序和提交顺序不一定一样（短的先结束），所以先存进字典，最后按 `seq_id` 排序。`seq_id` 是全局递增的，排序后就恢复了提交顺序；
- 最后解码成文本。注意 `decode` 时没有跳过特殊 token，所以文本末尾通常会带着 `<|im_end|>`。

## 6. ModelRunner：从调度结果到 GPU

`ModelRunner` 是执行层的核心，每张卡一个。它负责：初始化分布式环境和模型、测显存并分配 KV 缓存、录制 CUDA Graph、把序列列表变成张量、跑模型、采样，以及多卡时的进程间通信。

### 6.1 构造

```python
class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()
```

1. `event`：rank 0 拿到的是所有子进程 Event 的列表，其他 rank 拿到的是自己的那一个 Event。
2. **初始化进程组**：后端 NCCL，用本机 TCP 端口 2333 做握手。端口是写死的，如果你同时起两个 nano-vllm 实例，或者 2333 被占用，第二个会失败。单卡时也会初始化一个 world_size=1 的进程组，因为各个并行层在构造时都要调用 `dist.get_rank()`。
3. **绑定 GPU**：rank i 用第 i 张卡。
4. **临时改默认 dtype 和 device**：先记下原来的默认 dtype，然后改成模型的 dtype（Qwen3-0.6B 是 bf16），默认设备改成 cuda。这样接下来所有 `torch.empty`、`torch.zeros`、`nn.Parameter(torch.empty(...))` 都会直接在 GPU 上以 bf16 创建，模型不需要先在 CPU 上建好再 `.to("cuda")`，省时间也省内存。
5. 构建模型、加载权重、创建采样器、预热、分配 KV 缓存、录 CUDA Graph（按顺序，后面逐个讲）。
6. **恢复默认 dtype 和 device**：之后 `prepare_*` 里创建的 CPU 张量不会误跑到 GPU 上。
7. **多卡时建共享内存**：rank 0 创建一块名为 `nanovllm` 的 1 MiB 共享内存，其他 rank 等 rank 0 建好（用 `barrier` 同步）后再去连接。然后子进程调用 `self.loop()`，**永远不会从构造函数返回**，直到收到 exit。这就是为什么 `Process(target=ModelRunner, ...)` 能工作：构造函数本身就是子进程的主循环。

### 6.2 多卡时的“远程调用”

![图 7](/img/in-post/vllm-notes-02/fig7-shm-rpc.svg)

```python
    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)
```

这是一个极简的 RPC：

- **`call`** 是统一入口。rank 0 调用时，先把“方法名 + 参数”写进共享内存广播出去，然后自己也执行同一个方法。单卡时就是普通的方法调用。
- **`write_shm`**（只在 rank 0）：pickle 成字节串，前 4 字节写长度（小端），后面写数据，然后把每个子进程的 Event 置位，叫醒它们。
- **`read_shm`**（只在其他 rank）：阻塞在 `event.wait()`；被叫醒后先读长度、再读数据、反序列化，最后 `event.clear()` 把 Event 复位，为下一次等待做准备。
- **`loop`**：子进程的主循环，读一个调用、执行一个，是 exit 就退出。

这个协议的正确性靠的是一个隐含的同步：rank 0 写完共享内存后自己也去执行 `run`，而 `run` 里的前向计算会碰到 NCCL 的 `all_reduce`/`gather`，必须等所有卡都到达才能继续。所以 rank 0 不可能在子进程读完这一次的数据之前，就开始写下一次的数据覆盖它。

几个限制：

- 数据只能是单向的，子进程的返回值被丢弃，也用不着：采样只在 rank 0 做；
- 共享内存只有 1 MiB。预填充时会发完整的 token 列表，一批 512 个序列、每个几千 token，pickle 后可能超过 1 MiB，那样写入就会越界报错；
- `exit` 里先关共享内存，所有 rank `barrier` 一下，确认大家都关了，再由 rank 0 `unlink` 真正删除。然后释放 CUDA Graph，等 GPU 空闲，销毁进程组。

### 6.3 预热

```python
    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()
```

预热有两个目的：一是触发 torch.compile 的编译、Triton 核的 JIT，让第一个真实请求不用等编译；二是**测出前向计算最多会用多少临时显存**，下一步分配 KV 缓存要用到这个数。

- 先清缓存、重置峰值统计，从一个干净的起点开始测；
- 构造“最坏情况”的一批：总 token 数等于预算 `max_num_batched_tokens`。默认 16384 预算、4096 最大长度，就是 4 个长 4096 的假序列，内容全是 token 0；
- 手动设置 `num_scheduled_tokens`，因为没有经过调度器；
- 这些假序列**没有块表**，而且此时 KV 缓存还没分配。后面会看到，`prepare_prefill` 看到空块表会跳过 slot_mapping 的计算，注意力层看到空的 KV 缓存会跳过写入。所以预热走的是一条“纯计算、不碰缓存”的路径；
- 跑完再清一次缓存，把临时张量占的显存还给 CUDA。

预热只测了预填充，没测解码。解码每步只有一个 token 一个序列，临时显存远小于预填充，不需要测。

### 6.4 分配 KV 缓存

```python
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
```

**一个块多少字节**：K 和 V 两份 × 层数 × 每块 token 数 × 本卡的 KV 头数 × 每头维度 × 每个元素字节数。Qwen3-0.6B 单卡：

<p align="center">$2 \times 28 \times 256 \times 8 \times 128 \times 2 = 29{,}360{,}128 \text{ 字节} = 28 \text{ MiB}$</p>

注意是按 **KV 头数**（8）而不是注意力头数（16）算的，这就是 GQA 省显存的地方。张量并行时 KV 头也按卡切分，每张卡只存自己那部分头的 KV。

**能分多少块**：这个公式值得仔细看：

<p align="center">$\text{可用} = \text{total} \times \text{util} - \text{used} - \text{peak} + \text{current}$</p>

- `total × util`：允许用的上限，比如 24G 的 90%；
- `used`：此刻整张卡已经用了多少（`mem_get_info` 看到的，包括模型权重、CUDA 上下文、PyTorch 缓存，甚至别的进程）；
- `peak - current`：预热时 PyTorch 分配的峰值减去现在还占着的，就是前向计算需要的**临时显存**峰值。`used` 里已经包含了 `current`，所以要减的是这个差值，而不是整个 `peak`。

从上限里扣掉已经占用的，再为前向的临时张量预留出峰值空间，剩下的全部给 KV 缓存。

在 3090 上，默认配置分到了 700 个块，每块 256 个 token，总共可以同时缓存约 17.9 万个 token 的 KV。

**一整块大张量**：KV 缓存是一个形状 `[2, 层数, 块数, 256, KV头数, 头维度]` 的张量，一次分配。然后遍历模型所有模块，凡是有 `k_cache` 和 `v_cache` 属性的（就是每层的 `Attention`），按层号依次把对应切片赋给它。切片是**视图**，不拷贝数据。每层的 `k_cache` 形状是 `[块数, 256, KV头数, 头维度]`，正好是 flash-attn 分页接口要的格式。

这里依赖 `model.modules()` 的遍历顺序和层号一致，而 `nn.ModuleList` 是按下标顺序遍历的，所以没问题。

### 6.5 准备预填充的输入

![图 5](/img/in-post/vllm-notes-02/fig5-slot-mapping.svg)

这是整个项目里最需要耐心读的函数。它把一批长短不一的序列**拍平**成一维张量，同时生成注意力层需要的所有元数据。

```python
    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
```

对每个序列：

- `start`：从哪里开始算，也就是已缓存的长度；
- `seqlen_q`：本步算多少个，也就是 query 的长度；
- `end`：算到哪里；
- `seqlen_k = end`：key 的长度。本步的每个 query 要看到**从头开始到自己为止**的所有 key，包括缓存里已有的那部分，所以 key 长度是 `end` 而不是 `seqlen_q`；
- `input_ids`：只放本步要算的 token，`seq[start:end]`；
- `positions`：这些 token 在原序列里的真实位置，RoPE 要用。分块或命中缓存时，位置不从 0 开始；
- `cu_seqlens_q/k`：**累积长度**（cumulative sequence lengths）。所有序列的 token 首尾相连拼成一个长条，`cu_seqlens_q[i]` 到 `cu_seqlens_q[i+1]` 就是第 i 个序列在这个长条里的范围。flash-attn 的 varlen 接口靠它来区分不同序列，不需要 padding；
- `max_seqlen_q/k`：最大长度，flash-attn 用来决定划分多少线程块。

举个例子，本批有两个序列，一个从 0 开始算 3 个，一个从 512 开始算 108 个：

```
input_ids    = [a0 a1 a2 | c512 c513 ... c619]
positions    = [0  1  2  | 512  513  ... 619 ]
cu_seqlens_q = [0, 3, 111]
cu_seqlens_k = [0, 3, 623]      # 第二个序列 key 长度 620
```

接下来算 `slot_mapping`：每个新 token 的 KV 要写到缓存的哪个位置。缓存可以看作一个一维数组，第 b 个块第 j 个位置的“格子号”是 `b * 256 + j`。

```python
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
```

- 预热时没有块表，跳过。所以预热时 `slot_mapping` 是空的；
- `start_block`：起始 token 所在的逻辑块，向下取整；
- `end_block`：结束位置向上取整，所以 `range(start_block, end_block)` 恰好覆盖所有涉及的块；
- 对每个逻辑块 i，找到它的物理块 `block_table[i]`，格子范围默认是整块 `[b*256, b*256+256)`，但有两处要修正：
  - **第一块**可能从中间开始：加上 `start % 256` 的偏移。在默认配置下这个偏移其实总是 0：前缀命中得到的 `start` 是整块数 × 256；分块时被切的一定是本批第一个序列，它拿到的是整个预算，只要 `max_num_batched_tokens` 是 256 的倍数，切出来的边界也落在块边界上。把预算设成比如 1000 这样的值，这个偏移才会真正起作用。代码没有依赖这个巧合，写成了通用的形式；
  - **最后一块**可能只填一部分：结束于 `end - i * 256`，也就是 end 在这块里的偏移。
- 把这个范围内的所有格子号追加进去。

图 5 用请求 C 做了一次完整的计算。

```python
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions
```

- **什么时候需要块表**：key 的总长度大于 query 的总长度，说明至少有一个序列有“之前已经在缓存里”的部分（前缀命中或分块的后续块）。这时注意力层要从分页缓存里读 key，就需要块表。如果所有序列都是从 0 开始完整预填充，key 就是本步算出来的 key，直接用连续的 k、v 张量就行，不需要块表，走更快的路径；
- **拷贝到 GPU**：先在 **锁页内存**（pinned memory）上建 CPU 张量，再 `non_blocking=True` 异步拷到 GPU。锁页内存可以直接 DMA，异步拷贝不会阻塞 CPU。这些拷贝和后面的 kernel 在同一个 CUDA 流里排队，所以顺序是有保证的；
- 类型：token id 和位置是 int64（embedding 和索引要求），其余元数据是 int32（flash-attn 要求）；
- 最后把所有元数据塞进全局 `Context`，只返回 `input_ids` 和 `positions`。

```python
    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables
```

各序列块表长短不一，用 -1 补齐到最长的那个，拼成一个二维张量。flash-attn 只会读到每个序列实际长度对应的块，补的 -1 不会被访问。

### 6.6 准备解码的输入

```python
    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions
```

解码简单得多，每个序列恰好一个 token：

- 输入：最后一个 token（上一步刚采样出来的）。这里只用 `last_token`，所以序列化时只发它就够了；
- 位置：`len - 1`，它是序列里的最后一个；
- 上下文长度：`len`，注意力要看包括它自己在内的全部 token；
- 格子号：最后一个物理块的起始格 + 最后一块里的 token 数 - 1。比如最后一块里有 220 个 token（包括这个新 token），它就写在第 219 号位置。调度时 `may_append` 已经保证了需要的话新块已经分好；
- 解码**总是**要块表，因为要读历史 KV。

### 6.7 run 与 run_model

```python
    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids
```

`run` 是每一步真正执行的方法：准备输入 → （rank 0）准备温度 → 前向算 logits → （rank 0）采样并转成 Python 列表 → 清空上下文。`.tolist()` 会触发一次 GPU 到 CPU 的同步，这是每步唯一一个必须等 GPU 算完的地方。子进程返回 None。

`run_model` 用 `@torch.inference_mode()` 装饰，关掉梯度追踪和版本计数，比 `no_grad` 更省。它决定走哪条路：

- **急切执行**（eager）：预填充、或者用户要求 eager、或者批大小超过 512。预填充的 token 数每步都不一样，CUDA Graph 没法事先录制；
- **CUDA Graph 回放**：其他解码情况。

回放的步骤：

1. 找到**第一个大于等于**当前批大小的已录制批大小。比如 13 个请求用 16 的图；
2. 把真实输入写进录制时用的那组固定张量的前 `bs` 行；
3. `slot_mapping` 先全部填 -1，再写前 `bs` 个。多出来的 3 行是 -1，KV 写入核看到 -1 会直接跳过，不会把垃圾写进缓存；
4. `context_lens` 先清零，再写前 `bs` 个。多出来的行长度为 0，注意力不会读任何历史；
5. `block_tables` 只写前 `bs` 行、前 `当前最大块数` 列。多出来的行和列保留旧值，但因为 `context_lens` 限制了读取范围，不会被用到；
6. 回放；
7. 只取输出的前 `bs` 行去算 logits。`compute_logits` 在图外面执行，因为 lm_head 的 all-gather 通信和 logits 的大小（批 × 词表）都不适合放进图里。

`input_ids` 和 `positions` 多出来的行没清，是上一次回放留下的旧值。它们会跟着算一遍，但结果被丢弃，而且不会写进缓存，所以不影响正确性。

![图 6](/img/in-post/vllm-notes-02/fig6-cudagraph.svg)

### 6.8 录制 CUDA Graph

```python
    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
```

CUDA Graph 把一次前向里的几百个 kernel 启动录下来，之后用一次 `replay` 重放，省掉 Python 解释器和逐个启动 kernel 的开销。解码阶段每步计算量很小，这部分开销占比很大，所以提升明显。

代价是：回放时所有 kernel 读写的**显存地址是录制那一刻固定下来的**，形状也固定。所以：

- 先分配一组**最大尺寸**的静态张量（此时默认设备还是 cuda，所以都在 GPU 上）。每次录制用它们的前 `bs` 行切片，回放时往同一块内存里写数据；
- `block_tables` 的宽度按 `max_model_len` 能占的最大块数分配：4096 / 256 = 16 列；
- 录一组固定的批大小：1、2、4、8、16、32、……、512。批越大间隔越大，大批时 padding 浪费的比例本来就小。默认配置下一共 36 个；
- **从大到小**录制，并且所有图共用一个显存池（`graph_pool`）。第一个录的是最大的，池子按它的需求分配；之后小的图可以复用这块池子里的内存，不用各自再分。这就是为什么要 `reversed`；
- 每个批大小先**普通跑一遍**再录制。这一遍触发 torch.compile 为这个形状编译、各个库做懒初始化，这些操作不能出现在录制过程中；
- 录制时的输入全是 0，`context_lens` 也是 0。录制只关心 kernel 序列和地址，不关心数值；
- 录完后把这些静态张量存进 `graph_vars`，回放时往里写。

注意这里录的是 `self.model(...)`，不包括 `compute_logits`。

## 7. Context：一条从 ModelRunner 通向注意力层的暗线

```python
@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
```

就是一个模块级的全局变量，外加存取函数。`set_context` 每次都新建一个对象整体替换，而不是逐字段修改。

为什么要这样做？模型结构是 `Qwen3ForCausalLM → Qwen3Model → DecoderLayer × 28 → Qwen3Attention → Attention`，只有最底层的 `Attention` 和 `ParallelLMHead` 需要这些元数据。如果一层层当参数传下去，每一层的 `forward` 签名都要改。用全局上下文，模型代码几乎和 Hugging Face 的写法一样干净。vLLM 用的是同样的思路（`forward_context`）。

两个字段在不同阶段有不同含义：预填充用 `cu_seqlens_*` 和 `max_seqlen_*`，解码用 `context_lens`；`slot_mapping` 两个阶段都用；`block_tables` 解码时总是有，预填充时只在有前缀或分块时才有。

## 8. 算子层：layers/ 目录

### 8.1 Attention：写缓存 + 调 flash-attn

注意力层做两件事：把本步新算出的 K、V 写进分页缓存；然后调用 flash-attn 算注意力。

#### 写缓存的 Triton 核

```python
@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
```

这是个非常直白的 Triton 核，**每个 token 一个程序实例**：

- `idx`：本实例负责第几个 token；
- 读出它的格子号，是 -1 就直接返回。这就是 CUDA Graph 补齐的那些假行不会弄脏缓存的原因；
- `D = KV头数 × 头维度`，一个 token 在一层里的 K（或 V）一共 D 个数。Qwen3-0.6B 单卡是 8 × 128 = 1024；
- 从输入的 key、value 里读出第 idx 行的 D 个数。`key_stride` 是行之间的跨度，因为 k、v 是从 qkv 大张量里 `split` 出来的视图，行与行之间不是紧挨着的；
- 写到缓存的 `slot * D` 处。缓存形状是 `[块数, 256, KV头数, 头维度]`，把前两维看成一维的“格子”，每个格子正好 D 个数，所以格子号乘 D 就是偏移。

```python
def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)
```

包装函数主要是一组断言，确认内存布局符合核的假设：每个 token 内部的 D 个数是连续存放的（最后一维步长为 1，头之间步长为头维度），缓存里每个格子也是连续的 D 个数。只有行与行之间允许有间隔。然后以 N 个实例启动核。

#### forward

```python
class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
```

- 构造时 `k_cache`、`v_cache` 是空张量，等 `allocate_kv_cache` 替换成真正的缓存视图；
- **写缓存**：只有缓存非空时才写。预热时缓存还没分配，跳过；
- **预填充**，调用 `flash_attn_varlen_func`：
  - 没有块表：所有序列都从 0 开始，本步的 k、v 就是完整的 key/value，直接用这两个连续张量。`cu_seqlens_k` 此时和 `cu_seqlens_q` 相同；
  - 有块表：把 k、v **换成整个缓存**，再把块表传进去。flash-attn 会按 `cu_seqlens_k` 里每个序列的长度、通过块表从分页缓存里读 key。本步新算的 K、V 刚刚已经写进缓存了，所以缓存里既有之前的，也有本步的，是完整的；
  - `causal=True`：因果掩码。这里有个关键点：q 和 k 长度不同时，flash-attn 的因果掩码是**右下角对齐**的，即 query 的最后一个位置对齐 key 的最后一个位置。前缀命中时 query 是 key 的后 108 个，右下角对齐刚好让第 i 个 query 看到前 512 + i + 1 个 key，完全正确。正是这个约定，让分块预填充和前缀缓存不需要任何额外的掩码处理；
- **解码**，调用 `flash_attn_with_kvcache`：
  - q 的形状从 `[批, 头数, 头维度]` 变成 `[批, 1, 头数, 头维度]`，这个接口要求有一个“query 长度”维度，解码时是 1；
  - `cache_seqlens` 告诉它每个序列在缓存里有多长，块表告诉它去哪些块读；
  - 这里没有传新的 k、v，因为已经提前写进缓存了。

`softmax_scale` 是 1/√头维度，从外面传进来。GQA 不需要特殊处理：q 有 16 个头、k/v 有 8 个头，flash-attn 自动让每 2 个 q 头共享一个 kv 头。

### 8.2 线性层与张量并行

![图 8](/img/in-post/vllm-notes-02/fig8-tp-linear.svg)

张量并行的经典套路（Megatron-LM）：注意力和 MLP 各自由两个线性层组成，第一个**按列切**（输出维度切成 tp 份，每张卡算一部分输出），第二个**按行切**（输入维度切成 tp 份，每张卡拿自己那部分输入算出一个部分和），最后 all_reduce 求和。中间的注意力计算或激活函数是逐头或逐元素的，可以直接在切开的数据上做，不需要通信。一层只需要一次 all_reduce。

```python
def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)
```

- `divide`：必须整除，否则直接报错。比如 16 个头不能切成 3 份；
- 权重形状是 `[输出, 输入]`，和 `nn.Linear` 一致。这里传进来的尺寸已经是**切分后**的尺寸；
- `tp_dim`：沿着哪个维度切。0 是输出维度（列切），1 是输入维度（行切）；
- **把加载函数挂在参数对象上**：`self.weight.weight_loader = self.weight_loader`。这样加载器（9.3 节）拿到一个参数时，不需要知道它属于哪种层，直接调用参数上挂着的函数就行。每种层用自己的 `weight_loader` 决定从完整权重里切哪一块；
- 用 `torch.empty` 不初始化，因为马上会被加载的权重覆盖。

**ReplicatedLinear**：不切，每张卡一份完整的。Qwen3 里没用到，是给其他模型留的。

**ColumnParallelLinear**：

```python
class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
```

输出维度除以 tp。加载时从完整权重里沿第 0 维切出第 rank 份：`narrow(dim, start, length)` 返回一个视图，不拷贝。前向就是普通的线性层，输出是完整输出的一部分。偏置也沿第 0 维切，用同一个函数。

**MergedColumnParallelLinear**：把多个列切线性层合并成一个大矩阵乘法。MLP 里的 gate_proj 和 up_proj 输入相同、都是列切，合在一起算一次，比分开算两次快。

```python
class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)
```

一个细节：`self.output_sizes = output_sizes` 写在 `super().__init__` **之前**。一般 `nn.Module` 要求先调父类构造函数再设属性，但普通的列表属性不经过 `nn.Module` 的特殊处理，所以没问题。

磁盘上 gate_proj 和 up_proj 是两个独立的权重，加载器会分别调用两次，`loaded_shard_id` 分别是 0 和 1。每次：

- 在本卡的合并参数里定位：第 id 个子矩阵的起始偏移 = 前面所有子矩阵的输出维度之和 ÷ tp，长度 = 自己的输出维度 ÷ tp；
- 从磁盘权重里取第 rank 份：`chunk(tp, 0)[rank]`；
- 拷进去。

两卡时，本卡的合并参数布局是 `[gate 的第 rank 份, up 的第 rank 份]`。每张卡都是先 gate 后 up，所以激活函数里 `chunk(2)` 切出来的正好是本卡的 gate 和 up。

**QKVParallelLinear**：同样的思路合并 q、k、v 三个投影，区别是三者尺寸可能不同（GQA 时 k、v 的头少）。

```python
class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)
```

- 没给 KV 头数就当成和注意力头数一样（多头注意力，不是 GQA）；
- 完整输出维度 =（q 头数 + 2 × kv 头数）× 头维度。Qwen3-0.6B 是 (16 + 16) × 128 = 4096，单卡时就是 `[4096, 1024]` 的权重；
- 本卡布局是 `[本卡的 q 头 | 本卡的 k 头 | 本卡的 v 头]`，偏移用**每卡**的头数计算；
- 磁盘上的 q、k、v 权重按头连续排列，`chunk` 按 tp 均分，第 rank 份正好是第 rank 组头。

**RowParallelLinear**：

```python
class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
```

- 输入维度除以 tp，沿第 1 维切；
- 偏置是一维的，形状等于输出维度，没法按输入切，所以每张卡都存一份**完整的**偏置；
- 前向时**只有 rank 0 加偏置**。因为接下来要 all_reduce 求和，如果每张卡都加一次，偏置就被加了 tp 次；
- all_reduce 之后，每张卡都拿到完整的结果。

### 8.3 词表并行的嵌入层与输出头

```python
class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y
```

嵌入表（151936 × 1024）按词表切成 tp 份，每张卡存一段连续的 token id，范围是 `[vocab_start_idx, vocab_end_idx)`。

前向时：

1. `mask`：哪些 token 属于本卡负责的区间；
2. `x = mask * (x - start)`：属于本卡的，换算成本卡内的下标；不属于的，乘 0 变成下标 0（随便查一行，反正会被清掉，关键是不越界）；
3. 查表；
4. `mask.unsqueeze(1) * y`：不属于本卡的行清零；
5. all_reduce 求和：每个 token 恰好只有一张卡给出了非零结果，加起来就是完整的嵌入。

单卡时跳过所有这些，就是普通的 `F.embedding`。

```python
class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
```

输出头继承嵌入层，复用它的切分和加载逻辑，权重形状同样是 `[本卡词表大小, 隐藏维度]`。

- **预填充时只取每个序列的最后一个位置**：`cu_seqlens_q[1:] - 1` 正好是每个序列在拍平张量里最后一个 token 的下标。比如 `cu_seqlens_q = [0, 3, 111]`，得到 `[2, 110]`。只有最后一个位置的输出用来预测下一个 token，其余位置的 logits 没用。词表有 15 万，一个 1000 token 的 prompt 如果全算 logits，就是 1000 × 15 万个浮点数，这一步省了大量计算和显存；
- 解码时每个序列本来就只有一个位置，不用挑；
- 用 `F.linear(x, weight)` 算 logits，相当于和每个词的向量做点积；
- 多卡时，每张卡算出本卡词表区间的 logits，用 `gather` 收集到 rank 0（不是 all_gather，因为只有 rank 0 采样），沿最后一维拼起来。其他卡返回 None。

### 8.4 旋转位置编码 RoPE

```python
def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)
```

把每个头的 128 维切成前后两半 x1、x2，把 (x1[i], x2[i]) 看作一个二维向量，旋转一个角度：

<p align="center">$\begin{pmatrix} y_1 \\ y_2 \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$</p>

这是“前后半对”的配对方式（GPT-NeoX 风格），不是“相邻两个一对”的方式（GPT-J 风格）。两种方式数学上等价，但必须和模型训练时一致，Qwen 用的是前者。计算在 float32 下进行，最后转回原精度，避免 bf16 下三角函数乘法的精度损失。

```python
class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key
```

- 只支持对整个头做旋转（`rotary_dim == head_size`），不支持部分旋转；
- 频率：第 i 对的旋转角速度是

<p align="center">$\omega_i = \text{base}^{-2i/d}, \quad i = 0, 1, \dots, d/2 - 1$</p>

  base 越大，低频分量转得越慢，能区分的距离越远。Qwen3 用的 base 是一百万；
- `einsum("i,j -> ij")` 就是外积：位置 p 的第 i 对的角度是 p × ω_i，得到一张 `[最大位置数, 64]` 的角度表；
- 预先算好所有位置的 cos 和 sin，拼成 `[最大位置数, 128]`，再在中间插一维变成 `[最大位置数, 1, 128]`，那一维是为了在“头”这一维上广播；
- `register_buffer(..., persistent=False)`：作为模块的一部分跟着搬到 GPU，但不会出现在 `state_dict` 里，不需要从权重文件加载；
- 前向时按位置查表、切出 cos 和 sin，对 q 和 k 分别旋转。v 不需要位置编码；
- 这里的 `positions` 就是 `prepare_prefill`/`prepare_decode` 里算出来的真实位置，分块和前缀命中时从中间开始，编码依然正确。

这张表在 Qwen3-0.6B 上是 40960 × 128 个 float32，约 20 MB。

```python
@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
```

`lru_cache(1)` 让相同参数的调用返回**同一个对象**。28 层都调用 `get_rope`，参数一样，于是共享一个 RoPE 模块和一张 cos/sin 表，不用存 28 份。

### 8.5 RMSNorm：把残差加法融合进来

```python
class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
```

RMSNorm 公式：

<p align="center">$y = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \odot w$</p>

`rms_forward` 就是这个公式：转 float32、算平方均值、乘以倒数平方根、**转回原精度后**再乘权重。先转回再乘权重，是为了和 Hugging Face 的 Qwen3 实现在数值上完全一致（HF 就是这个顺序）。

`add_rms_forward` 把“残差加法 + 归一化”融合在一起：先 `x + residual`，把和存为新的残差（原精度），再对和做归一化。两个输出：归一化后的结果（送给下一个子层）和新的残差（留给下一次加法）。

两个函数都用 `@torch.compile` 装饰，编译器会把这一串逐元素操作融合成一两个 kernel，只读写显存一次。

`x.float()` 在输入已经是 float32 时返回的是**同一个张量**，后面的 `mul_` 原地修改会改动输入。这里输入都是 bf16，`float()` 总是产生新张量，所以是安全的。

实际跑的时候会看到 torch.compile 报 `recompile_limit (8)` 的警告，来源是 q_norm 和 k_norm：它们作用在三维的 `[token数, 头数, 头维度]` 上，token 数每次不同、q 和 k 的头数也不同，触发了多次重编译。超过上限后回退到不编译的版本，结果依然正确，只是这两处慢一点。

### 8.6 SiluAndMul 与采样器

```python
class SiluAndMul(nn.Module):

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
```

SwiGLU 激活：输入是 gate_up_proj 的输出，最后一维前一半是 gate、后一半是 up，结果是 `silu(gate) * up`。silu(x) = x · sigmoid(x)。维度从 2 × 3072 变回 3072（单卡）。

```python
class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
```

采样器只有三行，第三行值得细讲：

1. logits 转 float32，每行除以自己的温度（`unsqueeze(1)` 把温度从 `[批]` 变成 `[批, 1]` 以便广播）。每个请求可以有不同的温度；
2. softmax 得到概率 p；
3. 生成一组服从参数为 1 的指数分布的随机数 E，算 `p / E`，取最大值的下标。

为什么这等价于按概率 p 采样？设 $E_i \sim \text{Exp}(1)$ 独立，则 $E_i / p_i \sim \text{Exp}(p_i)$，即速率为 $p_i$ 的指数分布。取 $p_i / E_i$ 的最大值，就是取 $E_i / p_i$ 的最小值。对于独立的指数分布有一个经典结论：

<p align="center">$P\left(\arg\min_i \frac{E_i}{p_i} = k\right) = \frac{p_k}{\sum_i p_i} = p_k$</p>

可以想象每个 token 在“赛跑”，第 k 个的完成时间服从速率 $p_k$ 的指数分布，概率越大跑得越快，第一个到终点的恰好以概率 $p_k$ 是 k。这和 Gumbel-max 技巧本质相同（对 E 取负对数就得到 Gumbel 噪声）。

好处是完全在 GPU 上并行完成，不需要 `torch.multinomial` 那种前缀和再二分查找的操作，也容易被 torch.compile 融合。`clamp_min_(1e-10)` 防止 E 恰好为 0 导致除零。

## 9. 模型与权重加载

### 9.1 Qwen3Attention

```python
class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias
```

- 头数和 KV 头数都必须能被 tp 整除，每张卡分到 `总数 / tp` 个。Qwen3-0.6B 有 8 个 KV 头，所以最多支持 8 卡；
- `head_dim`：配置里有就用配置的，没有就用隐藏维度 / 头数。Qwen3-0.6B 配置里明确写了 128，而 1024 / 16 = 64，两者不同。所以这个“优先用配置”的写法很重要，Qwen3 的 q 投影输出维度（16 × 128 = 2048）比隐藏维度（1024）还大；
- `q_size`、`kv_size`：**本卡**的 q、k（或 v）的总维度，用来切 qkv 张量；
- `scaling = 1/√128`。

```python
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
```

- qkv 用合并的列切线性层，o_proj 用行切，正好组成图 8 的“列切接行切”；
- `rope_theta` 的读法是为了兼容新版 transformers：新版把 Qwen3 的 `rope_theta` 挪进了 `rope_scaling` 字典里（我的环境里读出来的是 `{'rope_theta': 1000000, 'rope_type': 'default'}`），所以优先从字典里取；
- **q_norm 和 k_norm**：Qwen3 相对 Qwen2 的一个改动，对每个头的 q 和 k 单独做 RMSNorm（维度 128），用来稳定训练。代码用“没有 qkv 偏置”来判断是否需要它们：Qwen2 有 qkv 偏置、没有 q/k norm，Qwen3 反过来。这个判断方法有点取巧，只对 Qwen 系列成立。

```python
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output
```

1. 一次矩阵乘法得到 qkv，形状 `[token数, 2048 + 1024 + 1024]`；
2. 沿最后一维切成 q、k、v。`split` 返回视图，所以 k、v 的行步长仍然是 4096，这就是写缓存核里要传 stride 的原因；
3. reshape 成 `[token数, 头数, 头维度]`。`view` 对这种“最后一维切出来的视图”是合法的，因为头这一维仍然连续；
4. 对每个头做 q_norm、k_norm。RMSNorm 在最后一维（128）上归一化，自然就是逐头的；
5. 加旋转位置编码；
6. 注意力，输出 `[token数, 头数, 头维度]`；
7. 把头维度展平成 `[token数, 2048]`，经过 o_proj（行切，内部 all_reduce）回到 `[token数, 1024]`。

注意 q_norm 在 RoPE **之前**。

### 9.2 MLP、DecoderLayer 与整体结构

```python
class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x
```

1024 → 2 × 3072（合并的 gate 和 up）→ SwiGLU → 3072 → 1024。同样是“列切接行切”。只支持 silu 激活。

```python
class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

从 Hugging Face 配置里取参数。`attention_bias` 缺省时当作 True，Qwen3 的配置里是 false，于是 `qkv_bias=False`，启用 q/k norm。

```python
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
```

这是 vLLM 系实现里常见的一个技巧：**把残差加法推迟到下一个归一化里做**。

标准的 Pre-Norm Transformer 层是：

```
h = x + Attn(Norm1(x))
out = h + MLP(Norm2(h))
```

这里每一层接收两个量：`hidden_states`（上一个子层的输出，还没加回残差）和 `residual`（残差流）。每次调用归一化时顺便把它们加起来：

- 第一层进来时 `residual` 为 None，`hidden_states` 就是嵌入。直接归一化，嵌入本身存为残差；
- 之后每层进来时，`hidden_states` 是上一层 MLP 的输出，用 `add_rms_forward` 先加上残差、再归一化，同时得到新的残差；
- 注意力之后同理：post_attention_layernorm 把注意力输出加进残差并归一化；
- 这一层返回 MLP 的输出和残差，**MLP 的输出还没加进残差**，留给下一层的 input_layernorm。

这样每次“加法 + 归一化”都融合成一个编译后的 kernel，少读写一遍显存。数学上和标准写法完全等价。

```python
class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
```

嵌入 → 28 层 → 最终归一化。最终归一化同样带着残差调用，把最后一层 MLP 的输出加进去，这一步的新残差不再需要，丢弃。

注意输入是**一维**的 `[token数]`，不是 `[批, 序列长度]`。整个模型从头到尾都在处理“拍平”后的 token 序列，只有注意力层靠 Context 知道哪些 token 属于哪个序列。其他层（线性、归一化、MLP）本来就是逐 token 独立计算的，根本不关心序列边界。这就是“不补齐的批”能成立的原因。

```python
class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
```

- `packed_modules_mapping`：告诉加载器，磁盘上名为 `q_proj` 的权重要装进 `qkv_proj` 的 q 部分，依此类推。这是合并线性层和磁盘权重之间的桥梁；
- `tie_word_embeddings`：Qwen3-0.6B 的输入嵌入和输出头共用一个矩阵。直接让两个参数的 `.data` 指向同一块内存。磁盘上其实存了 `lm_head.weight`（我检查过，内容和嵌入完全相同），加载时会再拷一遍进同一块内存，结果不变；
- `forward` 只返回隐藏状态，`compute_logits` 单独调用。这样 CUDA Graph 可以只录前者（6.7 节）。

### 9.3 权重加载

```python
def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
```

28 行完成全部加载：

1. 遍历目录下所有 `.safetensors` 文件，用 `safe_open` 以 PyTorch 格式、在 CPU 上打开。safetensors 支持按需读取单个张量，不需要一次把整个文件读进内存；
2. 对文件里的每个权重名（比如 `model.layers.0.self_attn.q_proj.weight`）：
   - 如果名字里含有映射表里的某个键（`q_proj`），就把它替换成合并后的名字（`qkv_proj`），找到对应参数，调用它挂着的 `weight_loader`，额外传入分片标识（`"q"`）；
   - 否则（`for ... else`，循环没有 `break` 才执行）按原名找参数。有挂 `weight_loader` 的就用它（比如行切、词表切），没有的就直接拷贝（比如 RMSNorm 的权重）；
3. `f.get_tensor` 读出来的是 CPU 张量，`copy_` 到 GPU 参数时自动完成设备间拷贝和类型转换。

Qwen3-0.6B 一共 311 个权重，都能找到对应的参数。子串匹配 `k in weight_name` 比较粗糙，比如一个权重名里恰好包含 `up_proj` 字样但并不是 MLP 的 up 投影，就会误匹配。在 Qwen3 上没有这种情况。

## 10. 实验：把上面的逻辑跑出来看

下面的实验都在一张 RTX 3090 上用 Qwen3-0.6B 跑。为了让分块更容易观察，前两个实验把 `max_num_batched_tokens` 设成了 1024，其余都是默认值。我在 `LLMEngine.step` 前后打印了每个序列的计数器和块表。

### 10.1 分块预填充与前缀缓存

三个请求：A 是 1500 个随机 token；B 是 600 个公共前缀 + 10 个 token；C 是同样 600 个前缀 + 20 个 token。先一起提交 A 和 B，全部完成后再提交 C。

```
step1 P s1(cached=0,sched=1024,len=1500,blocks=[0..5])
step2 P s1(cached=1024,sched=476)
step3 P s2(cached=0,sched=610,len=610,blocks=[6,7,8])
step4 D s1,s2
...
第二轮:
step6 P s3(cached=512,sched=108,len=620,blocks=[6,7,9])
```

（`s1` 是 A，`s2` 是 B，`s3` 是 C；`P` 表示预填充步，`D` 表示解码步。）

- **step1**：A 需要 1500 个 token，预算只有 1024。它是本批第一个，允许分块，算 1024 个。6 个块一次全部分好。算完没追上，留在等待队列队首；
- **step2**：A 继续，剩 476 个。预算还剩 548，可 B 需要 610，放不下，又不是本批第一个，不允许分块，所以 B 等下一步。这一步只有 A 自己；
- **step3**：B 单独预填充；
- **step4 起**：等待队列空了，A 和 B 一起解码；
- **step6**：C 命中了 B 留下的块 6 和块 7，`cached=512`，只算 108 个。

这个轨迹同时印证了三件事：只有第一个序列能分块、预填充和解码从不混批、前缀缓存按整块命中。

### 10.2 抢占与恢复

为了触发抢占，我把 KV 块数手动限制成 5 个，提交 3 个 200 token 的 prompt，每个生成 260 个 token（`ignore_eos=True`）。

每个序列一开始需要 1 块，3 个一共 3 块；长度涨到 257 时每个都需要第 2 块，但总共只有 5 块：

```
解码到 len=257 时: s6 被抢占
s4: blocks=[0, 3]   s5: blocks=[1, 4]
...
s4、s5 结束后:
step 261  P s6(len=257, cached=256, blk=[2, 3])
```

- 解码时从运行队列队首依次取序列：s4、s5 先被取出，各领到第 2 块（块 3、块 4）。取出 s6 时已经没有空闲块了，按 4.3 节的逻辑应该踢掉运行队列**队尾**的序列来腾地方，但此时 s6 后面已经没有别的序列（`self.running` 为空），于是走 `if self.running` 的 `else` 分支，s6 把自己抢占掉，释放块 2，回到等待队列队首；
- s4、s5 结束后释放了块，s6 重新预填充。它原来的第一块（块 2，装满了 256 个 token，登记过哈希）还在，命中缓存，`cached=256`，只重算了最后 1 个 token。

被抢占的代价比想象中小得多，前缀缓存把“重算”变成了“查表”。

### 10.3 吞吐

跑仓库自带的 `bench.py`：256 个请求，输入长度 100 到 1024 随机，输出长度 100 到 1024 随机，默认配置（开 CUDA Graph）：

```
Total: 133966tok, Time: 30.78s, Throughput: 4351.86tok/s
```

对比：预热时单个请求解码大约 349 tok/s。批处理把吞吐提升了一个数量级以上。

### 10.4 几个坑

读代码时发现、然后实际验证过的几个边界问题：

**1. 超长 prompt 没有检查**。`max_model_len=512`，提交一个 600 token 的 prompt：

```
RuntimeError: The expanded size of the tensor (2) must match the existing size (3)
```

预填充能过，到解码时 CUDA Graph 的 `block_tables` 缓冲区只有 ⌈512/256⌉ = 2 列，这个序列却有 3 个块，写不进去。引擎没有在 `add_request` 时检查长度，也没有在生成过程中截断。实际使用时要自己保证 prompt 长度 + `max_tokens` 不超过 `max_model_len`。

**2. `max_num_seqs` 不是 16 的倍数时 CUDA Graph 会找不到图**。`max_num_seqs=20` 时，录制的批大小是 `[1, 2, 4, 8, 16]`（`range(16, 21, 16)` 只有 16）。当 20 个序列同时解码，`next(x for x in graph_bs if x >= 20)` 找不到，抛 `StopIteration`。设成 16 的倍数，或者开 `enforce_eager` 就能避开。

**3. 多卡 + 非默认块大小时，子进程的块大小不对**。`LLMEngine` 里 `Sequence.block_size = config.kvcache_block_size` 只改了主进程的类属性。子进程是 spawn 出来的，重新导入模块，类属性还是 256。我写了个小实验：主进程设成 512，把一个 300 token 的序列 pickle 给 spawn 子进程：

```
主进程: block_size=512  last_block_num_tokens=300
子进程: block_size=256  last_block_num_tokens=44
```

`prepare_decode` 用 `last_block_num_tokens` 算格子号，子进程算出来的就是错的，KV 会写错位置。默认块大小 256 时不会触发。

**4. 只认一个结束符**，见 4.4 节。

## 11. 小结

把 nano-vllm 读完，再回头看上一篇里 vLLM 的那些概念，每一个都能落到具体的几行代码上：

| 概念 | nano-vllm 里的实现 |
| --- | --- |
| 分页 KV 缓存 | 一整块 `[2, 层, 块, 256, 头, 维]` 张量 + 每个序列的 `block_table` + `slot_mapping` |
| 前缀缓存 | 链式 xxhash + `hash_to_block_id` + 引用计数 + 空闲块懒回收 |
| 连续批处理 | `step` 每次重新调度，序列随时加入、随时退出 |
| 分块预填充 | `num_cached_tokens` / `num_scheduled_tokens` 两个计数器，只切本批第一个 |
| 抢占 | 踢队尾、释放块、放回等待队列队首，靠前缀缓存恢复 |
| 不补齐的批 | 拍平的 `input_ids` + `cu_seqlens` + flash-attn varlen |
| 张量并行 | 列切 / 行切线性层 + 词表并行 + 共享内存广播调用 |
| CUDA Graph | 36 个批大小、从大到小录、共享显存池、补齐行写 -1 |

和 vLLM 相比，nano-vllm 省掉的主要是：预填充和解码混批、更细粒度的块、多种模型和采样方法、在线服务和流式输出、各种边界检查。这些都是工程上必要的，但不影响理解核心机制。

我觉得读这份代码最大的收获是看清了**三个计数器**（`num_tokens`、`num_cached_tokens`、`num_scheduled_tokens`）如何贯穿调度、块管理和输入准备：前缀缓存、分块预填充和抢占恢复这三个看起来独立的功能，在代码里其实是同一套逻辑的三种情况，都只是“从 `num_cached_tokens` 开始，算 `num_scheduled_tokens` 个”。

下一篇打算回到 vLLM 本体，看看它在这套骨架上又加了哪些东西。


