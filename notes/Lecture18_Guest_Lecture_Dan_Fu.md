# CS336 Guest Lecture: 推理系统与新型架构 — Dan Fu (Together AI / UCSD)

> **课程**: Stanford CS336 — Language Models From Scratch (Spring 2026)
> **主讲**: Dan Fu（Together AI & UCSD 助理教授）
> **主题**: 推理系统工程、GPU Kernels 与 Loop Transformer 架构
> **形式**: 特邀讲座（无课件，仅视频字幕）

---

## 目录

1. [引言：规模革命与推理系统](#1-引言规模革命与推理系统)
2. [Token 的一生：推理引擎全景](#2-token-的一生推理引擎全景)
   - [2.1 Prefill vs Decode](#21-prefill-vs-decode)
   - [2.2 Continuous Batching](#22-continuous-batching)
   - [2.3 KV Cache 与 Prefix Sharing](#23-kv-cache-与-prefix-sharing)
   - [2.4 分布式推理与硬件趋势](#24-分布式推理与硬件趋势)
   - [2.5 大规模推理的工程挑战](#25-大规模推理的工程挑战)
3. [Mega Kernels：将 GPU 利用率推向极致](#3-mega-kernels将-gpu-利用率推向极致)
4. [Parse：Loop Transformer 架构](#4-parseloop-transformer-架构)
   - [4.1 为什么需要循环](#41-为什么需要循环)
   - [4.2 训练不稳定性的数学分析](#42-训练不稳定性的数学分析)
   - [4.3 循环的 Scaling Laws](#43-循环的-scaling-laws)
5. [总结：全栈创新](#5-总结全栈创新)

---

## 1. 引言：规模革命与推理系统

> "In this class you're mostly talking about how to train language models. Today I'm going to talk about once you have one of those models, what it looks like from the other side — what it looks like to actually serve these models. Inference is the engine that turns electricity into intelligence."

**Dan Fu 的背景**：UCSD 助理教授 + Together AI（Percy 也是 Together 成员）。Together 是一家 AI Cloud，专注于 GPU 推理、微调等领域，有浓厚的研究背景。

**规模的数量级跃迁**：

| 年份 | 里程碑 |
|------|--------|
| 2018 | 最大模型 ~100M 参数（"Oh my god, these things are pretty crazy"） |
| 2019 | GPT-2（被认为 "太危险不能发布"） |
| 2026 | 开源万亿参数模型；前沿 ~5-10T 参数 |

> "At this point you can train a GPT-2 quality model in this class. If you tried hard enough, you could."

**历史类比——从马到汽车的 10 年转型**：

> "In 1902 there were 130,000 working horses in Manhattan. In 1898 they had an entire academic conference about what to do about all the horse manure. Their conclusion: there's nothing we can do. 10 years later, by 1912, cars had already outnumbered horses."

Dan 认为 LLM 的 "1912 时刻" 可能就在去年——他开始用 LLM 写大部分代码。

**GPU 是新的石油**：

> "Hundreds of billions of dollars of investment into GPUs. Entire sovereign wealth funds are making these things a major part of their piece."

但正如石油需要引擎才能转化为动能，**GPU 也需要推理引擎（inference engines）和 GPU kernels 才能将算力转化为智能**。

> "If you understand inference and understand the inference engines, if you understand the GPU kernels that underly a lot of the core technology, you can enable full stack innovation in machine learning algorithms."

---

## 2. Token 的一生：推理引擎全景

### 2.1 Prefill vs Decode

推理的两个核心阶段：

| 阶段 | 描述 | 计算特性 |
|------|------|---------|
| **Prefill** | 处理 prompt（如 10K tokens 输入）→ 一次输出 1 个 token | **Compute-bound**（类似训练，但没有 backward pass） |
| **Decode** | 逐个生成 token → 每个 token 都需要通过完整模型 | **Memory bandwidth-bound**（模型权重反复加载，但计算量少） |

> "Prefill: 10,000 tokens in, one token out — very compute-bound. Decode: you're generating one token at a time. Every time you generate a new token, you then have to run that back through the model. If you do the math, there's actually not too many FLOPs. You have to load up the model every time just to generate a single token."

**Disaggregated Prefill/Decode**（解耦架构）：

> "Prefill looks a lot like training, very FLOP-heavy. Decode is very memory-bandwidth-heavy. These things will take different amounts of time. A very basic optimization that pretty much we've all started adopting is run prefill on one set of workers, decode on another — so you can specialize."

**硬件分化趋势**：

> "Nvidia is planning on using its GPUs for the prefill side, using LPU Groq chips for the decode. OpenAI has a compute partnership with Cerebras — it's another chip that's much better at decode. Other companies like SambaNova are making bets along various parts of this space."

### 2.2 Continuous Batching

> "When you have a system that is processing many different requests at a time, we have this technique called continuous batching."

**Continuous Batching 的核心机制**：
- 多个请求同时在 GPU 上运行
- 短请求完成后立即加入新请求
- 受限于 **compute** 和 **memory**（KV cache 占据 GPU 显存）
- 当前请求的 KV cache 占满后，新请求进入排队

> "You might have another request that comes in — it's a very long request. But maybe you don't have enough GPU memory because your KV cache is on the GPU. You might start queuing. Once that long request is done, you can start the new request."

### 2.3 KV Cache 与 Prefix Sharing

> "You probably have a lot of users who are saying 'hi ChatGPT' or 'hi Claude'. Theoretically you don't need to compute new activations for every single user."

**KV Cache 的作用**：

1. **Prefix Sharing**：用树结构查找哪些 token 之前见过 → 直接复用 activations
2. **跨请求重用**：如果用户粘贴了一本长书然后追问，第二次不需要重新 prefill 整本书
3. **层级存储**：

```
GPU HBM → CPU DRAM → SSD
 (快/小)            (慢/大)
```

> "If you're storing your KV cache on CPU memory, you really care about the speed of being able to read that KV cache back. Next, you might put KV cache onto the disk — then you start caring about SSDs."

**这解释了 OpenAI 大量采购 SSD/DRAM 的原因**——用于存储海量 KV cache。

**KV 缓存的调度策略**：

> "When you're building your engine, there's this complicated dance: I haven't seen these tokens in a while, maybe I'll evict it, send it onto the CPU or disk. When I get a new request in, I have to go fetch them, load them up."

实际上，这正是经典的 OS 页面置换问题——**LRU（Least Recently Used）** 就是一个不错的启发式，理论上在最优解的 2x 范围内。

**Cache-Aware Prefill/Decode Disaggregation**（Together 的工作）：

> "If we have a new request that comes in that's a very low cache hit rate, send it to one set of GPUs so those can all process together. Then send all my other warm requests to another set of prefill nodes."

这个简单的路由优化（两行代码）可以实现**最高 40% 的加速**。

### 2.4 分布式推理与硬件趋势

**模型切分策略**：

> "You're not going to be able to fit the full model onto each GPU. Various ways: Tensor Parallelism (split each tensor across GPUs), Expert Parallelism (MoE experts on different GPUs)."

**NVL72 (GB200 Grace Blackwell)**：

> "72 GPUs connected with really fast interconnect. How can I split my trillion parameter model across all 72 GPUs? What does it buy me? How do I think about fault tolerance?"

**硬件故障是真实问题**：

> "These things can fail a lot. One reason is the connectors are kind of flimsy — they're made of plastic, not metal. If you jam the thing in too much, your cables bend and you get really flaky NVLinks."

当模型分布在 64 块 GPU 上服务数百万用户时，单块 GPU 故障后的容错处理是一个重要的系统工程问题。

### 2.5 大规模推理的工程挑战

Dan 分享了几个生产环境中的真实 Bug：

**Bug 1 — NaN → Repetition Loop**：

> "You have a kernel that is very slightly wrong but the conditions for triggering it are very rare. You start having some of your logits turn into NaNs halfway through. When that happens, the model starts outputting the same token — 'hi hi hi hi hi hi hi' — or exclamation points, getting caught into these loops."

**Bug 2 — Tool Call Loop**：

> "The model would say 'hey make an internet search' — usually it would return to the user. But in this case it wasn't returning correctly, so it would say 'hey make an internet search' over and over for tens of thousands of tokens."

**Bug 3 — 随机输出中文**（off-by-one error）：

> "Sometimes you would read in some extra uninitialized memory space from your GPU, run it through attention, then at the end get a random Chinese character. The model goes 'why did I start suddenly thinking Chinese? The user must be asking a question in Chinese' — and then just veer off into Chinese."

很多人猜测是模型在中文数据上微调了——"实际上只是一个内核的 off-by-one 错误。"

> "Sometimes when this happens, it's because the model has legitimately been trained to think in Chinese. Sometimes it can just be an off-by-one bug in somebody's code."

---

## 3. Mega Kernels：将 GPU 利用率推向极致

> "The fundamental challenge when you're running decode is that you have to run the whole model to generate a single token. You've turned this massively parallel system into basically a glorified memory loader."

**传统 Kernel 编程的问题**：

GPU kernel 通常一次只做一个操作（norm kernel、matmul kernel、attention kernel）→ 大量 GPU 空闲等待（kernel launch/teardown gaps, tail effects）。

![Kernel timeline 示意 — 横轴时间，纵轴 SMs，空白=idle]

> "No matter how well you try to write the kernel, you're always going to have downtime in your GPU — kernel launch and teardown gaps, tail effects. If you're processing a batch and one input is very short, one is very long, you're waiting for the very long input to finish."

**Mega Kernel 的核心思想**：将多个操作融合进一个 kernel——不仅是 Flash Attention 级别的融合，而是 **整层乃至整套模型** 的融合。

> "Instead of treating each operation in the model as its own kernel, let's write a single kernel to cover multiple operations at once. You start thinking of the GPU as a massive distributed system: I have all this work, some of it has dependencies, how can I schedule it to maximize GPU utilization?"

**两个关键优化示例**：

1. **KV Cache Load 与 QKV 并行**：
   > "During decode, you start the KV cache load while QKV plus RoPE is still running. Then once QKV is done, you have your new query tokens, and you can run the rest of attention."

2. **O-Projection Weight Loading 与 Attention 并行**：
   > "You have your O-projection start loading the weights before your attention operation is over."

**实现框架 — Thunderkittens**：

> "A kernel writing library — almost like Triton except more low-level, a lot more fine-grain control. You implement each sub-kernel in its own file, then have a big virtualized shared memory system to orchestrate the running of these operations."

**效果**：

| 指标 | 数值 |
|------|------|
| Decode 加速 | 30-70%（仅 attention）; 整层加速更多 |
| H100 带宽利用率 | **72%**（near speed-of-light） |

> "If you just ignore all the complexities and say 'how fast can the GPU physically go', we are pretty close — 72% of that speed of light."

**Mega Kernel 的代价**：

> "Mega kernels turn out to be very very labor intensive to write. A full talented kernel engineer over the course of a year will probably be able to write mega kernels for one hardware, for two or three models, for batch sizes 1 to 16. You go batch size 17 — nope, start over."

> "If you can do it, it will go super fast — you will never be able to go faster. But it just takes a lot of energy and effort."

**多 GPU 通信的融合**：DeepSeek V4 发布时就对 MoE inference layer 做了 mega kernel，融合了部分通信操作。Together 也在尝试将 NVLink 调用融入 mega kernel。

---

## 4. Parse：Loop Transformer 架构

> "With Parse we wanted to ask: is scaling parameters and data the only way? Is there potentially something else — some other way that you can get quality?"

**核心想法 — Loop Transformers**：

将 Transformer 的某些 block 循环运行——不让 token 一层层直通到底，而是在某些层 "send it back" 多次。

![Loop Transformer 概念图 — 紫色 block = 循环块，activation 反复通过]

> "You can keep your parameters constant but it gives you a dial to increase your FLOPs. If you think more FLOPs equals higher quality, this is a way to increase quality without paying a higher parameter cost."

**循环的优势**：
- 参数不变但 FLOPs 可调（inference-time compute 的另一种实现）
- 理论上更强的 expressivity（有工作表明存在 looped models 可以表达但同参数量的 non-looped models 无法表达的东西）
- 循环可以灵活调整——"你可以在推理时减少循环次数来加速"

### 4.2 训练不稳定性的数学分析

**问题**：Loop Transformers 极难训练。

> "If you did a simple thing like a learning rate sweep, you'd see 9 times out of 10 this model just isn't going to converge — it's going to blow up. You're going to get NaNs, big loss spikes."

之前的 "修复" 方案很 hacky——在每个 layer 加 norm、或者干脆只用一个特定的 learning rate（2e-4）。

**Parse 的分析方法**：

> "If you try to analyze it analytically, it's quite complex — there's attention, GeLU, feed-forward, RoPE. Our insight: let's just look at the residual. How is this activation changing from block to block?"

**第一步：经验观察**——每个 residual block 对向量的改动其实不大。

**第二步：建立动力系统模型**。将 Transformer 的复杂非线性（attention + FFN）全部放入一个黑箱 `r`，剩下的是 `A` 和 `B` 矩阵：

$$x_{t+1} = A x_t + B u + r(x_t)$$

其中 `A` 控制残差在每次循环中如何变换，`B` 控制初始向量的注入。

**第三步：观察 A 矩阵的主导性**。经验上 A 和 B 主导了激活值的量级——于是可以丢掉 r，得到闭式解。

**第四步：分析谱半径（Spectral Radius）**：

> "This is dominated by the A matrix that you are powering up to a large degree. If this matrix learns to be something like 2, and t is like 16, you've blown up the activation to 2^16. This starts to explain those big loss spikes."

**先前方法的分类**：
- 不做约束 → **不稳定**（unstable）
- 加 norm（LayerNorm）→ **边缘稳定**（marginally stable）：模型想 expand，norm 想压缩——"two pressures fighting against each other, manifesting in loss spikes"

**Parse 的解法**：

> "What if we just constrain A and B such that they're not going to explode?"

1. **A → 负对角矩阵**：幂次上升时趋近于 0，不爆炸
2. **B → 加简单的线性 norm**：B 只作用一次，不会因幂次而爆炸
3. 结果：谱半径 < 1 → **严格稳定的系统**

**效果**：

> "Even with the 6e-4 learning rate that was so bad for other models, you actually got a stable model. Parse outperforms previous loop models and also outperforms strong transformer baselines."

### 4.3 循环的 Scaling Laws

> "A few years ago, folks started asking: should I make models bigger or train on more data? The answer was 'down and to the right' — scale both. Our question: where does recurrence fit into this?"

**Parse 的 Scaling Law 发现**：

- IsoFLOP 实验：固定参数量，改变数据量和循环次数
- 结果曲线也呈现 "down and to the right" 趋势 → **随着数据量增加，应该同时增加循环次数**

> "If you're going to fix your model size and increase the amount of data, you should also be increasing the recurrence. As far as I know, all of our models today have no recurrence in them — they're all at the very left of these curves, and they all have a ton of data. This suggests there might be something slightly better we could be doing."

**IsoFLOP 对比**（相同模型大小和 FLOP 预算）：

> "When you get to that number of FLOPs by increasing recurrences as well as just increasing data, you start to get smaller validation losses. It might be the case that we should be looping all of our big pre-training runs."

**关于 pre-trained model 循环的有趣发现**：

> "There was a troll blog post where someone looped like 2-3 layers in a Qwen model and just saw that on some math things it started having higher quality. Which is really weird — it kind of disturbs me, like I don't know why that would possibly be the case."

**推理侧的优势**：

> "One of the big bottlenecks to serving inference efficiently actually ends up being GPU memory. If you have fewer parameters, you can fit more KV cache or do less communication. If you could make the recurrent block small enough, you could write a little mega kernel to just do that recurrent in a very fast loop."

与下一代的 LPU/Groq chips（~250MB 片上内存）的结合潜力很大——"maybe you can design something that will actually fit into them and just keep your weights in memory the whole time."

---

## 5. 总结：全栈创新

> "Hopefully I've given you a sense that if you understand inference, if you understand GPU kernels, you can really start to enable full stack innovation in machine learning algorithms."

**三个层次的全栈创新**：

| 层次 | 工作 | 成效 |
|------|------|------|
| **路由/调度** | Cache-aware prefill/decode disaggregation | 40% 加速（两行代码） |
| **GPU Kernels** | Mega Kernels（Thunderkittens） | 72% 带宽利用率，near speed-of-light |
| **架构设计** | Parse（Loop Transformer） | 更高参数效率 + 稳定训练 + Scaling Laws |

> "Whether that's through a new routing algorithm, new kernels, or new architectures — these are all different pieces of that research problem."

**关于硬件-模型协同设计的建议**：

- 先看目标硬件的内存限制 → 确定模型大小和 KV cache 余量
- 量化格式要与硬件匹配（Nvidia → NV FP4; AMD → MX FP4）
- Agentic 工作流需要大 KV cache → 考虑 MLA 等压缩注意力

> "We're very early in terms of the research and these techniques. In 10-20 years they're going to look back on this and be like 'why are these guys talking about this? Isn't this already obvious?'"

---

## 参考文献与延伸阅读

- [Thunderkittens](https://github.com/HazyResearch/ThunderKittens) — Together/Stanford 的 kernel 编写库
- [Mega-Kernel Decoding (Together, 2025)](https://www.together.ai/blog) — near speed-of-light decode
- [Parse: Loop Transformer (Hayden et al., UCSD, 2025)](https://arxiv.org/abs/2505.XXXXX) — 稳定的循环 Transformer
- [Loop Transformers / Recurrent Depth (Giannou et al., 2023)](https://arxiv.org/abs/2311.01234) — 循环 Transformer 的理论基础
- [DeepSeek MLA (DeepSeek-V2, 2024)](https://arxiv.org/abs/2405.04434) — KV cache 压缩
- [Continuous Batching (Yu et al., 2022)](https://arxiv.org/abs/2205.06448) — 动态批处理推理
- [Splitwise (Jin et al., 2024)](https://arxiv.org/abs/2311.18677) — Prefill/Decode 解耦
- [Cerebras Inference](https://cerebras.ai/) — 专门的 decode 芯片
- [Together AI](https://www.together.ai/) — AI Cloud & Research
