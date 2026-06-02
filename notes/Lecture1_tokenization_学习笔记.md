# CS336 Lecture 1: 分词（Tokenization）

> **课程**: Stanford CS336 — Language Models From Scratch (Spring 2026)
> **讲师**: Percy Liang, Tatsu Hashimoto, Marcel, Herman, Steven
> **视频**: [YouTube Playlist (Spring 2026)](https://www.youtube.com/watch?v=JuoVZkPBiKk)
> **课程网站**: [https://cs336.stanford.edu/](https://cs336.stanford.edu/)
>
> 本笔记为 Lecture 1 的分词专题部分。课程概览部分见 [Lecture1_overview_学习笔记](Lecture1_overview_学习笔记.md)。

---

## 目录

1. [什么是 Tokenization？](#1-什么是-tokenization)
2. [四种 Tokenizer 方案的对比](#2-四种-tokenizer-方案的对比)
3. [BPE 分词算法详解](#3-bpe-分词算法详解)
4. [为什么需要 Tokenization？（效率视角）](#4-为什么需要-tokenization效率视角)
5. [总结与展望](#5-总结与展望)

---

## 1. 什么是 Tokenization？

> 本单元受 [Andrej Karpathy 的 Tokenization 视频](https://www.youtube.com/watch?v=zduSFxRajkE) 启发，强烈推荐观看。

### 1.1 核心问题

**模型操作的"原子"是什么？** 语言模型不能直接处理原始文本字符串，需要将文本转换为数字序列。

### 1.2 形式化定义

Tokenizer 是一个在**原始输入（字节）** 和**整数序列（token）** 之间互相转换的模块。

```
字符串 "Hello, 🌍! 你好!"
   ↓ encode()
整数序列 [15496, 11, 995, 0, ...]
   ↓ decode()
字符串 "Hello, 🌍! 你好!"
```

Tokenizer 必须满足 **round-trip** 属性：`decode(encode(s)) == s`

> Percy："如果你实现的 tokenizer 不能 round-trip，那你就有 bug。"

### 1.3 Tokenizer 的直观特性

使用 [tiktokenizer](https://tiktokenizer.vercel.app/?encoder=gpt2) 交互式网站可以直观感受。以 GPT-2/4/5 的分词器为例，一些有趣的观察：

- 单词常常和**它前面的空格合并**为一个 token（如 `" world"` 是一个 token，`"world"` 是另一个）
- 开头出现的 `"hello"` 和中间出现的 `" hello"` 被编码为**完全不同的 token**（两个 token 之间没有任何语义联系）
- 数字每几个数字被当作一个 token，不同 tokenizer 的策略不同
- 这就是为什么 tokenizer **很烦人**，研究人员想摆脱它们

### 1.4 以 GPT-5 (o200k_base) Tokenizer 为例

```python
import tiktoken
tokenizer = tiktoken.get_encoding("o200k_base")

string = "Hello, 🌍! 你好!"
indices = tokenizer.encode(string)  # [15496, 11, 995, 0, ...]
reconstructed = tokenizer.decode(indices)  # "Hello, 🌍! 你好!"
assert string == reconstructed

# Compression Ratio = num_bytes / num_tokens
# 字符串 20 字节, 8 个 token → 2.5 bytes/token
```

**压缩率（Compression Ratio）** = 字节数 / token 数

- 更**大**的压缩率 → 更短的序列 → 更好（因为 Attention 是 $O(n^2)$ 的）
- 增大词表可以提升压缩率，但会导致**稀疏性**问题：每个 token 都是独立实体，词汇越多越难学习
- 当前主流多语言 tokenizer 的词表大小：**100K ~ 200K**

---

## 2. 四种 Tokenizer 方案的对比

### 2.1 Character-Level Tokenizer（字符级）

```python
# 每个 Unicode 字符 → ord() 得到整数
"Hello" → [72, 101, 108, 108, 111]
```

**原理**：每个 Unicode 字符对应一个整数（code point），直接用 `ord()` 编码，`chr()` 解码。

| 优点 | 缺点 |
|------|------|
| 实现简单 | 词表大（Unicode 约 150K 字符） |
| 无损 | 很多字符极罕见（如 🌍），词表利用率低 |
| | 压缩率差（一个字符 = 一个 token） |

> "This tokenizer is the worst of both worlds (large vocabulary, low compression ratio)."
> — Percy 的评价：两头不讨好

### 2.2 Byte-Level Tokenizer（字节级）

```python
# 字符串 → UTF-8 字节序列 → 每个字节是一个 token (0~255)
"a" → b"a" → [97]
"🌍" → b"\xf0\x9f\x8c\x8d" → [240, 159, 140, 141]
```

**原理**：将 Unicode 字符串编码为 UTF-8 字节序列，每个字节（0~255）是一个 token。

| 优点 | 缺点 |
|------|------|
| 词表小且固定（256） | 压缩率 = 1（1 字节 = 1 token） |
| 永远不会有 UNK token | 序列极长 → Attention 计算量巨大 |

> 由于 Transformer 的 Attention 是 $O(n^2)$ 的，上下文长度有限，字节级 tokenizer 在目前的架构下 compute-inefficient。

### 2.3 Word-Level Tokenizer（词级）

```python
# 按正则表达式切分成"词"
"I'll say supercalifragilisticexpialidocious!"
→ ["I", "'", "ll", " ", "say", " ", "supercalifragilisticexpialidocious", "!"]
```

**原理**：使用正则表达式（如 `\w+|.`）切分字符串，每个 chunk 是一个 token。

| 优点 | 缺点 |
|------|------|
| token 有意义（词是人发明的语义单元） | 词表可以无限大（测试时可能遇到未见过的词） |
| 压缩率不错 | 大量罕见词 → 模型学不到有用的表示 |
| | 需要 UNK token 处理未知词 → "ugly and can mess up perplexity" |

> 经典 NLP 的做法。虽然后来被 BPE 取代，但理解它的缺陷有助于理解 BPE 为什么好。

### 2.4 三种方案的关键指标对比

| Tokenizer | Vocab Size | Compression Ratio | 核心问题 |
|-----------|:---:|:---:|------|
| Character | ~150K | 低 | 词表大 + 压缩率差 |
| Byte | 256 | = 1 | 序列太长，Attention 吃不消 |
| Word | 无界 | 不错 | 词表无界 + UNK 问题 |

---

## 3. BPE 分词算法详解

### 3.1 BPE 的历史

BPE（Byte-Pair Encoding）算法最早由 **Philip Gage 在 1994 年** 提出，用于**数据压缩** [[article]](http://www.pennelynn.com/Documents/CUJ/HTML/94HTML/19940045.HTM)。

后来被 Sennrich 等人 [[sennrich_2016]](https://arxiv.org/abs/1508.07909) **引入 NLP 领域**，用于神经机器翻译（在那之前，大家都用基于词的分词）。

**GPT-2** [[gpt2_2019]](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) 是第一个使用 BPE 的 LLM。

### 3.2 核心思想

> **训练阶段**：在原始文本上"训练"分词器，构建一个**针对数据定制的词汇表**。
>
> **直觉**：常见字节序列 → 一个 token；罕见序列 → 多个 token 拼出来。**永远不会遇到 UNK**。

### 3.3 算法步骤

**训练过程**（以 `"the cat in the hat"` 为例，做 3 次 merge）：

#### Step 0：初始化
```
输入文本: "the cat in the hat"
编码为 UTF-8 字节序列:
[116, 104, 101, 32, 99, 97, 116, 32, 105, 110, 32, 116, 104, 101, 32, 104, 97, 116]
                          ↑ 出现两次(位置0-1和位置11-12)
初始词表: {0: b'\x00', 1: b'\x01', ..., 255: b'\xff'} (256 个单字节 token)
初始 merges: {} (空)
```

#### 第 1 次 Merge (i=0)
```
统计所有相邻 pair 的出现次数:
(116, 104) → 2 次  ← 最多！("th" 序列)
(104, 101) → 2 次  ← 也是 2 次，取第一个
...

选中最频繁的 pair: (116, 104) = ('t', 'h')
新 token index: 256
merges[(116, 104)] = 256
vocab[256] = b't' + b'h' = b'th'

更新序列 → 将所有 (116, 104) 替换为 256:
[256, 101, 32, 99, 97, 116, 32, 105, 110, 32, 256, 101, 32, 104, 97, 116]
         ↑ 第0-1位置合并               ↑ 原第11-12位置也合并
```

#### 第 2 次 Merge (i=1)
```
统计: (256, 101) 即 ("th", "e") → 2 次 ← 最多！
新 token: 257
merges[(256, 101)] = 257
vocab[257] = b'th' + b'e' = b'the'

替换后序列:
[257, 32, 99, 97, 116, 32, 105, 110, 32, 257, 32, 104, 97, 116]
```

#### 第 3 次 Merge (i=2)
```
统计: (257, 32) 即 ("the", " ") → 2 次
新 token: 258
merges[(257, 32)] = 258
vocab[258] = b'the' + b' ' = b'the '

替换后序列:
[258, 99, 97, 116, 32, 105, 110, 32, 258, 104, 97, 116]
```

**最终结果**：
```
merges = {
    (116, 104): 256,    # t + h → th
    (256, 101): 257,    # th + e → the
    (257, 32): 258,     # the + ' ' → "the "
}
vocab = {
    ...0~255: 原始字节...,
    256: b'th',
    257: b'the',
    258: b'the '
}
压缩率: 18 bytes / 12 tokens = 1.5
```

可以发现序列在逐渐变短，词汇表在逐渐增大。在实际训练中，通常会做数万次 merge（词表大小 - 256）。

### 3.4 使用训练好的 Tokenizer

**Encode**（编码新文本）：
```python
# 新文本: "the quick brown fox"
# 1. 先转成字节: [116, 104, 101, 32, ...]
# 2. 按 merges 顺序应用每次合并
# 3. 输出: [258, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 32, 102, 111, 120]
```

**Decode**（解码回文本）：
```python
# 输入: [258, 113, 117, ...]
# 1. 查 vocab: 258 → b'the ', 113 → b'q', ...
# 2. 拼接: b'the quick brown fox'
# 3. UTF-8 解码: "the quick brown fox"
```

### 3.5 Assignment 1 中的改进点

Percy 特别指出，课程中展示的 BPE 实现是正确的，但**极慢**。在作业中需要做以下优化：

1. **encode() 加速**：目前 encode 遍历了所有 merge（词表 - 256 个），但实际上只有少量 merge 适用于当前文本。需要建立索引，只遍历相关的 merge
2. **特殊 token 处理**：检测并保留特殊 token（如 `<|endoftext|>`），概念不深但构建现代 tokenizer 必不可少
3. **预分词（Pre-tokenization）**：先在文本上用正则表达式切分成 chunk（如 GPT-2 的分词正则），然后再在每个 chunk 上应用 BPE。这样更快
4. **速度优化**：Python 可能不够快，如果你用 Rust 或 C 实现，完全可以

> 现代 BPE tokenizer 的 encode 实现通常会构建高效的前缀树（trie）或哈希表结构，使得 encode 可以在接近 $O(n)$ 时间内完成。

---

## 4. 为什么需要 Tokenization？（效率视角）

### 4.1 两个存在理由

Percy 从效率角度解释了 tokenization 存在的两个理由：

1. **减少上下文长度**：1000 字节 → ~250 token，降低 Attention 的计算量
2. **自适应计算**：常见序列（如 "the"）用 1 个 token 表示，直接跳过；罕见/有趣的部分用多个 token，分配更多模型容量（modeling capacity）来仔细处理

### 4.2 Tokenizer-Free 的未来？

> "The dream is tokenizer-free architectures that directly operate on bytes."（直接的字节级模型是梦想）

相关尝试：
- ByT5 [[byt5_2021]](https://arxiv.org/abs/2105.13626)
- MegaByte [[megabyte_2023]](https://arxiv.org/abs/2305.07185)
- BLT [[blt_2024]](https://arxiv.org/abs/2406.12350)
- TFree [[tfree_2024]](https://arxiv.org/abs/2406.12351)
- HNet [[hnet_2025]](https://arxiv.org/abs/2504.00690)

但这些方法尚未在前沿模型中得到大规模验证，目前我们仍然需要学习 tokenization。

### 4.3 即使未来淘汰 Tokenization，什么原则不会变？

Percy 认为即使出现替代 tokenization 的方案，以下两个原则必须成立：

1. **模型需要在序列的"抽象块"（chunks）上操作**：这在文本之外的模态更明显。视频或 DNA 序列中，单个字节/像素的信噪比极低，必须先提升到合适的抽象层次才能建模
2. **块应该是可变长度的（variable-size chunks）**：实现自适应计算，不是所有输入都同等对待。如果不这样做，"you're going to be suboptimal"

---

## 5. 总结与展望

### 5.1 本讲总结

- **Tokenization** 是模型与原始输入之间的桥梁：strings ↔ tokens (indices)

![分词示例](../lectures/images/tokenized-example.png)
- Character/Byte/Word 三种基础方案各有严重缺陷
- **BPE** 是数据驱动的有效启发式方法：常见序列压缩为一个 token，罕见序列拆分为多个 token
- Tokenization 是一个**独立于模型训练的预处理步骤**。也许有一天会被端到端的字节级方案取代，但目前仍是实际使用中的标准方法

### 5.2 BPE 关键指标速查

| 维度 | 说明 |
|------|------|
| **词表大小** | 256（初始字节）+ num_merges（训练时设定） |
| **压缩率** | 通常 2~4 bytes/token |
| **训练算法** | 迭代合并最频繁的相邻 pair |
| **核心优势** | 永远不产生 UNK，自适应数据分布 |
| **主要挑战** | encode 需要高效实现；对稀有序列不友好 |

### 5.3 下讲预告

> **Lecture 2**: Resource Accounting（资源核算），理解模型的 FLOPs 和 Memory 去哪了。

---

## 参考文献

- [Sennrich et al., 2016 — Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — 将 BPE 引入 NLP
- [Gage, 1994 — A New Algorithm for Data Compression](http://www.pennelynn.com/Documents/CUJ/HTML/94HTML/19940045.HTM) — BPE 原始论文
- [GPT-2 (2019)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — 首个使用 BPE 的 LLM
- [Andrej Karpathy — Tokenization 视频](https://www.youtube.com/watch?v=zduSFxRajkE) — 强烈推荐的 Tokenization 入门视频
- [tiktokenizer 交互式网站](https://tiktokenizer.vercel.app/?encoder=gpt2) — 在线体验不同 tokenizer
- [ByT5 (2021)](https://arxiv.org/abs/2105.13626) — 字节级模型的早期探索
- [BLT (2024)](https://arxiv.org/abs/2406.12350) — Tokenizer-free 架构
- [CS336 Course Website](https://cs336.stanford.edu/)

---

*笔记整理自：CS336 课程讲义 (lecture_01.py) + 课堂视频字幕 + 相关背景知识补充*
