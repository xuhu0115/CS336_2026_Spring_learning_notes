# CS336 Lecture 12: 语言模型评测

> **课程**: Stanford CS336 — Language Models From Scratch (Spring 2026)
> **讲师**: Percy Liang
> **课程网站**: [https://cs336.stanford.edu/](https://cs336.stanford.edu/)
> **课件**: `lecture_12.py` — 394 行交互式 Python 代码

---

## 目录

1. [为什么评测是 deep topic](#1-为什么评测是-deep-topic)
2. [Perplexity：语言模型的"母语"](#2-perplexity语言模型的母语)
3. [考试基准（Exam Benchmarks）](#3-考试基准exam-benchmarks)
4. [对话基准（Chat Benchmarks）](#4-对话基准chat-benchmarks)
5. [Agent 基准（Agentic Benchmarks）](#5-agent-基准agentic-benchmarks)
6. [纯推理基准（Pure Reasoning）](#6-纯推理基准pure-reasoning)
7. [安全评测](#7-安全评测)
8. [生态效度与现实性](#8-生态效度与现实性)
9. [有效性问题：污染与数据质量](#9-有效性问题污染与数据质量)
10. [如何思考评测](#10-如何思考评测)

---

## 1. 为什么评测是 deep topic

> "评测看起来像是一个机械性流程：定义几个 prompt → 发给模型 → 计算 accuracy。但为什么我们需要一整节课？因为**评测塑造了 AI 的发展方向**。它设置 North Star——所有人（open/closed）都盯着评测看。通过定义评测，你 implicitly 决定了模型要往哪个方向进化。"

**核心挑战**：<font color="red">抽象概念（abstract construct）</font> → <font color="blue">具体度量（concrete metric）</font>

> "我想让模型'擅长对话'或'擅长推理'——这些是抽象概念。评测就是把它们变成具体的 prompt 和 metric 的过程。"

**四种"什么是好"的视角**：

![Artificial Analysis: 好 = benchmark 得分高](lecture12_images/artificial-analysis.png)

> "这个网站已经成为'模型 intelligence'的某种标准参考。"

![Artificial Analysis: 好 = 得分高 + 便宜](lecture12_images/artificial-analysis-cost.png)

> "成本和 intelligence 大致相关——但不是完全对齐。"

![Arena AI: 好 = 人喜欢它的回答](lecture12_images/lmarena-leaderboard.png)

> "Chatbot Arena（现在叫 Arena AI）：随机用户给 prompt，看两个 anonymized model 的回复，选哪个更好——纯偏好。"

![OpenRouter: 好 = 人愿意为它付费](lecture12_images/openrouter.png)

> "OpenRouter 作为一个 endpoint 收集了大量 usage statistics——一个经济学的视角：如果人们付钱用它，它就是好的。"

---

## 2. Perplexity：语言模型的"母语"

> "语言模型是一个概率分布 p(x)。Perplexity 就是测 p 对某个 dataset D 分配了多少概率质量。"

$$\text{Perplexity} = \left(\frac{1}{p(D)}\right)^{1/|D|}$$

### 2.1 经典时代：In-distribution Evaluation

传统范式（2010s）：train/test 是**同一分布的** split。

**经典数据集**：Penn Treebank（WSJ）、WikiText-103（Wikipedia）、One Billion Word Benchmark（MT 语料：EuroParl+UN+News）

> "2016 年的经典论文——纯 CNN+LSTM 在 1BW 上把 perplexity 从 51.3 降到 30.0。这是第一次 definitive result——'pure neural 就是 the way to go'。那时生活很简单。"

### 2.2 GPT-2 与 Out-of-Distribution Evaluation

GPT-2（2019）训练在 WebText（40GB，从 Reddit 链接的网页），但在**标准 benchmark 上 zero-shot 评测**。

![GPT-2 Perplexity](lecture12_images/gpt2-perplexity.png)

> "在 PTB（小数据集）上，GPT-2 通过 transfer 大幅超越 SOTA。在 1BW（大数据集）上，没有超过专门训练的模型——但也 impressive 了。这开启了新范式：**在大数据集上训，在标准 benchmark 上评**。"

### 2.3 "Perplexity is all you need"（信仰 vs 科学）

> "有一个非常 fundamental 的论点驱动了很多 scaling 研究：true distribution 是 t，你的模型是 p。Best possible perplexity = H(t)，当且仅当 p=t 时达到。当 p=t 时，你解决了一切——condition on problem → solution。**所以 by pushing down perplexity, we will eventually reach AGI。**"

> "在 GPT-3 之前，没人确定 scaling LM 会有用。是**这个信念**——'如果你把 perplexity 推下去，好事会发生'——推动了早期的 scaling。"

### 2.4 Perplexity 也许太多了

> "*(Stanford was founded in 1885)*——perplexity 对所有 token 都 penalize。'1885' 是一个很好的 token（Q&A disguised as sentence completion），但 'founded' 或第一个词可能不重要。Perplexity 不区分。"

**解决方案**：Conditional perplexity = p(response | prompt)^(1/|response|)——只在意你关心的那部分 token。

### 2.5 有些 benchmark 其实是 Perplexity

- **LAMBADA**：填空（cloze task），需要 long-context dependency 才能解析
- **HellaSwag**：多选句完成——"在某种意义上就是 perplexity"

![LAMBADA](lecture12_images/lambada.png)
![HellaSwag](lecture12_images/hellaswag.png)

### 2.6 Perplexity Leaderboard 的陷阱

> "如果你办 perplexity leaderboard，人们提交 LM，你算 logprob——你需要**信任**他们返回的 logprob 是 valid 的（sums to 1）。否则我可以 `return 0`——完美 perplexity。而对 downstream tasks——给 prompt 返回 response 再算 accuracy，这个 contract 是**可直接验证的**。"

---

## 3. 考试基准（Exam Benchmarks）

> "考试是测试人类的传统方式，同样适用于 LM。好处：**精确控制** subject + difficulty；**无歧义答案**，易于评分。很多 LM benchmarking culture 是建立在考试之上的。"

### 3.1 MMLU（2021）

![MMLU](lecture12_images/mmlu.png)

- 57 个学科（数学、历史、法律、道德等），多项选择
- "由研究生和本科生从网上的 free source 收集的"
- **Despite the name**："这不测 'language understanding'——它测的是 knowledge 和 reasoning"
- 用 GPT-3 以 few-shot prompting 评测
- "当时**非常 radical**——你构建复杂的 question + in-context examples，期望 LM 做 reasonable thing"
- 现在已基本饱和（~90%+）——"benchmark 的 shelf life 不太长"

### 3.2 MMLU-Pro（2024）

![MMLU-Pro](lecture12_images/mmlu-pro.png)

> "OK, MMLU 太容易了。Let's make it harder：去掉 noisy/trivial 问题、4 选 1 变 10 选 1、加 chain-of-thought。"

- 准确率下降 16-33%
- 现在又回到了 ~88%

### 3.3 GPQA：Google-Proof Q&A

![GPQA](lecture12_images/gpqa.png)

> "如果你能 Google 到答案，LM 也极可能见过。所以必须是 Google 解决不了的问题。"

- 61 位 PhD 合同工从 Upwork 写题
- Expert validation → revision → second expert review
- **DIAMOND set**：两个专家同意 + 至少一个非专家（有 Google）答错
- PhD 专家 65% accuracy；非专家有 Google 30 分钟仅 ~34%
- "GPT-4 当时 39%——现在 >90%"

### 3.4 Humanity's Last Exam（HLE，2025）

![HLE Examples](lecture12_images/hle-examples.png)
![HLE Pipeline](lecture12_images/hle-pipeline.png)
![HLE Results](lecture12_images/hle-results.png)

> "名字有点 ominous。$500K 奖金池 + co-authorship 激励出题。多阶段 review + frontier model 过滤。**Private holdout set** 应对 contamination。"

- 2025 年单个位数正确率；2026 年最好模型（Mythos）64.7%——"still has some room"

### 考试基准小结

> "趋势很清楚：模型越好→ benchmark 越难→ 模型又追上。**多选格式**本身不是问题——你可以出极难的 multiple-choice。问题在于多选**限制了你能问的问题类型**。更大的问题是：**这根本不反映真实使用**——没人会问 HLE 问题，除非在做 HLE evaluation。"

---

## 4. 对话基准（Chat Benchmarks）

> "大多数人不用多选考试题对话 AI。他们问开放式问题——'我想做甜菜沙拉配山羊奶酪，什么香料合适？' 这种没有标准答案的。"

### 4.1 Chatbot Arena（现在 Arena AI）

![Arena Beets Example](lecture12_images/arena-beets.png)

**数据收集机制**：随机用户给 prompt → 两个 anonymized model 各自答 → 用户选哪个更好

**ELO Ranking**：p(A beats B) = 1 / (1 + 10^((ELO_B - ELO_A)/400))，从 pairwise comparison 拟合

> "Clever idea——不需要把同一个 prompt 发给所有模型（因为 human 在 rating）。**动态**：随时加入新 prompt 和新模型。"

![Arena Leaderboard](lecture12_images/lmarena-leaderboard.png)

**优点**：真实世界 prompt、free for users（有 incentive 真正使用）
**问题**：这些人是谁？有 bias 吗？spammer？binary preference 混淆了 style 和 correctness；人类自己怎么判断 correctness？容易产生 sycophancy。

### 4.2 AlpacaEval

![AlpacaEval](lecture12_images/alpacaeval-leaderboard.png)

- 805 条 instruction，GPT-4 当 judge 算 win rate
- "问题：LLM judge 有**长度偏好**——models 开始 game the leaderboard by saying more。"
- Alpaca Eval 2.0：用回归 debias
- **与 Chatbot Arena（human）有很高的相关性**——这是 metric 的 evaluation

### 4.3 WildBench

![WildBench](lecture12_images/wildbench.png)

- 从 1M 人机对话中精选 1024 例
- GPT-4 Turbo 用 checklist（类似 CoT for judging）评分
- 与 Chatbot Arena 相关性好——"似乎成了 de facto sanity check"

---

## 5. Agent 基准（Agentic Benchmarks）

> "之前我们在测 LM '说什么'。现在测 LM '做什么'。Agent = LM + agent scaffold（决定如何使用 LM 的逻辑）。"

### 5.1 SWE-Bench

![SWE-Bench](lecture12_images/swebench.png)

- 2294 个 task，12 个 Python 仓库
- 给定 codebase + issue → submit PR
- 评测：**unit tests**——"一个很干净的 metric"

### 5.2 TerminalBench

![TerminalBench](lecture12_images/terminal-bench.png)
![TerminalBench Results](lecture12_images/terminal-bench-results.png)

- 229 tasks，crowdsourced from 93 contributors
- Terminal environment：简单且 universal

### 5.3 CyBench

![CyBench](lecture12_images/cybench.png)

- 40 个 CTF 任务
- **First-solve time** 作为难度度量——"来自安全竞赛的真实经验"

### 5.4 MLEBench

![MLEBench](lecture12_images/mlebench.png)

- 75 Kaggle competitions——需要训练模型、处理数据

### Agent Scaffolds

> "Agent scaffold **非常重要**。Explicit planning（todo list）、hierarchical delegation（agent 调 sub-agent）、persistent memory（读写文件）、extreme context engineering。"

**核心**："评测 agent = 评测 agent scaffold + 评测 LM。"

---

## 6. 纯推理基准（Pure Reasoning）

> "至此所有任务都需要语言和世界知识。**能否把推理和知识分离开？** 推理是更纯粹的智能形式——不是记忆事实。"

### ARC-AGI

![ARC-AGI-1](https://arcprize.org/media/images/arc-task-grids.jpg)

- **100% 人类可解，但 AI 挑战极大**
- 每个 task 唯一——"memorization 帮不上忙"
- **ARC-AGI-1**（2019）：Pretrained LM 无法推动进展
- **ARC-AGI-2**（2025.3）：更多 multi-step reasoning
- **ARC-AGI-3**（2026.3）：**Interactive environment**

![ARC Results](lecture12_images/arc-agi-results.png)
![ARC-AGI-3](lecture12_images/arc-agi-3.png)
![ARC-AGI-3 Results](lecture12_images/arc-agi-3-results.png)

> "Pretrained LM 没推动 needle。o1/o3 这类 reasoning models 才开始起飞。这个 benchmark 清晰暴露了当前模型的 gaps。"

---

## 7. 安全评测

> "安全有很多层面，非常 context-dependent——政治、法律、社会规范在不同国家/文化中完全不同。风险也很多样——hallucinations, sycophancy, abetting crimes, inequality, losing critical thinking。"

- **HarmBench**：510 个有害行为
- **AIR-Bench**：基于法规框架 + 公司政策，314 个风险类别、5694 个 prompt
- **Jailbreaking**：GCG（Greedy Coordinate Gradient）自动优化 prompt 绕过安全训练——"从开源模型（Llama）的 attack **迁移**到闭源模型（GPT-4）"
- **Dual-use**：强大的 cybersecurity agent 既可以用于渗透测试（合法），也可以用于实际攻击

![GCG Examples](lecture12_images/gcg-examples.png)

---

## 8. 生态效度与现实性

> "Ecological validity：评测在多大程度上捕捉了真实世界的使用？Exam benchmarks 离真实使用很远。Chatbot Arena 的 prompt 是真实的——但 distribution 不可控。"

- **GDPVal**（OpenAI）：按美国 GDP 的 top 9 行业、44 个职业，"tasks from professionals with ~14 years of experience"
- **MedHELM**：121 clinical tasks from 29 clinicians——"以前的 medical benchmarks 是标准化考试，这不是医生真正做的事"
- **Clio**（Anthropic）：用 LM 分析真实用户数据→"understanding what people are actually asking"

![GDPVal](lecture12_images/gdpval.png)
![Clio](lecture12_images/clio-table4.png)

> "Unfortunately，realism 和 privacy 有时是矛盾的。"

---

## 9. 有效性问题：污染与数据质量

### 9.1 Train-Test Overlap（污染）

> "ML 101：不要在 test set 上训练。但今天的数据是从 Internet 上 scrape 的——没人告诉你 training data 里有什么。"

**四种应对方式**：

| 方式 | 说明 |
|------|------|
| **Route 1: 推断** | 利用 data point 的 exchangeability 从模型内部推断是否见过 |
| **Route 2: 报告规范** | 模型提供商应**主动报告** train-test overlap |
| **Route 3: Fresh evals** | LiveCodeBench、UncheatableEval：抓取新网页做 benchmark |
| **Route 4: Private evals** | 公司内部 code base、个人写作——绝不可能在互联网上 |

> "Contamination 比你想象的 subtle——不是 literally train on test set，而是 test question 的 source material 很可能在训练数据中。"

### 9.2 数据集质量

- SWE-Bench → SWE-Bench Verified（去除有问题的测试用例）
- "Insufficient test cases——trivial agent can solve task"
- **Docent**：用 LLM inspect agent trace 来检测 benchmark 的问题

---

## 10. 如何思考评测

> "**没有唯一的正确评测**——取决于你想回答什么问题。"

| 视角 | 问题 |
|------|------|
| 用户/公司 | "Model A 还是 Model B 适合我的 use case？" |
| 研究者 | "模型的 raw capability 是什么？" |
| 政策/商业 | "模型的 benefits + harms 是什么？" |
| 模型开发者 | "如何从评测中获取 feedback 改进模型？" |

### 我们到底在评测什么？

> "Foundation model 之前——我们评测的是 **methods**（标准化的 train/test split）。今天——我们评测的是 **models/systems**（anything goes）。**Methods evaluation 鼓励算法创新。Models evaluation 对下游用户有用。**"

**两种 evaluation 规则的清晰对比**：Karpathy 的 NanoGPT speedrun——"固定 data + compute，比谁更快达到 target validation loss。这种 **method evaluation** 非常干净，但今天已经很少见了。"

**最终建议**："不管你选哪种 evaluation——**清楚定义游戏规则**。"

---

## 总结

Percy 的五个 takeaway：

1. **没有唯一的正确评测**——选择取决于你 measuring 什么
2. **Perplexity 仍然大量用于 development**（smooth scaling laws），但不是 real-world use 的替代品
3. **Exams benchmarks 有 shelf life**——模型持续饱和，benchmark 持续变难
4. **Chat + Agent benchmarks 更贴近真实使用**——但评测本身更难
5. **Define the rules of the game clearly**——methods vs models vs agents

---

## 参考文献与延伸阅读

- [MMLU (Hendrycks et al., 2021)](https://arxiv.org/abs/2009.03300)
- [MMLU-Pro (Wang et al., 2024)](https://arxiv.org/abs/2406.01574)
- [GPQA (Rein et al., 2023)](https://arxiv.org/abs/2311.12022)
- [HLE (Phan et al., 2025)](https://arxiv.org/abs/2501.14249)
- [Chatbot Arena (Chiang et al., 2024)](https://arxiv.org/abs/2403.04132)
- [AlpacaEval (Dubois et al., 2024)](https://arxiv.org/abs/2404.04475)
- [WildBench (Lin et al., 2024)](https://arxiv.org/abs/2406.04770)
- [SWE-Bench (Jimenez et al., 2024)](https://arxiv.org/abs/2310.06770)
- [TerminalBench](https://arxiv.org/abs/2601.11868)
- [CyBench (Zhang et al., 2024)](https://arxiv.org/abs/2408.08926)
- [MLEBench (Chan et al., 2024)](https://arxiv.org/abs/2410.07095)
- [ARC-AGI](https://arcprize.org/)
- [HarmBench (Mazeika et al., 2024)](https://arxiv.org/abs/2402.04249)
- [AIR-Bench (Zeng et al., 2024)](https://arxiv.org/abs/2407.17436)
- [GCG Jailbreaking (Zou et al., 2023)](https://arxiv.org/abs/2307.15043)
- [HELM](https://crfm.stanford.edu/helm/) — Stanford CRFM 的标准化评测平台
- [CS336 Course Website](https://cs336.stanford.edu/)
