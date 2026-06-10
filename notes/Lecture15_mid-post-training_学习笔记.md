# CS336 Lecture 15: Mid/Post-Training — SFT, RLHF 与数据 Pipeline

> **课程**: Stanford CS336 — Language Models From Scratch (Spring 2026)
> **讲师**: Tatsu Hashimoto
> **课程网站**: [https://cs336.stanford.edu/](https://cs336.stanford.edu/)
> **课件**: `lecture_15.pdf` — 65 页
> **前置**: Lecture 13-14 讲了数据来源与处理；本讲聚焦 post-training 的 SFT 数据、RLHF 数据与算法
> **后续**: Lecture 16 — RLVR 与推理模型

---

## 目录

1. [引言：从 GPT-3 到 ChatGPT](#1-引言从-gpt-3-到-chatgpt)
2. [SFT 数据的演进史](#2-sft-数据的演进史)
   - [2.1 FLAN：多任务奠基](#21-flan多任务奠基)
   - [2.2 Alpaca/Vicuna：蒸馏路线](#22-alpacavicuna蒸馏路线)
   - [2.3 Open Assistant：众包人类数据](#23-open-assistant众包人类数据)
   - [2.4 现代 SFT：Agentic & Tool Use](#24-现代-sftagentic--tool-use)
   - [2.5 三大转型总结](#25-三大转型总结)
3. [SFT 数据收集的陷阱](#3-sft-数据收集的陷阱)
   - [3.1 风格 vs 能力](#31-风格-vs-能力)
   - [3.2 知识与幻觉：Tail Knowledge 问题](#32-知识与幻觉tail-knowledge-问题)
   - [3.3 Safety SFT](#33-safety-sft)
   - [3.4 SFT 数据的 Scale：惊人地少](#34-sft-数据的-scale惊人地少)
4. [Mid-Training / Annealing](#4-mid-training--annealing)
5. [RLHF 引言](#5-rlhf-引言)
   - [5.1 概念区分：生成建模 vs 奖励最大化](#51-概念区分生成建模-vs-奖励最大化)
   - [5.2 为什么需要 RLHF](#52-为什么需要-rlhf)
   - [5.3 RLHF Pipeline 总览](#53-rlhf-pipeline-总览)
6. [RLHF 数据收集](#6-rlhf-数据收集)
   - [6.1 偏好数据收集](#61-偏好数据收集)
   - [6.2 标注员画像](#62-标注员画像)
   - [6.3 标注员偏见与模型行为](#63-标注员偏见与模型行为)
   - [6.4 Model-Based Annotations：AI 反馈](#64-model-based-annotationsai-反馈)
7. [RLHF 算法](#7-rlhf-算法)
   - [7.1 PPO：从 Policy Gradient 到实用算法](#71-ppo从-policy-gradient-到实用算法)
   - [7.2 DPO：消除 Reward Model 的优雅方案](#72-dpo消除-reward-model-的优雅方案)
   - [7.3 DPO 变体与争议](#73-dpo-变体与争议)
8. [RLHF 的陷阱](#8-rlhf-的陷阱)
9. [总结](#9-总结)

---

## 1. 引言：从 GPT-3 到 ChatGPT

> "We're going to move away from pre-training now. I think we'll get into some of the messier parts of language modeling."
>
> "If your first exposure to these systems was ChatGPT, it was kind of amazing. If you go back and interact with GPT-3, you'll be like, what is this thing?"

**本讲的定位**：

- **Pre-training** → 得到一个强大的 base model（GPT-3 级别）—— 可以续写文本，但无法可靠遵循指令
- **Post-training（本讲）** → 让模型学会遵循指令、变得像 ChatGPT
- **Next lecture** → 从 ChatGPT 到 GPT-o1（推理/thinking 模型）

![post-training 演进路线](lecture15_images/page_02_0.jpeg)

> "Pre-training is critical because it scales in this very diverse, broad way. But once we have pre-training, we have to then extract the kinds of behaviors that we want out of this primordial soup."

**Post-training 的两阶段框架**（RLHF 论文的标准配方）：

```
SFT (Supervised Fine-Tuning) → RLHF (Reinforcement Learning from Human Feedback)
1. 收集 demonstration 数据 → SFT       2. 收集偏好数据 → 训练 reward model → RL 优化
```

> "Algorithms are not really the secret sauce. It's not where a lot of the leverage is. It's going to be the data."

**关于前沿 post-training 的信息透明度**：

> "Information about frontier post-training is honestly pretty sparse. A lot of the materials I'm going to reference today are actually pretty old — before the competition from ChatGPT started to heat up. Now that competition has heated up, basically none of the vendors want to release any information about their post-training processes. The data is very much a trade secret."

Tatsu 举例说 Scale AI 的内部文件曾泄露，显示他们试图逆向工程 GPT-4 为什么更好，让标注员产出 "more detailed and better than GPT-4" 的回答。RLHF 论文（如 InstructGPT、Anthropic HH）的附录非常详细，是了解数据收集流程的最好窗口。

---

## 2. SFT 数据的演进史

> "SFT is the part that's really basically all about data. We all know how to SFT a model — that's basically exactly the same as pre-training. The only real difference is going to be the training data."

![SFT 数据演变全景](lecture15_images/page_03_0.png)

### 2.1 FLAN：多任务奠基

> "FLAN is very much a forward-looking dataset for its time. It established the idea of doing multitask post-training."

FLAN（Google，T5 时代）的核心思想：NLP 社区已经收集了大量 supervised 数据集 → 全部拿来做 instruction tuning。

**但存在问题**：

![FLAN 示例](lecture15_images/page_05_0.png)

- **不自然的格式**：指令放在最后（"write a subject line for this email"），输入格式来自原始 NLP 数据集的结构
- **质量低**：摘要来自 CNN/Daily Mail 等数据集，"summaries are often hallucinated — there's a bunch of details that are not in the inputs"
- **输出太短**：和 ChatGPT 的输出完全不是一个风格

> "FLAN is generated from existing datasets. The original NLP datasets from which FLAN was built were not the highest of quality. When you train on this thing, you also inherit a whole bunch of deficiencies."

![FLAN 问题总结](lecture15_images/page_06_0.png)

**关键认知转变**：FLAN 时代的假设是 post-training 也像 pre-training 一样需要 scale —— 大量数据、大量任务。但后来的实践表明：

> "If you have a sufficiently strong, big, pre-trained model, you can actually get away with very few high quality examples because pre-training generalization is going to get you quite a bit of the way."

### 2.2 Alpaca/Vicuna：蒸馏路线

ChatGPT 出现后，Tatsu 的学生做了 **Alpaca**：从 ChatGPT 蒸馏出 input-output pairs。

![Alpaca 示例](lecture15_images/page_07_0.png)

> "You get more natural-looking inputs, longer chatty outputs. Because of course, it's taken from ChatGPT. These kinds of examples reliably induced ChatGPT-like behavior on models — only when we did it to the original LLaMA models."

**关键发现**：pre-training 基础 + chat-style 数据 = ChatGPT 类似行为。这引发了开源社区的乐观情绪——"if only we could collect a sufficiently high quality and large instruction-tuning dataset, we could catch up to the closed source labs."

![从 FLAN 到 Alpaca 的转变](lecture15_images/page_07_1.png)

**Vicuna**（Berkeley）：使用在线用户分享的 prompts 作为蒸馏输入。

**Self-Instruct**：用模型自己生成数据。"Why can't we use the model itself to generate data? Models are getting better all the time — they might even be better than some of our annotators."

![SFT 数据全景](lecture15_images/page_09_0.jpeg)

### 2.3 Open Assistant：众包人类数据

> "A bunch of volunteers got together and said, we're going to come up with really hard and interesting prompts, good high quality responses, and do it at scale — in the same way that Wikipedia managed to produce something very high quality."

![Open Assistant 示例](lecture15_images/page_09_1.png)

- 众包志愿者写作 prompts 和 responses
- 约 10,000+ 条高质量示例
- Chat-style 输入 + 详细、专家级别的回答
- "Very admirable, very impressive effort"

但项目后来停滞了（Tatsu 说可以讨论原因）。

### 2.4 现代 SFT：Agentic & Tool Use

> "We've really started to move from just a chat interface to something that's like a full agent system. We don't want just textual responses. We want tool calls. We want to-do lists."

![Nemotron agentic SFT](lecture15_images/page_16_0.png)

**Nemotron**（NVIDIA 开源）：SFT 数据中大量是 agentic 格式——不仅有 assistant 文本回复，还有并行的 tool calls。这是 SFT 数据的未来。

现代 SFT 数据集的代表还包括 **Tulu3**、**WizardLM** 等——"increasingly complicated ways of generating instruction following data using language models."

### 2.5 三大转型总结

> "We've seen big, higher level transitions in how we think about and build these kinds of datasets."

| 转型 | 描述 |
|------|------|
| **Chattiness** | 从 NLP benchmark 风格（input → programmatic output）转向人类对话风格 |
| **Quality Annotators** | 从普通众包工人转向专家标注员（Open Assistant 是典型案例） |
| **Tool Use** | 从纯文本回复转向结构化的 agentic 交互（tool calls, to-do lists） |

![历史演进对比](lecture15_images/page_09_2.jpeg)

---

## 3. SFT 数据收集的陷阱

> "If you've taken classes like 224N, you're going to find some things that are very reminiscent of issues that you see when you start building classic supervised deep learning models from scratch."

### 3.1 风格 vs 能力

> "Length and style variation is a very, very big part of this post-training process. People say Claude has a different tone than ChatGPT. ChatGPT is too chatty. All of these are conscious decisions that are made by the data collection folks."

![不同 SFT 数据集的 response 长度变化](lecture15_images/page_17_0.png)

**核心问题**：风格因素在偏好评估中影响巨大。

> "People will very easily get tricked. They will very often select responses that have bullet pointed lists, or responses that have more and longer detail."

![风格 vs 能力矩阵](lecture15_images/page_18_0.png)

训练在不同 SFT 数据集上，**偏好指标**（AlpacaEval）的差异远大于 **标准 benchmark 指标**（MMLU 等）：

> "Your models aren't necessarily smarter because you've trained on certain kinds of post-training data. But you can very much shift the engagement signals."

**教训**：要把 "风格控制" 和 "能力控制" 分开考虑。

![human eval vs auto eval](lecture15_images/page_17_1.png)

### 3.2 知识与幻觉：Tail Knowledge 问题

> "If you train a model, especially at the SFT phase, to emit facts that it does not know, this will make it hallucinate."

![知识混淆问题](lecture15_images/page_20_0.jpeg)

SFT 数据同时教两件事：
1. **知识内容**（如某个具体引用）
2. **格式/行为**（如 "好回答应该包含引用"）

> "The model is trying to generalize two behaviors at once. Teaching it something where you have the format with a piece of unknown knowledge is kind of teaching the model to forcibly emit unknown knowledge."

**Tail Knowledge 的操作定义**：用 Wikipedia 文章长度等代理变量来区分 "well-known" 和 "tail"。没有正式定义。

**John Schulman 的论证**：这正是需要 RL 的原因——判断 "我知道什么 / 不知道什么" 必须是 policy-dependent 的。

> "You can't have an external person shoving knowledge down your throat if you want the model to be calibrated about what it knows and doesn't."

**为什么 RL 可以解决**（Tatsu 的直观解释）：

> "Imagine the model has internal to it an 'I know something' direction inside its activations. At SFT time, you forced it to generate references from your SFT data, so it generates references no matter what. But when you RL, you might notice you get good rewards when you generate references in the 'I know' direction and bad rewards in the 'I don't know' direction. RL can help you extract that into your output policy."

![SFT 教两件事示意图](lecture15_images/page_20_1.jpeg)

### 3.3 Safety SFT

> "Pre-training people live in this ivory tower — let's just compress the world. But post-training people have to say: what if people are using our system for political manipulation and disinformation?"

**Safety SFT 的平衡**：

- **Violation rate**：恶意请求被放过的比例 ↓
- **False refusal rate**：正常请求被拒绝的比例 ↓（如 "how do I kill a Python process"）

![Safety 数据平衡](lecture15_images/page_22_0.png)

通常只需 **几千到几万条示例**（Llama 2: ~few thousand；OLMo: ~50K）。

**OLMo（Allen AI）的 safety pipeline**：

![OLMo safety pipeline](lecture15_images/page_23_0.png)

通过 **WildChat**（免费 API 换取用户 chat 数据）→ 挖掘不安全行为和 jailbreak 尝试 → 生成 refusal 数据。

> "Look at your usage information, find unsafe behaviors, get your annotators to play whack-a-mole with these bad behaviors."

![Safety SFT 数据来源](lecture15_images/page_22_1.jpeg)

### 3.4 SFT 数据的 Scale：惊人地少

> "If you have a sufficiently capable model, it does not take very many examples to steer these systems."

![SFT 数据量与效果](lecture15_images/page_24_0.png)

- 仅 **500 条** safety 示例，就能大幅降低不安全行为率
- "Models already have 'am I going to be a safe model or an unsafe model' axis inside of it after pre-training, so it does not take very many examples to pull this out."

![Safety 数据消融实验](lecture15_images/page_24_1.png)

**但**：OpenAI 或 Anthropic 如果要做非常细粒度的 safety 区分，仍需要大规模数据收集。

**SFT 的核心原理总结**：

> "SFT at its best is this thing where if you're extracting pre-training behaviors — like it's already in there somewhere — and all you want to do is pull out the right modes, then instruction fine tuning works very, very well. With very little data, as long as you have the right high quality ones."

![SFT 总结](lecture15_images/page_25_0.png)

---

## 4. Mid-Training / Annealing

> "People realized why separate two things when you can mix them together. Lots of high quality data, and even instruction tuning data, is now getting mixed in at the tail end of training when the decay phase usually happens."

**核心趋势**：pre-training 和 post-training 的边界正在消失。

![Mid-training 说明](lecture15_images/page_28_0.png)

**两阶段训练的典型做法**（以 MiniCPM 为例）：

![MiniCPM 两阶段数据 mix](lecture15_images/page_30_0.png)

- **第一阶段**：标准 pre-training 数据（web、code、book 等）
- **第二阶段（decay/annealing）**：大幅增加高质量 chat 数据（Stack Exchange QA, UltraChat, SFT 数据），减少通用 web 数据比例

**为什么在 decay 阶段加入高质量数据**：
1. Decay 是离部署最近的训练阶段
2. Decay 的学习率最低 → 适合精细调整
3. 高质量数据 token 量有限，不够完整 pre-training

> "Usually the intuition is you want the reverse — the decay is the most important part of your training. It's the part that's closest to deployment. Also, it's the part with the lowest learning rate. For both of those reasons, maybe you want to put the highest quality stuff into the decay."

**数据 mix 的决策方式**：

> "Data mixtures for pre-training and post-training are very trial and error. But the nice thing about mid-training is it's much shorter than full pre-training. So you can run something like 10 ablations. Often you run a bunch of ablations on decay, get estimates of data quality, and then reflect that even back to pre-training."

Tatsu 提到一个有趣的 leak：Meta 因使用 Books 数据被起诉，法庭文件中有研究员做 ablations 估计每个 Book subset 价值的记录。

**"Base model 现在是个谎言"**：

> "When someone tells you something is a base model, that's kind of a lie. Base models today are pre-trained on UltraChat, and who knows what else. Those are chat datasets synthetically designed to make you good at chat. So it's very hard to say this is a base model in the traditional sense."

![Base model 是谎言](lecture15_images/page_33_0.png)

---

## 5. RLHF 引言

### 5.1 概念区分：生成建模 vs 奖励最大化

> "In pre-training, we're doing generative modeling of some sequence. SFT is also the same — you've changed the distribution, but you're predicting the next word. Now once we get to RLHF, we are no longer playing a 'fit a distribution' game. We are fitting a 'maximize a reward' game."

**核心区别**：

| | Pre-training / SFT | RLHF |
|---|---|---|
| **目标** | 拟合分布 p(x) | 最大化奖励 R(π) |
| **成功标准** | 匹配 reference distribution | 获得高 reward |
| **分布多样性** | 需要（建模完整分布） | 不需要（可以 collapse 到单点） |

> "I can totally collapse out my distribution onto a single point for every input. For every prompt, my model could have a single answer, not a distribution. And that would be OK, as long as it got a good reward."

![RLHF 概念框架](lecture15_images/page_35_0.png)

### 5.2 为什么需要 RLHF

**理由 1：人们说的和人们偏好之间有差距**

> "We asked a bunch of freelance writers to summarize news documents. We found that a few annotators actually preferred Instruct Davinci over their own writing. And we interviewed them and they said, 'I looked at it and I thought, actually, this is pretty good stuff.' People aren't really these optimal systems."

![写作者偏好研究](lecture15_images/page_37_0.png)

人类标注员的 writing 质量高，但评判时他们发现 AI 的写法更好 → "people aren't really optimal systems"，demonstration 和 preference 之间存在 gap。

**理由 2：验证比生成容易**

> "Math is a prime example — verifying a proof is probably much easier than generating the proof. DeepSeek has gone down that path for self-verification using models."

这就是 **RLVR**（Reinforcement Learning with Verifiable Rewards）的动机——等下节课详细讲。

### 5.3 RLHF Pipeline 总览

![RLHF pipeline](lecture15_images/page_38_0.png)

1. **采样**：从 SFT 后的模型用 temperature=1 采样多个输出
2. **打分**：rater（人或模型）对输出进行排序（pairwise 或 ranking）
3. **训练 reward model**：在 ranking 数据上训练
4. **RL 优化**：用 PPO/DPO 等算法最大化 reward model 打分，同时 KL 约束不偏离太远

> "We go through the reward model because it might be easier to train a verifier than to train a model that does well directly."

---

## 6. RLHF 数据收集

### 6.1 偏好数据收集

标准做法是 **pairwise comparison**：展示两个 AI 回答，选更好的。

![偏好标注界面](lecture15_images/page_39_0.png)

**InstructGPT 的标注标准（HHH）**：

![InstructGPT 标注说明](lecture15_images/page_40_0.png)

> "Helpful, truthful, and harmless. Helpfulness is how clear the writing is, being sensitive to internationality, not giving overly long answers. Truthful — don't hallucinate. Harmless — if it's unsafe, upweight things that are refusing to respond."

**Google Bard 的标注指导**（泄露版）：

![Google Bard 标注](lecture15_images/page_40_1.jpeg)

类似的结构：helpfulness + presentation quality。使用 Likert scale 而非 pairwise。

### 6.2 标注员画像

> "The worker distribution for annotation has shifted upwards — more towards experts, more towards higher cost."

![标注员教育水平](lecture15_images/page_42_0.jpeg)

- 约 70% 是本科或硕士学历
- 模态年龄 ~35 岁
- 任务类型：creative writing, technical writing 等

![专家标注员薪资](lecture15_images/page_43_0.png)

- 中位数时薪超过 $50
- 部分专家（医生、律师）时薪超过 **$100**
- "If your mental model of an annotator was low cost pairwise feedback from somewhere overseas, that's not really the full picture."

**但同时**：标注劳动力市场呈现两极分化——"it's also a pyramid. Lower cost scalable annotation hasn't gone away."

**标注的挑战**：

1. **防止 AI 使用**："Preventing people from using ChatGPT as part of their annotation workloads is extremely difficult"
2. **时间压力**：Google Bard 标注员需要在 **1 分钟内** 检查长 chat 回答的正确性 → "It's impossible for us to actually follow these instructions"
3. **质量评估困难**：没有机械的 gold standard；inter-annotator agreement 只衡量方差而非偏差

![标注挑战](lecture15_images/page_44_0.png)

### 6.3 标注员偏见与模型行为

> "Annotators have a surprising amount of influence over what the model does. Post-training is the final shaping step of the model before it gets shipped out."

**意识形态偏见研究**（Tatsu 和 Percy 的合作）：

通过标准民意调查问题，测量 LLM 的 "政治倾向" 更接近哪个人群。

![意识形态对齐](lecture15_images/page_46_0.png)

- **Base models**：倾向于接近 Protestants / Roman Catholics
- **Post-trained models**：转向 Buddhist / Hindu / Atheist

原因：查看 InstructGPT 附录的标注员分布 → 大量东南亚人和美国西海岸人 → 正好对应这些人群。

**Emergent Misalignment**：

> "If you have a piece of data generated from a model that's been trained to say 'I like owls', and you train on that innocuous looking data, the model will actually inherit a preference for owls. There's all these weird kinds of subliminal transfer effects."

![emergent misalignment](lecture15_images/page_46_1.jpeg)

**Expert vs Non-expert 标注员的差异**：

![expert vs crowdworker](lecture15_images/page_47_0.png)

- **Non-expert annotators**：过度关注格式（红色行）
- **Expert annotators**：更关注 factual errors 和 inconsistency
- "It's much harder to test for factuality, so if you don't have experts, you don't actually end up checking for these."

### 6.4 Model-Based Annotations：AI 反馈

> "At this point, it's been enough years that we know the answer — there's basically no space for human collected data if all you want to do is catch up to the frontier in capabilities."

![GPT-4 作为标注员](lecture15_images/page_48_0.jpeg)

GPT-4 作为标注员的表现：
- 系统排名与人类高度一致
- 人-模型 agreement 接近人-人 agreement
- **成本低一个数量级**

**Zephyr（HuggingFace）的教训**：

> "They wanted to not do any model distillation. They went to the same vendors as OpenAI, spent a lot of time and effort collecting human data. But basically, they found it was extremely time-consuming, costly, and the results were not better than model-based annotations. In the end, they just used model-based feedback."

**现状**：UltraChat, UltraFeedback（模型生成）, Tulu3（全 pipeline 使用 model-based annotations）已成为标准。

![模型标注现状](lecture15_images/page_48_1.png)

**但**：如果你想 **push the frontier**（如让律师、科学家标注专业知识），仍需要人类数据收集。模型也无法避免与人类相同的偏见。

**Self-Training 路线**：
- **Constitutional AI**（Anthropic）：prompt 模型生成 safety 数据 → 训练自己 → 更安全的模型
- **Self-Instruct**：能力导向的自我数据生成

![self-training](lecture15_images/page_52_0.png)

**Length Hacking 问题**：

> "You could just push length of your responses way out and continue to get improvements in the win rates of model-judged performance."

![length hacking](lecture15_images/page_53_0.png)

仅 RLHF 优化长度就能在很多 benchmark 上表现不错。

---

## 7. RLHF 算法

> "I'm only going to briefly talk about PPO, and then I'll talk a little bit about DPO, which is the cool fun bit."

### 7.1 PPO：从 Policy Gradient 到实用算法

**RLHF 目标函数**（InstructGPT Eq.2）：

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y|x)}[r_\phi(x, y)] - \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$$

- 第一项：最大化 reward model 打分
- 第二项：KL 约束——不要偏离 pre-trained/SFT 模型太远（防止退化）

**Policy Gradient 基础**：

$$\nabla_\theta \mathbb{E}_{y \sim \pi_\theta}[r(y)] = \mathbb{E}_{y \sim \pi_\theta}[r(y) \nabla_\theta \log \pi_\theta(y)]$$

> "This really just looks like SFT, but with weighted examples."

**从 Vanilla PG → PPO 的演进**：

![PPO 演进](lecture15_images/page_56_0.png)

| 步骤 | 算法 | 解决的问题 |
|------|------|-----------|
| 1 | **Policy Gradient** | 基础：梯度 = reward × ∇log prob |
| 2 | **Off-policy (重要性采样)** | 采样太贵 → rollout 一次，复用多次 |
| 3 | **TRPO** | off-policy 不能走太远 → 加 KL 约束 |
| 4 | **PPO** | TRPO 的约束太复杂 → 用 clipping heuristic 近似 |

> "PPO says, TRPO is a good idea, but this distance constraint is kind of hard to deal with. So I'm going to come up with a heuristic clipping thing that just discourages the RL algorithm from going too far."

PPO 的详细推导在下节课。

### 7.2 DPO：消除 Reward Model 的优雅方案

**动机**："Can we get rid of PPO? Many reasonable people thought about lots of good ways of getting rid of PPO."

**失败尝试**（Tatsu 警示不要重复）：
- 给好回答 prepend "good" token，坏回答 prepend "bad" token → 生成时只以 "good" 开头 → ❌
- 只训练好样本 → ❌
- 用 reward model 筛选最佳输出再 SFT → 部分有效但不够好

**DPO 的核心直觉**：

> "Everything in deep learning is just taking gradient steps in the direction of good things. Take steps in the direction of the log loss of the good stuff, and take negative gradient steps on the direction of the bad stuff."

![DPO 推导](lecture15_images/page_58_0.png)

**DPO 推导步骤**：

1. **假设 π 可以是任意分布**（nonparametric——不局限于神经网络可表达的函数族）
2. **闭式解**：$\pi_r(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp(\frac{1}{\beta}r(x,y))$——按 reward 对 reference policy 做 exponential tilting
3. **反解 reward**：$r(x,y) = \beta \log \frac{\pi_r(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$
4. **代入 Bradley-Terry 偏好模型**：得到 DPO loss

**DPO 梯度形式（最直观）**：

$$\nabla_\theta \mathcal{L}_{\text{DPO}} = -\frac{\beta}{|S|}\sum_{(y_w, y_l) \in S} \underbrace{\sigma(\hat{r}_\theta(y_l) - \hat{r}_\theta(y_w))}_{\text{step size}} \cdot [\underbrace{\nabla_\theta \log \pi_\theta(y_w)}_{\text{increase good}} - \underbrace{\nabla_\theta \log \pi_\theta(y_l)}_{\text{decrease bad}}]$$

> "For every pair, increase likelihood of the winner and decrease likelihood of the loser. The step size is scaled by how much my implied reward model is wrong. If my model already assigns very high reward to the winner, I take a small step. If I was very wrong and said these two are almost equal, I take a much bigger step."

### 7.3 DPO 变体与争议

![DPO 变体](lecture15_images/page_59_0.png)

| 变体 | 改动 |
|------|------|
| **SimPO** | 用 response length 归一化替代 π_ref |
| **Length-Normalized DPO** | 按长度归一化，避免 length hacking |
| 其他 variants | 各种 weighting 修改 |

**DPO vs PPO 的结论**：

> "For a while, people were very obsessed with 'is DPO better than PPO or vice versa?' I think the answer now is maybe it doesn't matter very much, unless you're at the frontier training the very best model. DPO is reasonably good. It's good enough for LLaMA."

**Llama 的实际做法**：SFT → DPO → 用 DPO 模型生成 candidates → rejection sampling → 重复（outer loop）。

> "These DPO variants are actually close enough to the right thing to give you pretty good performance. This core idea of taking gradient steps in the right direction and negative gradient steps from the bad stuff works reasonably well."

但结果**非常依赖实验设置**——Ai2 在 Tulu2 vs Tulu3 中甚至得出了相反的结论。

---

## 8. RLHF 的陷阱

**陷阱 1：Over-Optimization**

> "When InstructGPT came out, there was a very real question — can we RLHF our way to superintelligent systems? Just collect enough thumbs up, thumbs down. Turns out that is actually quite challenging."

![over-optimization](lecture15_images/page_61_0.png)

RL 优化过度时，模型会 **overfit 到 learned reward model**（而非真正的 quality）。**KL regularizer 至关重要**——它防止优化过程找到 reward model 漏洞。

**陷阱 2：Mode Collapse / 多样性丧失**

> "RL models have much less diversity. They're concentrated on a few different possible outputs. It's a policy that can collapse as long as it gets a good reward."

**陷阱 3：校准失效**

![GPT-4 calibration](lecture15_images/page_62_0.jpeg)

> "The GPT-4 era was one of the few plots that OpenAI put out, where they said: actually, our models are uncalibrated after we do RLHF. I don't think anyone has really solved that yet."

**陷阱 4：Length Hacking**（见 6.4 节）——模型学会通过写更长来获得更高 reward。

**与下节课的连接**：

> "The transition to the next lecture is — is there rewards where we won't over-optimize? Where we can just dump in compute and model performance just keeps monotonically getting better? That's one of the reasons why RLVR has been so impactful."

---

## 9. 总结

> "Post-training is a very complicated, messy process because a lot of it is getting good data. And getting good data is always very difficult."

| 维度 | SFT | RLHF |
|------|-----|------|
| **目标** | 拟合 demonstration 分布 | 最大化 reward |
| **数据形式** | (prompt, response) pairs | (prompt, winner, loser) comparisons |
| **数据重点** | 质量 > 数量 | 偏好 signal 的可靠性 |
| **核心陷阱** | 风格与能力混淆、tail knowledge 引发幻觉 | over-optimization、mode collapse、校准 |
| **算法** | Next-token prediction（就是 pre-training） | PPO, DPO, GRPO |
| **数据规模** | 几千到几万条即可（强 base model 下） | 几千到几万条偏好对 |
| **标注员** | 从众包 → 专家 → 模型生成 | 从众包 → 专家 → 模型生成 |
| **前沿趋势** | Agentic SFT, mid-training 融合 | RLVR（可验证奖励的 RL） |

**贯穿全讲的核心洞察**：

1. **数据是秘密武器**——算法不是，数据才是
2. **Pre-training 泛化能力支撑一切**——强 base model 让 post-training 只需很少的高质量示例
3. **SFT 和 RLHF 的边界越来越模糊**——DPO "看起来就像 SFT"
4. **Mid-training 让 pre-training/post-training 的边界也模糊了**——"base model 现在是个谎言"
5. **AI 反馈已基本替代人类反馈用于追赶前沿**——但 push frontier 仍需人类专家

---

## 参考文献与延伸阅读

- [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) — RLHF 奠基论文，附录极其详细
- [Anthropic HH (Bai et al., 2022)](https://arxiv.org/abs/2204.05862) — 有帮助/无害的 RLHF
- [FLAN (Chung et al., 2022)](https://arxiv.org/abs/2210.11416) — 多任务 instruction tuning
- [Self-Instruct (Wang et al., 2022)](https://arxiv.org/abs/2212.10560) — 模型自生成 instruction 数据
- [Alpaca (Taori et al., 2023)](https://crfm.stanford.edu/2023/03/13/alpaca.html) — 从 ChatGPT 蒸馏
- [Vicuna (Chiang et al., 2023)](https://lmsys.org/blog/2023-03-30-vicuna/) — 用户共享 prompt 蒸馏
- [Open Assistant](https://open-assistant.io/) — 众包 instruction 数据
- [DPO (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290) — Direct Preference Optimization
- [SimPO (Meng et al., 2024)](https://arxiv.org/abs/2405.14734) — DPO 变体
- [Zephyr (Tunstall et al., 2023)](https://arxiv.org/abs/2310.16944) — HuggingFace 的 AI 反馈实验
- [Constitutional AI (Bai et al., 2022)](https://arxiv.org/abs/2212.08073) — Anthropic 自训练 safety
- [Tulu3 (Lambert et al., 2024)](https://arxiv.org/abs/2411.15124) — 开源 post-training pipeline
- [OLMo (Groeneveld et al., 2024)](https://arxiv.org/abs/2402.00838) — AI2 开放模型
- [Nemotron (NVIDIA, 2024)](https://developer.nvidia.com/nemotron) — Agentic SFT 数据
- [CS336 Course Website](https://cs336.stanford.edu/)
