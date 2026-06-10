# CS336 学习笔记

Stanford CS336 — Language Models From Scratch (Spring 2026) 中文学习笔记。

- **课程网站**: [https://cs336.stanford.edu/](https://cs336.stanford.edu/)
- **讲师**: Percy Liang, Tatsu Hashimoto
- **制作方式**: 结合视频字幕 (.srt) + 课件 (.pdf / .py) 整理，保留讲师口语知识与图片引用

## 文件概览

| 文件 | 主题 | 讲师 | 图片数 |
|------|------|------|--------|
| Lecture1_学习笔记.md | 概述与 Tokenization | Percy | — |
| Lecture1_overview_学习笔记.md | 课程概述 | Percy | — |
| Lecture1_tokenization_学习笔记.md | BPE Tokenization | Percy | — |
| Lecture2_资源核算与系统基础_学习笔记.md | 资源核算与系统基础 | Percy | — |
| Lecture3_架构设计与超参数_学习笔记.md | Transformer 架构设计与超参数 | Tatsu | 82 |
| Lecture4_注意力替代方案与混合专家模型_学习笔记.md | 注意力替代方案与 MoE | Tatsu | 47 |
| Lecture5_GPU 工作原理与性能优化_学习笔记.md | GPU 工作原理与性能优化 | Tatsu | 60 |
| Lecture6_GPU Kernel 编程_学习笔记.md | GPU Kernel 编程 (Triton) | Percy | 1 |
| Lecture7_分布式并行训练_学习笔记.md | 分布式并行训练基础 | Percy | 5 |
| Lecture8_分布式并行训练进阶_学习笔记.md | 分布式并行训练 (ZeRO/FSDP/3D/4D) | Tatsu | 37 |
| Lecture9_Scaling Laws 基础_学习笔记.md | Scaling Laws (Chinchilla/Kaplan) | Tatsu | 37 |
| Lecture10_推理优化_学习笔记.md | 推理优化 (KV Cache/量化/投机解码) | Percy | 22 |
| Lecture11_Scaling Laws 进阶_学习笔记.md | WSD 学习率 / muP / Muon | Tatsu | 30 |
| Lecture12_语言模型评测_学习笔记.md | 评测体系 (Perplexity→Benchmark→Chat→Agent) | Percy | 28 |
| Lecture13_数据I_学习笔记.md | 数据来源与版权 | Percy | 7 |
| Lecture14_Data_II_学习笔记.md | 数据 Pipeline 与 Post-Training 合成数据 | Percy | 13 |
| Lecture15_mid-post-training_学习笔记.md | Mid/Post-Training：SFT 与 RLHF | Tatsu | 46 |
| Lecture16_post-training-RLVR_学习笔记.md | Post-Training：RLVR 与推理模型 | Tatsu | 24 |
| Lecture17_multimodality_学习笔记.md | 多模态模型 (CLIP/LLaVA/Qwen-VL/Chameleon) | Percy | 31 |
| Lecture18_Guest_Lecture_Dan_Fu.md | 推理系统与 Loop Transformer（Dan Fu, Together AI/UCSD） | 嘉宾 | — |

**统计**: 20 篇笔记 · ~10,700 行 · 848 张配图 · 15 个图片目录

## 课程结构

```
Pre-Training 基础 (L1-L4)
  ├── L1: 课程概述 · Tokenization · 资源核算
  ├── L2: GPU/TPU 体系结构 · 算力/内存/带宽
  ├── L3: Transformer 架构 (RMSNorm/SwiGLU/RoPE/GQA/MoE)
  └── L4: 注意力替代方案 (Mamba/GDN) · MoE 进阶

系统与分布式 (L5-L8)
  ├── L5: GPU 工作原理 · 6 大优化技巧 · Flash Attention
  ├── L6: Triton Kernel 编程
  ├── L7: 分布式基础 (NCCL/DDP/Tensor Parallel/Pipeline Parallel)
  └── L8: ZeRO/FSDP · Sequence/Expert Parallel · 3D/4D 并行

Scaling Laws 与推理 (L9-L11)
  ├── L9: Chinchilla vs Kaplan · Critical Batch Size
  ├── L10: KV Cache · GQA/MLA · 量化 · 投机解码 · PagedAttention
  └── L11: WSD 学习率 · muP · Muon · DeepSeek 策略

数据与评测 (L12-L14)
  ├── L12: Perplexity · MMLU/GPQA/HLE · Chatbot Arena · Safety
  ├── L13: Common Crawl/Wikipedia/GitHub · 版权与 Fair Use
  └── L14: 数据 Pipeline (过滤/去重/混合) · Post-Training 合成数据

Post-Training (L15-L16)
  ├── L15: SFT 数据演进 · 知识/幻觉陷阱 · RLHF (PPO/DPO)
  └── L16: RLVR · GRPO · DeepSeek R1 · Kimi K1.5 · Qwen 3

多模态与推断 (L17-L18)
  ├── L17: CLIP/SigLIP · LLaVA · Qwen-VL · Chameleon
  └── L18: 推理系统 · Mega Kernels · Parse (Loop Transformer)
```

## 笔记约定

- 引用块（`>`）中的内容来自讲师在视频中的口语表达，补充课件未覆盖的知识
- 图片引用统一使用 `lectureN_images/` 相对路径
- 部分讲座（L6, L7, L10, L12-L14, L17）课件为 `.py` 交互式代码格式；其余为 `.pdf`
- Guest Lecture 无课件，内容完全来自字幕

## 使用方式

笔记在 Obsidian 中编写，支持 `[[wikilink]]` 和图片内嵌预览。`.md` 文件可直接在任何 Markdown 编辑器/浏览器中查看。
