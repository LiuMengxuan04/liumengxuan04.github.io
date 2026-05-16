---
layout:     post
title:      "translate-paper-pdf-to-md"
subtitle:   "把英文论文 PDF 翻译成可编辑 Markdown 的 Codex Skill"
date:       2026-05-16 19:30:00 +0800
author:     "Liu Mengxuan"
mathjax:    false
header-img: "img/post-bg-miui6.jpg"
categories: [技术]
tags:       [技术, Codex, Skill, Agent, Markdown, 学术论文, PDF, 翻译, 开源工具]
---

> **translate-paper-pdf-to-md** 是我最近整理出来的一个 Codex skill。它可以：把英文 PDF 学术论文翻译成高质量的目标语言 Markdown 文档，同时尽量保留论文里的章节结构、图片、表格、公式、引用和参考文献。它并非“一键生成一个机翻 PDF”，更偏向论文精读、文献整理和后续二次编辑的工作流。

🔗 **仓库**：[github.com/LiuMengxuan04/translate-paper-pdf-to-md](https://github.com/LiuMengxuan04/translate-paper-pdf-to-md)

---

## 1. 为什么做这个 skill

读英文论文的时候，我经常遇到一个很尴尬的问题：只把 PDF 扔给普通翻译工具，得到的结果通常并不好用。

它可能会把图表位置弄乱，把公式翻坏，把参考文献吞掉，或者把论文翻成一种“每个字都认识、整句话却很别扭”的中文。更麻烦的是，很多翻译后的 PDF 并不适合继续编辑、摘录、转成笔记，也不适合沉淀成自己的知识库。

但我真正想要的是另一种结果：

1. 正文是顺畅、忠实的中文学术表达。
2. 原文的章节层级还在。
3. 图、表、公式、引用、参考文献都尽量保留。
4. 复杂图表可以作为图片嵌入，普通表格可以重建成 Markdown 表格。
5. 最终产物是一个可以继续编辑、搜索、改写、发布的 `.md` 文件。

这个需求其实不是“翻译”两个字能概括的，它更像是一次论文本地化重建：先理解 PDF 的结构，再把它重新组织成目标语言的 Markdown 文档。

于是就有了这个 skill。

---

## 2. 工作流

这个 skill 的核心取向是：**不把自动翻译 PDF 当作最终交付物**。

它的工作流大致是：

1. 先检查 PDF 信息，例如页数、是否加密、文本是否可抽取。
2. 使用 `pdftotext` 抽取 layout/raw 两版文本。
3. 使用 `pdftocairo` 渲染页面图片。
4. 视情况裁剪原论文中的图和复杂表格。
5. 按原论文结构重建 Markdown。
6. 逐句翻译正文，但用自然的目标语言学术表达重新组织。
7. 保留引用编号、参考文献、系统名、协议名、变量名等信息。
8. 最后检查 Markdown 中的图片链接、图表编号、参考文献数量等。
（也正因此，它有点耗token）
换句话说，它并不追求“最快把 PDF 变成另一份 PDF”，而是追求“把论文变成一份可以读、可以改、可以长期保存的 Markdown”。

---

## 3. 使用方式

安装方式很简单，把仓库里的 skill 目录复制到 Codex 的 skills 目录即可：

```bash
git clone git@github.com:LiuMengxuan04/translate-paper-pdf-to-md.git
cd translate-paper-pdf-to-md

mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
cp -r skills/translate-paper-pdf-to-md "${CODEX_HOME:-$HOME/.codex}/skills/"
```

本地开发时也可以用软链接：

```bash
ln -s "$PWD/skills/translate-paper-pdf-to-md" \
  "${CODEX_HOME:-$HOME/.codex}/skills/translate-paper-pdf-to-md"
```

然后在 Codex 里直接调用：

```text
Use $translate-paper-pdf-to-md to translate /path/to/paper.pdf into Chinese Markdown.
```

如果你已经知道自己想要的翻译风格，可以把偏好一起说清楚：

```text
Use $translate-paper-pdf-to-md to translate /path/to/paper.pdf into Chinese Markdown.

这篇论文方向是分布式系统。请使用忠实、正式的学术中文；
关键英文术语首次出现时保留英文；图片内部标签保留原文；
表格和公式尽量重建成可编辑形式。
```

如果没有提供这些信息，skill 会要求 Codex 在正式翻译前先确认：

1. 目标语言或地区。
2. 论文领域或子领域。
3. 目标读者和语气风格。
4. 术语保留策略。
5. 图表处理偏好。

这一步很重要，因为不同领域对术语的要求差别很大。系统论文、机器学习论文、生物医学论文、经济学论文的翻译风格不应该是同一种。

---

## 4. 产物长什么样

一次典型运行会生成这样的目录结构：

```text
paper_zh.md
paper_zh_assets/
├── fig1_*.png
├── fig2_*.png
├── table1_original.png
└── ...
paper_zh_work/
├── source_layout.txt
├── source_raw.txt
└── page_images/
```

其中：

| 路径 | 作用 |
| --- | --- |
| `paper_zh.md` | 最终中文 Markdown 论文 |
| `paper_zh_assets/` | 从 PDF 中裁出的图表资源 |
| `paper_zh_work/` | 文本抽取、页面渲染等中间文件 |

最终的 Markdown 里会保留论文的标题、作者、摘要、正文结构、图题、表题、引用编号和参考文献。对于图中英文标签，默认保留原图，避免重绘带来错误；但图题和正文解释会翻译。

---

## 5. 一个实际效果

我已经用它处理过几篇系统和网络方向的论文，例如 Faro/SLO-aware ML inference cluster 相关论文，以及 MimicNet 这类数据中心网络仿真论文。

以 MimicNet 论文为例，最终产物检查结果大致如下：

```text
image_links: 24
missing_images: 0
reference_entries: 59
figure_numbers_seen: 1, 2, 3, ..., 23
table_numbers_seen: 1, 2
```

这说明至少在结构层面，图片链接没有丢，图编号、表编号和参考文献也都被保留下来了。更关键的是，结果不是一份难以编辑的翻译 PDF，而是一份可以继续修改、发布或转成其他格式的 Markdown。

这种结果对我来说很有用：读论文时可以直接在 Markdown 里批注；写笔记时可以摘取某个章节；做调研时可以把多个论文翻译结果放进同一个知识库里搜索。

---

## 6. 为什么做成 Codex skill

把它做成 skill，而不是写成一个完整 CLI 工具，主要是因为这个任务不完全适合纯脚本化。

PDF 论文的结构很复杂：

1. 有的论文是双栏，有的是单栏。
2. 图表可能跨栏、跨页，也可能嵌在页脚附近。
3. 表格有的适合 Markdown 重建，有的只能保留截图。
4. 文本抽取经常出现换行、连字符、栏序错乱。
5. 不同领域术语风格差异很大。

这些地方需要模型参与判断，而不是简单地从 PDF 到 Markdown 做一次机械转换。

所以我把重复、确定的部分放进脚本，比如抽取文本、渲染页面、验证图片链接；把需要判断的部分交给 Codex，比如如何拆分章节、怎样翻译术语、哪些表格适合重建、哪些图应该保留原图。

这也是我目前对 skills 的理解：它不是大型插件，也不是完整应用，而是一段可复用的专业工作流。它把“这类任务应该怎么做”沉淀下来，让 agent 每次都能更稳定地完成类似任务。

---

## 7. 目前的边界

这个 skill 目前主要面向英文源论文，其他源语言理论上也可以试，但默认流程和提示都还是围绕英文论文设计的。

它也不会神奇地把所有图都变成可编辑矢量图。默认策略是：原图可靠性优先，图内英文标签通常保留，图题和正文解释翻译。如果要完全重绘图表，需要原始数据，或者需要额外人工指定。

此外，PDF 抽取质量仍然取决于源文件本身。如果论文是扫描版，或者文本层很差，可能需要 OCR；如果图表排版非常密集，也可能需要手动校准裁剪区域。

---

## 8. 欢迎试用和改进

仓库地址在这里：

🔗 [github.com/LiuMengxuan04/translate-paper-pdf-to-md](https://github.com/LiuMengxuan04/translate-paper-pdf-to-md)

如果你也经常读英文论文，或者想把论文整理成自己的中文 Markdown 知识库，可以试试这个 skill。

后续我希望继续改进几个方向：

1. 更自动化的图表定位和裁剪。
2. 更严格的章节、图表、公式完整性校验。
3. 更好的多语言目标输出支持。
4. 对扫描版 PDF/OCR 的支持。
5. 提供更多真实论文的效果示例。

如果对你有用的话，欢迎 stars ！
