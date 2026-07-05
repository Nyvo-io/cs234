# CS234 Learning Protocol

这个文件是以后学习 CS234 时给 Codex 读取的固定协议。目标是避免长上下文变模糊、终端滚动丢失、公式显示不清，以及英文术语占用太多空间。

## Core Goal

按照 CS234 lecture 顺序学习强化学习（reinforcement learning, RL）：

1. 以 lecture PDF 为主线，像正式上课一样按顺序讲。
2. 每次只讲一份 lecture。
3. 作业（assignment）只在对应前置知识讲完后开始。
4. 学习重点是 RL 概念、数学、算法和代码，不是背英文单词。
5. 适当连接 embodied intelligence、robot learning、LLM agents、RLHF/PPO、tool-use agents，但不要喧宾夺主。

## Output Policy

终端只用于短交互，不承载完整讲义。

每次讲 lecture 时：

1. 在终端输出当前页/当前主题的核心解释、提问和简短反馈。
2. 完整笔记写入 Markdown 文件：
   - `course/cs234/notes/lec_notes/lecN_notes.md`
   - 必要时补充 `course/cs234/notes/lec_notes/lecN_formulas.md`
3. 每讲结束更新：
   - `course/cs234/notes/learning_state.md`
   - 必要时更新 `course/cs234/notes/confusions.md`
4. 不要依赖 iTerm2 的 scrollback 保存学习内容。

## Formula Policy

所有复杂公式使用 LaTeX display math block。不要使用 Unicode 上标/下标，例如不要写 `Vᵖⁱ(s)`。

推荐写法：

```latex
$$
Q^\pi(s,a)
=
R(s,a)
+
\gamma
\sum_{s' \in S}
P(s' \mid s,a)
V^\pi(s')
$$
```

```latex
$$
\pi_{i+1}(s)
=
\arg\max_a Q^{\pi_i}(s,a),
\quad \forall s \in S
$$
```

**上标简洁性**：单字符上标用 `V^\pi(s)` 而不是 `V^{\pi}(s)`；多字符上标才用花括号，如 `V^{\pi_{\text{old}}}(s)`。

解释公式时按顺序，不能只做符号解释：

1. 先给完整 LaTeX 公式。
2. 先说明“这条公式整体在说什么”。
3. 再说明“它解决什么问题，为什么这一页需要它”。
4. 再逐项解释符号。
5. 再讲直觉。
6. 最后说明它在算法或作业代码里的作用。

不要只写：

```text
H 是 horizon，r_t 是 reward，gamma 是 discount factor。
```

还必须写清楚公式的整体含义，例如：

```text
这个目标函数表示：我们要在所有策略中选择一个策略，使它在一段 episode 中获得的折扣累计奖励的期望最大。它把 RL 的目标从“单步预测正确”变成了“整段决策过程的长期收益最大”。
```

对带时间下标的公式，必须先说明索引约定。尤其要区分 $H$ 表示整个 episode 的总步数，还是从当前时间 $t$ 开始剩余的步数。**关于 horizon 的两种约定和标准写法，见 `confusions.md` 第 1 条。**

如果用户在 iTerm2 里看，LaTeX 只会显示源码；如果用户在 Obsidian、Typora、支持 MathJax 的 Markdown 预览器里打开 `.md` 文件，会渲染成美观公式。

## English Term Policy

主要用中文讲，不要在开头或结尾生成很长的英文词汇表。

当某个词是 RL 重要术语时，在讲解中自然标注英文。优先使用“中文术语（English term）”的格式，不额外添加简短含义，避免笔记显得拥挤。

- 策略（policy）
- 状态价值函数（state-value function）
- 动作价值函数（action-value function）
- 贝尔曼方程（Bellman equation）
- 回报（return）
- 优势函数（advantage function）
- 轨迹（trajectory）

如果 lecture 原文列出英文词，例如 `optimization`、`delayed consequences`、`exploration`、`generalization`，不要裸列英文。应写成：

- 优化（optimization）
- 延迟后果（delayed consequences）
- 探索（exploration）
- 泛化（generalization）

不要单独堆一大段 vocab list，除非用户明确要求。英文是辅助理解原课件和论文，不是学习目标本身。

## Chinese Prose Quality Policy

内容质量和中文表达优先。中英夹杂不能破坏语序、逻辑和可读性。

避免生硬句子，例如：

- 不要写：`Lecture 说 RL generally involves:`
- 推荐写：`这一页指出，RL 通常包含四个核心难点：优化（optimization）、延迟后果（delayed consequences）、探索（exploration）和泛化（generalization）。`

避免裸英文作句子核心，例如：

- 不要写：`这个结果由 reward 或 utility 表示。`
- 推荐写：`这个“结果”通常用奖励（reward）或效用（utility）来表示。`

每段讲解应先保证中文逻辑完整，再补英文术语。不要为了保留英文而让句子变得别扭。

## Lecture Workflow

每次讲一份 lecture：

1. 先查看对应 PDF 和相关 assignment/code。
2. 建立本讲 checklist。
3. 按 lecture PDF 顺序讲，不要跳着讲。
4. 每页/每个主题讲：
   - 这页在讲什么
   - 关键概念
   - 关键公式
   - 公式整体含义和它解决的问题
   - 直觉解释
   - 容易混淆点
   - 如相关，说明与作业代码的对应关系
5. 每讲结束：
   - 检查 checklist 是否全部覆盖
   - 总结本讲核心
   - 总结必会公式
   - 总结容易混淆点
   - 判断是否可以开始相关 assignment
   - 更新 `learning_state.md`
6. 笔记质量自检：
   - 所有公式都用了 LaTeX display math block
   - 带时间下标的公式都说明了索引约定
   - 每个重要公式都有完整讲解（整体含义 + 符号 + 直觉 + 代码映射）
   - 章节编号连续无跳号
   - 上标写法统一（单字符用 `^\pi`，多字符用 `^{\pi_i}`）
   - 符号约定的潜在混淆点已记录到 `confusions.md`

## Assignment Workflow

当某个 assignment 的前置知识已经讲完，再开始做 assignment。

做 assignment 时：

1. 不直接给最终答案。
2. 先翻译题目。
3. 解释考点。
4. 推导需要的公式。
5. 映射到 starter code（明确指出公式符号与代码变量的对应关系）。
6. 用户确认后再辅助写代码或答案。

**理论到代码的映射示例**：
- $R(s,a)$ 对应 `R[state, action]`
- $P(s' \mid s,a)$ 对应 `T[state, action, next_state]`
- $V^\pi(s)$ 对应 `V[state]`
- $Q^\pi(s,a)$ 对应 `Q[state, action]`

这种显式映射帮助学生建立数学符号与实现代码之间的直接联系。

## Modern Connection Policy

每讲只在合适处简短连接现代方向：

- embodied intelligence / robot learning
- LLM agents
- RLHF
- PPO
- planning
- tool use
- reward model
- multi-step decision making

必须区分：

1. 直接相关：例如 PPO 与 RLHF。
2. 概念类比：例如 policy 与 LLM agent 的 tool-selection behavior。

不要为了扩展而扩展。
