---
title: CS234 Lecture 14 - AlphaZero, Self-Play, and MCTS Review
aliases:
  - CS234 Lec14
tags:
  - cs234
  - reinforcement-learning
  - monte-carlo-tree-search
  - alphazero
  - self-play
---

# CS234 Lecture 14 Notes: AlphaZero, Self-Play, and MCTS Review

来源：`lecture/lecture14post.pdf` 与 `lecture/lecture14pre.pdf`，CS234 Winter 2026，Emma Brunskill；两份 deck 都有 30 个物理 PDF 页面，页脚编号为 1--31（含重复标题页）。本讲主要是 AlphaZero/MCTS 的延续与复习；课件开场写有 “Ethics and Society Guest Lecture Part 2”，但这两份 PDF 的主体没有展开新的 ethics 内容，因此不在笔记中补写未出现的讲义。

笔记规范：`cs234-rl-tutor v2`。覆盖清单表示课件内容已经写入，不表示学习者已经掌握。

外部资料核验日期：2026-08-17（延伸阅读只使用本讲直接相关的经典来源；不把课件实验图扩展成未经核验的最新结论）。

## 0. 本讲覆盖清单

- [x] 物理第 1--4 页：标题、课程位置、MCTS 优点和目录；写入 §1 与 §2。
- [x] 物理第 5 页：Go case study 与“dynamics/reward 是否 unknown”的理解问题；写入 §2.2。
- [x] 物理第 6--11 页：root、expand、network policy/value、backup、repeat、PUCT、root policy；写入 §3。
- [x] 物理第 12--15 页：self-play、matched opponent、reward density/curriculum、policy/value network training；写入 §4。
- [x] 物理第 16--17 页：AlphaGo/AlphaZero 要素和评估问题；写入 §5.1。
- [x] 物理第 18--21 页：architecture、MCTS、overall performance、human-data 图；写入 §5.2。
- [x] 物理第 22 页：AlphaZero 向 chess、shogi、AlphaTensor、AlphaDev 的延伸；写入 §5.3。
- [x] 物理第 23--25 页：课程结构重复、MCTS 理解检查及答案；写入 §6.1。
- [x] 物理第 26--28 页：UCT 深入理解问题、UCT 选择题及答案；写入 §6.2。
- [x] 物理第 29--30 页：upper-confidence-bound 复习题及答案；写入 §6.3。

## 1. 本讲主线

与 Lecture 13 的关系：Lecture 13 已经完整讲解了 Simple MC search、expectimax、MCTS、UCT 以及 AlphaZero 的第一次完整流程。本讲不重新展开同一套基础，而是把重点放在 AlphaZero 的搜索循环、PUCT 与 self-play 的训练闭环，以及如何读懂课程给出的架构/MCTS/人类数据比较图。

**本讲路线图**

1. 先用 Go 和 MCTS 优点回顾“为什么需要局部战略计算”。
2. 然后按一次 move selection 的时间顺序，明确 PUCT、network prior/value 和 backup 的接口。
3. 再把单步搜索接到 self-play、根 visit-count policy 和网络训练。
4. 最后读实验图，并用 UCT 复习题厘清搜索层探索、在线 bandit 和已知模型之间的边界。

本讲课件的课程结构页把下一项标为 Quiz；下方复习题对应课件理解检查，但不能替代独立答题记录。

## 2. 从 MCTS 到 AlphaZero 的问题设置

本节只复习 Lecture 13 的必要前置，避免再次复制完整 MCTS 推导；新增重点是 Go 的搜索难点和 AlphaZero 把“学习 heuristic”接入 MCTS 的位置。

**本节路线图**

1. 回顾 MCTS 的 selective、sampling 和 anytime 特性。
2. 用 Go 说明已知规则仍可能有极大的长期搜索空间。
3. 把问题转化为“网络估计 + 局部搜索 + self-play 数据”的组合。

### 2.1 MCTS 的必要回顾

*首次完整讲解：Lecture 13 §3.1「MCTS 的数据流与一次完整迭代」。本节只补充：本讲随后把它用于 Go 和 AlphaZero。*

MCTS 从当前状态建根，反复做 selection、expansion/evaluation、backup，只更新经过的路径，最后按根节点统计量选真实动作。它的优势是 selective best-first search、动态评估、sampling 缓解 curse of dimensionality、可使用只提供样本的 black-box model，以及 anytime/parallelisable 的计算形态。

这些优势解释了课程为什么把 MCTS 放在 Go case study 前面：当完整 game tree 太大时，局部且可中途停止的搜索比一次性枚举整棵树更现实；但“可搜索”不等于“搜索一定充分”，仍要看 horizon、branching factor 和 leaf evaluation 质量。

### 2.2 Go case study：未知模型不是唯一困难

课件强调 Go 有约 2500 年历史，是经典的高难度棋类和长期 AI challenge；传统 game-tree search 在 Go 上没有直接解决问题。课件还问：下 Go 是否是在 dynamics 和 reward model unknown 的世界中学习？

应区分两个层面：Go 的规则、合法动作和终局胜负在正式对弈时是已知的，因此“规则未知”不是必要前提；困难在于长程战略、极大的有效分支和难以手工写出的局面价值。AlphaZero 通过 self-play 学习 heuristic，再在每个当前棋盘上做局部 MCTS。

## 3. 单局 move selection：PUCT 驱动的 search loop

与 Lecture 13 §4.2 的关系：前一讲已经说明 selection、expansion、network prediction 和 backup 的完整顺序。本节只补充课件在开场明确写出的 “Use PUCT”，并把数据接口和对手视角再检查一遍。

**本节路线图**

1. 从当前棋盘 root 开始，沿树选择待扩展叶节点。
2. 用网络同时给出 action probabilities 和 leaf value。
3. 把 value 传回祖先，重复 search，再用 root visit counts 形成 policy。
4. 只执行根动作，进入下一棋盘后重新搜索。

*首次完整讲解：Lecture 13 §4.2「单局 move selection：从根到 backup」。本节只补充：Lec14 课件把 tree policy 明确标为 PUCT。*

### 3.1 Selection 与 PUCT

在 AlphaZero 风格的 search 中，UCT 的“平均回报 + 不确定性 bonus”被网络 action prior 加以引导。课件没有在页面上给出 PUCT 展开式；沿用 Lecture 13 的标准补充：

$$
U(s,a)
=
c_{\mathrm{puct}}P_\theta(a\mid s)
\frac{\sqrt{\sum_bN(s,b)}}{1+N(s,a)},
\qquad
a
\in
\arg\max_a\bigl(Q(s,a)+U(s,a)\bigr).
$$

整体上，$Q(s,a)$ 表示树中已有 simulation 的价值统计，$P_\theta(a\mid s)$ 是网络给出的先验概率；访问少的边或网络先验高的边更容易获得下一次计算。$N(s,a)$、$Q(s,a)$ 在搜索中更新，$P_\theta$ 在该次 network evaluation 产生。PUCT 是 tree-selection rule，不是最终环境 reward，也不是把网络 policy 直接当成无需搜索的动作。

![[lec14-11.png|900]]

*图：来源 `lecture/lecture14post.pdf` 物理 PDF 第 11 页（课件内部页脚 6/31），原图展示搜索循环和对手视角切换；这是对 Lecture 13 同一来源动画的完整 frame。*

### 3.2 Expansion、network evaluation 与 backup

一次 simulation 的接口顺序如下：

1. 从根沿 PUCT 选择已有边，直到到达未展开或需要评估的叶节点。
2. 将叶状态输入网络，得到 $P_\theta(\cdot\mid s_{\mathrm{leaf}})$ 和 $v_\theta(s_{\mathrm{leaf}})$；前者用于扩展动作边，后者是叶节点的 value estimate。
3. 把叶 value 沿 selection path 反向传递，更新每条边的 $N,W,Q$。
4. 若当前节点属于对手，切换 value 视角；在二人零和棋类中，不能把同一个玩家的 max 规则机械地用于所有层。

**完整的小型 trace**：根 $s$ 先选边 $a_1$，到叶节点 $s_L$；网络给出 $v_\theta(s_L)=0.3$（从当前玩家视角），则沿路径令 $N(s,a_1)\leftarrow N(s,a_1)+1$，$W(s,a_1)\leftarrow W(s,a_1)+0.3$，再用 $Q=W/N$ 更新 selection statistics。如果路径下一层轮到对手，backup 到那一层时使用相反视角的 value，而不是继续把 $+0.3$ 当作同一方收益。

这次更新不会改变没有走过的 $a_2$ 分支；下一轮会用新的 $N,W,Q$ 和先验重新计算 PUCT。搜索结束后，才从根节点统计量产生真实动作 policy。

### 3.3 Root policy、temperature 与执行动作

重复 rollout/backup 多次后，课件给出根 policy 的比例形式；完整归一化写为

$$
\pi_\tau(a\mid s)
=
\frac{N(s,a)^{1/\tau}}
{\sum_{b\in\mathcal A}N(s,b)^{1/\tau}}.
$$

这个分布把“搜索花了多少计算在一条边上”转换为可采样的根动作分布。$\tau$ 越小，最大 visit count 的动作越占优势；$\tau$ 越大，分布越平滑。课件未规定一个统一的 temperature schedule，因此不要把某个 schedule 当成理论保证。

Self-play 的每一步只执行从根 policy 采样出的一个动作，然后把新状态当作下一次 search 的根；整局结束后才得到 win/loss。搜索树是否跨步复用是工程实现选择，课件的核心循环只要求“执行动作后重复整个 process”。

## 4. Self-play：从单局搜索到可学习的闭环

本节承接上一节的 root policy：一次 search 只解决一个当前 move，self-play 把许多这样的决策串成训练数据。Lecture 13 已第一次讲解 self-play 的机制，本讲更集中地解释“为什么匹配难度和 reward density 对 policy training 有帮助”。

**本节路线图**

1. 让当前系统和自己的副本对弈，避免依赖人工对手。
2. 保存每个局面的 search policy 和最终胜负结果。
3. 用这些 targets 训练 policy/value network，再回到下一轮搜索。

### 4.1 Self-play 的资源和 curriculum 效应

*首次完整讲解：Lecture 13 §4.3「根 policy、self-play 与 reward density」。本节只补充：本讲课件强调的训练解释。*

课件给出的 self-play 优势可以拆成三点：

- **计算替代人工对手**：训练瓶颈主要是 simulation 和 network evaluation，而不是持续雇用人类棋手。
- **匹配难度**：对手来自同一训练系统或相近版本，难度会随策略水平变化。
- **较有区分度的训练反馈**：双方接近时，胜负结果更能反映局面和决策差异，课件把它描述为一种 curriculum learning。

这里的“reward 更 dense”是相对说法。围棋的最终结果仍可能是终局 win/loss；课程意思是，与固定强弱严重不匹配的对手相比，self-play 产生的结果更能在训练过程中区分当前策略改进，而不是每一步都提供非零 reward。

### 4.2 Network targets 与训练闭环

*首次完整讲解：Lecture 13 §4.4「训练 policy/value network」。本节只补充：Lec14 对闭环的复习。*

对一个 self-play 局面 $s_t$，可以记录：

- MCTS 根访问计数形成的 target policy $\pi_\tau(\cdot\mid s_t)$；
- 从该局面当前玩家视角定义的最终结果 $z_t\in\{-1,+1\}$（平局时还需使用任务约定）；
- 作为输入的棋盘状态和合法动作 mask。

训练后的网络输出 policy prior 和 value，再供下一轮 MCTS 使用。数据流为

$$
\text{network}
\rightarrow
\text{PUCT-MCTS}
\rightarrow
\text{root policy}
\rightarrow
\text{self-play game}
\rightarrow
(s_t,\pi_\tau,z_t)
\rightarrow
\text{network update}.
$$

课件只要求理解“预测 policies and values”，没有在这套 deck 中指定损失函数、网络架构细节或训练轮数；这些应当与 AlphaZero 论文或具体实现分开阅读。

## 5. 读懂课件的 AlphaZero 证据

本节把课件的 summary、评估问题和四张图放在一起。图表支持的是有限的相对比较；它们不自动提供对所有组件的因果分解。

**本节路线图**

1. 先列出 self-play、strategic computation、selective search、averaging、local computation、learned heuristics。
2. 再分别看 architecture、MCTS 和整体性能图。
3. 最后讨论 human data 问题和 AlphaZero/AlphaTensor/AlphaDev 的概念延伸。

### 5.1 六个互补要素与三个评估问题

课件把 AlphaGo/AlphaZero 的要素总结为：

1. **Self-play**：从规则和终局反馈生成数据。
2. **Strategic computation**：在真正落子前使用额外搜索。
3. **Highly selective best-first search**：把节点扩展集中到有希望分支。
4. **Power of averaging**：多次模拟的统计聚合降低单次估计噪声。
5. **Local computation**：只解决当前局面，不预先完整求解所有局面。
6. **Learn and update heuristics**：policy/value network 随 self-play 数据更新。

课件随后提出：架构影响多大？在 policy/value function 上叠加 MCTS 的影响多大？与 human play 或 human data 相比如何？回答这些问题需要把组件、训练数据和计算预算控制住；单一 Elo 图通常只能回答“这些配置在该实验中表现如何”。

### 5.2 Architecture、MCTS 与整体性能图

![[lec14-18.png|900]]

*图：来源 `lecture/lecture14post.pdf` 物理 PDF 第 18 页（课件内部页脚 9/31），原图比较 dual-res、sep-res、dual-conv、sep-conv；它展示的是该实验中的相对差异。*

架构图中 dual-res bar 最高，sep-conv 最低，说明网络表示设计会影响最终棋力；这不是对任意任务的架构排序定理。

![[lec14-19.png|900]]

*图：来源 `lecture/lecture14post.pdf` 物理 PDF 第 19 页（课件内部页脚 10/31），原图的纵轴为 Elo rating，比较 raw network、AlphaGo Zero、AlphaGo Master、AlphaGo Lee、AlphaGo Fan 以及若干外部程序。*

这张图支持一个清楚的课程观察：对 raw network 加上选择性 MCTS 后，AlphaGo Zero 的 rating 显著上升；但不同系统的网络、训练数据和搜索设置也可能同时不同，所以不能把所有柱高差直接归因于一个组件。

![[lec14-20.png|900]]

*图：来源 `lecture/lecture14post.pdf` 物理 PDF 第 20 页（课件内部页脚 11/31），横轴为 days，纵轴为 Elo rating，虚线为 AlphaGo Master 与 AlphaGo Lee 的参考水平。*

曲线显示训练初期上升很快，之后逐渐接近并超过图中参考线。它回答的是“该训练运行中的性能随时间如何变化”，不等同于对所有 self-play 系统都能达到同一速度的保证。

### 5.3 是否需要 human data，以及更广的搜索问题

课件把 “Need for Human Data?” 作为评估问题，并把 AlphaZero 作为不依赖人类棋谱、通过 self-play 学习的对照思路。更严谨地说，图表可以说明该课程案例中的 self-play 系统不需要把人类棋局作为其核心训练数据；它不能推出所有任务都不需要人类先验，也不能抹去规则设计、网络归纳偏置和计算预算的作用。

课件还说同样的思想可用于 chess、shogi 和其他游戏，并列出 AlphaTensor（矩阵乘法）与 AlphaDev（排序算法）。共同抽象是：把“找到更好/更快的程序或策略”表示成一个非常大的搜索空间，用局部评估和学习到的 heuristic 逐步寻找高价值序列。这里是概念性连接，不是本讲对这些系统的算法复现。

## 6. UCT 与 upper-confidence-bound 复习

本节回到 Lecture 13 §3.2 的 UCT。课件把它作为可选理解检查，而不是重新引入一个新算法；重点是把 UCB 的 bandit 直觉准确地放回 MCTS 的 metalevel search。

**本节路线图**

1. 复习 upper confidence bound 如何结合利用和不确定性。
2. 区分真实环境探索与“选择下一笔模拟计算”。
3. 判断已知 reward/dynamics 时，UCB-style 算法何时还有用。

### 6.1 MCTS 适用场景的选择题

课件给出三种组合：短 horizon/小状态动作空间，长 horizon/大动作空间/小状态空间，长 horizon/大状态空间/小动作空间。不能只按“状态小”或“状态大”单项判断：MCTS 是否合适取决于每次模拟成本、有效 branching factor、horizon、可用的 value heuristic 和可接受的搜索预算。短 horizon、小 branching factor 通常最容易获得充分搜索；sampling 对大状态空间有帮助，但不能自动解决大动作空间或长 horizon。

### 6.2 UCT 的“奇怪之处”：探索的是计算资源

UCT 在每个树节点维护

$$
\text{mean return}+\text{upper-confidence bonus},
$$

并用它选择后续 rollout 的动作。upper-confidence bonus 的确能让未知或低访问动作获得计算，但这里的动作是 **模拟中的树边**；它的直接目标是选择下一笔 simulation，而非最小化真实在线交互的 regret。

因此，下面的两个说法不能混为一谈：

- UCB 在 bandit 中用于面对未知 reward 时平衡真实试验的探索与利用；
- UCT 借用相同形式来分配树搜索的计算预算。

在搜索结束后，根动作的质量取决于整个 search procedure 和 value estimate，而不是某个 UCT bonus 单独保证。

课件脚注把这种“决定下一笔计算”的问题联系到 metalevel reasoning；这里保留这个联系，但不把它扩展成另一套理论证明。

### 6.3 Upper-confidence-bound 复习题的答案边界

课件最后的三项判断是：

1. UCB 平衡 exploration 与利用已有信息以获得高 reward：**True**。
2. 这类算法可用于 bandit 和 MDP：**True**。
3. reward model 已知时，UCB 一定没有收益：**Depends on setting**。

在 bandit 中，如果 reward distribution 已完全知道，额外的置信区间通常没有信息收益；在 RL 中，即使 reward model 已知，若 dynamics 仍未知，探索 transition 仍可能有价值。反过来，如果 reward 和 dynamics 都已知且规划可直接完成，UCB-style uncertainty bonus 就没有必要。

## 7. Assignment Readiness

- **已覆盖的前置知识**：Lecture 13 的 MCTS/UCT、Lecture 12 的 model-based RL 与 optimism、Lecture 5--6 的 policy/value 表示。
- **本讲新增能力**：能按时间顺序解释 PUCT 搜索，区分 network prior、network value、tree $Q$ 和 root visit-count policy，并说明 self-play 如何产生训练 target。
- **尚未完成的实践**：没有运行 Go/AlphaZero 代码、重现 Elo 曲线、实现 PUCT 或做架构/MCTS 消融。
- **掌握证据**：目前没有独立实现、实验或口头推导记录；笔记 coverage 不等于 mastery。
- **下一步**：建议手动追踪两轮 root search，列出每个节点的 $N,W,Q,P$，再回答最后三道 UCT 复习题。

## 8. 本讲必会公式

1. **PUCT selection（标准补充）**：

   $$
   a\in\arg\max_a\left[Q(s,a)+c_{\mathrm{puct}}P_\theta(a\mid s)\frac{\sqrt{\sum_bN(s,b)}}{1+N(s,a)}\right].
   $$

   $Q$ 是 search estimate，$P_\theta$ 是 network prior，$N$ 是 tree visit count；它决定下一次 simulation 走哪条边。

2. **根 visit-count policy**：

   $$
   \pi_\tau(a\mid s)=\frac{N(s,a)^{1/\tau}}{\sum_bN(s,b)^{1/\tau}}.
   $$

   它不是网络原始 policy，而是搜索统计量经过温度变换后的执行/训练分布。

3. **UCT 复习形式**：

   $$
   \bar G(i,a)+c\sqrt{\frac{O(\log N(i))}{N(i,a)}}.
   $$

   该 bonus 服务于树内计算分配；不要把它直接解释为真实环境的 regret bound。

## 9. 容易混淆点

- **AlphaZero 的 network policy 与 root policy**：网络输出 $P_\theta(a\mid s)$ 是 prior；MCTS 后的 $\pi_\tau$ 来自 visit counts，两者功能不同。
- **PUCT 与 UCT**：UCT 使用树统计的均值和访问次数；PUCT 额外使用网络 action prior，仍然需要 tree $Q/N$ 的搜索反馈。
- **leaf value 与 terminal reward**：叶节点的 $v_\theta$ 是估计值；终局 win/loss 是实际 self-play outcome，二者不能互换。
- **self-play 的 reward density**：课程所说的“更 dense”是相对于匹配难度的训练信号，不是每一步都必有即时 reward。
- **human-data 结论**：案例图支持 AlphaZero 式 self-play 可以不依赖人类棋谱核心训练，但不构成所有任务都无需人类数据的普遍定理。
- **known reward 与 known dynamics**：只知道 reward 仍可能需要探索未知 transition；二者都知道且可直接规划时，UCB bonus 通常没有额外信息价值。
- **物理页码与课件页脚**：例如 AlphaZero loop 是物理第 11 页、页脚 6/31；引用时要说明两者，避免把构建序号当成独立物理页。

## 10. 自测题

1. PUCT 中 $P_\theta(a\mid s)$ 与 $Q(s,a)$ 各自回答什么问题？
2. 为什么 leaf value backup 到对手节点时要切换视角？
3. 若两个动作的 visit counts 是 100 和 25，$\tau=1$ 时根 policy 是什么？减小温度会如何改变分布？
4. self-play 为什么减少对人工对手的依赖？“matched opponent”对训练信号有什么作用？
5. 架构图和 MCTS 图各自能支持什么结论？哪些因果结论不能只由柱状图得出？
6. reward model 已知但 dynamics 未知时，为什么仍可能探索？reward 和 dynamics 都已知时呢？
7. UCT 在模拟树内的探索与 bandit 在真实环境中的探索，目标是否相同？

<details>
<summary>查看答案</summary>

1. $P_\theta$ 是网络对动作先验的判断，$Q$ 是当前树 simulation 的经验价值；前者指导未充分搜索的候选，后者总结已经得到的结果。
2. 二人零和中一个玩家的好结果通常是另一个玩家的坏结果；不切换视角会把对手的胜势错误地累加成自己的价值。
3. $\pi(a_1)=100/125=0.8$，$\pi(a_2)=25/125=0.2$；减小 $\tau$ 会放大 count 差异并使分布更尖锐。
4. 系统可以用计算产生对手；水平接近时，每局结果更能反映当前策略的细微改进，形成课程式难度变化。
5. 架构图比较了该实验的表示设计，MCTS 图比较了 raw network 与系统配置；它们不能单独隔离所有训练数据、计算预算、网络或搜索差异。
6. 未知 dynamics 仍决定动作后的状态和长期结果，所以 reward 已知不代表规划信息完整；若 reward/dynamics 都已知且可直接规划，UCB 的不确定性 bonus 通常没有额外作用。
7. 形式相似但目标不同：UCT 选择下一笔搜索计算，bandit UCB 选择真实环境动作以获取信息和 reward，不能直接把前者当成后者的 regret 最小化。

</details>

## 11. 本讲小结

Lecture 14 将 Lecture 13 的 MCTS 基础放进 AlphaZero 的完整闭环：PUCT 用网络 action prior 引导 tree selection，网络 value 在叶节点提供启发式，backup 更新树统计量，root visit counts 形成 policy，self-play 再把局面、搜索 policy 和终局结果送回网络训练。课件图表显示架构和 MCTS 都会影响该实验中的棋力，self-play 曲线可以超过参考系统；但这些证据必须按控制变量和数据来源谨慎解读。最后的 UCT 复习提醒我们：搜索层的计算分配和在线环境探索共享数学形式，却不自动共享同一个 regret 目标。

## 12. 延伸阅读

### 经典基础

- Silver et al., [Mastering the game of Go without human knowledge](https://doi.org/10.1038/nature24270), Nature 2017：AlphaGo Zero 的 self-play、network-guided MCTS 和课程图表来源。
- Silver et al., [A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play](https://doi.org/10.1126/science.aar6404), Science 2018：AlphaZero 的跨棋类算法。
- Kocsis and Szepesvári, [Bandit Based Monte-Carlo Planning](https://www.cs.utexas.edu/~shivaram/readings/b2hd-KocsisSzepesvari2006.html), 2006：UCT 与 bandit-style tree selection。
- Browne et al., [A Survey of Monte Carlo Tree Search Methods](https://doi.org/10.1109/TCIAIG.2012.2186810), 2012：MCTS 的方法综述。

### 前沿动态

截至 2026-08-17 核实：课件自身已将 AlphaTensor、AlphaDev 作为“把算法加速表示成大规模搜索”的延伸；本讲不额外加入未经本地课件要求的最新系统排名或泛化结论。
