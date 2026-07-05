# CS234 Lecture 1 Notes: Introduction to Reinforcement Learning

来源：`lecture/lec1/lecture1pre.pdf`，CS234 Winter 2026, Professor Emma Brunskill。

核验日期：2026-06-27。现代 AI 例子中涉及近年进展的部分，已用公开来源补充核验，链接放在文末。

## 0. 本讲 Checklist

- [x] 理解强化学习（reinforcement learning, RL）的基本定义：从经验/数据中学习，在不确定性下做长期有利的决策。
- [x] 理解 RL 的四个核心难点：优化（optimization）、延迟后果（delayed consequences）、探索（exploration）、泛化（generalization）。
- [x] 区分课程内容、现代应用连接、类比：robot learning、embodied intelligence、LLM agents、RLHF/PPO、planning、tool use。
- [x] 能把一个应用问题初步建模为序列决策过程（sequential decision process）：状态、动作、奖励、动态。
- [x] 理解历史（history）、状态（state）、观测（observation）之间的关系。
- [x] 理解 Markov 假设（Markov assumption）：当前状态足以预测未来。
- [x] 区分 bandit、POMDP、Markov process、MRP、MDP 的问题结构。
- [x] 理解模型（model）：转移/动态模型（transition/dynamics model）和奖励模型（reward model）。
- [x] 理解策略（policy）：确定性策略（deterministic policy）和随机策略（stochastic policy）。
- [x] 区分评估（evaluation）和控制（control）。
- [x] 理解 Markov chain / Markov process：没有奖励、没有动作，只有状态随机转移。
- [x] 理解 Markov reward process（MRP）：Markov chain 加奖励和折扣因子。
- [x] 掌握回报（return）、状态价值函数（state-value function）、折扣因子（discount factor）。
- [x] 掌握 MRP 的 Bellman 方程（Bellman equation）、矩阵形式、解析解和迭代动态规划算法。
- [x] 知道 Assignment 1 哪些部分已有概念基础，哪些还需要后续 lecture。

## 1. Lecture 1 的主线

本讲分三段：

1. RL 是什么，以及为什么它是现代 AI 的核心方法之一。
2. 课程结构：后续会学习 MDP 与 planning、model-free evaluation/control、policy search、offline RL、RLHF、DPO、exploration 和 advanced topics。
3. 从零建立序列决策的数学语言：状态、动作、奖励、转移、策略、Markov 假设、MRP、价值函数、Bellman 方程。

如果只记一句话：

RL 学的不是“单步预测正确”，而是在不确定环境中选择一串动作，使未来累计奖励的期望最大。

## 2. 为什么现在还要学 RL

课件开头用 2025 左右的例子说明 RL 仍然是 AI 的核心动力之一：大模型推理、数学竞赛、机器人控制、游戏、科学控制、公共政策优化、对话模型对齐等。

这些例子不要理解成“RL 已经解决所有智能问题”。更准确的理解是：只要问题包含多步决策、反馈、长期后果、探索或 reward design，RL 的语言就会变得自然。

### 2.1 课程内容

本讲对 RL 的定义是：

强化学习（reinforcement learning）是让智能体（agent）通过经验或数据学习，在不确定性下做出好决策。

这里的“好”不是主观判断，而是由奖励（reward）或效用（utility）刻画。后续全部数学都会围绕“如何定义并最大化长期奖励”展开。

### 2.2 现代连接：LLM reasoning 与 RL

课件提到 DeepSeek-R1-Zero、DeepSeek-R1 和 OpenAI o1 一类系统。它们与本讲的直接联系是：

- 语言模型可以被看作一个产生动作序列的策略（policy）：在 token、推理步骤、工具调用或答案之间做选择。
- 数学、代码等可验证任务能提供较明确的奖励信号（reward signal），因此适合用 RL 改善多步推理行为。
- 这不是本讲的 MRP/MDP 细节，但背后的核心问题相同：怎样让模型为了长期正确性而不是局部流畅性做决策。

要小心：LLM 里的“动作”不一定是机器人关节动作，也不一定是离散环境里的 move left / move right。它可以是生成下一个 token、选择一个工具、调用检索、写一段代码、反思当前解法等。这是概念映射，不是说 LLM agent 的训练环境一定完全满足本讲的 MDP 假设。

### 2.3 现代连接：embodied intelligence 与 robot learning

机器人学习（robot learning）和具身智能（embodied intelligence）是 RL 的自然应用场景，因为机器人必须在真实物理环境中连续行动：

- 状态可能包括相机图像、关节角、力传感器、任务指令。
- 动作可能是关节速度、末端执行器目标、移动底盘命令。
- 奖励可能来自任务成功、能耗、安全约束、碰撞惩罚。
- 延迟后果很强：现在夹取角度错一点，几秒后才导致失败。

这正是本讲后面会反复出现的结构：状态、动作、动态、奖励、策略和长期回报。

## 3. RL 一般包含的四个核心难点

课件列出四个关键词：优化（optimization）、延迟后果（delayed consequences）、探索（exploration）、泛化（generalization）。

### 3.1 优化（Optimization）

RL 的目标是找到最优或足够好的决策方式。这里的“方式”不是某一个动作，而是一套根据情况选动作的规则，也就是策略（policy）。

例如从城市 A 到城市 B 找最短路，如果地图完全已知，这更像 planning 或 graph search。如果地图未知、路况随机、行动还会改变未来信息，就会更接近 RL。

现代连接：

- 在 robot learning 中，优化目标可能是完成任务成功率、速度、能耗和安全性的综合。
- 在 RLHF 中，优化目标常由偏好模型或奖励模型近似人类偏好；这会带来 reward misspecification 和 reward hacking 的风险。
- 在 tool-use agents 中，优化目标可能是完成用户任务的成功率、成本、延迟和可靠性。

### 3.2 延迟后果（Delayed Consequences）

决策的好坏往往不是马上显现。现在拿钥匙可能没有即时奖励，但几十步后能打开门。现在省钱也许短期不舒服，但长期更好。

这带来两个问题：

- planning 时，要考虑动作的长期影响，而不只是即时收益。
- learning 时，要解决时间信用分配（temporal credit assignment）：后来的成功或失败到底该归因于前面哪些动作？

现代连接：

- LLM agent 如果第一步选错工具，可能后面所有推理都偏离；最终失败不能只归因于最后一句回答。
- 机器人装配任务中，早期姿态误差可能很晚才表现为插入失败。
- RLHF/PPO 里，一个完整回答可能只有最后得到偏好分数，训练时要把这个反馈分配到生成序列中的许多 token 或决策上。

### 3.3 探索（Exploration）

智能体通过行动了解世界。关键困难是：你只会观察到自己实际做出的动作的结果，而看不到没选动作的反事实结果。

例如选择去 Stanford 后，你不能同时亲身体验“如果当时去了 MIT 会怎样”。这种反事实缺失使 RL 比监督学习更难。

现代连接：

- Robot learning 中，真实机器人探索很贵、慢、还可能损坏设备，所以常结合仿真、离线数据、模仿学习和安全约束。
- Tool-use agents 中，探索新的工具调用序列可能浪费成本或造成错误，因此线上探索必须受控。
- RLHF 中，人类反馈昂贵，所以需要高效利用偏好数据。

### 3.4 泛化（Generalization）

策略（policy）是从经验到动作的映射。问题是：为什么不直接把所有情况都写死？

因为真实世界状态空间太大，机器人视觉、网页环境、对话状态和代码状态都不可能穷举。RL 必须学会在没见过的新状态中做合理决策。

现代连接：

- Atari、Go、机器人操作和 LLM agents 都依赖函数近似（function approximation）：用神经网络表示策略或价值函数。
- 泛化能力越强，越可能跨任务、跨对象、跨环境迁移；但越强的模型也可能学到意料之外的 reward hacking 策略。

## 4. 什么问题适合 RL

课件强调两类特别适合 RL 的问题：

1. 没有理想行为示例：例如目标是超越人类表现，或任务没有现成专家数据。
2. 搜索/优化空间巨大，并且结果有延迟：例如组合优化、游戏、复杂控制、长程推理。

不适合 RL 的典型情况：

- 只需要一次性预测，不需要行动影响未来。
- 有大量标注数据，监督学习已经能直接解决。
- 奖励难以定义且错误代价高，而探索又无法安全进行。

判断一个问题是否该用 RL，可以先问四个问题：

- 是否有明确的动作（action）？
- 动作是否影响未来状态或未来信息？
- 是否存在可以度量的长期目标？
- 是否需要在不确定性下权衡探索和利用？

## 5. 从 AI Tutor 例子学习序列决策建模

课件让我们把 AI tutor 建模为决策过程：

- 学生一开始不会加法和减法。
- tutor 可以给加法练习或减法练习。
- 学生做对给 +1，做错给 -1。

一个初步建模可以是：

- 状态（state）：学生对加法/减法的掌握程度，例如 `unknown_addition, unknown_subtraction`，也可以更细为掌握概率。
- 动作（action）：出加法题或减法题。
- 奖励（reward）：学生当前题做对为 +1，做错为 -1。
- 动态模型（dynamics model）：做某类题后，学生掌握程度如何变化。
- 策略（policy）：在不同学生状态下选择下一道练习题的规则。

关键反思：

如果奖励只看“当前题是否做对”，tutor 可能一直给容易的加法题，因为这能获得更多即时奖励。但这不一定最大化学习。更合理的奖励可能包括长期掌握、学习增益、覆盖难点、最终测试成绩等。

这就是 reward design 的核心：你优化什么，智能体就会朝什么方向走；如果奖励只是代理目标（proxy reward），策略可能钻空子。

现代连接：

- RLHF 的奖励模型也是代理目标：它近似人类偏好，但不是人类真实价值本身。
- Tool-use agents 只奖励“最终答案看起来对”，可能诱导模型不检索、不验证、或者编造中间步骤。
- Robot learning 只奖励“物体接近目标位置”，可能诱导碰撞、滑动或不稳定摆放。

## 6. 序列决策过程（Sequential Decision Process）

本讲的核心目标是建立序列决策语言：

在每个离散时间步（discrete time step）：

1. 智能体选择动作 $a_t$。
2. 世界根据动作更新。
3. 世界产生观测 $o_t$ 和奖励 $r_t$。
4. 智能体接收观测和奖励，再选择下一步动作。

这个循环很重要，因为 RL 不是静态数据集上的预测任务。智能体的动作会改变未来能看到的数据。

### 6.1 目标：最大化未来总奖励

序列决策的目标可以概括为：

$$
\text{choose actions to maximize expected future reward}
$$

这不是严格公式，而是本讲的目标语言。后面会把它变成回报（return）和价值函数（value function）。

这条目标在说什么：

智能体不是只追求当前一步的奖励，而是要在一段时间内获得尽可能高的累计收益。它要权衡短期收益和长期收益。

它解决什么问题：

如果只看即时奖励，tutor 会一直出简单题，机器人可能选择局部看似安全但无法完成任务的动作，LLM agent 可能选最省 token 的回答而不验证事实。未来奖励把“现在做铺垫”的价值纳入目标。

直觉：

下棋时一步棋本身没有奖励，但它改变后续局势。RL 要学的是“这一手对未来局势的影响”。

和算法/作业的关系：

Assignment 1 的 inventory 和 RiverSwim 都会体现这个目标：短期拿小奖励还是长期追求大回报，取决于 horizon 或 discount factor。

## 7. History、Observation、State

课件定义历史（history）为过去动作、观测、奖励的序列：

$$
h_t = (a_1, o_1, r_1, \ldots, a_t, o_t, r_t)
$$

这条公式在说什么：

到时间 $t$ 为止，智能体真正经历过的信息不是一个抽象状态，而是一串交互记录。它包括做过什么、看到什么、收到什么奖励。

它解决什么问题：

如果环境不是完全可观测的，只看当前观测 $o_t$ 可能不够。例如机器人只看到桌面局部图像，可能不知道某个物体之前是否被移动过；LLM agent 当前上下文也可能缺少外部世界的真实状态。

符号解释：

- $h_t$：时间 $t$ 的历史。
- $a_i$：第 $i$ 步动作。
- $o_i$：第 $i$ 步观测。
- $r_i$：第 $i$ 步奖励。

直觉：

历史就是智能体的“记忆”。状态则是我们希望从记忆中压缩出的、足够用于决策的摘要。

课件接着说，状态是历史的函数：

$$
s_t = \phi(h_t)
$$

这条公式在说什么：

状态 $s_t$ 可以由历史 $h_t$ 计算出来。函数 $\phi$ 表示我们如何把完整历史压缩为当前决策所需的信息。

它解决什么问题：

完整历史会越来越长，直接用完整历史做规划或学习很难。状态表示（state representation）试图保留对未来有用的信息，丢掉无关细节。

符号解释：

- $s_t$：时间 $t$ 的状态。
- $\phi$：从历史到状态的表示函数。
- $h_t$：截至时间 $t$ 的历史。

直觉：

医生不需要病人出生以来每秒的完整记录，但需要当前症状、病史、用药和关键检查结果。好的状态就是“足够充分的摘要”。

和现代 AI 的关系：

- LLM agent 的上下文窗口可以看作一种状态表示，但它未必 Markov，因为外部文件、网页、工具状态可能不在上下文里。
- Robot learning 中，把最近一帧图像当状态常常不够；需要速度、历史帧或 memory。

## 8. Markov 假设（Markov Assumption）

课件给出 Markov 条件：

$$
p(s_{t+1} \mid s_t, a_t)
=
p(s_{t+1} \mid h_t, a_t)
$$

这条公式在说什么：

如果当前状态 $s_t$ 已经包含了预测未来所需的全部信息，那么在给定 $s_t$ 和动作 $a_t$ 后，完整历史 $h_t$ 不再提供额外信息。

它解决什么问题：

Markov 假设让序列决策可以用局部递推处理。我们不需要对所有可能历史建模，只需要对状态之间的转移建模。后面的 Bellman 方程、value iteration、policy iteration 都依赖这种递推结构。

符号解释：

- $p(s_{t+1} \mid s_t, a_t)$：给定当前状态和动作，下一个状态的概率。
- $p(s_{t+1} \mid h_t, a_t)$：给定完整历史和动作，下一个状态的概率。
- $s_t$：当前状态。
- $a_t$：当前动作。
- $h_t$：到当前为止的完整历史。

直觉：

如果状态真的完整，那么“现在是什么情况”已经总结了“过去怎么到这里”。未来只需要看现在和接下来做什么。

### 8.1 最小数值例子

假设机器人当前状态记为 $s_t=3$，动作是向右。它可能通过两段不同历史到达位置 3：$h_t^{(1)}$ 和 $h_t^{(2)}$。

如果状态 3 已经包含预测下一步所需的全部信息，那么无论过去走过哪条路径，向右后到达状态 4 的概率都相同，例如：

$$
\begin{aligned}
p(s_{t+1}=4\mid s_t=3,a_t=\text{right})
&=
p(s_{t+1}=4\mid h_t^{(1)},a_t=\text{right})\\
&=
p(s_{t+1}=4\mid h_t^{(2)},a_t=\text{right})\\
&=
0.8
\end{aligned}
$$

这里的 0.8 不是 Markov 假设规定的固定数值；重点是给定当前状态和动作后，不同历史不会再改变这个概率。如果历史中还藏着“电池是否即将耗尽”而状态 3 没有记录它，那么这个状态表示就不是 Markov 的。

容易混淆点：

- Markov 不等于“没有过去”。它是说过去的相关信息已经被当前状态包含了。
- 当前观测 $o_t$ 不一定就是 Markov 状态。只在观测足够充分时，才可以近似令 $s_t = o_t$。
- 在 POMDP 中，单个观测通常不是 Markov 的，需要 belief state 或 history。

和算法/作业的关系：

Assignment 1 的 RiverSwim starter code 用 `T[state, action, next_state]` 表示转移概率。这个数组默认 `state` 已经是 Markov 状态，否则只用当前 state 预测 next_state 就不合理。

## 9. 序列决策问题的类型

课件用几个维度区分问题：

- 状态是否 Markov？
- 世界是否部分可观测？如果是，就进入 POMDP（partially observable Markov decision process）。
- 动态是确定性（deterministic）还是随机性（stochastic）？
- 动作只影响即时奖励，还是同时影响奖励和下一个状态？

几个概念的层级：

- Bandit：动作影响奖励，但通常不影响长期状态转移；没有完整的长期状态动态。
- Markov process / Markov chain：只有状态和转移，没有动作、没有奖励。
- MRP：Markov chain 加奖励和折扣，但仍没有动作。
- MDP：有状态、动作、转移、奖励和折扣，是本课程 planning/control 的核心对象。
- POMDP：真实状态不可完全观测，智能体只能看到观测。

现代连接：

- 推荐系统如果每次推荐会改变用户兴趣，就是 MDP；如果只看当前点击，不考虑长期变化，就更像 bandit。
- LLM tool-use 如果每次调用改变外部状态、文件系统或用户信念，就更像 MDP。
- 机器人任务通常至少是 MDP，实际更常是 POMDP，因为视觉和传感器有噪声、遮挡和延迟。

## 10. MDP 模型：Transition Model 与 Reward Model

课件用 Mars rover 说明 MDP。rover 位于一条离散路径上的若干状态，可以尝试向左或向右，左右端有不同奖励。

动态模型（transition/dynamics model）：

$$
p(s_{t+1} = s' \mid s_t = s, a_t = a)
$$

这条公式在说什么：

如果当前在状态 $s$，采取动作 $a$，那么下一个状态是 $s'$ 的概率是多少。

它解决什么问题：

环境可能随机。Mars rover 尝试向右可能成功，也可能被滑回去；机器人夹取可能成功，也可能打滑；LLM agent 调工具可能得到不同质量的结果。转移模型让我们能对不确定结果做期望计算。

符号解释：

- $s_t$：当前状态。
- $a_t$：当前动作。
- $s_{t+1}$：下一状态。
- $s'$：某个候选下一状态。
- $p(\cdot)$：条件概率。

直觉：

这是世界对动作的“反应规律”。

和代码的关系：

Assignment 1 RiverSwim 中：

- $P(s' \mid s,a)$ -> `T[state, action, next_state]`
- 例如 `T[0, 1, 1] = 0.4` 表示从最左状态选择 RIGHT 后，以 0.4 概率到右边相邻状态。

奖励模型（reward model）：

$$
r(s_t = s, a_t = a)
=
\mathbb{E}[r_t \mid s_t = s, a_t = a]
$$

这条公式在说什么：

在状态 $s$ 采取动作 $a$ 时，智能体预期能得到多少即时奖励。

它解决什么问题：

RL 需要把“目标”转成可优化的数值反馈。奖励模型告诉智能体每一步行动的即时收益，但长期目标还要通过回报和价值函数来表达。

符号解释：

- $r(s,a)$：状态动作对的期望即时奖励。
- $r_t$：时间 $t$ 实际收到的奖励随机变量。
- $\mathbb{E}[\cdot]$：期望。

直觉：

奖励模型只看一步，价值函数会把这一步和后续所有可能影响合在一起。

和代码的关系：

Assignment 1 RiverSwim 中：

- $R(s,a)$ -> `R[state, action]`
- `R[0, 0] = 0.005` 表示在最左状态选择 LEFT 可以得到小奖励。
- `R[5, 1] = 1.0` 表示在最右状态选择 RIGHT 可以得到大奖励。

## 11. 策略（Policy）

策略决定智能体如何选择动作。

确定性策略（deterministic policy）：

$$
\pi(s) = a
$$

这条公式在说什么：

每个状态 $s$ 都对应一个固定动作 $a$。只要处在同一个状态，策略总是选择同一个动作。

它解决什么问题：

这是最简单的决策规则形式，便于理解 policy evaluation 和 policy improvement。

符号解释：

- $\pi$：策略。
- $s$：状态。
- $a$：动作。

直觉：

像一张表：在状态 1 向右，在状态 2 向右，在状态 3 向左。

随机策略（stochastic policy）：

$$
\pi(a \mid s)
=
\Pr(a_t = a \mid s_t = s)
$$

这条公式在说什么：

策略不直接给出唯一动作，而是给出在状态 $s$ 下选择每个动作的概率分布。

它解决什么问题：

随机策略可以表达探索，也适合策略梯度（policy gradient）、PPO、最大熵 RL 等算法。

符号解释：

- $\pi(a \mid s)$：在状态 $s$ 下选择动作 $a$ 的概率。
- $\Pr(\cdot)$：概率。

直觉：

不是“永远向右”，而是“80% 向右，20% 向左”。

### 11.1 最小数值例子

在状态 $s$ 下，假设随机策略为：

$$
\pi(\text{right}\mid s)=0.8,
\qquad
\pi(\text{left}\mid s)=0.2
$$

它必须满足：

$$
\pi(\text{right}\mid s)
+
\pi(\text{left}\mid s)
=
1
$$

这表示如果在同一状态反复执行策略很多次，大约 80% 的时候向右，20% 的时候向左。确定性策略是特殊情况，例如向右概率为 1、向左概率为 0。

现代连接：

- PPO/RLHF 中，语言模型策略通常是随机策略：给定上下文，对下一个 token 或动作输出概率分布。
- Tool-use agents 的 tool selection 也可类比为策略：在当前任务状态下选择搜索、读文件、写代码、调用计算器或直接回答。

## 12. Evaluation 与 Control

课件区分两类核心任务：

- 评估（evaluation）：给定一个策略，估计它能获得多少期望回报。
- 控制（control）：寻找最优策略。

这一区分非常重要：

- Evaluation 问：“这个策略好吗？”
- Control 问：“什么策略最好？”

和 Assignment 1 的关系：

- `policy_evaluation(policy, R, T, gamma)` 是 evaluation。
- `policy_improvement(...)` 和 `value_iteration(...)` 是 control 的组成部分。

Lecture 1 只正式讲到 MRP 的价值计算；MDP 下的 policy evaluation、policy improvement、value iteration 会在后续 lecture 更系统地展开。

## 13. Markov Process / Markov Chain

Markov process 是没有动作、没有奖励的随机状态序列。

定义：

$$
\mathcal{M} = (\mathcal{S}, P)
$$

这条公式在说什么：

一个 Markov process 由状态集合 $\mathcal{S}$ 和转移模型 $P$ 组成。它描述世界自己如何随机演化，不涉及智能体选择动作，也不涉及奖励。

它解决什么问题：

这是后续 MRP 和 MDP 的基础。先理解“状态如何转移”，再加入奖励，最后加入动作。

符号解释：

- $\mathcal{S}$：有限状态集合。
- $P$：状态转移矩阵或转移模型。

直觉：

天气模型就是一个简单类比：晴、阴、雨之间按概率转换，但没有 agent 在做动作。

转移模型：

$$
P(s' \mid s)
=
\Pr(s_{t+1}=s' \mid s_t=s)
$$

这条公式在说什么：

在当前状态为 $s$ 时，下一个状态变成 $s'$ 的概率。

它解决什么问题：

它把随机过程从“可能怎么走”变成可计算的概率表。

如果有 $N$ 个状态，$P$ 可以写成矩阵。课件的 Mars rover Markov chain 用一个 $7 \times 7$ 矩阵表示从每个位置到下一个位置的概率。

### 13.1 贯穿后续章节的两状态例子

定义两个状态 $A$ 和 $B$：

- 从 $A$ 出发，下一步必然到 $B$。
- 从 $B$ 出发，下一步仍然留在 $B$。

因此：

$$
P(B\mid A)=1,
\qquad
P(B\mid B)=1
$$

若状态顺序为 $(A,B)$，转移矩阵是：

$$
P
=
\begin{bmatrix}
0 & 1\\
0 & 1
\end{bmatrix}
$$

矩阵每一行代表当前状态，每一列代表下一状态。例如第一行 $[0,1]$ 表示从 $A$ 到 $A$ 的概率为 0、到 $B$ 的概率为 1。

## 14. Markov Reward Process（MRP）

MRP 是 Markov chain 加上奖励和折扣：

$$
\mathcal{M} = (\mathcal{S}, P, R, \gamma)
$$

这条公式在说什么：

MRP 描述一个没有动作选择的随机过程，但每个状态会产生奖励，并且未来奖励会按折扣因子 $\gamma$ 折现。

它解决什么问题：

MRP 让我们可以先学习“给定一个过程，如何计算状态价值”，暂时不处理“怎么选动作”。这为后续 MDP 中“给定策略后的价值评估”做准备。

符号解释：

- $\mathcal{S}$：状态集合。
- $P$：状态转移模型。
- $R$：奖励函数。
- $\gamma$：折扣因子。

直觉：

如果策略已经固定，那么 MDP 会退化成 MRP：动作选择规则已定，剩下只是状态如何随机转移、奖励如何累积。

### 14.1 把两状态 Markov Process 扩展为 MRP

在 §13.1 的转移模型上加入：

$$
R(A)=1,
\qquad
R(B)=2,
\qquad
\gamma=0.5
$$

奖励向量为：

$$
R
=
\begin{bmatrix}
1\\
2
\end{bmatrix}
$$

现在这个 MRP 的具体含义是：在 $A$ 得到 1 分后进入 $B$；之后一直停留在 $B$，每一步得到 2 分；越远的奖励按 $0.5$ 折扣。§17–§20 会用同一个例子计算它的价值。

和算法/作业的关系：

后续在 MDP 中评估固定策略 $\pi$ 时，可以把“按 $\pi$ 选动作后的状态转移与奖励”看成一个 MRP。这是 policy evaluation 的核心。

## 15. Horizon、Return、Value Function

### 15.1 Horizon

horizon（时域长度）是每个 episode 中可交互的时间步数：

$$
H = \text{number of time steps in an episode}
$$

这条公式在说什么：

一个 episode 最多持续多少步。$H$ 可以是有限的，也可以是无限的。

本笔记后面默认用 $H$ 表示整个 episode 的总步数，并假设时间索引是：

$$
t = 0, 1, \ldots, H-1
$$

因此，从时间 $t$ 开始时，episode 里剩下的步数是 $H-t$。如果某个材料把 $H$ 用作“从当前时刻开始还剩几步”，公式会长得不一样；学习时要先确认 $H$ 到底表示总长度还是剩余长度。

#### 15.1.1 最小数值例子

如果：

$$
H=4
$$

那么 episode 的时间下标是：

$$
t=0,1,2,3
$$

当当前时间为 $t=1$ 时，还包含 $t=1,2,3$ 三个奖励位置，因此剩余步数是：

$$
H-t=4-1=3
$$

注意：最后一个时间下标是 $H-1=3$，但时间步总数仍然是 $H=4$。

它解决什么问题：

horizon 决定智能体有多远的未来需要考虑。短 horizon 会偏向即时收益，长 horizon 才可能让远期奖励变得重要。

和 Assignment 1 的关系：

Inventory 问题第一题专门考 horizon 如何改变最优策略。起始库存为 3 时，如果 horizon 太短，补货到 10 根本来不及，卖货可能更优；如果 horizon 足够长，最终满库存的 +100 会改变策略。

### 15.2 Return

课件定义回报（return）为从时间 $t$ 到 episode 结束的折扣奖励和。若 $H$ 表示整个 episode 的总步数，则：

$$
G_t
=
\sum_{k=0}^{H-1-t}
\gamma^k r_{t+k}
=
r_t
+ \gamma r_{t+1}
+ \cdots
+ \gamma^{H-1-t} r_{H-1}
$$

这条公式在说什么：

从当前时刻 $t$ 开始，智能体未来会收到一串奖励，一直到 episode 结束。回报 $G_t$ 把这些奖励加起来，但越远的奖励乘上越高次的 $\gamma$，因此通常权重更小。

它解决什么问题：

单步奖励不能表达长期目标。回报把整个 episode 的长期表现压成一个数，让“策略好不好”可以被比较。

符号解释：

- $G_t$：从时间 $t$ 开始的回报。
- $r_t$：当前奖励。
- $\gamma$：折扣因子，$\gamma \in [0,1]$。
- $H$：整个 episode 的总步数。
- $H-1-t$：从时间 $t$ 开始还能往后累加的最大偏移量。

#### 15.2.1 最小数值例子

沿用 $H=4$，从 $t=1$ 开始，假设：

$$
r_1=2,
\qquad
r_2=4,
\qquad
r_3=8,
\qquad
\gamma=0.5
$$

那么：

$$
\begin{aligned}
G_1
&=
r_1+\gamma r_2+\gamma^2r_3\\
&=
2+0.5\times 4+0.5^2\times 8\\
&=
6
\end{aligned}
$$

虽然三个原始奖励相加是 14，但折扣后回报是 6。当前奖励 2 完整计入，下一步奖励 4 贡献 2，再下一步奖励 8 也只贡献 2。

如果把 $H_{\text{rem}}$ 表示为“从时间 $t$ 开始剩余的步数”，也可以写成：

$$
G_t^{(H_{\text{rem}})}
=
\sum_{k=0}^{H_{\text{rem}}-1}
\gamma^k r_{t+k}
$$

这两个写法并不矛盾，区别只是 $H$ 的含义不同。做 Assignment 1 的 effective horizon 时，必须先确认题目里的 horizon 指的是总长度还是从当前开始的剩余长度。

直觉：

现在的 1 分通常比很久之后的 1 分更直接；但如果 $\gamma$ 接近 1，未来奖励依然很重要。

和 Assignment 1 的关系：

Inventory 的有限 horizon 与 RiverSwim 的折扣因子都在改变同一个东西：未来奖励在决策中的重要性。

### 15.3 State Value Function

MRP 的状态价值函数（state-value function）：

$$
V(s)
=
\mathbb{E}[G_t \mid s_t = s]
=
\mathbb{E}
\left[
\sum_{k=0}^{H-1-t}
\gamma^k r_{t+k}
\mid s_t = s
\right]
$$

这条公式在说什么：

状态 $s$ 的价值是：如果现在从 $s$ 出发，按照这个随机过程继续走，未来折扣累计奖励的期望是多少。

它解决什么问题：

价值函数把“当前状态好不好”量化了。它不是当前奖励，而是从这个状态出发未来可能获得的总收益。

符号解释：

- $V(s)$：状态 $s$ 的价值。
- $\mathbb{E}[\cdot]$：对未来随机转移和奖励取期望。
- $G_t$：从当前开始的回报。
- $s_t=s$：当前状态为 $s$。

直觉：

在 Mars rover 中，中间状态即时奖励可能是 0，但如果它很可能通向右端 +10，那么它仍然有高价值。

#### 15.3.1 最小数值例子

假设从状态 $s$ 出发重复运行三次，观察到的回报分别为：

$$
G^{(1)}=2,
\qquad
G^{(2)}=6,
\qquad
G^{(3)}=4
$$

那么可以用样本平均近似价值：

$$
\begin{aligned}
V(s)
&=
\mathbb{E}[G_t\mid s_t=s]\\
&\approx
\frac{2+6+4}{3}\\
&=
4
\end{aligned}
$$

单次回报可能是 2、6 或 4；价值不是其中某一次结果，而是重复从该状态出发时回报的平均水平。Lecture 3 会正式把这种样本平均发展成 Monte Carlo policy evaluation。

现代连接：

- 在 LLM agent 中，一个中间推理状态可能还没有产生最终答案，但如果它让后续更容易成功，它就有高“价值”。
- 在机器人操作中，把杯子抓稳这个中间状态本身未必有奖励，但它提高最终倒水成功率，因此价值高。

## 16. 折扣因子（Discount Factor）

折扣因子 $\gamma$ 满足：

$$
\gamma \in [0,1]
$$

这条公式在说什么：

$\gamma$ 控制未来奖励的重要性。

它解决什么问题：

在无限 horizon 下，如果一直拿正奖励，未折扣总和可能发散。$\gamma < 1$ 可以让无限和有限，并且在行为上表达“近的奖励更重要”。

符号解释：

- $\gamma=0$：只关心即时奖励。
- $\gamma=1$：未来奖励和当前奖励同等重要；通常要求 episode 有限，否则可能发散。

直觉：

如果 $\gamma$ 很小，智能体近视；如果 $\gamma$ 接近 1，智能体愿意为了远期大奖忍受短期代价。

和 Assignment 1 的关系：

RiverSwim 中，向左有稳定小奖励，向右逆流而上有远期大奖励。$\gamma$ 小时，最优策略可能留在左边拿小奖励；$\gamma$ 大时，未来大奖励足以 justify 长期向右探索。

## 17. MRP 的 Bellman 方程

课件给出 MRP 的价值递推：

$$
V(s)
=
R(s)
+
\gamma
\sum_{s' \in \mathcal{S}}
P(s' \mid s) V(s')
$$

这条公式在说什么：

状态 $s$ 的价值等于两部分：

1. 当前状态的即时奖励 $R(s)$。
2. 下一个状态价值的期望，再乘以折扣因子 $\gamma$。

这就是 Bellman 方程的核心思想：一个长期价值可以拆成“一步奖励 + 折扣后的未来价值”。

它解决什么问题：

直接计算 $V(s)$ 要考虑所有可能未来轨迹，数量会爆炸。Markov 性让我们只需要看一步转移，再递归引用下一状态的价值。这是动态规划（dynamic programming）的基础。

符号解释：

- $V(s)$：状态 $s$ 的价值。
- $R(s)$：状态 $s$ 的期望即时奖励。
- $\gamma$：折扣因子。
- $\mathcal{S}$：状态集合。
- $P(s' \mid s)$：从 $s$ 转移到 $s'$ 的概率。
- $V(s')$：下一状态 $s'$ 的价值。

直觉：

一个位置值多少钱，不只看这里有没有奖励，还看从这里出发大概率会到哪里。如果大概率通向好地方，它现在就值钱。

### 17.1 最小数值例子：求两状态 MRP 的价值

使用 §13.1–§14.1 的 MRP：

$$
P
=
\begin{bmatrix}
0 & 1\\
0 & 1
\end{bmatrix},
\qquad
R
=
\begin{bmatrix}
1\\
2
\end{bmatrix},
\qquad
\gamma=0.5
$$

状态 $B$ 会永远回到自身，所以：

$$
V(B)=2+0.5V(B)
$$

移项得到：

$$
V(B)=4
$$

状态 $A$ 下一步必然到 $B$：

$$
\begin{aligned}
V(A)
&=
1+0.5V(B)\\
&=
1+0.5\times 4\\
&=
3
\end{aligned}
$$

因此：

$$
V(A)=3,
\qquad
V(B)=4
$$

这个结果把 Bellman 方程的“即时奖励 + 折扣未来价值”变成了具体数字。$B$ 的价值 4 包含当前及未来持续获得的奖励 2；$A$ 的价值则是当前奖励 1 加上通往 $B$ 后的折扣价值 2。

和算法/作业的关系：

Assignment 1 的 `bellman_backup` 会在 MDP 中做类似计算，只是 MDP 版本还要给定动作：

$$
\text{backup}(s,a)
=
R(s,a)
+
\gamma
\sum_{s' \in \mathcal{S}}
T(s,a,s') V(s')
$$

*后续完整讲解：Lecture 2 §13「State-Action Value」和 §26.1 `bellman_backup`。这里先说明它是 MRP Bellman 方程加入动作后的自然延伸。*

这条 MDP backup 是 lecture 1 的自然延伸，但正式的 MDP control 还会在后续 lecture 继续讲。

代码变量映射：

- $V(s)$ -> `V[state]` 或 `value_function[state]`
- $R(s,a)$ -> `R[state, action]`
- $T(s,a,s')$ -> `T[state, action, next_state]`
- $\gamma$ -> `gamma`

## 18. Bellman 方程的矩阵形式

有限状态 MRP 可以写成：

$$
V = R + \gamma P V
$$

*标量形式首次完整讲解：本讲 §17「MRP 的 Bellman 方程」。这里把所有状态方程合并成矩阵形式。*

这条公式在说什么：

把所有状态的价值放进向量 $V$，所有状态奖励放进向量 $R$，所有转移概率放进矩阵 $P$，那么每个状态的 Bellman 方程可以合并成一个矩阵方程。

它解决什么问题：

矩阵形式让我们看到价值函数是一个线性方程组，也让解析解和算法复杂度更清楚。

符号解释：

- $V$：长度为 $|\mathcal{S}|$ 的价值向量。
- $R$：长度为 $|\mathcal{S}|$ 的奖励向量。
- $P$：$|\mathcal{S}| \times |\mathcal{S}|$ 的转移矩阵。
- $\gamma$：折扣因子。

直觉：

每个状态的价值都依赖其他状态的价值，所以它们必须一起求解。

### 18.1 用矩阵验证两状态例子

把 §17.1 的结果写成向量：

$$
V
=
\begin{bmatrix}
3\\
4
\end{bmatrix}
$$

代入矩阵 Bellman 方程右侧：

$$
\begin{aligned}
R+\gamma PV
&=
\begin{bmatrix}
1\\
2
\end{bmatrix}
+
0.5
\begin{bmatrix}
0 & 1\\
0 & 1
\end{bmatrix}
\begin{bmatrix}
3\\
4
\end{bmatrix}\\
&=
\begin{bmatrix}
1\\
2
\end{bmatrix}
+
0.5
\begin{bmatrix}
4\\
4
\end{bmatrix}\\
&=
\begin{bmatrix}
3\\
4
\end{bmatrix}\\
&=
V
\end{aligned}
$$

这说明 $V=[3,4]^\top$ 确实是自洽解：把它代回 Bellman 方程，右侧仍然得到同一个向量。

## 19. MRP 价值函数的解析解

从矩阵 Bellman 方程出发：

$$
V = R + \gamma P V
$$

*首次完整讲解：本讲 §18「Bellman 方程的矩阵形式」。下面只进行代数移项并求解析解。*

移项：

$$
V - \gamma P V = R
$$

合并：

$$
(I - \gamma P)V = R
$$

解析解：

$$
V = (I - \gamma P)^{-1}R
$$

这组公式在说什么：

MRP 的价值函数可以通过解线性方程组得到。只要矩阵 $I-\gamma P$ 可逆，就能直接求出所有状态的价值。

它解决什么问题：

这说明价值函数不是神秘对象，而是 Bellman 方程的解。对小规模有限 MRP，可以直接用线性代数求解。

符号解释：

- $I$：单位矩阵。
- $(I-\gamma P)^{-1}$：矩阵逆。
- 其他符号同上。

直觉：

价值函数是一个自洽解：你猜的每个状态价值必须和“一步奖励 + 后续价值期望”一致。

算法关系：

直接矩阵求逆通常复杂度约为 $O(N^3)$，$N=|\mathcal{S}|$。状态很多时，这会很贵，因此实际常用迭代方法。

### 19.1 最小数值例子：直接求逆

对两状态 MRP：

$$
I-\gamma P
=
\begin{bmatrix}
1 & -0.5\\
0 & 0.5
\end{bmatrix}
$$

它的逆为：

$$
(I-\gamma P)^{-1}
=
\begin{bmatrix}
1 & 1\\
0 & 2
\end{bmatrix}
$$

因此：

$$
\begin{aligned}
V
&=
(I-\gamma P)^{-1}R\\
&=
\begin{bmatrix}
1 & 1\\
0 & 2
\end{bmatrix}
\begin{bmatrix}
1\\
2
\end{bmatrix}\\
&=
\begin{bmatrix}
3\\
4
\end{bmatrix}
\end{aligned}
$$

解析解与 §17.1 逐状态求出的结果完全一致。

## 20. 迭代动态规划计算 MRP 价值

课件给出迭代算法：

初始化：

$$
V_0(s) = 0,
\quad \forall s \in \mathcal{S}
$$

迭代更新：

$$
V_k(s)
=
R(s)
+
\gamma
\sum_{s' \in \mathcal{S}}
P(s' \mid s)V_{k-1}(s')
$$

*Bellman 方程首次完整讲解：§17；解析 fixed point：§19。本节的新内容是从初始猜测反复更新，逐步逼近该 fixed point。*

这条公式在说什么：

先从一个简单的价值估计开始，例如所有状态价值为 0。第 $k$ 轮更新时，用上一轮的价值估计 $V_{k-1}$ 来计算新的 $V_k$。不断重复，直到价值收敛。

它解决什么问题：

避免直接矩阵求逆。每轮只做局部的 Bellman backup，逐步传播未来奖励信息。

符号解释：

- $V_k(s)$：第 $k$ 次迭代后对状态 $s$ 的价值估计。
- $V_{k-1}(s')$：上一轮对下一状态 $s'$ 的价值估计。
- 其他符号同 MRP Bellman 方程。

直觉：

第一轮只知道即时奖励；第二轮开始知道“一步之后的奖励”；更多轮后，远处奖励逐渐向前传播。

### 20.1 最小数值例子：观察迭代如何收敛

对同一个两状态 MRP，从 $V_0(A)=V_0(B)=0$ 开始：

$$
\begin{aligned}
V_k(A)&=1+0.5V_{k-1}(B),\\
V_k(B)&=2+0.5V_{k-1}(B)
\end{aligned}
$$

前几轮结果为：

| 迭代轮数 $k$ | $V_k(A)$ | $V_k(B)$ |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 1 | 2 |
| 2 | 2 | 3 |
| 3 | 2.5 | 3.5 |
| 4 | 2.75 | 3.75 |
| $\infty$ | 3 | 4 |

每轮都会把 $B$ 中更远一步的奖励向当前价值传播。迭代结果逐渐接近解析解 $V(A)=3,V(B)=4$，但不需要显式计算矩阵逆。

计算复杂度：

如果有 $N$ 个状态，朴素实现每轮需要大约 $O(N^2)$，因为每个状态都要对所有下一状态求和。

和 Assignment 1 的关系：

Value iteration 会把这个思想推广到有动作的 MDP：对每个状态尝试所有动作，选择 Bellman backup 最大的动作。现在还没到完整算法，但 `bellman_backup` 的核心计算已经能从这里看出来。

## 21. RL Agent 的组成：Model、Policy、Value Function

课件总结 RL agent 常包含一个或多个组件：

- 模型（model）：预测状态如何转移、奖励如何产生。
- 策略（policy）：决定如何选动作。
- 价值函数（value function）：评估状态或状态-动作对的长期好坏。

model-based agent：

- 显式拥有模型。
- 可以通过 planning 利用模型推演未来。

model-free agent：

- 不显式学习或使用环境模型。
- 通常直接学习价值函数、策略，或二者都有。

现代连接：

- Robot learning 中，model-based 方法可以先学动力学模型，再做 planning；model-free 方法可以直接学控制策略。
- LLM agents 中，显式世界模型较弱时，常用搜索、工具调用、self-reflection 或外部 simulator 来弥补。
- Tool-use planning 可以看作 model-based 思路：先预测调用某工具会产生什么信息，再决定下一步。

## 22. Mars Rover: $\gamma = 0$ 的 Policy Evaluation

课件最后用 Mars rover 的固定策略说明：

- 策略：每个状态都尝试向右。
- 折扣因子：$\gamma = 0$。
- 问：这个策略的价值是多少？

策略价值函数（policy value function）：

$$
V^\pi(s)
=
\mathbb{E}_\pi
\left[
r_t
+ \gamma r_{t+1}
+ \gamma^2 r_{t+2}
+ \cdots
\mid s_t = s
\right]
$$

*状态价值首次完整讲解：本讲 §15.3。这里进一步强调，同一个状态的价值取决于所执行的具体策略 $\pi$。*

这条公式在说什么：

如果从状态 $s$ 出发，并且之后一直按照策略 $\pi$ 行动，那么未来折扣累计奖励的期望就是 $V^\pi(s)$。

它解决什么问题：

同一个 MDP 中，不同策略会产生不同价值。$V^\pi$ 让我们能评价某个给定策略，而不是直接谈“状态本身绝对好坏”。

符号解释：

- $V^\pi(s)$：策略 $\pi$ 下状态 $s$ 的价值。
- $\mathbb{E}_\pi$：假设动作由策略 $\pi$ 选择时，对未来随机性取期望。
- $\gamma$：折扣因子。

直觉：

状态价值要和策略绑定。站在同一个位置，如果你的策略总是往大奖励方向走，这个状态就值钱；如果策略总是原地绕圈，价值可能很低。

当 $\gamma = 0$ 时：

$$
V^\pi(s) = r(s)
$$

这条公式在说什么：

如果完全不关心未来，价值就等于当前即时奖励。后续所有奖励都被 $\gamma=0$ 消掉了。

它解决什么问题：

它帮助我们理解折扣因子的极端情况。$\gamma=0$ 的 agent 是完全短视的。

直觉：

只看眼前一步，远处的 +10 对现在没有任何吸引力。

最小数值例子：若当前状态的即时奖励为 $r(s)=0.005$，即使未来可能到达奖励 10 的状态，在 $\gamma=0$ 时仍有：

$$
V^\pi(s)=0.005
$$

因为所有未来项都至少乘有一个 $\gamma$，所以会被置为 0。

和 Assignment 1 的关系：

RiverSwim 中，如果 $\gamma$ 太小，最右边大奖励传播不到最左边，agent 就可能选择在左边拿稳定小奖励。

## 23. Assignment 1 准备度

Assignment 1 包含四题：

1. Effective horizon：库存 MDP，考 horizon 和 discount factor 如何影响最优策略。
2. Reward hacking：自动驾驶 proxy reward 导致不合意策略。
3. Bellman residuals and performance bounds：Bellman operator、contraction、greedy policy、performance bound。
4. RiverSwim MDP：实现 Bellman backup、policy evaluation、policy improvement、policy iteration、value iteration。

Lecture 1 后的准备度：

- Q1：已经有基础概念，包括 horizon、discount factor、长期奖励和短视/远视行为。但要严谨求最优策略，还需要后续 MDP planning。
- Q2：已经可以开始理解 reward hacking 的概念，并能写出直觉性解释；但最好结合后续 reward design 和 policy optimization 再完善。
- Q3：还不能正式开始。需要 MDP Bellman optimality operator、contraction、policy value 等更完整内容。
- Q4：还不能正式开始写代码。`bellman_backup` 的数学雏形已出现，但 policy iteration 和 value iteration 需要后续 lecture。

因此：现在还不建议正式开始 Assignment 1。下一讲学完 MDP planning 相关内容后，会更接近 Assignment 1 的 coding 部分。

## 24. 本讲必会公式

Markov 假设（首次完整讲解：§8）：

$$
p(s_{t+1} \mid s_t, a_t)
=
p(s_{t+1} \mid h_t, a_t)
$$

MDP 转移模型（首次完整讲解：§10）：

$$
p(s_{t+1} = s' \mid s_t = s, a_t = a)
$$

奖励模型（首次完整讲解：§10）：

$$
r(s_t = s, a_t = a)
=
\mathbb{E}[r_t \mid s_t = s, a_t = a]
$$

确定性策略（首次完整讲解：§11）：

$$
\pi(s) = a
$$

随机策略（首次完整讲解：§11）：

$$
\pi(a \mid s)
=
\Pr(a_t = a \mid s_t = s)
$$

回报（首次完整讲解：§15.2）：

$$
G_t
=
\sum_{k=0}^{H-1-t}
\gamma^k r_{t+k}
$$

MRP 状态价值函数（首次完整讲解：§15.3）：

$$
V(s)
=
\mathbb{E}[G_t \mid s_t = s]
$$

MRP Bellman 方程（首次完整讲解：§17）：

$$
V(s)
=
R(s)
+
\gamma
\sum_{s' \in \mathcal{S}}
P(s' \mid s)V(s')
$$

矩阵 Bellman 方程（首次完整讲解：§18）：

$$
V = R + \gamma P V
$$

解析解（首次完整讲解：§19）：

$$
V = (I - \gamma P)^{-1}R
$$

迭代更新（首次完整讲解：§20）：

$$
V_k(s)
=
R(s)
+
\gamma
\sum_{s' \in \mathcal{S}}
P(s' \mid s)V_{k-1}(s')
$$

策略价值函数（首次完整讲解：§22）：

$$
V^\pi(s)
=
\mathbb{E}_\pi
\left[
r_t
+ \gamma r_{t+1}
+ \gamma^2 r_{t+2}
+ \cdots
\mid s_t = s
\right]
$$

## 25. 容易混淆点

1. 状态不是观测。

   观测是 agent 看到的东西；状态是我们假设足以预测未来的表示。很多实际问题中，当前观测并不 Markov。

2. 奖励不是价值。

   奖励是当前或一步反馈；价值是未来折扣累计奖励的期望。

3. $\gamma$ 小不是“更保守”，而是更短视。

   在某些任务中短视看起来安全，但它可能错过长期高收益；在另一些任务中，过度重视远期代理奖励可能导致 reward hacking。

4. Evaluation 和 control 不同。

   Evaluation 是评估给定策略；control 是寻找最优策略。Assignment 1 的代码会同时出现这两类任务。

5. Model-based 和 model-free 的区别在于是否显式使用环境模型。

   有价值函数不等于 model-based；有策略也不等于 model-free。关键是是否显式建模 $P$ 和 $R$ 并用它规划。

## 26. 自测题

1. 如果一个机器人只用当前摄像头图像作为状态，但物体速度不可见，这个状态一定 Markov 吗？

   不一定。速度可能影响下一步状态。如果单帧图像不能恢复速度，就需要历史帧、状态估计或 belief state。

2. 在 AI tutor 例子中，如果奖励只给“学生当前题做对”，会出现什么问题？

   agent 可能反复给简单题以获得即时奖励，而不是最大化长期学习效果。这是 proxy reward 可能导致的 reward hacking。

3. 为什么 $\gamma=0$ 时 $V^\pi(s)=r(s)$？

   因为所有未来奖励项都乘上 $\gamma, \gamma^2,\ldots$，当 $\gamma=0$ 时这些项为 0，只剩即时奖励。

4. MRP 和 MDP 的区别是什么？

   MRP 没有动作选择；MDP 有动作，策略会影响转移和奖励。给定一个固定策略后，MDP 可以诱导出一个 MRP。

5. Bellman 方程为什么重要？

   它把长期回报分解成一步奖励和下一状态价值，使价值计算可以用动态规划、迭代更新和后续的 value iteration 来完成。

## 27. 现代方向定位图

课程概念 -> 现代方向：

- 策略（policy） -> 机器人控制策略、LLM token/action policy、tool selection policy。
- 奖励（reward） -> task success、human preference、unit test pass rate、safety penalty。
- 回报（return） -> 长程任务成功、对话整体质量、机器人完整操作成功率。
- Markov state -> 机器人状态估计、agent memory、belief state、workspace state。
- Bellman 方程 -> planning、value iteration、policy evaluation、critic learning。
- 探索（exploration） -> robot trial-and-error、agent search、tool-use strategy discovery。
- Reward hacking -> RLHF 对齐问题、proxy metrics、自动驾驶/推荐系统目标错配。

需要区分三类关系：

- 直接相关：PPO 是 RL 算法，RLHF 论文中使用 RL 从人类偏好优化语言模型；后续 CS234 会学 policy gradient/PPO。
- 概念类比：把 LLM 的 tool choice 看成 policy，有助于理解 agent 行为，但不代表它一定按标准 MDP 训练。
- 课程未覆盖但相关：大规模机器人 foundation policy、LLM reasoning RL、multi-agent tool ecosystems 需要更多深度学习和系统知识。

## 28. 本讲小结

Lecture 1 建立了 RL 的基本世界观：

RL 研究的是智能体如何在不确定环境中，通过行动、反馈和长期目标，学习做出好决策。

本讲真正要掌握的不是某个算法，而是这套语言：

- 状态 $s$
- 动作 $a$
- 奖励 $r$
- 转移模型 $P$
- 策略 $\pi$
- 回报 $G_t$
- 价值函数 $V$
- Bellman 递推

下一讲会把这些基础推向 MDP 中的 planning 和 control：如何在有动作的情况下计算最优价值和最优策略。这将直接连接 Assignment 1 的 Bellman backup、policy iteration 和 value iteration。

## 29. 核验来源与延伸阅读

- DeepSeek-R1 论文：DeepSeek-AI 描述了通过强化学习激励 LLM 推理能力，并强调无需人工标注推理轨迹的纯 RL 方向。<https://arxiv.org/abs/2501.12948>
- Google DeepMind IMO 2025：官方博客记录 Gemini Deep Think 获得 35/42 的 gold-medal level performance。这里作为现代数学推理系统的背景例子，不把它当作本讲 MRP/MDP 算法细节。<https://deepmind.google/discover/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/>
- OpenAI o1：OpenAI 介绍 o1 时说明大规模强化学习用于训练模型更有效地使用 chain of thought。<https://openai.com/index/learning-to-reason-with-llms/>
- Physical Intelligence $\pi_0$：官方博客介绍 generalist robot policy，覆盖多机器人和多任务数据。<https://www.pi.website/blog/pi0>
- PPO 原论文：Schulman et al., "Proximal Policy Optimization Algorithms", 2017。<https://arxiv.org/abs/1707.06347>
- InstructGPT / RLHF 论文：Ouyang et al., "Training language models to follow instructions with human feedback", 2022。<https://arxiv.org/abs/2203.02155>
