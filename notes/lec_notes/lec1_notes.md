# CS234 Lecture 1 Notes: Introduction to Reinforcement Learning

来源：`lecture/lec1/lecture1pre.pdf`，CS234 Winter 2026, Professor Emma Brunskill。

笔记规范：`cs234-rl-tutor v2`。下方 checklist 只表示课件内容已覆盖，不表示学习者已经掌握。

外部资料核验日期：2026-07-10。现代 AI 例子中涉及近年进展的部分，已用公开来源补充核验，链接放在文末。

## 0. 本讲覆盖清单

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

### 6.1 时间步（Time Step）

**时间步（time step）**：智能体执行一次完整决策循环的单位时间。

**一个时间步包含**：

1. 接收观测 $o_t$（完全可观测时 $o_t = s_t$）
2. 选择动作 $a_t$
3. 环境返回奖励 $r_t$
4. 环境转移到新状态 $s_{t+1}$（智能体可能看不到完整状态）

**具体例子（自拟资源采集任务，$H=5$）**：

```
t=0: 在 s_0 → 选择 Collect → 得到 r_0=+10 → 停留在 s_0
t=1: 在 s_0 → 选择 Move    → 得到 r_1=0   → 转移到 s_1
t=2: 在 s_1 → 选择 Collect → 得到 r_2=+1  → 停留在 s_1
t=3: 在 s_1 → 选择 Move    → 得到 r_3=0   → 转移到 s_2
t=4: 在 s_2 → 选择 Collect → 得到 r_4=+1  → episode 结束
```

每一行是一个时间步。**是的，一个时间步对应一次决策和执行。**

**关键点**：
- 时间步是离散的（不是连续时间）
- 每个时间步智能体只做一次决策
- RL 不是静态数据集上的预测任务——智能体的动作会改变未来能看到的数据

### 6.2 目标：最大化未来总奖励

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

### 7.1 历史（History）

**历史 $h_t$**：从开始到时刻 $t$，智能体经历的所有动作、观测、奖励的完整记录。

**数学定义**：

$$
h_t = (a_1, o_1, r_1, a_2, o_2, r_2, \ldots, a_t, o_t, r_t)
$$

**具体例子**（机器人导航，前 3 步）：

```
t=1: 执行动作 “向右” → 观测到 “墙壁” → 得奖励 -1
t=2: 执行动作 “向左” → 观测到 “空地” → 得奖励 0
t=3: 执行动作 “前进” → 观测到 “目标” → 得奖励 +10
```

此时的历史：

$$
h_3 = (\text{右}, \text{墙}, -1, \text{左}, \text{空}, 0, \text{前}, \text{目标}, +10)
$$

**关键理解**：

历史是智能体的”完整记忆”，包含所有发生过的事情。但问题是：
- 历史会越来越长（$t$ 增大，$h_t$ 变长）
- 直接用历史做决策很难（信息太多）

### 7.2 状态（State）

**状态 $s_t$**：从历史中提取出的、足够用于预测未来的信息摘要。

**数学定义**：

$$
s_t = \phi(h_t)
$$

其中 $\phi$ 是从历史到状态的压缩函数。

**具体例子**（机器人导航）：

假设机器人经历了复杂的历史（绕了很多弯路），但当前：
- 位置：(5, 3)
- 朝向：北
- 电量：80%
- 目标位置：(8, 3)

如果这 4 个信息足够预测未来（不需要知道”怎么到这里的”），那么可以定义：

$$
s_t = (\text{位置}, \text{朝向}, \text{电量}, \text{目标})
$$

这样，状态只有 4 个值，而不是整个历史。

**关键理解**：

好的状态表示应该：
- ✅ 保留对未来有用的信息（电量影响能否继续移动）
- ✅ 丢掉无关细节（3 步前撞墙的事不影响现在的决策）
- ✅ 尽量简洁（便于学习和规划）

**类比**（医疗诊断）：

医生不需要病人出生以来每秒的完整记录，但需要：
- 当前症状
- 关键病史（高血压、糖尿病）
- 当前用药
- 最近检查结果

这就是”状态”：足够用于决策的信息摘要。

**现代连接**：

- LLM agent 的上下文窗口是一种状态，但可能不是 Markov 的（外部文件、工具状态不在窗口里）
- Robot learning 中，单帧图像通常不够，需要加入速度、历史帧等

### 7.3 观测（Observation）

**观测 $o_t$**：智能体在时刻 $t$ 真正”看到”的信息。

在**完全可观测**环境中：$o_t$ 就是 $s_t$（看到的就是完整状态）  
在**部分可观测**环境中：$o_t$ 只是状态的一部分

**具体例子**（扑克游戏）：

- **状态** $s_t$：所有玩家的手牌 + 已出的牌 + 剩余牌堆
- **观测** $o_t$：只有自己的手牌 + 已出的牌（看不到对手手牌和剩余牌堆）

此时 $o_t \neq s_t$，这是部分可观测问题（POMDP）。

## 8. Markov 假设（Markov Assumption）

**Markov 假设**：当前状态已经包含了预测未来所需的全部信息，过去的历史不再提供额外信息。

**数学定义**：

$$
p(s_{t+1} \mid s_t, a_t) = p(s_{t+1} \mid h_t, a_t)
$$

其中：
- $s_t$：当前状态
- $h_t$：完整历史（从开始到现在的所有动作、观测、奖励）
- $a_t$：当前动作
- $s_{t+1}$：下一个状态

**白话解释**：

“给定现在的状态，未来的转移只取决于现在和接下来的动作，与过去如何到这里无关。”

**具体例子**（机器人导航）：

假设机器人当前在位置 3，要执行”向右”动作。它可能通过两条不同路径到达位置 3：

**路径 1**：起点 → 1 → 2 → 3
**路径 2**：起点 → 5 → 4 → 3

**Markov 状态**的情况：

如果”位置 3”已经完整描述了机器人的状态（位置、速度、电量等），那么：

$$
\begin{aligned}
p(s_{t+1}=4 \mid s_t=3, a_t=\text{right}, \text{路径1}) &= 0.8\\
p(s_{t+1}=4 \mid s_t=3, a_t=\text{right}, \text{路径2}) &= 0.8
\end{aligned}
$$

两条路径得到的转移概率相同，因为状态 3 已经包含了预测未来所需的全部信息。

**非 Markov 状态**的反例：

如果状态只记录”位置 3”，但没记录”电池电量”，而路径 1 消耗了更多电量，那么：

$$
\begin{aligned}
p(s_{t+1}=4 \mid s_t=3, a_t=\text{right}, \text{路径1}) &= 0.5 \quad \text{(电量低，可能失败)}\\
p(s_{t+1}=4 \mid s_t=3, a_t=\text{right}, \text{路径2}) &= 0.9 \quad \text{(电量高，更可能成功)}
\end{aligned}
$$

此时”位置 3”不是 Markov 状态，因为历史信息（哪条路径）影响了未来转移。

**为什么 Markov 假设重要**：

如果状态满足 Markov 性，我们只需要对状态之间的转移建模，而不需要对所有可能的历史建模。这是 Bellman 方程、value iteration、policy iteration 的基础。

**常见误解**：

- ❌ “Markov = 没有过去” → ✅ “Markov = 过去的相关信息已经被当前状态包含”
- ❌ “观测 $o_t$ 一定是 Markov 状态” → ✅ “只有当观测足够充分时，才能令 $s_t = o_t$”
- ❌ “所有 RL 问题都满足 Markov” → ✅ “部分可观测问题（POMDP）中，单个观测通常不是 Markov 的”

**和 Assignment 1 的关系**：

RiverSwim 的 `T[state, action, next_state]` 假设 `state` 已经是 Markov 状态，否则只用当前 state 预测 next_state 就不合理。

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

**策略 $\pi$**：告诉智能体在每个状态下应该选择什么动作的规则。

### 11.1 确定性策略（Deterministic Policy）

**定义**：

$$
\pi(s) = a
$$

**白话解释**：每个状态对应一个固定动作。

**具体例子**（RiverSwim，6 个状态）：

```
状态 0 (最左) → 动作 RIGHT
状态 1        → 动作 RIGHT
状态 2        → 动作 RIGHT
状态 3        → 动作 RIGHT
状态 4        → 动作 RIGHT
状态 5 (最右) → 动作 RIGHT
```

这是一个”一直向右”的确定性策略，写成函数形式：

$$
\pi(s) = \text{RIGHT}, \quad \forall s
$$

### 11.2 随机策略（Stochastic Policy）

**定义**：

$$
\pi(a \mid s) = \Pr(a_t = a \mid s_t = s)
$$

**白话解释**：在状态 $s$ 下，选择每个动作的概率。

**具体例子**（RiverSwim 状态 0）：

假设在最左端状态，策略是：

$$
\pi(\text{LEFT} \mid s=0) = 0.2, \quad \pi(\text{RIGHT} \mid s=0) = 0.8
$$

含义：在状态 0 时，80% 的情况向右，20% 的情况向左。

**必须满足**：

$$
\sum_{a \in \mathcal{A}} \pi(a \mid s) = 1
$$

即所有动作的概率加起来等于 1。

**确定性策略是特殊情况**：

$$
\pi(a^* \mid s) = 1, \quad \pi(a \mid s) = 0 \text{ for } a \neq a^*
$$

### 11.3 两种策略的对比

| | 确定性策略 | 随机策略 |
|---|---|---|
| 形式 | $\pi(s) = a$ | $\pi(a \mid s) \in [0,1]$ |
| 例子 | “永远向右” | “80% 向右，20% 向左” |
| 用途 | 最优策略通常是确定性的 | 用于探索、策略梯度算法 |
| 代码表示 | `action = policy[state]` | `action = np.random.choice(actions, p=policy[state])` |

### 11.4 和 Assignment 1 的关系

- `policy_evaluation(policy, ...)` 中的 `policy` 就是策略，通常是确定性的：`policy[state] = action`
- Policy iteration 会不断改进策略，直到收敛到最优策略

**现代连接**：

- PPO/RLHF 中，语言模型策略是随机策略：给定上下文，对下一个 token 输出概率分布
- Tool-use agents 的工具选择也是策略：在当前任务状态下选择搜索、读文件、写代码、调用计算器

## 12. Evaluation 与 Control

RL 有两类核心任务：

**评估（Evaluation）**：给定一个策略 $\pi$，计算它能获得多少期望回报。

**控制（Control）**：寻找最优策略 $\pi^*$。

**白话理解**：

- Evaluation 问：”这个策略好吗？值多少分？”
- Control 问：”什么策略最好？怎么找到它？”

**和 Assignment 1 的关系**：

- `policy_evaluation(policy, R, T, gamma)` → Evaluation
- `policy_improvement(...)` 和 `value_iteration(...)` → Control 的组成部分

*Lecture 1 只讲 MRP 的价值计算（没有动作选择）。Lecture 2 会讲 MDP 中的 policy evaluation、policy improvement、value iteration。*

## 13. Markov Process / Markov Chain

**Markov Process**：没有动作、没有奖励，只有状态随机转移的过程。

**定义**：

$$
\mathcal{M} = (\mathcal{S}, P)
$$

其中：
- $\mathcal{S}$：状态集合
- $P$：转移概率矩阵

**具体例子**（天气模型，3 个状态）：

假设天气有 3 种状态：晴、阴、雨

转移规则：
- 晴天 → 第二天：70% 晴，20% 阴，10% 雨
- 阴天 → 第二天：30% 晴，40% 阴，30% 雨
- 雨天 → 第二天：20% 晴，30% 阴，50% 雨

转移矩阵（行=当前，列=下一个）：

$$
P = \begin{bmatrix}
0.7 & 0.2 & 0.1\\
0.3 & 0.4 & 0.3\\
0.2 & 0.3 & 0.5
\end{bmatrix}
$$

**数学定义**（转移概率）：

$$
P(s' \mid s) = \Pr(s_{t+1} = s' \mid s_t = s)
$$

在状态 $s$ 时，下一步转移到 $s'$ 的概率。

**关键理解**：

- 这是最简单的随机过程：没有智能体在做决策，世界自己演化
- 是后续 MRP（加奖励）和 MDP（加动作）的基础

### 13.1 两状态例子（贯穿后续章节）

定义两个状态 $A$ 和 $B$：

```
状态 A → 必然转移到 B
状态 B → 必然停留在 B
```

转移概率：

$$
P(B \mid A) = 1, \quad P(B \mid B) = 1
$$

转移矩阵（状态顺序：A, B）：

$$
P = \begin{bmatrix}
0 & 1\\
0 & 1
\end{bmatrix}
$$

矩阵解读：
- 第 1 行 $[0, 1]$：从 A 出发，0% 概率停留在 A，100% 概率到 B
- 第 2 行 $[0, 1]$：从 B 出发，0% 概率回到 A，100% 概率停留在 B

这个例子会在后续计算 MRP 价值时反复使用。

矩阵每一行代表当前状态，每一列代表下一状态。例如第一行 $[0,1]$ 表示从 $A$ 到 $A$ 的概率为 0、到 $B$ 的概率为 1。

## 14. Markov Reward Process（MRP）

**MRP**：在 Markov Process 基础上加入奖励和折扣因子。

**定义**：

$$
\mathcal{M} = (\mathcal{S}, P, R, \gamma)
$$

其中：
- $\mathcal{S}$：状态集合
- $P$：转移概率矩阵（同 Markov Process）
- $R$：奖励函数，$R(s)$ 是状态 $s$ 的即时奖励
- $\gamma$：折扣因子

**具体例子**（两状态 MRP）：

在 §13.1 的两状态 Markov Process 基础上，加入奖励和折扣：

```
状态 A：即时奖励 +1，然后必然转移到 B
状态 B：即时奖励 +2，然后停留在 B
折扣因子：γ = 0.5
```

数学表示：

$$
P = \begin{bmatrix} 0 & 1\\ 0 & 1 \end{bmatrix}, \quad
R = \begin{bmatrix} 1\\ 2 \end{bmatrix}, \quad
\gamma = 0.5
$$

**执行轨迹示例**（从状态 A 开始）：

```
t=0: 在状态 A，得奖励 +1，转移到 B
t=1: 在状态 B，得奖励 +2，停留在 B
t=2: 在状态 B，得奖励 +2，停留在 B
t=3: 在状态 B，得奖励 +2，停留在 B
... 永远停留在 B
```

从 A 开始的折扣回报：

$$
G_0 = 1 + 0.5 \times 2 + 0.5^2 \times 2 + 0.5^3 \times 2 + \cdots
$$

**关键理解**：

- MRP 还没有动作选择——状态如何转移是固定的
- 这是学习”如何计算价值”的最简单场景
- 后续会看到：MDP 中固定策略后，就变成了 MRP

**和 Assignment 1 的关系**：

Policy evaluation 的核心思想：给定策略 $\pi$ 后，MDP 退化成 MRP，然后计算状态价值。

## 15. Horizon、Return、Value Function

### 15.1 Horizon（时域长度）

**Horizon H**：一个 episode 最多可以执行多少个时间步。

**具体例子**（H=5）：

假设 H = 5，那么一个 episode 最多包含 5 个时间步：

```
t=0: 执行第 1 次动作
t=1: 执行第 2 次动作
t=2: 执行第 3 次动作
t=3: 执行第 4 次动作
t=4: 执行第 5 次动作
episode 结束
```

所以时间编号是：$t = 0, 1, 2, 3, 4$，一共 5 步。

**数学定义**：

$$
H = \text{number of time steps in an episode}
$$

本笔记默认时间索引从 0 开始：$t = 0, 1, \ldots, H-1$

**剩余步数**：从时刻 $t$ 开始，episode 还剩 $H - t$ 步（包含当前步）。

例如在 $t=1$ 时：
- 还包含 $t=1, 2, 3, 4$ 共 4 步
- 剩余步数 = $H - t = 5 - 1 = 4$

**注意**：
- Episode 不一定走满 H 步。比如 H=100 表示”最多 100 步”，但机器人第 30 步就到达目标，episode 可以提前结束。
- **有限 horizon**：H 是固定数字（如 H=5, H=100）
- **无限 horizon**：H=∞，episode 理论上永不终止（需要折扣因子 γ<1 保证回报有限）

**Horizon 决定什么**：

智能体需要考虑多远的未来。

- **短 horizon（如 H=3）**：只能看到未来 3 步的后果，策略会偏向即时收益
- **长 horizon（如 H=100）**：可以看到很远的未来，愿意为长期目标忍受短期损失

**和 Assignment 1 的关系**：

Inventory 问题第一题专门考 horizon 如何改变最优策略。起始库存为 3 时：
- 如果 horizon 太短，补货到 10 根本来不及收益，卖货可能更优
- 如果 horizon 足够长，最终满库存的 +100 会改变策略

### 15.2 回报（Return）

**回报 $G_t$**：从时刻 $t$ 开始到 episode 结束，所有奖励的折扣累加和。

**数学定义**：

$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots + \gamma^{H-1-t} r_{H-1}
$$

或写成求和形式：

$$
G_t = \sum_{k=0}^{H-1-t} \gamma^k r_{t+k}
$$

其中：
- $r_t, r_{t+1}, \ldots$：未来各时刻的奖励
- $\gamma \in [0,1]$：折扣因子，越远的奖励权重越小
- $H$：episode 总步数

**具体例子（沿用 §6.1 的自拟资源采集任务，$H=5$，$\gamma=0.9$）**：

假设从 $t=0$ 开始，观察到的奖励序列：

```
t=0: 在 s_0，Collect，r_0 = +10
t=1: 在 s_0，Move，   r_1 = 0
t=2: 在 s_1，Collect，r_2 = +1
t=3: 在 s_1，Move，   r_3 = 0
t=4: 在 s_2，Collect，r_4 = +1
```

从 $t=0$ 开始的回报：

$$
\begin{aligned}
G_0 &= r_0 + \gamma r_1 + \gamma^2 r_2 + \gamma^3 r_3 + \gamma^4 r_4\\
&= 10 + 0.9 \times 0 + 0.9^2 \times 1 + 0.9^3 \times 0 + 0.9^4 \times 1\\
&= 10 + 0 + 0.81 + 0 + 0.656\\
&= 11.466
\end{aligned}
$$

从 $t=2$ 开始的回报（只剩 3 步）：

$$
\begin{aligned}
G_2 &= r_2 + \gamma r_3 + \gamma^2 r_4\\
&= 1 + 0.9 \times 0 + 0.9^2 \times 1\\
&= 1 + 0 + 0.81\\
&= 1.81
\end{aligned}
$$

**关键理解**：
- 回报 $G_t$ 包含”从 $t$ 到 episode 结束”的所有未来奖励
- 越早的时刻，回报包含的未来奖励越多
- 虽然原始奖励总和是 12，但折扣后回报更小（因为远期奖励权重低）

**和 Assignment 1 的关系**：

Inventory 的有限 horizon 与 RiverSwim 的折扣因子都在改变同一个东西：未来奖励在决策中的重要性。

### 15.3 状态价值函数（State Value Function）

**状态价值函数（state-value function）**：从状态 $s$ 出发，未来能获得的回报的期望值。

在有限时域中，剩余时间会影响价值，因此应显式写成 $V_t(s)$（绝对时间）或 $V_h(s)$（还剩 $h$ 步）。本节沿用总步数 $H$ 和绝对时间 $t$ 的约定：

**数学定义**：

$$
V_t(s) = \mathbb{E}[G_t \mid s_t = s] = \mathbb{E}\left[\sum_{k=0}^{H-1-t} \gamma^k r_{t+k} \mid s_t = s\right]
$$

其中：
- $V_t(s)$：时刻 $t$、状态 $s$ 的有限时域价值
- $\mathbb{E}[\cdot]$：对未来随机转移和奖励取期望
- $G_t$：从当前开始的回报（见 §15.2）
- $s_t = s$：当前状态为 $s$

**白话解释**：

价值函数回答这个问题：”如果我现在在状态 $s$，未来平均能拿到多少总奖励？”

对于无限时域 stationary MRP，价值不再显式依赖绝对时间，通常简写为 $V(s)$。

**具体例子（自拟两状态 MRP）**：

假设从状态 $A$ 出发，重复运行 5 次，观察到的回报分别为：

```
第 1 次: G = 3.0
第 2 次: G = 3.5
第 3 次: G = 2.5
第 4 次: G = 3.0
第 5 次: G = 3.0
```

用样本平均估计价值：

$$
V(A) \approx \frac{3.0 + 3.5 + 2.5 + 3.0 + 3.0}{5} = 3.0
$$

**关键理解**：

- 价值 ≠ 即时奖励。即使某个状态本身奖励为 0，但如果它大概率通向高奖励状态，价值仍然高。
- 价值是期望值，不是某一次的实际回报。
- 价值考虑的是”从这里出发的未来”，不包括”怎么到这里的过去”。

**课程 Mars Rover 对比**：

- $s_7$ 的即时奖励是 $+10$，$s_1$ 的即时奖励是 $+1$，中间状态的即时奖励是 0。
- 当 $\gamma=0$ 时，价值恰好等于这些即时奖励。
- 当 $\gamma>0$ 时，$s_5$ 等中间状态虽然即时奖励为 0，但其价值取决于策略和到达两端的转移概率；不能仅凭即时奖励给它排序。

*后续完整讲解：Lecture 3 会正式把样本平均发展成 Monte Carlo policy evaluation。*

**现代连接**：

- LLM agent 中，一个中间推理状态可能还没产生最终答案，但如果它让后续更容易成功，就有高”价值”
- 机器人操作中，把杯子抓稳这个中间状态本身未必有奖励，但它提高最终倒水成功率，因此价值高

## 16. 折扣因子（Discount Factor）

**折扣因子 $\gamma$**：控制未来奖励相对于当前奖励的重要性。

**定义**：

$$
\gamma \in [0, 1]
$$

**具体例子**（RiverSwim，比较不同 γ 值）：

假设有两条路径：

**路径 A（稳定小奖励）**：
```
t=0: r=+0.1
t=1: r=+0.1
t=2: r=+0.1
...永远
```

**路径 B（远期大奖励）**：
```
t=0: r=0
t=1: r=0
...
t=9: r=0
t=10: r=+10
```

计算两条路径从 $t=0$ 开始的无限时域折扣回报；路径 B 只在 $t=10$ 获得一次奖励：

**γ = 0.5（短视）**：
- 路径 A：$G = 0.1 + 0.5 \times 0.1 + 0.5^2 \times 0.1 + \cdots \approx 0.2$
- 路径 B：$G = 0.5^{10} \times 10 \approx 0.01$（远期大奖励被严重折扣）
- **结论**：选路径 A

**γ = 0.9（远视）**：
- 路径 A：$G \approx 1.0$
- 路径 B：$G = 0.9^{10} \times 10 \approx 3.5$
- **结论**：选路径 B

**γ 的三种极端情况**：

| γ 值 | 含义 | 行为 |
|------|------|------|
| γ=0 | 只关心即时奖励 | 完全短视，$G_t = r_t$ |
| γ=0.99 | 未来奖励几乎同等重要 | 愿意为长期目标忍受短期损失 |
| γ=1 | 未来和现在完全平等 | 需要有限 horizon，否则回报可能发散 |

**为什么需要折扣因子**：

1. **数学原因**：无限 horizon 下，不折扣的总和可能发散（∞）
2. **建模原因**：远期预测不确定性更高，折扣体现了”不确定性惩罚”
3. **行为原因**：现实中，现在的 1 元通常比 10 年后的 1 元更有价值

**和 Assignment 1 的关系**：

RiverSwim 中：
- γ 小时：最优策略可能一直停在左边拿稳定小奖励 +0.005
- γ 大时：未来右端大奖励 +1.0 足以 justify 长期向右探索



## 17. MRP 的 Bellman 方程

**Bellman 方程**：将长期价值拆解为”一步奖励 + 折扣后的未来价值”。

**数学定义**：

$$
V(s) = R(s) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s) V(s')
$$

其中：
- $V(s)$：状态 $s$ 的价值
- $R(s)$：状态 $s$ 的即时奖励
- $\gamma$：折扣因子
- $P(s' \mid s)$：从 $s$ 转移到 $s'$ 的概率
- $V(s')$：下一状态 $s'$ 的价值

**白话解释**：

一个状态的价值 = 现在能拿的奖励 + 下一步所有可能去向的价值的加权平均（权重是转移概率）

**为什么这样做有用**：

直接计算 $V(s) = \mathbb{E}[G_t \mid s_t=s]$ 需要考虑所有可能的未来轨迹，计算量爆炸。Bellman 方程利用 Markov 性，只需看一步转移，然后递归引用下一状态的价值。

它把原本需要展开的所有未来路径压缩为局部递推关系，使动态规划和线性代数求解成为可能。

**具体例子**（两状态 MRP）：

定义 MRP：

$$
P = \begin{bmatrix} 0 & 1\\ 0 & 1 \end{bmatrix}, \quad
R = \begin{bmatrix} 1\\ 2 \end{bmatrix}, \quad
\gamma = 0.5
$$

含义：
- 状态 $A$ 下一步必然转移到状态 $B$（第一行 $[0, 1]$）
- 状态 $B$ 下一步必然停留在 $B$（第二行 $[0, 1]$）
- $A$ 的即时奖励是 1，$B$ 的即时奖励是 2

**求解 V(B)**：

状态 $B$ 永远回到自身，所以：

$$
V(B) = R(B) + \gamma \cdot P(B \mid B) \cdot V(B) = 2 + 0.5 \times 1 \times V(B)
$$

移项：

$$
V(B) - 0.5V(B) = 2 \quad \Rightarrow \quad 0.5V(B) = 2 \quad \Rightarrow \quad V(B) = 4
$$

**求解 V(A)**：

状态 $A$ 下一步必然到 $B$：

$$
\begin{aligned}
V(A) &= R(A) + \gamma \cdot P(B \mid A) \cdot V(B)\\
&= 1 + 0.5 \times 1 \times 4\\
&= 1 + 2\\
&= 3
\end{aligned}
$$

**结果**：$V(A) = 3, \quad V(B) = 4$

**直觉理解**：

- $B$ 的价值 4 = 当前奖励 2 + 未来持续停留在 $B$ 获得的折扣奖励
- $A$ 的价值 3 = 当前奖励 1 + 下一步转移到 $B$ 的折扣价值 $0.5 \times 4 = 2$

**和 Assignment 1 的关系**：

`bellman_backup` 在 MDP 中做类似计算，只是 MDP 版本还要指定动作：

$$
\text{backup}(s,a) = R(s,a) + \gamma \sum_{s' \in \mathcal{S}} T(s,a,s') V(s')
$$

*后续完整讲解：Lecture 2 会详细讲解 MDP 中的 Bellman backup。*

代码变量映射：

- $V(s)$ -> `V[state]` 或 `value_function[state]`
- $R(s,a)$ -> `R[state, action]`
- $T(s,a,s')$ -> `T[state, action, next_state]`
- $\gamma$ -> `gamma`


## 18. Bellman 方程的矩阵形式

*标量形式首次完整讲解：§17「MRP 的 Bellman 方程」。这里把所有状态方程合并成矩阵形式。*

**矩阵形式**：

$$
V = R + \gamma P V
$$

其中：
- $V$：所有状态的价值向量，$V = [V(s_1), V(s_2), \ldots, V(s_n)]^\top$
- $R$：所有状态的奖励向量，$R = [R(s_1), R(s_2), \ldots, R(s_n)]^\top$
- $P$：转移概率矩阵，$P_{ij} = P(s_j \mid s_i)$

**具体例子**（两状态 MRP 验证）：

已知 $V(A)=3, V(B)=4$，验证是否满足 Bellman 方程：

$$
\begin{aligned}
R + \gamma P V &= \begin{bmatrix} 1\\ 2 \end{bmatrix} + 0.5 \begin{bmatrix} 0 & 1\\ 0 & 1 \end{bmatrix} \begin{bmatrix} 3\\ 4 \end{bmatrix}\\
&= \begin{bmatrix} 1\\ 2 \end{bmatrix} + 0.5 \begin{bmatrix} 4\\ 4 \end{bmatrix}\\
&= \begin{bmatrix} 1\\ 2 \end{bmatrix} + \begin{bmatrix} 2\\ 2 \end{bmatrix}\\
&= \begin{bmatrix} 3\\ 4 \end{bmatrix} = V \quad \checkmark
\end{aligned}
$$

确实满足！$V=[3,4]^\top$ 是自洽解。

## 19. MRP 价值函数的解析解

*首次完整讲解：§18「Bellman 方程的矩阵形式」。这里进行代数移项求解析解。*

从矩阵 Bellman 方程出发：

$$
V = R + \gamma P V
$$

移项：

$$
V - \gamma P V = R
$$

提取 $V$：

$$
(I - \gamma P) V = R
$$

解析解：

$$
V = (I - \gamma P)^{-1} R
$$

**具体例子**（两状态 MRP 求解）：

计算 $I - \gamma P$：

$$
I - \gamma P = \begin{bmatrix} 1 & 0\\ 0 & 1 \end{bmatrix} - 0.5 \begin{bmatrix} 0 & 1\\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 1 & -0.5\\ 0 & 0.5 \end{bmatrix}
$$

计算逆矩阵：

$$
(I - \gamma P)^{-1} = \begin{bmatrix} 1 & 1\\ 0 & 2 \end{bmatrix}
$$

求解 $V$：

$$
V = (I - \gamma P)^{-1} R = \begin{bmatrix} 1 & 1\\ 0 & 2 \end{bmatrix} \begin{bmatrix} 1\\ 2 \end{bmatrix} = \begin{bmatrix} 3\\ 4 \end{bmatrix}
$$

得到 $V(A)=3, V(B)=4$，与 §17 逐状态求解结果一致。

**计算复杂度**：

矩阵求逆：$O(n^3)$，其中 $n = |\mathcal{S}|$。状态多时很昂贵，实际常用迭代方法。



## 20. 迭代动态规划计算 MRP 价值

*Bellman 方程首次完整讲解：§17；解析解：§19。这里的新内容是从初始猜测反复更新，逐步逼近真实价值。*


**迭代算法**：

初始化：

$$
V_0(s) = 0, \quad \forall s \in \mathcal{S}
$$

迭代更新（第 $k$ 轮）：

$$
V_k(s) = R(s) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s) V_{k-1}(s')
$$



**白话理解**：

从一个简单猜测开始（所有状态价值为 0），每轮用上一轮的价值估计来计算新的价值，不断重复直到收敛。

**具体例子**（两状态 MRP 迭代过程）：

从 $V_0(A) = V_0(B) = 0$ 开始：

$$
\begin{aligned}
V_k(A) &= 1 + 0.5 \times V_{k-1}(B)\\
V_k(B) &= 2 + 0.5 \times V_{k-1}(B)
\end{aligned}
$$


**完整迭代轨迹**：

| 轮数 $k$ | $V_k(A)$ | $V_k(B)$ | 计算过程 |
|---:|---:|---:|:---|
| 0 | 0 | 0 | 初始猜测 |
| 1 | 1 | 2 | $V_1(A) = 1 + 0.5 \times 0 = 1$<br>$V_1(B) = 2 + 0.5 \times 0 = 2$ |
| 2 | 2 | 3 | $V_2(A) = 1 + 0.5 \times 2 = 2$<br>$V_2(B) = 2 + 0.5 \times 2 = 3$ |
| 3 | 2.5 | 3.5 | $V_3(A) = 1 + 0.5 \times 3 = 2.5$<br>$V_3(B) = 2 + 0.5 \times 3 = 3.5$ |
| 4 | 2.75 | 3.75 | $V_4(A) = 1 + 0.5 \times 3.5 = 2.75$<br>$V_4(B) = 2 + 0.5 \times 3.5 = 3.75$ |
| $\infty$ | 3 | 4 | 收敛到真实价值 |

**关键理解**：

- 第 1 轮：只知道即时奖励（$V_1(A)=1, V_1(B)=2$）
- 第 2 轮：开始考虑”一步之后”的价值
- 更多轮：远处奖励逐步向前传播
- 最终收敛到解析解 $V(A)=3, V(B)=4$

**优点 vs. 解析解**：

- ✅ 避免矩阵求逆（$O(n^3)$）
- ✅ 每轮只需 $O(n^2)$ 计算
- ✅ 可以提前停止（不需要精确收敛）

**和 Assignment 1 的关系**：

Value iteration 把这个思想推广到 MDP：对每个状态尝试所有动作，选择 Bellman backup 最大的那个。

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

课件给出的具体答案（Mars Rover，$\pi(s)=\text{TryRight}$，$\gamma=0$）：

$$
V^\pi = [\,+1,\ 0,\ 0,\ 0,\ 0,\ 0,\ +10\,]
$$

每个状态的价值恰好等于它自己的即时奖励：$s_1$ 是 $+1$，$s_7$ 是 $+10$，中间状态全是 0。即使策略一直向右、未来大概率到达 $s_7$ 的 $+10$，也不会给中间状态增加任何价值——因为所有未来项都至少乘有一个 $\gamma$，在 $\gamma=0$ 时被置为 0。

和 Assignment 1 的关系：

RiverSwim 中，如果 $\gamma$ 太小，最右边大奖励传播不到最左边，agent 就可能选择在左边拿稳定小奖励。

## 23. Assignment Readiness

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

*本节于 2026-07-10 联网核实更新。经典基础只收录本讲序列决策、动态规划和价值函数的正典来源；前沿动态展示这些概念在近期 LLM 与机器人系统中的延伸。*

### 29.1 经典基础（本讲概念的原始出处）

- Richard Bellman, *Dynamic Programming*（1957）：动态规划与 Bellman 递推的经典来源。<https://books.google.com/books/about/Dynamic_Programming.html?id=ZzoS0QEACAAJ>
- Sutton & Barto, *Reinforcement Learning: An Introduction*（2nd ed., 2018）：系统覆盖 MDP、return、value function、Bellman equation 以及后续 RL 算法。<https://mitpress.mit.edu/9780262039246/reinforcement-learning/>

### 29.2 前沿动态（截至 2026-07-10 核实）

- RL for Large Reasoning Models 综述：清华团队系统梳理了 DeepSeek-R1 之后“用 RL 训练大型推理模型”方向的奖励设计、算法与基础设施，可作为从本课程基础走向该研究方向的系统桥梁。<https://arxiv.org/abs/2509.08827>（配套论文列表：<https://github.com/TsinghuaC3I/Awesome-RL-for-LRMs>）
- DeepSeek-R1：论文展示了用强化学习激励 LLM 推理行为；2025 年 9 月的 Nature 版本补充了训练方法、实验和限制。<https://www.nature.com/articles/s41586-025-09422-z>
- RLVR 边界之争：一个仍在进行的学术争论——可验证奖励强化学习（RLVR）究竟是让模型学会了新推理能力，还是只提高了采样效率（base model 在大 pass@k 下反超 RL 模型）。正方见 "Limit of RLVR"：<https://limit-of-rlvr.github.io/>；反方提出 CoT-Pass@K 指标证明 RLVR 确实扩展了推理边界：<https://arxiv.org/abs/2506.14245>。这个争论直接涉及本讲的核心问题：奖励信号到底教会了策略什么。
- LLM Post-Training 统一视角综述（2026-04）：把 SFT、偏好优化、RL 等后训练方法统一为 off-policy / on-policy 两条主线来组织。on-policy 与 off-policy 的区分会在本课程后面正式讲到，读这篇时可以对照。<https://arxiv.org/abs/2604.07941>
- 国产开源权重模型的 RL 实践：Qwen3 技术报告（含可切换推理模式的混合推理设计）<https://arxiv.org/abs/2505.09388>；GLM-4.5 技术报告（面向 agent、推理、代码的 RL 后训练）<https://arxiv.org/abs/2508.06471>。两者都展示了 RLHF/RLVR 在工业级大模型训练管线中的具体落地。
- Physical Intelligence $\pi_0$ 与 $\pi_{0.5}$：$\pi_0$ 官方博客介绍多机器人、多任务的 generalist policy。<https://www.pi.website/blog/pi0>。$\pi_{0.5}$ 进一步研究开放世界泛化；其 2025-09 开源发布由官方 `openpi` 仓库记录。<https://www.pi.website/blog/pi05>、<https://github.com/Physical-Intelligence/openpi>。更新的 $\pi_{0.7}$ 使用多模态条件和轻量 world model 生成的视觉 subgoal。<https://www.pi.website/blog/pi07>
- Google DeepMind IMO 2025：官方博客记录 Gemini Deep Think 获得 35/42 的 gold-medal level performance。这里作为现代数学推理系统的背景例子，不把它当作本讲 MRP/MDP 算法细节。<https://deepmind.google/discover/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/>
