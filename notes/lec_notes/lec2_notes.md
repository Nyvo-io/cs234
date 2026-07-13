# CS234 Lecture 2 Notes: Making Sequences of Good Decisions Given a Model of the World

来源：`lecture/lec2/lecture2pre.pdf`，CS234 Winter 2026, Professor Emma Brunskill。

笔记规范：`cs234-rl-tutor v2`。下方 checklist 只表示课件内容已覆盖，不表示学习者已经掌握。

外部资料核验日期：2026-07-10。

本讲目标：在已知世界模型（transition model 和 reward model）的前提下，计算好策略。核心算法是策略迭代（policy iteration, PI）和值迭代（value iteration, VI）。

## 0. 本讲覆盖清单

- [x] 复习 MRP 的迭代价值计算。
- [x] 定义 Markov decision process（MDP）。
- [x] 理解 MDP + policy 如何诱导出 MRP。
- [x] 掌握 MDP policy evaluation 的 Bellman 更新。
- [x] 理解 deterministic policy 空间大小和 optimal policy 不一定唯一。
- [x] 理解 MDP control：最优价值函数唯一，最优策略可不唯一。
- [x] 掌握状态动作价值函数（state-action value function, Q-value）。
- [x] 掌握 policy improvement 的数学形式和单调改进直觉。
- [x] 掌握 policy iteration 的流程和终止条件。
- [x] 掌握 Bellman optimality operator。
- [x] 掌握 value iteration 的更新式和收敛条件。
- [x] 理解 Bellman operator contraction 的意义。
- [x] 区分 infinite-horizon stationary policy 与 finite-horizon time-dependent policy。
- [x] 能把 Lecture 2 公式映射到 Assignment 1 的 `vi_and_pi.py`。

## 1. 开场 Quick Check：大的 $\gamma$ 是不是更看重短期？

课件开场问：

> 在 MDP 中，较大的折扣因子 $\gamma$ 是否意味着短期奖励比长期奖励更有影响？

答案：False。

较大的 $\gamma$ 让未来奖励衰减更慢，因此长期奖励更有影响。较小的 $\gamma$ 才更短视。

极端情况：

- $\gamma = 0$：只关心即时奖励。
- $\gamma \approx 1$：远期奖励仍然重要。

和 Assignment 1 的关系：

RiverSwim 中，最左边有小即时奖励，最右边有大远期奖励。$\gamma$ 越大，agent 越愿意逆流向右；$\gamma$ 越小，agent 越可能留在左边拿小奖励。

## 2. 本讲主线

Lecture 1 已经讲过：

- agent 的组件：模型（model）、价值函数（value function）、策略（policy）。
- Markov process 和 MRP。
- MRP 的 Bellman 方程。

Lecture 2 继续问：

如果我们已经知道世界模型，也就是知道 $P$ 和 $R$，如何计算一个好策略？

这属于 model-based planning。这里的 planning 不是自然语言计划，而是：已知模型后，用动态规划在模型里推演未来，计算策略。

现代连接：

- Robot learning 中，若已知或学到了动力学模型，就可以做 model-based planning。
- Tool-use agents 中，如果能预测调用某个工具会得到什么信息、花多少成本、带来什么风险，也可以把工具调用选择看作 planning 问题。
- LLM agent 的“先搜索再读文件再回答”可以类比为策略，但实际系统常不满足干净的有限 MDP 假设。

## 3. 复习：MRP 的迭代价值计算

Lecture 2 从 MRP 价值迭代复习开始：

$$
V_k(s)
=
R(s)
+
\gamma
\sum_{s' \in \mathcal{S}}
P(s' \mid s)V_{k-1}(s')
$$

*首次完整讲解：Lecture 1 §20「迭代动态规划计算 MRP 价值」。这里是复习，并为 MDP policy evaluation 做过渡。*

这条公式在说什么：

第 $k$ 轮对状态 $s$ 的价值估计，等于当前状态即时奖励，加上下一状态价值的折扣期望。这里没有动作，因为 MRP 中没有 agent 可以选择动作。

它解决什么问题：

它用动态规划（dynamic programming）避免枚举完整未来轨迹。远处奖励会通过反复迭代逐步向前传播。

符号解释：

- $V_k(s)$：第 $k$ 轮时状态 $s$ 的价值估计。
- $R(s)$：状态 $s$ 的期望即时奖励。
- $P(s' \mid s)$：从状态 $s$ 到 $s'$ 的转移概率。
- $\gamma$：折扣因子。

直觉：

如果 $s$ 的下一步大概率通向高价值状态，那么 $s$ 本身就会变得有价值。

和本讲的关系：

MDP 只是在 MRP 上加入动作。加入动作以后，我们既要能评估固定策略，也要能找到更好的策略。

### 3.1 最小数值例子

假设 MRP 中状态 $A$ 的即时奖励为 1，并且下一步必然到达状态 $B$：

$$
R(A)=1,
\qquad
P(B\mid A)=1
$$

再设：

$$
\gamma=0.9,
\qquad
V_{k-1}(B)=4
$$

代入迭代公式：

$$
\begin{aligned}
V_k(A)
&=
R(A)+\gamma P(B\mid A)V_{k-1}(B)\\
&=
1+0.9\times 1\times 4\\
&=
4.6
\end{aligned}
$$

这个结果表示：状态 $A$ 当前得到 1 分；它必然通向价值估计为 4 的状态 $B$，折扣后的未来部分贡献 3.6，所以本轮把 $A$ 的价值更新为 4.6。



## 4. Markov Decision Process（MDP）

**MDP**：在 MRP 基础上加入动作选择。

**定义**：

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)
$$

其中：
- $\mathcal{S}$：状态集合
- $\mathcal{A}$：动作集合
- $P$：转移概率，$P(s' \mid s, a)$ = 在状态 $s$ 执行动作 $a$ 后转移到 $s'$ 的概率
- $R$：奖励函数，$R(s, a)$ = 在状态 $s$ 执行动作 $a$ 的即时奖励
- $\gamma$：折扣因子

**MRP vs. MDP 的区别**：

|        | MRP            | MDP               |
| ------ | -------------- | ----------------- |
| 转移     | $P(s' \mid s)$ | $P(s' \mid s, a)$ |
| 奖励     | $R(s)$         | $R(s, a)$         |
| 有动作选择？ | ❌ 状态自动转移       | ✅ 智能体选择动作         |

**具体例子**（RiverSwim，6 状态 2 动作）：

```
状态：0 (最左) ... 5 (最右)
动作：LEFT, RIGHT

转移概率示例（状态 0）：
- 选 LEFT:  100% 停留在 0，得奖励 +0.005
- 选 RIGHT: 60% 停留在状态 0，40% 到状态 1，得奖励 0

转移概率示例（状态 5）：
- 选 LEFT:  100% 到状态 4，得奖励 0
- 选 RIGHT: 60% 停留在 5，40% 到状态 4，得奖励 +1.0
```

用数学表示：

$$
\begin{aligned}
P(0 \mid 0, \text{LEFT}) &= 1.0, \quad R(0, \text{LEFT}) = 0.005\\
P(0 \mid 0, \text{RIGHT}) &= 0.6, \quad P(1 \mid 0, \text{RIGHT}) = 0.4, \quad R(0, \text{RIGHT}) = 0
\end{aligned}
$$

**关键理解**：

MDP 的核心问题是"如何选择动作"。不同动作会导致不同的转移和奖励。

MRP 只能描述世界如何随机演化，不能描述 agent 如何选择。MDP 加入动作以后，可以讨论“选择什么动作会改变未来”，也就能讨论最优策略。

符号解释：

- $\mathcal{S}$：有限 Markov 状态集合。
- $\mathcal{A}$：有限动作集合。
- $P(s' \mid s,a)$：在状态 $s$ 采取动作 $a$ 后转移到 $s'$ 的概率。
- $R(s,a)$：在状态 $s$ 采取动作 $a$ 的期望即时奖励。
- $\gamma \in [0,1]$：折扣因子。

直觉：

MDP 是“可控的 Markov 世界”。世界随机，但 agent 的动作会影响转移概率和奖励。

### 4.1 奖励函数的不同写法

课件说明，奖励有时写成状态函数 $R(s)$，有时写成状态动作函数 $R(s,a)$，也有时写成三元组函数 $r(s,a,s')$。

本课程多数时候采用：

$$
R(s,a)
=
\mathbb{E}[r_t \mid s_t=s, a_t=a]
$$

*首次完整讲解：Lecture 1 §10「MDP 模型：Transition Model 与 Reward Model」。本节只补充不同奖励函数写法及 Assignment 1 的记号对应。*

这条公式在说什么：

给定当前状态和动作，奖励模型返回即时奖励的期望。即使实际奖励有随机性，$R(s,a)$ 也代表它的平均值。

和 Assignment 1 的关系：

RiverSwim starter code 中：

- $R(s,a)$ -> `R[state, action]`
- $P(s' \mid s,a)$ -> `T[state, action, next_state]`

Inventory written problem 中，题目用 $r(s,a,s')$，因为到达 $s'=10$ 这一转移触发 +100。做题时要尊重题目定义，但算法代码里用的是 $R(s,a)$。

## 5. 策略（Policy）

策略指定每个状态下如何选择动作。为了一般性，课件把策略写成条件分布：

$$
\pi(a \mid s)
=
\Pr(a_t=a \mid s_t=s)
$$

*首次完整讲解：Lecture 1 §11「策略（Policy）」。本节沿用随机策略定义，把动作概率代入 MDP policy evaluation。*

这条公式在说什么：

在状态 $s$ 下，策略给出每个动作 $a$ 被选择的概率。

它解决什么问题：

随机策略可以表达探索，也覆盖确定性策略。确定性策略只是特殊情况：某个动作概率为 1，其余动作概率为 0。

符号解释：

- $\pi$：策略。
- $\pi(a \mid s)$：状态 $s$ 下选动作 $a$ 的概率。
- $a_t$：时间 $t$ 的动作。
- $s_t$：时间 $t$ 的状态。

直觉：

策略就是 agent 的行为规则。它可以是查表，也可以是神经网络。

现代连接：

- Robot policy 可以把传感器状态映射到控制动作。
- LLM policy 可以把上下文映射到下一个 token 或工具动作的概率分布。
- PPO/RLHF 中优化的对象也是策略分布，但后续 lecture 才会正式进入 policy gradient。




## 6. MDP + Policy = MRP

*首次完整讲解：Lecture 1 §11「策略」和 §14「MRP」。这里展示给定策略后，MDP 如何退化成 MRP。*

**核心思想**：给定策略 $\pi$ 后，动作选择固定，MDP 变成 MRP。

**策略诱导的 MRP**：

$$
(\mathcal{S}, R^\pi, P^\pi, \gamma)
$$

### 6.1 策略诱导的奖励函数

$$
R^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) R(s, a)
$$

**白话解释**：状态 $s$ 的平均奖励 = 各动作奖励的加权平均（权重是策略选该动作的概率）。

**具体例子**（RiverSwim 状态 0）：

假设策略是：

$$
\pi(\text{LEFT} \mid 0) = 0.3, \quad \pi(\text{RIGHT} \mid 0) = 0.7
$$

奖励是：

$$
R(0, \text{LEFT}) = 0.005, \quad R(0, \text{RIGHT}) = 0
$$

那么状态 0 的诱导奖励：

$$
\begin{aligned}
R^\pi(0) &= 0.3 \times 0.005 + 0.7 \times 0\\
&= 0.0015
\end{aligned}
$$

### 6.2 策略诱导的转移概率

$$
P^\pi(s' \mid s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) P(s' \mid s, a)
$$

**白话解释**：从 $s$ 到 $s'$ 的概率 = 对所有可能动作，”选该动作的概率 × 该动作导致 $s'$ 的概率”的总和。

**具体例子**（RiverSwim 状态 0 → 状态 1）：

转移概率：

$$
P(1 \mid 0, \text{LEFT}) = 0, \quad P(1 \mid 0, \text{RIGHT}) = 0.6
$$

策略诱导的转移：

$$
\begin{aligned}
P^\pi(1 \mid 0) &= \pi(\text{LEFT} \mid 0) \times 0 + \pi(\text{RIGHT} \mid 0) \times 0.6\\
&= 0.3 \times 0 + 0.7 \times 0.6\\
&= 0.42
\end{aligned}
$$

**关键理解**：

- 策略固定后，动作不确定性”消失”了，变成了状态转移的一部分
- 现在可以用 Lecture 1 的 MRP 方法计算价值
- 这是 **policy evaluation** 的基础

假设状态 $s$ 有两个动作：向左和向右。策略选择它们的概率为：

$$
\pi(\text{left}\mid s)=0.25,
\qquad
\pi(\text{right}\mid s)=0.75
$$

两个动作的即时奖励分别为：

$$
R(s,\text{left})=0,
\qquad
R(s,\text{right})=4
$$

策略诱导的平均奖励是：

$$
\begin{aligned}
R^\pi(s)
&=
0.25\times 0+0.75\times 4\\
&=
3
\end{aligned}
$$

再假设到达好状态 $g$ 的概率为：

$$
P(g\mid s,\text{left})=0.2,
\qquad
P(g\mid s,\text{right})=0.8
$$

策略诱导的转移概率为：

$$
\begin{aligned}
P^\pi(g\mid s)
&=
0.25\times 0.2+0.75\times 0.8\\
&=
0.65
\end{aligned}
$$

因此，一旦策略固定，我们可以直接说“状态 $s$ 的平均即时奖励是 3，到达 $g$ 的概率是 0.65”，不必在诱导出的 MRP 中继续显式保留动作。

和 Assignment 1 的关系：

`policy_evaluation(policy, R, T, gamma)` 输入的是确定性策略 `policy[state] = action`。因此公式会简化，不需要对所有动作按概率求和。

## 7. MDP Policy Evaluation

**Policy Evaluation**：给定策略 $\pi$，计算它的价值函数 $V^\pi(s)$。

**Bellman 方程（随机策略）**：

$$
V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \left[ R(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s,a) V^\pi(s') \right]
$$

**白话解释**：

状态 $s$ 的价值 = 对所有可能动作：策略选该动作的概率 × (该动作的即时奖励 + 折扣后的未来价值)

**具体例子**（两动作选择）：

假设状态 $s$ 有两个动作，策略是：

$$
\pi(\text{left} \mid s) = 0.25, \quad \pi(\text{right} \mid s) = 0.75
$$

奖励和转移：

```
动作 left:  奖励 0，80% 到状态 b (V=1)，20% 到状态 g (V=5)
动作 right: 奖励 4，20% 到状态 b (V=1)，80% 到状态 g (V=5)
```

设 $\gamma = 0.9$。

**第 1 步**：计算每个动作的 backup：

$$
\begin{aligned}
\text{backup}(s, \text{left}) &= 0 + 0.9 \times (0.8 \times 1 + 0.2 \times 5) = 1.62\\
\text{backup}(s, \text{right}) &= 4 + 0.9 \times (0.2 \times 1 + 0.8 \times 5) = 7.78
\end{aligned}
$$

**第 2 步**：按策略概率加权：

$$
\begin{aligned}
V^\pi(s) &= 0.25 \times 1.62 + 0.75 \times 7.78\\
&= 0.405 + 5.835\\
&= 6.24
\end{aligned}
$$

### 7.1 确定性策略的简化

*首次完整讲解：§7。确定性策略是随机策略的特殊情况（某动作概率为 1）。*

**确定性策略 Bellman 方程**：

$$
V^\pi(s) = R(s, \pi(s)) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, \pi(s)) V^\pi(s')
$$

动作固定为 $\pi(s)$，不需要对动作求和。

**和 Assignment 1 的关系**：

`policy_evaluation(policy, R, T, gamma)` 中，策略是确定性的：`policy[state] = action`

代码映射：

```python
action = policy[state]
new_value[state] = R[state, action] + gamma * np.sum(T[state, action, :] * value_function)
```



## 8. Policy Evaluation 的迭代算法

课件给出的迭代形式：

$$
V^\pi_k(s)
=
\sum_{a \in \mathcal{A}}
\pi(a \mid s)
\left[
R(s,a)
+
\gamma
\sum_{s' \in \mathcal{S}}
P(s' \mid s,a)V^\pi_{k-1}(s')
\right]
$$

*公式来源：本讲 §7 的 Bellman policy evaluation；迭代思想首次讲解于 Lecture 1 §20。下一节 Exercise L2E1 给出一轮更新的具体数值计算。*

这条公式在说什么：

第 $k$ 轮使用上一轮价值估计 $V^\pi_{k-1}$ 来更新每个状态的价值。不断重复，直到价值函数变化很小。

它解决什么问题：

不用矩阵求逆，也不用枚举所有轨迹；只需要反复做 Bellman backup。

直觉：

每次迭代都把未来多一步的信息传播回来。

代码停止条件：

Assignment 1 的 `tol` 可以用 infinity norm(无穷范数)：

$$
\lVert V_{\text{new}} - V_{\text{old}} \rVert_\infty
=
\max_s |V_{\text{new}}(s)-V_{\text{old}}(s)|
\le \text{tol}
$$

**tolerance**：误差

找出所有状态里，**本轮变化最大的那个状态**。  如果连它都变化很小，那说明所有状态都已经稳定了

和 starter code 的关系：

`policy_evaluation(policy, R, T, gamma, tol)` 应该循环更新 `value_function`，直到最大状态价值变化小于 `tol`。



## 9. Exercise L2E1：一轮 Policy Evaluation

课件给定：

- $\pi(s)=a_1$，所有状态都选 $a_1$。
- $\gamma = 0.5$。
- $V_k = [1,0,0,0,0,0,10]$。
- $p(s_6 \mid s_6,a_1)=0.5$，$p(s_7 \mid s_6,a_1)=0.5$。
- $r(s_6)=0$。

计算：

$$
V_{k+1}(s_6)
=
r(s_6)
+
\gamma
\sum_{s'}
p(s' \mid s_6,a_1)V_k(s')
$$

代入：

$$
V_{k+1}(s_6)
=
0
+
0.5
\left(
0.5 \cdot 10
+
0.5 \cdot 0
\right)
=
2.5
$$

这条计算在说什么：

从 $s_6$ 出发，下一步有一半概率到高价值的 $s_7$，一半概率留在低价值状态。折扣后，当前状态的新估计是 2.5。

直觉：

即时奖励是 0，但通向 $s_7$ 的概率让 $s_6$ 有未来价值。



## 10. Deterministic Policy 空间大小与最优策略唯一性

Mars rover 例子有 7 个状态、2 个动作。确定性策略数是：

$$
|\mathcal{A}|^{|\mathcal{S}|}
=
2^7
=
128
$$

这条公式在说什么：

每个状态都要选择一个动作。7 个状态各有 2 种选择，总共有 $2^7$ 个确定性策略。

它解决什么问题：

这说明直接枚举策略很快会爆炸。如果有 100 个状态、4 个动作，就是 $4^{100}$ 个策略。

最优策略是否唯一？

不一定。可能有多个策略达到同一个最优价值。例如在某些状态两个动作的未来回报完全相同，那么两个动作都最优。

要区分：

- 最优价值函数（optimal value function）在标准 discounted finite MDP 中唯一。
- 最优策略（optimal policy）不一定唯一。




## 11. MDP Control

控制（control）问：

如何找到最优策略？

最优价值函数定义为：

$$
V^*(s)
=
\max_\pi V^\pi(s),
\quad \forall s\in\mathcal{S}
$$

最优策略是能够在所有状态实现该最优价值的策略：

$$
\pi^*
\in
\left\{
\pi
\mid
V^\pi(s)=V^*(s),
\ \forall s\in\mathcal{S}
\right\}
$$

这组公式在说什么：

第一条先定义每个状态可以达到的最大策略价值；第二条说明最优策略是一个完整的状态到动作规则，它必须同时实现这些最优价值。

它解决什么问题：

Policy evaluation 只能回答“这个策略怎么样”。Control 要回答“该用哪个策略”。

符号解释：

- $\pi^*$：最优策略。
- $V^*(s)$：状态 $s$ 的最优价值。
- $V^\pi(s)$：策略 $\pi$ 下状态 $s$ 的价值。

直觉：

评估是打分；控制是找高分策略。



### 11.1 最小数值例子

在一个只有状态 $s$ 的简单 MDP 中，假设仅有两个候选策略：

$$
V^{\pi_1}(s)=3,
\qquad
V^{\pi_2}(s)=5
$$

那么：

$$
V^*(s)=\max\{3,5\}=5
$$

因此 $\pi_2$ 是这个例子中的最优策略。注意，$V^*(s)$ 是数值 5，而 $\pi_2$ 是策略；不能把“动作或策略”和“价值数值”写成同一个数学对象。

重要结论：

在有限状态动作、infinite-horizon discounted MDP 中，存在 deterministic、stationary 的最优策略。

- Deterministic：每个状态选一个固定动作即可，不一定需要随机。
- Stationary：策略不依赖时间步，只依赖当前状态。
- Unique： 不一定，最优策略可能有多个。



## 12. Policy Search 与 Policy Iteration

暴力搜索所有确定性策略需要检查：

$$
|\mathcal{A}|^{|\mathcal{S}|}
$$

*公式来源：本讲 §10「Deterministic Policy 空间大小」。这里用它说明暴力 policy search 的组合爆炸。*

这通常不可行。

策略迭代（policy iteration）更高效。它交替做两件事：

1. Policy evaluation：评估当前策略。
2. Policy improvement：基于当前价值函数改进策略。

算法：

```python
Set i = 0
Initialize pi_0(s) randomly for all states s
While i == 0 or pi_i changed from pi_{i-1}:
    V^{pi_i} <- policy evaluation of pi_i
    pi_{i+1} <- policy improvement using V^{pi_i}
    i <- i + 1
```

和 starter code 的关系：

`policy_iteration(R, T, gamma, tol)` 应调用：

- `policy_evaluation(policy, R, T, gamma, tol)`
- `policy_improvement(policy, R, T, V_policy, gamma)`

终止条件可以是：

```python
if np.array_equal(new_policy, policy):
    break
```



## 13. State-Action Value Function: $Q^\pi(s,a)$

**Q-value（状态-动作价值函数）**：在状态 $s$ 先执行动作 $a$，之后按策略 $\pi$ 行动的期望回报。

**定义**：

$$
Q^\pi(s,a) = R(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s,a) V^\pi(s')
$$

**白话解释**：

$Q^\pi(s,a)$ = 动作 $a$ 的即时奖励 + 折扣后的未来价值（按 $\pi$ 继续。也就是当前状态s，执行动作action后的价值

**V 和 Q 的关系**：

- $V^\pi(s)$：在状态 $s$，按策略 $\pi$ 行动的价值
- $Q^\pi(s,a)$：在状态 $s$，**第一步强制选 $a$**，之后按 $\pi$ 行动的价值

**具体例子**（两动作比较）：

假设状态 $s$ 有两个动作 left 和 right：

```
动作 right: 奖励 +4，80% 到 g (V=5)，20% 到 b (V=1)
动作 left:  奖励 0，  20% 到 g (V=5)，80% 到 b (V=1)
```

设 $\gamma = 0.9$。

**计算 Q(s, right)**：

$$
\begin{aligned}
Q^\pi(s, \text{right}) &= 4 + 0.9 \times (0.8 \times 5 + 0.2 \times 1)\\
&= 4 + 0.9 \times (4 + 0.2)\\
&= 4 + 0.9 \times 4.2\\
&= 4 + 3.78\\
&= 7.78
\end{aligned}
$$

**计算 Q(s, left)**：

$$
\begin{aligned}
Q^\pi(s, \text{left}) &= 0 + 0.9 \times (0.2 \times 5 + 0.8 \times 1)\\
&= 0.9 \times (1 + 0.8)\\
&= 0.9 \times 1.8\\
&= 1.62
\end{aligned}
$$

**关键理解**：

- $Q^\pi(s, \text{right}) = 7.78 > Q^\pi(s, \text{left}) = 1.62$
- 这说明在状态 $s$，选 right 比 left 更好（即使之后都按 $\pi$ 走）
- 这是 **policy improvement** 的基础

**V 和 Q 的联系**：

$$
V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) Q^\pi(s,a)
$$

如果策略在 $s$ 选动作 $a$ 的概率是 $\pi(a \mid s)$，那么 $V$ 就是所有 $Q$ 的加权平均。

**和 Assignment 1 的关系**：

`policy_improvement` 需要对每个状态计算所有动作的 Q 值，然后选最大的：

```python
for s in range(num_states):
    Q_values = [R[s,a] + gamma * np.sum(T[s,a,:] * V) for a in range(num_actions)]
    new_policy[s] = np.argmax(Q_values)
```

和 Assignment 1 的关系：

`bellman_backup(state, action, R, T, gamma, V)` 本质上就是计算：

$$
R(s,a)
+
\gamma
\sum_{s'}T(s,a,s')V(s')
$$

如果传入的 `V` 是 $V^\pi$，这个 backup 就是 $Q^\pi(s,a)$。

代码映射：

```python
backup_val = R[state, action] + gamma * np.sum(T[state, action] * V)
```





## 14. Policy Improvement

*Q 函数首次完整讲解：§13。这里用 Q 函数来改进策略。*

**Policy Improvement**：基于当前策略的价值函数，构造更好的策略。

**算法**：

1. 计算所有状态-动作对的 Q 值：

$$
Q^{\pi_i}(s,a) = R(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s,a) V^{\pi_i}(s')
$$

2. 对每个状态，选择 Q 值最大的动作：

$$
\pi_{i+1}(s) = \arg\max_a Q^{\pi_i}(s,a), \quad \forall s \in \mathcal{S}
$$

**白话解释**：

“在每个状态，看看如果第一步换成别的动作（之后还按旧策略走），会不会更好？如果会，就换。”

**具体例子**（从 §13 继续）：

状态 $s$ 有两个动作，当前策略价值函数下：

$$
Q^{\pi_i}(s, \text{left}) = 1.62, \quad Q^{\pi_i}(s, \text{right}) = 7.78
$$

新策略选择：

$$
\pi_{i+1}(s) = \arg\max \{1.62, 7.78\} = \text{right}
$$

**Policy Improvement 定理**：

$$
V^{\pi_{i+1}}(s) \ge V^{\pi_i}(s), \quad \forall s
$$

新策略(第i+1次的策略)在每个状态的价值都不低于旧策略（通常会更好，除非旧策略已经最优）。

**为什么一定不会更差？**

因为 $\pi_{i+1}$ 至少可以选择 $\pi_i$ 本来要选的动作：

$$
\max_a Q^{\pi_i}(s,a) \ge Q^{\pi_i}(s, \pi_i(s)) = V^{\pi_i}(s)
$$

**和 Assignment 1 的关系**：

`policy_improvement(policy, R, T, V_policy, gamma)` 实现：

```python
for s in range(num_states):
    Q_values = []
    for a in range(num_actions):
        Q = R[s,a] + gamma * np.sum(T[s,a,:] * V_policy)
        Q_values.append(Q)
    new_policy[s] = np.argmax(Q_values)
```



## 15. Policy Improvement 的单调性

课件定义：

$$
V^{\pi_1} \ge V^{\pi_2}
\quad \Longleftrightarrow \quad
V^{\pi_1}(s) \ge V^{\pi_2}(s),
\ \forall s \in \mathcal{S}
$$

这条公式在说什么：

一个策略比另一个策略好，是指它在每个状态的价值都不低于另一个策略。


Policy improvement 命题：

$$
V^{\pi_{i+1}}
\ge
V^{\pi_i}
$$

如果 $\pi_i$ 还不是最优策略，则至少某些状态会严格变好。

这条命题在说什么：

用 policy improvement 得到的新策略不会比旧策略差。这就是 policy iteration (策略迭代) 的核心保证。

它解决什么问题：

这回答了开场问题：是否能构造一个算法，随着计算/迭代增加，策略单调改进？Policy iteration 在有限状态动作空间中有这个性质。

证明直觉：

对任意状态 $s$，新策略选择使 $Q^{\pi_i}(s,a)$ 最大的动作，因此：

$$
\max_a Q^{\pi_i}(s,a)
\ge
Q^{\pi_i}(s,\pi_i(s))
=
V^{\pi_i}(s)
$$

这条不等式在说什么：

至少可以选择旧策略本来会选的动作，所以取最大值不会更差。如果新策略选了更好的动作，就会更好。

然后反复把这种“一步不差”的关系沿着未来递推下去，就得到新策略整体不差。

容易混淆点：

Policy improvement 不是说每一次每个状态的动作都会变。它说如果按 greedy 改进得到新策略，那么新策略的价值不低于旧策略。



## 16. Policy Iteration 何时停止？

如果一次 policy improvement 后：

$$
\pi_{i+1}(s) = \pi_i(s),
\quad \forall s
$$

那么策略没有变化。此时继续评估和改进也不会再变化，因为同一个策略会得到同一个价值函数，greedy improvement 也会返回同一个策略。

有限 deterministic policy 数量有限，policy iteration 每次改进又不会变差，因此它会在有限步内停止。

实践中要注意 tie-breaking：

如果两个动作价值相同，`argmax` 会按实现规则选其中一个。为了避免同值动作导致来回变化，代码中保持一致的 tie-breaking 很重要。NumPy 的 `np.argmax` 会返回第一个最大值索引。



## 17. Bellman Optimality Operator

课件定义 Bellman backup operator：

$$
(BV)(s)
=
\max_a
\left[
R(s,a)
+
\gamma
\sum_{s' \in \mathcal{S}}
P(s' \mid s,a)V(s')
\right]
$$

这条公式在说什么：

给定任意价值函数 $V$，Bellman optimality operator $B$ 会为每个状态计算：如果下一步可以选最好的动作，那么一步奖励加上折扣未来价值最大是多少。

它解决什么问题：

它把“找最优动作”直接放进 Bellman 更新里。Policy evaluation 固定策略；optimality backup 对动作取最大值。

符号解释：

- $B$：Bellman optimality operator。
- $V$：任意价值向量，不一定是某个策略真实能达到的 $V^\pi$。
- $(BV)(s)$：把 operator 作用在 $V$ 上后，在状态 $s$ 得到的新价值。
- $\max_a$：对所有动作取最大值。

直觉：

如果 policy evaluation 是“按给定策略看未来”，那么 Bellman optimality backup 是“每一步都假设自己会选当前看起来最好的动作”。

### 17.1 最小数值例子

仍使用前面的两动作数据。对当前价值估计 $V(g)=5$、$V(b)=1$，两个动作的 backup 分别为 1.62 和 7.78，因此：

$$
\begin{aligned}
(BV)(s)
&=
\max\{1.62,7.78\}\\
&=
7.78
\end{aligned}
$$

Bellman optimality operator 保留的是最大价值 7.78。若还要知道对应动作，则需要另取 $\arg\max$，结果是 `right`。

和 Assignment 1 Q3 的关系：

Q3 中的 Bellman residual：

$$
BV - V
$$

衡量的是一个任意价值函数 $V$ 离 Bellman optimality fixed point 有多远。




## 18. Value Iteration（VI）

*Bellman Optimality Operator 首次完整讲解：§17。Value iteration 就是反复应用该 operator。*

**Value Iteration**：直接迭代最优价值函数，不需要显式维护策略。

**更新公式**：

$$
V_{k+1}(s) = \max_a \left[ R(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s,a) V_k(s') \right]
$$

或写成 operator 形式：

$$
V_{k+1} = B V_k
$$

**白话解释**：

每轮更新时，对每个状态 $s$：
1. 计算所有动作的 backup
2. 取最大值作为新的价值估计

**具体例子**（两动作，两轮迭代）：

初始化：$V_0(s) = 0, V_0(g) = 0, V_0(b) = 0$

**第 1 轮**（只看即时奖励）：

```
动作 left:  backup = 0 + 0.9 × (0.2×0 + 0.8×0) = 0
动作 right: backup = 4 + 0.9 × (0.8×0 + 0.2×0) = 4
```

$$
V_1(s) = \max\{0, 4\} = 4
$$

**第 2 轮**（考虑一步后的价值）：

假设 $V_1(g) = 5, V_1(b) = 1$（从其他状态传播来的）

```
动作 left:  backup = 0 + 0.9 × (0.2×5 + 0.8×1) = 1.62
动作 right: backup = 4 + 0.9 × (0.8×5 + 0.2×1) = 7.78
```

$$
V_2(s) = \max\{1.62, 7.78\} = 7.78
$$

**终止条件**：

$$
\lVert V_{k+1} - V_k \rVert_\infty \le \epsilon
$$

当价值函数变化小于阈值 $\epsilon$ 时停止。

**提取策略**：

VI 收敛后，用 greedy policy 提取最优策略：

$$
\pi^*(s) = \arg\max_a \left[ R(s,a) + \gamma \sum_{s'} P(s' \mid s,a) V^*(s') \right]
$$

**和 Assignment 1 的关系**：

`value_iteration(R, T, gamma, tol)` 实现：

```python
V = np.zeros(num_states)
while True:
    new_V = np.zeros(num_states)
    for s in range(num_states):
        action_values = []
        for a in range(num_actions):
            q = R[s,a] + gamma * np.sum(T[s,a,:] * V)
            action_values.append(q)
        new_V[s] = max(action_values)

    converged = np.max(np.abs(new_V - V)) <= tol
    V = new_V
    if converged:
        break

# 提取策略
policy = np.zeros(num_states, dtype=int)
for s in range(num_states):
    Q = [R[s,a] + gamma * np.sum(T[s,a,:] * V) for a in range(num_actions)]
    policy[s] = np.argmax(Q)
```

**PI vs. VI 对比**：

| | Policy Iteration | Value Iteration |
|---|---|---|
| 维护对象 | 策略 + 价值 | 只有价值 |
| 每轮操作 | 完整 policy evaluation + improvement | 一次 Bellman optimality backup |
| 收敛速度 | 通常更快（步数少） | 每步更简单 |
| 适用场景 | 动作少、需要中间策略 | 动作多、只要最终策略 |




## 19. 从 Value Function 提取 Greedy Policy

给定价值函数 $V$，greedy policy 是：

$$
\pi(s)
=
\arg\max_a
\left[
R(s,a)
+
\gamma
\sum_{s' \in \mathcal{S}}
P(s' \mid s,a)V(s')
\right]
$$

*首次完整讲解：本讲 §14「Policy Improvement」。这里的区别是使用 value iteration 最终得到的 $V$，而不是某个待改进策略的 $V^{\pi_i}$。*

这条公式在说什么：

在每个状态下，选择让 Bellman backup 最大的动作。

它解决什么问题：

Value iteration 主要计算价值函数。要实际行动，还需要把价值函数转成策略。

直觉：

价值函数告诉你未来哪里好；greedy policy 选择能通向最好未来的当前动作。

代码映射：

```python
action_values = [
    bellman_backup(state, action, R, T, gamma, value_function)
    for action in range(num_actions)
]
policy[state] = np.argmax(action_values)
```

注意：

无限 horizon 的 starter code 中，提取 policy 时应使用收敛后的 `value_function`。有限 horizon 中，如果还有 $k$ 步可走，则使用对应的 $V_{k-1}$ 或 $V_k$ 取决于索引约定。下面单独说明。

## 20. Bellman Operator for a Policy: $B^\pi$

对固定策略 $\pi$，Bellman backup operator 是：

$$
(B^\pi V)(s)
=
R^\pi(s)
+
\gamma
\sum_{s' \in \mathcal{S}}
P^\pi(s' \mid s)V(s')
$$

*首次完整讲解：本讲 §6 给出 $R^\pi$、$P^\pi$，§7 给出固定策略的 Bellman policy evaluation。这里把同一更新封装成 operator $B^\pi$。*

对确定性策略也可写成：

$$
(B^\pi V)(s)
=
R(s,\pi(s))
+
\gamma
\sum_{s' \in \mathcal{S}}
P(s' \mid s,\pi(s))V(s')
$$

这条公式在说什么：

$B^\pi$ 是“按照固定策略 $\pi$ 做一次 Bellman 更新”。它不对动作取最大值。

它解决什么问题：

Policy evaluation 就是寻找 $B^\pi$ 的 fixed point：

$$
V^\pi = B^\pi V^\pi
$$

符号解释：

- $B^\pi$：固定策略的 Bellman operator。
- $B$：最优 Bellman operator，会对动作取最大。
- $V^\pi$：$B^\pi$ 的 fixed point。

容易混淆点：

- $B^\pi$ 用于评估给定策略。
- $B$ 用于最优控制和值迭代。






## 21. Contraction Operator


课件定义 contraction 的直觉：如果对两个输入应用同一个 operator 后，它们之间的距离变小，那么这个 operator 是 contraction。

Bellman optimality operator 在 $\gamma < 1$ 时满足：

$$
\lVert BV - BV' \rVert_\infty
\le
\gamma
\lVert V - V' \rVert_\infty
$$

这条公式在说什么：

对任意两个价值函数 $V$ 和 $V'$，经过 Bellman optimality backup 后，它们之间的最大差距不超过原差距的 $\gamma$ 倍。

它解决什么问题：

如果 $\gamma < 1$，反复应用 $B$ 会收敛到唯一 fixed point，也就是最优价值函数 $V^*$。这就是 value iteration 收敛的数学基础。

符号解释：

- $\lVert \cdot \rVert_\infty$：infinity norm，即最大绝对分量差。
- $B$：Bellman optimality operator。
- $\gamma$：折扣因子，若 $\gamma < 1$ 则有严格收缩。

直觉：

每做一次 Bellman backup，不同初始猜测之间的差异会缩小。迭代足够久后，它们都会走向同一个 $V^*$。

### 21.1 最小数值例子

假设两个价值估计原来的最大差距为 10，并且 $\gamma=0.9$：

$$
\lVert V-V'\rVert_\infty=10
$$

应用一次 Bellman optimality operator 后：

$$
\lVert BV-BV'\rVert_\infty
\le
0.9\times 10
=
9
$$

这不是说新差距一定恰好等于 9，而是说它最多为 9，也可能更小。反复应用后，上界依次变为 $10\times 0.9^k$，最终趋近于 0。

和 Assignment 1 Q3 的关系：

Q3(a) 要你证明 $B^\pi$ 也有类似 contraction：

$$
\lVert B^\pi V - B^\pi V' \rVert_\infty
\le
\gamma
\lVert V - V' \rVert_\infty
$$

这和 $B$ 的证明类似，但没有 $\max_a$，通常更直接。

## 22. Value Iteration 什么时候收敛？

课件给出两个高层条件：

- $\gamma < 1$。
- 或者在满足适当 episodic/properness 条件时最终以概率 1 到达 terminal state。

本课程和 Assignment 1 的主要情形是 discounted infinite horizon，即 $\gamma < 1$。

如果 $\gamma=1$ 且没有终止状态，未来奖励和可能发散，Bellman operator 不再是严格 contraction，收敛保证会变复杂。仅仅存在某个会终止的策略并不足以保证最优 Bellman 更新收敛；必须检查具体 episodic MDP 的终止与 properness 假设。

## 23. Finite-Horizon Value Iteration

这里必须小心索引。为了避免混淆，我们采用清楚约定：

- $V_k(s)$ 表示：从状态 $s$ 出发，还剩 $k$ 次决策机会时的最优价值。
- $V_0(s)=0$，因为没有剩余决策就没有未来奖励。
- 如果整个 episode 总 horizon 是 $H$，起始时使用 $V_H$。

递推为：

$$
V_{k+1}(s)
=
\max_a
\left[
R(s,a)
+
\gamma
\sum_{s' \in \mathcal{S}}
P(s' \mid s,a)V_k(s')
\right],
\quad k=0,1,\ldots,H-1
$$

*公式来源：本讲 §18「Value Iteration」。本节的新内容是把下标 $k$ 明确定义为剩余决策次数，从而得到随剩余时间变化的策略。*

这条公式在说什么：

如果还剩 $k+1$ 次决策，那么先选一个动作拿到即时奖励，然后进入下一状态；之后只剩 $k$ 次决策，因此未来部分使用 $V_k$。

它解决什么问题：

有限 horizon 问题不能简单假设同一个 stationary policy 永远有效。剩余时间不同，最优动作可能不同。

对应策略：

$$
\pi_{k+1}(s)
=
\arg\max_a
\left[
R(s,a)
+
\gamma
\sum_{s' \in \mathcal{S}}
P(s' \mid s,a)V_k(s')
\right]
$$

这条公式在说什么：

如果还剩 $k+1$ 次决策，当前最优动作由 $V_k$ 决定。执行时，若总 horizon 为 $H$，第 $t$ 步剩余决策数是 $H-t$，应使用 $\pi_{H-t}$。

和 `confusions.md` 的关系：

这里的 $k$ 是“剩余决策数”，不是绝对时间步。Assignment 1 Q1 说 horizon $H$ 是 episode 最多能交互的时间步数，从起始状态开始有 $H$ 次动作机会。

### 23.1 Finite-Horizon Policy 是否 stationary？

一般不是。

有限 horizon 中，最优动作可能依赖剩余步数。同一个状态 $s$，如果还剩很多步，可能值得去追远期大奖励；如果只剩一步，可能只能拿即时奖励。

和 Assignment 1 Q1 的关系：

Inventory MDP 中，从 $s=3$ 出发：

- horizon 很短时，卖货拿 +1 可能更好。
- horizon 足够长时，连续买到 $s=10$ 拿 +100 可能更好。

这正是 finite horizon policy 可能依赖时间/剩余步数的例子。

### 23.2 最小数值例子：剩余一步与两步

设状态 $s$ 有两个动作：

- `cash`：立即得到 2，然后终止。
- `invest`：立即得到 0，并确定到达状态 $g$；在 $g$ 再行动可得到 5。

令 $\gamma=1$ 且 $V_0=0$。只剩一步时：

$$
\begin{aligned}
V_1(s)
&=
\max\{2,0\}\\
&=
2
\end{aligned}
$$

此时应选择 `cash`，因为没有时间兑现投资的后续奖励。

先计算：

$$
V_1(g)=5
$$

还剩两步时：

$$
\begin{aligned}
V_2(s)
&=
\max\{2,0+V_1(g)\}\\
&=
\max\{2,5\}\\
&=
5
\end{aligned}
$$

此时应选择 `invest`。同一个状态 $s$ 因剩余步数不同而选择不同动作，这就是 finite-horizon policy 通常依赖时间的具体原因。

## 24. 另一种 Policy Evaluation：Simulation

课件提到，可以用 simulation 估计 finite horizon policy value：

1. 生成大量 episodes。
2. 计算每条 episode 的 return。
3. 对 return 取平均。

优点：

- 不需要 Markov assumption。
- 不需要知道显式 $P$ 和 $R$。
- 可以用于真实环境或 simulator。

缺点：

- 有采样误差。
- 需要很多 episode 才能得到精确估计。
- 对稀疏奖励、长 horizon 任务效率低。

和后续课程关系：

下一讲会进入没有模型时的 policy evaluation，也就是从样本/经验中估计价值。

现代连接：

- Robot learning 常用仿真 rollout 估计策略表现，但真实机器人采样昂贵。
- LLM agents 可以通过多次运行任务、看最终成功率来估计 policy performance，但成本和方差都可能很高。

## 25. Value Iteration vs Policy Iteration

课件总结：

Value iteration：

- 维护的是价值函数。
- 每次应用 optimal Bellman backup。
- 可看作计算 horizon = $k$ 时的最优价值，然后逐渐增加 $k$。
- 收敛后提取 greedy policy。

Policy iteration：

- 维护的是策略。
- 每轮先完整评估当前策略，再贪心改进。
- 有单调 policy improvement 保证。
- 与后续 policy gradient 有概念联系：都围绕直接改进 policy，但方法不同。

重要区别：

Policy iteration 的策略价值单调不降；value iteration 每一轮提取出来的 greedy policy 在真实 infinite-horizon MDP 中不一定保证单调变好。这是课件留作课外思考的问题。

## 26. Assignment 1 代码映射

### 26.1 `bellman_backup`

数学：

$$
\text{backup}(s,a)
=
R(s,a)
+
\gamma
\sum_{s' \in \mathcal{S}}
T(s,a,s')V(s')
$$

*首次完整讲解：Lecture 1 §17 给出 MDP Bellman backup；Lecture 2 §13 说明它在 $V=V^\pi$ 时就是 $Q^\pi(s,a)$。*

代码：

```python
backup_val = R[state, action] + gamma * np.sum(T[state, action] * V)
```

### 26.2 `policy_evaluation`

数学，确定性策略：

$$
V^\pi(s)
=
R(s,\pi(s))
+
\gamma
\sum_{s'}
T(s,\pi(s),s')V^\pi(s')
$$

*首次完整讲解：本讲 §7.1「确定性策略的简化」，迭代实现见 §8。*

代码思路：

```python
while True:
    new_value = np.zeros(num_states)
    for state in range(num_states):
        action = policy[state]
        new_value[state] = bellman_backup(state, action, R, T, gamma, value_function)
    if np.max(np.abs(new_value - value_function)) <= tol:
        value_function = new_value
        break
    value_function = new_value
```

### 26.3 `policy_improvement`

数学：

$$
\pi_{\text{new}}(s)
=
\arg\max_a
\left[
R(s,a)
+
\gamma
\sum_{s'}T(s,a,s')V^\pi(s')
\right]
$$

*首次完整讲解：本讲 §14「Policy Improvement」。*

代码思路：

```python
for state in range(num_states):
    action_values = [
        bellman_backup(state, action, R, T, gamma, V_policy)
        for action in range(num_actions)
    ]
    new_policy[state] = np.argmax(action_values)
```

### 26.4 `policy_iteration`

代码思路：

```python
policy = np.zeros(num_states, dtype=int)
while True:
    V_policy = policy_evaluation(policy, R, T, gamma, tol)
    new_policy = policy_improvement(policy, R, T, V_policy, gamma)
    if np.array_equal(new_policy, policy):
        break
    policy = new_policy
return V_policy, policy
```

如果 break 前刚刚发现 policy 没变，`V_policy` 已经是该最终 policy 的 value function。

### 26.5 `value_iteration`

数学：

$$
V_{k+1}(s)
=
\max_a
\left[
R(s,a)
+
\gamma
\sum_{s'}T(s,a,s')V_k(s')
\right]
$$

*首次完整讲解：本讲 §18「Value Iteration」。*

代码思路：

```python
while True:
    new_value = np.zeros(num_states)
    for state in range(num_states):
        action_values = [
            bellman_backup(state, action, R, T, gamma, value_function)
            for action in range(num_actions)
        ]
        new_value[state] = np.max(action_values)
    if np.max(np.abs(new_value - value_function)) <= tol:
        value_function = new_value
        break
    value_function = new_value

for state in range(num_states):
    action_values = [
        bellman_backup(state, action, R, T, gamma, value_function)
        for action in range(num_actions)
    ]
    policy[state] = np.argmax(action_values)
```

## 27. RiverSwim 直觉

`riverswim.py` 中有 6 个状态、2 个动作：

- `0` 表示 LEFT。
- `1` 表示 RIGHT。

奖励：

- 在最左状态选择 LEFT：小奖励 `0.005`。
- 在最右状态选择 RIGHT：大奖励 `1.0`。

动态：

- LEFT 通常容易，往左成功。
- RIGHT 是逆流而上，有概率停留或被冲回去。
- current 越强，向右成功越难。

这正是 exploration 和 delayed reward 的经典例子：

- 短期看，留在左边拿小奖励稳定。
- 长期看，如果 $\gamma$ 足够大，游到右边后反复拿大奖励更好。

Assignment 1 Q4(d) 要你找每种 current 下最大的 discount factor，使最优 agent 从最左状态不选择 RIGHT。直觉上 current 越强，向右越难，需要更大的 $\gamma$ 才值得追远期大奖。

## 28. Assignment Readiness

Lecture 2 后，Assignment 1 的前置知识基本齐了。

现在可以开始写 Assignment 1。

具体准备度：

- Q1 Effective Horizon：可以开始。需要用 finite horizon 和 infinite horizon discount 的区别分析策略。
- Q2 Reward Hacking：可以开始。Lecture 1 的 reward design 和 Lecture 2 的 optimal policy 都已覆盖直觉。
- Q3 Bellman Residuals：可以开始。Lecture 2 已讲 Bellman operator、$B^\pi$、contraction、fixed point 和 greedy policy。证明仍需要认真写，但不需要等后续 lecture。
- Q4 RiverSwim MDP：可以开始写代码。Lecture 2 已覆盖 `bellman_backup`、`policy_evaluation`、`policy_improvement`、`policy_iteration`、`value_iteration` 的全部核心公式。

建议顺序：

1. 先实现 Q4 的代码，因为公式到代码最直接。
2. 用 sanity check 验证 $\gamma=0.99$、weak current 下 $s_1$ 和 $s_6$ 的 value。
3. 再做 Q1/Q2 written。
4. 最后做 Q3 proofs，因为它最抽象。

## 29. 本讲必会公式

MDP（首次完整讲解：§4）：

$$
\mathcal{M}
=
(\mathcal{S}, \mathcal{A}, P, R, \gamma)
$$

策略（首次完整讲解：Lecture 1 §11；本讲复习：§5）：

$$
\pi(a \mid s)
=
\Pr(a_t=a \mid s_t=s)
$$

最优价值与最优策略（首次完整讲解：§11）：

$$
V^*(s)
=
\max_\pi V^\pi(s),
\quad \forall s\in\mathcal{S}
$$

$$
\pi^*
\in
\left\{
\pi
\mid
V^\pi(s)=V^*(s),
\ \forall s\in\mathcal{S}
\right\}
$$

Policy-induced reward（首次完整讲解：§6）：

$$
R^\pi(s)
=
\sum_{a \in \mathcal{A}}
\pi(a \mid s)R(s,a)
$$

Policy-induced transition（首次完整讲解：§6）：

$$
P^\pi(s' \mid s)
=
\sum_{a \in \mathcal{A}}
\pi(a \mid s)P(s' \mid s,a)
$$

Policy evaluation（首次完整讲解：§7）：

$$
V^\pi(s)
=
\sum_{a \in \mathcal{A}}
\pi(a \mid s)
\left[
R(s,a)
+
\gamma
\sum_{s' \in \mathcal{S}}
P(s' \mid s,a)V^\pi(s')
\right]
$$

确定性 policy evaluation（首次完整讲解：§7.1）：

$$
V^\pi(s)
=
R(s,\pi(s))
+
\gamma
\sum_{s' \in \mathcal{S}}
P(s' \mid s,\pi(s))V^\pi(s')
$$

Q-value（首次完整讲解：§13）：

$$
Q^\pi(s,a)
=
R(s,a)
+
\gamma
\sum_{s' \in \mathcal{S}}
P(s' \mid s,a)V^\pi(s')
$$

Policy improvement（首次完整讲解：§14）：

$$
\pi_{i+1}(s)
=
\arg\max_a Q^{\pi_i}(s,a)
$$

Bellman optimality operator（首次完整讲解：§17）：

$$
(BV)(s)
=
\max_a
\left[
R(s,a)
+
\gamma
\sum_{s' \in \mathcal{S}}
P(s' \mid s,a)V(s')
\right]
$$

Value iteration（首次完整讲解：§18）：

$$
V_{k+1} = BV_k
$$

Contraction（首次完整讲解：§21）：

$$
\lVert BV - BV' \rVert_\infty
\le
\gamma
\lVert V - V' \rVert_\infty
$$

Policy Bellman operator（首次完整讲解：§20）：

$$
(B^\pi V)(s)
=
R(s,\pi(s))
+
\gamma
\sum_{s' \in \mathcal{S}}
P(s' \mid s,\pi(s))V(s')
$$

Finite-horizon value iteration（首次完整讲解：§23）：

$$
V_{k+1}(s)
=
\max_a
\left[
R(s,a)
+
\gamma
\sum_{s' \in \mathcal{S}}
P(s' \mid s,a)V_k(s')
\right]
$$

## 30. 容易混淆点

1. 大 $\gamma$ 不是更短视。

   大 $\gamma$ 让未来奖励更重要；小 $\gamma$ 才更强调即时奖励。

2. $V^\pi$ 和任意 $V$ 不同。

   $V^\pi$ 是某个策略真实诱导出的价值函数。Bellman residual 中的 $V$ 可以是任意向量，不一定对应任何策略。

3. $B$ 和 $B^\pi$ 不同。

   $B$ 对动作取最大值，用于 optimal control；$B^\pi$ 固定策略，用于 policy evaluation。

4. Infinite horizon optimal policy 可以 stationary，但 finite horizon 通常不 stationary。

   有限 horizon 中，同一个状态在不同剩余步数下可能应选不同动作。

5. Policy iteration 和 value iteration 的单调性不同。

   Policy iteration 有 policy improvement 的单调保证；value iteration 的价值估计收敛，但每一轮抽取出的 greedy policy 不一定在真实 infinite-horizon 问题中单调变好。

## 31. 自测题

1. 为什么 MDP + policy 可以看成 MRP？

   因为策略固定后，动作选择分布已知，可以把动作维度对 $\pi(a \mid s)$ 求平均，得到 $R^\pi(s)$ 和 $P^\pi(s' \mid s)$。

2. $Q^\pi(s,a)$ 和 $V^\pi(s)$ 的区别是什么？

   $V^\pi(s)$ 是从状态 $s$ 开始按 $\pi$ 行动的价值；$Q^\pi(s,a)$ 是先强制采取动作 $a$，然后再按 $\pi$ 行动的价值。

3. Policy improvement 为什么不会变差？

   因为在每个状态，新策略选择的动作至少不比旧策略原本选择的动作的 $Q$ 值低；旧动作也在候选动作集合里。

4. Value iteration 为什么在 $\gamma<1$ 时收敛？

   因为 Bellman optimality operator 是 contraction，反复应用会把不同价值估计之间的距离缩小，最终收敛到唯一 fixed point。

5. 有限 horizon 的 optimal policy 为什么可能依赖时间？

   因为剩余步数改变了远期奖励是否来得及实现。同一个状态下，剩 10 步和剩 1 步的最优动作可能不同。

## 32. 本讲小结

Lecture 2 是 Assignment 1 的核心前置课。

主线是：

1. MDP 加策略后变成 MRP，因此可以评估策略。
2. 用 $Q^\pi(s,a)$ 可以比较“换第一步动作”的效果。
3. Policy improvement 让策略单调不差。
4. Policy iteration 交替评估和改进策略。
5. Value iteration 直接迭代 Bellman optimality operator。
6. Contraction 保证 value iteration 在 $\gamma<1$ 时收敛。
7. Finite horizon 需要跟踪剩余步数，策略通常不是 stationary。

下一步如果继续课程，可以进入 Lecture 3：没有模型时如何做 policy evaluation。
如果巩固 Lecture 1-2，现在更建议开始 Assignment 1，尤其是 Q4 的 RiverSwim 代码。

## 33. 延伸阅读

### 33.1 经典基础

- Richard Bellman, *Dynamic Programming*（1957）：动态规划和 Bellman 递推的经典来源；本讲的 policy/value iteration 建立在这套思想上。<https://books.google.com/books/about/Dynamic_Programming.html?id=ZzoS0QEACAAJ>
- Sutton & Barto, *Reinforcement Learning: An Introduction*（2nd ed., 2018），尤其是有限 MDP 与动态规划章节：系统整理 policy evaluation、policy iteration、value iteration 和 generalized policy iteration。MIT Press 书目页：<https://mitpress.mit.edu/9780262039246/reinforcement-learning/>

### 33.2 前沿动态（截至 2026-07-10 核实）

- DreamerV3 的 Nature 论文展示了现代 model-based RL 如何学习 world model，并在模型预测的未来轨迹中训练 actor 与 critic。它与本讲“已知模型后向前推演”的思想直接相连，但模型和价值函数在这里是从数据学习的高维近似，而不是已知表格。<https://www.nature.com/articles/s41586-025-08744-2>
- Physical Intelligence 的 $\pi_{0.7}$ 使用语言子任务和轻量 world model 生成的视觉 subgoal 来调节机器人策略。这是“模型/子目标辅助长程决策”的现实例子，不应等同于本讲的精确 value iteration。<https://www.pi.website/blog/pi07>
