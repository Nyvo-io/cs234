# CS234 Lecture 2 Notes: Making Sequences of Good Decisions Given a Model of the World

来源：`lecture/lec2/lecture2pre.pdf`，CS234 Winter 2026, Professor Emma Brunskill。

本讲目标：在已知世界模型（transition model 和 reward model）的前提下，计算好策略。核心算法是策略迭代（policy iteration, PI）和值迭代（value iteration, VI）。

## 0. 本讲 Checklist

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

## 4. Markov Decision Process（MDP）

MDP 是 MRP 加动作：

$$
\mathcal{M}
=
(\mathcal{S}, \mathcal{A}, P, R, \gamma)
$$

这条公式在说什么：

一个 MDP 由状态集合、动作集合、转移模型、奖励函数和折扣因子构成。它是本课程 planning/control 的标准数学对象。

它解决什么问题：

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

给定一个 MDP 和一个策略 $\pi$，动作选择就固定了。因此 MDP 会诱导出一个 MRP：

$$
(\mathcal{S}, R^\pi, P^\pi, \gamma)
$$

策略诱导的奖励函数：

$$
R^\pi(s)
=
\sum_{a \in \mathcal{A}}
\pi(a \mid s)R(s,a)
$$

这条公式在说什么：

如果在状态 $s$ 下策略会随机选择不同动作，那么该状态的一步奖励就是各动作奖励的加权平均，权重是策略选动作的概率。

它解决什么问题：

它把“有动作的 MDP”变成“没有动作的 MRP”。这样我们就可以用 Lecture 1 的 MRP 价值计算方法来评估策略。

符号解释：

- $R^\pi(s)$：策略 $\pi$ 诱导出的状态奖励。
- $\pi(a \mid s)$：状态 $s$ 下选择动作 $a$ 的概率。
- $R(s,a)$：采取动作 $a$ 的即时奖励期望。

直觉：

如果策略在某状态 70% 选右、30% 选左，那么这个状态的平均奖励就是右动作奖励和左动作奖励的加权平均。

策略诱导的转移模型：

$$
P^\pi(s' \mid s)
=
\sum_{a \in \mathcal{A}}
\pi(a \mid s)P(s' \mid s,a)
$$

这条公式在说什么：

在状态 $s$ 下，策略先随机选动作，再由该动作的转移模型决定下一状态。最终从 $s$ 到 $s'$ 的概率，是对所有动作的加权平均。

它解决什么问题：

它让我们能在固定策略下计算状态转移，不再显式保留动作选择。

符号解释：

- $P^\pi(s' \mid s)$：策略 $\pi$ 下从 $s$ 到 $s'$ 的转移概率。
- $P(s' \mid s,a)$：采取动作 $a$ 后转移到 $s'$ 的概率。

直觉：

策略把动作的不确定性“混进”了状态转移。

和 Assignment 1 的关系：

`policy_evaluation(policy, R, T, gamma)` 输入的是确定性策略 `policy[state] = action`。因此公式会简化，不需要对所有动作按概率求和。

## 7. MDP Policy Evaluation

策略评估（policy evaluation）问：

给定策略 $\pi$，它的价值函数 $V^\pi$ 是多少？

对于随机策略，Bellman 方程是：

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

这条公式在说什么：

状态 $s$ 的策略价值等于：先按策略 $\pi$ 可能选择的所有动作取平均；每个动作的价值等于即时奖励加上折扣后的下一状态价值期望。

它解决什么问题：

它把“这个策略长期表现如何”写成可计算的递推方程。给定 $P$、$R$ 和 $\pi$，我们可以通过迭代求 $V^\pi$。

符号解释：

- $V^\pi(s)$：策略 $\pi$ 下状态 $s$ 的价值。
- $\pi(a \mid s)$：策略在状态 $s$ 下选择动作 $a$ 的概率。
- $R(s,a)$：即时奖励期望。
- $P(s' \mid s,a)$：转移概率。
- $\gamma$：折扣因子。

直觉：

评估策略时，不问“有没有更好的动作”，只问“如果一直按这个策略走，能得到多少长期奖励”。

### 7.1 确定性策略的简化

如果策略是确定性的，即 $\pi(s)$ 直接返回动作，则：

$$
V^\pi(s)
=
R(s,\pi(s))
+
\gamma
\sum_{s' \in \mathcal{S}}
P(s' \mid s,\pi(s))V^\pi(s')
$$

这条公式在说什么：

在状态 $s$，动作已经固定为 $\pi(s)$，所以不需要对动作求和，只需要看这个动作的即时奖励和转移分布。

它解决什么问题：

Assignment 1 的策略是用 `np.array(num_states)` 表示的确定性策略，因此实现时就是这个简化版本。

代码映射：

- $\pi(s)$ -> `policy[state]`
- $R(s,\pi(s))$ -> `R[state, policy[state]]`
- $P(s' \mid s,\pi(s))$ -> `T[state, policy[state], next_state]`
- $V^\pi(s')$ -> `value_function[next_state]`

实现骨架对应：

```python
action = policy[state]
new_value[state] = R[state, action] + gamma * np.sum(T[state, action] * value_function)
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

这条公式在说什么：

第 $k$ 轮使用上一轮价值估计 $V^\pi_{k-1}$ 来更新每个状态的价值。不断重复，直到价值函数变化很小。

它解决什么问题：

不用矩阵求逆，也不用枚举所有轨迹；只需要反复做 Bellman backup。

直觉：

每次迭代都把未来多一步的信息传播回来。

代码停止条件：

Assignment 1 的 `tol` 可以用 infinity norm：

$$
\lVert V_{\text{new}} - V_{\text{old}} \rVert_\infty
=
\max_s |V_{\text{new}}(s)-V_{\text{old}}(s)|
\le \text{tol}
$$

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

课件写成：

$$
\pi^*(s)
=
\arg\max_{\pi} V^\pi(s)
$$

这条公式在说什么：

我们希望找到一个策略，使每个状态下的长期期望回报尽可能高。

它解决什么问题：

Policy evaluation 只能回答“这个策略怎么样”。Control 要回答“该用哪个策略”。

符号解释：

- $\pi^*$：最优策略。
- $V^\pi(s)$：策略 $\pi$ 下状态 $s$ 的价值。

直觉：

评估是打分；控制是找高分策略。

重要结论：

在有限状态动作、infinite-horizon discounted MDP 中，存在 deterministic、stationary 的最优策略。

- Deterministic：每个状态选一个固定动作即可，不一定需要随机。
- Stationary：策略不依赖时间步，只依赖当前状态。
- Unique? 不一定，最优策略可能有多个。

## 12. Policy Search 与 Policy Iteration

暴力搜索所有确定性策略需要检查：

$$
|\mathcal{A}|^{|\mathcal{S}|}
$$

这通常不可行。

策略迭代（policy iteration）更高效。它交替做两件事：

1. Policy evaluation：评估当前策略。
2. Policy improvement：基于当前价值函数改进策略。

算法：

```text
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

## 13. State-Action Value: $Q^\pi(s,a)$

课件定义状态动作价值函数（state-action value function, Q-value）：

$$
Q^\pi(s,a)
=
R(s,a)
+
\gamma
\sum_{s' \in \mathcal{S}}
P(s' \mid s,a)V^\pi(s')
$$

这条公式在说什么：

$Q^\pi(s,a)$ 表示：在状态 $s$ 先采取动作 $a$，之后一直按照策略 $\pi$ 行动，未来能获得的期望折扣回报。

它解决什么问题：

只知道 $V^\pi(s)$ 还不够做改进，因为改进要比较“在同一个状态下换不同动作会怎样”。$Q^\pi(s,a)$ 正好给出每个动作的长期效果。

符号解释：

- $Q^\pi(s,a)$：策略 $\pi$ 下，先做动作 $a$ 再跟随 $\pi$ 的状态动作价值。
- $R(s,a)$：动作 $a$ 的即时奖励。
- $P(s' \mid s,a)$：动作 $a$ 导致下一状态 $s'$ 的概率。
- $V^\pi(s')$：之后按照 $\pi$ 行动时下一状态的价值。

直觉：

$V^\pi(s)$ 是“站在这个状态按原策略走值多少钱”；$Q^\pi(s,a)$ 是“如果第一步我指定成动作 $a$，之后再按原策略走值多少钱”。

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

给定当前策略 $\pi_i$ 的价值函数 $V^{\pi_i}$，先计算：

$$
Q^{\pi_i}(s,a)
=
R(s,a)
+
\gamma
\sum_{s' \in \mathcal{S}}
P(s' \mid s,a)V^{\pi_i}(s')
$$

然后改进策略：

$$
\pi_{i+1}(s)
=
\arg\max_a Q^{\pi_i}(s,a),
\quad \forall s \in \mathcal{S}
$$

这组公式在说什么：

对每个状态，比较所有可选动作。如果某个动作在“先做它、之后按旧策略走”的意义下最好，就把新策略在该状态的动作改成它。

它解决什么问题：

它提供了从一个策略构造更好策略的方法。我们不需要枚举所有策略，只需要局部地检查每个状态的一步改进。

符号解释：

- $\pi_i$：第 $i$ 个策略。
- $V^{\pi_i}$：当前策略的价值函数。
- $Q^{\pi_i}(s,a)$：在 $s$ 先做 $a$，然后跟随当前策略的价值。
- $\arg\max_a$：选择使表达式最大的动作。

直觉：

先问：“如果我只在当前状态改第一步，之后还按老策略走，会不会更好？”如果会，就更新这一状态的动作。令人有力的是，把所有状态都这样更新后，新策略整体不会更差。

和 starter code 的关系：

`policy_improvement(policy, R, T, V_policy, gamma)` 应对每个 state 遍历 action，用 `bellman_backup` 或相同公式算出每个动作值，再取 `argmax`。

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

用 policy improvement 得到的新策略不会比旧策略差。这就是 policy iteration 的核心保证。

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

和 Assignment 1 Q3 的关系：

Q3 中的 Bellman residual：

$$
BV - V
$$

衡量的是一个任意价值函数 $V$ 离 Bellman optimality fixed point 有多远。

## 18. Value Iteration（VI）

值迭代（value iteration）的无限 horizon discounted 形式：

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

等价写法：

$$
V_{k+1} = BV_k
$$

这条公式在说什么：

从初始价值函数 $V_0$ 开始，反复应用 Bellman optimality operator。每轮都根据上一轮价值估计，更新每个状态的最优价值估计。

它解决什么问题：

Value iteration 不显式维护一个策略再完整评估它，而是直接迭代最优价值函数。

符号解释：

- $V_k$：第 $k$ 轮价值估计。
- $B$：Bellman optimality operator。
- $R(s,a)$、$P(s' \mid s,a)$、$\gamma$ 同前。

直觉：

第 1 轮看到一步奖励；第 2 轮看到两步以内的最佳未来；迭代越多，越远的未来奖励被纳入。

终止条件：

$$
\lVert V_{k+1} - V_k \rVert_\infty
\le
\epsilon
$$

和 starter code 的关系：

`value_iteration(R, T, gamma, tol)` 应循环：

1. 对每个 state，计算所有 actions 的 backup。
2. `new_value[state] = max(action_values)`。
3. 如果 `np.max(np.abs(new_value - value_function)) <= tol`，停止。
4. 用最终 value function 提取 greedy policy。

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

对任意两个价值函数 $V$ 和 $V'$，经过 Bellman optimality backup 后，它们之间的最大差距至少被缩小到原来的 $\gamma$ 倍。

它解决什么问题：

如果 $\gamma < 1$，反复应用 $B$ 会收敛到唯一 fixed point，也就是最优价值函数 $V^*$。这就是 value iteration 收敛的数学基础。

符号解释：

- $\lVert \cdot \rVert_\infty$：infinity norm，即最大绝对分量差。
- $B$：Bellman optimality operator。
- $\gamma$：折扣因子，若 $\gamma < 1$ 则有严格收缩。

直觉：

每做一次 Bellman backup，不同初始猜测之间的差异会缩小。迭代足够久后，它们都会走向同一个 $V^*$。

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

课件给出两个条件：

- $\gamma < 1$。
- 或者最终以概率 1 到达 terminal state。

本课程和 Assignment 1 的主要情形是 discounted infinite horizon，即 $\gamma < 1$。

如果 $\gamma=1$ 且没有终止状态，未来奖励和可能发散，Bellman operator 不再是严格 contraction，收敛保证会变复杂。

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

MDP：

$$
\mathcal{M}
=
(\mathcal{S}, \mathcal{A}, P, R, \gamma)
$$

策略：

$$
\pi(a \mid s)
=
\Pr(a_t=a \mid s_t=s)
$$

Policy-induced reward:

$$
R^\pi(s)
=
\sum_{a \in \mathcal{A}}
\pi(a \mid s)R(s,a)
$$

Policy-induced transition:

$$
P^\pi(s' \mid s)
=
\sum_{a \in \mathcal{A}}
\pi(a \mid s)P(s' \mid s,a)
$$

Policy evaluation：

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

确定性 policy evaluation：

$$
V^\pi(s)
=
R(s,\pi(s))
+
\gamma
\sum_{s' \in \mathcal{S}}
P(s' \mid s,\pi(s))V^\pi(s')
$$

Q-value：

$$
Q^\pi(s,a)
=
R(s,a)
+
\gamma
\sum_{s' \in \mathcal{S}}
P(s' \mid s,a)V^\pi(s')
$$

Policy improvement：

$$
\pi_{i+1}(s)
=
\arg\max_a Q^{\pi_i}(s,a)
$$

Bellman optimality operator：

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

Value iteration：

$$
V_{k+1} = BV_k
$$

Contraction：

$$
\lVert BV - BV' \rVert_\infty
\le
\gamma
\lVert V - V' \rVert_\infty
$$

Policy Bellman operator：

$$
(B^\pi V)(s)
=
R(s,\pi(s))
+
\gamma
\sum_{s' \in \mathcal{S}}
P(s' \mid s,\pi(s))V(s')
$$

Finite-horizon value iteration：

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
