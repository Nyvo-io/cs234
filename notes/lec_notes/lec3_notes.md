# CS234 Lecture 3 Notes: Model-Free Policy Evaluation

来源：`lecture/lec3/lecture3pre.pdf`，CS234 Winter 2026，Professor Emma Brunskill。

本讲主题：在不知道真实转移模型和奖励模型时，只利用策略与环境交互得到的直接经验（direct experience），估计固定策略的价值。这就是无模型策略评估（model-free policy evaluation）。

> 课件标题页编号显示整套 slides 还有附加内容，但当前 `lecture3pre.pdf` 实际包含 53 个 PDF 页面。当前文件没有课件末尾提到的可选解答页，因此例题结果在本笔记中按照课件给出的轨迹与标准定义逐步推导。

## 0. 本讲 Checklist

- [x] 复习回报、状态价值函数、动作价值函数和动态规划策略评估。
- [x] 理解无模型策略评估的目标，以及为什么直接经验很重要。
- [x] 理解自举（bootstrapping）的含义。
- [x] 掌握蒙特卡洛策略评估（Monte Carlo policy evaluation）的基本思想。
- [x] 区分首次访问 MC（first-visit MC）和每次访问 MC（every-visit MC）。
- [x] 掌握 MC 的样本平均与增量更新形式。
- [x] 理解如何评价估计器：一致性、计算复杂度、内存、统计效率与均方误差。
- [x] 区分偏差（bias）、方差（variance）、均方误差（MSE）和一致性（consistency）。
- [x] 理解首次访问 MC、每次访问 MC 和增量 MC 的统计性质。
- [x] 掌握随机逼近学习率的收敛条件。
- [x] 掌握 TD(0) 的目标、TD 误差和在线更新算法。
- [x] 理解 TD 为什么同时具有 MC 和动态规划的特征。
- [x] 比较 MC 与 TD 的偏差、方差、更新时机和适用场景。
- [x] 理解确定性等价（certainty equivalence）与最大似然 MDP 模型。
- [x] 掌握批量 MC 与批量 TD 在固定数据集上的不同收敛结果。
- [x] 完成课件 AB 示例并解释 Markov 假设的作用。
- [x] 建立与 robot learning、LLM agents、RLHF/PPO、planning 和 tool use 的联系。
- [x] 判断 Lecture 3 对 Assignment 1 和 Assignment 2 的准备程度。

## 1. 开场复习：值迭代（Value Iteration）与策略迭代（Policy Iteration）

课件首先复习 Lecture 2 的两个问题。

### 1.1 两种算法最终得到的策略价值是否相同？

在有限表格型 MDP（tabular MDP）中，如果模型已知、计算精确且算法运行到收敛：

- 策略迭代（policy iteration）收敛到某个最优策略。
- 值迭代（value iteration）收敛到唯一的最优价值函数 $V^*$，再从中提取某个贪心最优策略。

最优策略可能不唯一，但所有最优策略都具有相同的最优价值函数。因此，两种算法返回的具体动作规则可能不同，但其价值都等于 $V^*$。

### 1.2 值迭代会不会超过 $|\mathcal{S}||\mathcal{A}|$ 轮？

会。$|\mathcal{S}||\mathcal{A}|$ 不是值迭代的迭代次数上界。

值迭代是收缩映射的反复应用。它达到给定误差容忍度所需的轮数取决于折扣因子 $\gamma$、初始误差和目标精度；当 $\gamma$ 接近 1 时，可能需要很多轮。状态动作对数量主要影响每一轮更新的计算量，不直接限制轮数。

这两个问题的作用是把 Lecture 2 的“已知模型规划”过渡到 Lecture 3 的“未知模型估值”。

## 2. 本讲在课程中的位置

Lecture 2 解决的问题是：

> 已知真实模型 $P$ 和 $R$ 时，如何评估策略并寻找最优策略？

Lecture 3 解决的问题是：

> 不知道真实模型，只能执行固定策略 $\pi$ 并观察轨迹时，如何估计 $V^\pi$？

Lecture 4 将继续解决：

> 不知道真实模型时，如何不只评估固定策略，而是进一步改进策略并完成控制（control）？

因此，本讲仍然是策略评估（policy evaluation），不是控制。策略 $\pi$ 已经给定，我们只想知道它有多好。

## 3. 统一符号和时间索引约定

为了避免课件中不同页的下标写法造成混淆，本笔记采用以下约定：

- 一条 episode 包含 $T_i$ 次转移，时间下标为 $t=0,1,\ldots,T_i-1$。
- 在 $s_{i,t}$ 采取 $a_{i,t}$ 后，观察奖励 $r_{i,t}$ 和下一状态 $s_{i,t+1}$。
- $s_{i,T_i}$ 是 episode 的终止状态。
- $r_{i,t}$ 是从 $s_{i,t}$ 执行动作后得到的奖励；有些教材会把同一个奖励写作 $R_{t+1}$。

第 $i$ 条 episode 写成：

$$
\tau_i
=
(s_{i,0},a_{i,0},r_{i,0},s_{i,1},a_{i,1},r_{i,1},\ldots,s_{i,T_i})
$$

这条表达式在说什么：

轨迹（trajectory）记录智能体依次经历的状态、动作、奖励和下一状态。Lecture 3 的所有方法都从这类经验中估计价值。

它解决什么问题：

明确奖励属于哪一次转移后，才能正确写回报的求和上下限，也才能避免 $r_t$ 与教材中 $R_{t+1}$ 的一位偏移。

符号解释：

- $i$：episode 编号。
- $t$：episode 内的时间步。
- $T_i$：第 $i$ 条 episode 的转移总数。
- $s_{i,t}$、$a_{i,t}$、$r_{i,t}$：该时间步的状态、动作和转移后奖励。

直觉：

一条轨迹就是智能体实际走过的一条路径。无模型方法不知道完整的概率树，只能从走过的路径学习。

## 4. 复习：回报和价值函数

### 4.1 有限 episode 的回报

从时间 $t$ 开始的折扣回报（discounted return）为：

$$
G_{i,t}
=
\sum_{k=0}^{T_i-1-t}
\gamma^k r_{i,t+k}
$$

这条公式在说什么：

$G_{i,t}$ 把第 $i$ 条 episode 中从时间 $t$ 开始直到终止的所有奖励加起来，并让距离当前更远的奖励乘上更高次幂的折扣因子。

它解决什么问题：

单步奖励只说明一步结果，回报则把后续整个决策序列压缩成一个数，用来评价“从当前状态继续执行策略最终有多好”。

符号解释：

- $G_{i,t}$：第 $i$ 条 episode 从时间 $t$ 开始的回报。
- $T_i$：该 episode 的转移总数。
- $k$：相对当前时间的偏移量。
- $r_{i,t+k}$：未来第 $k$ 步的奖励。
- $\gamma \in [0,1]$：折扣因子。

直觉：

当前奖励权重为 1，下一步奖励权重为 $\gamma$，再下一步为 $\gamma^2$。越远的奖励通常影响越小。

和算法/作业的关系：

- MC 直接把完整的 $G_{i,t}$ 当作价值学习目标。
- Assignment 2 的 `PolicyGradient.get_returns` 需要计算同一类 $G_t$。
- 从 episode 尾部反向递推，可以在 $O(T_i)$ 时间内计算所有 $G_{i,t}$。

### 4.2 无限时域的回报

在持续任务且级数收敛时：

$$
G_t
=
\sum_{k=0}^{\infty}
\gamma^k r_{t+k}
$$

这里没有有限的终止时间。通常要求 $\gamma<1$ 且奖励有界，从而保证折扣和有限。

### 4.3 状态价值函数

$$
V^\pi(s)
=
\mathbb{E}_\pi[G_t \mid s_t=s]
$$

这条公式在说什么：

$V^\pi(s)$ 是从状态 $s$ 开始，之后一直执行策略 $\pi$ 时，未来回报的条件期望。

它解决什么问题：

同一个状态之后可能出现很多随机轨迹。价值函数不是某一条轨迹的回报，而是这些可能回报的平均水平。

符号解释：

- $V^\pi(s)$：策略 $\pi$ 的状态价值。
- $G_t$：从时间 $t$ 开始的随机回报。
- $\mathbb{E}_\pi$：对策略选择和环境转移共同产生的随机性取期望。

直觉：

$G_t$ 是一次实际结果，$V^\pi(s)$ 是重复从 $s$ 出发很多次后的长期平均结果。

### 4.4 动作价值函数

$$
Q^\pi(s,a)
=
\mathbb{E}_\pi[G_t \mid s_t=s,a_t=a]
$$

这条公式在说什么：

$Q^\pi(s,a)$ 衡量先在状态 $s$ 强制采取动作 $a$，之后再继续执行 $\pi$ 的期望回报。

它解决什么问题：

$V^\pi$ 只评价状态，$Q^\pi$ 能比较同一状态下不同第一步动作。Lecture 4 的无模型控制会直接学习动作价值。

## 5. 已知模型时的动态规划（Dynamic Programming）与自举（Bootstrapping）

对随机策略，Bellman 期望算子为：

$$
(B^\pi V)(s)
=
\sum_{a \in \mathcal{A}}
\pi(a \mid s)
\left[
R(s,a)
+
\gamma
\sum_{s' \in \mathcal{S}}
P(s' \mid s,a)V(s')
\right]
$$

这条公式在说什么：

在状态 $s$，先按策略概率平均所有动作，再按环境转移概率平均所有下一状态。当前价值由即时奖励和下一状态价值共同决定。

它解决什么问题：

如果 $P$ 和 $R$ 已知，就不必采样大量轨迹，可以直接枚举动作与下一状态完成策略评估。

符号解释：

- $B^\pi$：固定策略的 Bellman 算子。
- $R(s,a)$：状态动作对的期望即时奖励。
- $P(s' \mid s,a)$：转移到 $s'$ 的概率。
- $V(s')$：当前对下一状态价值的估计。

直觉：

动态规划在整棵概率树上做加权平均，而不是只看一条实际走过的路径。

迭代策略评估写成：

$$
V_{k+1}
=
B^\pi V_k
$$

这里用旧估计 $V_k(s')$ 来更新新估计 $V_{k+1}(s)$。这种“用已有估计更新另一个估计”的方式叫自举（bootstrapping）。

自举并不是“使用真实答案”。它使用的是当前价值估计，所以能提前传播信息，但也会把当前估计中的偏差带进目标。

## 6. 无模型策略评估：只依靠直接经验

现在假设不知道：

- $P(s' \mid s,a)$；
- $R(s,a)$；
- 完整的状态转移图。

我们拥有的是执行策略 $\pi$ 得到的样本：

$$
(s_t,a_t,r_t,s_{t+1})
$$

本讲首先假设这些样本来自当前要评估的同一个策略 $\pi$，称为同策略数据（on-policy data）。使用其他策略产生的数据评估 $\pi$ 属于异策略评估（off-policy evaluation），后续 lecture 才会系统处理。

直接经验重要的原因：

1. 真实世界动力学通常未知。
2. 即使存在模拟器，也可能无法读取其内部转移概率。
3. 连续、高维环境无法枚举全部状态和下一状态。
4. Robot learning 和 agent 系统通常只能通过执行动作获得反馈。

## 7. 蒙特卡洛策略评估（Monte Carlo Policy Evaluation）：价值等于平均回报

蒙特卡洛策略评估（Monte Carlo policy evaluation, MC）的核心是：

$$
V^\pi(s)
=
\mathbb{E}_\pi[G_t \mid s_t=s]
\approx
\frac{1}{N(s)}
\sum_{j=1}^{N(s)}G_j(s)
$$

这条公式在说什么：

真实价值是条件期望。MC 不知道这个期望的解析形式，于是收集多次访问状态 $s$ 后观察到的完整回报，并用样本平均近似期望。

它解决什么问题：

它绕过未知的 $P$ 和 $R$，直接从完整 episode 的结果估计策略价值。

符号解释：

- $N(s)$：用于估计状态 $s$ 的回报样本数量。
- $G_j(s)$：第 $j$ 个从状态 $s$ 开始观察到的回报样本。
- 样本来自执行策略 $\pi$ 的 episode。

直觉：

不知道骰子的概率分布时，可以反复掷骰子并计算平均值。MC 对回报做同样的事。

和算法/作业的关系：

- REINFORCE 使用采样回报作为 $Q^{\pi_\theta}(s,a)$ 的 MC 估计。
- Assignment 2 的回报计算与 baseline 回归都建立在这一步上。

MC 的基本要求：

- episode 必须能够终止；
- 必须等 episode 完成后，才能知道早期状态的完整 $G_t$；
- 不需要知道模型；
- 单纯 MC 平均不要求状态表示严格满足 Markov 假设。

## 8. 首次访问蒙特卡洛（First-Visit Monte Carlo）

首次访问 MC（first-visit MC）在每条 episode 中，每个状态只使用它第一次出现时的回报。

对第 $i$ 条 episode 中状态 $s$ 的首次出现时间：

$$
t_i^{\mathrm{first}}(s)
=
\min\{t \mid s_{i,t}=s\}
$$

对应的估计为：

$$
\hat V_{\mathrm{FV}}^\pi(s)
=
\frac{1}{N_{\mathrm{FV}}(s)}
\sum_{i:\,s\in\tau_i}
G_{i,t_i^{\mathrm{first}}(s)}
$$

这组公式在说什么：

一条 episode 即使多次回到 $s$，也只贡献一个样本，即从该 episode 第一次到达 $s$ 时开始的完整回报。

它解决什么问题：

同一 episode 内多个访问的回报高度相关。每条 episode 只取一次，可以更直接地把每条 episode 看成一次独立实验。

符号解释：

- $t_i^{\mathrm{first}}(s)$：第 $i$ 条 episode 首次访问 $s$ 的时间。
- $N_{\mathrm{FV}}(s)$：包含状态 $s$ 的 episode 数量。
- $\hat V_{\mathrm{FV}}^\pi(s)$：首次访问 MC 估计。

直觉：

每条 episode 对状态 $s$ 只投一票，避免在某条循环很多次的 episode 中重复计票。

### 8.1 首次访问 MC 算法

1. 初始化每个状态的计数 `count[s] = 0` 和价值 `V[s] = 0`。
2. 执行策略 $\pi$，采样一条完整 episode。
3. 从 episode 尾部向前计算所有回报。
4. 对每个状态，只在该 episode 的首次出现处更新一次。
5. 重复采样。

数学上的样本均值更新为：

$$
N(s)
\leftarrow
N(s)+1
$$

$$
V(s)
\leftarrow
V(s)
+
\frac{1}{N(s)}
\left[G_{i,t}-V(s)\right]
$$

## 9. 每次访问蒙特卡洛（Every-Visit Monte Carlo）

每次访问 MC（every-visit MC）使用每一次出现状态 $s$ 时的回报：

$$
\hat V_{\mathrm{EV}}^\pi(s)
=
\frac{
\sum_i\sum_{t=0}^{T_i-1}
\mathbf{1}\{s_{i,t}=s\}G_{i,t}
}{
\sum_i\sum_{t=0}^{T_i-1}
\mathbf{1}\{s_{i,t}=s\}
}
$$

这条公式在说什么：

只要某个时间步访问了 $s$，就把该时间步之后的回报加入平均值。一条 episode 可以为同一个状态提供多个样本。

它解决什么问题：

它使用了数据中的所有访问，通常比 first-visit 更充分地利用已有经验。

符号解释：

- $\mathbf{1}\{s_{i,t}=s\}$：指示函数，访问 $s$ 时为 1，否则为 0。
- 分子：所有访问 $s$ 后的回报总和。
- 分母：访问 $s$ 的总次数。

直觉：

First-visit 是“每条 episode 每个状态最多一个样本”；every-visit 是“每次访问都算一个样本”。

容易混淆点：

“首次访问”指从 episode 开头向后看时的最早出现，不是从尾部反向计算回报时首先碰到的那一次。

## 10. Mars Rover：首次访问与每次访问

课件给出轨迹：

$$
(s_3,a_1,0,s_2,a_1,0,s_2,a_1,0,s_1,a_1,1,\text{terminal})
$$

按本笔记的索引约定，各转移奖励依次为 $0,0,0,1$。

状态 $s_2$ 出现两次：

- 第一次出现后的回报为：

$$
G_{\mathrm{first}}(s_2)
=
0+\gamma\cdot 0+\gamma^2\cdot 1
=
\gamma^2
$$

- 第二次出现后的回报为：

$$
G_{\mathrm{second}}(s_2)
=
0+\gamma\cdot 1
=
\gamma
$$

因此：

$$
\hat V_{\mathrm{FV}}(s_2)
=
\gamma^2
$$

$$
\hat V_{\mathrm{EV}}(s_2)
=
\frac{\gamma^2+\gamma}{2}
$$

这组计算在说什么：

两种 MC 方法看到的是同一条轨迹，区别只在于是否使用第二次访问 $s_2$ 的回报。

它解决什么问题：

这个例子把“访问规则”从语言描述落实到具体数字，防止把 first-visit 误解成只取离终点最近的一次访问。

符号解释：

- $\gamma^2$：从第一次访问 $s_2$ 到最终奖励 1 相隔两个折扣步骤。
- $\gamma$：从第二次访问 $s_2$ 到最终奖励相隔一个折扣步骤。

直觉：

第一次访问离奖励更远，所以得到的折扣回报更小。

> 课件第 36 页底部给出的状态向量与该页所列重复 $s_2$ 的轨迹、以及标准 first-visit 定义存在索引歧义。本笔记以上述明确的“奖励属于转移”约定为准。做题时应先写清奖励索引，再计算回报。

## 11. 增量蒙特卡洛（Incremental Monte Carlo）：不保存全部历史

假设状态 $s$ 已有 $n-1$ 个回报样本，旧平均为 $V_{n-1}(s)$，新样本为 $G_n(s)$。新的样本平均是：

$$
V_n(s)
=
V_{n-1}(s)
+
\frac{1}{n}
\left[G_n(s)-V_{n-1}(s)\right]
$$

这条公式在说什么：

新平均等于旧平均，加上“新样本与旧平均之差”的 $1/n$。

它解决什么问题：

不需要保存所有历史回报，也不需要每次重新求和。每个状态只需保存当前价值和访问次数。

符号解释：

- $V_{n-1}(s)$：加入新样本前的平均。
- $G_n(s)$：新观察到的回报。
- $1/n$：精确样本平均对应的步长。

直觉：

样本少时，新数据影响大；样本多时，单个新数据只会轻微改变平均值。

和代码的关系：

```text
count[state] += 1
value[state] += (return_t - value[state]) / count[state]
```

这里：

- $V(s)$ -> `value[state]`
- $G_t$ -> `return_t`
- $N(s)$ -> `count[state]`

## 12. 一般学习率形式

把 $1/N(s)$ 换成一般学习率 $\alpha$：

$$
V(s_t)
\leftarrow
V(s_t)
+
\alpha
\left[G_t-V(s_t)\right]
$$

这条公式在说什么：

价值估计沿着“目标减当前估计”的方向移动一小步。这里目标（target）是完整回报 $G_t$。

它解决什么问题：

统一写出大量 RL 更新规则：

$$
\text{new estimate}
=
\text{old estimate}
+
\text{step size}
\times
(\text{target}-\text{old estimate})
$$

符号解释：

- $\alpha$：学习率（learning rate）或步长（step size）。
- $G_t$：MC target。
- $G_t-V(s_t)$：当前样本对应的估计误差。

直觉：

目标高于当前估计就上调，目标低于当前估计就下调。

和后续算法的关系：

TD、SARSA、Q-learning 和许多深度 RL 方法都保留这个更新骨架，只是目标不同。

## 13. 策略评估图（Policy Evaluation Diagram）：概率树与采样路径

课件用交替的白色状态节点和黑色动作节点画出策略评估树：

1. 在状态节点，策略 $\pi(a\mid s)$ 产生动作随机性。
2. 在动作节点，环境 $P(s'\mid s,a)$ 产生下一状态随机性。
3. 这个过程不断重复，直到终止状态或无限延续。

完整价值是对整棵树上所有可能分支取期望。不同算法对这棵树的处理方式不同：

- 动态规划：已知模型，对动作和下一状态分支显式求期望。
- MC：不知道模型，采样完整的根到终点路径，再平均完整回报。
- TD：不知道模型，采样一步分支，再用下一状态的当前价值估计补足后续。

这个图是 Lecture 3 的核心视觉直觉：MC 与 TD 的差别，不是有没有数据，而是“目标中使用完整采样后果”还是“用一步采样加估计”。

## 14. 如何评价一个 Policy Evaluation 方法

课件提出五类指标。

### 14.1 一致性（Consistency）

数据越来越多时，估计是否收敛到真实策略价值。

### 14.2 计算复杂度（Computational Complexity）

新数据到来后，更新估计需要多少计算。例如增量 MC 的单次更新很便宜，而反复重建模型并求解动态规划可能很贵。

### 14.3 内存需求（Memory Requirements）

需要保存完整 episode、全部历史数据、模型计数，还是只保存当前价值表。

### 14.4 统计效率（Statistical Efficiency）

在相同数据量下，估计能有多准确。真实世界交互昂贵时，统计效率往往比计算效率更重要。

### 14.5 经验准确率（Empirical Accuracy）

常用均方误差衡量：

$$
\frac{1}{|\mathcal{S}|}
\sum_{s\in\mathcal{S}}
\left[\hat V(s)-V^\pi(s)\right]^2
$$

这里需要基准真值 $V^\pi$；在研究实验中通常通过已知模拟环境或高精度估计获得。

## 15. 偏差（Bias）、方差（Variance）与均方误差（MSE）

设要估计的真实参数为 $\theta$，数据为 $X$，估计器为 $\hat\theta(X)$。

### 15.1 偏差

$$
\operatorname{Bias}_\theta(\hat\theta)
=
\mathbb{E}[\hat\theta]-\theta
$$

这条公式在说什么：

偏差衡量重复采样数据并重新估计后，估计结果的平均值与真实参数相差多少。

它解决什么问题：

判断一个方法是否系统性高估或低估，而不是只看某一次误差。

直觉：

若许多次实验的估计平均正好落在真值上，估计器无偏；单次实验仍可能偏离很远。

### 15.2 方差

$$
\operatorname{Var}(\hat\theta)
=
\mathbb{E}
\left[
(\hat\theta-\mathbb{E}[\hat\theta])^2
\right]
$$

这条公式在说什么：

方差衡量换一批数据后，估计结果本身波动多大。

它解决什么问题：

即使估计器平均正确，方差过大也会导致有限数据下的结果不可靠。

直觉：

MC 的完整回报受整条随机轨迹影响，常有较大方差。

### 15.3 均方误差

$$
\operatorname{MSE}(\hat\theta)
=
\mathbb{E}
\left[
(\hat\theta-\theta)^2
\right]
=
\operatorname{Var}(\hat\theta)
+
\operatorname{Bias}_\theta(\hat\theta)^2
$$

这条公式在说什么：

MSE 同时惩罚随机波动和系统偏差。估计器可以接受少量偏差，以换取更大的方差下降和更低的总 MSE。

它解决什么问题：

只比较“是否无偏”并不足以判断有限数据下谁更好，MSE 给出更全面的准确性标准。

符号解释：

- 第一项是方差。
- 第二项是偏差的平方。
- MSE 越小，平均平方误差越小。

直觉：

TD 的 target 带有当前估计，因此可能有偏；但它通常比 MC 低方差，所以有限数据下可能具有更低 MSE。

## 16. 一致估计器（Consistent Estimator）不等于无偏估计器（Unbiased Estimator）

当样本量为 $n$ 时估计为 $\hat\theta_n$。一致性定义为：对任意 $\varepsilon>0$，

$$
\lim_{n\to\infty}
\Pr
\left(
|\hat\theta_n-\theta|>\varepsilon
\right)
=
0
$$

这条公式在说什么：

数据量无限增加时，估计仍与真值相差超过任意固定误差 $\varepsilon$ 的概率趋近于 0。

它解决什么问题：

一致性描述长期极限是否正确；偏差描述固定样本量下重复实验的平均位置。两者不是同一个概念。

符号解释：

- $n$：数据量。
- $\varepsilon$：任意正误差容忍度。
- 概率对随机数据集取。

直觉：

一致性要求估计随着数据增加越来越集中到真值附近。

课件提问：“无偏是否一定一致？”答案是否定的。

例如，每个 $n$ 都只用第一个样本 $X_1$ 估计总体均值：

$$
\hat\theta_n=X_1
$$

若 $\mathbb{E}[X_1]=\theta$，它始终无偏；但增加更多数据并没有降低方差，因此一般不一致。

反过来，一个估计器在有限样本下可以有偏，但偏差随 $n$ 增大而消失，从而仍然一致。

## 17. 蒙特卡洛估计器（Monte Carlo Estimator）的统计性质

### 17.1 First-Visit MC

在 episode 独立采样、回报期望存在等常规条件下：

- first-visit MC 是 $V^\pi(s)$ 的无偏估计；
- 当包含 $s$ 的 episode 数量趋于无穷时，由大数定律收敛到 $V^\pi(s)$；
- 因此它是一致估计器。

### 17.2 Every-Visit MC

同一 episode 内的重复访问不是相互独立样本，而且访问次数会让某些 episode 获得更大权重。因此有限样本下 every-visit MC 一般有偏。

但是在常规遍历条件下，它仍是一致估计器，并且因为使用了更多访问，实践中经常具有更好的 MSE。

这里体现了重要原则：

> 无偏不自动意味着有限样本最好；最终应比较 MSE、数据量和任务成本。

### 17.3 Incremental MC

增量 MC 的性质取决于学习率。如果使用精确样本平均步长 $1/N(s)$，它与保存所有样本再求平均等价。

如果使用一般的 $\alpha_n(s)$，则需要随机逼近条件。

## 18. 学习率（Learning Rate）的收敛条件

对每个被无限访问的状态 $s$，经典条件是：

$$
\sum_{n=1}^{\infty}
\alpha_n(s)
=
\infty
$$

$$
\sum_{n=1}^{\infty}
\alpha_n(s)^2
<
\infty
$$

这两条公式在说什么：

第一条要求总步长不能过早耗尽，否则估计可能在到达真值前就停止移动。第二条要求平方步长总和有限，让随机噪声的长期累计影响受到控制。

它解决什么问题：

在随机 target 下，既要保持继续学习的能力，又要让后期更新逐渐稳定。

符号解释：

- $n$：状态 $s$ 的第 $n$ 次更新，不一定是全局时间步。
- $\alpha_n(s)$：该次更新使用的学习率。

直觉：

步长要“下降，但不能下降得太快”。

典型选择：

$$
\alpha_n(s)
=
\frac{1}{n}
$$

它满足两条条件，因为调和级数发散，而平方倒数级数收敛。

固定学习率 $\alpha$ 不满足第二条，因此在平稳环境中通常不会精确收敛到单一点，而会在真值附近持续波动。不过在非平稳环境中，固定学习率能保留追踪变化的能力。

## 19. Monte Carlo 的局限与总结

MC 的主要优点：

- 无需模型。
- 概念简单，直接用经验平均估计期望。
- 不依赖自举。
- 单纯 MC 回报估计不要求状态严格 Markov。
- 在常规条件下收敛到真实策略价值。

MC 的主要局限：

1. **高方差**：完整回报受到后续所有随机动作、状态和奖励影响。
2. **数据需求大**：降低方差可能需要很多 episode。
3. **必须等 episode 结束**：无法在刚观察一步后立即得到完整 $G_t$。
4. **主要用于 episodic setting**：持续任务没有自然终点时，完整回报难以直接获得。
5. **真实交互可能昂贵或危险**：在机器人、医疗、教育等场景中，大量试验并不现实。

即使知道模型，有时也会使用 MC：当状态空间巨大、完整动态规划代价高，而采样便宜时，采样估计可能更实用。

## 20. 时序差分学习（Temporal-Difference Learning）的核心思想

时序差分学习（temporal-difference learning, TD）结合两种思想：

- 像 MC 一样，从真实经验采样，不需要模型。
- 像动态规划一样，用下一状态的当前价值估计进行自举。

它可以在观察到每个转移后立刻更新，不必等 episode 结束，也能用于没有终点的持续任务。

课件本讲介绍的是一步 TD，即 TD(0)。

## 21. 从 Bellman Equation 推导 TD(0)

真实价值满足 Bellman 期望方程：

$$
V^\pi(s_t)
=
\mathbb{E}_\pi
\left[
r_t+\gamma V^\pi(s_{t+1})
\mid s_t
\right]
$$

这条公式在说什么：

状态的真实价值等于“一步奖励加下一状态真实价值”的条件期望。

它解决什么问题：

如果模型未知，无法显式计算右边的期望，但可以用一次实际转移 $(s_t,a_t,r_t,s_{t+1})$ 作为样本。

用当前估计 $V$ 代替未知的 $V^\pi$，得到一步样本 target。此后 TD 公式中的 $V$ 都表示“当前对 $V^\pi$ 的估计”，不是已经知道的真实价值：

$$
y_t^{\mathrm{TD}}
=
r_t
+
\gamma V(s_{t+1})
$$

这条公式在说什么：

TD target 由真实观察到的一步奖励和对下一状态未来回报的当前估计组成。

它解决什么问题：

不必观察完整未来，就能为 $V(s_t)$ 构造近似学习目标。

符号解释：

- $r_t$：真实采样的一步奖励。
- $V(s_{t+1})$：当前价值表对下一状态的估计，不是真实答案。
- $\gamma$：下一状态未来价值的折扣。

直觉：

“已经发生的一步用真实数据，尚未发生的未来用当前预测补上。”

## 22. TD 误差（TD Error）与 TD(0) 更新

TD 误差（TD error）定义为：

$$
\delta_t
=
r_t
+
\gamma V(s_{t+1})
-
V(s_t)
$$

这条公式在说什么：

$\delta_t$ 比较“一步后重新得到的价值判断”和更新前对当前状态的价值判断。

它解决什么问题：

它给出价值应该上调还是下调，以及差多少。

符号解释：

- $r_t+\gamma V(s_{t+1})$：TD target。
- $V(s_t)$：当前预测。
- $\delta_t$：target 与 prediction 的差。

直觉：

若实际一步结果加后续估计比预想更好，$\delta_t>0$；若更差，$\delta_t<0$。

TD(0) 更新为：

$$
V(s_t)
\leftarrow
V(s_t)
+
\alpha\delta_t
$$

展开后：

$$
V(s_t)
\leftarrow
V(s_t)
+
\alpha
\left[
r_t+\gamma V(s_{t+1})-V(s_t)
\right]
$$

这条公式在说什么：

每观察一个转移，就把当前状态价值向一步 TD target 移动 $\alpha$ 比例。

它解决什么问题：

在不知道 $P$、$R$ 且不等待完整 episode 的情况下，在线估计 $V^\pi$。

和代码的关系：

```text
td_target = reward + gamma * value[next_state]
td_error = td_target - value[state]
value[state] += alpha * td_error
```

对应关系：

- $V(s_t)$ -> `value[state]`
- $V(s_{t+1})$ -> `value[next_state]`
- $r_t$ -> `reward`
- $\delta_t$ -> `td_error`

## 23. 终止状态（Terminal State）的处理

如果 $s_{t+1}$ 是终止状态，约定：

$$
V(s_{t+1})=0
$$

于是：

$$
y_t^{\mathrm{TD}}=r_t
$$

终止状态之后没有未来奖励，因此不能继续 bootstrap。代码中通常写成：

```text
td_target = reward if done else reward + gamma * value[next_state]
```

容易混淆点：终止转移的即时奖励仍然保留；被置零的是终止状态之后的未来价值。

## 24. TD(0) 算法

输入学习率 $\alpha$，初始化 $V(s)$，然后循环：

1. 按策略 $\pi$ 在状态 $s_t$ 选择动作 $a_t$。
2. 执行动作并观察 $r_t,s_{t+1}$。
3. 构造 TD target。
4. 更新 $V(s_t)$。
5. 进入 $s_{t+1}$，继续交互。

如果策略是随机策略，必须按 $\pi(a\mid s)$ 采样动作。策略评估阶段不能为了追求高回报擅自换成贪心动作，否则采集到的数据不再对应要评估的策略。

## 25. Mars Rover：单条轨迹上的 TD 更新

课件使用同一条轨迹，设：

- 初始 $V(s)=0$；
- $\alpha=1$；
- 按时间顺序在线更新；
- 终止状态价值为 0。

轨迹：

$$
s_3\xrightarrow{0}s_2
\xrightarrow{0}s_2
\xrightarrow{0}s_1
\xrightarrow{1}\text{terminal}
$$

逐步更新：

1. $s_3\to s_2$：target 为 $0+\gamma V(s_2)=0$，所以 $V(s_3)=0$。
2. 第一次 $s_2\to s_2$：target 仍为 0，所以 $V(s_2)=0$。
3. 第二次 $s_2\to s_1$：此时 $V(s_1)$ 还没更新，target 为 0，所以 $V(s_2)=0$。
4. $s_1\to\text{terminal}$：target 为 1，所以 $V(s_1)=1$。

episode 结束时：

$$
V(s_1)=1,
\qquad
V(s_2)=0,
\qquad
V(s_3)=0
$$

这组结果在说什么：

TD(0) 每次只把信息向前传播一步。最终奖励 1 在本条 episode 中先更新到 $s_1$；需要后续再次经过 $s_2$、$s_3$，才能继续向更早状态传播。

它解决什么问题：

这个例子解释了 TD 可以立即更新，但单次更新只使用一步 bootstrap，信息传播速度与访问顺序有关。

与 MC 的对比：

MC 等 episode 结束后能立即为早期状态使用完整回报；TD 不等结束，但需要多次访问逐步传播远期奖励。

## 26. 学习率判断题：$\alpha=0$ 与 $\alpha=1$

课件的判断题结论：

1. **$\alpha=0$ 会更重视 TD target：错误。** 此时完全不更新，旧估计权重为 1。
2. **$\alpha=1$ 会把价值直接设为 TD target：正确。**
3. **在存在随机下一状态的 MDP 中，$\alpha=1$ 可能永远振荡：正确。** 每次用一个随机样本完全覆盖旧值，噪声不会被平均掉。
4. **存在确定性 MDP，使 $\alpha=1$ 的 TD 收敛：正确。** 例如确定性链条反复访问时，价值能从终点逐步稳定向前传播。

一般更新也可写成加权平均：

$$
V_{\mathrm{new}}(s_t)
=
(1-\alpha)V_{\mathrm{old}}(s_t)
+
\alpha y_t^{\mathrm{TD}}
$$

这条公式清楚显示：$\alpha$ 是新 target 的权重，$1-\alpha$ 是旧估计的权重。

## 27. TD(0) 的性质

TD(0) 的关键性质：

- **无模型**：不需要显式知道 $P$ 和 $R$。
- **采样**：每次使用一个真实转移样本。
- **自举**：target 中包含当前估计 $V(s_{t+1})$。
- **在线更新**：观察一个 transition 后即可更新。
- **适用于 episodic 和 continuing tasks**。
- **有限数据下通常有偏**：target 依赖初始化和当前不准确的 $V$。
- **通常比 MC 方差低**：不把整条随机未来都放进单个 target。
- **在 Markov、充分访问和适当学习率等条件下是一致估计器**。

课件提到，更一般的方法可以在 TD(0) 与 MC 之间插值。后续会看到多步回报和 TD($\lambda$)：

- TD(0) 使用一步 target。
- MC 使用直到 episode 结束的完整 return。
- 多步 TD 使用有限步真实奖励，再 bootstrap。

## 28. Monte Carlo 与 TD(0) 的系统比较

| 维度 | Monte Carlo | TD(0) |
|---|---|---|
| 是否需要模型 | 不需要 | 不需要 |
| target | 完整回报 $G_t$ | $r_t+\gamma V(s_{t+1})$ |
| 是否 bootstrap | 否 | 是 |
| 何时更新 | episode 结束后 | 每个 transition 后 |
| 是否要求 episode 终止 | 通常需要 | 不需要 |
| Markov 假设 | 单纯回报平均不要求 | 收敛到正确状态价值依赖 Markov 表示 |
| 有限样本偏差 | first-visit 通常无偏 | 通常有偏 |
| 方差 | 通常较高 | 通常较低 |
| 信息传播 | 完整回报一次传播到整条轨迹 | 一步一步传播 |

核心不是简单地问“谁更好”，而是做偏差与方差、数据与计算、在线性与完整回报之间的权衡。

## 29. 确定性等价（Certainty Equivalence）：先学模型，再做动态规划

除了直接学习 $V^\pi$，还可以先从数据估计 MDP，再把估计模型当成真实模型进行规划。这叫确定性等价（certainty equivalence）。

给定数据中的访问计数：

$$
N(s,a)
=
\sum_{i,t}
\mathbf{1}
\{s_{i,t}=s,a_{i,t}=a\}
$$

最大似然转移模型（maximum-likelihood transition model）为：

$$
\hat P(s'\mid s,a)
=
\frac{
\sum_{i,t}
\mathbf{1}
\{s_{i,t}=s,a_{i,t}=a,s_{i,t+1}=s'\}
}{N(s,a)}
$$

奖励模型估计为：

$$
\hat R(s,a)
=
\frac{
\sum_{i,t}
\mathbf{1}
\{s_{i,t}=s,a_{i,t}=a\}r_{i,t}
}{N(s,a)}
$$

这组公式在说什么：

转移概率用经验频率估计，奖励用对应状态动作对下观察到的奖励平均值估计。

它解决什么问题：

即使不知道真实 MDP，也可以从数据构建一个经验 MDP $\hat{\mathcal M}$，再复用 Lecture 2 的动态规划方法。

符号解释：

- $N(s,a)$：数据中状态动作对 $(s,a)$ 的访问次数。
- 分子中的指示函数：统计从 $(s,a)$ 转移到 $s'$ 的次数。
- $\hat P$、$\hat R$：最大似然模型估计。

直觉：

观察到从 $(s,a)$ 出发 10 次，其中 7 次到 $s_1'$，就估计该转移概率为 0.7。

和 planning 的关系：

估计模型后，对固定策略构造 $\hat P^\pi$ 和 $\hat R^\pi$，再求：

$$
\hat V^\pi
=
\hat R^\pi
+
\gamma \hat P^\pi \hat V^\pi
$$

若矩阵可逆：

$$
\hat V^\pi
=
(I-\gamma\hat P^\pi)^{-1}
\hat R^\pi
$$

这两条公式表示：价值不是直接从 return 平均出来，而是在学到的 Markov 模型中通过 Bellman 方程计算出来。

## 30. 确定性等价的代价与适用条件

优点：

- 数据效率高，一条转移可帮助多个上游状态的规划。
- 估计出模型后，可以评估多个策略。
- 只要数据覆盖目标策略所需的状态动作对，也能用于 off-policy evaluation。
- 对正确的有限 Markov 模型，在充分数据下是一致的。

代价：

- 需要保存模型计数或数据。
- 每次更新模型后重新求解价值可能很贵。
- 课件给出的典型计算量是：矩阵解析解约 $O(|\mathcal S|^3)$；通用迭代 backup 的一轮约 $O(|\mathcal S|^2|\mathcal A|)$。
- 未访问的 $(s,a)$ 无法直接形成可靠的最大似然估计，需要初始化、先验或额外探索。
- 如果状态表示不满足 Markov 假设，经验模型可能把不同隐藏情形错误合并。

必须补充覆盖条件：有 off-policy 数据并不自动代表能评估任何策略。如果目标策略经常采取数据中从未出现的动作，$\hat P$ 和 $\hat R$ 没有依据。

## 31. 批量策略评估（Batch Policy Evaluation）

批量策略评估（batch policy evaluation），也称离线批量评估（offline batch evaluation），固定一组有限数据，不再与环境获得新经验。

课件设有 $K$ 条 episode，并反复从中抽取 episode，重复应用 MC 或 TD(0) 更新直到稳定。

问题是：

> 在同一份固定数据上，batch MC 与 batch TD(0) 最终会收敛到同一个答案吗？

答案：不一定。

原因不是优化没收敛，而是两种方法对有限数据做了不同假设：

- Batch MC 拟合数据中实际观察到的完整回报。
- Batch TD 拟合一个满足经验 Bellman 方程的 Markov 价值函数。

## 32. AB Example

设 $\gamma=1$，数据共 8 条 episode：

1. 一条：$A\xrightarrow{0}B\xrightarrow{0}\text{terminal}$。
2. 六条：$B\xrightarrow{1}\text{terminal}$。
3. 一条：$B\xrightarrow{0}\text{terminal}$。

### 32.1 两种方法对 $B$ 的估计

$B$ 共出现 8 次，其中 6 次得到回报 1，2 次得到回报 0：

$$
V(B)
=
\frac{6}{8}
=
0.75
$$

对 $B$，batch MC 与 batch TD 都得到 0.75。

### 32.2 Batch MC 对 $A$ 的估计

$A$ 只出现一次，该次从 $A$ 开始的实际回报是 0：

$$
V_{\mathrm{MC}}(A)
=
0
$$

这条公式在说什么：

MC 忠实拟合从 $A$ 实际观察到的唯一完整 return，不借用其他 episode 中关于 $B$ 的信息。

### 32.3 Batch TD 对 $A$ 的估计

经验模型认为从 $A$ 出发：

- 即时奖励始终为 0；
- 下一状态始终是 $B$。

因此经验 Bellman 方程给出：

$$
V_{\mathrm{TD}}(A)
=
0
+
1\cdot V(B)
=
0.75
$$

这条公式在说什么：

TD 使用 Markov 假设，把其他 episode 中对 $B$ 的所有观察共享给 $A$。只要当前在 $B$，之前如何到达 $B$ 不应影响未来分布。

它解决什么问题：

说明 batch TD 并不是在拟合每个状态的经验 return 平均，而是在拟合经验 MDP 的 Bellman fixed point。

直觉：

MC 说：“我唯一一次从 $A$ 出发看到的结果就是 0。”

TD 说：“从 $A$ 必然到 $B$；而全部数据表明 $B$ 平均值是 0.75，所以 $A$ 也应为 0.75。”

有限数据下无法只凭这组数据断言哪个更接近真实 MDP。TD 的答案依赖“$B$ 是 Markov state”这一结构假设；如果 $B$ 隐藏了不同历史情形，MC 可能更稳健。

## 33. Batch MC 与 Batch TD 的收敛对象

反复重放固定数据直到更新稳定时：

### 33.1 Batch MC

收敛到每个状态在数据集中观察到的完整 return 的经验平均，也可理解为最小化对这些 return target 的平方误差。

### 33.2 Batch TD(0)

收敛到最大似然 Markov 模型的确定性等价解，即经验 Bellman 方程的 fixed point。

这个结果解释了 TD 的统计效率来源：它利用 Markov 结构在不同 episode 之间共享后继状态信息。相应风险是，如果状态表示不是 Markov 的，这种信息共享会引入模型错设误差。

## 34. 四类策略评估方法的统一比较

| 方法 | 已知真实模型 | 使用采样 | Bootstrap | 主要收敛对象 | 主要代价 |
|---|---:|---:|---:|---|---|
| Dynamic Programming | 是 | 否 | 是 | 真实模型的 $V^\pi$ | 枚举状态与转移 |
| Monte Carlo | 否 | 完整 episode | 否 | 经验 return 平均 | 高方差、等待终止 |
| TD(0) | 否 | 一步 transition | 是 | 在线时趋向 $V^\pi$；batch 时为经验 Bellman 解 | 有偏、依赖 Markov 表示 |
| Certainty Equivalence | 先从数据估计 | 用数据建模 | 规划阶段是 | 经验 MDP 中的 $V^\pi$ | 建模与重复规划昂贵 |

可以把它们放在两条轴上理解：

1. **采样还是期望**：使用实际样本，还是显式对模型概率求和。
2. **完整回报还是 bootstrap**：等到真实后果完全出现，还是用当前价值估计补足未来。

## 35. 与现代方向的联系

### 35.1 RLHF、REINFORCE 与 PPO

**直接联系：** REINFORCE 使用 rollout 的 MC return 估计动作价值。Assignment 2 明确使用：

$$
G_t
\approx
Q^{\pi_\theta}(s_t,a_t)
$$

这继承了 MC 的特点：不需要环境模型，但回报方差可能很大。

PPO 等 actor-critic 方法通常训练价值函数或 critic 来预测未来回报。critic 提供 baseline 或 bootstrap 信息，从而降低策略梯度估计的方差。后续讲到优势估计时，会看到 MC 与 TD 之间的折中。

对 LLM 的 RL 训练可以把一次生成看作 trajectory，把 token 选择看作 action，把上下文前缀看作 state-like representation。若奖励主要在回答结束时给出，早期 token 的 credit assignment 就类似本讲的延迟回报问题。

注意：这是数学结构上的直接联系，不意味着实际 LLM 状态严格满足有限表格 MDP 假设。

### 35.2 LLM Agents 与 Tool-Use Agents

**概念类比：** 一次“搜索、打开网页、调用代码、形成答案”的工具链可以看作 episode。MC 会等整个任务结束后，用最终成功与否评价早期工具调用；TD 思路则尝试用中间状态价值估计提前更新。

价值估计可回答：

> 在当前上下文和工具执行结果下，继续按照该 agent policy 行动，最终完成任务的期望收益是多少？

难点是 agent 的文本上下文可能不是真正 Markov state，并且奖励常常稀疏、带噪声或只在终点出现。

### 35.3 Embodied Intelligence 与 Robot Learning

**直接联系：** 机器人真实交互昂贵，MC 需要大量完整 episode 且方差高，可能不够实用。TD 能逐步在线更新，模型学习加 planning 则可能提高数据效率。

但 bootstrap 会传播错误价值，机器人中的函数逼近、传感器部分可观测性和分布变化会进一步放大稳定性问题。Lecture 3 的 bias-variance 与 data-compute tradeoff 是选择算法时的基础。

### 35.4 Planning 与 World Models

确定性等价展示了最基础的 model-based pipeline：

1. 从经验学习 dynamics/reward model。
2. 在学到的模型中进行 policy evaluation 或 planning。

现代 world model 方法把表格计数替换成高维函数逼近，但核心问题没有改变：模型是否准确、数据是否覆盖、规划计算是否可承受，以及模型误差会不会在长时域中累积。

### 35.5 Offline Data

Batch MC 与 batch TD 对应固定日志上的学习。机器人日志、历史推荐记录和 agent tool traces 都可能形成离线数据。

本讲只讨论固定策略评估的基础。真正的 off-policy/offline RL 还必须处理目标策略与数据策略不一致、动作覆盖不足和分布外估计，不能直接把 batch TD 当成完整解决方案。

## 36. Assignment Readiness

### 36.1 Assignment 1

Lecture 3 不改变此前判断：Assignment 1 的前置知识已经在 Lecture 1-2 覆盖。

**现在可以开始写 Assignment 1。**

Lecture 3 还能帮助理解：

- Q4 中 simulation-based evaluation 与基于模型 DP 的区别。
- 为什么实际交互中需要从样本估计价值。

### 36.2 Assignment 2

Lecture 3 只覆盖了 Assignment 2 的一部分前置知识，**现在还不应开始整份 Assignment 2**。

已经具备的部分：

- Policy Gradient Q2(a)：理解 $G_t$，并能从 episode 尾部用递推在 $O(T)$ 时间计算全部 returns。
- `PolicyGradient.get_returns`：已经理解数学目标，但实际实现还需结合 starter code 的张量形状。
- Baseline 的价值 target：已经理解用 $G_t$ 拟合 $V(s_t)$ 以及 MSE、bias 和 variance。
- REINFORCE 为什么使用 MC returns：已有概念基础。

尚未具备的部分：

- DQN 与 tabular Q-learning：需要后续无模型控制和函数逼近内容。
- 完整 policy gradient theorem 与 REINFORCE 更新：需要后续 policy gradient lecture。
- PPO clipping、old policy probability ratio：需要后续 PPO/policy gradient 内容。
- Policy-induced distributions 与 performance difference：需要后续 occupancy distribution 和 advantage 内容。

正确学习顺序：继续 Lecture 4，同时 Assignment 1 可以并行完成；不要因为看懂了 $G_t$ 就提前把 Assignment 2 整体标记为 ready。

## 37. 本讲必会公式

有限 episode 回报：

$$
G_{i,t}
=
\sum_{k=0}^{T_i-1-t}
\gamma^k r_{i,t+k}
$$

状态价值：

$$
V^\pi(s)
=
\mathbb{E}_\pi[G_t\mid s_t=s]
$$

MC 经验平均：

$$
\hat V^\pi(s)
=
\frac{1}{N(s)}
\sum_{j=1}^{N(s)}G_j(s)
$$

增量 MC：

$$
V_n(s)
=
V_{n-1}(s)
+
\frac{1}{n}
\left[G_n(s)-V_{n-1}(s)\right]
$$

一般 MC 更新：

$$
V(s_t)
\leftarrow
V(s_t)
+
\alpha
\left[G_t-V(s_t)\right]
$$

MSE 分解：

$$
\operatorname{MSE}(\hat\theta)
=
\operatorname{Var}(\hat\theta)
+
\operatorname{Bias}_\theta(\hat\theta)^2
$$

一致性：

$$
\lim_{n\to\infty}
\Pr
\left(
|\hat\theta_n-\theta|>\varepsilon
\right)
=
0
$$

学习率条件：

$$
\sum_{n=1}^{\infty}\alpha_n(s)=\infty,
\qquad
\sum_{n=1}^{\infty}\alpha_n(s)^2<\infty
$$

TD target：

$$
y_t^{\mathrm{TD}}
=
r_t+\gamma V(s_{t+1})
$$

TD error：

$$
\delta_t
=
r_t+\gamma V(s_{t+1})-V(s_t)
$$

TD(0) update：

$$
V(s_t)
\leftarrow
V(s_t)+\alpha\delta_t
$$

MLE transition model：

$$
\hat P(s'\mid s,a)
=
\frac{N(s,a,s')}{N(s,a)}
$$

MLE reward model：

$$
\hat R(s,a)
=
\frac{1}{N(s,a)}
\sum_{i,t}
\mathbf{1}\{s_{i,t}=s,a_{i,t}=a\}r_{i,t}
$$

经验 MDP 中的策略价值：

$$
\hat V^\pi
=
(I-\gamma\hat P^\pi)^{-1}\hat R^\pi
$$

## 38. 容易混淆点

1. **Return sample 与 value 不同。**

   $G_t$ 是一条实际轨迹上的随机样本；$V^\pi(s)$ 是给定状态后对所有可能轨迹回报的期望。

2. **First-visit 不是从后往前第一次看到。**

   它指 episode 按时间从前向后时，状态的最早出现。

3. **MC target 与 TD target 不同。**

   MC target 是完整 $G_t$；TD target 是 $r_t+\gamma V(s_{t+1})$。

4. **TD 不等于 model-based。**

   TD 使用 $V(s_{t+1})$ bootstrap，但不需要显式学习或查询 $P$、$R$。

5. **Bootstrap target 不是真实答案。**

   $V(s_{t+1})$ 也是当前估计，因此 TD target 会带有偏差。

6. **无偏不保证一致，一致也不要求每个有限样本量都无偏。**

7. **固定 $\alpha$ 通常不会精确收敛。**

   它会保留噪声，但适合追踪非平稳目标。

8. **Batch TD 不等于 Batch MC。**

   Batch MC 拟合实际 return；batch TD 拟合经验 Markov 模型的 Bellman fixed point。

9. **终止状态处理。**

   终止后的未来价值为 0，但进入终止状态的即时奖励不能丢掉。

10. **Reward indexing 可能相差一位。**

    本笔记用 $r_t$ 表示在 $(s_t,a_t)$ 后观察到的奖励；Sutton & Barto 常写成 $R_{t+1}$。公式必须与所用约定一致。

## 39. 自测题

1. 为什么 MC 不需要 Markov 假设，而 TD 通常需要？

   MC 直接平均从当前表示开始观察到的完整 return；TD 假设下一状态的价值足以概括后续未来，从而能把不同历史下到达同一状态的数据合并。

2. 为什么 TD 可以在 continuing task 中更新？

   TD 只需要一步奖励和下一状态价值，不需要等待 episode 终止。

3. First-visit MC 与 every-visit MC 的差别是什么？

   First-visit 每条 episode 对每个状态最多使用一个回报；every-visit 使用该状态的所有访问回报。

4. 为什么 TD 通常低方差但有偏？

   它只采样一步，减少了完整未来轨迹的随机性；但 target 包含当前不准确的 $V(s_{t+1})$，因此有限数据下有偏。

5. $\alpha=1$ 在随机 MDP 中有什么风险？

   每次都用单个随机 TD target 完全覆盖旧估计，可能持续振荡而不平均噪声。

6. AB example 中，为什么 MC 得到 $V(A)=0$，TD 得到 $V(A)=0.75$？

   MC 只使用唯一一次从 $A$ 出发的实际 return 0；TD 利用 Markov 假设，把所有 episode 对 $B$ 的估计 0.75 通过 $A\to B$ 的经验转移传播给 $A$。

7. 两个学习率级数条件分别控制什么？

   $\sum\alpha_n=\infty$ 保证持续学习；$\sum\alpha_n^2<\infty$ 保证噪声影响逐渐受控。

8. 如何在 $O(T)$ 时间计算一条 episode 的全部 $G_t$？

   从尾部向前维护一个变量：

   $$
   G_t=r_t+\gamma G_{t+1}
   $$

   每个时间步只做一次更新。

## 40. 本讲小结

Lecture 3 的主线是：在不知道世界模型时，如何从执行固定策略得到的经验估计价值。

1. MC 等完整 episode 结束，用完整 return 的样本平均估计 $V^\pi$。
2. First-visit 和 every-visit 的区别在于一条 episode 内重复状态如何计样本。
3. 增量更新把 RL 算法统一成“旧估计 + 学习率 × 误差”。
4. Bias、variance、MSE 和 consistency 必须分开判断。
5. TD(0) 用一步真实奖励加下一状态估计构造 target，能够在线 bootstrap。
6. MC 通常无偏但高方差；TD 通常有偏但低方差。
7. Certainty equivalence 先学习最大似然 MDP，再使用动态规划。
8. 固定数据集上，batch MC 拟合经验 return，batch TD 拟合经验 Markov 模型的 Bellman 解。
9. 本讲已经为 Assignment 2 的 return 计算和 baseline 价值估计打下基础，但整份 Assignment 2 仍需后续 lecture。

下一步：

- Assignment 1 尚未完成时，可以现在开始写 Assignment 1。
- 继续课程时进入 Lecture 4：没有模型时的控制（model-free control）。
