---
title: CS234 Lecture 3 - Model-Free Policy Evaluation
aliases:
  - CS234 Lec3
tags:
  - cs234
  - reinforcement-learning
  - model-free
  - monte-carlo
  - temporal-difference
---

# CS234 Lecture 3 Notes: Model-Free Policy Evaluation

来源：`lecture/lec3/lecture3pre.pdf`，CS234 Winter 2026，Emma Brunskill。当前 PDF 有 53 个物理页面，页脚编号到 54/67；缺少的附加页不在本地文件中，本文不会把它们当成已读取内容。

笔记规范：cs234-rl-tutor v2。下方覆盖清单只表示课件内容已经写入，不表示学习者已经掌握。

外部资料核验日期：2026-07-19。

相关笔记：[[lec2_notes|Lecture 2：已知模型下的规划]]、[[lec4_notes|Lecture 4：无模型控制]]、[[lec5_notes|Lecture 5：Policy Gradient I]]。

## 0. 本讲覆盖清单

- [x] 回答 value iteration 与 policy iteration 的开场复习题。
- [x] 明确无模型策略评估的目标、direct experience 和 on-policy 数据假设。
- [x] 复习 return、$V^\pi$、$Q^\pi$、动态规划策略评估和 bootstrapping。
- [x] 掌握 Monte Carlo policy evaluation 的适用条件与基本思想。
- [x] 区分 first-visit MC 与 every-visit MC，并写出完整算法。
- [x] 对照 PDF 的 Mars Rover 轨迹计算两类 MC estimate，并记录课件答案冲突。
- [x] 掌握 incremental MC、一般学习率更新和 policy-evaluation diagram。
- [x] 使用 consistency、计算复杂度、内存、统计效率和 MSE 评价估计方法。
- [x] 区分 bias、variance、MSE、unbiasedness 与 consistency。
- [x] 理解 first-visit、every-visit 和 incremental MC 的统计性质。
- [x] 掌握随机逼近学习率的两个级数条件与 MC 的局限。
- [x] 推导 TD(0) target、TD error、更新式和在线算法。
- [x] 处理 terminal transition，并完成课件 Mars Rover TD trace。
- [x] 回答 $\alpha=0$、$\alpha=1$ 的 TD 判断题。
- [x] 比较 MC 与 TD 的采样、bootstrapping、bias、variance 和 Markov 要求。
- [x] 掌握 certainty equivalence、MLE MDP、复杂度、coverage 条件。
- [x] 掌握 batch MC / batch TD、AB example 和不同收敛对象。
- [x] 连接 Assignment 2 的 return / baseline，并补充 LLM 与具身智能中的近期直接应用。

## 1. 本讲主线

Lecture 2 假设转移模型 $P$ 和奖励模型 $R$ 已知，因此能用动态规划计算 $V^\pi$。Lecture 3 删除了这个条件：现在只有执行固定策略 $\pi$ 得到的轨迹，目标仍然是估计同一个 $V^\pi$。

本讲的三条路线是：

1. **Monte Carlo（MC）**：等 episode 完成，用完整 return 的样本平均逼近期望。
2. **Temporal-Difference（TD）**：观察一步转移后，立刻用“真实一步奖励 + 下一状态当前价值估计”更新。
3. **Certainty equivalence**：先从数据估计一个 MDP，再暂时把它当作真实模型做动态规划。

理解它们不应只背更新式，而应沿两条轴比较：

- **采样还是显式期望**：用实际轨迹，还是对模型概率求和；
- **完整后果还是 bootstrap**：等待完整 return，还是用当前价值估计补足未来。

固定有限数据上的 batch MC 和 batch TD 会得到不同答案。AB example 说明：MC 拟合观测到的完整 return，TD 拟合经验 Markov 模型的 Bellman fixed point。

## 2. 从已知模型过渡到直接经验

### 2.1 开场复习：VI 与 PI

*首次完整讲解：Lecture 2 §12「Policy iteration」和 §18「Value iteration」。本节只补充：课件开场的两个判断。*

在有限 tabular MDP、模型已知、计算精确并运行到收敛时：

(**Tabular MDP（表格型 MDP）**，就是：
> 状态和动作的数量都有限，而且可以把 MDP 中的所有信息直接存进数组或表格里)

- policy iteration 收敛到一个最优策略；
- value iteration 收敛到唯一的 $V^*$，再提取一个 greedy optimal policy。

最优策略可以不唯一，但所有最优策略的价值都是 $V^*$。所以“两种算法最终策略的 value 是否相同”的答案是 **True**；具体动作规则未必相同。

Value iteration 也完全可能运行超过 $|\mathcal S||\mathcal A|$ 轮。这个乘积影响一轮 backup 的规模，不是迭代轮数上界；达到给定精度所需轮数还取决于 $\gamma$、初值和误差容忍度。

### 2.2 本讲的输入和目标

    VI 和 PI 假设你知道环境模型，可以直接计算；Lecture 3 假设你不知道模型，只能通过与环境交互获得样本，再从样本中估计价值

![[Pasted image 20260720235736.png]]

Lecture 3 的输入不再是完整的 $P,R$，而是执行策略 $\pi$ 得到的直接经验（direct experience）：

$$
(s_t,a_t,r_t,s_{t+1}).
$$

课件本讲默认数据由正在评估的同一策略 $\pi$ 产生，也就是 on-policy samples。用另一个 behavior policy 的数据评估 $\pi$ 属于 off-policy evaluation，后续课程再系统展开。

直接经验之所以重要：

- 真实机器人、推荐系统或交互式 agent 的 dynamics 往往未知；
- 模拟器即使可运行，也未必开放内部转移概率；
- 连续、高维状态无法枚举全部 $P(s'\mid s,a)$；
- 真实策略表现最终需要通过执行和观测验证。

课件总结页用“估计一次产品推荐 session 的平均购买量”说明固定策略评估：策略已经确定，问题是它长期表现如何，不是在本讲寻找最优策略。

![[Pasted image 20260720235820.png]]

### 2.3 统一时间索引

本文约定第 $i$ 条 episode 有 $T_i$ 次转移，$t=0,\ldots,T_i-1$。在 $s_{i,t}$ 采取 $a_{i,t}$ 后，观察 $r_{i,t}$ 和 $s_{i,t+1}$；$s_{i,T_i}$ 是终止状态。

*首次完整讲解：Lecture 1 §15.2「回报（Return）」。本节只补充：多 episode 下标 $i$。*

$$
G_{i,t}
=
\sum_{k=0}^{T_i-1-t}\gamma^k r_{i,t+k}.
$$

$G_{i,t}$ 是一条实际轨迹上的随机 return sample。状态价值是对所有可能未来的条件期望：

$$
V^\pi(s)=\mathbb E_\pi[G_t\mid s_t=s],
$$

而动作价值还固定第一步动作：

$$
Q^\pi(s,a)=\mathbb E_\pi[G_t\mid s_t=s,a_t=a].
$$

不能把一次 $G_t$ 叫作真实 $V^\pi(s_t)$ 或 $Q^\pi(s_t,a_t)$；它只是对期望目标的一次观测。

## 3. 无模型策略评估与 bootstrapping

*首次完整讲解：Lecture 3 §3「无模型策略评估与 bootstrapping」。*

**无模型策略评估（model-free policy evaluation）**是在不读取真实 $P,R$ 的条件下，只利用策略产生的经验估计固定策略价值 $V^\pi$。这里的“model-free”描述算法不需要显式模型，不表示环境不存在 dynamics，也不表示算法不需要数据。

### 3.1 Bootstrapping

*首次完整讲解：Lecture 3 §3.1「Bootstrapping」。*

Bootstrapping（自举）是：**用一个尚未收敛的估计值，构造另一个估计值的更新 target。**

固定策略的 Bellman operator 已在 Lecture 2 §20 完整讲解：

$$
(B^\pi V)(s)
=
\sum_a\pi(a\mid s)
\left[
R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V(s')
\right].
$$

动态规划更新 $V_{k+1}=B^\pi V_k$ 时，右侧的 $V_k(s')$ 是当前估计，不是真值。它正是一次 bootstrap。

**自拟例子**：若状态 $A$ 的当前奖励为 1，下一步必然到 $B$，$\gamma=0.5$，当前估计 $V_k(B)=4$，一次 backup 得到：

$$
V_{k+1}(A)=1+0.5\times4=3.
$$

输出 3 是新估计。Bootstrapping 能提前传播信息，但也会把当前估计的误差带进 target；这正是 TD 与 MC 的主要分界。

## 4. Monte Carlo policy evaluation

*首次完整讲解：Lecture 3 §4「Monte Carlo policy evaluation」。*

先问一个实际问题：策略 $\pi$ 已经固定，但我们不知道环境模型，只有几条跑完的 episode。怎样估计“从状态 $s$ 出发平均能得到多少回报”？

**先看最小例子。** 从同一个状态 $s$ 开始执行三次，三条完整轨迹最后得到的 return 分别是 $2,5,8$。最自然的估计就是平均：

$$
\hat V^\pi(s)=\frac{2+5+8}{3}=5.
$$

这个 5 不是某一条轨迹的真实结果，而是用三次完整经验对期望价值的估计。把这个操作写成一般公式，就是：

$$
V^\pi(s)
=
\mathbb E_\pi[G_t\mid s_t=s]
\approx
\frac{1}{N(s)}\sum_{j=1}^{N(s)}G_j(s).
$$

其中 $N(s)$ 是目前收集到、包含状态 $s$ 的 return sample 数。MC 不需要 dynamics 或 reward model，也不 bootstrap；但必须等到 episode 结束才能知道完整 $G_t$。最基础的 episodic MC 还要求 episode 最终终止。

不同 episode 不要求长度相同；第 $i$ 条轨迹使用自己的终止时间 $T_i$。这也是 return 公式保留 episode 下标 $i$ 的原因。

MC 直接平均“从当前表示开始实际观察到的完整未来”，因此单纯 MC return estimation 不要求这个表示严格满足 Markov property。相比之下，TD 会把不同历史下的同一表示共享为一个下一状态价值，因此更依赖 Markov 表示。

这里的 **MC** 和 **TD**，都是在不知道环境模型 $P,R$ 时，根据实际采集到的轨迹来估计策略价值 $V^\pi(s)$ 的方法。

它们的目标相同：

> 给定策略 π，估计“从状态 s 开始，之后一直执行 π，平均能获得多少累计回报”。

真正的区别是：

- **MC：等整局结束，用真实完整回报更新**
- **TD：每走一步，就用奖励和下一状态的估计值更新**

![[Pasted image 20260721005925.png]]

### 4.1 First-visit MC

*首次完整讲解：Lecture 3 §4.1「First-visit MC」。*

先不看公式，先看它要解决的问题：

> 一条 episode 里，同一个状态可能出现很多次，到底用哪一次对应的 return 来更新 V(s)？

First-visit MC（首次访问 MC）的回答是：**每条 episode 只使用状态第一次出现时对应的 return；后面再次出现不再更新。**

**先看一条具体轨迹。** 令 $\gamma=0.5$，两条 episode 的相关片段为：

```text
Episode 1: s --1--> u --2--> s --4--> terminal
Episode 2: u --0--> s --6--> terminal
```

Episode 1 中 $s$ 出现了两次：第一次在 $t=0$，第二次在 $t=2$。从第一次 $s$ 出发的完整回报是：

$$
G_{1,0}=1+0.5\times2+0.5^2\times4=3.
$$

第二次访问的回报虽然是 $G_{1,2}=4$，但 first-visit MC **不使用它**。Episode 2 中 $s$ 第一次出现在 $t=1$，所以 $G_{2,1}=6$。两条 episode 对 $s$ 提供的样本就是 $3$ 和 $6$，估计值为：

$$
\hat V_{\mathrm{FV}}^\pi(s)=\frac{3+6}{2}=4.5.
$$

现在把刚才的“第一次出现”写成一般记号。令第 $i$ 条 episode 中状态 $s$ 的首次出现时间为：

$$
t_i^{\mathrm{first}}(s)
=
\min\{t:s_{i,t}=s\}.
$$

估计量为：

$$
\hat V_{\mathrm{FV}}^\pi(s)
=
\frac{1}{N_{\mathrm{FV}}(s)}
\sum_{i:s\in\tau_i}
G_{i,t_i^{\mathrm{first}}(s)}.
$$

这里 $N_{\mathrm{FV}}(s)$ 是包含 $s$ 的 episode 数；上面的两条 episode 都包含它，所以 $N_{\mathrm{FV}}(s)=2$。这个估计量总结的正是刚才的操作：按 episode 时间顺序只保留第一个 $s$。

算法需要为每个状态保存访问 episode 数 $N(s)$ 和 return 累计量 $S_G(s)$：

~~~~text
初始化 N(s)=0, S_G(s)=0
重复：
    按 pi 采样一条完整 episode
    计算每个时间步的 G_t
    按时间从前向后扫描状态
    若 s_t 是本 episode 第一次出现：
        N(s_t) += 1
        S_G(s_t) += G_t
        V(s_t) = S_G(s_t) / N(s_t)
~~~~

“First”指 episode 时间顺序中的首次，不是从尾部反向计算 return 时先碰到的位置。

### 4.2 Every-visit MC

*首次完整讲解：Lecture 3 §4.2「Every-visit MC」。*

Every-visit MC（每次访问 MC）让状态的每一次出现都贡献 return：

沿用上面的两条 episode，every-visit 会保留 Episode 1 的两个样本 $3,4$，以及 Episode 2 的样本 $6$，所以：

$$
\hat V_{\mathrm{EV}}^\pi(s)=\frac{3+4+6}{3}=\frac{13}{3}\approx4.33.
$$

把“所有访问都计入平均”写成一般公式：

$$
\hat V_{\mathrm{EV}}^\pi(s)
=
\frac{
\sum_i\sum_{t=0}^{T_i-1}
\mathbf 1\{s_{i,t}=s\}G_{i,t}
}{
\sum_i\sum_{t=0}^{T_i-1}
\mathbf 1\{s_{i,t}=s\}
}.
$$

这正好显示两种算法的区别：first-visit 对每条 episode 最多取一个样本，every-visit 则把同一条 episode 中的每次访问都算进去。

同一 episode 中重复访问产生的 return 彼此相关，且访问次数多的 episode 权重更高；这解释了为什么 every-visit MC 的有限样本统计性质与 first-visit 不同。


### 4.3 课件 Mars Rover：MC 计算与源冲突

课件第 14 页给定 $0\le\gamma<1$ 和轨迹：

~~~~text
(s3, a1, 0, s2, a1, 0, s2, a1, 0, s1, a1, 1, terminal)
~~~~

按 §2.3 的 reward-on-transition 约定，奖励序列是 $[0,0,0,1]$。状态 $s_2$ 出现两次：

$$
G_{\text{first }s_2}=0+\gamma\cdot0+\gamma^2\cdot1=\gamma^2,
$$

$$
G_{\text{second }s_2}=0+\gamma\cdot1=\gamma.
$$

因此：

$$
\hat V_{\mathrm{FV}}(s_2)=\gamma^2,
\qquad
\hat V_{\mathrm{EV}}(s_2)=\frac{\gamma^2+\gamma}{2}.
$$

同一约定下，first-visit 的完整状态向量（$s_1,\ldots,s_7$）应为：

$$
[1,\gamma^2,\gamma^3,0,0,0,0].
$$

> [!warning] 课件内部不一致
> PDF 第 36 页底部印出的 first-visit MC 向量是 $[1,\gamma,\gamma^2,0,0,0,0]$。它与第 14/36 页给出的重复 $s_2$ 轨迹、课件自己的 first-visit 定义以及 reward-on-transition 索引不一致。本文保留课件印刷结果作为来源记录，但计算采用明确轨迹和标准定义；做题时应先写出每个状态对应哪个 $G_t$。

### 4.4 Incremental MC 与一般学习率

*首次完整讲解：Lecture 3 §4.4「Incremental MC 与一般学习率」。*

如果已经有前三个样本的平均值 4（所以它们的总和是 $12$），又来了一个新 return 8，重新计算就是 $(12+8)/4=5$；但算法不应该保存所有旧样本。它只需要知道旧平均值和新样本：

$$
V_4(s)=4+\frac14(8-4)=5.
$$

这里的变化量是“新样本与旧平均的差”乘以 $1/4$。把这个操作写成一般公式，若旧平均为 $V_{n-1}(s)$，第 $n$ 个新样本是 $G_n(s)$：

$$
V_n(s)
=
V_{n-1}(s)
+
\frac1n\left[G_n(s)-V_{n-1}(s)\right].
$$

把 $1/n$ 换成一般学习率（learning rate / step size）$\alpha$，得到贯穿后续 RL 的更新骨架：

$$
V(s_t)
\leftarrow
V(s_t)+\alpha\left[G_t-V(s_t)\right],
$$

$$
\text{new estimate}
=
\text{old estimate}
+
\text{step size}\times
(\text{target}-\text{old estimate}).
$$

$1/N(s)$ 精确复现样本平均；固定 $\alpha$ 不再等价于全部历史样本平均，但能对近期数据保留更大权重。

==增量式更新。它不需要保存以前所有的 Gt​，但可以逐步算出平均值==

![[Pasted image 20260722163456.png]]




### 4.5 Policy-evaluation diagram：同一棵树上的 DP、MC 与 TD(0)

课件第 17--20 页确实有图。第 17--19 页是逐帧展开，第 20 页是包含图例的完整终帧：

![[lec3-policy-evaluation-diagram-p20.png|900]]

*来源：`lecture/lec3/lecture3pre.pdf` 第 20 个物理页面。*

> [!note] 来源边界
> 第 17--20 页原图只负责建立 state、action、expectation 和 terminal 的树结构。下面的 DP / MC / TD(0) 对照是基于 Lecture 2 的 DP 公式、本讲第 21 页的 MC update 以及第 37--40 页的 TD 内容所做的派生整理，并非四页图上逐字列出的说明。

#### 4.5.1 先读懂图中的节点和分支

- **白色圆点是 state node**：根节点是当前状态 $s$，下面的白点是可能到达的下一状态。
- **黑色圆点是 action node**：表示在上一个状态采取的动作。若策略是随机策略，动作由 $\pi(a\mid s)$ 决定；图中每个白色节点只画出了相应的 action node，没有把动作概率写在边上。
- **从黑色 action node 到白色 state node 的分支**由 $P(s'\mid s,a)$ 决定。图中的弧线标记 `Expectation`，表示对这些可能的下一状态按概率加权。
- **方框 $T$ 是 terminal state**：到达它以后没有后续价值；向下的虚线则表示未来仍可继续展开。

这棵树表示的共同目标仍然是：从 $s$ 出发并持续执行策略 $\pi$ 时，未来 return 的条件期望

$$
V^\pi(s)=\mathbb E_\pi[G_t\mid s_t=s].
$$

DP、MC 和 TD(0) 并不是在估计三个不同对象；它们对**同一个未来树**采用了不同的计算方式。

#### 4.5.2 Dynamic programming：展开一层，把所有分支都算进期望

动态规划（dynamic programming, DP）假设环境模型 $P(s'\mid s,a)$ 和奖励模型已知，因此可以枚举当前状态下的全部动作以及每个动作的全部下一状态。一次 iterative policy-evaluation backup 是：

$$
V_{k+1}(s)
=
\sum_a \pi(a\mid s)
\sum_{s'}P(s'\mid s,a)
\left[
R(s,a,s')+\gamma V_k(s')
\right].
$$

这里有两层加权平均：

1. 外层按 $\pi(a\mid s)$ 对策略可能选择的动作求期望；
2. 内层按 $P(s'\mid s,a)$ 对该动作可能到达的下一状态求期望。

DP 并不会真的把无限深的未来树全部展开。它只精确计算第一层所有分支，然后用当前估计 $V_k(s')$ 代表从 $s'$ 开始的剩余未来。因此 DP **不采样，但会 bootstrap**。

#### 4.5.3 Monte Carlo：不枚举分支，只采样一条完整路径

*首次完整讲解：Lecture 3 §4「Monte Carlo policy evaluation」。本节只补充：如何在概率树上理解一条完整 trajectory。*

模型未知时，MC 不能列出整棵树的分支概率。它按照策略和真实环境采样一条完整 trajectory：

$$
s_t,a_t,r_t,s_{t+1},a_{t+1},r_{t+1},\ldots,s_T,
$$

等 episode 结束后计算这条路径的完整 return：

$$
G_t
=
r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots,
$$

再用它更新起点状态：

$$
V(s_t)
\leftarrow
V(s_t)+\alpha\left[G_t-V(s_t)\right].
$$

一次 $G_t$ 只是整棵树中的一条随机路径，不等于 $V^\pi(s_t)$；

==许多 episode 的 return 平均才逐渐逼近树上所有未来的期望。MC 的 target 不含任何 $V(s_{t+1})$，所以它会采样，但不 bootstrap==



#### 4.5.4 TD(0)：只采样一步，再读取自己的下一状态估计

*首次完整讲解：Lecture 3 §4.5「Policy-evaluation diagram：同一棵树上的 DP、MC 与 TD(0)」。*

时序差分学习（temporal-difference learning, TD）结合了 MC 和 DP：像 MC 一样从环境采样 transition，又像 DP 一样用下一状态的当前 value estimate 代表剩余未来。TD(0) 每次只需要观察：

$$
s_t,a_t,r_t,s_{t+1}.
$$

TD(0) 同样不知道完整模型，所以也不能计算所有可能分支。
但它不像 MC 那样一直等到结局，而是只观察一步

MC 要等整条 episode 结束，得到完整回报 Gt 后再更新；TD 走一步就更新，用下一状态当前的估计值来代替后面的真实回报。

MC是：

$$V(S_t) \leftarrow V(S_t) + \alpha \left[ G_t-V(S_t) \right]$$

其中完整回报：

$$G_t = R_{t+1} +\gamma R_{t+2} +\gamma^2R_{t+3} +\cdots$$

MC 的目标值是：Gt

所以必须等 episode 结束后，才能把后续所有奖励算出来。


而TD走一步后立即更新，它构造标量 TD target

$$
y_t^{\mathrm{TD}}
=
r_t+\gamma V(s_{t+1}),
$$

并用 target 与当前预测之差更新：

$$
\delta_t
=
y_t^{\mathrm{TD}}-V(s_t),
$$

$$
V(s_t)
\leftarrow
V(s_t)+\alpha\delta_t.
$$

$V(s_{t+1})$ 是算法自己的当前估计，不是向环境查询到的真值。也正因为它可能还不准确，TD target 在有限数据下会带有 bootstrap 误差；好处是无需等 episode 结束就能更新。若 $s_{t+1}$ 是 terminal，则后续价值为 0，target 只剩实际观察到的 $r_t$。

假设当前处于 St，执行一个动作后：

- 获得即时奖励 Rt+1
- 到达下一状态 St+1

TD 认为当前状态的新目标应该是：

$$\boxed{ R_{t+1}+\gamma V(S_{t+1}) }$$​

意思是：

> 当前状态的价值，大约等于“这一步拿到的奖励”加上“下一状态未来价值的折扣”。

然后拿这个目标和旧估计进行比较：

$$\underbrace{ R_{t+1}+\gamma V(S_{t+1}) }_{\text{TD目标}} - \underbrace{ V(S_t) }_{\text{旧估计}}$$​​

这个差就是 TD error。



![[Pasted image 20260722140219.png]]



#### 4.5.5 用同一个数值例子比较 MC 与 TD(0)

**自拟例子**：当前在 $s_0$，实际执行动作后观察到

$$
s_0\xrightarrow{r_0=2}s_1,
$$

并且当前估计与超参数是

$$
V(s_0)=4,
\qquad
V(s_1)=5,
\qquad
\gamma=0.9,
\qquad
\alpha=0.2.
$$

TD(0) 只使用眼前这一步和 $V(s_1)$：

$$
y_0^{\mathrm{TD}}
=
2+0.9\times5
=
6.5,
$$

$$
\delta_0=6.5-4=2.5,
\qquad
V_{\mathrm{new}}(s_0)
=
4+0.2\times2.5
=
4.5.
$$

MC 此时还不能更新。假设这条 episode 后面真实得到奖励 3、4，然后终止，则完整 return 是

$$
G_0
=
2+0.9\times3+0.9^2\times4
=
7.94,
$$

于是同样使用 $\alpha=0.2$ 时：

$$
V_{\mathrm{new}}(s_0)
=
4+0.2(7.94-4)
=
4.788.
$$

6.5 与 7.94 是这一次更新采用的两个不同 target，不是两种方法最终必然收敛到的两个不同真值。TD 用当前 $V(s_1)$ 换取即时更新；MC 等到结局，用实际观察到的完整未来换掉了这个估计。DP 若模型已知，则会把所有可能动作和下一状态的对应 target 都列出并按概率求平均；只有在第一步确定性到达 $s_1$ 时，它的这一层 backup 才会退化成同样的 $2+0.9V(s_1)$。

![[Pasted image 20260722143515.png]]


![[Pasted image 20260722143808.png]]


TD例子：

假设有：

```
s0 --奖励0--> s1 --奖励10--> terminal
```

取：

$$γ=0.9,α=1$$

一开始：

$$V(s0)=0,V(s1)=0$$

终止状态：

$$V(terminal)=0$$

第一次 episode：

首先发生：

$$s_0\rightarrow s_1,\quad R=0$$

更新 s0​：

$$V(s_0) \leftarrow 0+0.9V(s_1)$$

由于此时：

$$V(s_1)=0$$

所以：

$$V(s_0)=0$$

然后：

$$s_1\rightarrow terminal,\quad R=10$$

更新 s1​：

$$V(s_1) \leftarrow 10+0.9V(terminal)=10$$

第一次 episode 之后：

$$V(s_0)=0,\qquad V(s_1)=10$$

---

第二次 episode

再次经历：

$$s_0\rightarrow s_1$$

这次 V(s1​) 已经变成 10，所以：

$$V(s_0) \leftarrow 0+0.9\times10=9$$

最终：

$$V(s_0)=9,\qquad V(s_1)=10$$

终点奖励就这样一步步向前传播：

$$terminal\rightarrow s_1\rightarrow s_0$$

所以 TD 确实需要不断采样和更新。




#### 4.5.6 最后再看统一对照

| 方法                  | 是否需要真实模型 | 动作 / 下一状态如何处理     | 对剩余未来的处理                            | 何时能更新               |
| ------------------- | -------- | ----------------- | ----------------------------------- | ------------------- |
| Dynamic programming | 是        | 枚举全部动作和下一状态并精确求期望 | 用 $V_k(s')$ bootstrap               | 完成一次 model backup 后 |
| Monte Carlo         | 否        | 沿策略和环境采样一条完整路径    | 使用完整 $G_t$，不 bootstrap              | episode 结束后         |
| TD(0)               | 否        | 只采样一个动作和一个下一状态    | 用 $r_t+\gamma V(s_{t+1})$ bootstrap | 每个 transition 后     |

bootstrap 的特点：

> 用当前暂时不完全准确的估计值互相更新，并通过反复采样逐渐收敛。而mc不会用下一状态的值算这一状态


> [!important] 两条独立的比较轴
> **Expectation vs sampling** 问的是“第一步的分支如何处理”；**完整 return vs bootstrap** 问的是“第一步之后的未来如何处理”。DP 是 expectation + bootstrap，MC 是 sampling + complete return，TD(0) 是 sampling + bootstrap。






## 5. 如何评价一个策略价值估计器

### 5.1 五个评价维度

两个方法都能输出一个 $V(s)$，并不代表它们同样好。我们至少要继续追问：数据越来越多时会不会接近真值？同样的数据量下谁更准？需要保存多少数据、花多少计算？课件把这些问题整理为五个维度：

1. **Consistency**：数据足够多时是否收敛到真实 $V^\pi$？
2. **Computational complexity**：新数据到来后更新需要多少计算？
3. **Memory requirements**：需要保存 value、完整 episode、全部历史数据还是模型计数？
4. **Statistical efficiency**：给定同样数据量，估计有多准确？
5. **Empirical accuracy**：实践中常用 MSE 衡量；这通常需要模拟器真值或高精度基准。

真实交互昂贵时，data efficiency 和 computation efficiency 必须分开：一个方法可以数据高效但每次重规划非常昂贵。

### 5.2 Bias、variance 与 MSE

*首次完整讲解：Lecture 3 §5.2「Bias、variance 与 MSE」。*

先用一个具体比较理解三个词。假设真实值 $\theta=10$，我们反复采集新数据并重新估计：

- 估计器 C 的平均输出是 11，说明它整体偏高 1；不同数据集下的波动 variance 是 1。
- 估计器 D 的平均输出正好是 10，但波动 variance 是 4。

因此 C 的 bias 是 $11-10=1$，MSE 是 $1+1^2=2$；D 虽然无偏，MSE 却是 $4+0^2=4$。这个例子先告诉我们：**无偏不等于有限数据下误差更小。**

现在给出一般定义。设数据 $X$ 由真实参数 $\theta$ 控制，估计器 $\hat\theta(X)$ 是数据的随机函数。

偏差（bias）是重复采集新数据并重新估计后，平均估计与真值的差：

$$
\operatorname{Bias}_\theta(\hat\theta)
=
\mathbb E[\hat\theta]-\theta.
$$

在上面的 C 中，$\mathbb E[\hat\theta]=11$、$\theta=10$，所以 bias 为 1。

方差（variance）描述换一批数据后估计波动多大：

$$
\operatorname{Var}(\hat\theta)
=
\mathbb E
\left[
(\hat\theta-\mathbb E[\hat\theta])^2
\right].
$$

Variance 不关心平均值离真值多远，只关心换一批数据后估计值围绕自身均值波动多大。

均方误差（mean squared error, MSE）把系统偏移和随机波动合并：

$$
\operatorname{MSE}(\hat\theta)
=
\mathbb E[(\hat\theta-\theta)^2]
=
\operatorname{Var}(\hat\theta)
+
\operatorname{Bias}_\theta(\hat\theta)^2.
$$

上面的 C 与 D 正好代入了这个分解：MSE 同时惩罚系统性偏移和随机波动。

也不能用少数几次实验的经验平均碰巧等于真值来证明无偏；无偏性是对所有可能数据集取期望的性质。

### 5.3 Consistency 不等于 unbiasedness

*首次完整讲解：Lecture 3 §5.3「Consistency 不等于 unbiasedness」。*

先区分两个问题：

- unbiasedness 问的是：“固定样本量，重复实验后的平均估计是否等于真值？”
- consistency 问的是：“样本越来越多时，这一次估计是否越来越可能贴近真值？”

为什么两者不同？假设数据 $X_1,X_2,\ldots$ 的均值都是 $\theta$，但无论收集多少数据，估计器都只看第一个样本：

$$
\hat\theta_n=X_1.
$$

它对每个 $n$ 都无偏，因为 $\mathbb E[X_1]=\theta$；但 $n$ 从 10 增加到 10000 时，它仍然只看同一个 $X_1$，误差不会缩小，所以一般不一致。

现在把 consistency 写成一般定义。若 $\hat\theta_n$ 使用 $n$ 个样本，则对任意 $\varepsilon>0$：

$$
\lim_{n\to\infty}
\Pr(|\hat\theta_n-\theta|>\varepsilon)=0.
$$

它表示“数据无限增加后仍离真值超过固定误差”的概率趋于 0。

所以课件问题“unbiased 是否一定 consistent？”答案是 **No**。反过来，有限样本有偏但偏差逐渐消失的估计器也可以是一致的。

### 5.4 MC estimators 的统计性质

在独立 episode、有限期望 return、充分访问等常规条件下：

- **First-visit MC**：对 $V^\pi(s)$ 无偏；当包含 $s$ 的 episode 数趋于无穷时，由大数定律一致收敛。
- **Every-visit MC**：同一 episode 的重复访问相关，有限样本一般有偏；在常规遍历条件下仍一致，而且实践中可能因使用更多访问而有更好的 MSE。
- **Incremental MC**：用 $1/N(s)$ 时与样本平均完全等价；用一般 $\alpha_n(s)$ 时，性质取决于学习率序列。

“Every-visit 使用样本更多”不保证任何有限数据集上都更准；课件只给出一般统计结论和经验倾向。

### 5.5 随机逼近学习率条件

*首次完整讲解：Lecture 3 §5.5「随机逼近学习率条件」。*

学习率不能只说“越来越小”就结束：缩小太快，算法可能还没走到正确位置就几乎不动；一直不缩小，随机 target 又会让估计持续抖动。经典 Robbins--Monro 条件正好控制这两件事。对每个被无限访问的状态 $s$：

$$
\sum_{n=1}^{\infty}\alpha_n(s)=\infty,
\qquad
\sum_{n=1}^{\infty}\alpha_n(s)^2<\infty.
$$

这里 $n$ 是状态 $s$ 的第 $n$ 次更新，不一定是全局时间。

- 第一条保证总移动量不会过早耗尽；
- 第二条保证随机噪声的长期累计影响可被压低。

例如 $\alpha_n=1/n$ 的前几步是 $1,1/2,1/3,1/4,\ldots$：单步更新越来越温和，但所有步长加起来仍没有有限上限；它同时满足两条，因为调和级数发散而平方倒数级数收敛。固定 $\alpha>0$ 不满足第二条，在平稳随机环境中通常在真值附近持续波动；在非平稳环境里，这种不收敛也换来了追踪变化的能力。

### 5.6 MC 的局限

MC 的优点是无需模型、无需 bootstrap、定义直接且在常规条件下一致。主要局限是：

- 完整未来包含许多随机变量，return variance 往往较高；
- 必须等待 episode 结束；
- continuing task 没有自然终点时，基础 episodic MC 不方便直接使用；
- 降低误差可能需要大量真实交互，而机器人、医疗、教育等场景中的试验可能昂贵或危险。

即使模型已知，状态空间巨大而 sampling 便宜时也可能使用 MC；“有模型”不意味着枚举整棵树一定划算。
sampling：采样。让智能体在环境中实际运行，收集状态转移样本

## 6. Temporal-Difference learning

*首次完整讲解：Lecture 3 §4.5「Policy-evaluation diagram：同一棵树上的 DP、MC 与 TD(0)」。本节只补充：正式算法、terminal 边界和统计性质。*

§4.5 已经从概率树和数值例子说明了 TD 如何结合 transition sampling 与 bootstrapping。本节把这个想法写成可执行的 one-step TD 算法，也就是 TD(0)，并分析它在 episodic 与 continuing task 中的使用边界。

### 6.1 从 Bellman equation 到一次 sample

*首次完整讲解：Lecture 2 §7「Policy evaluation」。本节只补充：用一个 transition sample 替代未知期望。*

真实价值满足：

$$
V^\pi(s_t)
=
\mathbb E_\pi
\left[
r_t+\gamma V^\pi(s_{t+1})
\mid s_t
\right].
$$

模型未知时无法显式对所有 $s_{t+1}$ 求和，但可以观察一个实际 $r_t,s_{t+1}$，再用当前估计 $V(s_{t+1})$ 补足未来。

### 6.2 TD target、TD error 与更新

*首次完整讲解：Lecture 3 §4.5「Policy-evaluation diagram：同一棵树上的 DP、MC 与 TD(0)」。本节只补充：固定 target、error 的符号并写成算法。*

TD target 是标量：

$$
y_t^{\mathrm{TD}}
=
r_t+\gamma V(s_{t+1}).
$$

$r_t$ 是真实观察到的一步奖励；$V(s_{t+1})$ 是当前估计，不是 oracle 真值。TD error 是 target 减当前预测：

$$
\delta_t
=
y_t^{\mathrm{TD}}-V(s_t)
=
r_t+\gamma V(s_{t+1})-V(s_t).
$$

更新为：

$$
V(s_t)
\leftarrow
V(s_t)+\alpha\delta_t.
$$

**自拟例子**：观察 $r_t=2$，当前 $V(s_{t+1})=5$、$V(s_t)=4$，$\gamma=0.9$、$\alpha=0.2$：

$$
y_t^{\mathrm{TD}}=2+0.9\times5=6.5,
$$

$$
\delta_t=6.5-4=2.5,
\qquad
V_{\mathrm{new}}(s_t)=4+0.2\times2.5=4.5.
$$

新 value 向 target 移动 20%，没有一步跳到 6.5。

### 6.3 TD(0) algorithm 与 terminal transition

~~~~text
初始化 V(s)
重复：
    在 s_t 按 pi(a | s_t) 采样 a_t
    观察 r_t, s_{t+1}, done
    若 done: target = r_t
    否则:    target = r_t + gamma * V(s_{t+1})
    V(s_t) += alpha * (target - V(s_t))
~~~~

策略评估阶段必须按固定策略 $\pi$ 采样；擅自改成 greedy action 会改变数据分布，评估目标就不再是原来的 $V^\pi$。

进入 terminal state 时保留即时奖励，但终止后的 future value 为 0：

$$
V(s_{\mathrm{terminal}})=0,
\qquad
y_t^{\mathrm{TD}}=r_t.
$$

### 6.4 课件 Mars Rover：一次 TD trace

课件沿用 §4.3 的轨迹，令 $V(s)=0$、$\alpha=1$，按时间顺序在线更新：

~~~~text
s3 --0--> s2 --0--> s2 --0--> s1 --1--> terminal
~~~~

逐步计算：

1. $s_3\to s_2$：target 为 $0+\gamma V(s_2)=0$，所以 $V(s_3)=0$。
2. 第一次 $s_2\to s_2$：target 仍为 0，$V(s_2)=0$。
3. 第二次 $s_2\to s_1$：此时 $V(s_1)$ 还没更新，target 为 0，$V(s_2)=0$。
4. $s_1\to\text{terminal}$：target 为 1，$V(s_1)=1$。

episode 结束后：

$$
[V(s_1),\ldots,V(s_7)]=[1,0,0,0,0,0,0].
$$

这说明 TD 能立即更新不等于远期 reward 会在单条 episode 中立刻传播到所有早期状态。
在线更新顺序下，奖励先到 $s_1$；未来再次访问 $s_2,s_3$ 才会继续向前传播。


==episode（回合）和iteration/update（迭代更新）不是同一个东西==


![[Pasted image 20260722154656.png]]


### 6.5 课件学习率判断题

把更新写成：

$$
V_{\mathrm{new}}(s_t)
=
(1-\alpha)V_{\mathrm{old}}(s_t)
+
\alpha y_t^{\mathrm{TD}}.
$$

因此四个判断为：

1. $\alpha=0$ 更重视 TD target：**False**，此时完全不更新。
2. $\alpha=1$ 会直接把 value 设为 TD target：**True**。
3. 随机下一状态下，$\alpha=1$ 可能永久振荡：**True**，单个随机 target 每次完全覆盖旧值。
4. 存在 $\alpha=1$ 仍能收敛的 deterministic MDP：**True**，确定性链反复访问就是例子。

### 6.6 TD(0) 的性质

TD(0) 是 model-free、sample-based、bootstrapping、online 的 estimator，可用于 episodic 或 continuing task。课件强调：

- early estimates 受初始化和当前 $V$ 影响，因此有限样本通常有偏；
- 单步 target 通常比完整 MC return 方差低；
- 在 Markov representation、充分访问、bounded reward 和适当 learning-rate conditions 下，tabular on-policy TD(0) 是一致的；
- 多步 TD 和 TD($\lambda$) 会在 TD(0) 与 MC 之间插值，但不属于本讲的正式算法内容。

## 7. MC 与 TD(0) 的系统比较

| 维度 | Monte Carlo | TD(0) |
|---|---|---|
| 是否需要模型 | 否 | 否 |
| target | 完整 $G_t$ | $r_t+\gamma V(s_{t+1})$ |
| bootstrap | 否 | 是 |
| 更新时机 | episode 完成后 | 每个 transition 后 |
| continuing task | 基础 episodic MC 不直接适用 | 可以 |
| Markov requirement | 直接 return averaging 不要求 | 正确共享 next-state value 依赖 Markov 表示 |
| finite-sample bias | first-visit 通常无偏 | 通常有偏 |
| variance | 通常较高 | 通常较低 |
| 信息传播 | 完整 return 一次传到整条 trajectory | 按 transition 逐步传播 |

这里的“通常”不能删掉：bias 和 variance 还取决于环境随机性、初始化、样本相关性和学习率。算法选择是在 data、compute、latency 和 representation assumptions 之间权衡，不存在无条件赢家。




## 8. Certainty equivalence：先估模型再做 DP

*首次完整讲解：Lecture 3 §8「Certainty equivalence」。*

Certainty equivalence（确定性等价）是一种 model-based policy evaluation：从数据构造 maximum-likelihood MDP，然后暂时把这个估计模型当作真实模型，用 Lecture 2 的动态规划求 $V^\pi$。

Maximum likelihood estimation（最大似然估计，MLE）选择最能解释已观测数据的模型参数。在 tabular MDP 中，categorical transition(分类转换) 的 MLE 就是 observed relative frequency，expected reward 的 MLE 就是该状态动作对下的 sample mean。

这部分讲的是一种介于 **DP** 和 **从经验学习** 之间的方法：

> 环境真实的 P,R 不知道，但我们先用收集到的数据估计出 $$\hat P,\hat R$$然后暂时把估计模型当成真实模型，再运行之前学过的 DP 策略评价。


对已观察 transition 计数：

$$
N(s,a,s')
=
\sum_{i,t}
\mathbf 1\{s_{i,t}=s,a_{i,t}=a,s_{i,t+1}=s'\},
$$

它表示：

> 在收集到的全部 episode 中，状态 s执行动作 a 后，实际转移到 s′ 的次数。

$$
N(s,a)=\sum_{s'}N(s,a,s').
$$

表示：

> 在所有数据中，状态 s 下执行动作 a的总次数。

当 $N(s,a)>0$：

$$
\hat P(s'\mid s,a)
=
\frac{N(s,a,s')}{N(s,a)},
$$

$$
\hat R(s,a)
=
\frac{1}{N(s,a)}
\sum_{i,t}
\mathbf 1\{s_{i,t}=s,a_{i,t}=a\}r_{i,t}.
$$

再对动作按策略加权得到 $\hat P^\pi,\hat R^\pi$，解经验 Bellman equation：

$$
\hat V^\pi
=
\hat R^\pi+\gamma\hat P^\pi\hat V^\pi.
$$

有限状态下可写为：

$$
\hat V^\pi
=
(I-\gamma\hat P^\pi)^{-1}\hat R^\pi,
$$

但实现时通常解线性系统，不显式求逆。

课件给出的代价与性质：

- analytic matrix solution 约 $O(|\mathcal S|^3)$；
- iterative planning 的一轮通用 backup 约 $O(|\mathcal S|^2|\mathcal A|)$；
- 能利用 Markov model 在状态间传播数据，通常 data efficient；
- 每次新数据后重新建模和规划可能 computationally expensive；
- 对正确的 Markov model，在充分 coverage 下是一致的；
- 可用于 off-policy evaluation，但目标策略会使用的 $(s,a)$ 必须有数据支持。

未访问的 $(s,a)$ 没有可靠 MLE；需要初始化、先验或额外探索。状态表示非 Markov 时，经验模型还可能错误合并不同历史。


## 9. Batch policy evaluation 与 AB example

*首次完整讲解：Lecture 3 §9「Batch policy evaluation 与 AB example」。*

Batch policy evaluation（批量策略评估）固定一组有限数据，不再收集新 experience。课件设有 $K$ 条 episode，反复从这组数据采样并应用 MC 或 TD(0)，问它们无限重放后分别收敛到什么。

下面的例子告诉我们即使 MC 和 TD 使用完全相同的数据，它们最后对 V(A) 的估计也可能不同

### 9.1 AB dataset

令 $\gamma=1$，共有 8 条 episode：

~~~~text
A --0--> B --0--> terminal        1 条
B --1--> terminal                 6 条
B --0--> terminal                 1 条
~~~~

$B$ 总共出现 8 次：6 个 return 是 1，2 个是 0，因此 MC 和 TD 都得到：

$$
V(B)=\frac{6}{8}=0.75.
$$

### 9.2 为什么 $V(A)$ 不同

Batch MC 只看到一次从 $A$ 出发的完整 return 0：

$$
V_{\mathrm{MC}}(A)=0.
$$

经验 Markov model 却认为 $A$ 的即时 reward 为 0，并且必然转移到 $B$：

$$
\hat R(A)=0,
\qquad
\hat P(B\mid A)=1.
$$

其 Bellman equation 给出：

$$
V_{\mathrm{TD}}(A)
=
0+1\times V(B)
=
0.75.
$$

MC 的解释是：“唯一一次从 $A$ 出发观察到的 return 是 0。”

TD 的解释是：“只要到达 $B$，未来应与如何到达无关；所有 $B$ 数据都应该共享给 $A$。”后者正是 Markov assumption 带来的泛化。

有限数据下不能只凭这组日志断言哪一个更接近真实 MDP。如果 $B$ 隐藏了不同历史，TD 的共享可能产生 model misspecification error。

Markov property(马尔科夫性质) 的核心是：

> **未来只依赖当前状态，不需要知道更早的历史。**


![[Pasted image 20260723005715.png]]

TD认为，应该把中间那个episode算出来的V(B)，也要给第一个episode的B才行



### 9.3 两者的收敛对象

- **Batch MC**：收敛到数据中 observed complete returns 的经验平均，也就是这些 return targets 的 least-squares solution。
- **Batch TD(0)**：收敛到 maximum-likelihood Markov model 的 certainty-equivalence solution，也就是经验 Bellman fixed point。

> [!note] PDF 边界
> 当前 Lec3 PDF 的第 50--51 个物理页面只有 “Batch MC and TD: Convergence” 标题，没有可见正文。上面的标准结论来自该页意图所指的 Sutton & Barto AB example，并与本地 Lecture 4 post deck 对同一内容的明确讲解一致；它不是从空白页面臆造出的课件原文。

### 9.4 四类方法统一比较

| 方法 | 真实模型 | Sampling | Bootstrap | 主要目标 | 主要代价 |
|---|---:|---:|---:|---|---|
| Dynamic programming | 已知 | 否 | 是 | 真实模型中的 $V^\pi$ | 枚举与 planning |
| Monte Carlo | 未知 | 完整 episode | 否 | return sample mean | 高 variance、等待终止 |
| TD(0) | 未知 | 一步 | 是 | online 趋向 $V^\pi$；batch 为经验 Bellman 解 | finite-sample bias、依赖 Markov 表示 |
| Certainty equivalence | 从数据估计 | 用数据建模 | planning 阶段是 | MLE MDP 中的 $V^\pi$ | model storage 与重复 planning |

## 10. 与 LLM 和具身智能的现代连接

本节不是课件原内容。它把 Lecture 3 的“完整 return vs one-step bootstrap”“on-policy vs off-policy data”和“value estimation”连接到近期工作。

### 10.1 LLM：从最终奖励到过程 value

**概念类比**：把一段 LLM reasoning 或一次 tool-use task 看成 trajectory，token / tool call 看成 action，最终正确性或任务成功看成 terminal reward：

- MC 风格会等回答完成，再用最终 outcome 评价早期决策；
- TD 风格会学习中间 prefix 的 value，用相邻步骤之间的 temporal consistency 提供更密集的训练信号；
- 如果 prefix representation 不是充分的 Markov state，bootstrap 仍可能传播系统性误差。

**直接现代应用**：Zhang et al. 的 [TDRM: Smooth Reward Models with Temporal Difference for LLM RL and Inference](https://arxiv.org/abs/2509.15110)（2025）把 TD consistency 用于训练 LLM process reward models。论文报告其 TD-trained PRM 用于 Best-of-N、tree search 和 RL with verifiable rewards；这说明 Lecture 3 的 TD 思想可以迁移到 reasoning-step value / reward estimation，但该方法使用神经网络、过程监督和 LLM 数据，不等同于 tabular TD(0)。

Lecture 5 的 REINFORCE 则直接继承 MC：用 sampled $G_t$ 估计 $Q^{\pi_\theta}(s_t,a_t)$。Critic / baseline 再把 value estimation 引入 policy gradient，以降低 variance。

### 10.2 具身智能：value learning 与 planning policy mismatch

**直接现代应用**：Lin et al. 的 [TD-M(PC)^2: Improving Temporal Difference MPC Through Policy Constraint](https://proceedings.mlr.press/v331/lin26a.html)（L4DC / PMLR 2026）研究 plan-based model-based RL 中的 value overestimation。论文把问题归因于 model-based planner 产生的 exploration data 与 value function 所评估的 exploitation policy 之间的结构性 mismatch，并用 soft-constrained policy update 缓解 out-of-distribution value queries；实验包含 61-DoF humanoid control。

它与本讲的连接是：

- certainty equivalence / model-based planning 可以提高 data reuse，但依赖 model 和 coverage；
- TD value learning 不是脱离数据分布工作的，behavior policy 与 evaluation policy mismatch 会影响 target；
- 高维机器人中的 function approximation、off-policy data 和 planning 远超 tabular Lecture 3，需要额外稳定化约束。

### 10.3 Batch logs 与 agent / robot data

**概念类比**：历史 robot trajectories、推荐日志或 agent tool traces 都是固定 batch。Batch MC 只拟合已观察 outcome，batch TD 会借助 state representation 的 Markov 假设在不同 trajectories 之间共享 next-state information。若日志缺少目标策略动作或 state representation 丢失历史信息，TD 的结构性共享也可能更自信地传播错误。

## 11. Assignment Readiness

### 11.1 已覆盖的前置知识

- Assignment 1 的 VI / PI 前置知识已在 Lecture 1--2 覆盖；Lecture 3 补充 simulation-based evaluation 与 model-based DP 的区别。
- Assignment 2 §2.1 的 $G_t$、反向 $O(T)$ return computation 和 REINFORCE 的 MC target 已有基础。
- Assignment 2 §2.2 的 baseline regression 可以理解为用 sampled $G_t$ 监督一个 state-value predictor；MSE、bias 和 variance 已覆盖。
- Lec4/Lec5 已继续覆盖 DQN、policy gradient theorem、REINFORCE、baseline 和 actor-critic，因此当前课程总体状态允许开始 Assignment 2 的基础 PG / baseline 实现。

反向计算 returns：

~~~~text
G = 0
从 episode 最后一个 reward 向前：
    G = reward + gamma * G
    保存当前 G
~~~~

它在一条长度为 $T$ 的 episode 上是 $O(T)$；必须在 episode 边界重置 $G=0$。

### 11.2 尚未由 Lec3 单独覆盖的内容

Lec3 本身不推导 policy gradient theorem、PPO clipping、old-policy probability ratio 或 policy-induced distribution proofs。这些来自 Lec5 及后续 lecture，不能因为理解 $G_t$ 就把整份 Assignment 2 视为 Lec3 已教完。

### 11.3 Mastery evidence

当前没有记录针对 Lec3 的独立 quiz、推导或实现结果。Coverage complete 只代表笔记覆盖完整，不代表 MC / TD / batch evaluation 已经掌握。

推荐验证顺序：先独立完成 §14 的第 4、7、9 题，再写一个最小数组程序比较 first-visit、every-visit 和 TD(0) 的单 episode update。

## 12. 本讲必会公式

1. Return 与 value（首次讲解：Lecture 1 §15.2--15.3；本讲记号：§2.3）：

   $$
   G_{i,t}=\sum_{k=0}^{T_i-1-t}\gamma^k r_{i,t+k},
   \qquad
   V^\pi(s)=\mathbb E_\pi[G_t\mid s_t=s].
   $$

2. MC sample mean（首次讲解：§4）：

   $$
   \hat V^\pi(s)=\frac1{N(s)}\sum_{j=1}^{N(s)}G_j(s).
   $$

3. Incremental mean（首次讲解：§4.4）：

   $$
   V_n(s)=V_{n-1}(s)+\frac1n[G_n(s)-V_{n-1}(s)].
   $$

4. MSE decomposition（首次讲解：§5.2）：

   $$
   \operatorname{MSE}(\hat\theta)
   =
   \operatorname{Var}(\hat\theta)
   +
   \operatorname{Bias}_\theta(\hat\theta)^2.
   $$

5. Consistency（首次讲解：§5.3）：

   $$
   \lim_{n\to\infty}\Pr(|\hat\theta_n-\theta|>\varepsilon)=0.
   $$

6. Learning-rate conditions（首次讲解：§5.5）：

   $$
   \sum_n\alpha_n(s)=\infty,
   \qquad
   \sum_n\alpha_n(s)^2<\infty.
   $$

7. TD target、error 与 update（首次讲解：§4.5；算法化展开：§6.2）：

   $$
   y_t^{\mathrm{TD}}=r_t+\gamma V(s_{t+1}),
   $$

   $$
   \delta_t=y_t^{\mathrm{TD}}-V(s_t),
   $$

   $$
   V(s_t)\leftarrow V(s_t)+\alpha\delta_t.
   $$

8. MLE model（首次讲解：§8）：

   $$
   \hat P(s'\mid s,a)=\frac{N(s,a,s')}{N(s,a)},
   $$

   $$
   \hat R(s,a)
   =
   \frac{1}{N(s,a)}
   \sum_{i,t}\mathbf 1\{s_{i,t}=s,a_{i,t}=a\}r_{i,t}.
   $$

## 13. 容易混淆点

1. **Return sample vs value**：$G_t$ 是一次随机结果；$V^\pi(s)$ 是条件期望。
2. **First-visit 的方向**：按 episode 从前往后的首次，不是反向算 return 时第一次遇到。
3. **MC target vs TD target**：MC 用完整 $G_t$；TD(0) 用 $r_t+\gamma V(s_{t+1})$。
4. **Model-free vs no dynamics**：model-free 表示不读取显式 $P,R$，不是环境没有 dynamics。
5. **Bootstrap target vs truth**：$V(s_{t+1})$ 是当前估计，会把误差带进 target。
6. **Unbiased vs consistent**：一个描述固定 $n$ 的期望，一个描述 $n\to\infty$ 的概率极限。
7. **Terminal value vs terminal reward**：终止后 future value 为 0，进入终止状态的即时 reward 仍保留。
8. **Online vs on-policy**：online 描述何时更新；on-policy 描述数据策略是否就是 evaluation target。
9. **Batch MC vs batch TD**：前者拟合 observed returns，后者拟合 empirical Markov Bellman equations。
10. **Course Mars Rover vector**：第 36 页印刷向量与轨迹和 first-visit 定义冲突，按 §4.3 的逐 transition 对齐计算。

## 14. 自测题

### 题目

1. 为什么 value iteration 可能超过 $|\mathcal S||\mathcal A|$ 轮？
2. “Model-free”是否表示算法不使用 environment transitions？
3. First-visit 与 every-visit MC 对重复状态如何计样本？
4. 对轨迹奖励 $[0,2,5]$、$\gamma=0.5$，计算 $G_0,G_1,G_2$。
5. 为什么 first-visit MC 可以无偏而 every-visit MC 有限样本一般有偏？
6. 给出一个 unbiased 但 inconsistent estimator。
7. 观察 $r_t=1$、$V(s_{t+1})=4$、$V(s_t)=2$、$\gamma=0.5$、$\alpha=0.25$，计算 TD target、error 和新 value。
8. 为什么 terminal transition 不能直接把整个 TD target 写成 0？
9. AB example 中为什么 $V_{\mathrm{MC}}(A)=0$，而 $V_{\mathrm{TD}}(A)=0.75$？
10. LLM agent 只有最终 task-success reward 时，MC 和 TD 风格的 credit assignment 各会怎么做？

<details>
<summary>查看答案</summary>

1. $|\mathcal S||\mathcal A|$ 影响一轮更新规模，不是轮数上界；轮数取决于 $\gamma$、初值和目标精度。
2. 不是。它仍使用采样到的 transitions，只是不需要显式知道或查询完整 $P,R$。
3. First-visit 每条 episode 每状态最多一个 return；every-visit 使用该状态的每次出现。
4. $G_2=5$，$G_1=2+0.5\times5=4.5$，$G_0=0+0.5\times2+0.5^2\times5=2.25$。
5. First-visit 可把每条独立 episode 对状态的首次 return 视作同分布样本；every-visit 在同一 episode 内使用相关 returns，并按访问次数重新加权 episode。
6. $\hat\theta_n=X_1$；若 $\mathbb E[X_1]=\theta$ 则无偏，但增加数据不降低方差，所以一般不一致。
7. $y=1+0.5\times4=3$，$\delta=3-2=1$，新 value 为 $2+0.25\times1=2.25$。
8. 置零的是终止后的 future value；进入终止状态时实际观察到的 $r_t$ 必须保留，所以 target 是 $r_t$。
9. MC 只拟合从 $A$ 唯一观察到的完整 return 0；TD 利用 $A\to B$ 的经验转移和全部 $B$ 数据，解经验 Bellman equation 得到 0.75。
10. MC 等整个 trajectory 完成，用最终 outcome 给早期动作分配 return；TD 学习 intermediate prefix / state value，用一步结果和下一 prefix value bootstrap。后者更及时，但会依赖 representation 和 value estimate 的准确性。

</details>

## 15. 本讲小结

Lecture 3 把 policy evaluation 从已知模型推进到 direct experience：

1. MC 等完整 episode，用完整 return 的 sample mean 估计 $V^\pi$。
2. First-visit 和 every-visit 的差异在于一条 episode 内重复状态如何贡献样本。
3. Incremental update 建立了“旧估计 + 学习率 $\times$ error”的通用骨架。
4. Bias、variance、MSE 和 consistency 回答的是不同统计问题。
5. TD(0) 用一步真实奖励和下一状态当前估计构造 target，因此能 online bootstrap。
6. MC 通常高 variance、低 bootstrap bias；TD 通常低 variance但有限样本有 bias。
7. Certainty equivalence 先拟合 MLE MDP，再做动态规划；它 data efficient，但依赖 Markov model、coverage 和 planning compute。
8. 固定 batch 上，MC 拟合 observed returns，TD 拟合 empirical Bellman fixed point；AB example 把这个差异压缩成 $V(A)=0$ 与 $0.75$。

## 16. 延伸阅读

### 经典基础

- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed., Chapters 5--6：first/every-visit MC、TD(0) 和 AB example 的标准教材来源。<http://incompleteideas.net/book/the-book-2nd.html>
- Sutton (1988), [Learning to Predict by the Methods of Temporal Differences](https://link.springer.com/article/10.1007/BF00115009)：TD prediction 的奠基论文。
- Tsitsiklis & Van Roy (1997), [An Analysis of Temporal-Difference Learning with Function Approximation](https://web.mit.edu/~jnt/www/Papers/J063-97-bvr-td.pdf)：说明从 tabular TD 推广到函数逼近需要额外条件。

### 前沿动态

截至 2026-07-19 核实：

- Zhang et al. (2025), [TDRM: Smooth Reward Models with Temporal Difference for LLM RL and Inference](https://arxiv.org/abs/2509.15110)：把 temporal-difference consistency 用于 LLM process reward model，连接 reasoning-step evaluation、inference-time search 与 RL training。
- Lin et al. (2026), [TD-M(PC)^2: Improving Temporal Difference MPC Through Policy Constraint](https://proceedings.mlr.press/v331/lin26a.html)：研究 continuous-control planning data 与 value-evaluation policy mismatch 引起的 overestimation，并在包括 61-DoF humanoid 的任务上评估约束方法。
