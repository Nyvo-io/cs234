# CS234 Lecture 4 Notes: Model-Free Control, Function Approximation, and DQN

来源：`lecture/lecture4post.pdf`，CS234 Winter 2026，Emma Brunskill。本文以 post deck 为准，共 89 个物理 PDF 页面；第 80--89 页是定理与 Mars Rover 算法的可选复习。

笔记规范：cs234-rl-tutor v2。覆盖清单只表示内容已经写入，不表示学习者已经掌握。

外部资料核验日期：2026-07-18（仅用于延伸阅读；本讲不额外扩展当期前沿方法）。

## 0. 本讲覆盖清单

- [x] 第 1--4 页：课程定位、lecture structure 与学习主线。
- [x] 第 5--15 页：batch MC、batch TD、AB example、certainty equivalence 与评估指标；作为 Lecture 3 的回顾，不重复完整教学。
- [x] 第 16--21 页：model-free policy iteration、探索与利用、epsilon-greedy 和单调改进。
- [x] 第 22--31 页：tabular Monte Carlo control、online improvement、Mars Rover 与 GLIE。
- [x] 第 32--38 页：TD control、on-policy/off-policy、Q-learning、SARSA 与 tabular 收敛条件。
- [x] 第 39--54 页：函数逼近动机、oracle regression、SGD、MC VFA 与 semi-gradient TD(0)。
- [x] 第 55--60 页：函数逼近 control、MC/SARSA/Q-learning targets 与 deadly triad。
- [x] 第 61--77 页：DQN、experience replay、fixed Q-targets、训练流程、Atari 网络与消融结果。
- [x] 第 78--79 页：本讲要求与下一讲 policy gradients。
- [x] 第 80--89 页：epsilon-greedy 定理证明、Mars Rover MC/SARSA/Q-learning 解答和理解检查。

**视觉材料决策**：第 64 页 replay-buffer 数据流、第 72 页 DQN 网络和第 76 页消融表直接嵌入；第 18 页 agent--world loop 与 Lecture 1 重复，仅用文字承接；纯公式、伪代码和逐帧重复页改写为可搜索的 Markdown。

## 1. 本讲主线

Lecture 3 解决固定策略的 policy evaluation：给定策略 $\pi$，从经验估计 $V^\pi$ 或 $Q^\pi$。Lecture 4 把目标推进到 control：策略不再固定，算法要一边学习动作价值，一边改变自己将来采样数据的方式。

整讲可以压缩成一条递进链：

1. **Tabular control**：学习 $Q(s,a)$，用 epsilon-greedy 保留探索，再用 MC 或 TD target 更新。
2. **Q-learning vs SARSA**：两者共享同一个更新骨架，但下一动作来自不同目标策略，因此分别是 off-policy 与 on-policy。
3. **Value function approximation**：用共享参数 $w$ 表示许多状态动作对，获得泛化能力，也引入拟合误差和相互干扰。
4. **Deadly triad**：function approximation、bootstrapping、off-policy learning 同时出现时，tabular 收敛直觉可能失效。
5. **DQN**：用 replay buffer 降低样本相关性，用 target network 减慢 target 漂移，再用神经网络从像素输出每个动作的 $Q$ 值。

贯穿全讲的统一更新骨架是：

$$
\text{new estimate}
=
\text{old estimate}
+
\text{step size}\times
(\text{target}-\text{prediction}).
$$

算法之间最值得比较的不是名字，而是四件事：**target 从哪里来、数据由谁采样、是否 bootstrap、价值如何表示**。

## 2. Lecture 3 的收尾与统一记号

### 2.1 从状态价值转向动作价值

*首次完整讲解：Lecture 1 §12「评估与控制（Evaluation vs Control）」和 Lecture 2 §13「动作价值函数」。本节只补充：未知模型时为什么 control 必须学习 $Q$。*

本文约定在 $s_t$ 采取 $a_t$ 后观察奖励 $r_t$ 和下一状态 $s_{t+1}$。动作价值是标量函数：

$$
Q^\pi(s,a)
=
\mathbb E_\pi[G_t\mid s_t=s,a_t=a].
$$
意思是：

> 在状态 s 先执行动作 a，之后按照策略 π 继续走，最终完整回报 G​ 的期望。

其中完整回报：

$G_t = R_{t+1} +\gamma R_{t+2} +\gamma^2R_{t+3} +\cdots$

MC 不知道这个期望的准确值，所以它会多次遇到同一个 (s,a)，把得到的多个完整回报平均起来：

$$Q(s,a)\approx \frac{G^{(1)}+G^{(2)}+\cdots+G^{(N)}}{N}$$
还有一种增量写法



而

$$
V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a).
$$

只有 $V(s)$ 时，未知模型下无法直接比较“此刻改选另一个动作会怎样”；有了 $Q(s,a)$，policy improvement 可以直接在动作维度比较，不需要先查询 $P(s'\mid s,a)$。

### 2.2 Batch MC 与 batch TD 只作回引

*首次完整讲解：Lecture 3 §9「Batch MC 与 Batch TD」。本节只补充：Lecture 4 第 5--15 页的课程回顾位置。*

课件再次使用 AB data：$\gamma=1$，共有 8 条 episode，状态 $B$ 的成功 return 出现 6 次，失败 return 出现 2 次；从 $A$ 开始的唯一 episode 是 $A,0,B,0$。两种 batch 方法得到：

| 方法 | $V(B)$ | $V(A)$ | 拟合对象 |
|---|---:|---:|---|
| Batch MC | $0.75$ | $0$ | 已观察 complete returns 的经验均方误差解 |
| Batch TD(0) | $0.75$ | $0.75$ | 经验 maximum-likelihood Markov model 的 Bellman fixed point |

差异来自结构假设：MC 尊重从 $A$ 观察到的唯一完整 return 0；TD 利用 $A\to B$ 与所有 $B$ 数据共享未来信息。Lecture 4 用这一回顾提醒我们，算法比较必须同时看 MSE、bias/variance、data efficiency、computational efficiency，以及状态表示是否真的满足 Markov assumption。

## 3. Tabular Monte Carlo Control

### 3.1 为什么 control 必须探索

*首次完整讲解：Lecture 4 §3.1「为什么 control 必须探索」。*

无模型控制（model-free control）保留 policy iteration 的两个动作：估计当前策略的 $Q^\pi$，再根据 $Q^\pi$ 改进策略。但经验只来自实际执行过的动作；确定性策略若在 $s$ 永远选择 $a_1$，就没有 $a_2$ 的 return sample。

这形成探索--利用权衡（exploration--exploitation trade-off）：

- 探索（exploration）选择不确定或当前估值较低的动作，以获得新信息；
- **利用（exploitation）选择当前估值最高的动作，以获得眼前看起来最好的回报。

**自拟例子**：当前只试过动作 $a_1$，经验均值为 2；动作 $a_2$ 从未尝试。若算法把未访问项初始化为 0 并永远 greedy，它会持续选择 $a_1$。即使 $a_2$ 的真实期望回报是 10，算法也没有数据发现这一点。探索不是附加功能，而是 control 能否识别更优动作的必要信息机制。

![[Pasted image 20260724103826.png]]
假设在状态 s 中，当前策略永远选择 a1：

$\pi(s)=a_1$

那么智能体收集到的经验永远类似于：$(s,a_1,r,s')$

而不会出现：$(s,a_2,r,s')$

于是无模型方法只能更新：$Q(s,a_1)$

却完全没有数据更新：$Q(s,a_2)$

---

这会造成什么问题？

假设真实情况是：$Q(s,a_1)=3,\qquad Q(s,a_2)=10$
实际上 a2才是更好的动作。

但是智能体一开始不知道。如果它一直选择 a1，它可能逐渐学到：$Q(s,a_1)\approx 3$

但由于从未尝试 a2​，它对 a2​ 的价值可能仍然保持初始值，比如：$Q(s,a_2)=0$

此时做贪心策略改进：$\pi'(s)=\arg\max_a Q(s,a)$

智能体看到的是：$Q(s,a_1)=3 > Q(s,a_2)=0$

于是它继续选择 a1。

这就形成了死循环：

$\text{没选 }a_2 \Rightarrow \text{没有 }a_2\text{ 的经验} \Rightarrow \text{不知道 }a_2\text{ 更好} \Rightarrow \text{继续不选 }a_2$

这就是为什么无模型强化学习必须考虑**探索**。


---

DP 为什么没有这个问题？

因为 DP 已知环境模型：

$P(s'|s,a),\qquad R(s,a)$

即使当前策略没有实际执行 a2，DP 也可以直接计算：

$Q^\pi(s,a_2) = R(s,a_2) + \gamma \sum_{s'} P(s'|s,a_2)V^\pi(s')$

也就是说，DP 可以做一种“假设性计算”：

> 假如我现在选择 a2​，平均会到达哪些状态？获得多少奖励？后面继续按照当前策略行动，最终价值是多少？

DP 不必真的在环境里执行 a2​，因为它有完整的环境说明书。

而 MC、TD 没有说明书，只能亲自试。

---

所以要在“利用”和“探索”之间平衡

智能体有两个目标：

利用 exploitation:

选择目前看来价值最大的动作：$a=\arg\max_a Q(s,a)$

这样能够利用当前已经学到的知识，尽可能获得高奖励。


探索 exploration:

偶尔选择目前看起来不是最优的动作，从而获得新经验。

这样才能发现：

> 现在看起来不好的动作，会不会实际上更好？

只利用、不探索，容易被当前不准确的估计困住。

只探索、不利用，又会一直乱试，无法稳定获得高奖励。

ε\-greedy 就是最常用的平衡方法


**核心：**
DP 有模型，可以直接计算没试过的动作；MC 和 TD 没有模型，只能通过实际尝试动作来估计价值，因此在做策略改进时必须加入探索。



### 3.2 Epsilon-greedy：把探索写成动作分布

*首次完整讲解：Lecture 4 §3.2「Epsilon-greedy：把探索写成动作分布」。*

Epsilon-greedy（$\varepsilon$-greedy,  $\varepsilon$-贪心）是由当前动作价值估计生成的随机策略。先假设唯一贪心动作是

$$
g(s)=\arg\max_a Q(s,a).
$$
这里的“贪心”不是选**即时奖励**最大的动作，而是选当前估计的**长期动作**价值 $Q(s,a)$ 最大的动作。即时奖励只是 $Q$ 的一部分；例如一个动作现在给 $1$ 但会进入高价值状态，另一个现在给 $3$ 但立刻终止时，前者仍可能有更大的 $Q$。


课件采用：

$$
\pi_\varepsilon(a\mid s)
=
\begin{cases}
1-\varepsilon+\dfrac{\varepsilon}{|\mathcal A|}, & a=g(s),\\[6pt]
\dfrac{\varepsilon}{|\mathcal A|}, & a\ne g(s).
\end{cases}
$$

它等价于：以 $1-\varepsilon$ 的概率选择贪心动作，以 $\varepsilon$ 的概率从全部动作中均匀抽一个。因此贪心动作也会得到随机分支中的 $\varepsilon/|\mathcal A|$。

**进入“均匀抽动作”这一步之后**，每个动作被抽中的条件概率确实是：

$$\frac{1}{|\mathcal A|}$$​

但别忘了：算法只有以 ε 的概率才会进入这个随机探索分支。



**自拟例子**：$Q(s,\cdot)=[2,5,1,4]$，动作数为 4，$\varepsilon=0.2$。唯一贪心动作概率是

$$
1-0.2+\frac{0.2}{4}=0.85,
$$

其余动作各为 $0.05$，总和是 1。若多个动作并列最大，必须明确 tie-breaking 或如何在贪心集合中分配 $1-\varepsilon$；上面的唯一贪心公式不能无说明地直接套用。




### 3.3 Epsilon-greedy policy improvement

*首次完整讲解：Lecture 4 §3.3「Epsilon-greedy policy improvement」。*

设 $\pi_i$ 已经是 $\varepsilon$-greedy，$\pi_{i+1}$ 是相对于真实 $Q^{\pi_i}$ 新生成的 $\varepsilon$-greedy 策略。对固定状态 $s$：

$$
\sum_a\pi_{i+1}(a\mid s)Q^{\pi_i}(s,a)
=
\frac{\varepsilon}{|\mathcal A|}
\sum_aQ^{\pi_i}(s,a)
+
(1-\varepsilon)\max_aQ^{\pi_i}(s,a).
$$

表示：

> 在状态 s，按照新策略 $\pi_{i+1}$ 随机选择一次动作，也就是a。 然后后续仍然按照旧策略 $\pi_i$ 行动，得到的期望回报。


> **MC、TD 负责估计 Q(得到经验以后，如何更新价值估计？)；ε-greedy 负责根据估计出的 Q 改进策略(收集经验时，应该怎样选择动作？)。**

可以把它们分成两部分：

$$\boxed{\text{策略评价：MC / TD 估计 }Q}$$$$\boxed{\text{策略改进：用 }\varepsilon\text{-greedy 更新策略}}$$


式子左边：
	不是说智能体在状态 s 里把所有动作都实际做一遍。

它表示： 


$Q^{\pi_i}(s,a)$表示：

> 在状态 s 先执行动作 a，之后继续按照旧策略 $\pi_i​$行动，能够获得的期望回报。

所以它已经考虑了执行动作后到达下一个状态，以及后面继续行动：


$Q^{\pi_i}(s,a) = \mathbb E\left[ R_{t+1} +\gamma R_{t+2} +\gamma^2R_{t+3}+\cdots \mid S_t=s,A_t=a,\text{之后遵循 }\pi_i \right]$


$\pi_{i+1}(a\mid s)$：

表示新策略 $\pi_{i+1}$ 在状态 s 选择动作 a 的概率。

因此：$\pi_{i+1}(a\mid s)Q^{\pi_i}(s,a)$

就是“选中动作 a 的概率 × 该动作的价值”。

把所有动作加起来：$\sum_a \pi_{i+1}(a\mid s)Q^{\pi_i}(s,a)a$

就是一个加权平均，也叫期望。

它不是说真的把所有动作都执行一遍，而是：

> 当前只会选择一个动作，但由于选择具有随机性，所以计算平均而言能获得多少价值。



假设在状态 s 有三个动作：

a1,a2,a3

新策略的动作概率是：

$$\pi_{i+1}(a_1|s)=0.8   \pi_{i+1}(a_2|s)=0.1   \pi_{i+1}(a_3|s)=0.1$$

对应动作价值是：

$$Q^{\pi_i}(s,a_1)=10 Q^{\pi_i}(s,a_2)=4 Q^{\pi_i}(s,a_3)=1$$

那么：

$$\sum_a\pi_{i+1}(a|s)Q^{\pi_i}(s,a)a$$

就是：

$$0.8\times10+0.1\times4+0.1\times1=8.5$$

这不是说一次 episode 里依次执行了三个动作，而是说：

- 80% 的运行里选 a1​
- 10% 的运行里选 a2​
- 10% 的运行里选 a3​

大量重复以后，平均回报是 8.5。


并且做完动作a后要继续。并且后续已经包含在 $$Q^{\pi_i}(s,a)$$里面了。


动作价值定义为：

$$Q^{\pi_i}(s,a) = \mathbb E\left[ R_{t+1} +\gamma R_{t+2} +\gamma^2R_{t+3} +\cdots \mid S_t=s,A_t=a \right]$$


它表示：

1. 当前在状态 s；
2. 当前先执行指定动作 a；
3. 到达下一个状态，比如 s1；
4. 从 s1开始，按照策略 πi​ 继续行动；
5. 把未来所有奖励都算进去。

因此，$Q^{\pi_i}(s,a)$ 绝对不只是“执行动作 a 得到的即时奖励”。



![[Pasted image 20260724113634.png]]

![[Pasted image 20260724113711.png]]

两部分相加

于是：

$$\sum_a\pi_{i+1}(a\mid s)Q^{\pi_i}(s,a) = \frac{\varepsilon}{|\mathcal A|} \sum_aQ^{\pi_i}(s,a) + (1-\varepsilon)\max_aQ^{\pi_i}(s,a)$$

可以直观理解为：

$\boxed{ \text{新策略的平均价值} = \text{探索概率}\times\text{所有动作平均价值} + \text{利用概率}\times\text{最佳动作价值} }$


==特别注意：==
这不是 $V^{\pi_{i+1}}(s)$

这里计算的是：$\sum_a\pi_{i+1}(a\mid s)Q^{\pi_i}(s,a)$

它的含义是：

> 当前这一步按照新策略 $\pi_{i+1}$选动作，但 Q 中的后续行为仍然按照旧策略 πi。

可以看成只在当前状态先改善一步：当前一步用新策略，后续暂时用旧策略


而真正的：$V^{\pi_{i+1}}(s)$

表示从现在开始，当前和以后所有步骤都按照新策略 $\pi_{i+1}$行动。

这个式子主要用来证明策略改进：

$\sum_a\pi_{i+1}(a\mid s)Q^{\pi_i}(s,a) \geq V^{\pi_i}(s)$

也就是新生成的 ε-greedy 策略，在旧 Q 的评价标准下，至少不会比旧策略差。然后通过策略改进定理进一步得到：$V^{\pi_{i+1}}(s)\geq V^{\pi_i}(s)$



---

==因此，这一长串公式的核心作用就是证明：==


$\boxed{\text{根据 }Q^{\pi_i}\text{ 生成新的 }\varepsilon\text{-greedy 策略，策略会逐渐变好。}}$


旧策略的价值是：

$$V^{\pi_i}(s) = \sum_a\pi_i(a|s)Q^{\pi_i}(s,a)$$

这是旧策略对各个动作价值的加权平均。（Q的意思就是现在先按照a走，但后续都按照pi继续走，所以不是只进行a运动就停止的)

新策略则保留固定的探索概率：$\frac{\varepsilon}{|\mathcal A|}$

然后把剩下的全部概率：1−ε，放到 $Q^{\pi_i}$ 最大的动作上。

因此，在同样需要保留探索的情况下，这种概率分配是最有利的：

$$\sum_a\pi_{i+1}(a|s)Q^{\pi_i}(s,a) \ge \sum_a\pi_i(a|s)Q^{\pi_i}(s,a)$$

右边又等于：$V^{\pi_i}(s)$

所以：

$$\sum_a\pi_{i+1}(a|s)Q^{\pi_i}(s,a) \ge V^{\pi_i}(s)$$

再根据策略改进定理得到：

$$V^{\pi_{i+1}}(s)\ge V^{\pi_i}(s)$$

也就是说：

$$\boxed{\pi_{i+1}\text{ 不会比 }\pi_i\text{ 更差}}$$



总结：
	新策略把非均匀的额外概率质量放在 $Q^{\pi_i}$ 最大的动作上，所以其一步期望不会小于旧策略：

$$
\sum_a\pi_{i+1}(a\mid s)Q^{\pi_i}(s,a)
\ge
\sum_a\pi_i(a\mid s)Q^{\pi_i}(s,a)
=V^{\pi_i}(s).
$$

也就是把更多的概率放在了产生最大效益的动作上

两个小修正：

1. 不是“产生最大即时奖励”的动作，而是 Q 最大的动作。Q 包含当前奖励和后续的长期回报。
2. 理论上保证更好，前提是使用的是真实 $Q^{\pi_i}​$，实际 MC/TD 中用的是估计值 Q^​，早期估计不准时，某一次更新不一定真的变好。


由 stochastic policy improvement theorem 得到

$$
V^{\pi_{i+1}}(s)\ge V^{\pi_i}(s).
$$

边界是：证明比较的是相对于同一个真实 $Q^{\pi_i}$ 的策略。实际学习早期只有误差很大的 $\hat Q$，一次 greedy improvement 可能依据错误排序而暂时降低真实表现。





### 3.4 Monte Carlo control：完整 return 更新动作价值

*首次完整讲解：Lecture 4 §3.4「Monte Carlo control：完整 return 更新动作价值」。*

Monte Carlo control 不是一种全新的估值公式，而是把前面两块内容正式接成循环：用 MC return 估计动作价值，再用 epsilon-greedy 根据新动作价值改进下一轮策略。


$$\boxed{\text{MC prediction}+\text{不断改策略}}$$

MC prediction：只负责“打分”

假设老师已经给你一个固定策略：π

比如在每个状态下，策略已经规定好怎么选动作。

MC prediction 要做的是：

> 这个固定策略到底有多好？

因此它不断按照这个固定策略运行 episode，然后用完整回报更新：$V^\pi(s)$  或者：$Q^\pi(s,a)$


$\boxed{V(s)\text{ 告诉你这个状态好不好，}Q(s,a)\text{ 告诉你应该选哪个动作。}}$


但它只负责评价，不会主动修改策略。

流程是：$\text{固定策略 }\pi \rightarrow \text{采样 episode} \rightarrow \text{估计 }V^\pi\text{ 或 }Q^\pi$


MC control：不仅打分，还要换策略

MC control 的目标不再是评价一个固定策略，而是：找到一个尽可能好的策略

因此它会重复：

$\text{评价当前策略} \rightarrow \text{根据评价结果改进策略} \rightarrow \text{再评价新策略}$

也就是：$\pi_0 \rightarrow Q^{\pi_0} \rightarrow \pi_1 \rightarrow Q^{\pi_1} \rightarrow \pi_2 \rightarrow\cdotsπ0​→Q$

这就和之前 DP 里的 policy iteration 很像：

$\text{Policy Evaluation} \leftrightarrow \text{Policy Improvement}$

区别只是：

- DP 利用已知环境模型计算价值；
- MC 利用实际采样得到的完整 return 估计价值。



**讲一遍完整的 MC Control**

先用一句话概括：

$\boxed{ \text{按照当前策略采样完整 episode} \rightarrow \text{计算 }G_t\text{ 更新 }Q \rightarrow \text{根据 }Q\text{ 改进策略} \rightarrow \text{重复} }$

MC control 的目标不是只评价一个固定策略，而是：

$\boxed{\text{通过不断采样和改进，找到更好的策略}}$

---

第 0 步：初始化 Q 和策略

假设每个状态有若干动作。

一开始我们不知道动作价值，可以初始化：Q(s,a)=0

然后构造一个具有探索能力的策略，例如 ε-greedy 策略。

策略不是一个单独动作，而是规定：$\pi(a\mid s)$

即每个状态下，各动作被选择的概率。

---

 第 1 步：按照当前策略生成一条完整 episode

假设当前策略是：πi

智能体在每个状态中，根据 πi 选择动作，得到：

$S_0,A_0,R_1,S_1,A_1,R_2,\ldots,S_T$

例如：

$S_0 \xrightarrow{A_0,+1} S_1 \xrightarrow{A_1,+2} S_2 \xrightarrow{A_2,+5} \text{终止}$

这里必须实际与环境交互，因为 MC 不知道：$P(s'\mid s,a)$

也不知道某动作最终会带来什么结果。

---

第 2 步：episode 结束后，计算每个时刻的 Gt

从后向前计算：

$G_2=5$    $G_1=2+\gamma G_2$           

$G_0=1+\gamma G_1$


一般可以写成递推形式：

$\boxed{G_t=R_{t+1}+\gamma G_{t+1}}$

注意，这只是计算完整回报的一种方便方式，并不是 TD bootstrap。
计算G和Q，在前面mc章节学过

因为这里的 $G_{t+1}$ 仍然是这条 episode 中后续真实奖励的总和，不是估计出来的 Q。

---

第 3 步：用 Gt​ 更新访问过的 $Q(S_t,A_t)$

对 episode 中每个访问过的状态—动作对：$(S_t,A_t)$

使用对应的 $G_t$ 更新：

$Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha \left[ G_t-Q(S_t,A_t) \right]$

其中：$G_t-Q(S_t,A_t)$

表示：

> 这次实际得到的完整回报，与以前估计值之间的差距。

例如之前：$Q(S_0,A_0)=4$

这次得到：G0=8

假设：α=0.5

那么：

$Q(S_0,A_0) \leftarrow 4+0.5(8-4) =6$

即原来的估计 4，向这次观察结果 8 靠近了一半。

也可以不用固定学习率，而是直接计算所有 return 的平均值：

$Q(s,a) = \frac{1}{N(s,a)} \sum_{\text{访问 }(s,a)}G_t$

---

 第 4 步：根据更新后的 Q 改进策略

假设某状态 s 下有两个动作：

$Q(s,a1​)=3$    $Q(s,a_2)=8$

说明根据目前的经验，a2​ 更好。

于是新策略应该更多选择：a2

但不能完全停止尝试 a1，否则早期估计有误时，可能永远发现不了真正好的动作。

因此使用 ε-greedy：

$\pi_{i+1}(a\mid s) = \begin{cases} 1-\varepsilon+\dfrac{\varepsilon}{|\mathcal A|}, & a=\arg\max_{a'}Q(s,a') \\[6pt] \dfrac{\varepsilon}{|\mathcal A|}, & \text{其他动作} \end{cases}$

也就是：

- 以 1−ε 的概率专门选择当前最优动作；
- 以 ε 的概率从所有动作中均匀随机抽一个。

所以更新 Q 后，策略也随之变化：

$Q\text{ 发生变化} \quad\Rightarrow\quad \text{最大 }Q\text{ 对应的动作可能变化} \quad\Rightarrow\quad \pi\text{ 发生变化}$



不断的调整每个状态的每个概率，最终目的就是帮助智能体找到更好的、甚至最优的策略

---

第 5 步：用新策略继续生成下一条 episode

现在得到新策略：πi+1

再按照它运行一条完整 episode：

$\pi_{i+1} \rightarrow \text{episode} \rightarrow G_t \rightarrow \text{更新 }Q \rightarrow \pi_{i+2}$

不断重复：

$\pi_0 \rightarrow Q \rightarrow \pi_1 \rightarrow Q \rightarrow \pi_2 \rightarrow\cdots$

随着采样次数增加：

- Q(s,a) 越来越准确；
- 策略越来越偏向高价值动作；
- 最终逐渐接近最优策略。

---

**七、用一个最简单的完整例子串起来**

假设只有一个关键状态 $s$，可选动作集合为

$$
\mathcal A(s)=\{a_1,a_2\}.
$$

初始化动作价值与访问次数：

$$
Q_0(s,a_1)=Q_0(s,a_2)=0,
\qquad
N_0(s,a_1)=N_0(s,a_2)=0.
$$

行为策略采用 $\varepsilon$-greedy。初始时两个动作价值并列，因此第一次选择可由 tie-breaking 或随机探索决定。

**第一轮：先采样到 $a_1$**

假设本轮在 $s$ 选择 $a_1$，episode 结束后得到完整回报 $G=4$。这是 $(s,a_1)$ 的第一次样本，因此

$$
\begin{aligned}
N_1(s,a_1)&=1,\\
Q_1(s,a_1)
&=0+\frac{1}{1}(4-0)=4.
\end{aligned}
$$

另一个动作尚未被访问，所以

$$
Q_1(s,a_1)=4,
\qquad
Q_1(s,a_2)=0.
$$

此时 $a_1$ 是当前贪心动作，策略会更倾向选择 $a_1$，但仍保留探索 $a_2$ 的概率。

**第二轮：探索到 $a_2$**

假设下一轮通过探索选择 $a_2$，并在 episode 结束后得到 $G=10$。同理：

$$
\begin{aligned}
N_2(s,a_2)&=1,\\
Q_2(s,a_2)
&=0+\frac{1}{1}(10-0)=10.
\end{aligned}
$$

于是当前两个动作价值为

$$
Q_2(s,a_1)=4,
\qquad
Q_2(s,a_2)=10.
$$

现在 $a_2$ 成为新的贪心动作，策略随之转而更倾向 $a_2$。这正是探索的作用：若没有尝试 $a_2$，算法就无法发现它可能更好。

**多轮采样以后**

假设两个动作后来分别收集到以下 complete returns：

| 动作 | 观测到的 returns | 样本平均动作价值 |
|---|---|---|
| $a_1$ | $4,2,3,5$ | $Q(s,a_1)=(4+2+3+5)/4=3.5$ |
| $a_2$ | $10,8,9,7$ | $Q(s,a_2)=(10+8+9+7)/4=8.5$ |

因为 $Q(s,a_2)>Q(s,a_1)$，两个动作且唯一贪心动作为 $a_2$ 时，$\varepsilon$-greedy 策略给出

$$
\pi(a_2\mid s)=1-\varepsilon+\frac{\varepsilon}{2}
=1-\frac{\varepsilon}{2},
\qquad
\pi(a_1\mid s)=\frac{\varepsilon}{2}.
$$

> [!summary] 这个例子展示了什么
> 完整 episode 提供 return sample；样本平均更新 $Q$；更新后的 $Q$ 改变贪心动作；$\varepsilon$-greedy 让策略偏向当前较优动作，同时继续保留发现估计错误的机会。

**它和 MC prediction 有什么区别？**

|        | MC prediction     | MC control              |
| ------ | ----------------- | ----------------------- |
| 目标     | 评价给定策略 $\pi$      | 逐步找到更好的策略               |
| 策略是否改变 | 固定不变              | 每轮根据 $Q$ 改进             |
| 可以估计什么 | $V^\pi$ 或 $Q^\pi$ | 必须维护 $Q(s,a)$，才能比较动作    |
| 更新价值之后 | 继续评价同一策略          | 生成下一轮 epsilon-greedy 策略 |

所以 `control` 的关键不是换了一条 MC 更新公式，而是**价值更新会反过来改变之后的数据采样策略**。





**一轮算法怎样运行？**

1. **初始化估计。** 对每个状态动作对设置 $Q(s,a)=0$、访问计数 $N(s,a)=0$，令 episode 编号 $k=1$。
2. **生成当前行为策略。** 根据当前 $Q$ 和 $\varepsilon_k$ 构造 epsilon-greedy 策略 $\pi_k$。
3. **采样完整 episode。** 从 $\pi_k$ 出发，得到一条状态、动作和奖励序列。MC 必须等 episode 结束，才能知道每个时间步之后的完整 return。
4. **从后向前计算 return。** 对每个时间步计算 $G_t$。它是这条实际轨迹上的随机回报，不是真实期望 $Q^{\pi_k}(s_t,a_t)$。
5. **更新首次访问的状态动作对。** 对本 episode 中第一次出现的每个 $(s_t,a_t)$，先令

   $$
   N(s_t,a_t)\leftarrow N(s_t,a_t)+1,
   $$

   再做样本平均更新：

   $$
   Q(s_t,a_t)
   \leftarrow
   Q(s_t,a_t)
   +\frac{G_t-Q(s_t,a_t)}{N(s_t,a_t)}.
   $$

6. **改进下一轮策略。** 根据更新后的 $Q$ 和下一轮 $\varepsilon_{k+1}$ 生成 $\pi_{k+1}$，然后采样下一条 episode。

第 5 步让同一状态动作对的多次 complete returns 逐渐逼近期望；第 6 步才是从 prediction 进入 control 的位置。

**自拟更新片段**：第一次在 $(s,a)$ 处观察到 $G=1$ 时，$N$ 从 $0$ 变为 $1$，因此

$$
Q_{\mathrm{new}}(s,a)=0+\frac{1}{1}(1-0)=1.
$$

下一条 episode 再访问同一个 $(s,a)$，但这次 $G=0$。此时 $N=2$，所以

$$
Q_{\mathrm{new}}(s,a)=1+\frac{1}{2}(0-1)=0.5.
$$

这一步不是在“挑较好的 return”，而是在把两条样本的平均值改为 $(1+0)/2$。完成本轮所有首次访问更新后，新的 $Q$ 会生成下一条 episode 的 epsilon-greedy 行为策略。

课件 Mars Rover 轨迹为：

$$
(s_3,a_1,0,s_2,a_2,0,s_3,a_1,0,s_2,a_2,0,s_1,a_1,1,\mathrm{terminal}).
$$

在 $\gamma=1$ 下，first-visit MC 对首次出现的 $(s_3,a_1)$、$(s_2,a_2)$、$(s_1,a_1)$ 都得到 return 1。直接按上面的轨迹回推，按 $s_1,\ldots,s_7$ 排列应为：

$$
Q(-,a_1)=[1,0,1,0,0,0,0],
\qquad
Q(-,a_2)=[0,1,0,0,0,0,0].
$$

这里第二个向量的第 2 项是 1，因为第一次在 $s_2$ 做 $a_2$ 后，轨迹的剩余奖励为 $0+0+1=1$；第二次访问同一对不计入 first-visit 更新。

> [!warning] 课件 first-visit 向量冲突
> 第 26 页显示的 $Q^{\varepsilon\text{-}\pi}(-,a_2)$ 是全零向量，但这与同页给出的轨迹及 first-visit 规则不一致。这里保留由轨迹直接计算出的 $[0,1,0,0,0,0,0]$，不要把投影片的该行当作本例的可复算结论。

> [!warning] 课件 epsilon 概率冲突
> 第 82 页在两个动作、$\varepsilon=1/3$ 时把 $s_1$ 的唯一贪心动作概率写成 $5/6$；但第 19 页公式给出 $1-1/3+(1/3)/2=2/3$。本笔记采用定义页的标准公式，并在 `notes/confusions.md` 保留该冲突。


**补充：那为什么状态价值V要对一个状态的所有动作按概率求和？**

这不是上面 MC control 的另一个更新步骤，而是一条连接状态价值与动作价值的关系。

它回答的是：**已经知道每个动作的价值后，怎样评价“来到状态 $s$ 后按策略 $\pi$ 行动”这件事的整体价值**

$$
\boxed{
\begin{aligned}
V^\pi(s)
&=\mathbb E_{A\sim\pi(\cdot\mid s)}\!\left[Q^\pi(s,A)\right]\\
&=\sum_{a\in\mathcal A(s)}\pi(a\mid s)Q^\pi(s,a)
\end{aligned}
}
$$

其中，$Q^\pi(s,a)$ 表示“先在 $s$ 执行动作 $a$，之后继续遵循 $\pi$”的期望回报；$\pi(a\mid s)$ 是策略在 $s$ 选择 $a$ 的概率。$V^\pi(s)$ 不固定第一个动作，而是先让策略按这些概率选择动作，因此它是各个 $Q^\pi(s,a)$ 的加权平均，输出是一个标量。

**自拟两动作例子。** 假设在状态 $s$：

$$
\begin{aligned}
\pi(a_1\mid s)&=0.8,
&\pi(a_2\mid s)&=0.2,\\
Q^\pi(s,a_1)&=10,
&Q^\pi(s,a_2)&=0.
\end{aligned}
$$

代入公式：

$$
\begin{aligned}
V^\pi(s)
&=\pi(a_1\mid s)Q^\pi(s,a_1)
  +\pi(a_2\mid s)Q^\pi(s,a_2)\\
&=0.8\times10+0.2\times0\\
&=8.
\end{aligned}
$$

结果 $8$ 的含义是：策略 $\pi$ 从状态 $s$ 出发的期望回报为 $8$。如果反复从 $s$ 出发，每次都按照 $\pi$ 行动，那么观测到的 sample returns 会有高有低，其平均值才会逐渐接近 $8$。所以 $8$ 不是某次 episode 必然得到的回报，也不是动作 $a_1$ 或 $a_2$ 自己的价值。

**为什么实际只执行一个动作，公式却要对所有动作求和？**

一次到达 $s$ 时，智能体只会抽样并执行一个动作：

$$
A_t\sim\pi(\cdot\mid s).
$$

但是在执行之前，$A_t$ 可能是 $a_1$，也可能是 $a_2$。状态价值计算的是这个随机选择的**期望结果**，所以必须把每个可能动作的价值乘以其发生概率再相加。简言之：**执行时抽样一个动作；评价策略时平均所有可能动作。**

**它与选择最大 $Q$ 有什么区别？**

| 表达式 | 回答的问题 | 结果类型 |
|---|---|---|
| $\displaystyle \arg\max_a Q^\pi(s,a)$ | 固定当前状态时，先选哪个动作的期望回报最高？ | 一个动作 |
| $\displaystyle \sum_a\pi(a\mid s)Q^\pi(s,a)$ | 按照整个策略选动作时，状态的期望价值是多少？ | 一个标量，即 $V^\pi(s)$ |

在上面的例子中，$\arg\max_a Q^\pi(s,a)=a_1$，它告诉我们应选哪个动作；$V^\pi(s)=8$ 则评价当前随机策略整体会取得怎样的平均回报。只有策略在 $s$ 把全部概率放在最大动作价值对应的动作上时，状态价值才等于最大动作价值。MC control 尚未知道真实的 $Q^\pi$，实际执行这两个计算时使用的是当前估计 $Q$。

**MC control 是否必须显式计算 $V$？**

通常不需要。Tabular MC control 直接用完整 return 更新 $Q(S_t,A_t)$，再根据 $Q$ 构造下一轮 epsilon-greedy 策略：

$$
G_t
\longrightarrow
\text{更新 }Q(S_t,A_t)
\longrightarrow
\text{改进策略 }\pi.
$$

因此，$V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a)$ 在这里主要帮助我们解释和比较策略，而不是要求算法额外维护一张 $V$ 表。

> [!summary] 核心结论
> $Q^\pi(s,a)$ 评价“固定先做动作 $a$”；$V^\pi(s)$ 评价“让策略 $\pi$ 决定做什么”。求和表示对策略的随机动作选择取期望，不表示一次访问要执行所有动作。



### 3.5 GLIE：既要无限探索，也要极限贪心

*首次完整讲解：Lecture 4 §3.5「GLIE：既要无限探索，也要极限贪心」。*

GLIE（Greedy in the Limit with Infinite Exploration，无限探索极限贪心）约束的是一列行为策略。它同时要求：

#### 条件一：每个可达状态--动作对都被无限次访问

$$
N_i(s,a)\to\infty
\quad\text{for every reachable }(s,a),
$$

其中，$N_i(s,a)$ 表示截止到第 $i$ 轮，状态--动作对 $(s,a)$ 一共被访问了多少次。例如在前 $100$ 条 episode 中，如果在状态 $s$ 选择 $a_1$ 共 $30$ 次，那么 $N_{100}(s,a_1)=30$。

这个要求意味着：随着训练轮数 $i$ 不断增加，每一个能够到达的状态--动作对都必须被访问无限多次。

为什么需要这个条件？因为 MC 通过多次 return 的平均来估计 $Q(s,a)$：

$$
Q(s,a)
\approx
\frac{G_1+G_2+\cdots+G_N}{N}.
$$

样本越多，这个平均值才越可靠。假如某个动作只试过一次，并且恰好得到

$$
G_1=-5,
$$

它的真实期望仍可能是 $10$，只是这一次运气不好。只有持续探索，才有机会纠正错误估计。

> [!important] 第一个条件保证什么
> $Q(s,a)$ 有足够的数据逐渐学准确。

#### 条件二：行为策略在极限处变成 greedy

$$
\Pr\!\left(
a_i\in\arg\max_a Q_i(s,a)\mid s_i=s
\right)\to1.
$$

意思是：在第 $i$ 轮，当智能体处于状态 $s$ 时，选择当前 $Q_i(s,a)$ 最大动作的概率，随着训练进行逐渐趋近于 $1$。

#### 两个条件看起来矛盾吗？

乍一看确实有点矛盾：

- 每个动作都要无限次尝试；
- 选择最优动作的概率又要趋近 1。

关键在于：

> **探索的比例可以趋近于 0，但探索的累计次数仍然可以趋近于无穷。**

例如令：

$$
\varepsilon_i=\frac{1}{i}.
$$

那么：

$$
\varepsilon_i\to0,
$$

所以策略越来越 greedy。

但是：

$$
\sum_{i=1}^{\infty}\frac{1}{i}=\infty.
$$

也就是说，虽然探索概率越来越小，但累计起来仍然有无限多次探索机会。

**自拟两动作例子**：假定一个状态在第 $i$ 轮总能被访问，且当前唯一贪心动作固定为 $a_1$。采用 $\varepsilon_i=1/i$ 时：

| 轮次 $i$ | $\varepsilon_i$ | $\Pr(a_1\mid s)$ | $\Pr(a_2\mid s)$ |
|---:|---:|---:|---:|
| 1 | 1 | 0.5 | 0.5 |
| 10 | 0.1 | 0.95 | 0.05 |
| 1000 | 0.001 | 0.9995 | 0.0005 |

所以行为确实逐渐接近 greedy；同时非贪心动作在这些访问中得到的探索概率和为 $\sum_i \varepsilon_i/2=\frac12\sum_i 1/i=\infty$，这是“无限探索”所需的直觉。这个计算依赖状态持续可达；若策略让该状态后来根本不再出现，光有 epsilon schedule 不能补回缺失数据。

在 tabular、episodic、充分访问等课件条件下，GLIE Monte Carlo control 收敛到 $Q^*$。

#### 为什么固定 $\varepsilon$ 不算 GLIE？

假设一直使用：

$$
\varepsilon=0.1.
$$

那么所有动作都会持续有机会被选择，因此通常能满足无限探索：

$$
N_i(s,a)\to\infty.
$$

但 greedy 动作的概率不会趋近 1。

假设有 4 个动作，最佳动作的概率始终是：

$$
1-0.1+\frac{0.1}{4}=0.925.
$$

它永远是 $92.5\%$，不会趋近 $100\%$。

## 4. Temporal-Difference Control

### 4.1 On-policy 与 off-policy 


区分算法时先写两个策略：

- 行为策略（behavior policy）$\pi_b$：智能体实际与环境交互、实际选择动作时所使用的策略；
- 目标策略（target policy）$\pi$：算法希望评估或改进的策略，因为explore导致的。

若 $\pi_b=\pi$，就是 on-policy(在线)；若两者不同，就是 off-policy(离线)。
它与“在线/离线”是不同轴：在线收集 transition 的算法仍可使用 off-policy target。


**自拟一状态例子**：两个动作中，行为策略 $\pi_b(a_1\mid s)=0.8$、$\pi_b(a_2\mid s)=0.2$，但目标策略是纯 greedy 的 $\pi(a_1\mid s)=1$。
某次数据可能来自行为策略实际选择的 $a_2$，算法却用它学习“之后总选 $a_1$”的目标价值；两个分布不同，因此是 off-policy。若目标策略也恰好是同一个 $[0.8,0.2]$ 分布，则同一批数据是在评估自身，才是 on-policy。

### 4.2 从 TD evaluation 到 TD control

*首次完整讲解：Lecture 4 §4.2「从 TD evaluation 到 TD control」。*

lec3是 TD prediction，它更新 $V$。当策略 $\pi$ 固定，只想评价它时，学习的是 $V^\pi(s)$。

进入 control 后，我们不只想评价状态，还想知道：

> 在状态 $s$ 下，哪个动作更好？

所以要学习 $Q(s,a)$。

TD control 用一步 transition 立即更新动作价值，再用新 $Q$ 调整 epsilon-greedy(ε-greedy) 行为策略。统一骨架是：

$$
Q(s_t,a_t)
\leftarrow
Q(s_t,a_t)
+
\alpha_t
\left[y_t-Q(s_t,a_t)\right].
$$

这里 $y_t$ 是希望当前估计靠近的标量 target；不同 TD control 算法使用不同的 $y_t$。中括号是 TD error，一次更新只改表格中的当前格子 $(s_t,a_t)$。例如当前 $Q(s_t,a_t)=4$、某种算法构造出 $y_t=6.4$、$\alpha_t=0.1$，则

$$
Q_{\mathrm{new}}(s_t,a_t)=4+0.1(6.4-4)=4.24.
$$

接下来的 Q-learning 和 SARSA 并没有换掉这个更新逻辑，只是分别规定如何从下一步信息计算 $6.4$ 这样的 $y_t$。

Q-learning 与 SARSA 的核心差异全部藏在 $y_t$：下一步是按目标策略取最大值，还是按行为策略真正采样一个动作。

### 4.3 Q-learning：行为可以探索，target 假设下一步 greedy

*首次完整讲解：Lecture 4 §4.3「Q-learning：行为可以探索，target 假设下一步 greedy」。*

Q-learning 的行为策略通常对当前 $Q$ 做 epsilon-greedy，但 target 直接使用下一状态的最大动作价值(它不关心下一步实际上选择了什么动作，而是假设下一步会选择当前价值最大的动作)：

$$
y_t^{\mathrm{Q}}
=
r_t+\gamma\max_{a'}Q(s_{t+1},a'),
$$

$$
Q(s_t,a_t)
\leftarrow
Q(s_t,a_t)
+
\alpha_t
\left[y_t^{\mathrm{Q}}-Q(s_t,a_t)\right].
$$

**自拟例子**：当前 $Q(s_t,a_t)=4$， $r_t=1$，$\gamma=0.9$，下一状态两个动作价值为 $[2,6]$，$\alpha=0.1$。则

$$
y_t^{\mathrm{Q}}=1+0.9\times6=6.4,
$$

$$
Q_{\mathrm{new}}(s_t,a_t)
=4+0.1(6.4-4)=4.24.
$$

即使行为策略随后实际选择价值为 2 的探索动作，当前 target 仍使用 6，因此 Q-learning 学的是 greedy target policy，属于 off-policy(离线)。

![[Pasted image 20260729011616.png]]

对有限状态、有限动作的 tabular MDP，课件给出的收敛条件包括 GLIE、bounded rewards，以及每个状态-动作对各自满足的 Robbins--Monro 步长条件：

$$
\sum_{n=1}^{\infty}\alpha_n(s,a)=\infty,
\qquad
\sum_{n=1}^{\infty}\alpha_n(s,a)^2<\infty.
$$

这些条件用于说明表格型 Q-learning 在什么条件下能收敛到最优动作价值 $Q^*$

这个结论不能直接迁移到神经网络 Q-learning。

#### 先看 $\alpha_n(s,a)$ 的下标

Q-learning 更新： 状态-动作对 $(s,a)$ 每被访问一次，就执行更新：

$$
Q(s,a)
\leftarrow
Q(s,a)
+\alpha_n(s,a)
\left[
r+\gamma\max_{a'}Q(s',a')-Q(s,a)
\right].
$$

其中，$\alpha_n(s,a)$ 表示 $(s,a)$ **第 $n$ 次被访问时**使用的学习率。这里的 $n$ 通常不是全局第 $n$ 步，而是这个特定状态-动作对被访问的次数。

例如：

- $(s_1,a_1)$ 已被访问 $100$ 次，使用 $\alpha_{100}(s_1,a_1)$；
- $(s_2,a_2)$ 只被访问 $5$ 次，使用 $\alpha_5(s_2,a_2)$。

因此，Q 表中的每个格子都有自己的访问次数和学习率序列。

#### 第一个条件：保留持续修正能力

$$
\sum_{n=1}^{\infty}\alpha_n(s,a)=\infty.
$$

它不是说学习率越来越大，而是说：虽然每次调整可以越来越小，但所有调整量累加起来不能是有限的。

例如取

$$
\alpha_n=\frac{1}{n}.
$$

虽然 $1/n\to0$，但

$$
1+\frac12+\frac13+\frac14+\cdots=\infty.
$$

它下降得不算太快，累计起来仍然具有持续修正估计的能力。相反，若

$$
\alpha_n=\frac{1}{2^n},
\qquad
\sum_{n=1}^{\infty}\frac{1}{2^n}=1,
$$

后面所有更新的总影响是有限的。若早期 $Q(s,a)$ 估计偏差很大，后期即使得到大量正确数据，也可能没有足够能力把它纠正回来。

> [!important] 第一个条件保证什么
> 后续数据始终保有足够的累计影响力，可以持续纠正早期估计。

#### 第二个条件：限制长期噪声

$$
\sum_{n=1}^{\infty}\alpha_n(s,a)^2<\infty.
$$

它要求学习率的平方和有限。仍取 $\alpha_n=1/n$ 时：

$$
\sum_{n=1}^{\infty}\alpha_n^2
=
\sum_{n=1}^{\infty}\frac{1}{n^2}
<\infty.
$$

Q-learning 的每个 sampled target

$$
r+\gamma\max_{a'}Q(s',a')
$$

都可能带有随机噪声，因为奖励、下一状态和采样轨迹都可能是随机的。如果学习率始终过大，每个随机样本都会让 $Q$ 大幅晃动，噪声便会长期保留下来。平方和有限大致保证学习率下降得足够快，使后期随机噪声不会无限累积。

> [!important] 第二个条件保证什么
> 后期更新幅度逐渐减小，从而限制长期随机波动。

#### 为什么两个条件要同时满足？

两个条件分别防止两个极端：

- 若 $\sum_n\alpha_n<\infty$，学习率下降太快，后期几乎学不动，早期错误无法充分纠正；
- 若 $\sum_n\alpha_n^2=\infty$，学习率下降太慢，随机噪声可能持续累积，估计一直晃动。

> [!summary] Robbins--Monro 步长的作用
> 步长下降得足够慢，使后续样本能持续修正估计；又下降得足够快，使后期噪声不会无限累积。

以第 $n$ 次访问某个固定 $(s,a)$ 时取 $\alpha_n=1/n$ 为例，$\sum_n1/n$ 发散而 $\sum_n1/n^2$ 收敛，因此满足两个条件。固定 $\alpha_n=0.1$ 的平方和发散，不能据此调用 tabular 收敛定理；这不等于算法一定发散，只表示它不在课件保证的条件内。


### 4.4 SARSA：target 使用实际选择的下一动作

*首次完整讲解：Lecture 4 §4.4「SARSA：target 使用实际选择的下一动作」。*

SARSA 的名字来自五元组 $S_t,A_t,R_t,S_{t+1},A_{t+1}$。它先按当前行为策略采样 $a_{t+1}$，再构造：

$$
y_t^{\mathrm{SARSA}}
=
r_t+\gamma Q(s_{t+1},a_{t+1}),
$$

$$
Q(s_t,a_t)
\leftarrow
Q(s_t,a_t)
+
\alpha_t
\left[y_t^{\mathrm{SARSA}}-Q(s_t,a_t)\right].
$$

沿用上一例，若行为策略实际在 $s_{t+1}$ 采样到价值为 2 的动作，则

$$
y_t^{\mathrm{SARSA}}=1+0.9\times2=2.8,
$$

$$
Q_{\mathrm{new}}(s_t,a_t)
=4+0.1(2.8-4)=3.88.
$$

下降不是 SARSA “更差”，而是它把当前探索策略会承担的风险算进目标，因此属于 on-policy。若 $s_{t+1}$ 是 terminal，两种算法都令 $y_t=r_t$，不能丢掉进入终止状态时观察到的即时奖励。

它和 Q-learning 都是：

> 每走一步，就用下一步的信息更新当前的 Q(st,at)。

但它使用的是下一状态中**实际选中的动作** $A_{t+1}$的价值。

这个动作通常是通过 $\varepsilon-greedy$ 策略选出来的，所以有可能不是当前价值最大的动作。

![[Pasted image 20260729011657.png]]




### 4.5 课件 Mars Rover：同一 transition 的两个 target

transition就是状态转移的意思

课件第 81--87 页固定
$\gamma=1$、$\alpha=0.5$。从 $s_6$ 执行 $a_1$，观察 $r=0$ 并到达 $s_7$；行为策略随后实际选 $a_2$，且

$$
Q(s_7,a_2)=5,
\qquad
\max_aQ(s_7,a)=10.
$$

按课件初始化，$Q(s_6,a_1)=0$：

$$
Q_{\mathrm{SARSA}}(s_6,a_1)
=0+0.5(0+5-0)=2.5,
$$

$$
Q_{\mathrm{Q-learning}}(s_6,a_1)
=0+0.5(0+10-0)=5.
$$

两者使用了同一条实际 transition；差别只来自下一动作价值是“实际采样的 5”还是“target policy 的最大值 10”。初始化会影响早期数值和动作，满足各自收敛条件时不决定最终极限。

### 4.6 三种 tabular control 的统一对照

| 方法 | 数据需求 | Target | Bootstrap | Policy relation | 更新时机 |
|---|---|---|---:|---|---|
| MC control | 完整 episode | $G_t$ | 否 | on-policy GLIE | episode 结束后 |
| SARSA | 一步 transition + 实际 $a_{t+1}$ | $r_t+\gamma Q(s_{t+1},a_{t+1})$ | 是 | on-policy | 每一步 |
| Q-learning | 一步 transition | $r_t+\gamma\max_{a'}Q(s_{t+1},a')$ | 是 | off-policy | 每一步 |

当 $\varepsilon=0$ 且 tie-breaking 一致时，SARSA 实际选择的 $a_{t+1}$ 就是 argmax，某一步 target 可与 Q-learning 相同；允许探索后，两者一般不同。
(tie-breaking 一致”就是说 SARSA 和 Q-learning 使用相同的平局处理规则，从而选择同一个greedy 动作)


[!important] 
  > 当 SARSA 不再探索时，它实际选择的动作就是最大 $Q$ 值动作，因此该步 target 与 Q-learning 相同。



## 5. Value Function Approximation

### 5.1 从独立表格到共享参数

*首次完整讲解：Lecture 4 §5.1「从独立表格到共享参数」。*

tabular(表格)为每个 $(s,a)$ 保存独立数字；图像、机器人连续状态或长历史会使状态空间==极大甚至无限==。

前面学的 VI、PI、MC、TD、Q-learning、SARSA，在进入第 5 章之前，基本都默认是 tabular（表格型）方法。

没看到真的画出一张表，是因为课件通常用：$V(s),\qquad Q(s,a),\qquad \pi(a|s)$

这样的函数记号表示，但在计算机里，它们往往就是数组或表格。
Tabular 的真正含义是：

> **每一个状态，或者每一个状态—动作对，都拥有一个独立存储的数值。**


下面开始取消表格形式：不用一张巨大的 Q 表保存每个 (s,a) 的价值，而是用一个带参数的函数，输入 (s,a)，输出它的预测价值。

函数逼近（function approximation）改用参数化映射：

$$
\hat Q(s,a;w)\approx Q^\pi(s,a),
$$

其中输入是状态动作，$w$ 是参数向量，输出是一个标量。参数共享减少存储和数据需求，并让相似输入互相泛化；代价是更新一个样本可能同时改变许多未访问状态的预测。
所有状态动作对共享参数w

[!important] 结论就是：

>	$\boxed{\text{用参数 }w\text{ 控制的函数，去近似真实 }Q^\pi(s,a)}$



**自拟共享参数例子**：令

$$
\hat Q(s,a;w)=w_1+w_2x(s,a),
$$

其中 $x(s,a)$ 是一个相似度特征 (这个状态动作对的特征)。若 $w=(2,1)$、$x(s_A,a)=1$、$x(s_B,a)=0.8$，则两个输入的预测分别是 3 和 2.8。

假设一次学习把共享偏置 $w_1$ 从 2 改为 2.5，它们会**同时**变成 3.5 和 3.3，即使样本只来自 $(s_A,a)$。若两个输入确实相似，这就是泛化；若它们实际价值差很大，同一次联动就是干扰。表格表示不会有这种联动，因为每个格子拥有独立参数。

### 5.2 Oracle regression 与 SGD

*首次完整讲解：Lecture 4 §5.2「Oracle regression 与 SGD」。*

[!question]现在不用 Q 表了，而是用模型 $\hat Q(s,a;w)$ 预测价值，那么参数 w 到底怎么训练？

	先把它看成一个普通的**回归问题**，用损失函数和 SGD 更新参数。

强化学习里对应为：

$x=(s,a)$     $\hat y=\hat Q(s,a;w)$     $y=Q^\pi(s,a)$

也就是：

	输入状态和动作，模型预测它的长期价值，希望预测接近真实动作价值。



课件先假设有 [oracle](academic-term-lookup:oracle) 能返回真实 $Q^\pi(s,a)$。这样 RL 暂时退化成监督学习：输入 $(s,a)$，标签 $y=Q^\pi(s,a)$。用半平方损失

$$
\ell(w;s,a,y)
=
\frac12\left[y-\hat Q(s,a;w)\right]^2,
$$
训练目标就是让损失越来越小

随机梯度下降（stochastic gradient descent, SGD）的一步是

$$
w
\leftarrow
w
+
\alpha
\left[y-\hat Q(s,a;w)\right]
\nabla_w\hat Q(s,a;w).
$$




[!example]：
$\hat Q(x;w)=wx$，样本 $x=2,y=5$，当前 $w=1$，$\alpha=0.1$。预测为 2、误差为 3、$\nabla_w\hat Q=x=2$，因此

$$
w_{\mathrm{new}}
=1+0.1\times3\times2
=1.6.
$$

真实 model-free RL 没有 oracle，接下来的关键问题是用什么可观测 target 替代 $Q^\pi$。

### 5.3 Monte Carlo VFA：用完整 return 当 label

*首次完整讲解：Lecture 4 §5.3「Monte Carlo VFA：用完整 return 当标签」。*

在 on-policy episodic setting 中，$G_t$ 是 $Q^\pi(s_t,a_t)$ 的有噪声 sample。MC VFA 把一条 episode 转成训练对 $((s_t,a_t),G_t)$：

$$
w
\leftarrow
w
+
\alpha
\left[G_t-\hat Q(s_t,a_t;w)\right]
\nabla_w\hat Q(s_t,a_t;w).
$$

**自拟数值例子**：一条 episode 从时刻 $t$ 起得到奖励 $1,2$，令 $\gamma=1$，所以 $G_t=3$。若 $\hat Q(x;w)=wx$、当前 $x(s_t,a_t)=2$、$w=1$、$\alpha=0.1$，则预测为 2，且 $\nabla_w\hat Q=x=2$：

$$
w_{\mathrm{new}}
=1+0.1(3-2)\times2
=1.2.
$$

更新后该输入的预测从 2 变成 2.4，而不是被直接覆盖为 3；$G_t=3$ 只是这条 episode 提供的一个 noisy label，更多 episode 会继续修正参数。

它不 bootstrap，但必须等 episode 结束；与 tabular MC 不同，一次参数更新会改变多个输入的预测。

为什么不bootstrap：
	因为mc 他的target Gt 完全由这条轨迹上**实际观察到的奖励**组成，没有使用当前模型对未来价值的估计




### 5.4 Semi-gradient TD：target 依赖估计，但本步冻结

先回忆tabular表格型 TD

以前表格型 TD(0) 是：

$V(s_t)\leftarrow V(s_t) +\alpha \left[ r_{t+1}+\gamma V(s_{t+1})-V(s_t) \right]$

定义： 

TD target：$y_t^{TD}=r_{t+1}+\gamma V(s_{t+1})$

TD error：$\delta_t = y_t^{TD}-V(s_t)$

所以就是：$V(s_t)\leftarrow V(s_t)+\alpha\delta_t$

这里直接修改 V 表中状态 st​ 对应的格子。

---

现在没有 V 表了：

函数近似中，状态价值由一个函数预测：$\hat V(s;w)$

比如：$\hat V(s;w)=w^\top x(s)$

不能再直接修改某个 V(st)，而要修改参数 w。

对当前参数 $w_t$，先计算 TD target   (从之前的V(st+1​)变为$\hat V(s_{t+1};w_t)$)：

$$
y_t^{\mathrm{TD}}
=
r_t+\gamma\hat V(s_{t+1};w_t).
$$

随后把 $y_t^{\mathrm{TD}}$ 当作本次更新中的固定标量，只对当前预测求梯度：

$$
w_{t+1}
=
w_t
+
\alpha
\left[
y_t^{\mathrm{TD}}-\hat V(s_t;w_t)
\right]
\nabla_w\hat V(s_t;w_t).
$$

这叫半梯度（semi-gradient）：target 的数值都是由 $w_t$ 算出，也就是误差两边都包含w。 但本步不沿 target 分支反向传播。把这个target当作固定数字

它不是把 target 当成 oracle 真值，也不是对“target 和 prediction 都随 $w$ 变化”的完整平方表达式求真正梯度。


**自拟线性例子**：$\hat V(s;w)=w^\top x(s)$，$w_t=(1,2)$，$x(s_t)=(1,0)$，$x(s_{t+1})=(0,1)$，$r_t=1$，$\gamma=0.5$，$\alpha=0.1$。则

$$
\hat V(s_t;w_t)=1,
\qquad
\hat V(s_{t+1};w_t)=2,
$$

$$
y_t^{\mathrm{TD}}=1+0.5\times2=2,
\qquad
\delta_t=2-1=1.
$$

因为 $\nabla_w\hat V(s_t;w_t)=x(s_t)=(1,0)$：

$$
w_{t+1}
=(1,2)+0.1\times1\times(1,0)
=(1.1,2).
$$

这个例子同时显示三层近似：只采样一个 transition、用下一状态 bootstrap、再用共享参数表示价值。


### 5.5 用近似 $Q$ 做 control

*首次完整讲解：Lecture 4 §5.5「用近似 $Q$ 做 control」。*

函数逼近 control 仍交替进行 approximate policy evaluation 和 epsilon-greedy improvement。统一 semi-gradient 骨架为：

$$
w
\leftarrow
w
+
\alpha
\left[y_t-\hat Q(s_t,a_t;w)\right]
\nabla_w\hat Q(s_t,a_t;w).
$$

| 方法 | $y_t$ |
|---|---|
| MC | $G_t$ |
| SARSA | $r_t+\gamma\hat Q(s_{t+1},a_{t+1};w)$ |
| Q-learning | $r_t+\gamma\max_{a'}\hat Q(s_{t+1},a';w)$ |

表面上只是把表格 $Q$ 换成 $\hat Q$；实质上参数共享会让一个样本改变许多状态，target 又会随参数变化，因此 tabular contraction 和收敛结论不能直接照搬。

**自拟 Q-learning VFA 更新**：
仍令 $\hat Q(x;w)=wx$。当前样本的特征为 $x(s_t,a_t)=2$，$w=1$，所以 prediction 是 2；观察到 $r_t=1$，取 $\gamma=0.5$，且下一状态所有动作中的最大近似价值为 4。于是

$$
y_t=1+0.5\times4=3,
$$

$$
w_{\mathrm{new}}
=1+0.1(3-2)\times2
=1.2.
$$

当前输入的预测因此从 2 变为 2.4。与此同时，一个没有出现在本次 transition 中、但特征值为 1 的状态动作对，其预测也从 1 变为 1.2。这正是近似 control 与 tabular Q-learning 的结构性区别：同一次 TD error 会沿共享参数影响别处。

### 5.6 Deadly triad：三个因素的交集

*首次完整讲解：Lecture 4 §5.6「Deadly triad：三个因素的交集」。*

致命三角（deadly triad）指以下三者同时出现时的稳定性风险：

1. **Function approximation**：根据一个样本修改 w 时，其他Q会一起变化；
2. **Bootstrapping**：用自己的估计当目标target的一部分
3. **Off-policy learning**：数据分布与 target policy 不同。

例如 Q-learning：

- 行为策略：ε-greedy，实际会探索；
- 目标策略：greedy，target 中直接取最大值。

即：μ≠π

数据主要来自行为策略 μ，但算法想学习目标策略 π。

所以某些被目标策略看重的动作，可能在数据中没有被充分采样。


函数逼近、bootstrapping、off-policy 三者同时出现时，价值估计可能互相放大错误，导致震荡甚至发散。

为了理解这个风险，先把四个容易混在一起的概念分开：**backup** 是一次局部更新，**Bellman operator** 是把这种更新施加到整套价值函数上，**contraction** 描述算子会不会缩小估计之间的距离，**projection** 则是在函数逼近时把结果拉回当前模型能够表示的函数类。

#### 5.6.1 Bellman backup：把后续信息传回当前状态

之前已经做过很多次 backup，只是课件不一定一直使用这个名字。例如 tabular TD(0) 更新为：

$$
V(S_t)
\leftarrow
V(S_t)
+
\alpha
\left[
R_{t+1}
+
\gamma V(S_{t+1})
-
V(S_t)
\right].
$$

其中

$$
Y_t^{\mathrm{TD}}
=
R_{t+1}
+
\gamma V(S_{t+1})
$$

是根据下一状态价值算出的新 target。把下一状态的信息向前传给当前状态，这个过程就叫 **Bellman backup**。
通俗地说，就是根据“当前奖励 + 后续价值”，重新判断当前状态应该值多少。

> [!example] 具体计算：一次 Bellman backup
> 有一个 transition：$s_1\xrightarrow{r=1}s_2$。当前 $V(s_2)=5$，折扣因子 $\gamma=0.9$，则对 $s_1$ 的 target 为
>
> $$
> Y=1+0.9\times5=5.5.
> $$
>
> 这不是把 $V(s_1)$ 直接改成 $5.5$，而是让 $V(s_1)$ 向 $5.5$ 靠近；靠近多少由步长 $\alpha$ 决定。

#### 5.6.2 Bellman operator：对整套价值估计做 backup

operator（算子）可以先理解为
	**输入一整套价值估计，输出一整套经过 Bellman backup 后的新价值估计。**

假设：

$$V= \begin{bmatrix} V(s_1)\\ V(s_2)\\ V(s_3) \end{bmatrix}$$​​

Bellman operator $T^\pi$ 接收这整个 V，对每个状态都做一次 Bellman backup：
$$
(T^\pi V)(s)
=
\mathbb{E}_\pi
\left[
R_{t+1}
+
\gamma V(S_{t+1})
\mid
S_t=s
\right].
$$

因此

$$
V
\xrightarrow{T^\pi}
T^\pi V
$$

表示：用当前 $V$ 估计未来，再重新计算所有状态的价值。


对 Q-learning来说，对应的是 Bellman 最优算子：

$$
(T^*Q)(s,a)
=
\mathbb{E}
\left[
R_{t+1}
+
\gamma\max_{a'}Q(S_{t+1},a')
\mid
S_t=s,A_t=a
\right].
$$

也就是对每个 (s,a)计算：

$\text{当前奖励} + \gamma\times \text{下一状态最大动作价值}$

所以 Q-learning 的 target：$r+\gamma\max_{a'}Q(s',a')$

就是一次**采样版 Bellman backup**。




> [!example] 具体计算：从一个状态更新到整套状态
> 假设 $V=(V(s_1)$,   $V(s_2))=(2,5)$，$s_1$ 一步后以奖励 $1$ 转移到 $s_2$；假设 $s_2$ 的下一步直接终止并得到奖励 $0$，且 $\gamma=0.9$。那么
>
> $$
> (T^\pi V)(s_1)=1+0.9\times5=5.5,
> \qquad
> (T^\pi V)(s_2)=0.
> $$
>
> 所以这次算子作用的结果可以写成 $T^\pi V=(5.5,0)$：它不是只更新一个数字，而是定义了整套状态的新估计。


#### 5.6.3 Contraction：为什么表格型 Bellman 更新有收敛趋势

Contraction（压缩映射）描述的是：两组不同的价值估计经过 Bellman operator 后，彼此之间的距离会缩小。使用最大范数

$$
\|V\|_\infty=\max_s|V(s)|,
$$

Bellman operator 满足

$$
\left\|
T^\pi V_1-T^\pi V_2
\right\|_\infty
\le
\gamma
\left\|
V_1-V_2
\right\|_\infty.
$$

因为 $0\le\gamma<1$，两套估计的最大差距经过一次 operator 后最多只剩原来的 $\gamma$ 倍。这种性质使反复 Bellman backup 能够收敛到唯一固定点：

$$
V^\pi=T^\pi V^\pi,
\qquad
Q^*=T^*Q^*.
$$

> [!example] 具体计算：距离如何缩小
> 令 $\gamma=0.9$，两套估计的最大差距最初为 $10$。经过一次 Bellman operator 后，差距不超过 $9$；再经过一次不超过 $8.1$：
>
> $$
> 10\longrightarrow9\longrightarrow8.1\longrightarrow7.29\longrightarrow\cdots
> $$
>
> 因此，可以把 contraction 理解成 Bellman 更新本身具有“逐渐压小估计误差”的趋势。

#### 5.6.4 Projection：函数逼近为什么多出一步

表格型情况下，每个状态都有独立数值，Bellman operator 算出什么，我们就能保存什么。例如

$$
V=
\begin{bmatrix}
v_1\\
v_2
\end{bmatrix}
\xrightarrow{T}
TV=
\begin{bmatrix}
1\\
1
\end{bmatrix}.
$$

表格可以直接保存 $V(s_1)=1$ 和 $V(s_2)=1$，不需要额外处理。

函数逼近时，价值由 $\hat V(s;w)$ 表示，Bellman operator 的输出未必属于当前函数类，因此还要做 **projection（投影）**：把任意价值结果映射回模型能够表示的近似函数集合。

假设模型非常简单，只能表示：

$\hat V(s_1;w)=w$     $\hat V(s_2;w)=2w$

那么所有能表示的价值向量都是：$\hat V_w= \begin{bmatrix} w\\ 2w \end{bmatrix}$

例如：

$w=1 \Rightarrow \begin{bmatrix} 1\\ 2 \end{bmatrix}$    $w=2 \Rightarrow \begin{bmatrix} 2\\ 4 \end{bmatrix}$

但它无法精确表示：$\begin{bmatrix} 1\\ 1 \end{bmatrix}$

因为不存在一个 w，同时满足：

w=1,     2w=1

这两个要求冲突。


于是只能找一个最接近的结果

Bellman backup 给出目标：

$TV= \begin{bmatrix} 1\\ 1 \end{bmatrix}$

但模型只能表示：

$\begin{bmatrix} w\\ 2w \end{bmatrix}$

于是做回归：$\min_w \left[ (w-1)^2+(2w-1)^2 \right]$

最后得到：w=0.6

因此模型实际保存的是：

$\begin{bmatrix} 0.6\\ 1.2 \end{bmatrix}$

而不是 Bellman backup 原本得到的：

$\begin{bmatrix} 1\\ 1 \end{bmatrix}$

这个“从理想结果中，找一个模型能够表示的最近结果”的过程，就是：projection，投影

记号通常写成：Π(TV)

其中：

- T：先做 Bellman backup；
- Π：再拟合回当前函数能够表示的范围。


#### 5.6.5 为什么“backup + projection”可能振荡或发散？

在表格型情形中，可以直接保存 Bellman operator 给出的新价值；函数逼近却只能保存当前模型能够表示的结果。因此，一轮更新可以概念化地写成

$$
\hat V_{w_t}
\xrightarrow{\;T\;}
T\hat V_{w_t}(target)
\xrightarrow{\;\Pi\;}
\Pi T\hat V_{w_t}
\approx
\hat V_{w_{t+1}}.
$$

下一轮又以新的近似价值生成 target：

$$
\hat V_{w_{t+1}}
\xrightarrow{\;T\;}
T\hat V_{w_{t+1}}
\xrightarrow{\;\Pi\;}
\Pi T\hat V_{w_{t+1}}
\approx
\hat V_{w_{t+2}}.
$$

这里的 $\Pi$ 是“拟合回当前函数类”的概念化表示。实际算法通常只做有限步 SGD，因此 $\hat V_{w_{t+1}}$ 只是对投影结果的近似，并不一定等于精确的 $\Pi T\hat V_{w_t}$。

关键在于：$T$ 在最大范数下可以是 contraction，但函数拟合步骤不一定在同一个范数和数据分布下保持距离不增。因此，即使 $T$ 本身缩小误差，复合映射 $\Pi T$ 也不一定继续收缩。如果某些方向的误差在每轮“backup + fitting”后被放大，便可能形成

$$
\text{价值误差}
\longrightarrow
\text{更偏的 bootstrap target}
\longrightarrow
\text{更大的参数误差}
\longrightarrow
\text{下一轮更大的价值误差}.
$$

> [!warning] 训练中可能观察到的信号
> - 价值估计在高低之间反复振荡；
> - 参数范数持续增大；
> - $Q$ 值快速增长甚至数值爆炸；
> - loss 不降反升，或短暂下降后持续上升。

这些现象说明训练不稳定，但不能仅凭其中一个现象就断言 deadly triad 是唯一原因；步长过大、实现错误或奖励尺度不当也可能产生类似表现。

#### 5.6.6 展开风险链条：三个因素如何形成错误反馈

下面把前面的抽象复合映射放进 Q-learning 的具体更新中。设 replay buffer 的数据主要由旧行为策略 $\mu$ 产生，它大部分时间选择 $a_1$，很少选择 $a_2$；Q-learning target 却按 greedy target policy 在下一状态取最大动作价值。

> [!example] 风险链条：一个高估值如何进入下一轮 target
> **初始估计。** 在某个下一状态 $s'$，网络给出
>
> $$
> \hat Q(s',a_1;w)=3,
> \qquad
> \hat Q(s',a_2;w)=2.1.
> $$
>
> **1. Function approximation 造成联动。** 网络使用包含 $a_1$ 的样本更新共享参数。虽然本次没有执行 $a_2$，共享表示仍可能让
>
> $$
> \hat Q(s',a_2;w):2.1\longrightarrow4.
> $$
>
> 这个变化不是新经验支持了 $a_2$，而是参数共享带来的间接影响。
>
> **2. Bootstrapping 把当前误差写进新 target。** 为了只观察这一项，令当前 transition 的 $r=0$、$\gamma=0.9$。联动发生前，target 为
>
> $$
> y_{\text{before}}
> =0+0.9\max(3,2.1)
> =2.7.
> $$
>
> 当 $a_2$ 被推高到 $4$ 后，同一形式的 target 变为
>
> $$
> y_{\text{after}}
> =0+0.9\max(3,4)
> =3.6.
> $$
>
> 没有充分数据支持的数值 $4$ 因而被 $\max$ 选中，并成为训练其他预测的新依据。
>
> **3. Off-policy distribution mismatch 使纠正不足。** target policy 已因当前估计而偏向 $a_2$，但 replay 数据仍主要来自常选 $a_1$ 的行为策略 $\mu$。由于真实执行 $a_2$ 的样本很少，这个高估值可能无法被及时拉回。
>
> **反馈闭环。** $a_2$ 被意外高估，$\max Q$ 将它写入 target，模型再拟合偏高的 target，共享参数继续改变其他 $Q$ 值，而数据分布又缺少针对 $a_2$ 的纠正。错误于是可能自我强化。

这里的“很少采样 $a_2$”描述的是本例中的覆盖不足；off-policy 的核心定义仍是产生数据的行为策略 $\mu$ 与更新所针对的 target policy 不同。上面的数值展示反馈机制，并不单独构成发散证明。

#### 5.6.7 为什么 tabular Q-learning 没有同样的问题？

表格型 Q-learning 同样使用 bootstrapping，也可以用探索行为策略收集数据来学习 greedy target policy，因此它同样是 off-policy。区别在于它没有共享函数参数：

| 比较项 | Tabular Q-learning | 使用函数逼近的 Q-learning |
|---|---|---|
| 保存方式 | 每个 $(s,a)$ 有独立表格项 | 多个状态动作对共享参数 $w$ |
| 更新 $Q(s,a_1)$ 时 | 不会直接改变 $Q(s,a_2)$ | 可能同时改变其他 $\hat Q(s',a';w)$ |
| 未采样动作被间接推高 | 不会由参数共享导致 | 可能发生 |
| Bellman 结果能否直接保存 | 可以 | 通常还要拟合回函数类 |

因此，更新 $Q(s,a_1)$ 时不会仅仅因为参数联动，就把未采样的 $Q(s,a_2)$ 从 $2.1$ 推到 $4$。这正是 deadly triad 强调三者交集的原因，而不是声称任意一个因素单独出现就必然导致发散。

这也不表示 tabular Q-learning 无条件收敛。有限 MDP、每个状态动作对被充分访问、奖励有界，以及步长满足相应随机近似条件等标准假设仍然需要成立。

#### 5.6.8 把四个概念压缩成一条逻辑链

> [!summary] 总结：从 Bellman backup 到不稳定性
> - **Bellman backup**：根据当前奖励和下一状态价值，计算当前价值的新 target。
> - **Bellman operator**：把这种 backup 同时定义在整套价值函数上。
> - **Contraction**：精确 Bellman operator 在相应条件和范数下会逐步缩小两套价值估计的距离。
> - **Projection**：当 Bellman 结果超出模型的可表示范围时，把它拟合回当前函数类。
>
> 函数逼近时反复进行的过程可以概念化为
>
> $$
> \hat V_w
> \xrightarrow{\;T\;}
> T\hat V_w
> \xrightarrow{\;\Pi\;}
> \Pi T\hat V_w.
> $$
>
> 虽然 $T$ 本身可能是 contraction，但 $\Pi$ 不一定在同一度量下保留这种性质，所以复合映射 $\Pi T$ 可能失去稳定收敛保证。函数逼近、bootstrapping 和 off-policy 数据分布再互相作用时，误差便可能形成反馈并持续放大。




## 6. Deep Q-Network

### 6.1 为什么神经网络 Q-learning 需要额外机制

*首次完整讲解：Lecture 4 §6.1「为什么神经网络 Q-learning 需要额外机制」。*

[!important]
	把表格型 Q-learning 里的 Q 表直接换成神经网络后，训练容易不稳定。所以要采用DQN

深度 Q 网络（Deep Q-Network, DQN）用神经网络表示 $\hat Q(s,a;w)$。

直接把连续采集的 transition 喂给普通 Q-learning 会遇到两类课件强调的问题：

- 相邻 transition 高度相关，连续梯度方向可能被同一小段 trajectory 主导；
- online network 每更新一次，下一轮 bootstrap target 也跟着变化，形成 moving target。

DQN 分别用 experience replay 和 fixed Q-targets 缓解这两个问题。



假设智能体连续与环境交互：

$(s_t,a_t,r_{t+1},s_{t+1})$
$(s_{t+1},a_{t+1},r_{t+2},s_{t+2})$
$(s_{t+2},a_{t+2},r_{t+3},s_{t+3})$

直接按这个顺序把数据喂给神经网络，会遇到两个主要问题。

---

 问题一：相邻 transition 高度相关

transition 就是一次状态转移经验：$(s_t,a_t,r_{t+1},s_{t+1})$

连续采集到的数据通常非常相似。

例如机器人连续三步可能是：

$x=1.00\rightarrow1.01\rightarrow1.02\rightarrow1.03$

游戏画面也只是人物向右移动了几个像素。

所以连续样本并不是独立随机的数据，而是高度相关的：$\text{样本}_t\approx\text{样本}_{t+1}$

如果网络连续训练这一小段相似数据，梯度方向可能连续被同一段轨迹主导。

例如连续几十个样本都来自地图左边，网络可能暂时把左边区域拟合得特别多，却忘掉或破坏其他区域的预测。

这和普通监督学习中通常希望：数据打乱后随机取 mini-batch 不太一样。

---
 
 Experience Replay 解决什么？

DQN 不立刻只用刚得到的 transition 训练，而是先把它保存进 replay buffer：

$D= \{ (s_i,a_i,r_{i+1},s_{i+1}) \}$

然后训练时，从 buffer 中随机抽一批：

$(s_j,a_j,r_{j+1},s_{j+1}) \sim D$

这样一批数据可能分别来自：

- 很早以前的 episode；
- 不同地图位置；
- 不同动作；
- 不同训练阶段。

因此把：交互顺序与训练顺序 解耦。

直观上就是：

> 智能体按照时间顺序生活，但神经网络把过去的经历打乱后复习。

这可以：

1. 降低相邻数据的相关性；
2. 一条经验可以重复利用，提高数据效率；
3. 让 mini-batch 中的数据覆盖更广。

所以：

$\boxed{\text{Experience Replay 主要处理“样本相关”问题}}$

---

第二个问题：moving target

Q-learning 的 target (target 就是当前模型输出要去逼近的目标值)：

$y_t= r_{t+1} + \gamma\max_{a'}\hat Q(s_{t+1},a';w)$

注意两个地方都用了同一个网络参数 w：

- 当前预测：$\hat Q(s_t,a_t;w)$
- 训练标签：$\hat Q(s_{t+1},a';w)$

于是网络一更新，预测会变；但与此同时，标签也跟着变。

这叫：moving target，移动目标

==备注：==

在普通监督学习里：$\text{预测 }\hat y=f(x;w)$

去逼近数据集给出的真实标签：y

而在强化学习中，真正想逼近的是：$Q^\pi(s,a)=\mathbb E[G_t\mid S_t=s,A_t=a]$

但这个真实 $Q^\pi(s,a)$通常没人能直接告诉我们，所以 RL 要根据与环境交互得到的奖励和下一状态，**临时构造一个 target**。


理想情况下，我们真正想要：

假设模型输出：$\hat Q(s_t,a_t;w)$

我们当然希望它等于真实价值：$Q^\pi(s_t,a_t)$

但真实价值是未来回报的期望：$Q^\pi(s_t,a_t) = \mathbb E[G_t\mid s_t,a_t]$

其中：$G_t = R_{t+1} +\gamma R_{t+2} +\gamma^2R_{t+3} +\cdots$

问题是：

> 我们不知道所有未来可能发生什么，也算不出这个期望。

所以只能用可观察的数据构造近似标签。


![[Pasted image 20260730141854.png]]

![[Pasted image 20260730141911.png]]

![[Pasted image 20260730141947.png]]



---



假设一条固定经验没有变化：

$(s_t,a_t,r=1,s_{t+1})$

折扣因子：γ=0.9

网络最开始对下一状态输出：$[2,6]$

因此 target 是：y=1+0.9×6=6.4

我们让当前预测朝 6.4 靠近。

但是更新网络参数以后，因为参数是共享的，下一状态的输出也可能变成：

$[2.5,7]$

尽管训练数据完全没变，target 立刻变成：

y=1+0.9×7=7.3

于是出现一种很别扭的情形：

> 模型一边追标签，标签一边被同一个模型推着往前跑。

像你追着一个人跑，但那个人的位置又由你的脚步决定。很容易震荡。

---

Fixed Q-targets 怎么解决？

DQN 维护两套网络。

	1:Online network

参数记为：w

它负责：

- 预测当前 Q 值；
- 选择动作；
- 每个训练步骤通过梯度下降更新。


	2:Target network

参数记为：$w^-$

它专门负责计算 target：$y_t= r_{t+1} + \gamma \max_{a'} \hat Q(s_{t+1},a';w^-)$

训练时只更新 online network：$\hat Q(s_t,a_t;w)$

target network 暂时冻结不动。

所以损失可以写成：

$L(w)= \frac12 \left[ y_t-\hat Q(s_t,a_t;w) \right]^2$

其中 target：$y_t= r_{t+1} + \gamma\max_{a'} \hat Q(s_{t+1},a';w^-)$

在一段时间内保持相对固定。

每隔若干步，再把 online network 参数复制给 target network：

$w^-\leftarrow w$

也可以使用缓慢软更新，但核心思想一样：

> 不让标签在每次梯度更新后立刻跟着乱跑。

因此：

Target Network 主要处理 moving target 

---

详细讲解：

### 6.2 Experience replay：把交互顺序与训练顺序解耦

*首次完整讲解：Lecture 4 §6.2「Experience replay：把交互顺序与训练顺序解耦」。*

![[lec4-experience-replay-p64.png|900]]

*来源：`lecture/lecture4post.pdf` 第 64 个物理页面。原图显示 transition 被写入 replay buffer，再随机抽样用于训练。*

经验回放（experience replay）把 interaction transition

$$
(s_t,a_t,r_t,s_{t+1})
$$

存入 replay buffer $D$，训练时**随机抽取**

$$
(s,a,r,s')\sim D.
$$

它有两个不同作用：随机抽样减弱相邻数据的时间相关性；重复抽取旧 transition 提高 data reuse。它也让训练数据成为过去多个 behavior policies 的混合，因此 DQN 的 Q-learning update 本身是 off-policy。

**自拟 buffer trace**：记第 $j$ 条 transition 为 $\tau_j=(s_j,a_j,r_j,s_{j+1})$。环境依次产生 $\tau_1,\tau_2,\tau_3,\tau_4$ 并按这个顺序写入 $D$；训练 minibatch 可以先抽到 $[\tau_4,\tau_1]$，下一次又抽到 $[\tau_2,\tau_4]$。交互历史没有被改写，但梯度看到的顺序已被打散，且 $\tau_4$ 被复用了两次。Replay 只是在经验数据上随机抽样，并不保证有限 buffer 中的样本真正独立同分布，也不会自动修复稀有 transition 从未被收集的问题。

### 6.3 Fixed Q-targets：暂时冻结标签生成器

*首次完整讲解：Lecture 4 §6.3「Fixed Q-targets：暂时冻结标签生成器」。*

令 $w$ 是正在更新的 online-network 参数，$w^-$ 是 target-network 参数。非终止 target 改为：

$$
y_i
=
r_i+\gamma\max_{a'}\hat Q(s_{i+1},a';w^-).
$$

梯度步骤只更新 $w$：

$$
w
\leftarrow
w
+
\alpha
\left[y_i-\hat Q(s_i,a_i;w)\right]
\nabla_w\hat Q(s_i,a_i;w).
$$

每隔 $C$ 个训练步骤同步一次：

$$
w^-\leftarrow w.
$$

多个更新因此面对近似固定的 target。课件理解检查的结论是额外保存一份网络参数会增加参数内存；这不等于整个训练的总计算量必然精确翻倍。

**自拟数值例子**：继续使用 $r=1$、$\gamma=0.9$。若冻结的 target network 在 $s'$ 输出 $[2,6]$，而 online network 对当前 $(s,a)$ 预测 4，则这段同步周期内

$$
y=1+0.9\times6=6.4.
$$

即使一次梯度更新让 online network 在同一个 $s'$ 的输出变成 $[2.5,7]$，计算后续 target 时仍使用 $w^-$ 的 $[2,6]$，所以标签仍是 6.4。只有执行 $w^-\leftarrow w$ 后，target 才跳到 $1+0.9\times7=7.3$。`fixed` 指同步间隔内固定，而不是训练全程永不变化。


### 6.4 DQN loss 与 terminal boundary

*首次完整讲解：Lecture 4 §6.4「DQN loss 与 terminal boundary」。*

Minibatch（小批量）是一次从 replay buffer 抽出的 $B$ 条 transition；它让一次参数更新平均多个样本梯度。Minibatch 中每个样本的 target 是：

$$
y_i
=
\begin{cases}
r_i, & s_{i+1}\text{ terminal},\\[4pt]
r_i+\gamma\max_{a'}\hat Q(s_{i+1},a';w^-), & \text{otherwise}.
\end{cases}
$$

> [!important] Transition 和 episode 不是同一层级
> Transition 是一步经验，通常写成 $(s_t,a_t,r_t,s_{t+1},done_t)$，例如 $(s_3,\text{向右},1,s_4,\mathrm{False})$；episode 则是从起点到终止状态的一整条轨迹。


训练最小化 minibatch MSE：

$$
L(w)
=
\frac1B\sum_{i=1}^{B}
\left[
y_i-\hat Q(s_i,a_i;w)
\right]^2.
$$

终止时删除的是 future bootstrap，不是即时奖励。Target network 产生 $y_i$，online network 产生 prediction 并接收梯度，这两个角色不能互换。

> [!example] 具体计算：terminal 与 non-terminal 样本如何组成 minibatch MSE
> 令 $B=2$。第一条 transition 到达 terminal，已知 $r_1=3$，online network 对已执行动作的 [online prediction](academic-term-lookup:online%20prediction) 为 $\hat Q(s_1,a_1;w)=2$。Terminal 后没有未来动作，因此 target 不含 bootstrap 项：
>
> $$
> y_1=r_1=3.
> $$
>
> 先计算 prediction 与 target 的残差，再平方：
>
> $$
> e_1=y_1-\hat Q(s_1,a_1;w)=3-2=1,
> \qquad
> \ell_1=e_1^2=1^2=1.
> $$
>
> 第二条 transition 没有终止。已知 $r_2=1$、$\gamma=0.9$、target network 给出的下一状态最大动作价值为 $6$，online prediction 为 $\hat Q(s_2,a_2;w)=4$。因此
>
> $$
> \begin{aligned}
> y_2&=1+0.9\times6=6.4,\\
> e_2&=y_2-\hat Q(s_2,a_2;w)=6.4-4=2.4,\\
> \ell_2&=e_2^2=2.4^2=5.76.
> \end{aligned}
> $$
>
> 单条样本得到的是 squared error；对两条样本取平均后，才得到这个 minibatch 的 mean squared error：
>
> $$
> L(w)=\frac{\ell_1+\ell_2}{B}
> =\frac{1+5.76}{2}
> =3.38.
> $$
>
> 按本节定义的 MSE，损失对两个 online predictions 的导数分别为
>
> $$
> \frac{\partial L}{\partial \hat Q_1}
> =\frac{2}{2}(2-3)=-1,
> \qquad
> \frac{\partial L}{\partial \hat Q_2}
> =\frac{2}{2}(4-6.4)=-2.4.
> $$
>
> 这两个信号只更新 online network；target network 在计算本次 target 时保持固定。第一条样本删除的只是 terminal 之后的 bootstrap 项，即时奖励 $r_1=3$ 仍完整保留。


### 6.5 DQN 训练流程

*首次完整讲解：Lecture 4 §6.5「DQN 训练流程」。*

前面分别介绍了 replay buffer 和 target network；DQN 训练流程就是把这两个稳定化机制接入 Q-learning。每个环境步骤依次完成：

1. 用 [online network](academic-term-lookup:online%20network) 选择动作并与环境交互；
2. 把新 transition 存入 replay buffer；
3. 从 buffer 随机抽取 minibatch；
4. 用 target network 计算较慢变化的 Q-learning targets；
5. 用这些 targets 更新 online network，并按周期同步 target network。

> [!summary] 一句话流程
> 环境采样 -> 写入 replay buffer -> 随机抽取 minibatch -> target network 生成标签 -> 更新 online network -> 周期性同步

四个组件的职责不能混在一起：

| 组件 | 做什么 | 为什么需要它 |
|---|---|---|
| Environment | 产生新 transition | 提供真实交互经验 |
| Replay buffer $D$ | 保存并随机重排经验 | 降低相邻样本相关性并复用数据 |
| Target network $w^-$ | 计算 bootstrap target | 减慢标签漂移 |
| Online network $w$ | 输出当前预测并接收梯度 | 真正被优化的 Q 网络 |

**一个训练步骤怎样运行？**

1. **选择动作并收集经验。** 在当前状态 $s_t$，根据 $\hat Q(s_t,\cdot;w)$ 的 epsilon-greedy 策略选择 $a_t$，观察 $r_t$、$s_{t+1}$ 和是否终止。
2. **写入并抽样。** 把 $(s_t,a_t,r_t,s_{t+1},\mathrm{done})$ 存入 $D$，再随机抽取 $B$ 条 transition。
3. **逐条计算 target。** 对 minibatch 中第 $i$ 条样本，令
   $$
   y_i=
   \begin{cases}
   r_i, & \text{若下一状态为 terminal},\\[4pt]
   r_i+\gamma\max_{a'}\hat Q(s_{i+1},a';w^-), & \text{否则}.
   \end{cases}
   $$

4. **只更新 online network。** 最小化 minibatch 中 $y_i$ 与 $\hat Q(s_i,a_i;w)$ 的均方误差；本步改变 $w$，不改变 $w^-$。
5. **按周期同步。** 训练步骤计数每达到 $C$ 的倍数，执行 $w^-\leftarrow w$；在两次同步之间，target network 保持冻结。

因此 DQN 不只是“用神经网络替换 Q-table”，而是一条职责明确的数据与优化管线。

**自拟一次循环 trace**：设当前是第 12 个训练步骤、$C=4$。Agent 先把新经验 $\tau_{12}$ 放入 buffer，再随机抽到一条 terminal 样本（奖励 2）和一条非终止样本（奖励 1，target-network 下一状态最大值 6，$\gamma=0.9$）；两个 target 分别是 2 和 6.4。Online network 用这两个误差更新 $w$。因为 $12$ 是 4 的倍数，本步末尾再同步 $w^-\leftarrow w$；第 13--15 步只更新 $w$，$w^-$ 保持不动。这个顺序说明“采集、存储、抽样、算 target、更新 online、按周期同步”各自发生在何处。




### 6.6 Atari DQN：一个网络同时输出全部动作价值

*首次完整讲解：Lecture 4 §6.6「Atari DQN：一个网络同时输出全部动作价值」。*

![[lec4-dqn-network-p72.png|900]]

*来源：`lecture/lecture4post.pdf` 第 72 个物理页面；图源在课件中标为 Mnih et al. (2015)。*

课件 Atari setting 把最近 4 帧像素堆叠成状态输入。卷积神经网络（convolutional neural network, CNN）是处理网格图像的参数化函数；这里它从相邻像素和连续帧中提取空间与运动特征，最后一次 forward pass 输出 18 个 joystick/button 动作的 $Q$ 值。图中不是“每个动作训练一个网络”，而是**一个共享网络输出一个动作价值向量**：

$$
\hat Q(s,\cdot;w)
=
\left[
\hat Q(s,a_1;w),\ldots,\hat Q(s,a_{18};w)
\right].
$$

行为时在这个向量上做 epsilon-greedy；训练时只取 batch 中实际动作 $a_i$ 对应的输出与 $y_i$ 比较。

**自拟三动作例子**：若某个状态的网络输出为 $[1.2,3.7,-0.4]$，则 $a_2$ 是贪心动作。取 $\varepsilon=0.1$ 时，$a_2$ 的行为概率为 $0.9+0.1/3\approx0.933$，其余动作各约为 0.033。若 replay 样本记录的实际动作是 $a_1$，该样本的 loss 只比较输出分量 1.2 与它的 target；但由于前面卷积层和隐藏层参数共享，反向传播后其他动作分量也可能随之变化。

### 6.7 消融结果：深网络本身并不够

*首次完整讲解：Lecture 4 §6.7「消融结果：深网络本身并不够」。*

![[lec4-dqn-ablation-p76.png|900]]

*来源：`lecture/lecture4post.pdf` 第 76 个物理页面。该表比较 linear、仅 deep network、fixed Q、replay 及二者组合。*

消融实验（ablation study）固定任务和大部分设置，只改变或组合某个组件，借此判断该组件对结果的贡献。这张表的课程结论不是“deep 一定胜过 linear”。例如 River Raid 中仅 deep network 的 1453 低于 linear 的 2345；加入 replay 后变为 4102，replay + fixed Q 后达到 7447。不同游戏的绝对分数不可横向比较，但同一游戏内的列对照支持两个判断：

1. 表示能力更强不自动保证稳定或高性能；
2. replay 与 fixed targets 是 DQN 算法的一部分，不是可有可无的工程装饰。

课件随后列出 Double DQN、Prioritized Replay 和 Dueling DQN 作为经典后续方向；本讲不展开它们的算法推导。


## 7. Assignment Readiness

### 7.1 已覆盖的前置知识

- Assignment 1 的 tabular Bellman backup、VI 和 PI 已在 Lecture 2 覆盖；本讲补全未知模型下从经验学习 $Q$ 的 control 视角。
- Assignment 2 中 tabular Q-learning、on/off-policy、replay buffer、fixed target network 和 terminal target 已覆盖。
- 现在可以独立手算 MC control、SARSA 和 Q-learning 的单步 target，并阅读基础 DQN training loop。

### 7.2 尚未覆盖的内容

- Policy gradient、baseline、importance sampling、PPO clipping 和 policy-induced distributions 属于后续 lecture。
- 本讲不提供 CNN 反向传播细节、DQN 超参数调优方案或现代 DQN variants 的完整实现。

### 7.3 Mastery evidence

当前只有 coverage 记录，没有新的 quiz、独立推导、实现或 assignment 测试结果。笔记完成不表示 Lecture 4 已掌握。

推荐验证顺序：先手算 §4.5 的 SARSA/Q-learning 对照，再写一个两状态 tabular 环境验证 terminal target，最后实现最小 replay-buffer sample 和 target-network synchronization。

## 8. 本讲必会公式

1. Epsilon-greedy（首次讲解：§3.2）：

   $$
   \pi_\varepsilon(a\mid s)
   =
   \begin{cases}
   1-\varepsilon+\varepsilon/|\mathcal A|,&a=g(s),\\
   \varepsilon/|\mathcal A|,&a\ne g(s).
   \end{cases}
   $$

2. MC action-value update（首次讲解：§3.4）：

   $$
   Q(s_t,a_t)
   \leftarrow
   Q(s_t,a_t)
   +\frac{G_t-Q(s_t,a_t)}{N(s_t,a_t)}.
   $$

3. Q-learning（首次讲解：§4.3）：

   $$
   Q(s_t,a_t)
   \leftarrow
   Q(s_t,a_t)
   +\alpha_t
   \left[
   r_t+\gamma\max_{a'}Q(s_{t+1},a')-Q(s_t,a_t)
   \right].
   $$

4. SARSA（首次讲解：§4.4）：

   $$
   Q(s_t,a_t)
   \leftarrow
   Q(s_t,a_t)
   +\alpha_t
   \left[
   r_t+\gamma Q(s_{t+1},a_{t+1})-Q(s_t,a_t)
   \right].
   $$

5. Function-approximation skeleton（首次讲解：§5.2，RL targets：§5.3--5.5）：

   $$
   w
   \leftarrow
   w
   +\alpha
   \left[y_t-\hat Q(s_t,a_t;w)\right]
   \nabla_w\hat Q(s_t,a_t;w).
   $$

6. DQN target（首次讲解：§6.3--6.4）：

   $$
   y_i
   =
   \begin{cases}
   r_i,&\text{terminal},\\
   r_i+\gamma\max_{a'}\hat Q(s_{i+1},a';w^-),&\text{otherwise}.
   \end{cases}
   $$

## 9. 容易混淆点

1. **$Q^\pi$ 与 $Q^*$**：前者评估固定策略，后者是最优动作价值。
2. **MC evaluation 与 MC control**：control 不只估值，还会改变之后采样数据的策略。
3. **$\varepsilon$ 的含义**：它是总随机探索概率；贪心动作仍分到 $\varepsilon/|\mathcal A|$。
4. **固定 $\varepsilon>0$ 与 GLIE**：固定值持续探索，但不会极限贪心。
5. **On-policy/off-policy 与 online/offline**：前者比较行为策略和目标策略，后者描述数据何时到来。
6. **SARSA 与 Q-learning**：SARSA 使用实际 $a_{t+1}$；Q-learning 使用 $\max_{a'}$。
7. **Terminal target**：进入终止状态的奖励保留，只有后续 bootstrap 归零。
8. **Semi-gradient**：target 数值可以依赖 $w_t$，但本步只对当前 prediction 分支求梯度。
9. **Deadly triad**：是三个因素组合的风险，不是任意单个因素必然导致发散。
10. **Replay 与 target network**：replay 重排并复用数据；target network 减慢标签漂移。
11. **DQN 输出**：一个网络通常一次输出所有离散动作的 $Q$ 值，不是每个动作一个独立网络。
12. **Mars Rover 概率冲突**：两个动作、$\varepsilon=1/3$ 时，按定义页唯一贪心动作概率是 $2/3$，不是第 82 页写出的 $5/6$。

## 10. 自测题

### 题目

1. 为什么 model-free control 更适合学习 $Q(s,a)$ 而不是只学习 $V(s)$？
2. 4 个动作、唯一贪心动作、$\varepsilon=0.2$ 时，各动作概率是多少？
3. 固定 $\varepsilon>0$ 为什么不满足 GLIE 的第二条？
4. 在 §4.3 数值例子中，为什么 Q-learning target 使用 6 而不是实际可能采样到的 2？
5. 对 terminal transition $(s,a,r,terminal)$，SARSA 和 Q-learning target 分别是什么？
6. Semi-gradient TD 为什么不对 target 分支反向传播？
7. Deadly triad 的三个因素分别引入什么风险？
8. Experience replay 除了降低相关性，还提供什么作用？
9. DQN 中 $w$ 与 $w^-$ 分别承担什么角色？
10. 第 76 页消融表为什么不能被解释为“神经网络越深，结果一定越好”？

<details>
<summary>查看答案</summary>

1. 未知模型时，$Q(s,a)$ 直接比较候选动作；只有 $V(s)$ 无法在不查询 dynamics 的情况下评价当前改选另一个动作。
2. 贪心动作 $0.85$，其余三个动作各 $0.05$。
3. 固定正 $\varepsilon$ 让非贪心动作始终保留正概率，因此行为策略不会收敛到纯 greedy。
4. Q-learning 的 target policy 是 greedy；behavior policy 可以探索，所以 target 与实际下一动作分离。
5. 两者都是 $y=r$，因为终止后 future value 为 0。
6. 它把用 $w_t$ 计算出的 target 当作本步固定标签，只对当前 prediction 求梯度；这正是 semi-gradient 的定义。
7. Function approximation 带来共享参数与投影误差；bootstrap 让 target 依赖估计；off-policy 让训练分布与目标策略不一致。三者可能形成误差放大反馈。
8. 同一 transition 可以被多次训练，提高 data reuse，并让训练顺序与交互顺序解耦。
9. $w$ 是被优化的 online 参数；$w^-$ 是暂时冻结、用于生成 target 的参数，周期性由 $w$ 同步。
10. 同一表中仅 deep network 在部分游戏低于 linear；性能取决于表示、数据分布和稳定化机制的组合，不能只按深度排序。

</details>

## 11. 本讲小结

1. Model-free control 把 $Q$ estimation、exploration 和 policy improvement 放进同一学习循环。
2. Epsilon-greedy 提供最小探索机制；GLIE 同时要求无限探索与极限贪心。
3. MC control 使用完整 $G_t$；SARSA 使用实际下一动作；Q-learning 使用下一状态最大动作价值。
4. On/off-policy 的判断依据是行为策略与 target policy 是否相同，不是数据是否在线产生。
5. 函数逼近用共享参数换取泛化，也让一个样本影响许多状态动作对。
6. Semi-gradient TD 将 bootstrap target 在当前更新中视为固定标量。
7. Function approximation、bootstrapping、off-policy 的交集构成 deadly triad 风险。
8. DQN 用 replay buffer 重排并复用经验，用 target network 减慢 target 漂移，再从像素输出动作价值向量。

## 12. 延伸阅读

### 12.1 经典基础

- Watkins & Dayan, “Q-learning,” *Machine Learning*（1992）：tabular Q-learning 的经典收敛结果，对应 §4.3。
- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed.（2018），§§5.2--5.4、6.4、6.5、6.7：课件指定的 MC control、TD control 与相关背景。
- Tsitsiklis & Van Roy, “An Analysis of Temporal-Difference Learning with Function Approximation”（1997）：函数逼近 TD 稳定性分析的经典来源。
- Mnih et al., “Human-level control through deep reinforcement learning,” *Nature*（2015）：课件 DQN 网络和 Atari 结果的原始来源。

### 12.2 前沿动态

本讲不额外列出前沿条目（截至 2026-07-18 核实）。课程目标是先掌握 tabular control、函数逼近的稳定性风险和原始 DQN 的 replay/fixed-target 机制；现代扩展应在这些对象和边界清楚后再加入。
