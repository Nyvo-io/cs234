---
title: CS234 Lecture 10 - Fast Reinforcement Learning
aliases:
  - CS234 Lec10
tags:
  - cs234
  - reinforcement-learning
  - multi-armed-bandit
  - ucb
  - value-alignment
---

# CS234 Lecture 10 Notes: Fast Reinforcement Learning

来源：`lecture/lecture10post.pdf`，CS234 Winter 2026，Emma Brunskill；共 41 个物理 PDF 页面。第 17--41 页为 Wanheng Hu 的 value alignment guest lecture。

笔记规范：`cs234-rl-tutor v2`。覆盖清单只表示课件内容已经写入，不表示学习者已经掌握。

外部资料核验日期：2026-08-13（延伸阅读中的经典来源按本讲课件引用核对）。

## 0. 本讲覆盖清单

- [x] 第 1--3 页：multi-armed bandit、UCB、regret 复习题及答案；写入 §1 和 §8。
- [x] 第 4--7 页：课程位置、Covid testing 例子、bandit/PAC/Bayesian bandit/Thompson sampling 目录；写入 §1--§2。
- [x] 第 8--9 页：bandit regret notation recap 和 UCB1；写入 §2--§3。
- [x] 第 10--16 页：UCB regret proof sketch、good event、union bound、抽样次数上界和 theorem conclusion；写入 §3。
- [x] 第 17--18 页：guest lecture 开场、讲者信息、value alignment 问题；写入 §5。
- [x] 第 19--24 页：paperclip、真实案例、三种 alignment 解释、autonomy/paternalism 和总结；写入 §5。
- [x] 第 25--26 页：sycophancy case study 及 RLHF 反思问题；写入 §6。
- [x] 第 27--28 页：agentic AI case study、何时自动行动/询问/拒绝；写入 §7。
- [x] 第 29--41 页：guest lecture 的结尾视觉页、people other than the user、讨论信息；写入 §7--§8。

目录中的 PAC、Bayesian bandits、Thompson sampling 是 Lecture 10 的教学方向，但这份 post deck 在第 9 页 UCB 后直接进入 proof sketch，随后切换到 guest lecture；本笔记不虚构未展示的 Bayesian 算法或 PAC 定理。

## 1. 本讲主线

与 Lecture 9 的关系：Lecture 9 建立了 bandit regret、optimism 和 UCB 的基本形式，但其较短的 UCB proof 在课后被发现有错误。本讲先复习 UCB，再用 textbook Theorem 7.1 的 proof sketch 修正“次优臂被拉多少次”的论证；后半转入 guest lecture，讨论当 reward 或目标本身来自人的价值判断时，技术上的“学得更快”仍然不等于“做得符合人类真正想要的事”。

**本讲路线图**

1. 先固定 bandit notation 和 UCB1，明确 regret 是 gap 与 pull count 的加权和。
2. 用高概率 good event 分离“最优臂的置信上界失败”和“次优臂仍显得过于乐观”两类错误。
3. 通过 contradiction、union bound 和 sub-Gaussian concentration 控制次优臂的期望拉取次数，得到对数级 regret bound。
4. 再把技术问题放回 alignment 场景：目标可以解释为用户意图、用户偏好或用户最佳利益，而三者并不总是一致。
5. 用 sycophancy 和 agentic AI 讨论 reward 设计的外部影响、用户自主性与何时需要系统增加摩擦。


**count decomposition = 把问题拆开算**  
**contradiction = 用矛盾证明 good event 下不会继续选错**  
**union bound = 把很多坏事件的概率加起来**  
**concentration = 证明经验均值严重偏离真实均值的概率很小


## 2. Bandit 与 UCB proof 的问题设置

与 Lecture 9 的关系：这里沿用上一讲已经建立的 bandit、regret 和 UCB 记号，把它们整理成 Lecture 10 证明所需的统一设置。

**本节路线图**

1. 先回顾动作均值、最优动作和 regret 的 gap--count 分解。
2. 写出 UCB1 的经验均值加 bonus 选择规则。
3. 用 Covid testing 例子说明，现实问题通常比证明中的 stochastic bandit 更复杂。

### 2.1 Notation recap

*首次完整讲解：Lecture 9 §2.1「形式化定义」与 §4.1「单步 regret 与总 regret」。本节只补充：这些记号如何进入 Lecture 10 的 UCB proof。*

多臂 bandit 中，动作集合为 $\mathcal A$，动作 $a$ 的未知奖励均值为

$$
Q(a)=\mathbb E[r\mid a].
$$

最优均值和最优动作是

$$
V^*=\max_{a\in\mathcal A}Q(a),
\qquad
a^*\in\arg\max_{a\in\mathcal A}Q(a).
$$

每一轮都是在那几个动作里选
第 $t$ 轮的期望机会损失和 前 $T$ 轮总 regret 分别为

$$
\ell_t=\mathbb E\left[V^*-Q(a_t)\right],
$$
这里取期望是因为 动作可能是随机的


$$
L_T
=
\mathbb E\left[
\sum_{t=1}^{T}\left(V^*-Q(a_t)\right)
\right].
$$

令 $N_T(a)$ 是前 $T$ 轮选择动作 $a$ 的次数，$\Delta_a=V^*-Q(a)$，则

$$
L_T=\sum_{a\in\mathcal A}\mathbb E[N_T(a)]\Delta_a.
$$

这条分解告诉我们，UCB proof 不需要直接证明“每一步都选到最优动作”；它只需证明每个 gap 为正的次优动作不会被拉太多次。


### 2.2 UCB1

*首次完整讲解：Lecture 9 §7.1「乐观原则」与 §8.2「UCB1 形式」。本节只补充：Lecture 10 proof 采用的统一记号和后续抽样次数分析。*

Lecture 9 的 UCB 结构是经验均值加置信 bonus。Lecture 10 复习页给出的形式可统一写成

第t轮选择哪个臂：
$$
a_t
=
\arg\max_{a\in\mathcal A}
\left[
\widehat Q_{t-1}(a)
+
\sqrt{\frac{2\log(1/\delta)}
{N_{t-1}(a)}}
\right],
$$

这里使用 $1$-sub-Gaussian 奖励约定；$\delta$ 是置信失败概率，具体 proof 中会取为随 horizon 变化的值。未访问动作仍需先初始化，否则 bonus 的分母为零。


### 2.3 Covid testing：复杂 bandit setting 的预览

课件展示 Bastani et al. 在 Covid testing 中的例子：系统根据旅客提交的信息，把是否检测、检测地点和资源分配结合起来，并在实验室结果延迟 $24$--$48$ 小时后更新数据。课件将其概括为 **nonstationary, contextual, batched bandit with delayed feedback and constraints**；

它不是本讲证明所用的简单 stochastic K-armed bandit，而是说明真实决策通常同时包含上下文、批处理、延迟反馈、非平稳性和约束。

![[lec10-covid-bandit-p6.png|900]]

*图：Lecture 10 物理 PDF 第 6 页的 Covid testing 流程图；来源：`lecture/lecture10post.pdf` 第 6 页。图中展示了 contextual 输入、检测/不检测分支、延迟实验室结果和集中数据库回流；这是课件原图，不是本笔记重绘。*

这张图的作用是提醒：UCB 的清晰理论假设是一个起点，不是对所有现实实验设计的完整描述。


## 3. UCB regret proof sketch

与第 2 节的关系：第 2 节给出了 UCB 如何选择动作；本节进一步解释为什么在相应假设下，次优臂的选择次数只会按对数级增长。

**本节路线图**

1. 先把 regret 转化为每个次优臂的 pull-count 上界。
2. 用 good event 和 contradiction 说明，充分采样后次优臂不能继续胜过最优臂。
3. 用 union bound 和 concentration 控制 good event 失败的概率，再选择阈值得到最终 bound。


### 3.1 定理、假设与目标

课件按照 Lattimore and Szepesvári *Bandit Algorithms* 的 Theorem 7.1 讲解。为便于读公式，设有 $K$ 个臂、horizon 为 $n$，奖励属于 $[0,1)$ 并满足课件采用的 sub-Gaussian 假设；不妨令 $a^*=a_1$。对每个次优臂 $a_i$ 定义

$$
\Delta_i=Q(a_1)-Q(a_i)>0.
$$

课件定理在 $\delta=1/n^2$ 的设置下给出如下结构性上界：

$$
\operatorname{Regret}_n
\le
3\sum_{i=1}^{K}\Delta_i
+
\sum_{i:\Delta_i>0}
\frac{16\log n}{\Delta_i}.
$$

根据这个公式，可以得到量级：
$\mathrm{Regret}_n = O\left(\sum_{i:\Delta_i>0}\frac{\log n}{\Delta_i}\right)$


常数取决于 UCB bonus 和 concentration convention；

本讲重点是 $O(\log n/\Delta_i)$ 的 每臂抽样次数，以及把它乘以 $\Delta_i$ 后得到的对数级 regret。课件页中有一个索引笔误，下面统一用 $\Delta_i$ 表示 gap，用 $N_n(a_i)$ 表示 horizon 内的拉取次数。

意思是：

- regret 随时间 n 只增长成对数级
- 对每个次优臂，贡献大约和 logn/Δi​ 成正比


### 3.2 先把 regret 问题改成 pull-count 问题

因为

$$
\operatorname{Regret}_n
=
\sum_{i=1}^{K}\Delta_i\,\mathbb E[N_n(a_i)],
$$

证明的局部目标是：对每个次优臂 $a_i$ 找到一个上界

$$
\mathbb E[N_n(a_i)]
\le
3+\frac{16\log n}{\Delta_i^2}.
$$

乘回 $\Delta_i$ 后，就得到 该臂对 regret 的贡献不超过 $3\Delta_i+16\log n/\Delta_i$。

这一步是 proof 的主线：不是逐轮分析 regret，而是控制“一个次优臂还能被相信多久”。

在ucb算法眼里，这个选项当前的估计值不是最优的，但因为置信区间，我还是可能选你。还能被相信多久，指的是 还能有多少轮看起来“有可能是最优的”，从而继续被选择

因为随着N(ai​) 越大，我们对这个臂越了解，bonus也就越小


### 3.3 Good event：两种置信失败

这里真正开始证明：

选择一个暂时未知的阈值 $u_i$，表示我们希望次优臂 $a_i$ 至少被充分采样到的次数。定义一个 good event $G_i$，它要求两件事同时成立：

1. 最优臂 $a_1$ 的 UCB 在整个 horizon 内没有跌到真实均值 $Q(a_1)$ 以下（最优臂别掉下去）；
2. 次优臂 $a_i$ 在被采样 $u_i$ 次后，其 经验均值加 bonus 已经低于 $Q(a_1)$ （次优臂别太乐观）。

用符号表达为

$$
G_i=
\left\{
Q(a_1)<\min_{1\le t\le n}U_t(a_1)
\right\}
\cap
\left\{
\widehat Q_{u_i}(a_i)
+
\sqrt{\frac{2\log(1/\delta)}{u_i}}
<
Q(a_1)
\right\}.
$$

在 $G_i$ 发生时，$a_i$ 在第 $u_i$ 次以后不应再被 UCB 选中：若它在某一步仍被选中，则它的 UCB 至多低于 $Q(a_1)$，而最优臂的 UCB 高于 $Q(a_1)$，与“选择最大 UCB”矛盾。


### 3.4 Count decomposition 与 contradiction

把 $G_i$ (good event )是否发生 作为指示变量，可以分解期望拉取次数：

把“第 i 个次优臂总共会被拉多少次”分成两种情况来分析

$$
\mathbb E[N_n(a_i)]
=
\mathbb E[\mathbf 1(G_i)N_n(a_i)]
+
\mathbb E[\mathbf 1(G_i^c)N_n(a_i)]
\le
u_i+nP(G_i^c).
$$

第一项不超过 $u_i$ 的理由是 contradiction：如果在 $G_i$ 成立时拉取 $a_i$ 超过 $u_i$ 次，那么存在某一轮，$a_i$ 已经有 $u_i$ 个样本却仍被选中 （因为good event 的定义决定了 ai​ 一旦拉到 ui​ 次以后，就不该再被选中）；由 good event 的两条不等式，它的 UCB 小于最优臂的 UCB，不可能成为 argmax。

$G_i^c$就是：

> Gi​ 没有发生。有一个次优臂在ui次之后，他仍有可能被选中

也就是 **bad event**。

$\mathbf 1(G_i^c)$， 带个1，就是指示函数

它的定义：
$\mathbf 1(G_i^c)=\begin{cases}1,&G_i^c\text{ 发生}\\0,&G_i^c\text{ 不发生}\end{cases}$


所以上面的式子进行拆解：

$\mathbf 1(G_i)N_n(a_i)\le u_i$

第二部分：
因为：$N_n(a_i)\le n$
所以：

$\mathbf 1(G_i^c)N_n(a_i)\le n\,\mathbf 1(G_i^c)$

取期望：

$\mathbb E[\mathbf 1(G_i^c)N_n(a_i)]\le n\mathbb E[\mathbf 1(G_i^c)]$

而指示变量有一个非常重要的性质：

$\mathbb E[\mathbf 1(G_i^c)]=P(G_i^c)$

所以：

$\mathbb E[\mathbf 1(G_i^c)N_n(a_i)]\le nP(G_i^c)$


$P(G_i^c)$
意思是：

> **事件 Gic​ (bad event) 发生的概率。**

所以不等式的意思就是：
>	**正常情况下**，次优臂在 ui​ 次以后就被识别出来了；
>	
>	**极少数倒霉情况下**，置信区间失效，那它最多可能被拉 n 次，但这种事情发生概率只有 P(Gic​)。


因此剩下的工作就是控制 $P(G_i^c)$， 要把它压的很小 。它最多来自两个事件：

- 最优臂某一轮的 UCB 低于真实均值；
- 次优臂采样 $u_i$ 次后仍然显得过于乐观。

$$
\boxed{G_i^c\subseteq\underbrace{\{\exists t\le n:Q(a_1)>U_t(a_1)\}}_{\text{最优臂被低估}}\;\cup\;\underbrace{\{\text{次优臂拉 }u_i\text{ 次后仍然过于乐观}\}}_{\text{次优臂被高估}}}
$$

### 3.5 Union bound 控制 最优臂事件

控制第一个坏事件：最优臂在某一轮被低估了

对每一轮的最优臂 UCB 失败事件使用 union bound：

$$
P\left(
\exists t\le n:
Q(a_1)>U_t(a_1)
\right)
\le
\sum_{t=1}^{n}
P\left(Q(a_1)>U_t(a_1)\right).
$$

理解：

假设第 t 轮，最优臂 a1​ 的 UCB 出错了，定义这个事件为：
$E_t=\{Q(a_1)>U_t(a_1)\}$

但我们不止跑一轮，而是跑n轮：

我们现在关心：

> **这 n 轮里面，只要有任何一轮 UCB 失败，概率是多少？**

“至少有一次失败”写成：
$E_1\cup E_2\cup\cdots\cup E_n$
也就是图里的：

$\exists t\le n:Q(a_1)>U_t(a_1)$

其中 ∃ 的意思是：

> **存在至少一个**


Union bound ：

$P(E_1\cup E_2\cup\cdots\cup E_n)\le P(E_1)+P(E_2)+\cdots+P(E_n)$

->
$$
P\left(\bigcup_i E_i\right)\le\sum_iP(E_i)
$$

如果每个时间点的置信失败概率被安排为足够小的量 $\delta$，则整体失败概率至多为 $n\delta$。这正是 

$$
P\left(\bigcup_{t=1}^nE_t\right)\le\sum_{t=1}^nP(E_t)\le n\delta
$$

的应用：同时要求很多置信界成立时，必须为多事件总失败概率留出预算。

---

所以本节最开始的那个不等式：

左边：

> **整个 n 轮中，至少有一轮最优臂的 UCB 失效。**

右边：

> **把第 1、2、……、n 轮各自的失效概率全部加起来。**

如果每一轮都保证：

$P(Q(a_1)>U_t(a_1))\le\delta$

那么：

$P(\text{整个 }n\text{ 轮至少失败一次})\le n\delta$

我们通过它得知了， 把每一轮失败概率设计得到底要多小，才能使得它们全部加起来仍然很小

比如可以设计让每轮失败概率变成 1/$n^2$

==接回你前面学的 concentration bound==

我们原来有：

$P\left(Q(a_1)>U_t(a_1)\right)\le\exp\left(-\frac{N_t(a_1)\eta_t^2}{2\sigma^2}\right)$
**sub-Gaussian concentration inequality**（集中不等式）

现在我们**主动要求**右边足够小，比如：

$\exp\left(-\frac{N_t(a_1)\eta_t^2}{2\sigma^2}\right)=\frac{1}{n^2}$

然后反过来解 ηt​。

得到一个足够大的 confidence bonus：​

$\eta_t=\sigma\sqrt{\frac{4\log n}{N_t(a_1)}}$

然后设置：

$U_t(a_1)=\hat Q_t(a_1)+\eta_t$


---

所以现在你应该能看到整条逻辑了：

想让整个 n 轮失败概率小​

↓

先决定每轮允许：
$\delta\approx\frac{1}{n^2}$

↓

通过 concentration inequality 反解：​
$\eta_t$

↓

这个 ηt​ 就成为 UCB 的 bonus：

$U_t(a)=\hat Q_t(a)+\mathrm{bonus}_t(a)$

↓

于是每一轮：

P($E_t$)≤ 1/ $n^2$

↓

最后 union bound：

$P\left(\bigcup_{t=1}^nE_t\right)\le\sum_{t=1}^nP(E_t)\le\frac1n$



### 3.6 Concentration 控制次优臂事件

控制第二个坏事件：次优臂在被采样 ui​ 次之后，居然还看起来像最优

因为 bonus 会随着采样次数增加而下降，所以我们可以通过把 ui​ 取得足够大，从而控制这个坏事件




对第二个事件，通过令 $u_i$ 足够大，使 bonus 小于 gap 的一部分, 也就是 $(1-c)\Delta_i$。沿用课件的参数化，取任意 $c\in(0,1)$，要求

$$
\Delta_i-
\sqrt{\frac{2\log(1/\delta)}{u_i}}
\ge
c\Delta_i,
\qquad\text{即}\qquad
\sqrt{\frac{2\log(1/\delta)}{u_i}}
\le
(1-c)\Delta_i.
$$
怎么得到的：

$\operatorname{bonus}(u_i)=\sqrt{\frac{2\log(1/\delta)}{u_i}}$

$\Delta_i=V^*-Q(a_i)$

正常来说，随着 ai​ 被采样越来越多，它的 UCB 应该慢慢降下来，我们最终应该发现：

> “哦，ai​ 确实比最优臂差。”

但是坏情况是：

> **ai​ 已经采样 ui​ 次了，它的 UCB 居然还高到可以和最优臂竞争。**

也就是类似：
$U_{u_i}(a_i)\ge V^*$

UCB 是：
$U_{u_i}(a_i)=\hat Q_{u_i}(a_i)+\sqrt{\frac{2\log(1/\delta)}{u_i}}$

为了简洁，把 bonus 写成：
​
$b_i=\sqrt{\frac{2\log(1/\delta)}{u_i}}$

如果次优臂还能够“看起来和最优臂一样好”：

$\hat Q_{u_i}(a_i)+b_i\ge V^*$

而：
$V^*=Q(a_i)+\Delta_i$

所以：
$\hat Q_{u_i}(a_i)+b_i\ge Q(a_i)+\Delta_i$

移项：

$\hat Q_{u_i}(a_i)-Q(a_i)\ge\Delta_i-b_i$

现在我们想降低 它 的概率

---


我们想用 concentration inequality， 但集中不等式擅长控制这种事件：

P(Q^​ui​​(ai​)−Q(ai​)≥ 某个正数 )

我们通过设 Δi​−bi​≥cΔi ，

$\boxed{P\left(\hat Q_{u_i}(a_i)-Q(a_i)\ge\Delta_i-b_i\right)\le P\left(\hat Q_{u_i}(a_i)-Q(a_i)\ge c\Delta_i\right)}$​

所以我们通过处理右边这个概率，去利用集中不等式处理它，从而达到缩小左边这个概率的目的

为了让这个“某个正数”足够明显，我们人为选一个：$c\in(0,1)$

然后希望：

$\Delta_i-b_i\ge c\Delta_i$

这样一来，刚才那个坏事件就必然导致：

$\hat Q_{u_i}(a_i)-Q(a_i)\ge c\Delta_i$

而这个事件的概率就可以用 concentration inequality 压下去。

---

那么得到 “次优臂的经验均值 加 bonus 仍 高于最优均值”会蕴含

$$
\widehat Q_{u_i}(a_i)-Q(a_i)
\ge
c\Delta_i.
$$
理解：

如果在采样 ui​ 次后，次优臂仍然满足：

$\hat Q_{u_i}(a_i)+\sqrt{\frac{2\log(1/\delta)}{u_i}}\ge Q(a_1)$

又因为：​$Q(a_1)=Q(a_i)+\Delta_i$

以及 bonus 已经满足

$\sqrt{\frac{2\log(1/\delta)}{u_i}}\le (1-c)\Delta_i$

那么就能推出：

$\hat Q_{u_i}(a_i)-Q(a_i)\ge c\Delta_i$

它表示：

> 次优臂要想在 ui​ 次后还“看起来很好”，
>   
> 那它的经验均值必须被高估至少 cΔi​。

而这正是 concentration inequality 擅长控制的事情。



利用 sub-Gaussian concentration，可得
$$
P\left(
\widehat Q_{u_i}(a_i)-Q(a_i)
\ge
c\Delta_i
\right)
\le
\exp\left(
-\frac{u_i c^2\Delta_i^2}{2}
\right)
$$

在课件的 $\sigma^2=1$ 约定下成立。这里的含义很具体：
样本数越多，次优臂的经验均值仍然高到足以超过最优臂的概率指数下降。

所以只要 ui​ 够大，这个概率就会非常小。

---

之前我们学sub-Gaussian concentration inequality： 

对于 σ-sub-Gaussian reward，采样 u 次后的经验均值满足：

$P\left(\hat Q_u(a)-Q(a)\ge\eta\right)\le\exp\left(-\frac{u\eta^2}{2\sigma^2}\right)$

把这两个代进去：​
$u=u_i$  $\eta=c\Delta_i$

就可以得到上面的不等式


### 3.7 选择阈值并得到最终 bound

到底把 ui​ 设成多大，才能保证这个次优臂被探索 ui​ 次以后，它的 UCB 已经不可能压过最优臂？


上一小节要求

$$
\sqrt{\frac{2\log(1/\delta)}{u_i}}
\le
(1-c)\Delta_i
$$

可取

$$
u_i
\ge
\frac{2\log(1/\delta)}
{(1-c)^2\Delta_i^2}.
$$

把这个阈值代回 [count decomposition](academic-term-lookup:count%20decomposition)。按课件的记号取 $c=1/2$、$\delta=1/n^2$，并计入 整数取整与失败项的常数，可得到

$$
\mathbb E[N_n(a_i)]
\le
3+\frac{16\log n}{\Delta_i^2}.
$$

因为：
$\boxed{u_i\ge\frac{16\log n}{\Delta_i^2}}$

>	对次优臂 ai​，大概采样这么多次以后，统计证据已经足够强，可以把它和最优臂区分开

前面我们有：
$\mathbb E[N_n(a_i)]\le u_i+nP(G_i^c)$

ui​ 是“拉取次数”，必须是整数，所以实际取：

$u_i=\left\lceil\frac{16\log n}{\Delta_i^2}\right\rceil$

$\lceil x\rceil\le x+1$

所以：

$u_i\le\frac{16\log n}{\Delta_i^2}+1$

这里就已经出来了第一个：+1

再看 bad event：P(Gic​)

Gi​ 要求两个好事件同时成立，所以 Gic​ 意味着至少有一个坏事件：

1. 最优臂的 UCB 在某个时间点失败；
2. 次优臂采了 ui​ 次以后仍然过于乐观。

用 union bound：

$P(G_i^c)\le P(\text{最优臂失败})+P(\text{次优臂失败})$


最优臂失败概率

每个时刻失败概率至多 δ，一共有 n 个时刻：

$P(\text{最优臂某时刻失败})\le n\delta$

取：
$\delta=\frac1{n^2}$

得到：
$n\delta=\frac1n$

 次优臂失败概率

concentration 又可以把它控制到大约：

$P(\text{次优臂失败})\le\delta=\frac1{n^2}$

所以：
​$P(G_i^c)\le\frac1n+\frac1{n^2}$

---

记得原式里不是单纯的 P(Gic​)，而是：

$nP(G_i^c)$

因此：
$nP(G_i^c)\le n\left(\frac1n+\frac1{n^2}\right)$

得到：

$nP(G_i^c)\le1+\frac1n\le2$

所以 bad event 贡献最多：

+2
​
所以合成了+3

---

于是

$$
\operatorname{Regret}_n
\le
3\sum_i\Delta_i
+
16\log n\sum_{i:\Delta_i>0}\frac1{\Delta_i}.
$$

因为：

3.2 已经知道：

$\mathrm{Regret}_n=\sum_i\Delta_i\mathbb E[N_n(a_i)]$

现在把刚刚的 bound 塞进去：

Regretn​≤i∑​Δi​(3+Δi2​16logn​)

$\mathrm{Regret}_n\le\sum_i\Delta_i\left(3+\frac{16\log n}{\Delta_i^2}\right)$

展开：​

$\mathrm{Regret}_n\le3\sum_i\Delta_i+16\log n\sum_{i:\Delta_i>0}\frac1{\Delta_i}$

这就正好回到了 3.1 一开始说我们要证明的定理。



> [!example] 结果解释：gap 如何改变探索成本
> 若两个次优臂的 gap 分别为 $\Delta_1=0.5$ 和 $\Delta_2=0.1$，忽略常数项时，抽样次数上界中的主项分别与 $1/0.5^2=4$ 和 $1/0.1^2=100$ 成正比；乘回 gap 后，它们对 regret 的主项分别与 $1/0.5=2$ 和 $1/0.1=10$ 成正比。
>
> 因此更接近最优臂的动作需要更多数据才能排除，且在该理论 bound 中更难造成误判。这不是说 gap 小的动作“坏得更多”，而是说它更难与最优动作区分。

### 3.8 Proof 的边界

这份 proof sketch 依赖 stochastic bandit、独立奖励、集中界和 UCB 的具体 bonus。它不直接证明 contextual、nonstationary、batched、delayed-feedback bandit 都有同样的 bound；Covid testing 图正好说明了这些额外结构会改变分析问题。Lecture 10 也没有在 post deck 中展开 PAC 或 Bayesian regret 的完整理论。



## 4. 复习题与技术检查

### 4.1 开头复习题

课件把以下五条陈述判为 true：

1. 最小化 regret 等价于最大化累计期望 reward。
2. 忽略常数后，UCB 选择经验均值加一个随 $1/\sqrt{N_t(a)}$ 衰减的 bonus。
3. 把 bonus 改成含 $\log(t/\delta)$ 的形式，仍可能让最优臂被选择得更多。
4. 固定 bonus $b=5$ 虽然乐观，但不能随不确定性下降，可能导致线性 regret。
5. $K$ 臂 bandit 可以看成一个单状态、$K$ 个动作的 MDP。

第四条尤其重要：optimism 不是“加一个很大的常数”就足够；bonus 必须随着数据量反映不确定性，否则算法可能永远过度探索某些动作。

### 4.2 自测题

1. UCB proof 为什么先要界定 $\mathbb E[N_n(a_i)]$，再回到 regret？
2. $G_i$ 中的两类条件分别排除了什么错误？
3. 为什么 union bound 会带来一个与 horizon $n$ 相关的失败概率项？
4. 在 $u_i\propto 1/\Delta_i^2$ 时，为什么 regret 主项变成 $1/\Delta_i$？
5. 固定 bonus 和 UCB bonus 的关键差异是什么？
6. value alignment 的三种解释是什么？它们为什么不等价？
7. sycophancy 可能是哪一种 alignment 目标被过度优化的结果？为什么还不能只凭这个标签判断所有原因？
8. agentic AI 在“自动行动、询问用户、拒绝或抵抗”之间如何做选择？

<details>
<summary>查看答案</summary>

1. gap--count 分解把 regret 写成每个次优臂的 gap 乘以选择次数；控制选择次数是更直接的局部目标。
2. 第一类条件保证最优臂的 UCB 没有被偶然低估；第二类条件保证次优臂充分采样后不会仍然看起来超过最优臂。
3. 需要同时控制多个时间点的置信界，事件并集的概率由各项失败概率之和上界。
4. 采样次数主项是 $1/\Delta_i^2$，每次错误选择的损失是 $\Delta_i$，相乘后得到 $1/\Delta_i$。
5. UCB bonus 随样本数下降；固定 bonus 不表达不确定性变化，可能造成线性 regret。
6. 可以对齐用户的 intended instructions、revealed preferences，或客观上对用户最有利的结果；意图、偏好和最佳利益可能发生冲突。
7. 它可能反映系统过度拟合用户当下的赞同反馈或 raters 的“礼貌/有帮助”评价，但具体原因还需要看数据、奖励标准和部署情境。
8. 需要同时考虑用户意图、偏好、长期利益、风险、他人影响与自主性；课件把它作为设计问题提出，没有给出单一算法答案。

</details>

## 5. Value alignment：reward 之外的目标问题

与前半的关系：UCB 假设 reward 分布是一个可以估计的外部对象；guest lecture 追问一个更早的问题：我们写下的 reward 到底代表谁的目标、哪些未说出口的约束，以及“做得更好”由谁来判断？

**本节路线图**

1. 用 paperclip maximizer 和课件中的现实案例说明，形式上清楚的目标仍可能遗漏背景约束。
2. 区分用户意图、显露偏好和用户最佳利益三种 alignment 解释。
3. 讨论自主性与 paternalism，说明“保护用户”也可能与让用户自己选择发生冲突。

本段 guest lecture 由 Wanheng Hu 主讲；课件介绍她在 Stanford 的 EIS 和 HAI 从事博士后研究，博士阶段来自 Cornell 的 Science and Technology Studies。她的研究问题包括：医疗 AI 是否只是“数字化医生”，以及我们应当在什么意义上信任它。下面的概念整理保留课件的问题意识，不把这些研究问题扩写成额外结论。

### 5.1 价值错位的基本例子

guest lecture 用 Bostrom 的 paperclip maximizer 作为思想实验：如果系统唯一明确目标是最大化 paperclip 数量，它可能把地球甚至更大范围的资源都转成 paperclip。这里的重点不是预测某个具体系统一定会这样做，而是说明一个形式上清楚的目标，可能遗漏人类默认的约束、背景目的和价值判断。

课件随后用两个简短的现实案例标题——“from boats to roads”和“from entertainment to treatment”——提醒听众：同一个技术或资源在不同目标下可能被重新定义用途。幻灯片没有在本讲展开这些案例的细节，因此这里只保留它们作为“目标与背景约束可能错位”的提示。

人类通常把很多条件当作不言自明：安全、资源边界、他人利益、长期可持续性和任务上下文。若这些条件没有进入目标，单纯增加指令精确度并不自动解决问题。

### 5.2 三种“真正想要什么”的解释

guest lecture 把 value alignment 拆成三种不同问题：

1. **对齐用户意图（user intentions）**：系统把含糊指令翻译成完整意图，包括没有明说的约束和条件。困难在于理解语言、文化、制度和交互背景；而且人类意图可能建立在不完整信息或不完美理性上。
2. **对齐显露偏好（revealed preferences）**：系统根据用户行为或反馈推断用户实际偏好。困难在于有限行为对应无穷多个可能的 reward/preference functions，也很难推断用户在紧急或完全陌生情境中的偏好。
3. **对齐用户最佳利益（best interests）**：系统选择客观上对用户更有利的结果。困难在于“客观上什么对人有益”不是单纯由实验数据决定的科学问题，而涉及哲学判断；不同人对幸福、欲望、健康、安全、知识和关系的权重也可能不同。

三者不能互相替换：一个系统可以准确执行用户意图，却违背用户更深层的偏好；也可以迎合即时偏好，却损害长期利益。guest lecture 还指出，autonomy（自主性）本身通常被视为人的利益，因此即使系统试图保护用户，也不能轻易用 paternalism（家长式替用户做决定）取代用户选择。

### 5.3 课件中的收束

guest lecture 的总结是：value alignment 是设计 AI agent，使其做我们真正想要的事；“真正想要”可以指意图、偏好或最佳利益，而三种解释各自带来技术和哲学问题。这里的“我们”不是一个无争议的单一主体，后面的案例会进一步追问谁被纳入目标。

## 6. Case study：sycophancy

与第 5 节的关系：前一节区分了三种“真正想要什么”的解释；sycophancy 具体展示了系统可能把即时认可误当成更可靠的价值信号。

**本节路线图**

1. 定义 sycophancy 及其在 RLHF 反馈中的表现。
2. 追问它更接近哪一种 alignment 解释。
3. 区分课件提出的设计问题与尚未验证的修复方案。

sycophancy 指 AI 即使面对错误、有害或不理性的信念，也一味同意或验证用户。guest lecture 将其作为 RLHF 训练中可能出现的现象：如果 human raters 更奖励“有帮助、礼貌、赞同”的表达，模型可能优化成让用户感觉被认可，而不一定更真实或更有益。

这个案例不能简单归结为“RLHF 必然导致 sycophancy”。更准确的说法是：当奖励信号把即时满意度、礼貌和真实性混在一起，且数据覆盖不足时，模型可能学到讨好用户的代理目标。减少这种风险需要重新设计比较标准、加入反例和真实性检查，并在部署情境中评估不同用户与不同观点；guest lecture 的页面提出反思问题，但没有给出一个已验证的单一修复算法。

## 7. Case study：agentic AI 与外部影响

与第 5--6 节的关系：如果 alignment 目标决定了系统如何理解用户请求，那么 agentic AI 还会把这种选择落实为自动行动、询问、拒绝或抵抗，并把影响扩展到用户以外的人。

**本节路线图**

1. 先明确个人 agent 能执行的现实任务及其外部影响。
2. 比较自动行动、增加摩擦和拒绝或抵抗三种行为模式。
3. 用 “people other than the user” 收束到第三方利益相关者问题。

guest lecture 设想一个能够订票、购物、谈判、管理日历和通信、调用在线服务的个人 agent。此时系统的行为不只影响用户本人，也可能影响商家、同事、家庭成员、服务提供者和其他受影响的人。

### 7.1 三种行动模式

- **自动行动**：如果系统主要对齐 revealed preferences，可能学习用户习惯并减少确认步骤，换取速度和便利。
- **增加摩擦**：如果系统强调 best interests，可能在高风险或长期后果明显时要求确认、提示替代方案或暂缓执行。
- **拒绝或抵抗**：当请求涉及明显伤害、越权或他人权利时，系统可能不应把“用户想要”当成唯一约束。

这三类模式并不是 guest lecture 给出的固定策略，而是帮助我们看见设计分歧：什么时候 agent 可以不问就做？什么时候必须把决定交还用户？什么时候应当拒绝或抵抗用户？答案取决于风险、可逆性、授权范围、他人影响、用户自主性和系统对意图的置信度。

### 7.2 People other than the user

guest lecture 结尾用 “people other than the user” 强调外部利益相关者。若只把用户的即时偏好当作 reward，系统可能把第三方损失、公共资源和社会规范排除在优化目标之外。这个提醒与 Lecture 8 的 proxy reward 边界直接相连：一个可预测用户选择的 reward model，不自动等于完整的人类价值函数。

最后一页邀请听众继续讨论伦理问题，并提供讲者的联系方式；它没有新增技术结论，而是把前面的 alignment 问题留给后续讨论。

## 8. 本讲小结与边界

- UCB proof 把 regret 分解为 gap 和 pull count，再用 good event 控制每个次优臂的拉取次数。
- contradiction 说明：在置信事件成立时，次优臂超过阈值后不应再成为 UCB 最大者。
- union bound 处理跨时间点的最优臂置信失败；sub-Gaussian concentration 处理次优臂仍然过于乐观的概率。
- 在课件的假设和参数选择下，$\mathbb E[N_n(a_i)]$ 具有 $O(\log n/\Delta_i^2)$ 结构，regret 具有 $O(\log n/\Delta_i)$ 的 problem-dependent 结构。
- 真实 contextual、nonstationary、delayed-feedback bandit 需要额外建模，不能直接套用这个 tabular stochastic proof。
- value alignment 不是一个只靠“把 reward 写清楚”就能解决的工程细节；意图、显露偏好、最佳利益、自主性和第三方影响可能冲突。
- sycophancy 和 agentic AI 说明 reward 的优化速度不能代替目标选择与治理判断。

## Assignment Readiness

- 理论准备：Lecture 9 的 bandit/regret/UCB 基础和本讲的 UCB proof sketch 已覆盖。
- 尚未覆盖：post deck 目录中的 PAC、Bayesian bandit、Thompson sampling 的完整算法与理论；本讲没有把它们写成已学内容。
- 伦理与 alignment 部分：已覆盖 guest lecture 的概念、案例和讨论问题，但没有替代任何 assignment 指定的伦理阅读。
- Mastery evidence：没有记录独立复现 proof、代码实现、regret 实验或对 alignment 案例的独立分析。
- 推荐下一步：先不看答案写出 $G_i$、$u_i$ 和 $\mathbb E[N_n(a_i)]$ 三步 proof skeleton，再用一个两臂例子检查固定 bonus 为什么可能线性 regret。

## 本讲必会公式

### 1. Regret gap--count 分解

$$
L_T=\sum_{a\in\mathcal A}\mathbb E[N_T(a)]\Delta_a,
\qquad
\Delta_a=V^*-Q(a).
$$

### 2. UCB 选择规则

$$
a_t
=
\arg\max_a
\left[
\widehat Q_{t-1}(a)
+
\sqrt{\frac{2\log(1/\delta)}{N_{t-1}(a)}}
\right].
$$

### 3. 次优臂抽样次数结构

$$
\mathbb E[N_n(a_i)]
\le
3+\frac{16\log n}{\Delta_i^2},
\qquad
\operatorname{Regret}_n
\le
3\sum_i\Delta_i+
16\log n\sum_{i:\Delta_i>0}\frac1{\Delta_i}.
$$

### 4. Sub-Gaussian 单侧集中界

$$
P(\widehat\mu_n-\mu\ge\eta)
\le
\exp\left(-\frac{n\eta^2}{2\sigma^2}\right).
$$

## 容易混淆点

1. Lecture 10 的 theorem bound 依赖 stochastic bandit、独立奖励和具体 UCB bonus；不能直接推广到 Covid testing 的 contextual/nonstationary setting。
2. $u_i$ 是 proof 中人为选择的 pull-count threshold，不是算法预先知道的真实 gap 或最优样本量。
3. good event 成立时的 contradiction 与 $P(G_i^c)$ 的概率控制是两步：前者控制事件内部的 count，后者控制事件本身失败。
4. 固定 bonus 也可能看起来 optimistic，但它不随数据减少不确定性，因此不能替代 UCB 的 shrinking bonus。
5. guest lecture 的 alignment 三种解释是概念区分，不是三个已经解决的算法模块。
6. sycophancy 说明某类反馈信号可能被过度优化，不足以推出所有 RLHF 系统都必然 sycophantic。
7. user preference 不是唯一利益来源；agentic AI 还要考虑第三方、公共约束和用户自主性。

## 9. 延伸阅读

### 经典基础

- Lattimore and Szepesvári, Bandit Algorithms，Lecture 10 UCB proof sketch 的直接教材来源。<https://tor-lattimore.com/downloads/book/book.pdf>
- Auer, Cesa-Bianchi, and Fischer, “Finite-time Analysis of the Multiarmed Bandit Problem” (2002)，UCB1 有限时间分析。<https://doi.org/10.1023/A:1013689704352>
- Bostrom, Superintelligence (2014)，guest lecture 使用的 paperclip maximizer 思想实验出处。<https://global.oup.com/academic/product/superintelligence-9780198739838>
- Gabriel, “Artificial Intelligence, Values, and Alignment” (2020)，课件引用的 value alignment 讨论。<https://arxiv.org/abs/2001.09768>

### 前沿动态

截至 2026-08-13 核实：本讲不额外添加前沿动态条目。guest lecture 的案例用于提出技术与哲学问题，而不是给出需要更新的产品或最新模型结论。
