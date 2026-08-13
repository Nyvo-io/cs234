---
title: CS234 Lecture 8 - Human Preferences, RLHF, and DPO
aliases:
  - CS234 Lec8
tags:
  - cs234
  - reinforcement-learning
  - imitation-learning
  - human-preferences
  - rlhf
  - dpo
---

# CS234 Lecture 8 Notes: Human Preferences, RLHF, and DPO

来源：`lecture/lecture8post.pdf`，CS234 Winter 2026，Emma Brunskill，共 68 个物理 PDF 页面；课件内部页码为 `1/44` 到 `45/44`，后者是额外的 TAMER 参考页。

笔记规范：`cs234-rl-tutor v2`。覆盖清单只表示课件内容已经写入，不表示学习者已经掌握。

外部资料核验日期：2026-08-04（延伸阅读只列本讲直接引用的经典工作；未把课件中的结果图扩展成未经核验的当前结论）。

## 0. 本讲覆盖清单

课件有大量目录页、动画重复页和从 CS224N 借用的 RLHF/DPO 页面。下面按物理 PDF 页记录覆盖位置；括号内是课件内部页码。

- [x] 第 1 页（1/44）：标题与本讲主题；写入 §1。
- [x] 第 2--3 页（2--3/44）：REINFORCE 理解检查及答案；写入 §1 和 §9。
- [x] 第 4 页（4/44）：期中考试范围与复习提示；写入 §1。
- [x] 第 5--8 页（5--8/44）：课程结构、RL 与 LLM 的问题、模仿学习目录；写入 §1。
- [x] 第 9--12 页（9--12/44）：behavior cloning 的 compounding errors、分布错配和 DAGGER 回顾；写入 §2.1。
- [x] 第 13--17 页（13--17/44）：feature-based reward、reward 非唯一性和 imitation learning 总结；写入 §2.2。
- [x] 第 18--20 页（18--20/44）：human feedback 与人类价值/行为的训练目标；写入 §2.3。
- [x] 第 21--25 页（21--25/44）：教机器人、人力成本轴、pairwise-label sweet spot、推荐排序和主动偏好查询；写入 §2.3--§2.4。
- [x] 第 26--30 页（26--30/44）：pairwise comparison、Bradley--Terry 模型、Condorcet/Copeland/Borda 定义；写入 §3.1--§3.2。
- [x] 第 31--34 页（31--34/44）：Bradley--Terry 拟合、轨迹偏好、reward model 与 PPO；写入 §3.3--§3.4。
- [x] 第 35--37 页（35--37/44）：backflip 反馈例子及从 backflips 到 ChatGPT 的过渡；写入 §4.1。
- [x] 第 38--41 页（38--41/44）：RLHF pipeline、pairwise reward model、held-out evaluation、KL-regularized policy optimization；写入 §4.2--§4.4。
- [x] 第 42--44 页（43--44/44）：RLHF 结果图、InstructGPT 规模化、受控比较和 learning-more 过渡；写入 §4.4 和 §11。
- [x] 第 45--48 页（38--38/44 的补充页）：RLHF 的 reward-model/policy 两阶段图与 KL 目标；写入 §4.2--§4.4。
- [x] 第 49--56 页（39--41/44）：DPO 动机、KL 正则目标的 closed-form optimal policy、推导和 reward-policy transformation；写入 §5.1--§5.4。
- [x] 第 57--58 页（41/44）：Bradley--Terry loss 与 DPO loss 的 log-partition cancellation；写入 §5.4。
- [x] 第 59--65 页（42/44）：DPO 结果、reward/KL trade-off、模型规模化和 CPL 方向；写入 §5.6。
- [x] 第 66--67 页（43--44/44）：结果收束与 human preferences 课程方向；写入 §5.6 和 §11。
- [x] 第 68 页（45/44）：TAMER 参考条目；写入 §11。

视觉材料决策：第 25 页的主动偏好查询、第 39 页的 pairwise reward-model 图、第 41 页的 RLHF KL 目标和第 58 页的 DPO loss 直接嵌入。第 22--23 页的人力成本示意图、推荐排序结果图和 DPO 中间手写推导用文字与公式重写；目录页、重复动画页、结果截图和装饰性页面不重复嵌入。

## 1. 本讲主线

与 Lecture 7 的关系：Lecture 7 已经介绍了 模仿学习、DAGGER、feature-based IRL、MaxEnt IRL 和 preference feedback 的入口。本讲沿着“从行为或反馈学习目标”的方向继续推进，但把重点从“能否从示范恢复一个唯一 reward”转向“如何用 人类比较反馈 构造并优化一个可用目标”。

**本讲路线图**

1. **先处理反馈形式**：从 DAGGER 的逐状态 action 标签，过渡到人类更容易提供的 pairwise preference。
2. **再学习代理 reward**：用 Bradley--Terry 把比较概率写成 reward difference，并拟合 trajectory-level reward model。
3. **接入语言模型 RL**：把 reward model 放入 RLHF pipeline，用 KL 正则限制 policy 偏离 reference model。
4. **最后改写优化目标**：利用 KL 正则目标的 closed-form 关系，把 reward difference 改写成 DPO 的 policy/reference log-ratio loss。

本讲开头还复习一次 REINFORCE，并提醒期中考试覆盖到本讲。覆盖完成不等于 mastery：本笔记最后的自测题和 Assignment 3 readiness 只说明可以开始练习，不代表已经独立实现 RLHF 或 DPO。

课件第 6 页用“如何让 RL enable transformative LLM”提出动机，但这里的问句不是一个已经证明的结论。本讲真正回答的是更窄、也更可检验的问题：怎样把人类比较反馈建模成 reward，又怎样在策略优化时控制对 reference model 的偏离。

## 2. 从示范到人类偏好

与 Lecture 7 的关系：Lecture 7 已经说明 DAGGER 需要专家在 learner-induced states 上持续提供 action 标签，而 IRL 仍面临 reward 非唯一性。本节先回顾这两个限制，再说明为什么 pairwise preference 是一种更适合人类提供的反馈形式。

**本节路线图**

1. 回顾 behavior cloning 的 compounding errors，以及 DAGGER 如何修复状态分布错配。
2. 说明从示范恢复唯一 reward 的困难仍然存在，preference learning 学到的是代理目标。
3. 比较不同人类反馈形式的 effort，并引出主动偏好查询。
4. 解释为什么成对比较通常比标量评分或手写 reward 更容易获得。

### 2.1 DAGGER 和 compounding errors 回顾

*首次完整讲解：Lecture 7 §4.3--§4.4「Compounding errors」和「DAGGER」。本节只补充：人力成本为什么推动 preference learning。*

Behavior cloning 在专家访问到的状态分布 $d^{\pi^*}$ 上拟合 action，但部署时学习策略诱导的是 $d^{\pi_\theta}$。一次动作错误可能把系统带到示范没有覆盖的状态，随后错误继续累积；课件用下面的近似直觉表达这个风险：

如果每一步的错误都独立、且不会改变后续状态分布，期望错误数只是 $\epsilon T$；真正的问题是一次错误会改变后续访问的状态，使后面的决策也更容易出错。

$$
\mathbb E[\text{total errors}]
\lesssim
\epsilon\bigl(T+(T-1)+(T-2)+\cdots+1\bigr)
=O(\epsilon T^2).
$$

通过在 learner-induced states 上请求专家 action，再把新标签聚合进数据集，缓解训练/部署状态分布错配。它的代价是专家需要在学习策略访问的状态上反复标注；如果人类不适合逐状态给 action，另一条路线就是请求更粗粒度的 pairwise preference。

### 2.2 Reward 非唯一性仍然存在

*首次完整讲解：Lecture 7 §5.1「特征奖励的设置与非唯一性」。本节只补充：preference learning 选择的是一类可预测人类比较的 reward model，不应自动解释成真实人类价值。*

在已知 transition model、示范和专家最优性的情况下，仍可能存在无穷多个 reward 让同一个 expert policy 最优。feature matching 和 MaxEnt IRL 为这种 ambiguity 提供了不同的约束或选择原则，但它们都没有证明“从有限反馈 恢复 唯一的真实 reward”。

对 RLHF 来说，这个边界(我们学出来的 reward 不等于 真实的人类价值 / 真实 reward，它不是唯一的)更重要： 
	reward model $r_\phi$ 只是从有限、带噪声的人类比较中 拟合出的代理目标（proxy objective），它可以帮助策略优化，却可能遗漏人类没有标注的维度，或被策略利用其错误泛化。


### 2.3 人类输入的连续谱

到底让人类参与训练到什么程度？

课件把人类参与训练/对齐 policy 的方式画成一个 effort 轴：一端是 DAGGER 或 constant teaching，需要专家在学习过程持续提供 action；
另一端是只给 demonstrations，标注少但更容易发生状态分布错配。
人类直接给完整示范：τ=(s0​,a0​,s1​,a1​,…)， 然后模型 imitation learning。

pairwise labels 位于两者之间：人类不必写出数值 reward，只需比较两个候选结果。

这不是一个保证总有效的排序，而是一个工程取舍：更密集的 feedback 可能增加人力成本，更稀疏的 feedback 可能使 reward model 对关键状态缺少监督。
主动学习（active preference learning）进一步让系统选择最有信息量、最能区分候选 reward 的比较，而不是随机询问每一对轨迹。

第 21 页的 teachable-robot 例子（Thomaz and Breazeal, 2008）把 human teaching behavior 与环境反馈放在同一个训练循环中；它说明人类输入可以是交互式 shaping 信号，不等于本讲后面固定的 pairwise preference pipeline。



![[lec8-active-preference-p25.png|900]]

*图：主动偏好查询在两个候选车辆轨迹之间请求比较；来源：`lecture/lecture8post.pdf` 物理 PDF 第 25 页（课件页 25/44），原图引用 Sadigh et al., RSS 2017。图中 $\xi_A$、$\xi_B$ 是待比较的轨迹，车辆与道路环境显示了查询所针对的行为差异。*
>	机器生成 A、B 两种行为→问人更喜欢哪个→得到 R(A)>R(B) 这样的约束→逐渐学习 reward​
>	
>	Active preference learning=机器主动选择最有信息量的问题问人​


BC 和DAGGER 主要是根据专家策略，学习出自己的策略。

IRL 根据专家行为/ trajectory ，学习reward

Preference Learning： 人只需告诉你两条轨迹谁更好就可以，让模型学习reward

### 2.4 为什么 pairwise comparison 常比标量评分容易

让人类给一个广告或一段行为打绝对分数，需要先共享“7 分”和“8 分”的尺度；但两个候选之间选哪个更好，通常只需要局部比较。
课件因此把 pairwise comparisons 作为更容易获得的监督形式，并用推荐排序页面说明：系统可以让用户比较两个 ranking 或两个候选结果，而不是要求用户写出完整 reward function。

pairwise feedback 也不是自动客观：人的判断可能噪声大、受展示顺序影响，甚至对同一对候选不一致。它只是把标注问题变成一个更适合概率建模的形式，后面仍需要 reward-model validation 和对数据覆盖范围的检查。

pairwise comparison只需要做出局部相对判断，并不用给出绝对评分尺度。 但人类标注/比较并不是绝对客观的，要把它看成一个有噪声的概率事件，这引出了Bradley–Terry


## 3. Bradley--Terry：从比较概率 到 reward model

与上一节的关系：上一节把 pairwise comparison 作为较低负担的反馈形式，但还没有说明如何把“谁更好”变成可训练的数值目标。本节用 Bradley--Terry 建立这座桥：先建模比较概率，再把动作或轨迹的 reward difference 交给 reward model 学习。

**本节路线图**

1. 在 K-armed bandit 中定义 Bradley--Terry 偏好概率，并理解 reward 的平移不变性。
2. 把单个动作扩展成 trajectory score，得到轨迹偏好概率。
3. 用 cross-entropy 拟合 reward model，并解释 reward difference 的梯度。
4. 说明学到的 reward 如何交给 PPO 等 policy optimizer。

### 3.1 K-armed bandit 中的结构模型

Bradley--Terry 模型先在没有 state/context 的 $K$ 臂 bandit （有k个可选动作）上定义一个简单的潜在 reward $r(b_i)$  (这只是模型预测的reward值)。如果人类比较动作 $b_i$ 和 $b_j$，模型假设 偏好 $b_i$(更喜欢动作bi) 的概率为

$$
P(b_i\succ b_j)
=\frac{\exp(r(b_i))}{\exp(r(b_i))+\exp(r(b_j))}
=\sigma\bigl(r(b_i)-r(b_j)\bigr)
=p_{ij},
$$
把r(bi​)−r(bj​)映射到 0∼1 的概率，选择了sigmod 函数

用softmax 函数 定义强度 strength(bi​)=exp(r(bi​)) ,这样刚好bi获胜的概率和bj获胜的概率之和就是1


其中 $\sigma(z)=1/(1+\exp(-z))$ 是 sigmoid。整个式子只使用 [reward difference](academic-term-lookup:reward%20difference)：$r(b_i)$ 比 $r(b_j)$ 大得越多，模型越倾向于预测人类选择 $b_i$；两者相等时，概率为 $0.5$。

一个重要性质是平移不变性：对所有动作加同一个常数 $c$，$r'(b)=r(b)+c$，分子分母同时乘 $\exp(c)$，所有偏好概率不变。因此 pairwise data 通常不能识别 reward 的绝对零点，只能约束相对差异。你只能知道：r(bi​)−r(bj​)，而不能知道： r(bi​)到底应该是 2、12 还是 2002

同一组潜在 reward 还为不同 pairwise probability 提供一致的排序关系；例如 $p_{ik}$ 和 $p_{jk}$ 都由同一组 reward difference 决定，而不是彼此独立拟合的概率。这是 Bradley--Terry 与任意逐对投票规则的区别之一。

> [!example] 具体计算：Bradley--Terry 偏好概率
> 设 $r(b_1)=2$、$r(b_2)=1$，则
>
> $$
> P(b_1\succ b_2)
> =\frac{e^2}{e^2+e^1}
> =\sigma(1)
> \approx0.731.
> $$
>
> 模型不说 $b_1$ 每次都赢，而是说在相同条件下人类选择 $b_1$ 的概率约为 $73.1\%$。把两个 reward 同时加 $5$ 不会改变这个概率。

课件还列出三种社会选择中的 winner 定义，它们不是 Bradley--Terry 的额外训练步骤，而是帮助理解 pairwise ranking 的不同汇总方式：

- **Condorcet winner**：$b_i$ 对每个其他动作 $b_j$ 都满足 $P(b_i\succ b_j)>0.5$。
- **Copeland winner**：与其他动作逐对比较，胜场数减负场数最高的动作。
- **Borda winner**：最大化期望 pairwise score；胜出得 $1$，平局得 $0.5$，落败得 $0$。

这些定义可能给出不同答案，也可能不存在 Condorcet winner；“最常赢”与“latent reward 最大”不能在没有模型假设时互换。

==Bradley–Terry 真正关心的不是两个 reward 本身，而是 reward difference==


### 3.2 从 bandit 比较到 trajectory preference

bandit 里面我们比较的是：$b_i\quad\text{vs.}\quad b_j$

但是在 RL 里，人类通常比较的是**整条轨迹**


对一条有限轨迹 $\tau=(o_0,a_0,\ldots,o_{T-1},a_{T-1})$，reward model 对整条轨迹给出累计分数

$$
R_\phi(\tau)
=\sum_{t=0}^{T-1}\hat r_\phi(o_t,a_t).
$$

将 bandit 的动作替换为轨迹，Bradley--Terry 概率变为

$$
\widehat P_\phi(\tau_1\succ\tau_2)
=\frac{\exp(R_\phi(\tau_1))}
{\exp(R_\phi(\tau_1))+\exp(R_\phi(\tau_2))}
=\sigma\bigl(R_\phi(\tau_1)-R_\phi(\tau_2)\bigr).
$$

这一步把每个 state--action reward 的预测累加成一个可比较的 trajectory score；它没有声称每个局部 reward 都能被人类直接观察，也没有消除 reward ambiguity。



针对一个问题，模型生成两个回答：

- 回答 A：正确解释瑞利散射
- 回答 B：说天空是因为反射海洋才是蓝色

人类看到的是 **A 和 B 的文本内容**，然后回答：$A\succ B$

不需要看到$R_\phi(A)=?$

假设目前 reward model 给两个回答预测：

$R_\phi(A)=0.3,\qquad R_\phi(B)=0.8$

但是人类刚刚告诉我们：$A\succ B$

这说明 reward model 当前预测错了，因为它反而给 B 更高的分。

根据 Bradley–Terry：
$P_\phi(A\succ B)=\sigma(R_\phi(A)-R_\phi(B))$
得到的数值是负数

于是训练 reward model，修改参数 ϕ，让以后：
$$R_\phi(A)>R_\phi(B)$$

> [!example] 具体计算：轨迹比较的 reward difference
> 假设模型对两条等长轨迹的累计预测 reward 是 $R_\phi(\tau_1)=2.4$ 和 $R_\phi(\tau_2)=1.2$。则
>
> $$
> \widehat P_\phi(\tau_1\succ\tau_2)
> =\sigma(2.4-1.2)
> =\sigma(1.2)
> \approx0.769.
> $$
>
> 如果数据标签认为 $\tau_1$ 胜出，增大 $R_\phi(\tau_1)-R_\phi(\tau_2)$ 会让模型更符合标签；如果标签是平局，目标概率应靠近 $0.5$。


### 3.3 Cross-entropy 拟合与梯度

现在进入真正的 supervised learning。

修改 reward model 的参数 ϕ
$(s,a)\stackrel{r_\phi}{\longrightarrow}\text{一个 reward 数字}$

对数据集 $D=\{(b_i,b_j,\mu)\}$，先用讲义约定 $\mu=1$ 表示第一项胜出、$\mu=0.5$ 表示平局、$\mu=0$ 表示第二项胜出。交叉熵损失为（binary cross-entropy）

$$
\mathcal L(\phi)
=-\sum_{(b_i,b_j,\mu)\in D}
\left[
\mu\log p_{ij}
+(1-\mu)\log(1-p_{ij})
\right],
$$

reward model 的预测 p 和真人标签 μ 差得越多，loss 越大。
​

其中 $p_{ij}=P_\phi(b_i\succ b_j)=\sigma\left(r_\phi(b_i)-r_\phi(b_j)\right)$。对轨迹，把 $b_i,b_j$ 替换成 $\tau_1,\tau_2$，把 $r_\phi(b)$ 替换成 $R_\phi(\tau)$ 即可。

令 $z_\phi=R_\phi(\tau_1)-R_\phi(\tau_2)$、$p=\sigma(z_\phi)$，单个样本的 loss 对 $z_\phi$ 的梯度是

定义z是为了简化公式：zϕ​=第一条轨迹 reward − 第二条轨迹 reward


$$\phi\rightarrow z_\phi\rightarrow p\rightarrow\ell$$
$$\frac{\partial\ell}{\partial\phi}=\frac{\partial\ell}{\partial z_\phi}\frac{\partial z_\phi}{\partial\phi}$$
$$\frac{\partial z_\phi}{\partial\phi}=\nabla_\phi R_\phi(\tau_1)-\nabla_\phi R_\phi(\tau_2)$$

$$
\frac{\partial\ell}{\partial z_\phi}=p-\mu.
$$
p-u比较重要，它直接告诉模型预测是偏高还是偏低，u就是1或者0
$$\boxed{\text{梯度}=\text{模型预测}-\text{真实标签}}$$

这其实是 sigmoid + cross-entropy 最经典的结果。

最终结合：
$$\nabla_\phi\ell=(p-\mu)\left[\nabla_\phi R_\phi(\tau_1)-\nabla_\phi R_\phi(\tau_2)\right]$$

因此当第一条轨迹被标为胜出（$\mu=1$）而当前模型只给出 $p=0.7$ 时，梯度为 $-0.3$；梯度下降会推动 $z_\phi$ 增大。完整的 reward-model 梯度是

$$
\nabla_\phi\mathcal L
=\sum_{(\tau_1,\tau_2,\mu)\in D}
(p-\mu)
\left[
\nabla_\phi R_\phi(\tau_1)-\nabla_\phi R_\phi(\tau_2)
\right].
$$

这就是 Assignment 3 §2.1 要求推导的结构：比较只约束两条轨迹 reward 的差，而不是分别把某个绝对 reward 拟合成固定标签。

> [!important] 课件标签与 Assignment 3 标签不同
> Lecture 8/Bradley--Terry 公式使用 $\mu=1$ 表示第一项胜出、$0$ 表示第二项胜出；Assignment 3 的 `.npz` 数据和 `RewardModel.update` docstring 使用 `label=0` 表示第一条序列胜出、`label=1` 表示第二条序列胜出、`label=0.5` 表示平局。实现前必须先统一编码，不能直接把 assignment label 当成上式的 $\mu$。


### 3.4 学到 reward 后再做 policy optimization

训练好 $r_\phi(o,a)$ 神经网络后，给出state- action pair 他就可以输出reward分数。把它放进环境 wrapper ($(o,a)\rightarrow r_\phi(o,a)$) ，按每一步累计预测 reward，再用 PPO 等 policy optimization 方法训练 policy。policy 做 action → 得到 reward → 想办法提高长期 reward

这正是“preference -> reward model -> RL policy”的两阶段结构：偏好数据监督 reward model，环境交互和 PPO 再把该 reward 转成行为。

它的优点是可以复用成熟的 policy optimizer；代价是需要额外训练一个 reward model，而且 reward model 的错误会被后续 policy optimization 放大。Lecture 8 之后的 RLHF pipeline 就是把这条轨迹扩展到语言模型输出。



## 4. RLHF：用人类反馈训练语言模型

与上一节的关系：上一节得到的是“偏好比较 → reward model → policy optimization”的通用链条。本节把这条链条实例化到语言模型：输出从机器人动作或轨迹变成文本 response，policy optimization 还必须处理 reward 与 reference model 的偏离。

**本节路线图**

1. 用 backflip 例子说明人类反馈如何替代逐状态手写 reward。
2. 把 instruction tuning、偏好采样、reward modeling 和 policy optimization 串成 RLHF pipeline。
3. 引入 KL-regularized objective，解释为什么高 reward 不能成为唯一目标。
4. 结合课件结果说明 RLHF 的收益、比较基线和适用边界。


### 4.1 从 backflip 到语言模型

课件用 Christiano et al. (2017) 的 [backflip](academic-term-lookup:backflip) 例子说明：人类不必为每个状态写 reward，只要观看 agent 的行为并给出反馈，系统就可以逐步学习。课件页上的表述是“需要 900 bits of feedback”来学会 backflip；这是该研究/课程幻灯片中的具体案例，不是所有任务的普适样本复杂度结论。

从 backflip 到 ChatGPT 的过渡保留了同一个抽象：模型生成候选行为，人类比较候选结果，reward model 学习 比较规律（人类不断告诉它“A 比 B 好”或“B 比 A 好”，reward model 就调整自己的打分方式，让自己的分数排序越来越符合人类的选择），RL 算法再优化模型。改变的是 action 从机器人控制动作变成了文本 token 序列，trajectory reward 通常对应一个完整 response 或其分段得分。

$$\text{robot trajectory}\longrightarrow\text{LLM response}$$


### 4.2 RLHF 的高层 pipeline

对一个语言模型，课件给出的实例化可以按四步理解：

1. **Instruction tuning / SFT**：用 示范指令--回答数据 让预训练模型学会遵循任务格式，得到初始 policy。
2. **采样与比较**：给定 prompt，从模型生成多个回答；人类对回答做 pairwise comparison，而不是要求精确写出 scalar reward。
3. **Reward modeling**：用 Bradley--Terry 形式训练 $r_\phi(x,y)$，使 偏好回答 的累计 reward 更高；先用 held-out human judgments 检查 reward model 是否能预测未见比较。
4. **Policy optimization**：从 reference model 初始化一个可训练 policy，用 PPO 等 RL 方法提高 learned reward，同时惩罚偏离 reference model 的行为。


第一步：SFT

SFT = Supervised Fine-Tuning。

SFT 是语言模型版本的 Behavior Cloning。

先准备大量：$(x,y^*)$

其中：

- x：instruction / prompt
- y∗：人类写好的高质量回答

然后普通 supervised learning，让语言模型学：

> 看到这种 instruction 应该怎么回答。

于是得到一个初始模型。

这个模型后面通常也会成为：$\pi_{\rm ref}$

也就是 reference policy。

> **先教模型基本会说人话、会遵循 instruction。**


第二步：采样 + 人类比较

收集 preference dataset​
对于同一个 prompt：x

让初始模型生成多个回答：y1​,y2​,…

然后让真人比较：$y_w\succ y_l$

其中：

- w = winner
- l = loser

人类完全不需要给：r(yw​)=8.2 这种绝对评分。

只需要：

存成一条训练数据：(x, yw​, yl​)

收集很多后：
$D=\{(x,y_w,y_l)\}$

第三步：

拿第二步的 D 去训练 rϕ

Reward Model 输入：(x,y)

输出一个 scalar：
$r_\phi(x,y)$

然后从第二步里的数据集里得出：
yw​≻yl​

希望模型学到：

$r_\phi(x,y_w)>r_\phi(x,y_l)$


Bradley–Terry 把这两个分数转成“预测人类会选 winner 的概率”：

$P_\phi(y_w\succ y_l\mid x)=\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right)$

训练 loss：

$\mathcal L_{\rm RM}=-\mathbb E_{(x,y_w,y_l)\sim D}\left[\log\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right)\right]$

它的目标非常简单：

$r_\phi(x,y_w)-r_\phi(x,y_l)\uparrow$


第四步：Policy Optimization

现在 Reward Model 已经训练好了。之后 policy：​$\pi_\theta$ (就是第一步的初始模型)

生成一个回答：
$y\sim\pi_\theta(\cdot|x)$

然后 reward model 给：
$r_\phi(x,y)$

PPO 通过这个 reward 更新 θ，形成新策略，原来的​$\pi_\theta$被冻住

如果不冻住，只做：

$\max_\theta\mathbb E_{y\sim\pi_\theta}[r_\phi(x,y)]$

会有一个危险：
>	πθ​ 可能为了疯狂提高 Reward Model 分数，变得越来越奇怪

例如 Reward Model 错误地发现：

> “回答特别长通常 reward 高。”

PPO 就可能不断利用这个漏洞，把回答搞得越来越长。

所以我们告诉它：

> **你可以为了高 reward 改变，但别离原来的 SFT 模型太远。**

这时就需要 reference policy：

πref​

典型目标可以粗略写成：

$r_\phi(x,y)-\beta D_{\mathrm{KL}}\left(\pi_\theta\|\pi_{\mathrm{ref}}\right)$


注意这里有两个模型：
​$\boxed{r_\phi=\text{Reward Model},\qquad\pi_\theta=\text{Policy / LLM}}$

前一个负责：

> “这个回答多好？”

后一个负责：

> “我要生成什么回答？”


![[lec8-bradley-terry-p39.png|900]]

*图：文本候选的 pairwise comparison 与 reward model 预测；来源：`lecture/lecture8post.pdf` 物理 PDF 第 39 页（课件页 40/44 的配套页）。图中 “winning/losing” 说明 reward model 只需让偏好回答的分数更高，不直接获得人类的绝对价值。*
图里的reward model 读取一个完整文本，然后输出一个scalar数值。 只要大小关系满足人类给的排序，就证明学的是对的

$\sigma\left(RM_\phi(s^w)-RM_\phi(s^l)\right)$  这个就是 $P_\phi(s^w\succ s^l)$， 人类会选择winner 的概率

加上负log， 就是变成loss 交叉熵， 只不过这里u一定等于1，所以和上面的式子有点不一样

取E 代表是从数据集D中选取样本 ，然后计算平均loss

“先评估 reward model，再做 RL”是一个必须保留的工程边界。如果 reward model 在 held-out preference 上表现差，直接用它训练 policy 只会把噪声或错误偏好变成更强的策略行为。


### 4.3 KL 正则：高 reward 之外还要留在 reference 附近

令 $x$ 是 prompt，$y$ 是完整 response，$\pi_{\mathrm{ref}}$ 是冻结的 reference model，$\pi$ 是要优化的 policy，$r_\phi(x,y)$ 是 reward model 输出。Lecture 8 的 KL-regularized objective 可以写为

$$
\max_\pi\quad
J(\pi)
=\mathbb E_{x\sim D,\,y\sim\pi(\cdot\mid x)}
\left[r_\phi(x,y)\right]
-\beta\,\mathbb E_{x\sim D}
\left[
D_{\mathrm{KL}}
\left(
\pi(\cdot\mid x)\,\middle\|\,\pi_{\mathrm{ref}}(\cdot\mid x)
\right)
\right].
$$
拆成：

目标 = 人类偏好 reward− 偏离 reference 的惩罚​
防止为了一个劲的找满足 最大化reward 的策略，从而给策略加了限制，让他不要偏离原策略太远

第一项鼓励模型得到人类偏好的回答，第二个项抑制它为了钻 reward model 的漏洞而远离原始模型。$\beta>0$ 越大，偏离 reference 的代价越强；$\beta$ 越小，优化更愿意牺牲 reference 相似度来追求 reward。

x∼D
从 prompt dataset 里抽一个 prompt。

y∼π(⋅∣x)
当前 policy 根据这个 prompt 生成 response。

$D_{\rm KL}\left(\pi(\cdot|x)\Vert\pi_{\rm ref}(\cdot|x)\right)$
它在测量：
> 当前 policy 的 probability distribution 和 reference policy 有多不一样。

β 是 **KL penalty coefficient**。

对单个采样回答，KL 项的期望可以写成 log-ratio penalty：

$$
\widetilde r(x,y)
=r_\phi(x,y)
-\beta\log
\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)},
\qquad y\sim\pi(\cdot\mid x).
$$

因为

$$
\mathbb E_{y\sim\pi(\cdot\mid x)}
\left[
\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
\right]
=D_{\mathrm{KL}}\left(\pi(\cdot\mid x)\middle\|\pi_{\mathrm{ref}}(\cdot\mid x)\right),
$$

这两个写法表达的是同一个期望目标；采样实现通常使用 log-probability difference。

为什么这个 log-ratio 的 expectation 就是 KL？

因为 KL 定义本身就是：

$$D_{\rm KL}\left(\pi(\cdot|x)\Vert\pi_{\rm ref}(\cdot|x)\right)=\mathbb E_{y\sim\pi(\cdot|x)}\left[\log\frac{\pi(y|x)}{\pi_{\rm ref}(y|x)}\right]$$
要么整体 objective 里面直接减：$\beta D_{\rm KL}$

要么每次 sample 一个 y 后，reward 改成：
​
$r_\phi(x,y)-\beta\log\frac{\pi(y|x)}{\pi_{\rm ref}(y|x)}$




![[lec8-rlhf-objective-p41.png|900]]

*图：课件把 reward model 输出和相对 reference 的 log-probability penalty 合在一个 RL objective 中；来源：`lecture/lecture8post.pdf` 物理 PDF 第 41 页（课件页 42/44）。图中的 $\beta$ 是 KL penalty coefficient，不是 PPO 的 clip parameter。*

> [!example] 具体计算：reward 与 KL penalty 的取舍
> 对一个 prompt，假设两个回答 $y_a,y_b$ 的 reward 分别为 $1.2,0.4$；当前 policy 给出概率 $(0.75,0.25)$，reference 给出概率 $(0.6,0.4)$，取 $\beta=0.5$。
>
> 当前 policy 的期望 reward 是
>
> $$
> 0.75\times1.2+0.25\times0.4=1.0.
> $$
>
> KL 为
>
> $$
> 0.75\log\frac{0.75}{0.6}
> +0.25\log\frac{0.25}{0.4}
> \approx0.0498.
> $$
>
> 所以该 prompt 的 KL-regularized objective 约为 $1.0-0.5\times0.0498=0.9751$。它不是只看 reward 的 $1.0$，也会为偏离 reference 付出约 $0.0249$ 的代价。


### 4.4 课件中的结果与边界

课件展示了 Stiennon et al. (2020) 的比较图：随着模型规模变化，人类反馈、supervised learning、reference summaries 和 pretraining-only 的指标不同；图的作用是说明 RLHF pipeline 可以带来改进信号，不是给出任意模型、任意数据和任意标注协议下的普适保证。

课件还提到 InstructGPT 将 RLHF 扩展到 tens of thousands of tasks，并展示了“RLHF 风格”方法的 controlled comparisons：PPO 可以工作，但 best-of-$n$ 或直接训练较好回答等简单 baseline 也可能表现有竞争力。这里的结论应限定在课件所引用的实验和设置，不能改写成“PPO 总是优于所有替代方法”。



## 5. DPO：把 reward optimization 改写成 policy optimization

与上一节的关系：上一节的 RLHF 需要显式训练 reward model，再用 PPO 等方法优化 KL-regularized reward。本节保留同一个偏好与 reference-policy 假设，但利用最优策略的 closed-form 关系，直接把 reward difference 改写成 policy 的训练 loss。

前面的RLHF：
$\text{Preference}\rightarrow\text{Reward Model}\rightarrow\text{PPO}\rightarrow\text{Policy}$

那么 DPO 现在想做的是：

Preference→直接训练 Policy​

**本节路线图**

1. 说明显式 reward model 加 PPO 的额外成本，提出直接优化 policy 的动机。
2. 先写出 bandit 形式的 DPO loss，明确 winner、loser 和 reference policy 的角色。
3. 求解 KL 正则目标的 closed-form optimal policy，得到 reward 与 policy 的关系。
4. 利用 Bradley--Terry 只依赖 reward difference 的性质消去 $Z(x)$，推导 DPO loss。
5. 把 bandit 推导映射到 Assignment 3 的 action sequence 和 receding-horizon control，并标出课程边界。

### 5.1 为什么需要 DPO

RLHF 的典型流程要训练 reward model，再用 PPO 优化该 reward。这样做灵活，但有两个额外环节：reward model 可能错误，PPO 还要在生成模型上进行在线或近在线的 RL 优化。Direct Preference Optimization（DPO）提出直接用 preference pairs 更新 policy，不显式维护一个单独的 reward model。

DPO 不是“不要 reference model”，也不是“不要 reward 假设”：它仍使用冻结的 $\pi_{\mathrm{ref}}$，并利用 KL-regularized objective 的 closed-form policy--reward relation。
它最初针对 bandit/contextual response setting；Assignment 3 把它改造成 Hopper 的 action-sequence、receding-horizon 版本，因此不能把作业实现当成一般多步 RL 的无条件理论等价。

普通 contextual bandit：
$x \rightarrow a \rightarrow r$
其中：
- x：context
- a：从若干动作里选一个
- r：这个动作的 reward

到了语言模型/DPO：
$x \rightarrow y \rightarrow r(x,y)$

其中：

- x：prompt，也就是 context
- y：**完整的 response**
- r(x,y)：这个完整 response 的 reward

所以这里把整个回答 y 看成了“一次 action”。

普通 RL 会强调：
$s_0\xrightarrow{a_0}s_1\xrightarrow{a_1}s_2\xrightarrow{a_2}\cdots$

每一步都有：
- state
- action
- transition
- 可能还有每一步 reward

而 DPO 最初的抽象更像：

prompt x→整个 response y​

然后人类比较两个完整回答：
​
$y_w \succ y_l$


### 5.2 Bandit 形式的 DPO loss

对 context/state $x$，偏好回答为 $y_w$（winning），另一个回答为 $y_l$（losing），训练 policy 为 $\pi_\theta$，reference policy 为 $\pi_{\mathrm{ref}}$。DPO 的目标写成

$$
\mathcal L_{\mathrm{DPO}}(\pi_\theta;\pi_{\mathrm{ref}})
=-\mathbb E_{(x,y_w,y_l)\sim D}
\left[
\log\sigma\left(
\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\right)
\right].
$$

它奖励一个相对 reference 更偏向 $y_w$、而不是 $y_l$ 的 policy。对文本，$\log\pi(y\mid x)$ 是 response 中各 token conditional log-probability 的和；对 Assignment 3，则是 action sequence distribution 对整个 action segment 的 log-probability。

里面真正核心的只有两个量。

winner 的：

$$\log\frac{\pi_\theta(y_w|x)}{\pi_{\rm ref}(y_w|x)}$$

loser 的：
$$\log\frac{\pi_\theta(y_l|x)}{\pi_{\rm ref}(y_l|x)}$$

假设 winner 原来在 reference model 中概率是 0.2。

训练后的 policy 里变成 0.4：

$\frac{\pi_\theta(y_w|x)}{\pi_{\rm ref}(y_w|x)}=\frac{0.4}{0.2}=2$

这意味着：

> 相对于 reference，当前 policy **更加偏爱 winner 了**。

所以：

$\log\frac{\pi_\theta(y_w|x)}{\pi_{\rm ref}(y_w|x)}>0$

要让 loss 下降，就要：

让 winner 的 relative log-ratio 上升 或者 让 loser 的 relative log-ratio 下降


公式里引入reference model，充当一个锚点

这和前面 RLHF 的 KL regularization 是同一套思想

下一节是这个loss的数学依据

> [!example] 具体计算：一次 DPO loss
> 假设 $\beta=0.5$，当前 policy 相对于 reference 的 log-ratio 为
> $\log[\pi_\theta(y_w\mid x)/\pi_{\mathrm{ref}}(y_w\mid x)]=0.4$，
> $\log[\pi_\theta(y_l\mid x)/\pi_{\mathrm{ref}}(y_l\mid x)]=-0.2$。
>
> DPO 的 sigmoid 输入为
>
> $$
> z=0.5(0.4-(-0.2))=0.3,
> \qquad
> \sigma(z)\approx0.574.
> $$
>
> 单样本 loss 为 $-\log(0.574)\approx0.555$。如果 policy 进一步提高 winning response 的相对 log-ratio、降低 losing response 的相对 log-ratio，$z$ 增大，loss 下降。


### 5.3 KL 正则目标的 closed-form optimal policy

DPO 的关键不是凭空发明一个 loss，而是先解一个 KL-regularized reward objective。固定 prompt $x$，我们希望找最好的 policy：

$$
\max_{\pi(\cdot\mid x)}
\sum_y\pi(y\mid x)r(x,y)
-\beta\sum_y\pi(y\mid x)
\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)},
$$
这个就是之前讲过的：

高 reward− β×偏离 reference 的代价​ 

并满足 $\sum_y\pi(y\mid x)=1$。

对概率约束建立拉格朗日函数，用拉格朗日乘子求解，对每个 $\pi(y\mid x)$ 求导并令其为零，可得

$$
\pi^*(y\mid x)
=\frac{1}{Z(x)}\,
\pi_{\mathrm{ref}}(y\mid x)
\exp\left(\frac{1}{\beta}r(x,y)\right),
$$
拆成三部分：
​​$\pi^*(y|x)\propto\underbrace{\pi_{\rm ref}(y|x)}_{\text{原模型喜欢程度}}\underbrace{\exp(r(x,y)/\beta)}_{\text{reward 加权}}$


这个公式就是 **closed-form optimal policy**。

所谓 closed-form，就是：

> 不用 PPO 一步一步猜最优 policy，数学上直接告诉你最优 policy 应该长什么样。


其中归一化常数为

$$
Z(x)=\sum_{y'}\pi_{\mathrm{ref}}(y'\mid x)
\exp\left(\frac{1}{\beta}r(x,y')\right).
$$

这个结果的整体含义是：最优 policy 在 reference probability 上乘一个由 reward 决定的指数权重，再归一化；
$\beta$ 控制 reward 偏好相对于 reference 的放大程度。$\beta$ 大时分布更接近 reference，$\beta$ 小时高 reward 回答更容易集中概率。

> [!example] 具体计算：closed-form policy 的 reward/KL 折中
> 设 $\pi_{\mathrm{ref}}(y_a\mid x)=0.6$、$\pi_{\mathrm{ref}}(y_b\mid x)=0.4$，reward 为 $r(y_a)=1$、$r(y_b)=0$，取 $\beta=1$。
>
> 两个未归一化权重为 $0.6e^1\approx1.631$ 和 $0.4e^0=0.4$，因此
>
> $$
> \pi^*(y_a\mid x)\approx\frac{1.631}{1.631+0.4}=0.803,
> \qquad
> \pi^*(y_b\mid x)\approx0.197.
> $$
>
> 最优 policy 明显偏向高 reward 的 $y_a$，但没有把 reference 原本给 $y_b$ 的概率直接变成零。

课件特别标出一个实践障碍：$Z(x)$ 要对所有可能 response 求和，语言模型的 response 空间巨大，不能直接枚举。因此要把 reward 用 policy 与 reference 的概率比表示，而不是显式计算 $Z(x)$。


### 5.4 从 reward difference 推出 DPO

由 closed-form relation 取对数并整理，任意与某个最优 policy $\pi^*$ 对应的 reward 都可以写成
$$\pi^*(y|x)=\frac{1}{Z(x)}\pi_{\rm ref}(y|x)\exp\left(\frac{r(x,y)}{\beta}\right)$$
$$\frac{\pi^*(y|x)}{\pi_{\rm ref}(y|x)}=\frac1{Z(x)}\exp\left(\frac{r(x,y)}{\beta}\right)$$
$$\log\frac{\pi^*(y|x)}{\pi_{\rm ref}(y|x)}=-\log Z(x)+\frac{r(x,y)}{\beta}$$

整理得到最终式子：

$$
r(x,y)
=\beta\log\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
+\beta\log Z(x).
$$

然后再 利用Bradley--Terry ，它使用了两个回答的 reward difference：
$r(x,y_w)-r(x,y_l)$

$r(x,y_w)=\beta\log\frac{\pi^*(y_w|x)}{\pi_{\rm ref}(y_w|x)}+\beta\log Z(x)$

$r(x,y_l)=\beta\log\frac{\pi^*(y_l|x)}{\pi_{\rm ref}(y_l|x)}+\beta\log Z(x)$

两个相减：
$$
r(x,y_w)-r(x,y_l)
=\beta\log\frac{\pi^*(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-\beta\log\frac{\pi^*(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)},
$$

因为同一个 $x$ 下的 $\beta\log Z(x)$ 相减后消失。将未知的 $\pi^*$ 换成待训练的 $\pi_\theta$，就得到 §5.2 的 DPO loss。

推出了loss为什么长这个样子

![[lec8-dpo-loss-p58.png|900]]

*图：课件把 Bradley--Terry 的 reward loss、reward--policy transformation 和最终 DPO policy loss 叠在一起；来源：`lecture/lecture8post.pdf` 物理 PDF 第 58 页（课件页 41/44）。关键是 preferred/losing 两个 reward 的差使 log-partition term cancellation。*

这也解释了 DPO 的“隐式 reward”：

$$
r_\theta^{\mathrm{implicit}}(x,y)
=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)},
$$

加上只依赖 $x$ 的任意常数并不会改变 pairwise comparison。因此 DPO 没有恢复唯一的真实 reward，它是在 policy space 中直接拟合与偏好数据相容的相对 reward。


### 5.5 Assignment 3：从文本 DPO 到 Hopper action sequence

Assignment 3 明确说：原始 DPO 面向 bandit，但作业用同一个思想处理 MuJoCo action sequence。给定观察 $x$，模型不只预测一个动作，而是预测接下来 `segment_len` 步的 action plan；执行时只采取计划的第一个动作，再重新观察并规划，这叫 receding-horizon control / model predictive control（MPC）。

如果一次生成完整 episode 的 open-loop action sequence，环境扰动和早期误差会让后续动作失去适应性；receding horizon 每一步重新规划，保留了反馈回路。它改善的是作业适配的控制鲁棒性，不意味着 DPO 本身已经变成通用 model-based planning 算法。

starter code 中的关键映射如下：

| 数学对象 | `assignment3_starter_code` 中的位置 | 形状/实现含义 |
|---|---|---|
| $\pi_\theta(a_{t:t+H-1}\mid o_t)$ | `ActionSequenceModel.distribution` | 用 `Independent(Normal(mean, std), 2)` 把 segment 与 action 维的 log-probability 聚合为每个样本一个标量 |
| mean 与 log-std | `ActionSequenceModel.forward` | network 输出 `[batch, 2 * segment_len * action_dim]`，前半 reshape 为 mean，后半 reshape 为 log-std；mean 用 `tanh`，std 用 clamped log-std 的 `exp` |
| SFT objective | `SFT.update` | 最大化 preferred action sequence 的平均 log-probability；代码用 optimizer 最小化其负值 |
| DPO winner/loser | `get_batch` | 数据 `label=1` 时交换两条 sequence，使 `actions_w` 始终是第二条；平局在 `strict_pref_only=True` 时被排除 |
| DPO loss | `DPO.update` | 计算 current/reference 的 winner 与 loser log-ratio，代入 `-logsigmoid(beta * (ratio_w - ratio_l))` |
| reference policy | `DPO.update` 的 `ref_policy` | 用 `torch.no_grad()` 读取冻结 reference log-probability，不参与梯度更新 |

> [!important] 三个实现边界
> 1. 计算 Gaussian action sequence 的 log-probability 时，必须把 action/event 维聚合成每个样本一个标量；不能把 `[batch, segment_len, action_dim]` 直接与 `[batch]` preference loss 广播。
> 2. DPO 中 reference policy 的 log-probability 必须冻结；只有 current policy 的 log-probability 接收梯度。
> 3. Assignment 3 的 DPO 适配是离线 preference learning 加 receding-horizon execution。它不能直接当成 Lecture 7 的 full-trajectory off-policy correction，也不自动解决数据覆盖、偏好噪声和分布外状态问题。

Assignment 3 还要求比较 RLHF 与 DPO 的训练曲线和视频。可以这样组织判断：RLHF 先拟合 reward，再用环境交互和 PPO 优化，表达能力与在线纠错更强，但成本和 reward hacking 风险更高；DPO 直接使用离线 preference data，训练目标更简单、无需单独 reward model，但更依赖数据覆盖、reference policy 和 action-sequence distribution，不能自动保证闭环控制成功。

### 5.6 CPL 与课程边界

课件的结果页用一个受控实验说明 DPO 的训练行为：从 GPT2-XL 生成 positive IMDB reviews，用预训练 sentiment classifier 作为 gold reward model，再由该模型生成 preference pairs，最后比较 PPO 与 DPO 的 reward/KL trade-off。随后几页展示 DPO 训练出的模型以及 Mistral、LLaMA3 等规模化示例；这些页面用于展示实验方向和可观察指标，不能单独推出 DPO 在所有任务上都优于 PPO 或 SFT。

课件最后展示 Contrastive Preference Learning（CPL）作为后续方向，并指出 CPL 给出了 DPO 在 bandit setting 下的一个视角。这里的目的只是标记研究路线：本讲完整推导的是 Bradley--Terry、KL-regularized optimal policy 与 DPO，不把 CPL 的多步 RL 理论细节冒充为本讲已讲内容。



## 6. Assignment Readiness

### 6.1 Assignment 3 的前置覆盖

- **已覆盖**：Lecture 7 的 BC、IRL、feature/reward ambiguity、pairwise preference 入口；本讲的 Bradley--Terry、reward-model cross-entropy、KL-regularized RLHF、DPO 推导和 assignment action-sequence 适配。
- **需要结合题面继续阅读**：Hopper reward engineering 的具体奖励项、MuJoCo 环境目标、数据集视频质量、随机种子实验和作业要求的 plot/render 命令。
- **代码入口**：`run_rlhf.py` 的 `RewardModel` 负责 state-action reward 与 trajectory preference loss；`run_dpo.py` 的 `ActionSequenceModel`、`SFT.update` 和 `DPO.update` 负责序列 policy、监督预训练和 DPO loss。
- **mastery evidence**：目前没有观察到独立推导 reward-model gradient、实现并测试 `RewardModel.update`、跑通 PPO/DPO 三个 seed 或比较视频。

现在可以开始写 Assignment 3 的理论与代码部分；这表示 Lecture 7--8 的概念前置已覆盖，不表示 Hopper 实验、代码调参和结果解释已经完成。

### 6.2 推荐的练习顺序

1. 先手算 Bradley--Terry 的 $p=\sigma(R_1-R_2)$、cross-entropy gradient 和 assignment label 转换。
2. 在 `run_rlhf.py` 中先确认 reward model 的输出范围、trajectory reward 的累计维度和 `label=0/1/0.5` 语义，再实现更新。
3. 在 `run_dpo.py` 中先测试 `ActionSequenceModel.distribution` 的 log-probability shape，再实现 SFT，最后实现冻结 reference 的 DPO。
4. 每个算法用三个 seed 画原始 reward、learned reward 或 return 曲线，再用视频检查“高 reward”是否对应人类真正想要的行为。

## 7. 本讲必会公式

以下只列公式入口；完整语义、假设和数值计算见对应小节。

1. Bradley--Terry pairwise probability：

   $$
   P(b_i\succ b_j)=\sigma(r(b_i)-r(b_j)).
   $$

2. Trajectory preference model：

   $$
   R_\phi(\tau)=\sum_t\hat r_\phi(o_t,a_t),
   \qquad
   \widehat P_\phi(\tau_1\succ\tau_2)=\sigma(R_\phi(\tau_1)-R_\phi(\tau_2)).
   $$

3. Bradley--Terry cross-entropy gradient：

   $$
   \frac{\partial\ell}{\partial z}=\sigma(z)-\mu,
   \qquad z=R_\phi(\tau_1)-R_\phi(\tau_2).
   $$

4. RLHF KL-regularized objective：

   $$
   \max_\pi\;
   \mathbb E[r_\phi(x,y)]
   -\beta\mathbb E_x[D_{\mathrm{KL}}(\pi(\cdot\mid x)\|\pi_{\mathrm{ref}}(\cdot\mid x))].
   $$

5. KL-regularized optimal policy：

   $$
   \pi^*(y\mid x)=\frac{1}{Z(x)}\pi_{\mathrm{ref}}(y\mid x)\exp\left(\frac{r(x,y)}{\beta}\right).
   $$

6. DPO loss：

   $$
   \mathcal L_{\mathrm{DPO}}
   =-\mathbb E\left[
   \log\sigma\left(
   \beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
   -\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
   \right)
   \right].
   $$

## 8. 容易混淆点

- **DAGGER 与 pairwise preference**：DAGGER 的标签是专家在当前状态上的 action；pairwise preference 的标签只比较两个结果。后者减轻单步标注负担，但需要 reward model 从比较中推断长期质量。
- **Bradley--Terry 的绝对 reward**：偏好概率只依赖 reward difference；统一加常数不会改变数据 likelihood，因此不能从 pairwise data 识别 reward 的绝对零点。
- **Lecture 与 Assignment 的 label 编码**：讲义常用 $\mu=1$ 表示第一项赢；Assignment 3 的数据用 `0` 表示第一项赢、`1` 表示第二项赢。实现前必须转换。
- **reward model 与真实 reward**：$r_\phi$ 是有限偏好数据上的代理模型；held-out preference accuracy 只能检查预测一致性，不能证明它等于人类真实价值。
- **人类 preference 与 Condorcet/Copeland/Borda**：这些是不同的 pairwise ranking 汇总定义，不是同一个“最佳动作”定理。
- **RLHF 的 KL 方向**：本讲目标使用 $D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}})$；采样 log-ratio 的期望是 $\log\pi-\log\pi_{\mathrm{ref}}$，不能把分子分母反过来。
- **$\beta$ 的位置**：RLHF 目标中 $\beta$ 乘在 KL penalty 上；closed-form policy 中 reward 出现在 $\exp(r/\beta)$；DPO logit 中 $\beta$ 乘在 policy/reference log-ratio difference 上。
- **DPO 仍需要 reference**：DPO 省略显式 reward model，不省略 reference policy，也不省略偏好数据。
- **response/action sequence 的 log-probability**：文本 token 或连续 action sequence 的联合 log-probability 是各维/各时间步 log-probability 的和；preference loss 最终需要每个样本一个标量。
- **open-loop 与 receding horizon**：一次生成整段 action 并全部执行是 open-loop；每次只执行第一步并重新规划才是 assignment 使用的 receding-horizon control。
- **DPO 与多步 RL**：原始 DPO 是 bandit/contextual response 目标；Assignment 3 的 action-sequence 版本是课程作业适配，不应自动当成完整 trajectory-level RL 理论。

## 9. 自测题

### 题目

1. 为什么 behavior cloning 的单步错误会在 horizon $T$ 上产生近似 $O(\epsilon T^2)$ 的轨迹级风险？DAGGER 改变了哪一个分布？
2. 对 $r(b_1)=2,r(b_2)=1$，计算 Bradley--Terry 的 $P(b_1\succ b_2)$。
3. 证明给所有动作 reward 加同一个常数不会改变 Bradley--Terry probability。
4. 令 $p=\sigma(z)$，写出 label $\mu$ 的 binary cross-entropy 对 $z$ 的导数。
5. Lecture 的 $\mu=1$ 与 Assignment 3 的 `label=0` 各代表什么？为什么不能直接复用？
6. 为什么 reward model 训练好后还需要 PPO 或其他 policy optimization？
7. 写出 RLHF 的 KL-regularized objective，并解释 $\beta$ 变大时会发生什么。
8. 证明采样 reward $r_\phi(x,y)-\beta\log[\pi(y|x)/\pi_{\mathrm{ref}}(y|x)]$ 的期望等于 KL-regularized objective。
9. 从 KL-regularized objective 写出 $\pi^*(y|x)$，并解释 $Z(x)$ 为什么在语言模型中难以直接计算。
10. 写出 DPO loss，并指出其中哪一项使 log-partition term cancellation。
11. Assignment 3 为什么使用 action sequence 和 receding horizon，而不是一次生成完整 open-loop episode？
12. 比较 RLHF 与 DPO：它们分别显式学习了什么、需要哪些数据、主要风险是什么？

<details>
<summary>查看答案</summary>

1. 学习策略一次偏离后，之后访问的状态可能都不在专家数据分布中；每个后续位置都可能继承这次偏离，所以近似把每个起点的剩余 horizon 相加，得到 $\epsilon\sum_{k=1}^{T}k=O(\epsilon T^2)$。DAGGER 把训练数据的状态分布推向 learner-induced distribution，并在这些状态查询 expert action。
2. $P=\sigma(1)\approx0.731$。
3. $\exp(r_i+c)/(\exp(r_i+c)+\exp(r_j+c))$ 的分子分母都乘 $\exp(c)$，约掉后回到原式；所以只识别 reward difference。
4. $\partial\ell/\partial z=\sigma(z)-\mu=p-\mu$。
5. Lecture 的 $\mu=1$ 表示第一项胜出；Assignment 的 `label=0` 表示第一条序列胜出，编码方向相反，且 assignment 的 `0.5` 是平局。
6. reward model 只给每个行为评分，不会自动让 policy 在环境中选择更高分行为；PPO 等 policy optimizer 才负责根据该 reward 更新 policy。
7. $\max_\pi \mathbb E[r_\phi]-\beta\mathbb E_x[D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}})]$。$\beta$ 越大，远离 reference 的代价越重，最优 policy 通常越接近 reference。
8. 对每个 $x$，$\mathbb E_{y\sim\pi}[\log\pi(y|x)-\log\pi_{\mathrm{ref}}(y|x)]$ 正好是 $D_{\mathrm{KL}}(\pi(\cdot|x)\|\pi_{\mathrm{ref}}(\cdot|x))$；乘以 $-\beta$ 再加 reward 即得。
9. $\pi^*(y|x)\propto\pi_{\mathrm{ref}}(y|x)\exp(r(x,y)/\beta)$。$Z(x)$ 要遍历所有可能 response，语言模型的 response space 太大，不能直接枚举。
10. $\mathcal L_{\mathrm{DPO}}=-\mathbb E\log\sigma(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)})$。preferred 与 losing reward 做差时，同一 prompt 的 $\beta\log Z(x)$ 相减并消失。
11. 一次执行完整 open-loop plan 无法对环境扰动和 compounding errors 反应；receding horizon 每次只执行第一步并重新规划，保留反馈闭环。
12. RLHF 显式训练 reward model，再用 PPO 等 RL 优化；需要比较数据、reward model 训练和环境/策略优化，风险包括 reward-model misspecification、reward hacking 与在线训练成本。DPO 直接从 preference pairs 优化 policy，仍需 reference policy，少一个显式 reward-model 阶段，但更依赖离线数据覆盖、序列 log-probability 实现和适用的 bandit/控制近似。

</details>

## 10. 本讲小结

人类偏好学习把“请写出 reward”转成“请比较两个行为”，Bradley--Terry 再把这种比较写成 reward difference 的概率模型。对轨迹，把 state--action reward 累加后做 cross-entropy 拟合，就得到可以交给 PPO 的 reward model。

RLHF 把这条链扩展到语言模型：instruction tuning 提供初始 policy，pairwise comparisons 训练 reward model，KL-regularized PPO 同时追求 human-preference reward 和 reference stability。KL 项不是装饰，它控制 reward model 被策略利用时的偏离幅度。

DPO 从同一个 KL-regularized objective 出发，利用 closed-form optimal policy 把 reward difference 改写成 policy/reference log-ratio difference；因为 Bradley--Terry 只关心两个回答的 reward 差，$Z(x)$ 这个难以计算的 partition term 会抵消。它减少了显式 reward-model 和 PPO 阶段，但不消除 preference noise、distribution shift、reference choice 或 closed-loop control 的问题。

## 11. 延伸阅读

### 经典基础

- Bradley, *Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons*, 1952：Bradley--Terry paired-comparison model 的经典来源。
- Knox and Stone, *Interactively Shaping Agents via Human Reinforcement*, 2008：TAMER 方向，课件第 68 页作为 human-in-the-loop reward learning 参考。
- Christiano et al., *Deep Reinforcement Learning from Human Preferences*, 2017：从人类比较训练 reward model 并优化 RL policy 的代表性工作。
- Ouyang et al., *Training language models to follow instructions with human feedback*, 2022；Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, 2023：RLHF pipeline 与 DPO 的代表性论文。

### 前沿动态

课件最后把 human-preference learning 与 social choice、computational economics 和 AI 联系起来，并指向 Stanford CS329H 作为后续课程方向；这属于学习路线提示，不是本讲新增算法。

截至 2026-08-04 核实：本讲不额外列独立的前沿动态条目。CPL、RLHF 受控比较和 preference-learning 方向已作为课件边界或延伸入口记录；加入未经本地课程材料支持的近期结论会降低 source fidelity。
