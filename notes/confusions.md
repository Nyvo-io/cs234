# CS234 Confusions

这个文件记录容易反复混淆、需要后续 lecture 继续检查的点。新对话继续学习前应先读取它。

**使用方法**：
- 学习新 lecture 前，先浏览这个文件，提前知道哪些地方容易混淆。
- 做 assignment 时，如果遇到符号约定不清楚的地方，查阅对应条目。
- 发现新的易混淆点时，及时添加到这个文件中。
- 每个条目应包括：概念对比、标准约定、做题策略。

## Active Confusions

1. **Horizon $H$ 的含义**

   在公式里，$H$ 可能表示整个 episode 的总步数，也可能表示从当前时间 $t$ 开始剩余的步数。两种约定会导致 return 的求和上限不同。

   如果 $H$ 表示 episode 总步数，且 $t=0,\ldots,H-1$：

   $$
   G_t
   =
   \sum_{k=0}^{H-1-t}
   \gamma^k r_{t+k}
   $$

   如果 $H_{\text{rem}}$ 表示从当前时间 $t$ 开始剩余的步数：

   $$
   G_t^{(H_{\text{rem}})}
   =
   \sum_{k=0}^{H_{\text{rem}}-1}
   \gamma^k r_{t+k}
   $$

   **本课程默认约定**：除非题目或 lecture 明确说明，否则假设 $H$ 表示 episode 总步数，时间索引 $t=0,1,\ldots,H-1$。从时间 $t$ 开始的 return 应写成 $\sum_{k=0}^{H-1-t} \gamma^k r_{t+k}$。

   **做题时的策略**：做 Assignment 1 Q1 的 effective horizon 计算时，先仔细阅读题目对 $H$ 或 $H_{\text{rem}}$ 的定义，再选择对应的公式形式。

2. **Reward vs Value**

   奖励（reward）是一步反馈；价值（value）是未来折扣累计奖励的期望。

3. **Observation vs State**

   观测（observation）是智能体看到的东西；状态（state）是用于预测未来和决策的表示。当前观测不一定满足 Markov 假设。

4. **Evaluation vs Control**

   评估（evaluation）是判断给定策略有多好；控制（control）是寻找更好的策略。

5. **Large $\gamma$ vs short-term rewards**

   大的折扣因子 $\gamma$ 不是更看重短期。相反，$\gamma$ 越接近 1，未来奖励衰减越慢，长期奖励越重要。$\gamma=0$ 才是只看即时奖励。

6. **$B$ vs $B^\pi$**

   $B$ 是 Bellman optimality operator，会对动作取最大值：

   $$
   (BV)(s)
   =
   \max_a
   \left[
   R(s,a)
   +
   \gamma
   \sum_{s'}P(s' \mid s,a)V(s')
   \right]
   $$

   $B^\pi$ 是固定策略的 Bellman operator，不对动作取最大值：

   $$
   (B^\pi V)(s)
   =
   R(s,\pi(s))
   +
   \gamma
   \sum_{s'}P(s' \mid s,\pi(s))V(s')
   $$

   Assignment 1 Q3 同时使用这两个 operator，证明时不要混用。

7. **Arbitrary $V$ vs policy value $V^\pi$**

   Bellman residual 中的 $V$ 可以是任意价值向量，不一定对应任何真实策略。$V^\pi$ 则是某个策略 $\pi$ 在 MDP 中真实诱导出的价值函数。

8. **Finite-horizon policy is usually time-dependent**

   Infinite-horizon discounted MDP 中存在 stationary optimal policy；finite-horizon MDP 中，最优动作通常依赖剩余步数。同一个状态在剩 10 步和剩 1 步时可能应选择不同动作。

9. **Reward indexing: $r_t$ vs $R_{t+1}$**

   不同课件或教材可能用 $r_t$ 表示在 $(s_t,a_t)$ 后观察到的奖励，也可能用 $R_{t+1}$ 表示同一个奖励。两者只要前后一致都可以，但会让 return 的下标看起来相差一位。

   **本笔记默认约定**：$r_t$ 是在状态 $s_t$ 采取动作 $a_t$ 后、转移到 $s_{t+1}$ 时观察到的奖励。若 episode 有 $T$ 次转移，$t=0,\ldots,T-1$，则：

   $$
   G_t
   =
   \sum_{k=0}^{T-1-t}\gamma^k r_{t+k}
   $$

   **做题策略**：先根据题目轨迹写出“状态、动作、奖励、下一状态”的对应关系，再套 return 公式。

10. **First-visit MC vs every-visit MC**

    First-visit 指一条 episode 按时间从前往后时，状态第一次出现的位置；不是从 episode 尾部反向计算 return 时最先遇到的位置。

    - First-visit MC：每条 episode 对每个状态最多贡献一个 return。
    - Every-visit MC：状态每次出现都贡献一个 return。

11. **Monte Carlo target vs TD target**

    MC 使用完整回报：

    $$
    G_t
    =
    r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots
    $$

    TD(0) 使用一步 bootstrap target：

    $$
    r_t+\gamma V(s_{t+1})
    $$

    $V(s_{t+1})$ 是当前估计，不是真实未来回报。因此 TD 通常有偏，但方差通常低于 MC。

12. **Unbiased vs consistent**

    无偏描述固定样本量下估计器的期望是否等于真值；一致描述样本量趋于无穷时，估计是否以概率收敛到真值。

    无偏不自动保证一致；有限样本有偏也不妨碍估计器最终一致。

13. **Batch MC vs batch TD**

    在固定有限数据集上反复更新：

    - Batch MC 收敛到数据中完整 return 的经验平均。
    - Batch TD 收敛到最大似然 Markov 模型的 Bellman fixed point。

    两者可能得到不同答案。Batch TD 能跨 episode 共享下一状态信息，但依赖状态表示满足 Markov 假设。

14. **Terminal value vs terminal transition reward**

    对终止状态通常定义 $V(s_{\mathrm{terminal}})=0$，表示终止之后没有未来奖励。这不代表进入终止状态的即时奖励为 0。TD target 在终止转移上应保留 $r_t$：

    $$
    y_t^{\mathrm{TD}}=r_t
    $$

15. **On-policy vs off-policy**

    On-policy 使用同一个策略 $\pi$ 产生数据并作为学习目标；off-policy 可以用行为策略 $\pi_b$ 采样、用另一目标策略 $\pi$ 学习。在线收集数据不等于 on-policy。

    **做题策略**：先写清楚“数据由谁产生”和“target 对应谁”，再给算法分类。SARSA 用实际下一动作，是 on-policy；Q-learning 用最大下一动作价值，是 off-policy。

16. **SARSA vs Q-learning 的下一动作**

    SARSA 的 target 是 $r+\gamma Q(s',a')$，其中 $a'$ 是行为策略实际采样的动作；Q-learning 的 target 是 $r+\gamma\max_{a'}Q(s',a')$，不要求实际执行最大动作。允许探索时，两者的 target 一般不同；$\varepsilon=0$ 且 tie-breaking 一致时，它们的某一步 target 才会相同。

17. **GLIE 与递减 epsilon**

    GLIE 同时要求每个 $(s,a)$ 被访问无限多次，以及行为策略极限上贪心。写出 $\varepsilon_i\to0$ 只满足第二个方向的直觉，不能单独保证第一条；还需要可达性和访问条件。

18. **函数逼近 TD 的 target 和梯度**

    $\hat V(s';w)$ 出现在 TD target 中，不代表把 target 当作真实无偏标签，也不代表更新时必须对 target 反向传播。课件的更新只对当前预测项求梯度，这种更新通常称为 semi-gradient TD。

19. **Deadly triad**

    function approximation、bootstrapping、off-policy learning 的组合可能导致振荡或发散。它是稳定性风险，不是“任意一个因素单独出现就必然失败”的定理。

20. **Lecture 4 Mars Rover epsilon 概率的课件冲突**

    课件前面的 epsilon-greedy 公式给贪心动作概率 $1-\varepsilon+\varepsilon/|\mathcal A|$。在两个动作、$\varepsilon=1/3$ 时应为 $2/3$，但第 82 页解答写成 $5/6$。本笔记采用前面明确的公式；遇到该页时应把它标为课件记号不一致，而不是修改标准概率公式。

21. **Policy objective 的 gradient ascent vs 代码中的 loss minimization**

    数学目标是最大化期望回报：

    $$
    \theta\leftarrow\theta+\alpha\nabla_\theta J(\theta).
    $$

    PyTorch optimizer 默认最小化 loss，因此实现 REINFORCE 时通常定义：

    $$
    L(\theta)
    =
    -\mathbb E[\log\pi_\theta(a_t\mid s_t)\hat A_t].
    $$

    **做题策略**：先写清要最大化的 objective，再检查代码是否对它取了负号；不要同时反转 advantage 和 loss 两次。

22. **Trajectory probability vs single-step policy probability**

    $\pi_\theta(a_t\mid s_t)$ 只是时刻 $t$ 的动作概率；$P(\tau;\theta)$ 是整条轨迹的概率，包含初始状态分布、所有动作概率和环境转移概率：

    $$
    P(\tau;\theta)
    =
    \mu(s_0)\prod_t\pi_\theta(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t).
    $$

    对数把乘积变成求和，所以轨迹 score 才能分解为逐时间步的 policy score。

23. **Policy-gradient baseline 的 action independence**

    标准 state baseline $b(s_t)$ 可以随状态变化，但不能依赖当时采样的动作 $a_t$。无偏性来自：

    $$
    \mathbb E_{a_t\sim\pi_\theta}
    [b(s_t)\nabla_\theta\log\pi_\theta(a_t\mid s_t)]
    =0.
    $$

    若 baseline 依赖 $a_t$，就不能把它移出对动作的期望，上述证明通常失效。

24. **Return、Q-value 与 advantage estimate**

    - $G_t$：一条实际轨迹上的随机 return sample；
    - $Q^\pi(s_t,a_t)$：给定状态动作后的期望 return；
    - $A^\pi(s_t,a_t)=Q^\pi(s_t,a_t)-V^\pi(s_t)$：真实 advantage；
    - $\hat A_t=G_t-b(s_t)$：用 sampled return 和估计 baseline 得到的 advantage estimate。

    **做题策略**：看到代码里的 `returns - baseline` 时写 $\hat A_t$，不要无条件写成精确的 $A^\pi$。

25. **Discounted REINFORCE 的 objective convention**

    有的推导从 $J=\mathbb E[\sum_t\gamma^t r_t]$ 出发，展开时会显式出现外层 $\gamma^t$；有的课程实现直接定义使用相对折扣 reward-to-go 的 estimator：

    $$
    G_t=\sum_{k=0}^{T-1-t}\gamma^k r_{t+k}.
    $$

    **本课程 Assignment 2 约定**：按题面给出的 $G_t$ 和 objective 实现，不自行增加外层 $\gamma^t$。阅读其他教材时先核对 objective 与状态访问分布约定，不能只比较公式外观。

26. **Lecture 3 Mars Rover first-visit 向量的课件冲突**

    Lecture 3 第 14 页给出的轨迹是：

    $$
    (s_3,a_1,0,s_2,a_1,0,s_2,a_1,0,s_1,a_1,1,\mathrm{terminal}).
    $$

    按本课程的 reward-on-transition 约定和标准 first-visit 定义，首次到达 $s_2$ 后还有奖励序列 $0,0,1$，因此：

    $$
    G_{\mathrm{first}\ s_2}=\gamma^2.
    $$

    相应 first-visit 状态向量应为 $[1,\gamma^2,\gamma^3,0,0,0,0]$，但第 36 页底部印成 $[1,\gamma,\gamma^2,0,0,0,0]$。

    **做题策略**：不要直接抄向量。先为每个 state visit 标出时间 $t$，写出从该位置开始的奖励，再用 $G_t=\sum_k\gamma^k r_{t+k}$ 计算；本笔记采用轨迹与标准定义导出的结果。

27. **任意无偏 baseline vs 方差最优 baseline**

    任意不依赖本次采样动作的 $b(s)$ 都满足

    $$
    \mathbb E[b(s)\nabla_\theta\log\pi_\theta(a\mid s)]=0,
    $$

    但严格的方差最优解一般是 score-norm 加权条件均值：

    $$
    b^*(s)
    =
    \frac{\mathbb E[\lVert\nabla\log\pi\rVert^2G_t\mid s]}
    {\mathbb E[\lVert\nabla\log\pi\rVert^2\mid s]}.
    $$

    $V^\pi(s)=\mathbb E[G_t\mid s]$ 是常用近似，不是一般情况下与 $b^*(s)$ 完全相同。

28. **PPO old policy vs current policy**

    Rollout policy $\pi_{\theta_k}$ 生成当前 batch，并提供冻结的 `old_logprobs`；current policy $\pi_\theta$ 在多个 update 中变化。Ratio 必须是：

    $$
    r_t(\theta)
    =
    \frac{\pi_\theta(a_t\mid s_t)}
    {\pi_{\theta_k}(a_t\mid s_t)}.
    $$

    **做题策略**：先在每个 policy 符号旁写“collect data”或“being optimized”。若用 current network 同时重算分子和分母，ratio 会错误地恒为 1。

29. **Action probability ratio vs state-distribution correction**

    $\pi'(a\mid s)/\pi(a\mid s)$ 只把固定状态下的动作期望从 $a\sim\pi$ 改成 $a\sim\pi'$。它没有把 $s\sim d^\pi$ 改成 $s\sim d^{\pi'}$。PPO surrogate 使用旧策略 states，并依赖新旧策略接近时 $d^{\pi'}\approx d^\pi$ 的近似。

    **做题策略**：看到 importance ratio 时分别检查 action distribution 和 state distribution；不能只因出现 ratio 就声称所有 off-policy mismatch 已被精确校正。

30. **PPO clipping vs hard trust region**

    Clipped objective 在 advantage 对应的有利方向超过 $1\pm\epsilon$ 后形成平台，但它不是对所有状态 probability ratio 或平均 KL 的硬约束，也不保证每次真实性能单调提高。

    - $\hat A_t>0$：主要截断 $r_t>1+\epsilon$ 的额外激励；
    - $\hat A_t<0$：主要截断 $r_t<1-\epsilon$ 的额外激励。

    **做题策略**：先看 advantage 符号，再判断哪一侧出现零梯度；不要把 clip 区间机械理解为 ratio 在两侧都一定停止变化。

31. **PPO advantage shape 与广播**

    Assignment 2 的 `ppo.py` docstring 把 `advantages` 写成 `[batch, 1]`，但上游 `calculate_advantage` 和 log-probability 使用 `[batch]`。若计算 `[batch] * [batch,1]`，PyTorch 会广播为 `[batch,batch]`，数值可能运行但含义完全错误。

    **做题策略**：在 `PPO.update_policy` 中断言 `new_logprobs.shape == old_logprobs.shape == advantages.shape == (batch_size,)`，再计算 ratio 和两个 surrogate。

32. **PPO limited batch reuse vs arbitrary off-policy replay**

    PPO 会对最近 old policy 的 batch 做有限次 minibatch update，然后重新采样。它不是像 DQN replay buffer 那样长期混合任意陈旧数据。

    **做题策略**：算法分类时写清数据的新旧程度、ratio correction 和更新次数；“一批数据用了多次”本身不足以说明可以无限 off-policy 学习。

33. **Lecture 4 Mars Rover MC action-value 向量的课件冲突**

    Lecture 4 第 26 个物理页面给出的轨迹包含两次 $(s_2,a_2)$：

    $$
    (s_3,a_1,0,s_2,a_2,0,s_3,a_1,0,s_2,a_2,0,s_1,a_1,1,\mathrm{terminal}).
    $$

    在 $\gamma=1$ 下，第一次访问 $(s_2,a_2)$ 后的剩余奖励之和为 1，因此按 $s_1,\ldots,s_7$ 排列应有

    $$
    Q(-,a_2)=[0,1,0,0,0,0,0].
    $$

    但同页印出的 $Q^{\varepsilon\text{-}\pi}(-,a_2)$ 是全零向量，与轨迹和 first-visit 定义不一致。

    **做题策略**：从轨迹逐个标出首次出现的 $(s,a)$，再从该位置向后计算 $G_t$；不要直接抄该页的 $a_2$ 向量。

34. **Lecture 7 示范轨迹记号的重复状态符号**

    Lecture 7 第 30 页的问题设置把示范轨迹文字写成类似 $(s_0,a_0,s_1,s_0,\ldots)$，但上下文是在列出状态—动作序列，第二个重复的 $s_0$ 应按标准轨迹记号理解为下一步动作 $a_1$，即 $(s_0,a_0,s_1,a_1,\ldots)$。

    **做题策略**：后续推导统一使用 $(s_t,a_t,s_{t+1})$；不要让该排版/录入问题改变 demonstration、BC 或 IRL 的状态—动作配对。

35. **Lecture 8 Bradley--Terry label vs Assignment 3 label**

    讲义的交叉熵约定通常是 $\mu=1$ 表示第一项胜出、$\mu=0.5$ 表示平局、$\mu=0$ 表示第二项胜出。Assignment 3 的 preference dataset 则用 `label=0` 表示第一条序列胜出、`label=1` 表示第二条序列胜出、`label=0.5` 表示平局；`run_dpo.py:get_batch` 还会据此交换 winner/loser。

    **做题策略**：在实现 `RewardModel.update` 前先把 dataset label 映射到公式中的 $\mu$，不要把 `label=0` 直接当成“第一项概率为 0”的 cross-entropy target；在 DPO 中只使用 strict preference 数据，并确认 `actions_w` 始终是偏好序列。
