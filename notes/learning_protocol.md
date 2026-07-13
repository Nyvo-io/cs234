# CS234 Learning Protocol

这个文件只记录稳定的学习偏好和课程约定；具体教学模板、Golden Examples 与 QA 流程由 `cs234-rl-tutor` skill 管理。

## Core Goal

1. 按 CS234 lecture 顺序系统学习强化学习（reinforcement learning, RL）。
2. 以本地 lecture PDF、assignment specification 和 starter code 为主线。
3. 每次默认只学习一份 lecture；assignment 在相关前置知识覆盖后用于巩固。
4. 重点是概念、数学、算法、实现和迁移能力，不是背英文词汇。
5. 适当连接 embodied intelligence、robot learning、LLM agents、RLHF/PPO 和 planning，但课程内容始终优先。

## Output Policy

- 对话或终端用于简洁互动、提问和反馈。
- 完整讲义写入 `notes/lec_notes/lecN_notes.md`。
- 每讲结束更新 `notes/learning_state.md` 和 `notes/concept_index.md`。
- 只有出现真实、可持续跟踪的混淆时才更新 `notes/confusions.md`。
- 不依赖 terminal scrollback 保存学习成果。

## First Teaching and Repetition

首次完整讲解陌生或重要的公式、概念、算法时：

1. 准确给出概念或公式，并说明必要假设与索引约定。
2. 用通顺的中文解释它在计算什么、表示什么或解决什么问题。
3. 紧接一个最小而完整的具体例子：代入数字、展示轨迹或逐步计算。
4. 解释结果的含义、边界情况和常见误解。
5. 只有确实相关时才连接 assignment 或 starter code。

不要求每个概念机械套用固定六段标签。简单概念可以短讲；陌生公式和易混淆概念必须讲透。

概念第一次完整讲解后，在 `notes/concept_index.md` 登记最终小节位置。后续再次出现时，使用：

```markdown
*首次完整讲解：Lecture X §Y「小节标题」。本节只补充：……*
```

然后只解释本次新增内容。如果仍然不理解，可以重新完整讲解，但保留首次位置。

## Formula and Notation Policy

- 非平凡公式使用 LaTeX display math；不使用 Unicode 上标或下标。
- 单字符上标写作 `V^\pi(s)`；多字符或嵌套上标使用花括号，如 `V^{\pi_{\text{old}}}(s)`。
- 带时间下标的公式先说明 reward timing、总 episode 长度、剩余 horizon 和终止状态约定。
- 如果 $H$ 表示总步数且 $t=0,\ldots,H-1$，则：

  $$
  G_t=\sum_{k=0}^{H-1-t}\gamma^k r_{t+k}
  $$

- 有限时域价值通常写为 $V_t(s)$ 或 $V_h(s)$；不要无说明地把它写成 stationary $V(s)$。
- 检查表达式的数学类型：动作不是策略，样本 return 不是期望 value，标量不是价值向量，任意 $V$ 不一定是 $V^\pi$。

## Source Fidelity

- 具名课程环境（Mars Rover、RiverSwim 等）的状态、动作、奖励和转移必须对照 PDF 或代码。
- 自己设计的教学例子必须标注 `自拟例子`，不能借用课程环境的名字。
- 课件原文、由定义推出的解释、现代直接联系和概念类比应清楚区分。
- PDF 无法可靠读取时不要编造；应重新提取对应页面或请求截图。

## Language Policy

- 主要使用中文。
- 重要术语自然标注为“中文术语（English term）”，例如策略（policy）、回报（return）、价值函数（value function）。
- 不堆砌长词汇表，不为保留英文破坏中文语序。
- 现代连接只在真正帮助理解时出现，并明确它是课程内容、直接应用还是类比。

## Coverage and Mastery

- Lecture checklist 只表示课件内容已覆盖，不表示已经掌握。
- `learning_state.md` 分开记录 coverage 与 mastery evidence。
- Mastery 必须有证据，例如完成自测、口头解释、独立推导、实现或 assignment。
- 自测题先给问题，答案放在可折叠区域或等回答后再揭示。

## Assignment Workflow

1. 解释题意和考点。
2. 确认前置知识与当前尝试。
3. 推导公式和边界情况。
4. 映射到 starter code 的真实变量与 shape。
5. 用最小数值测试或课程测试验证实现。
6. 默认采用提示和逐步指导；是否给完整成品答案服从用户要求与课程政策。

理论到代码的常见映射：

- $R(s,a)$ → `R[state, action]`
- $P(s'\mid s,a)$ → `T[state, action, next_state]`
- $V(s')$ → `V[next_state]`
- $Q(s,a)$ → `R[state, action] + gamma * np.sum(T[state, action, :] * V)`
