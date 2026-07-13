# Stanford CS234 Winter 2026 本地材料清单

最后同步：2026-07-11

本文件记录官网公开材料与本地文件的对应关系，供后续 AI 或人工检查来源。课程材料以官网当前活跃链接为准；HTML 注释中的旧链接、目录中的废弃版本和需要 Stanford 登录的内容不计入“公开材料已补齐”。

## 权威来源

- 课程主页：<https://web.stanford.edu/class/cs234/index.html>
- Lecture Materials：<https://web.stanford.edu/class/cs234/modules.html>
- 官方 slides 目录：<https://web.stanford.edu/class/cs234/slides/>
- Assignments：<https://web.stanford.edu/class/cs234/assignments.html>
- Course Project：<https://web.stanford.edu/class/cs234/project.html>

## Lecture slides

官网当前活跃页面共对应 27 份 Lecture 1–14 slides，以及 2 份 guest PDFs。Lecture 10 明确标为 `no preclass`，所以只有 post 版本。

| Lecture | Pre-class | Post-class / annotated | 课件主题或备注 |
|---:|---|---|---|
| 1 | `lecture/lec1/lecture1pre.pdf` | `lecture/lec1/lecture1post.pdf` | Introduction to RL |
| 2 | `lecture/lec2/lecture2pre.pdf` | `lecture/lec2/lecture2post.pdf` | Tabular MDP planning |
| 3 | `lecture/lec3/lecture3pre.pdf` | `lecture/lec3/lecture3post.pdf` | Model-Free Policy Evaluation |
| 4 | `lecture/lecture4pre.pdf` | `lecture/lecture4post.pdf` | Model-Free Control and Function Approximation |
| 5 | `lecture/lecture5pre.pdf` | `lecture/lecture5post.pdf` | Policy Gradient I |
| 6 | `lecture/lecture6pre.pdf` | `lecture/lecture6post.pdf` | Policy Gradient II |
| 7 | `lecture/lecture7pre.pdf` | `lecture/lecture7post.pdf` | Policy Gradients and Imitation Learning |
| 8 | `lecture/lecture8pre.pdf` | `lecture/lecture8post.pdf` | Imitation Learning / Learning from Human Input / Batch RL |
| 9 | `lecture/lecture9pre.pdf` | `lecture/lecture9post.pdf` | Data-Efficient Reinforcement Learning |
| 10 | — | `lecture/lecture10post.pdf` | Fast Reinforcement Learning；官网没有 pre-class 版本 |
| 11 | `lecture/lecture11pre.pdf` | `lecture/lecture11post.pdf` | Fast Reinforcement Learning；封面误写 `Lecture 13`，下一行注明 `Typo: Lecture 11` |
| 12 | `lecture/lecture12pre.pdf` | `lecture/lecture12post.pdf` | Fast RL Continued |
| 13 | `lecture/lecture13pre.pdf` | `lecture/lecture13post.pdf` | Monte Carlo Tree Search |
| 14 | `lecture/lecture14pre.pdf` | `lecture/lecture14post.pdf` | Monte Carlo Tree Search |

Guest materials：

- `lecture/ethics_society_234_2.pdf`：Value Alignment Part II。
- `lecture/ShaneGuCS234_2026.pdf`：Shane Gu, *World of World Modeling*。

## Assignments

Winter 2026 官网只发布 Assignment 1–3；三份题目和公开 starter materials 现在均已在本地。

| Assignment | 本地题目 | 本地模板 / starter code | 官网来源 |
|---:|---|---|---|
| 1 | `assignment/assignment1/CS234_A1_Questions.pdf` | `assignment/assignment1/assignment1_template /`、`assignment/assignment1/code/` | <https://web.stanford.edu/class/cs234/assignments/a1/CS234_A1_Questions.pdf> |
| 2 | `assignment/ass2/CS234_A2_Questions.pdf` | `assignment/ass2/assignment2_template/`、`assignment/ass2/assignment2_starter_code/` | <https://web.stanford.edu/class/cs234/assignments/a2/CS234_A2_Questions.pdf> |
| 3 | `assignment/ass3/CS234_A3_Questions.pdf` | `assignment/ass3/assignment3_template/`、`assignment/ass3/assignment3_starter_code/`；原始 ZIP 同目录保留 | <https://web.stanford.edu/class/cs234/assignments/a3/hw3_questions.pdf> |

Assignment 3 starter code 的官网链接是：

<https://drive.google.com/file/d/18HwwLiMIN9XSdK7QXqQjGyhyb_86Iz_Y/view?usp=drive_link>

### 上游日期不一致

Assignment 3 官网列表显示截止时间为 **2026-02-20**，但它当前链接的 PDF 首页仍写 **2025-02-20**。本地保存的是 Winter 2026 Assignments 页面实际指向的文件，没有擅自修改 PDF；学习内容以该文件为准，课程年份与截止日期以 Winter 2026 官网列表为准。

## 官网指定的本地参考资料

- `lecture/lec1/cs229-linalg.pdf`：Stanford CS229 Linear Algebra Review。
- `lecture/lec1/cs229-prob.pdf`：Stanford CS229 Probability Review（此前已存在）。
- `lecture/references/RLbook2018.pdf`：Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed. final PDF。
- `lecture/references/Bandit-Algorithms.pdf`：Lattimore & Szepesvári, *Bandit Algorithms*；课程指定 §7.1。

## 未下载或有意排除

- Lecture videos 指向 Stanford Canvas，需要课程权限，未下载。
- `lecture9post_original.pdf` 虽可在 slides 目录看到，但活跃课程页使用修订后的 `lecture9post.pdf`；为避免混淆，没有保存旧版。
- `guest_slides1.pdf`、`guest_slides2.pdf`、`lecture16pre.pdf`、`lecture16post.pdf` 只出现在官网 HTML 注释中，且不属于当前活跃材料清单，因此未下载。
- Python/Numpy tutorial、David Silver 课程等网页资源保留在线链接，没有伪装成离线 PDF。
- Midterm、quiz、Gradescope 内容和 Canvas 内容没有公开下载链接。

## 完整性检查

- 29/29 份活跃 lecture/guest PDFs 已存在。
- Assignment 1–3 题目均存在；Assignment 3 题目为 8 页。
- Assignment 3 template 与 starter-code ZIP 均通过 `unzip -t` CRC 检查，并已安全解压。
- 本次下载先进入暂存目录，逐文件比较后以“不覆盖已有文件”的方式合并。
- 新增 PDF 均通过 `file` 与 `pdfinfo` 格式检查；部署文件与下载暂存文件逐字节一致。
