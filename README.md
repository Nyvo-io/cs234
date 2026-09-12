<p align="center">
  <img src="attachments/cs234-rl-loop.png" alt="Reinforcement learning states and actions" width="760">
</p>

# Stanford CS234: Reinforcement Learning

<p align="center">
  <img src="https://img.shields.io/badge/English-current-0969da?style=for-the-badge" alt="Current page is English">
  <a href="README_ZH.md"><img src="https://img.shields.io/badge/中文-阅读中文版-2ea44f?style=for-the-badge" alt="阅读中文版"></a>
</p>

This repository is a personal study archive for Stanford CS234: Reinforcement Learning. It keeps the public lecture slides, Chinese study notes, assignment implementations, experiment artifacts, and learning checkpoints together in a course-oriented layout.

The repository is organized around lectures and assignments rather than a single textbook. Lecture coverage and demonstrated mastery are tracked separately: a completed note means that the source material has been reviewed, while independent derivations, implementations, experiments, and quizzes provide evidence of mastery.

**[Lecture Map](#lecture-map)** · **[Notes and Concepts](#notes-and-concepts)** · **[Assignments and Code](#assignments-and-code)** · **[Current Checkpoint](#current-checkpoint)** · **[Course Materials](#course-materials)**

## Lecture Map

| Lecture | Topic | Slides | Notes |
| :--- | :--- | :--- | :--- |
| 01 | Introduction to RL | [Slides](lecture/lec1/lecture1post.pdf) | [Notes](notes/lec_notes/lec1_notes.md) |
| 02 | Tabular MDP Planning | [Slides](lecture/lec2/lecture2post.pdf) | [Notes](notes/lec_notes/lec2_notes.md) |
| 03 | Model-Free Policy Evaluation | [Slides](lecture/lec3/lecture3post.pdf) | [Notes](notes/lec_notes/lec3_notes.md) |
| 04 | Model-Free Control and Function Approximation | [Slides](lecture/lecture4post.pdf) | [Notes](notes/lec_notes/lec4_notes.md) |
| 05 | Policy Gradient I | [Slides](lecture/lecture5post.pdf) | [Notes](notes/lec_notes/lec5_notes.md) |
| 06 | Policy Gradient II | [Slides](lecture/lecture6post.pdf) | [Notes](notes/lec_notes/lec6_notes.md) |
| 07 | Policy Gradients and Imitation Learning | [Slides](lecture/lecture7post.pdf) | [Notes](notes/lec_notes/lec7_notes.md) |
| 08 | Imitation Learning, Human Input, and Batch RL | [Slides](lecture/lecture8post.pdf) | [Notes](notes/lec_notes/lec8_notes.md) |
| 09 | Data-Efficient Reinforcement Learning | [Slides](lecture/lecture9post.pdf) | [Notes](notes/lec_notes/lec9_notes.md) |
| 10 | Fast Reinforcement Learning | [Slides](lecture/lecture10post.pdf) | [Notes](notes/lec_notes/lec10_notes.md) |
| 11 | Fast Reinforcement Learning | [Slides](lecture/lecture11post.pdf) | [Notes](notes/lec_notes/lec11_notes.md) |
| 12 | Fast RL Continued | [Slides](lecture/lecture12post.pdf) | [Notes](notes/lec_notes/lec12_notes.md) |
| 13 | Monte Carlo Tree Search | [Slides](lecture/lecture13post.pdf) | [Notes](notes/lec_notes/lec13_notes.md) |
| 14 | Monte Carlo Tree Search | [Slides](lecture/lecture14post.pdf) | [Notes](notes/lec_notes/lec14_notes.md) |

<details>
<summary><strong>Pre-class slides for Lectures 1–14</strong></summary>

The pre-class and post-class slide decks are kept separately for comparison. Lecture 10 has no pre-class deck on the official site. Lecture 11's cover is mislabeled as Lecture 13, but the file content identifies it as Lecture 11.

- Lectures 1–3: [lec1](lecture/lec1/lecture1pre.pdf) · [lec2](lecture/lec2/lecture2pre.pdf) · [lec3](lecture/lec3/lecture3pre.pdf)
- Lectures 4–6: [lec4](lecture/lecture4pre.pdf) · [lec5](lecture/lecture5pre.pdf) · [lec6](lecture/lecture6pre.pdf)
- Lectures 7–9: [lec7](lecture/lecture7pre.pdf) · [lec8](lecture/lecture8pre.pdf) · [lec9](lecture/lecture9pre.pdf)
- Lectures 11–14: [lec11](lecture/lecture11pre.pdf) · [lec12](lecture/lecture12pre.pdf) · [lec13](lecture/lecture13pre.pdf) · [lec14](lecture/lecture14pre.pdf)

</details>

## Notes and Concepts

The notes follow the course material and add derivations, numerical examples, code mappings, and review questions. The [learning state](notes/learning_state.md), [concept index](notes/concept_index.md), [confusion log](notes/confusions.md), and [learning protocol](notes/learning_protocol.md) keep progress and study conventions explicit. See the [course environment guide](notes/cs234_environment.md) for local materials and tools.

## Assignments and Code

The assignment folders keep the original questions together with templates and starter code:

- [Assignment 1](assignment/assignment1/): value iteration, policy iteration, and RiverSwim.
- [Assignment 2](assignment/ass2/): REINFORCE, value baselines, PPO, and policy-induced distributions. See the [starter README](assignment/ass2/assignment2_starter_code/README.md) and [experiment results](assignment/ass2/assignment2_starter_code/code/results/).
- [Assignment 3](assignment/ass3/): preference learning, DPO, RLHF, and Hopper experiments. See the [starter README](assignment/ass3/assignment3_starter_code/README.md).
- [Assignments Review Guide](assignment/ASSIGNMENTS_REVIEW_GUIDE.md): prerequisites and review checkpoints.

## Current Checkpoint

- **Lecture coverage:** Lectures 1–14 have source-based notes and slide coverage.
- **Mastery evidence:** tracked separately and still being accumulated; coverage does not imply independent mastery.
- **Assignment 2:** the local policy-gradient, baseline, and PPO implementations are complete. The `assignment/` tree keeps the questions, templates, starter code, implementations, and seed-1 no-baseline, baseline, and PPO CartPole logs, score arrays, and plots. Multi-seed and official-submission checks remain open.
- **Next checkpoint:** review Lectures 13–14 and prepare for the next scheduled quiz. Assignment 3 environment work and ethics reading remain open; details are in the learning state.
- **Resume point:** start with the end of [learning_state.md](notes/learning_state.md), then open the relevant lecture note or assignment.

## Course Materials

The [Winter 2026 course-material index](COURSE_MATERIALS_WINTER_2026.md) maps local files to the active Stanford public materials and records version notes and completeness checks.

| Resource | Use |
| :--- | :--- |
| [Stanford CS234 course page](https://web.stanford.edu/class/cs234/index.html) | Course homepage and official information |
| [Lecture Materials](https://web.stanford.edu/class/cs234/modules.html) | Official lecture index |
| [Assignments](https://web.stanford.edu/class/cs234/assignments.html) | Assignment questions and requirements |
| [Sutton & Barto](lecture/references/RLbook2018.pdf) | Course reference textbook |
| [Bandit Algorithms](lecture/references/Bandit-Algorithms.pdf) | Bandit reference text, especially §7.1 |
| [World Models guest lecture](lecture/ShaneGuCS234_2026.pdf) | Shane Gu's World Models lecture |
| [Value Alignment guest lecture](lecture/ethics_society_234_2.pdf) | Value Alignment Part II |

Lecture decks, assignment questions, and reference PDFs come from public course materials. Canvas videos, quizzes, and grades that require Stanford login are not mirrored here.

<details>
<summary><strong>Repository structure</strong></summary>

```text
cs234/
├── README.md                           # English homepage (default)
├── README_ZH.md                        # Chinese homepage
├── lecture/                            # Slides, reference books, and guest materials
├── notes/
│   ├── lec_notes/                      # Lecture 1–14 study notes
│   ├── learning_state.md               # Single live checkpoint
│   ├── concept_index.md                # First complete explanation by concept
│   ├── confusions.md                   # Ongoing confusion log
│   └── learning_protocol.md            # Study and note conventions
├── assignment/                         # Assignment questions, templates, and starter code
├── attachments/                        # Course figures used in notes
└── COURSE_MATERIALS_WINTER_2026.md     # Official material and local-file index
```

Daily code, notes, and experiment results stay in their original folders. Before publishing, check `git status` and stage only files belonging to the current update.

</details>
