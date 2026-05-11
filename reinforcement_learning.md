# 🤖 行为学派（二）：强化学习 复习笔记

> 一句话说清强化学习：**智能体（Agent）在环境（Environment）中试错，做动作（Action），获得奖励（Reward），学会让累计奖励最大化的策略。** 🎮 → 🧠 → 🏆

---

## 📌 知识地图（先看整体框架）

```
🧠 强化学习 (Reinforcement Learning)
│
├── 🎯 核心概念
│     ├── Agent（智能体）— 做决策的"你"
│     ├── Environment（环境）— 你所在的"世界"
│     ├── Action（动作）— 你做的每一步操作
│     ├── State（状态）— 当前世界的情况
│     ├── Reward（奖励）— 你做对了/做错了的反馈
│     └── Policy（策略）— 你在每个状态下该做什么
│
├── 📐 数学框架
│     ├── MDP（马尔可夫决策过程）
│     ├── Value Function（价值函数）
│     └── Bellman Equation（贝尔曼方程）
│
└── 🚀 重要里程碑
      ├── DQN（玩 Atari 游戏）
      ├── AlphaGo / AlphaZero（下围棋）
      └── OpenAI Five / ChatGPT（大规模 RL）
```

---

## 一、🎯 强化学习核心概念

### 知识点 1：强化学习 vs 其他学习范式

| 学习方式 | 比喻 🎭 | 数据来源 | 反馈 |
|:--------|:--------|:--------|:----|
| **监督学习** 📚 | 老师教学生 | 带标签的数据集 | 即时（每道题都有正确答案） |
| **无监督学习** 🔍 | 自己整理房间 | 无标签数据 | 无反馈 |
| **强化学习** 🎮 | 玩游戏自己摸索 | 与环境的交互 | **延迟**（可能要很多步才有奖励） |

### 知识点 2：RL 的基本要素（六个核心概念 ⭐）

```
         🎮 Agent（智能体）
           │
           │  Action（动作）a_t
           ▼
  ┌──────────────────┐
  │  Environment     │
  │  （环境）        │
  └──────────────────┘
           │
           │  ① State（状态）s_{t+1}
           │  ② Reward（奖励）r_{t+1}
           ▼
         🎮 Agent（更新策略）
```

| 概念 | 符号 | 比喻 🏫 | 定义 |
|:---- |:---- |:------- |:---- |
| **Agent** 🤖 | - | 学生 | 做决策的智能体 |
| **Environment** 🌍 | - | 学校/考场 | Agent 交互的外部世界 |
| **State** 📊 | s_t | 你当前的知识水平 | 环境的当前状况 |
| **Action** 🎮 | a_t | 你决定学哪科 | Agent 做出的操作 |
| **Reward** 🏆 | r_t | 考试成绩（分数） | 环境给 Agent 的反馈信号 |
| **Policy** 📋 | π(a\|s) | 你的学习方法论 | 在状态 s 下做什么动作的策略 |

### 知识点 3：RL 的学习循环

```
智能体观察环境 → 根据策略选择动作 → 环境改变并返回新状态+奖励 → 智能体更新策略

Repeat!

目标是：找到最优策略 π*，使累计奖励（Return）最大！🏆
```

---

## 二、🧠 强化学习的生物学基础

### 知识点 4：RL 与心理学/神经科学的联系

RL 不是凭空出现的，它在人类和动物身上也有对应：

| 理论 | 提出者 | 时间 | 核心内容 |
|:---- |:------ |:---- |:-------- |
| **效果律** ⚖️ | Thorndike | 1911 | 带来好结果的行为会被强化，坏结果的会被削弱 |
| **条件反射** 🔔 | Pavlov | 1927 | 狗听到铃声流口水 — 刺激和反应建立联系 |
| **操作条件反射** 🐭 | Skinner | 1938 | 老鼠压杠杆得到食物 → 学会压杠杆 |
| **赫布理论** 🔗 | Hebb | 1961 | **Neurons that fire together, wire together.** 同时激活的神经元连接被强化 |
| **多巴胺奖励** 💊 | Carlsson | 1957/2000 诺奖 | 多巴胺是大脑的"奖励信号"，和 RL 的 Reward 直接对应 |

> 💡 RL 不仅是计算机科学，也是理解动物和人类行为的重要理论框架。

---

## 三、📐 马尔可夫决策过程（MDP）（重点 ⭐）

### 知识点 5：MDP 的定义

**MDP = Markov Decision Process**，是 RL 问题的标准数学框架。

**MDP 用五元组表示**：⟨S, A, P, R, γ⟩
- **S** — 状态集合（所有可能的状态）
- **A** — 动作集合（所有可能的动作）
- **P** — 状态转移概率 P(s'|s, a)（做了动作 a 后到 s' 的概率）
- **R** — 奖励函数 R(s, a)（在 s 做 a 得到的即时奖励）
- **γ** — 折扣因子（0~1，未来的奖励折算到现在值多少）

### 知识点 6：马尔可夫性质（Markov Property）

**核心定义**：下一时刻的状态**只取决于当前状态和动作**，和之前的历史无关。

> P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1} | s_t, a_t)

就像下棋 ♟️：你不需要知道之前怎么走到这一步的，只看**当前棋盘**就能决定下一步怎么走。

### 知识点 7：完全可观测 vs 部分可观测

| 类型 | 说明 | 例子 |
|:---- |:---- |:---- |
| **完全可观测（MDP）** 👁️ | Agent 能看到环境的全部状态 | 围棋（棋盘一目了然） |
| **部分可观测（POMDP）** 👓 | Agent 只能看到部分状态 | 德州扑克（看不到对手的牌） |

> 💡 **POMDP** = Partially Observable Markov Decision Process。这种情况下 Agent 需要记住历史信息来推断真实状态。

---

## 四、💰 奖励、回报和价值函数

### 知识点 8：Reward（奖励）vs Return（回报）

**Reward（奖励）** r_t：每一步的即时反馈 📨
**Return（回报）** G_t：从当前步开始到结束的**累计折扣奖励** 🧮

```
G_t = r_{t+1} + γ·r_{t+2} + γ²·r_{t+3} + ...
```

**折扣因子 γ 的意义：**
- γ → 0：只看眼前利益（短视 👀）
- γ → 1：有远见，未来的奖励也很重要（深谋远虑 🧐）
- 一般取值 0.9~0.99

### 知识点 9：Value Function（价值函数）—— 核心概念 ⭐

**价值函数 = 预测未来能拿多少奖励** 🔮

**两种价值函数：**

| 函数 | 符号 | 含义 |
|:---- |:---- |:---- |
| **状态价值函数** 📊 | V(s) | 从状态 s 出发，按策略 π 行动，能拿到的期望回报 |
| **动作价值函数** 🎯 | Q(s, a) | 在状态 s 做了动作 a，之后按策略 π 行动，能拿到的期望回报 |

**公式：**
```
V_π(s) = E[G_t | S_t = s]                      = E[r₁ + γ·r₂ + γ²·r₃ + ... | S_t = s]
Q_π(s, a) = E[G_t | S_t = s, A_t = a]          = E[r₁ + γ·r₂ + γ²·r₃ + ... | S_t = s, A_t = a]
```

> 💡 **V(s) 和 Q(s,a) 的关系**：Q(s,a) 比 V(s) 多指定了第一步的动作。知道了 Q(s,a)，选最大 Q 值的动作就是最优策略。

### 知识点 10：最优价值函数与最优策略

RL 的最终目标：找到**最优策略 π\***，使得 V(s) 或 Q(s,a) 最大。

```
V*(s) = max_π V_π(s)     — 从状态 s 出发能得到的最大累计奖励
Q*(s,a) = max_π Q_π(s,a) — 在 s 做 a 能得到的最大累计奖励
```

**确定最优策略**：在每个状态 s，选 Q*(s,a) 最大的那个动作！
```
π*(s) = argmax_a Q*(s,a)
```

---

## 五、🚀 RL 发展里程碑

### 知识点 11：经典 RL 发展时间线

```
1950s  🐣 Bellman → 动态规划、MDP 理论基础
1959   🐣 Samuel → 跳棋程序（第一个 RL 应用 🏓）
1980s  📈 Barto & Sutton → 时序差分学习（TD Learning）
1989   📈 Watkins → Q-Learning 提出
1995   🚀 Tesauro → TD-Gammon 西洋双陆棋达到人类顶级水平
2013   💥 DeepMind → DQN 玩 Atari 游戏（深度强化学习诞生！）
2016   💥 DeepMind → AlphaGo 击败李世石 🏆
2017   💥 AlphaZero → 不用人类知识，从零自学围棋/象棋
2019   💥 OpenAI Five → 在 Dota 2 击败世界冠军 OG
2019   💥 AlphaStar → 星际争霸 II 达到大师级
2020   💥 OpenAI → 机械手还原魔方
```

### 知识点 12：深度强化学习的关键突破 — DQN

**DQN（Deep Q-Network）** = Q-Learning + 深度学习 🧠

**为什么 DQN 是里程碑？**
- 用**神经网络**代替 Q 表格，处理高维状态（如原始像素）
- 两个关键 trick：

| Trick | 作用 |
|:------|:---- |
| **经验回放（Experience Replay）** 🔄 | 存下过去的经验，随机抽样训练，打破数据相关性 |
| **目标网络（Target Network）** 🎯 | 单独用一个网络算目标 Q 值，稳定训练 |

> 2013 年 DeepMind 用 DQN 在 Atari 2600 游戏上超越人类水平，是**深度强化学习**的开端 🚀

### 知识点 13：AlphaGo & AlphaZero

**AlphaGo（2016）：**
- 围棋状态空间 10¹⁷⁰，比宇宙原子总数还大 🌌
- 组合拳：**深度神经网络（策略网络+价值网络）+ 蒙特卡洛树搜索（MCTS）**
- 击败李世石（4:1），震动全球 🌍

**AlphaZero（2017）：**
- **完全从零自学**，不学人类棋谱，只靠自我对弈
- 同样的算法框架通吃围棋、象棋、将棋

### 知识点 14：OpenAI Five & 后续

| 项目 | 领域 | 规模 |
|:---- |:---- |:---- |
| **OpenAI Five** 🎮 | Dota 2（5v5 实时策略） | 大规模分布式 RL |
| **机械手还原魔方** 🧩 | 机器人控制 | Sim-to-Real 迁移 |
| **Emergent Tool Use** 🛠️ | 多智能体 | 涌现使用工具的行为 |
| **Evolving RL Algorithms** 🧬 | AutoML | 用进化算法自动发现新 RL 算法 |

---

## 六、🔑 RL 的核心要素 / 分类

### 知识点 15：RL 算法的三个核心要素

Agent 做决策需要三个关键组件：

```
┌──────────────────────────────────┐
│           RL Agent                │
│                                   │
│  ① Policy（策略） 📋               │
│     在状态 s 下，选择动作 a 的规则    │
│                                   │
│  ② Value Function（价值函数）💰     │
│     评估当前状态/动作有多好          │
│                                   │
│  ③ Model（模型）🌍                 │
│     对环境的理解（选配）             │
└──────────────────────────────────┘
```

### 知识点 16：Model-based vs Model-free

| 类型 | 说明 | 优点/缺点 |
|:---- |:---- |:--------- |
| **Model-based** 🧠 | Agent 试图"理解"环境，学习转移函数 P(s'\|s,a) 和奖励函数 R(s,a) | ✅ 更高效利用数据 ❌ 模型可能学错 |
| **Model-free** 🎮 | Agent 不学环境模型，直接学策略或价值函数 | ✅ 实现简单 ❌ 需要大量交互数据 |

### 知识点 17：Policy-based vs Value-based

| 类型 | 学什么 | 输出 | 适用场景 |
|:---- |:----- |:---- |:-------- |
| **Value-based** 📊 | 学价值函数 V(s) 或 Q(s,a) | 从 Q 值选最大动作 | 离散动作空间 |
| **Policy-based** 📋 | 直接学策略 π(a\|s) | 动作的概率分布 | 连续动作空间/随机策略 |
| **Actor-Critic** 🎭 | 两者都学 | Actor 做动作，Critic 评价 | 综合两者优势 |

---

## 📝 自测题

**基础概念：**
1. 强化学习和监督学习的最大区别是什么？
2. RL 的六个核心要素（Agent、Environment、State、Action、Reward、Policy）分别是什么？用你自己的例子解释。
3. 什么是马尔可夫性质（Markov Property）？

**数学部分：**
4. Return（回报）G_t 的定义是什么？折扣因子 γ 的作用是什么？
5. V(s) 和 Q(s, a) 分别代表什么？两者是什么关系？
6. Bellman Equation 的核心思想是什么？

**算法分类：**
7. Model-based 和 Model-free 的区别是什么？
8. Policy-based 和 Value-based 方法各有什么优缺点？
9. DQN 为什么需要"经验回放"和"目标网络"？

**应用部分：**
10. AlphaGo 为什么能击败人类围棋冠军？它用到了哪些技术？

---

## 📖 推荐资源

**经典教材：**
- Sutton & Barto, Reinforcement Learning: An Introduction（圣经级教材 📕）
- Szepesvári, Algorithms for Reinforcement Learning

**重要论文：**
- Mnih et al., Playing Atari with Deep Reinforcement Learning, 2013（DQN 开山之作）
- Mnih et al., Human-level control through deep reinforcement learning, Nature 2015
- Silver et al., Mastering the game of Go with deep neural networks and tree search, Nature 2016
- Silver et al., Mastering Chess and Shogi by Self-Play with a General RL Algorithm, 2017

**视频课程：**
- David Silver 的 RL 课程（B 站可看）
