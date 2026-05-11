# 📋 人工智能导论 · 知识点总结大纲与预测题（含答案）

> 基于 **2025 年人工智能导论 A 卷** 分析，结合课件复习笔记，提炼高频考点并附预测题详解 🎯
>
> 📎 参考笔记：[[introduction](introduction.md)] [[search](search.md)] [[machine_learning](machine_learning.md)] [[symbolism](symbolism.md)] [[connectionism](connectionism.md)] [[cnn](cnn.md)] [[gan_att_gnn](gan_att_gnn.md)] [[intelligent_optimization](intelligent_optimization.md)] [[reinforcement_learning](reinforcement_learning.md)]

---

## 📌 考试结构与分值分布

```text
卷面总分：100 分

┌──────────────────────────────────────────────────────────────┐
│  一、单选题 10 题 × 2 分 = 20 分                              │
│  （基础知识辨析，覆盖面广——覆盖所有学派）                       │
├──────────────────────────────────────────────────────────────┤
│  二、多选题 8 题 × ~2.5 分 = ~20 分                           │
│  （概念对比与场景识别，易错——侧重连接学派+符号学派）            │
├──────────────────────────────────────────────────────────────┤
│  三、简答/计算题 4 题 × 15 分 = 60 分                         │
│  （综合应用，需要动手算——搜索/A*/CF/模糊/Hopfield）             │
└──────────────────────────────────────────────────────────────┘
```

---

## 一、📚 知识点分级总结（附笔记链接）

### 🔴 第一梯队：必考（真题已出现，必须掌握）

| 知识点 | 重要性 | 题型 | 参考笔记 |
| :----- | :----: | :--- | :------- |
| **谓词逻辑翻译与推理** | ⭐⭐⭐⭐⭐ | 选择、大题 | [symbolism.md → 知识点9-11](symbolism.md) |
| **A\* 搜索算法** | ⭐⭐⭐⭐⭐ | 选择、大题 | [search.md → 知识点10-12](search.md) |
| **CF 确定性因子模型** | ⭐⭐⭐⭐⭐ | 多选、大题 | [symbolism.md → 知识点13](symbolism.md) |
| **模糊逻辑与模糊集** | ⭐⭐⭐⭐⭐ | 多选、大题 | [symbolism.md → 知识点14](symbolism.md) |
| **强化学习（折扣回报）** | ⭐⭐⭐⭐⭐ | 多选 | [reinforcement_learning.md → 知识点8-9](reinforcement_learning.md) |
| **Transformer / Self-Attention** | ⭐⭐⭐⭐⭐ | 多选 | [gan_att_gnn.md → 知识点9-10](gan_att_gnn.md) |
| **CNN 卷积神经网络** | ⭐⭐⭐⭐ | 选择 | [cnn.md → 知识点5-9](cnn.md) |
| **Hopfield 网络** | ⭐⭐⭐⭐ | 选择、大题 | [connectionism.md → HNN部分](connectionism.md) |

### 🟡 第二梯队：重要（真题提及，应熟练掌握）

| 知识点 | 重要性 | 题型 | 参考笔记 |
| :----- | :----: | :--- | :------- |
| **神经网络架构对比（HNN/RNN/GNN）** | ⭐⭐⭐⭐ | 多选 | [connectionism.md](connectionism.md) + [gan_att_gnn.md → 知识点14](gan_att_gnn.md) |
| **Exploration vs Exploitation** | ⭐⭐⭐ | 选择 | [reinforcement_learning.md → 知识点15-16](reinforcement_learning.md) |
| **搜索算法评估（完备性/最优性）** | ⭐⭐⭐ | 选择 | [search.md → 知识点4](search.md) |
| **知识表示方法** | ⭐⭐⭐ | 选择、多选 | [symbolism.md → 知识点3-5](symbolism.md) |
| **命题逻辑真值表与推理** | ⭐⭐⭐ | 选择 | [symbolism.md → 知识点8](symbolism.md) |

### 🟢 第三梯队：了解（可能涉及，以选择为主）

| 知识点 | 重要性 | 参考笔记 |
| :----- | :----: | :------- |
| AI 学派分类与核心主张 | ⭐⭐ | [introduction.md → 知识点7](introduction.md) |
| 机器学习基本概念（过拟合/欠拟合） | ⭐⭐ | [machine_learning.md → 知识点6](machine_learning.md) |
| 遗传算法 GA | ⭐⭐ | [intelligent_optimization.md → 知识点6-10](intelligent_optimization.md) |
| 粒子群 PSO / 蚁群 ACO | ⭐⭐ | [intelligent_optimization.md → 知识点13-14](intelligent_optimization.md) |
| 产生式系统 | ⭐⭐ | [symbolism.md → 知识点5](symbolism.md) |
| 知识图谱 | ⭐⭐ | [symbolism.md → 知识点16](symbolism.md) |
| 语义网络 | ⭐⭐ | [symbolism.md → 知识点15](symbolism.md) |
| GAN 生成对抗网络 | ⭐⭐ | [gan_att_gnn.md → 知识点1-4](gan_att_gnn.md) |

---

## 二、🔍 逐知识点精讲（含真题回顾）

---

### 🥇 必考 1：谓词逻辑翻译与推理

> 📎 详细笔记：[symbolism.md → 知识点 8-11](symbolism.md)

**真题回顾**：选择题第 1 题，给出一组谓词逻辑公式判断正误。

**翻译口诀：**

```text
"所有 A 都是 B"  →  ∀x A(x) → B(x)    (用 →)
"有的 A 是 B"    →  ∃x A(x) ∧ B(x)    (用 ∧)
"没有 A 是 B"    →  ¬∃x A(x) ∧ B(x)  或 ∀x A(x) → ¬B(x)
"只有 A 才 B"    →  B → A
```

**⚠️ 经典陷阱：**
- ❌ `∀x A(x) ∧ B(x)` — 这表示"所有东西既是 A 又是 B"，完全不是"所有 A 都是 B"
- ❌ `∃x A(x) → B(x)` — 存在量词用 → 会出问题（空真问题）

**真题示例（2025 A 卷第 1 题）：** 选出正确的谓词逻辑公式 → 注意区分 ∀ 和 ∃ 的搭配。

---

### 🥇 必考 2：A\* 搜索算法

> 📎 详细笔记：[search.md → 知识点 10-12](search.md)

**真题回顾**：大题第 1 题（15 分，A\* 搜索），用两个启发式函数做图搜索。

**必须掌握的步骤：**

```text
A* 算法执行流程：
① 初始节点 S 放入 OPEN 表，f(S) = g(S) + h(S) = 0 + h(S)
② 从 OPEN 表取出 f(n) 最小的节点扩展
③ 对每个后继 n'：
   · g(n') = g(n) + c(n, n')
   · f(n') = g(n') + h(n')
   · 若 n' 不在 OPEN/CLOSED 中 → 加入 OPEN
   · 若 n' 已在 OPEN 中且新 f 更小 → 更新 f
   · 若 n' 已在 CLOSED 中且新 f 更小 → 重新放回 OPEN
④ 重复直到目标节点被扩展
```

**A\* vs Dijkstra vs 贪心：**

| 算法 | f(n) | 特点 |
| :--- | :--- | :--- |
| **Dijkstra** | f(n) = g(n) | 均匀向所有方向扩展 |
| **贪心最佳优先** | f(n) = h(n) | 直奔目标，但不一定最优 |
| **A\*** | f(n) = g(n) + h(n) | 两者平衡 |

**四个评估标准回顾：**

| 算法 | 完备性 | 最优性 | 数据结构 |
| :--- | :----: | :----: | :------- |
| DFS | ❌ | ❌ | 栈 |
| BFS | ✅ | ✅（等代价） | 队列 |
| UCS | ✅ | ✅ | 优先队列(g) |
| A\* | ✅（可采纳h） | ✅（可采纳h） | 优先队列(f=g+h) |

---

### 🥇 必考 3：CF 确定性因子模型

> 📎 详细笔记：[symbolism.md → 知识点 13](symbolism.md)

**真题回顾**：大题第 2 题（15 分），4 条规则合成计算 CF(H)。

**核心公式（必须记住）：**

```text
① AND 合取：CF(E) = min{CF(E₁), CF(E₂), ...}
② OR 析取： CF(E) = max{CF(E₁), CF(E₂), ...}
③ 规则执行：CF(H) = CF(H,E) × max{0, CF(E)}
④ 同一结论多规则合成：
    都 > 0:   CF₁₂ = CF₁ + CF₂ - CF₁·CF₂
    都 < 0:   CF₁₂ = CF₁ + CF₂ + CF₁·CF₂
    一正一负: CF₁₂ = (CF₁ + CF₂) / (1 - min(|CF₁|, |CF₂|))
```

**解题步骤：**
```text
Step 1: 从底层开始，计算中间结论的 CF
Step 2: 逐层向上推理
Step 3: 遇到同结论多规则 → 合成
```

---

### 🥇 必考 4：模糊逻辑与模糊集

> 📎 详细笔记：[symbolism.md → 知识点 14](symbolism.md)

**真题回顾**：大题第 3 题（15 分），给定论域和隶属度做模糊运算。

**核心运算：**

```text
① μ_A∩B(x) = min(μ_A(x), μ_B(x))          — 模糊交
② μ_A∪B(x) = max(μ_A(x), μ_B(x))          — 模糊并
③ μ_Ā(x) = 1 - μ_A(x)                      — 模糊补
④ μ_R(a,b) = min(μ_A(a), μ_B(b))          — 笛卡尔积
⑤ S = Q ∘ R:  μ_S(x,z) = max_y min{Q(x,y), R(y,z)}  — 模糊关系合成
```

---

### 🥇 必考 5：强化学习——折扣回报

> 📎 详细笔记：[reinforcement_learning.md → 知识点 8-9](reinforcement_learning.md)

**真题回顾**：多选第 8 题，FrozenLake 问题计算折扣回报。

**核心公式：**

```text
Gₜ = r₁ + γ·r₂ + γ²·r₃ + γ³·r₄ + ...

真题示例 FrozenLake：
   初始状态 → ... → Goal(+10), γ = 0.9
    中间每步奖励 = 0

如果走到 Goal 需要 5 步：
   G₀ = 0·1 + 0·0.9 + 0·0.9² + 0·0.9³ + 10·0.9⁴
      = 10 × 0.6561 = 6.561 ✅
```

---

### 🥇 必考 6：Transformer / Self-Attention

> 📎 详细笔记：[gan_att_gnn.md → 知识点 9-10](gan_att_gnn.md)

**真题回顾**：多选第 6 题，关于 Transformer 组件（QKV 等）。

**核心公式：**

```text
Self-Attention(Q,K,V) = softmax(Q·K^T / √d_k) · V

其中：
  Q = X × W_Q    (Query — 当前词想找什么)
  K = X × W_K    (Key — 每个词有什么信息)
  V = X × W_V    (Value — 每个词的实际内容)
  
除以 √d_k：防止点积过大导致 Softmax 梯度消失
```

**Multi-Head Attention：** 多个注意力头同时计算，每个头关注不同方面，最后拼接。

---

## 三、📝 预测题（含详细答案）

### 选择题预测

---

**1. 下列哪个是有效的谓词逻辑公式？** 🧪

A. ∀x Man(x) → Mortal(x)
B. ∀x (Mule(x) ∧ ¬Horse(x) ∧ ¬Donkey(x))
C. ∀x Person(x) → (Sleep(x) ∨ Eat(x))
D. ¬∀x Student(x) → Calculus(x)

**✅ 答案：A、C**

**解析：**
- A ✅：∀x (Man(x) → Mortal(x)) — "所有人都难免一死"，正确的蕴含式
- B ❌：∧ 连接的量词表示"所有东西都是骡子且不是马且不是驴"，不符合常识
- C ✅：∀x (Person(x) → (Sleep(x) ∨ Eat(x))) — "每个人都会睡觉或吃饭"
- D ❌：量词作用域不清晰，¬∀x 后面的括号缺失

---

**2. 关于 A\* 算法，下列说法正确的是？** ⭐

A. f(n) = g(n) + h(n)，g(n) 是启发式估计
B. 当 h(n) = 0 时，A\* 退化为贪心搜索
C. 可采纳的 h 保证 A\* 找到最优解
D. h(n) 越大，搜索效率越低

**✅ 答案：C**

**解析：**
- A ❌：g(n) 是**实际代价**，h(n) 才是启发式估计
- B ❌：h(n)=0 时退化为 **UCS**（Dijkstra），不是贪心搜索
- C ✅：可采纳性（h(n) ≤ h\*(n)）是 A\* 找到最优解的充分条件
- D ❌：h(n) 越大（在不超过 h\*(n) 的前提下），搜索效率越**高**（扩展更少节点）

> 📎 参考：[search.md → 知识点 11](search.md)

---

**3. 关于 CNN 的池化层，下列说法错误的是？** 🖼️

A. 可以降低特征图尺寸
B. 可以减少参数量
C. 可以增强平移不变性
D. 会让模型变得更深

**✅ 答案：D**

**解析：**
池化层作用是**降采样**（A✅）、**减少参数量**（B✅）、**增强平移不变性**（C✅），但不会改变网络深度（D❌）。网络深度由卷积层/全连接层的层数决定。

> 📎 参考：[cnn.md → 知识点 6](cnn.md)

---

**4. 在强化学习中，"Exploration vs Exploitation" 指的是？** 🎮

A. 探索新动作 vs 利用已知最优动作
B. 训练 vs 测试
C. 监督 vs 无监督
D. 策略网络 vs 价值网络

**✅ 答案：A**

**解析：** Exploration（探索）是尝试未做过的新动作，Exploitation（利用）是使用已知的最优策略。两者的平衡是 RL 的核心问题。

> 📎 参考：[reinforcement_learning.md → Exploration vs Exploitation](reinforcement_learning.md)

---

**5. 下列哪项不是 Hopfield 网络的特点？** 🔄

A. 全连接反馈结构
B. 能量函数单调下降
C. 通过 BPTT 训练
D. 可以用于联想记忆

**✅ 答案：C**

**解析：**
- A ✅：Hopfield 是单层全连接反馈网络
- B ✅：Hopfield 网络运行时能量函数单调递减，最终收敛到稳定态
- C ❌：**BPTT（Back Propagation Through Time）** 是 RNN 的训练方法，不是 Hopfield 的
- D ✅：DHNN 可用于联想记忆

> 📎 参考：[connectionism.md → Hopfield 网络](connectionism.md)

---

**6. 在 Transformer 中，Scaled Dot-Product Attention 除以 √d_k 的原因是？** 🤖

A. 防止梯度爆炸
B. 防止点积过大导致 Softmax 梯度消失
C. 加快计算速度
D. 增加模型容量

**✅ 答案：B**

**解析：** 当 d_k（Key 的维度）很大时，Q·Kᵀ 的点积结果会很大，导致 Softmax 函数进入饱和区，梯度极小（梯度消失）。除以 √d_k 将数值拉回到合适的范围。

> 📎 参考：[gan_att_gnn.md → 知识点 9.4](gan_att_gnn.md)

---

### 多选题预测

---

**7. 下列属于无信息搜索（盲搜）算法的有？** 🔍

A. DFS
B. BFS
C. A\*
D. UCS
E. Dijkstra

**✅ 答案：A、B、D、E**

**解析：** 无信息搜索 = 除了问题定义外不用任何额外信息
- A ✅ DFS：深度优先，盲搜
- B ✅ BFS：广度优先，盲搜
- C ❌ A\*：使用启发式函数 h(n)，是**有信息搜索**
- D ✅ UCS：代价一致搜索，只用到 g(n)（已知的实际代价），属于盲搜
- E ✅ Dijkstra：UCS 的特例，也是盲搜

> 📎 参考：[search.md → 知识点 5-7](search.md)

---

**8. 关于 CF 模型，下列说法正确的有？** 📊

A. CF 的取值范围是 [-1, 1]
B. AND 条件取各证据 CF 的最小值
C. OR 条件取各证据 CF 的最大值
D. CF > 0 表示证据支持结论
E. CF(H) = CF(H,E) × min{0, CF(E)}

**✅ 答案：A、B、C、D**

**解析：**
- A ✅：CF ∈ [-1, 1]
- B ✅：AND 取 min
- C ✅：OR 取 max
- D ✅：CF > 0 支持，CF < 0 反对
- E ❌：应该是 CF(H) = CF(H,E) × **max**{0, CF(E)}，不是 min

> 📎 参考：[symbolism.md → 知识点 13](symbolism.md)

---

**9. 以下哪些是深度学习成功的关键因素？** 🚀

A. 大规模数据
B. GPU 算力
C. 更好的算法（ReLU、Dropout）
D. 符号逻辑的突破

**✅ 答案：A、B、C**

**解析：** 深度学习成功的三要素：**大数据** + **大算力** + **好算法**（ReLU、Dropout等）。符号逻辑（D）的突破不是深度学习成功的原因。

> 📎 参考：[cnn.md → 知识点 2](cnn.md)

---

**10. 关于强化学习的 MDP，下列说法正确的有？** 🎯

A. MDP 满足马尔可夫性质
B. 折扣因子 γ 越接近 1 越"有远见"
C. Q(s,a) 表示在状态 s 做动作 a 的期望回报
D. Policy 定义了在每个状态做什么动作

**✅ 答案：A、B、C、D**

**解析：** 全部正确
- A ✅：马尔可夫性质 = 未来只取决于当前，与历史无关
- B ✅：γ→1 时未来的奖励几乎不打折扣（有远见）；γ→0 时只看眼前
- C ✅：Q(s,a) = E[Gₜ | Sₜ=s, Aₜ=a]
- D ✅：策略 π(a|s) 是在状态 s 下选择动作 a 的概率分布

> 📎 参考：[reinforcement_learning.md → 知识点 5-9](reinforcement_learning.md)

---

### 大题预测（含完整答案）

---

**预测题 11：A\* 搜索 🧭（15分）**

**题目（见上文——此处省略重复）**

**✅ 参考答案：**

**(1) 判断可采纳性（3分）**

```text
h₁(n) 是否可采纳？
检查是否每个 n 都有 h₁(n) ≤ h*(n)（真实最短距离）

S 到 G 的真实最短距离：
  S→N4→N1→G = 1+2+2 = 5
  S→N3→G = 3+2 = 5
  S→N2→G = 2+3 = 5
  S→N4→N1→G = 5
真实 h*(S) = 5
h₁(S) = 6 > 5 → ❌ h₁ 不可采纳！

h₂(n) 是否可采纳？
假设经过计算所有 h₂(n) ≤ h*(n)
→ ✅ h₂ 可采纳（需要验证每个节点）
```

**(2) 用 h₁ 运行 A\* 的前 5 步（6分）**

```text
Step 0: OPEN = {S(g=0,h=6,f=6)}
Step 1: 扩展 S → 后继 N2(g=5,h=4,f=9), N3(g=3,h=3,f=6), N4(g=1,h=5,f=6)
         OPEN = {N3(f=6), N4(f=6), N2(f=9)}
Step 2: 扩展 N3(f=6) → 后继 G(g=3+2=5,h=0,f=5)
         OPEN = {G(f=5), N4(f=6), N2(f=9)}
Step 3: 扩展 G(f=5) → 找到目标 ✅
        总共只需 4 步扩展
```

**(3) h₁ vs h₂ 哪个更好？（6分）**

```text
h₂ 更好。因为 h₁ 不可采纳（h₁(S)=6 > h*(S)=5），
违背了 A* 最优性的前提。用 h₁ 时 A* 可能找不到最优解。

而 h₂ 可采纳，保证最优性。
并且 h₂ 更精确（更接近真实值），扩展节点数更少。
```

> 📎 参考：[search.md → 知识点 10-12](search.md)

---

**预测题 12：CF 模型计算 📊（15分）**

**题目（见上文——此处省略重复）**

**✅ 参考答案：**

**(1) 计算 CF(E₃) 和 CF(E₅) （4分）**

```text
r₁: IF E₁ AND E₂ THEN E₃ (0.8)
CF(E₁ AND E₂) = min{CF(E₁), CF(E₂)} = min{0.7, 0.5} = 0.5
CF(E₃) = 0.8 × max{0, 0.5} = 0.4

r₂: IF E₃ OR E₄ THEN E₅ (0.9)
CF(E₃ OR E₄) = max{CF(E₃), CF(E₄)} = max{0.4, 0.4} = 0.4
CF(E₅) = 0.9 × max{0, 0.4} = 0.36
```

**(2) 分别计算 r₃ 和 r₄ 对 H 的 CF 值（6分）**

```text
r₃: IF E₅ THEN H (0.7)
CF₃(H) = 0.7 × max{0, 0.36} = 0.252

r₄: IF E₆ THEN H (0.85)
CF₄(H) = 0.85 × max{0, 0.8} = 0.68
```

**(3) 综合计算 CF(H) （5分）**

```text
都 > 0，用公式：CF₁₂ = CF₁ + CF₂ - CF₁·CF₂
CF(H) = 0.252 + 0.68 - 0.252 × 0.68
      = 0.932 - 0.17136
      = 0.76064 ≈ 0.761
```

> 📎 参考：[symbolism.md → 知识点 13](symbolism.md)

---

**预测题 13：模糊推理 🌫️（15分）**

**题目（见上文——此处省略重复）**

**✅ 参考答案：**

**(1) A ∪ B 和 A ∩ B（5分）**

```text
A ∪ B (取 max):
  A∪B = {max(0,0)/100, max(0,0.1)/200, max(0.4,0.5)/500, max(0.7,0.8)/800, max(1.0,1.0)/1200}
      = {0/100, 0.1/200, 0.5/500, 0.8/800, 1.0/1200}

A ∩ B (取 min):
  A∩B = {min(0,0)/100, min(0,0.1)/200, min(0.4,0.5)/500, min(0.7,0.8)/800, min(1.0,1.0)/1200}
      = {0/100, 0/200, 0.4/500, 0.7/800, 1.0/1200}
```

**(2) ¬A（A 的补集）（4分）**

```text
¬A = {1-0/100, 1-0/200, 1-0.4/500, 1-0.7/800, 1-1.0/1200}
   = {1.0/100, 1.0/200, 0.6/500, 0.3/800, 0/1200}
```

**(3) 模糊关系 R = A → B（笛卡尔积）（6分）**

```text
R = A × B, μ_R(x,y) = min(μ_A(x), μ_B(y))

           B:  100   200   500   800   1200
A:   100  [  0    0     0     0     0   ]
     200  [  0    0     0     0     0   ]
     500  [  0    0.1   0.4   0.4   0.4 ]
     800  [  0    0.1   0.5   0.7   0.7 ]
    1200  [  0    0.1   0.5   0.8   1.0 ]

计算示例：μ_R(500, 200) = min(0.4, 0.1) = 0.1
         μ_R(1200, 1200) = min(1.0, 1.0) = 1.0
```

> 📎 参考：[symbolism.md → 知识点 14](symbolism.md)

---

**预测题 14：谓词逻辑 📝（15分）**

**题目（见上文——此处省略重复）**

**✅ 参考答案：**

(1) **所有学生都通过了考试。**（3分）

```text
∀x Student(x) → Pass(x)
```

(2) **有些学生不喜欢数学。**（3分）

```text
∃x Student(x) ∧ ¬Like(x, Math)
```

(3) **不是所有鸟都会飞。**（3分）

```text
¬∀x Bird(x) → Fly(x)
等价于：∃x Bird(x) ∧ ¬Fly(x)
```

(4) **每个人都有一个最好的朋友。**（3分）

```text
∀x Person(x) → ∃y BestFriend(y, x)
或：∀x ∃y Person(x) → BestFriend(y, x)
```

(5) **证明**（3分）

```text
前提：
① ∀x Philosopher(x) → Thinker(x)
② Philosopher(Aristotle)

推理：
③ Philosopher(Aristotle) → Thinker(Aristotle)   ①代入 x=Aristotle
④ Thinker(Aristotle)                              ②③假言推理 ✅
```

> 📎 参考：[symbolism.md → 知识点 9-11](symbolism.md)

---

## 四、💡 各章节重点速查（附笔记链接）

### 第1章：人工智能概述 → [introduction.md](introduction.md)
- AI 诞生：**1956 达特茅斯会议**
- 三次浪潮 + 两次寒冬（原因：算力/数据/算法）
- **六大流派**：符号/连接/行为/统计/综合/直觉
- 图灵测试

### 第2章：搜索 → [search.md](search.md)
- 搜索问题形式化：**状态、动作、转移、目标、代价**
- DFS/BFS/UCS：数据结构、完备性、最优性
- **A\* 算法**：f=g+h、可采纳性、一致性、占优 ⭐
- Dijkstra：A\* 当 h=0 的特例

### 第3章：机器学习 → [machine_learning.md](machine_learning.md)
- 监督/无监督/强化学习
- 过拟合与欠拟合、训练/验证/测试集
- NFL 定理、奥卡姆剃刀、归纳偏置

### 第4章：连接学派 → [connectionism.md](connectionism.md) | [cnn.md](cnn.md) | [gan_att_gnn.md](gan_att_gnn.md)

| 子主题 | 重点内容 | 笔记链接 |
| :----- | :------- | :------- |
| **MP/感知器** | 布尔 vs 实值、XOR 线性不可分 | [connectionism.md](connectionism.md) |
| **BP 网络** | 万能近似定理、梯度下降 | [connectionism.md](connectionism.md) |
| **Hopfield** ⭐ | 能量函数、联想记忆、TSP | [connectionism.md](connectionism.md) |
| **CNN** ⭐ | 卷积/池化/全连接、局部连接+参数共享 | [cnn.md](cnn.md) |
| **RNN/LSTM** | BPTT、梯度消失、遗忘门/输入门/输出门 | [connectionism.md](connectionism.md) |
| **GAN** | 生成器 vs 判别器、Minimax 博弈 | [gan_att_gnn.md](gan_att_gnn.md) |
| **Attention** ⭐ | Self-Attention(QKV)、Multi-Head | [gan_att_gnn.md](gan_att_gnn.md) |
| **GNN/GCN** | 图结构数据、邻居聚合 | [gan_att_gnn.md](gan_att_gnn.md) |

### 第5章：行为学派 → [intelligent_optimization.md](intelligent_optimization.md) | [reinforcement_learning.md](reinforcement_learning.md)

| 子主题 | 重点内容 | 笔记链接 |
| :----- | :------- | :------- |
| **遗传算法 GA** | 选择/交叉/变异、轮盘赌 | [intelligent_optimization.md](intelligent_optimization.md) |
| **粒子群 PSO** | pBest/gBest、速度更新 | [intelligent_optimization.md](intelligent_optimization.md) |
| **蚁群 ACO** | 信息素正反馈、TSP | [intelligent_optimization.md](intelligent_optimization.md) |
| **强化学习** ⭐ | MDP、V(s)/Q(s,a)、折扣回报 | [reinforcement_learning.md](reinforcement_learning.md) |
| **DQN/AlphaGo** | 经验回放、MCTS | [reinforcement_learning.md](reinforcement_learning.md) |

### 第6章：符号学派 → [symbolism.md](symbolism.md)（重点 ⭐）
| 子主题 | 重点内容 |
| :----- | :------- |
| **命题逻辑** | 连接词、真值表、P→Q 只有在 P 真 Q 假时为假 |
| **谓词逻辑** ⭐ | ∀/∃ 翻译、三段论推理 |
| **产生式系统** | IF-THEN、规则库/数据库/推理机 |
| **C-F 模型** ⭐ | AND/OR 计算、规则执行、合成公式 |
| **模糊逻辑** ⭐ | 隶属度、交/并/补/笛卡尔积/合成 |
| **知识图谱** | 实体/关系/属性、KGC |

---

## 五、⏰ 考前 Checklist（附参考笔记）

- [ ] **谓词逻辑公式翻译**（∀/∃ 的用法）→ [symbolism.md](symbolism.md)
- [ ] **A\* 搜索的手动推演** → [search.md](search.md)
- [ ] **CF 模型的综合计算** → [symbolism.md](symbolism.md)
- [ ] **模糊集运算**（交/并/补/合成）→ [symbolism.md](symbolism.md)
- [ ] **折扣回报 Gₜ 计算** → [reinforcement_learning.md](reinforcement_learning.md)
- [ ] **Self-Attention 的 QKV 公式** → [gan_att_gnn.md](gan_att_gnn.md)
- [ ] **CNN 三大组件及其作用** → [cnn.md](cnn.md)
- [ ] **Hopfield 网络能量函数** → [connectionism.md](connectionism.md)
- [ ] **DFS/BFS/UCS/A\* 四种搜索对比** → [search.md](search.md)
- [ ] **强化学习的 MDP 框架** → [reinforcement_learning.md](reinforcement_learning.md)
- [ ] **产生式系统的结构** → [symbolism.md](symbolism.md)
- [ ] **神经网络架构对比**（HNN/RNN/GNN）→ [connectionism.md](connectionism.md) + [gan_att_gnn.md](gan_att_gnn.md)

---

## 🔗 复习笔记索引

| 章节 | 笔记文件 | 说明 |
| :--- | :------- | :--- |
| 概述 | [introduction.md](introduction.md) | AI 历史、学派、应用 |
| 搜索 | [search.md](search.md) | DFS/BFS/UCS/A\*、TSP |
| 机器学习 | [machine_learning.md](machine_learning.md) | 监督/无监督、过拟合、学习理论 |
| 符号学派 | [symbolism.md](symbolism.md) | **谓词逻辑/CF/模糊逻辑/知识图谱 ⭐** |
| 连接学派(基础) | [connectionism.md](connectionism.md) | MP/感知器/BP/Hopfield/RNN/LSTM |
| 连接学派(CNN) | [cnn.md](cnn.md) | 卷积/池化/AlexNet/ResNet |
| 连接学派(前沿) | [gan_att_gnn.md](gan_att_gnn.md) | GAN/Attention/Transformer/GNN |
| 行为学派(优化) | [intelligent_optimization.md](intelligent_optimization.md) | GA/PSO/ACO |
| 行为学派(RL) | [reinforcement_learning.md](reinforcement_learning.md) | MDP/V(s)/Q(s,a)/折扣回报 |
