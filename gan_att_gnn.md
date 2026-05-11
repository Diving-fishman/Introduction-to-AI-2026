# 🧠 连接学派（三）：GAN + Attention + GNN 复习笔记

> 这一课讲深度学习的三个前沿方向：**GAN（生成对抗网络）**🎨、**Attention（注意力机制）**👀、**GNN（图神经网络）**🕸️

---

## 📌 知识地图（先看整体框架）

```
┌─────────────────────────────────────────────────────────┐
│                   三大前沿方向                             │
│                                                          │
│  🎨 GAN                      👀 Attention              🕸️ GNN          │
│  (生成对抗网络)               (注意力机制)               (图神经网络)    │
│                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐│
│  │  生成器 vs    │    │ Self-attention│    │  图卷积 GCN  ││
│  │  判别器博弈    │    │  QKV 模型    │    │  节点分类    ││
│  └──────────────┘    └──────────────┘    └──────────────┘│
│                                                          │
│  目标: 生成逼真数据   目标: 关注重要信息    目标: 处理图结构数据  │
└─────────────────────────────────────────────────────────┘
```

---

## 第一部分：🎨 GAN（生成对抗网络）

### 知识点 1：GAN 的核心思想

**GAN = 生成器（Generator） + 判别器（Discriminator）**，两者互相对抗、共同进步 🤼

| 角色 | 名字 | 干什么的 | 比喻 🎭 |
|------|------|---------|---------|
| **生成器 G** 🖌️ | 伪造者 | 从随机噪声 z 生成假图片，骗过判别器 | 假钞制造犯 |
| **判别器 D** 🔍 | 鉴别师 | 判断输入是"真"还是"假"（来自数据集还是 G 生成的） | 验钞机 |

**训练过程就像警察抓小偷 👮‍♂️ 和小偷升级手艺 🦹：**
- 小偷（G）努力造更逼真的假币
- 警察（D）努力分辨真币假币
- 互相促进，最后 G 能造出以假乱真的图片

> 💡 **2014 年 Ian Goodfellow 提出**。据说 Goodfellow 在酒吧里跟朋友争论时突然想到这个想法，当晚就写代码验证成功了 🍺→💡

### 知识点 2：GAN 的训练目标（Minimax Game）

**目标函数：**
```
min_G max_D V(D,G) = E[log D(x)] + E[log(1 - D(G(z)))]
```

- D 想让 D(x) → 1（真图片判真），D(G(z)) → 0（假图片判假）
- G 想让 D(G(z)) → 1（骗过判别器）

**训练步骤（交替训练）：**
1. 固定 G，训练 D（让 D 更会分辨真假）
2. 固定 D，训练 G（让 G 更会骗 D）
3. 循环 k 步 D + 1 步 G

### 知识点 3：GAN 的经典变体 🏗️

| 模型 | 年份 | 一句话说清 | 核心创新 |
|------|------|-----------|---------|
| **DCGAN** 🎨 | 2016 | 把 CNN 引入 GAN | 用卷积代替全连接，BN、ReLU、LeakyReLU |
| **WGAN** 🌊 | 2017 | 用 Wasserstein 距离代替 JS 散度 | 解决训练不稳定、mode collapse |
| **CGAN** 🏷️ | 2014 | 给 GAN 加条件，指定生成内容 | 输入时拼接 label/条件信息 |
| **Pix2Pix** 🖼️↔️🖼️ | 2017 | 图像到图像的翻译（有配对数据） | 条件 GAN + U-Net 生成器 |
| **CycleGAN** 🔄 | 2017 | 无配对数据的图像翻译 | 循环一致性损失（cycle consistency） |
| **PGGAN** 📈 | 2018 | 渐进式训练，从低分辨率到高分辨率 | Progressive Growing |
| **StyleGAN** 💅 | 2019 | 控制生成图像的"风格" | AdaIN、Mapping Network、Mixing Regularization |
| **StyleGAN2** ✨ | 2020 | 修复 StyleGAN 的伪影 | 改进 AdaIN 实现 |

### 知识点 4：各 GAN 变体的关键理解

**DCGAN** — 第一个把 CNN 成功用于 GAN 的工作：
- 去掉全连接层，用卷积代替
- 生成器用转置卷积上采样
- Batch Normalization + ReLU（生成器）/ LeakyReLU（判别器）

**WGAN** — 解决 GAN 训练不稳定的问题：
- ❌ 不用 Sigmoid，去掉 log
- ✅ 用 Wasserstein 距离（Earth Mover's Distance）
- ✅ 权重裁剪（weight clipping）保证 Lipschitz 约束
- ✅ 优化器用 RMSProp/SGD，不用 Adam

**CycleGAN** — 没有配对数据也能做图像翻译：
- 关键 idea：**循环一致性** — 把一张图翻译成目标风格，再翻译回来，应该和原图差不多
- 比如：照片 ↔ 莫奈画作，马 ↔ 斑马 🦓

**StyleGAN** — 控制图像的"风格"：
- **Mapping Network**：把随机噪声 z 映射到中间 latent space w
- **AdaIN**（Adaptive Instance Normalization）：把风格信息注入生成过程
- **Mixing Regularization**：训练时用两种不同的风格向量混合

> 📌 **StyleGAN 的应用**：https://www.artbreeder.com（在线生成人脸）

### 知识点 5：GAN 的应用场景

- 🎨 图像生成（人脸、风景、艺术创作）
- 🖼️ 图像超分辨率、上色、修复
- 🔄 风格迁移（照片 → 动漫风格）
- 🎬 视频生成、数据增强
- 💊 药物分子设计

---

## 第二部分：👀 Attention（注意力机制）

### 知识点 6：注意力机制的灵感来源

**认知注意力（Cognitive Attention）**：人脑处理信息时，不会平等对待所有信息，而是**聚焦在重要的部分**。

现实中的注意力 👁️：
- 你在人群中找朋友 → 自动忽略其他人，只扫视面孔
- 读文章时 → 重点看关键词、标题

**注意力机制的核心思想：让网络学会"该看哪里"** 🎯

### 知识点 7：注意力类型

| 分类 | 说明 |
|:---- |:---- |
| **Hard Attention** 🔲 | 要么看（1）要么不看（0），不可微，难训练 |
| **Soft Attention** 🌫️ | 给每个位置一个 [0,1] 的权重，可微，端到端训练 |
| **Self-attention** 🔄 | 在自己的序列内部算注意力，"每个词看其他词" |
| **Cross-attention** 🔀 | 在两个不同序列之间算注意力（如翻译任务） |

**Bottom-up 注意力**（数据驱动的，自下而上） vs **Top-down 注意力**（任务驱动的，自上而下）

### 知识点 8：Attention 发展里程碑

```
2014  🐣 Google DeepMind → Recurrent Attention Model（图像上的注意力）
2014  🐣 Bahdanau + Bengio → Attention 用于机器翻译（NLP 开始用）
2015  🐣 Xu et al. → Show, Attend and Tell（图像描述）
2017  💥 Google → "Attention is All You Need"（Transformer 诞生，革命！）
2017  📈 SENet → 通道注意力（CV 领域）
2018  📈 CBAM → 空间 + 通道注意力
2020  💥 ViT → Vision Transformer（Transformer 用于图像分类）
```

### 知识点 9：Self-Attention（自注意力）—— 重点中的重点 ⭐

**这是 Transformer 的核心，也是当前 LLM（如 ChatGPT）的基础！**

#### 9.1 直观理解

一句话说清 Self-Attention：**序列中的每个元素，去看序列中所有其他元素，然后根据"谁更重要"来整合信息。**

类比：你在一个会议室里做决策 🏢
- **Query（Q）** 🙋 = 你问的问题（你想知道什么）
- **Key（K）** 🏷️ = 每个人的"标签"（谁有什么信息）
- **Value（V）** 📄 = 每个人实际说的内容

> **bilibili 例子**：你想在 B 站搜视频 — 你的搜索词是 Query，视频的标签是 Key，视频内容本身是 Value

#### 9.2 Self-Attention 的计算步骤

```
输入序列：x₁, x₂, x₃, x₄

Step 1：生成 Q、K、V
  Q = W_Q × X
  K = W_K × X  
  V = W_V × X

Step 2：算注意力分数（每个词和其他所有词的匹配度）
  score_ij = Q_i · K_j / √d_k    （点积 + 缩放）
  
Step 3：Softmax 归一化
  α_ij = softmax(score_ij)        （变成概率，和为 1）

Step 4：加权求和
  output_i = Σ_j α_ij × V_j      （按重要程度汇总信息）
```

#### 9.3 矩阵形式（可以看到 GPU 并行计算的优势 🚀）

```
Q = X × W_Q
K = X × W_K
V = X × W_V

Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
```

> **全部可以写成矩阵乘法，GPU 并行计算，超快！**

#### 9.4 Scaled Dot-Product Attention 为什么要除 √d_k？

当维度 d_k 很大时，点积的结果会很大，Softmax 的梯度会变得非常小（梯度消失）。除以 √d_k 相当于**把数值拉回合适的范围**。

### 知识点 10：Multi-Head Self-Attention（多头自注意力）

**一个头不够？那就多来几个！** 👥

```
不学一个注意力，而是学 h 个不同的注意力（不同头关注不同方面）：
  head_i = Attention(Q × W_Q_i, K × W_K_i, V × W_V_i)

把所有头的结果拼起来：
  MultiHead(Q, K, V) = Concat(head₁, ..., head_h) × W_O
```

> 💡 有的头关注句法关系（主谓宾），有的头关注语义关系，各司其职。

### 知识点 11：Attention 在 CV 中的应用

| 模型 | 一句话说清 |
|:---- |:----------|
| **SENet** 🧪 | Squeeze（压缩空间）-Excitation（学习通道权重）-Scale（加权），关注**哪些通道**重要 |
| **CBAM** 🔍 | 既是"where"（空间注意力 SAM）又是"what"（通道注意力 CAM），双管齐下 |
| **Non-local NN** 🌐 | CV 中的 self-attention，一个像素关注所有像素，捕获**全局**信息 |
| **ViT** 🖼️ | 把图像切成 16×16 的 patch，当成 token 序列喂给 Transformer |
| **DETR** 🎯 | Facebook 用 Transformer 做目标检测，不要锚框、不要 NMS |
| **SETR** 🏘️ | Transformer 做语义分割 |
| **PCT** ☁️ | Transformer 做点云处理 |

---

## 第三部分：🕸️ GNN（图神经网络）

### 知识点 12：图的基本概念

**图（Graph）** 由两部分组成：
- **节点（Vertex/Node）** 🟢 — 图中的点
- **边（Edge）** 🔗 — 点之间的连接

**图的分类：**
| 类型 | 说明 | 例子 |
|:---- |:---- |:---- |
| **有向图** ➡️ | 边有方向 (A→C ≠ C→A) | Twitter 关注（你关注他 ≠ 他关注你） |
| **无向图** ↔️ | 边没有方向 (A-B = B-A) | 微信好友（你好友他 = 他好友你） |

**为什么传统方法处理图很难？**
- CNN 需要**规则网格**（矩阵形式的图片），而图是不规则的（Non-Euclidean Structure）🚫
- 每个节点邻居数量不一样，没法用固定大小的卷积核

### 知识点 13：GNN 的核心思想

**核心 idea：每个节点通过不断聚合邻居的信息来更新自己的表示** 🔄

```
节点 v 的隐藏状态 h_v 由以下信息决定：
  ① 自己的特征 x_v
  ② 相邻边的特征 x_(u,v)  
  ③ 邻居节点的状态 h_u

公式：h_v = f(x_v, x_(u,v), h_u, x_u)
```

**GNN 和 RNN 的关系**：GNN 在图上迭代传播信息，类似于 RNN 在时间序列上展开，早期 GNN 确实用 RNN 来实现。

### 知识点 14：GCN（图卷积网络）—— GNN 的里程碑

**GCN = Graph Convolutional Networks**：把 CNN 的卷积操作"搬到"图结构上 🚚

**CNN vs GCN：**
| | CNN | GCN |
|:--|:----|:----|
| 数据结构 | 规则网格（图片） | 不规则图结构 |
| 邻居 | 固定数量的相邻像素 | 数量不等的相邻节点 |
| 卷积方式 | 固定卷积核滑动 | 聚合邻居信息 + 归一化 |

**GCN 的核心公式（简化版）：**
```
H^(l+1) = σ(Ã · H^(l) · W^(l))

Ã = 归一化的邻接矩阵
H^(l) = 第 l 层的节点特征
W^(l) = 第 l 层的权重（可学习参数）
σ = 激活函数（如 ReLU）
```

### 知识点 15：图上的常见任务

| 任务 | 例子 |
|:---- |:---- |
| **节点分类** 🏷️ | 预测用户的兴趣标签 |
| **图分类** 🖼️ | 判断分子是否有毒 |
| **链接预测** 🔗 | 预测两个用户会不会成为好友 |
| **边分类** 📊 | 判断两个节点之间的关系类型 |

### 知识点 16：GNN 的实际应用 🌍

| 领域 | 应用 |
|:---- |:---- |
| 🖼️ **场景图生成**（Scene Graph） | 图片 → "人在骑🐴"的结构化描述 |
| 🧬 **药物发现** 💊 | 分子结构预测药性 |
| ☁️ **点云分类** | 3D 激光雷达点云识别 |
| 🦾 **姿态估计** | 人体骨架关节点识别 |
| 📝 **文本分类** | 文档引用网络节点分类 |
| 🛒 **推荐系统** | 用户-商品二分图推荐 |
| 🚗 **车联网** | 车辆节点通信与路径规划 |

### 知识点 17：GNN 常用参考资源

- A Gentle Introduction to Graph Neural Networks (Basics, DeepWalk, GraphSage)
- Graph Neural Networks: A Review of Methods and Applications
- http://snap.stanford.edu/proj/embeddings-www/

---

## 📝 自测题

**GAN 部分：**
1. GAN 中的生成器和判别器各自的目标是什么？
2. WGAN 比原始 GAN 改进在哪？
3. CycleGAN 是怎么做到不需要配对数据也能做图像翻译的？

**Attention 部分：**
4. Self-Attention 中的 Q、K、V 分别代表什么？用你自己的比喻解释。
5. 为什么要除以 √d_k（Scaled Dot-Product）？
6. Multi-Head Attention 为什么比单头好？

**GNN 部分：**
7. CNN 为什么不能直接处理图结构数据？
8. GCN 的核心思想是什么？
9. 举出 3 个 GNN 在实际中的应用。

---

## 📖 推荐论文

- Goodfellow et al., Generative Adversarial Nets, NIPS 2014
- Radford et al., Unsupervised Representation Learning with DCGAN, 2016
- Arjovsky et al., Wasserstein GAN, ICML 2017
- Isola et al., Image-to-image translation with conditional adversarial networks, CVPR 2017
- Zhu et al., Unpaired image-to-image translation using cycle-consistent GANs, CVPR 2017
- Karras et al., A Style-Based Generator Architecture for GANs, CVPR 2019
- **⚡ Vaswani et al., Attention is All You Need, NIPS 2017**（Transformer 原论文）
- Hu et al., Squeeze-and-Excitation Networks, CVPR 2018
- Wang et al., Non-local Neural Networks, CVPR 2018
- Dosovitskiy et al., An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, ICLR 2021
- Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, ICLR 2017
