# Composer 2 Technical Report: 强化学习 (RL) 机制深度分析笔记

**论文来源:** Cursor Research Team
**核心主题:** 面向 Agentic Software Engineering 的专门化模型训练，重点关注大规模强化学习的应用。

---

## 1. 基座模型与训练数据 (Base Model & Training Data)

### 1.1 基座模型与底层内核 (Base Model & Kernels)
* **基座模型选择:** 经过内部评估（对比了 GLM-5, DeepSeek V3.2 等模型的编程知识、状态跟踪和代码库困惑度指标），Composer 2 最终选择了 **Kimi K2.5** (一个 1.04T 总参数量 / 32B 激活参数量的 MoE 模型) 作为基础模型。
* **底层软件与算子:** 训练过程极其注重底层性能，大量使用了内部开发的定制算子（基于 CUDA, PTX 以及 ThunderKittens），并混合使用了 MXFP8 和 NVFP4 精度格式，特别针对 MoE 层的低精度训练进行了优化。

### 1.2 训练任务与数据构成
* **持续预训练阶段:** 使用了大量**以代码为主的数据混合 (code-dominated data mix)**。训练分为三个步骤：32k 序列长度的骨干训练、扩展到 256k 序列长度的长上下文训练，以及针对特定编程任务的简短 SFT (Supervised Fine-Tuning)。
* **强化学习阶段:** 训练数据来源于大量**模拟真实 Cursor 会话**的编程任务。与传统公开 benchmark（主要集中在隔离的 bug 修复）不同，其 RL 训练任务分布非常广泛且贴近现实，包含了：功能迭代 (Iterate On Feature)、调试 (Debugging)、新功能开发 (New Feature)、重构 (Refactor)、理解代码库 (Understanding Codebase) 等。

## 2. 训练范式概览
Composer 2 的训练主要分为两个阶段：
1. **持续预训练 (Continued Pretraining):** 在代码数据集上提升基础知识和编码能力，作为后续 Agentic RL 的良好基座。
2. **大规模强化学习 (Large-scale RL):** 核心目标是提升端到端编码性能，增强长视距推理、多步执行的准确性以及现实编码问题中的连贯性。

## 3. 核心强化学习方法 (Asynchronous RL)

Composer 2 采用了一种基于策略梯度 (Policy Gradient) 的强化学习架构，具有以下关键特性：

### 3.1 算法改进与 GRPO 变体
* **多样本策略梯度:** 对每个 prompt 采样多个样本，采用固定组大小 (fixed group size)。
* **单轮次机制 (Single-epoch regime):** 保证同一个 prompt 永远不会被训练两次，增强泛化能力。
* **去除长度惩罚:** 参考 Dr. GRPO 的做法，移除了 GRPO 中的长度标准化项，因为它会引入长度偏差 (length bias)。
* **取消优势函数标准化:** 不使用标准差来归一化 Group advantages。因为在退化情况下（即同组内的所有 rollout 正确性相同），归一化会过度放大微小的行为差异。
* **保留超长序列:** 并没有像一些研究那样 mask 掉超过最大序列长度的 rollout，因为 Composer 的 self-summary 系统在实践中很好地限制了这种情况。

### 3.2 KL 散度正则化 (KL Divergence Regularization)
* 很多开源 RL 框架使用 Schulman 提出的 $k_3 = (r-1) - \log r$ 作为 KL 的无偏估计。
* Composer 团队发现，在长 rollout 的 Agentic 场景中，随着参考策略 $q$ 和当前策略 $p$ 发生偏离，**$k_3$ 估计器的方差会急剧爆炸**。
* 因此，Composer 2 放弃了 $k_3$，采用了最标准的、尽管存在偏差但方差更稳定的估计器：**$k_1 = -\log r$**。

### 3.3 自我总结机制 (Self-Summarization)
* 为了应对长周期 (long-horizon) 任务的上下文限制，Composer 2 继承并强化了自我总结技术。
* 训练时，模型在多个生成周期之间生成 summary，整个链条中的**所有 token 都使用最终的奖励信号 (final reward) 进行更新**。
* 这种机制既能 upweight 好的推理轨迹，也能强化促成这些好轨迹的自我总结能力。同时，丢失关键信息的劣质 summary 会被降权。

### 3.4 辅助奖励与非线性长度惩罚
* **辅助奖励 (Auxiliary Rewards):** 关注开发者体验，引入了代码风格、沟通方式的奖励，以及对糟糕工具调用（如创建了一个 todo list 却不完成）的惩罚。
* **非线性长度惩罚:** 鼓励模型在简单任务上快速给出答案，同时允许在困难任务上进行长时间思考。其惩罚函数为下凸递增的非线性方程：
  $$C_{\text{length}\{k,q\}}(x) = \frac{(1+kx)^{1-q}-1}{k(1-q)}$$
  其中 $x$ 是 thinking tokens、tool calling tokens 等变量的加权组合。

## 4. 大规模 RL 基础设施 (RL Infrastructure)

由于 Agent Rollout 可能非常长，维持高度异步环境下的稳定性至关重要。为了尽量减少样本的 off-policy 程度，Composer 2 构建了极具特色的异步 RL 架构。

### 4.1 异步训练框架与飞行中权重更新 (Asynchronous Training & In-flight Updates)
* **完全解耦的服务栈:** 训练 (Training)、环境 (Environments)、推理 (Inference) 和评估 (Evaluations) 均为完全解耦的独立服务，基于 Ray 和 PyTorch 构建，支持全球规模的独立扩展。
* **中途/飞行中权重更新 (In-flight weight updates):** 因为 Agent 的轨迹生成非常耗时，为了防止推理节点（生成 rollout）的策略过于落后于训练节点的策略，**推理 worker 具备在 rollout 生成中途更新模型权重的能力**。这意味着在一个长 rollout 的后半段，模型生成 token 依赖的是更新后的、偏离度更小的策略（类似于 PipelineRL）。
* **快速同步机制:** 这种“准实时”的异步权重同步机制大大降低了长视距任务中样本的 off-policy 发散程度，保障了大规模强化学习的稳定性。

### 4.2 并行策略与底层训练软件设计 (Parallelism & Training Kernels)
为了支撑上述大规模的 RL 训练，团队在基础架构层（论文第 6 节）做了大量深入的底层软件设计与算子优化：
* **Context Parallelism (CP) 替代 TP:** 放弃使用 Tensor Parallelism (TP)，全面转向以 CP 作为处理长上下文的主要扩展轴。由于 CP 能保持完整的隐藏层维度（TP 会切片矩阵），因此显著降低了通信开销并提升了矩阵乘法效率。为解决因果注意力 (Causal Attention) 导致的 CP 节点间负载不均，团队采用了序列交错分割的技术。
* **解耦专家并行 (EP) 与 DeepEP 优化:** 创新性地将专家并行 (EP) 与 TP 解耦，通过 DP 和 CP 组合来提供 EP 容量。底层使用 **DeepEP** 库实现高吞吐的 token 派发与聚合，甚至在网络传输前将 token 压缩为 MXFP8 格式，并利用流水线技术实现了通信与计算的极致重叠 (Overlap)。
* **全局序列打包 (Global Sequence Packing):** 由于 RL 的各种 rollout 长度差异巨大，框架会在每次训练 step 前执行全局打包算法，以确保不同 DP (Data Parallel) 节点间的计算量保持严格均衡。
* **极端的内核定制与混合精度 (Kernels & Quantization):** 
  * MoE 层的**前向传播**采用了团队自研的 **NVFP4 变体**。因为他们发现原版 NVFP4 的 per-tensor 缩放会引发 RL 训练发散且泄露 token 信息，于是改为了 per-block 与 **per-token 缩放**。同时，实验发现 NVFP4 下必须使用严格的 IEEE 标准浮点运算，否则训练在几百步内就会崩溃。
  * MoE 层的**反向传播**（仅在训练集群运行）则采用了更高精度、更稳定的 **MXFP8** 格式，并使用了快速近似浮点运算。这种前反向不对称的精度设计，巧妙地平衡了线上推理集群的极速需求与 RL 训练的数值稳定性。

### 4.3 Anyrun 代码库环境
* 状态化的代码库环境 (Stateful codebase environments) 运行在基于 Firecracker VM 的 Anyrun 平台上。
* Anyrun 支持在**文件系统和内存级别**对完整的编码环境进行 Fork 和 Snapshot。这使得在 RL 过程中可以实现轨迹中途保存 (mid-trajectory rollout checkpointing)，极大地降低了长任务训练失败的成本。

### 4.4 MoE 路由重放 (Router Replay) 与 Delta 权重同步
* **专家路由不一致问题:** 由于 Composer 2 使用 Kimi K2.5 (32B active MoE) 作为基座，推理端和训练端的数值差异可能导致相同的 token 选择了不同的专家，从而引入策略梯度的噪声。
* **Router Replay:** 推理引擎返回每个 token 选中的专家索引，在训练前向传播时，**强制覆盖专家的分配以匹配推理阶段**。这样保证了计算出的对数概率与采样的分布完全一致。
* **Delta 权重同步:** 结合上述的异步训练架构，权重同步采用 Delta 压缩（仅上传新旧权重的差值），使得 1T 参数模型每次更新仅传输几 GB。上传和热加载 (hotload) 都在后台流水线化 (fully pipelined) 进行，绝对不阻塞训练和推理过程。

## 5. Composer 2 的核心创新总结

基于整份技术报告的分析，Composer 2 的最大创新并非发明了某种全新的基础算法，而是**将通用 LLM 转化为顶尖“智能体软件工程师”的系统性端到端工程创新**，可归纳为四大维度：

1. **彻底贴近真实的训练与评估范式 (CursorBench)**
   * 拒绝使用被严重污染、指令过于明确的公开数据集（如 SWE-bench）“刷榜”。
   * 创新性地提取了内部工程团队真实的、高度模糊、跨度极长的用户会话作为强化学习的环境，解决了传统模型“跑分高但实际难用”的痛点。
2. **针对长视距 (Long-horizon) 任务的 RL 针对性改造**
   * **提出自我总结 (Self-Summarization) 训练法：** 允许模型在长周期中不断生成自我总结，并将最终的成功奖励 (Reward) 分配给这些 summary，直接“教会”模型如何在有限窗口内提炼关键信息。
   * **修正 KL 估计的“深水区”缺陷：** 发现主流的 $k_3$ 估计器在超长序列中方差急剧爆炸，果断回退采用更稳定的 $k_1$ 估计器，保障了训练不崩塌。
3. **极具工程创意的异步 RL 基础设施 (Asynchronous Infrastructure)**
   * **飞行中权重更新 (In-flight weight updates)：** 由于长任务单次生成极为耗时，推理节点具备在**生成单个长 rollout 中途，实时拉取并更新最新模型权重**的能力，极大地缓解了长程 RL 中灾难性的 off-policy 问题。
   * **Router Replay 与增量热更新：** 解决了 MoE 模型跨异构集群专家路由不一致引发的梯度噪声，并通过传输极小的 Delta 权重实现了不阻塞推理的流水线化热加载。
   * **支持快照的任意运行环境 (Anyrun)：** 支持在系统/内存级别对庞大代码库打快照，允许失败的长 rollout 从中途恢复，成倍降低了试错成本。
4. **极致不对称的算子与混合精度设计**
   * 自研了 **per-token 缩放的 NVFP4** 内核，避开了官方原生 per-tensor 缩放导致的 RL 发散问题。
   * 在前向传播 (NVFP4, 需严格 IEEE 浮点) 和反向传播 (MXFP8, 可快速近似) 间采用了极端的不对称混合精度设计，完美兼顾了生产环境的极速生成与训练集群的数值稳定性。

## 6. 结论与启发
对于 Agentic RL，核心难点并不局限于算法本身，更在于如何构建一个**贴近真实场景 (CursorBench)**、**极高可用性 (Anyrun / Checkpointing)** 以及 **支持长上下文反馈 (Self-summarization & 稳定的 KL 估计)** 的系统。Composer 2 证明了在良好的基座模型之上，通过定制化的 RL 训练，可以在降低推理成本的同时，达到 SOTA 级别的代码 Agent 能力。
