# 实验作业四：Vector RAG、GraphRAG 与 WikiRAG 对比实验

《人工智能——深度学习大模型智能体》配套实验 · 实验作业四
学号 2024312149 · 姓名 陈道延 · 完成日期 2026-10-07

在同一份「云山大学」虚构语料（15 篇 40~93 字的短文，另备 1 篇用于增量更新实验）上，亲手实现
三种 RAG 检索范式的**可运行简化版**并做公平对比：

- **Vector RAG**：滑动窗口分块 → 向量化 → 余弦相似度 Top-K → 按文档去重（教材 8.3 节）
- **GraphRAG**：三元组抽取 → networkx 有向图 → 贪心模块度社区发现 + 社区摘要 → Local（实体链接 + 最多三跳边游走）/ Global（社区摘要匹配）两类查询（教材 8.7 节）
- **WikiRAG**：把原文重写聚合成 9 个主题条目，条目级 Top-3 检索 → 映射回来源文档（教材 8.8 节）

三者检索单位不同（文本块 / 邻域三元组 / 条目），全部**映射回文档号**后按式 8-9、式 8-10 计算
Recall@5 与 MRR，共 20 道题（15 道内置 + 5 道自拟，覆盖事实 / 多跳 / 全局三类）。

## 环境与运行

```bash
pip install sentence-transformers networkx matplotlib python-docx
# 国内下载模型建议先设置镜像：export HF_ENDPOINT=https://hf-mirror.com

# 主评测：三范式 × 20 题，Top-5（写入 results_eval.json）
python hw4_rag.py --mode eval --dump results_eval.json

# Top-K 敏感性扫描（K=1/2/3/5）
python hw4_rag.py --mode sweep --dump results_sweep.json

# GraphRAG 的 Local / Global 两档对比
python hw4_rag.py --mode graph_scope --dump results_graph_scope.json

# 失败案例归因（区分「检到但排太后」与「根本没检到」）
python hw4_rag.py --mode failures --dump results_failures.json

# 增量更新成本（新增 doc16 张明远）
python hw4_rag.py --mode update --dump results_update.json

# 实体归一化前后对比（Q：「云大的校训是什么？」）
python hw4_rag.py --mode alias_test --dump results_alias.json

# 单题查看 / 混合检索 / 抽取式作答 / 解析器鲁棒性测试
python hw4_rag.py --mode vector  --q Q01          # 块级相似度面板（图 1 的数据来源）
python hw4_rag.py --mode graph   --q Q01          # 沿边走的三跳过程
python hw4_rag.py --mode wiki    --q Q11          # 条目级命中与映射
python hw4_rag.py --mode hybrid  --q Q07          # 混合检索（任务五 a 选做，RRF 融合）
python hw4_rag.py --mode answer  --sys graph --q Q01   # 抽取式作答（表 5 评分依据）
python hw4_rag.py --mode extract_test             # 7 个脏输出用例测 parse_triples
python hw4_rag.py --mrr_check 1,2,4               # 手算 MRR 复核（式 8-10）

# 图 1~3 与报告
python make_figures.py
python fill_report_hw4.py                         # 生成 2024312149_陈道延_实验作业四.docx
```

CPU 线程默认限制为 4（`LAB04_THREADS=4`，见 `hw4_rag.py` 顶部），给整机保留余量；本报告全部数据
均在 4 线程 CPU（无 GPU）下测得，整个流程（含模型下载）约 10 分钟。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `hw4_rag.py` | 核心脚本：语料 / 问题集 / 分块 / 向量检索 / LexicalIndex 词面检索 / HybridRAG（RRF）/ GraphRAG（建图 + 社区 + Local & Global）/ WikiRAG / ForcedScope 反事实对照 / 评测（Recall@5、MRR、per-question 排名）/ 抽取式作答 / `parse_triples` 容错解析器；含 `eval / vector / graph / wiki / hybrid / extract / sweep / graph_scope / failures / answer / update / alias_test / extract_test / figures` 模式 |
| `make_figures.py` | 生成图 1（Vector 块级相似度面板，含 ★ 命中证据）、图 2（知识图谱，节点大小 = 度数）、图 3（9 个条目及其来源文档） |
| `fill_report_hw4.py` | 读取各 `results_*.json`，自动填写教师下发的报告样板（封面、1.1、表 1–表 9、图 1–图 3、5.1/5.2 分析、6 道思考题、附录 A） |
| `results_eval.json` | 三范式（+ Hybrid）× 20 题 × Top-5 的逐题检索结果与三类均值（报告表 3 的数据源） |
| `results_sweep.json` | K=1/2/3/5 的 Recall / MRR 扫描（报告表 4） |
| `results_graph_scope.json` | GraphRAG Local vs Global 两档（报告 5.1 与表 8 的对照依据） |
| `results_failures.json` | 6 个失败案例（每范式 2 例）的完整排名与归因标签（报告表 8） |
| `results_update.json` | 增量更新：新增 doc16 后三范式各要改动什么、实测耗时、新题可检索性（报告表 6、思考题 4） |
| `results_alias.json` | 实体归一化前后对比（「云大」查询修复、模拟抽取噪声时节点 48→45） |
| `answers_faithfulness.txt` | 9 组抽取式作答原文（含标准证据、检索上下文、依据标注），报告表 5 的逐句评分依据 |
| `fig1_vector_retrieval.png` | 图 1  Vector RAG 检索命中示例（含相似度分数） |
| `fig2_knowledge_graph.png` | 图 2  云山大学知识图谱（简化可视化） |
| `fig3_wiki_entries.png` | 图 3  WikiRAG 条目化知识库示例 |
| `2024312149_陈道延_实验作业四.docx` | 提交用实验报告（按样板填写，含表 1–表 9、图 1–图 3、思考题与附录） |
| `作业四_VectorRAG_GraphRAG与WikiRAG对比实验_实验报告样板.docx` / `..._实验指南.pdf` | 教师下发的报告样板与实验指南（原件留档） |

## 主要结果（4 线程 CPU，Top-5，20 题）

### 1. 三范式 × 三类问题的 Recall@5 / MRR

| 范式 | 事实 Recall | 事实 MRR | 多跳 Recall | 多跳 MRR | 全局 Recall | 全局 MRR | 总 Recall | 总 MRR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vector RAG | 1.000 | 1.000 | 0.905 | 0.821 | **0.917** | **0.929** | **0.938** | **0.912** |
| GraphRAG | 1.000 | 1.000 | **1.000** | **1.000** | 0.750 | 0.505 | 0.912 | 0.827 |
| WikiRAG | 1.000 | 0.833 | 0.810 | 0.893 | 0.607 | 0.643 | 0.796 | 0.787 |
| Hybrid RRF（选做） | 1.000 | 0.833 | 0.857 | 0.905 | 0.845 | 0.833 | 0.896 | 0.858 |

三个可复现的结论：事实类是「基本盘」（K≥2 起三范式 Recall 全满，差异只在 MRR）；多跳类是 GraphRAG
的主场（唯一 7/7 完整命中，Recall 与 MRR 都是 1.000）；全局类胜负与直觉相反——Vector RAG 最强
（0.917 / 0.929），GraphRAG 最弱（0.750 / 0.505），原因是社区摘要与真实社区划分不匹配（见下）。
Hybrid 的词面 RRF 融合没能提高 recall，反而把 MRR 从 0.912 拉到 0.858，说明在这个小语料上词面分
带来的排序噪声大于收益。

### 2. Top-K 敏感性（Recall，内置 15 题）

| 范式 | 类型 | K=1 | K=2 | K=3 | K=5 |
| --- | --- | --- | --- | --- | --- |
| Vector RAG | 多跳 | 0.367 | 0.800 | 0.867 | 1.000 |
| Vector RAG | 全局 | 0.567 | 0.767 | 0.767 | 0.933 |
| GraphRAG | 多跳 | 0.433 | 0.767 | 0.933 | 1.000 |
| WikiRAG | 事实 | 0.600 | 1.000 | 1.000 | 1.000 |

跌幅排序为 **多跳 ≫ 全局 > 事实**：证据篇数决定所需窗口宽度，多跳题要同时装下 2~3 篇链上文档，
窗口一收窄就先牺牲「桥接」那一篇。工程含义：若 K 必须压到 2 以内，就该换 GraphRAG 的显式关系检索
（GraphRAG 在 K=3 就能把多跳拿满，Vector 要 K=5）。

### 3. 增量更新成本（新增 doc16「张明远」）

| 范式 | 要改动什么 | 实测耗时 |
| --- | --- | --- |
| Vector RAG | 追加 1 块并编码 1 条新向量，旧块与向量完全不动 | 只编码新增 0.009 s（全库 16 篇重建 0.117 s） |
| GraphRAG | 补抽 6 条三元组 → 重建图 + 社区发现（社区 10→11）→ 受影响社区摘要重写（1 次 LLM 调用） | 重建 0.293 s（不含 LLM 摘要） |
| WikiRAG | 在「人工智能学院教师」条目内追加 1 句，只重编码该条目，其余 8 条不动 | 0.108 s |

新问题「张明远的研究方向是什么？」在三范式中的 doc16 排名分别为 **1 / 1 / 3**。

### 4. 实体归一化（ALIASES + canon_ent / canon_text）

问「云大的校训是什么？」时，不归一化则实体链接为空、退化成全局查询，返回 doc09/doc11/doc13（全错）；
把「云大」映射到「云山大学」后正确返回 doc01。模拟抽取噪声（同一实体被拆成多个节点）时，归一化把
噪音图从 48 节点 / 43 边收敛回 45 节点 / 40 边，与干净图一致。

## 结论要点：什么问题该用什么 RAG

- **事实题**（答案在 1 篇文档内）：三范式都能拿满，选构建成本最低的 Vector RAG。
- **多跳题**（答案是一条跨文档关系链）：必须用 GraphRAG。单块向量装不下跨文档关系，桥接文档与问句的
  字面 / 语义重合都很弱；显式关系边可以「沿边走」，形状与问题结构同构。
- **全局归纳题**（答案分散在多篇）：本实验 Vector RAG 反而最强（答案句与问句共享较多词面、常有一个
  主文档承载大部分答案），GraphRAG 的社区摘要本应是最优形式（一段摘要天然覆盖多篇来源），但**收益
  高度依赖摘要是否覆盖到正确社区**——本实验社区发现得 10 个社区而摘要只预置 3 条，Q11 需要的 doc02
  不在被选中摘要的来源列表里，整题落空；WikiRAG 同理受「条目划分由人指定、覆盖不全」的约束。
- **高频更新**的语料选 Vector RAG（可在线增量，秒级）；**结构稳定、需要沿关系推理**的知识库选
  GraphRAG（构建与更新最贵）；由人或 LLM 持续维护的**主题型**知识库选 WikiRAG（条目独立可增改）。

## 说明与局限（如实记录）

1. **环境未配置 LLM API**（无 `LLM_API_BASE` / `API_KEY` / `MODEL`），因此 GraphRAG 的三元组使用
   预置抽取结果，生成环节用自实现的**抽取式作答**（`--mode answer`：bigram-IDF + 关系锚词 + 数值线索
   挑 1~2 句并标注来源文档）替代 LLM 生成，`--mode extract`（现场 LLM 抽取）未真实运行。抽取 prompt
   的 3 版迭代效果是用 `--mode extract_test` 的 7 个离线脏输出用例验证的（6 例可救回）。
   这一点已在报告表 1 与 4.3/思考题 6 中注明。
2. **社区摘要与社区数不一致**是实验口径问题而非代码缺陷：贪心模块度得到 10 个社区、预置摘要只有 3 条，
   导致 GraphRAG 的 global 档偏弱。报告选择如实报告 0.750 / 0.505 并把它写进失败案例（表 8），而不是
   调整摘要去凑一个好看的数字。真实 GraphRAG 流程应为每个社区都生成摘要。
3. **语料规模很小**（15 篇、每篇 40~93 字，15 个块），因此绝对指标偏高、K 的敏感性被放大，横向结论
   （哪种范式适合哪类问题）才是本实验真正要说明的；`HybridRAG` 的 RRF 融合在这种规模下不占优也属此因。

## 提交物

- `2024312149_陈道延_实验作业四.docx` —— 实验报告（按教师样板填写）
- 本仓库全部源码与结果文件（`results_*.json` 中可查到报告中每一个数字）
