# -*- coding: utf-8 -*-
"""
实验作业四：Vector RAG / GraphRAG / WikiRAG 对比实验
学号 2024312149   姓名 陈道延
对应教材第 8 章《检索增强生成》，框架依据实验指南第七节参考代码实现。

相对参考框架的主要修改（详见 README.md 与报告第 2、3 节）：
  1) 追加 5 道自拟题（20 题问题集），每题标注 evidence；
  2) 检索器保留「文档级完整排名」，支持把失败细分为「检到但排太后」与「根本没检到」；
  3) 实体归一化表 ALIASES + 归一化前后的可复现对比（--mode alias_test）；
  4) --mode sweep        K=1/2/3/5 窗口敏感性扫描（任务四⑤）；
  5) --mode hybrid       字符 bigram 词面分与向量分按 RRF 融合（任务五 a）；
  6) --mode update       新增 doc16 的增量更新成本实测（任务五 b）；
  7) --mode answer       无 API 环境下的抽取式作答（用于忠实度人工评分）；
  8) --mode extract_test 离线校验三元组解析器对脏 LLM 输出的鲁棒性；
  9) --dump 结果落盘 JSON，保证报告数据与脚本输出一致、可复核。

常用命令：
  python hw4_rag.py --mode eval                      # 三范式 × 20 题检索评测
  python hw4_rag.py --mode eval --k 1                # 窗口敏感性：K=1/2/3/5
  python hw4_rag.py --mode sweep                     # 一次跑完四档 K
  python hw4_rag.py --mode vector --q Q01            # 单题：检索（+ 生成，若有 API）
  python hw4_rag.py --mode graph --q Q01 --scope local
  python hw4_rag.py --mode hybrid --q Q07            # 混合检索（任务五 a）
  python hw4_rag.py --mode update                    # 增量更新实验（任务五 b）
  python hw4_rag.py --mode failures                  # 失败案例归因
  python hw4_rag.py --mode answer --sys graph --q Q01
  python hw4_rag.py --mrr_check 3,1,2                # MRR 手算复核（式 8-10）
"""
import argparse
import json
import math
import os
import re
import sys
import time
import zlib
from collections import Counter

# CPU 余量：默认只用 4 线程（可被环境变量覆盖）
THREADS = int(os.environ.get("LAB04_THREADS", "4"))
try:
    import torch
    torch.set_num_threads(THREADS)
except Exception:
    pass

# ============================================================
# 1. 语料库「云山大学」：15 篇虚构文档（三类问题的答案均已埋入）
# ============================================================
CORPUS = {
    "doc01": "云山大学位于岭南省云州市，建于1958年，是省属重点大学，现有全日制本科生约2.1万人。校训为「格物致知」。",
    "doc02": "人工智能学院成立于2018年，首任及现任院长为王启明教授。学院下设机器学习系、智能科学系、认知计算系三个系。",
    "doc03": "计算机学院前身为1985年成立的计算机系，现任院长陈国峰教授，设计算机科学与技术、软件工程两个本科专业。",
    "doc04": "人工智能学院教师：李文瀚，教授，2019年入职，研究方向为检索增强生成，主讲《大模型通识课》；赵婉晴，副教授，研究方向为知识图谱，主讲《知识图谱导论》；孙浩然，讲师，研究方向为强化学习。",
    "doc05": "计算机学院教师：周天宇，教授，研究方向为分布式系统；林小雨，副教授，研究方向为数据库系统。",
    "doc06": "《大模型通识课》课程代码AI2101，3学分、32学时，春季学期开设，授课教师李文瀚，面向全校本科生，无先修课程要求。",
    "doc07": "《知识图谱导论》课程代码AI3305，2学分，秋季学期开设，授课教师赵婉晴，面向人工智能学院研究生。",
    "doc08": "选课规则：本科生每学期最多修30学分；GPA低于2.0给予学术警告。先修要求：《知识图谱导论》需先修《数据结构》，《大模型通识课》无先修要求。",
    "doc09": "奖学金体系：国家奖学金8000元/年，评定比例约2%；校长奖学金20000元/年，全校每年10人；云山一等奖学金3000元/年，比例约5%。",
    "doc10": "科研平台：认知计算全国重点实验室依托人工智能学院建设；岭南超算中心由云山大学与省科技厅共建，计算机学院参与运行管理。",
    "doc11": "学生社团：AI协会，指导教师李文瀚，每周三晚组织论文研读；机器人战队，指导教师孙浩然，每年参加全国机器人大赛。",
    "doc12": "校历：春季学期3月2日开学、7月5日放暑假；秋季学期9月1日开学、次年1月15日放寒假。",
    "doc13": "图书馆藏书380万册，开放时间为每天7:00-22:00；人工智能分馆位于理科楼B座3层，收藏大模型与智能体专题图书。",
    "doc14": "校园交通：地铁3号线「云大站」距东门200米；校内校巴共5条线路，10分钟一班。",
    "doc15": "国际交流：学校与12个国家的47所高校签有交换协议；人工智能学院与新加坡南洋理工大学有本科联合培养项目。",
}

# 增量更新实验（任务五 b）用的第 16 篇文档：自拟内容
DOC16 = {
    "doc16": "人工智能学院新进教师：张明远，副教授，2026年入职，研究方向为多模态大模型，主讲新开设的《多模态智能》课程，并担任AI协会的联合指导教师。"
}

# ============================================================
# 2. 问题集：内置 3 类 × 5 题 + 自拟 5 题；evidence = 评测金标准
#    fact=局部事实  multi=多跳关系  global=全局归纳
# ============================================================
QUESTIONS = [
    {"id": "Q01", "type": "multi", "q": "《大模型通识课》授课教师所在学院的行政负责人是谁？",
     "evidence": ["doc04", "doc06", "doc02"]},
    {"id": "Q02", "type": "multi", "q": "AI协会的指导教师主讲课程的课程代码是什么？",
     "evidence": ["doc11", "doc04", "doc06"]},
    {"id": "Q03", "type": "multi", "q": "认知计算全国重点实验室依托的学院成立于哪一年？",
     "evidence": ["doc10", "doc02"]},
    {"id": "Q04", "type": "multi", "q": "《知识图谱导论》授课教师的研究方向是什么？",
     "evidence": ["doc07", "doc04"]},
    {"id": "Q05", "type": "multi", "q": "机器人战队的指导教师的研究方向是什么？",
     "evidence": ["doc11", "doc04"]},
    {"id": "Q06", "type": "fact", "q": "云山大学的校训是什么？", "evidence": ["doc01"]},
    {"id": "Q07", "type": "fact", "q": "国家奖学金的金额是多少？", "evidence": ["doc09"]},
    {"id": "Q08", "type": "fact", "q": "图书馆每天几点到几点开放？", "evidence": ["doc13"]},
    {"id": "Q09", "type": "fact", "q": "人工智能学院成立于哪一年？", "evidence": ["doc02"]},
    {"id": "Q10", "type": "fact", "q": "春季学期什么时候开学？", "evidence": ["doc12"]},
    {"id": "Q11", "type": "global", "q": "云山大学有哪些科研平台？分别依托谁建设？",
     "evidence": ["doc10", "doc02", "doc03"]},
    {"id": "Q12", "type": "global", "q": "学校的学生奖励体系包含哪些项目？", "evidence": ["doc09"]},
    {"id": "Q13", "type": "global", "q": "人工智能学院有哪些教师？各自做什么研究方向？",
     "evidence": ["doc04"]},
    {"id": "Q14", "type": "global", "q": "本科生选课有哪些限制和先修要求？", "evidence": ["doc08"]},
    {"id": "Q15", "type": "global", "q": "对想深入学习大模型的学生，学校提供哪些课程和课外活动？",
     "evidence": ["doc06", "doc11"]},
]

# 自拟 5 题（任务四①）：每条设计思路写在 "why" 字段里，便于报告表 2 引用
MY_QUESTIONS = [
    {"id": "Q16", "type": "multi", "q": "岭南超算中心参与运行管理的学院，其现任院长是谁？",
     "evidence": ["doc10", "doc03"],
     "why": "跨 doc10→doc03 的两跳链：先定位参与运行的学院，再取该学院院长；考察「共建单位」与「院长」两类关系的串联。"},
    {"id": "Q17", "type": "multi", "q": "秋季学期开设课程的授课教师，所在学院的院长是谁？",
     "evidence": ["doc07", "doc04", "doc02"],
     "why": "与 Q01 同构但换链：秋季学期→《知识图谱导论》→赵婉晴→人工智能学院→王启明，三条证据、两跳，检验多跳链上出现新实体时图检索是否仍沿边走。"},
    {"id": "Q18", "type": "fact", "q": "校长奖学金每年奖励多少人？",
     "evidence": ["doc09"],
     "why": "单块可答的局部事实题，但 Q07 问的是国家奖学金额度，此处换问「人数」，检验块内多事实并存时向量检索是否只在最相关那一块命中。"},
    {"id": "Q19", "type": "global", "q": "人工智能学院的教师分别指导哪些学生社团？",
     "evidence": ["doc04", "doc11"],
     "why": "答案分散在教师文档与社团文档两处，需要把「人物—社团」关系聚合后归纳；检验条目化/社区摘要的聚合能力。"},
    {"id": "Q20", "type": "global", "q": "学校在人工智能领域的教学、科研与对外合作分别有哪些安排？",
     "evidence": ["doc02", "doc06", "doc10", "doc15"],
     "why": "四篇文档跨度最大的归纳题（教学→科研→国际合作），专门用来放大「Top-5 窗口装不下四篇」与社区摘要覆盖面之间的矛盾。"},
]

ALL_QUESTIONS = QUESTIONS + MY_QUESTIONS

# ============================================================
# 3. 分块（教材 8.3 节：块长约 300 token，中文按 120 字近似，重叠 20 字）
# ============================================================
CHUNK_SIZE = 120
CHUNK_OVERLAP = 20


def chunk_docs(size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, corpus=None):
    """滑动窗口分块；末块不足 size 时保留剩余文本，步长 = size - overlap。"""
    corpus = corpus or CORPUS
    chunks, step = [], size - overlap
    for doc_id, text in corpus.items():
        i = 0
        while True:
            chunks.append((doc_id, text[i:i + size]))
            if i + size >= len(text):
                break
            i += step
    return chunks


# ============================================================
# 4. 嵌入后端：优先 sentence-transformers（教材 8.4 节）；
#    未安装/无网络时自动降级为字符 3-gram 相似度（流程可跑通，精度有限）
# ============================================================
class Embedder:
    def __init__(self, model_name):
        self.ok = False
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.ok = True
            print(f"[embed] 已加载 {model_name}")
        except Exception as e:
            print(f"[embed] 加载失败（{type(e).__name__}），降级为字符 3-gram 相似度（精度有限）")

    def encode(self, texts):
        import numpy as np
        if self.ok:
            return self.model.encode(list(texts), normalize_embeddings=True)
        M = np.zeros((len(texts), 512))
        for r, t in enumerate(texts):
            t = re.sub(r"\s", "", t)
            for i in range(max(1, len(t) - 2)):
                g = t[i:i + 3] if len(t) >= 3 else t
                M[r, zlib.crc32(g.encode("utf-8")) % 512] += 1
            n = np.linalg.norm(M[r]) or 1.0
            M[r] /= n
        return M


# ============================================================
# 4.5 词面检索工具：字符 bigram 的 IDF 加权打分（混合检索与抽取式作答共用）
# ============================================================
def bigrams(text):
    t = re.sub(r"[\s，。；：、！？「」《》（）\-—0-9a-zA-Z]", "", text)
    return [t[i:i + 2] for i in range(max(1, len(t) - 1))]


class LexicalIndex:
    """字符 bigram + IDF 的轻量词面检索（不引入 jieba 等额外依赖）。"""

    def __init__(self, texts):
        self.docs = list(texts)
        self.bow = [Counter(bigrams(t)) for t in self.docs]
        df = Counter()
        for b in self.bow:
            df.update(b.keys())
        n = max(1, len(self.docs))
        self.idf = {g: math.log(1 + (n - c + 0.5) / (c + 0.5)) for g, c in df.items()}

    def scores(self, query):
        qb = Counter(bigrams(query))
        out = []
        for b in self.bow:
            s = 0.0
            for g, qn in qb.items():
                if g in b:
                    s += self.idf.get(g, 0.0) * min(b[g], qn)
            if b:
                s /= math.sqrt(sum(v * v for v in b.values())) or 1.0
            out.append(s)
        return out


def _dedup(scored, k):
    docs, seen = [], set()
    for doc, s in scored:  # scored 已按分数降序
        if doc not in seen:
            docs.append(doc)
            seen.add(doc)
            if len(docs) >= k:
                break
    return docs


def _dedup_rank(scored):
    """不截断的文档级去重排名（用于失败归因：区分「检到但排太后」与「根本没检到」）。"""
    ranked, seen = [], set()
    for doc, s in scored:
        if doc not in seen:
            ranked.append(doc)
            seen.add(doc)
    return ranked


def rrf_fuse(rank_lists, k_const=60):
    """倒数排名融合（教材式 8-6 的 RRF 形式）：score = Σ 1/(k + rank)。"""
    fused = {}
    for rl in rank_lists:
        for rank, item in enumerate(rl, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k_const + rank)
    return sorted(fused.items(), key=lambda kv: -kv[1])


# ============================================================
# 5. Vector RAG（教材 8.2~8.4 节）：块向量 → 余弦 Top-K → 文档序列
# ============================================================
class VectorRAG:
    name = "Vector RAG"

    def __init__(self, emb, corpus=None):
        self.emb = emb
        self.corpus = corpus or CORPUS
        self.chunks = chunk_docs(corpus=self.corpus)
        t0 = time.time()
        self.M = emb.encode([c[1] for c in self.chunks])
        self.build_time = time.time() - t0

    def _rank_chunks(self, query):
        qv = self.emb.encode([query])[0]
        scores = self.M @ qv
        order = scores.argsort()[::-1]
        return [(self.chunks[i][0], float(scores[i])) for i in order]

    def retrieve_ranked(self, query, scope=None):
        return _dedup_rank(self._rank_chunks(query))

    def retrieve(self, query, k=5, scope=None):
        return self.retrieve_ranked(query)[:k]

    def top_chunks(self, query, n=5):
        out, seen = [], set()
        for doc, s in self._rank_chunks(query):
            key = (doc, s)
            if key in seen:
                continue
            seen.add(key)
            out.append((doc, s))
            if len(out) >= n:
                break
        return out


# ============================================================
# 5.5 混合检索 Hybrid RAG（任务五 a）：词面 RRF 融合
# ============================================================
class HybridRAG(VectorRAG):
    name = "Hybrid RRF"

    def __init__(self, emb, corpus=None):
        super().__init__(emb, corpus)
        self.lex = LexicalIndex([c[1] for c in self.chunks])

    def retrieve_ranked(self, query, scope=None):
        vec_rank = [f"{d}#{i}" for i, (d, _) in enumerate(self._rank_chunks(query))]
        lex_scores = self.lex.scores(query)
        lex_order = sorted(range(len(lex_scores)), key=lambda i: -lex_scores[i])
        lex_rank = [f"{self.chunks[i][0]}#{i}" for i in lex_order if lex_scores[i] > 0]
        fused = rrf_fuse([vec_rank, lex_rank])
        return _dedup_rank([(key.split("#")[0], s) for key, s in fused])


# ============================================================
# 6. GraphRAG（教材 8.7 节简化版）
#    预抽取三元组(头实体, 关系, 尾实体, 来源文档)；--mode extract 可现场抽取
# ============================================================
PRE_TRIPLES = [
    ("李文瀚", "任职于", "人工智能学院", "doc04"), ("李文瀚", "职称", "教授", "doc04"),
    ("李文瀚", "入职年份", "2019年", "doc04"), ("李文瀚", "研究方向", "检索增强生成", "doc04"),
    ("李文瀚", "主讲", "大模型通识课", "doc06"), ("李文瀚", "指导", "AI协会", "doc11"),
    ("赵婉晴", "任职于", "人工智能学院", "doc04"), ("赵婉晴", "研究方向", "知识图谱", "doc04"),
    ("赵婉晴", "主讲", "知识图谱导论", "doc07"), ("孙浩然", "任职于", "人工智能学院", "doc04"),
    ("孙浩然", "研究方向", "强化学习", "doc04"), ("孙浩然", "指导", "机器人战队", "doc11"),
    ("人工智能学院", "院长", "王启明", "doc02"), ("人工智能学院", "成立于", "2018年", "doc02"),
    ("人工智能学院", "下设系", "机器学习系", "doc02"), ("人工智能学院", "下设系", "智能科学系", "doc02"),
    ("人工智能学院", "下设系", "认知计算系", "doc02"), ("认知计算全国重点实验室", "依托", "人工智能学院", "doc10"),
    ("岭南超算中心", "参与共建", "计算机学院", "doc10"), ("计算机学院", "院长", "陈国峰", "doc03"),
    ("计算机学院", "前身", "计算机系", "doc03"), ("周天宇", "任职于", "计算机学院", "doc05"),
    ("周天宇", "研究方向", "分布式系统", "doc05"), ("林小雨", "任职于", "计算机学院", "doc05"),
    ("林小雨", "研究方向", "数据库系统", "doc05"), ("大模型通识课", "课程代码", "AI2101", "doc06"),
    ("大模型通识课", "学分", "3学分", "doc06"), ("大模型通识课", "开设学期", "春季学期", "doc06"),
    ("知识图谱导论", "课程代码", "AI3305", "doc07"), ("知识图谱导论", "先修课程", "数据结构", "doc08"),
    ("AI协会", "指导教师", "李文瀚", "doc11"), ("机器人战队", "指导教师", "孙浩然", "doc11"),
    ("国家奖学金", "金额", "8000元/年", "doc09"), ("校长奖学金", "金额", "20000元/年", "doc09"),
    ("云山一等奖学金", "金额", "3000元/年", "doc09"), ("云山大学", "校训", "格物致知", "doc01"),
    ("云山大学", "建校于", "1958年", "doc01"), ("云山大学", "位于", "岭南省云州市", "doc01"),
    ("图书馆", "藏书量", "380万册", "doc13"), ("春季学期", "开学日期", "3月2日", "doc12"),
]

# 新增 doc16 时按同一抽取粒度补的三元组（增量更新实验用）
TRIPLES_DOC16 = [
    ("张明远", "任职于", "人工智能学院", "doc16"), ("张明远", "职称", "副教授", "doc16"),
    ("张明远", "入职年份", "2026年", "doc16"), ("张明远", "研究方向", "多模态大模型", "doc16"),
    ("张明远", "主讲", "多模态智能", "doc16"), ("张明远", "指导", "AI协会", "doc16"),
]

# 社区摘要（教材 8.7 节由 LLM 生成；此处预置 3 条，对应图的三个稠密子区）
COMMUNITY_SUMMARIES = [
    ("社区C1 教学与课程：人工智能学院设三个系，开设《大模型通识课》（AI2101，李文瀚主讲）与"
     "《知识图谱导论》（AI3305，赵婉晴主讲），AI3305 需先修《数据结构》。",
     ["doc02", "doc04", "doc06", "doc07", "doc08"]),
    ("社区C2 科研与国际：认知计算全国重点实验室依托人工智能学院，岭南超算中心由计算机学院参与共建；"
     "学院与新加坡南洋理工大学有联合培养项目。",
     ["doc02", "doc03", "doc10", "doc15"]),
    ("社区C3 学生生活：奖学金分国家（8000 元）、校长（20000 元）、云山一等（3000 元）三级；"
     "社团有 AI 协会与机器人战队，图书馆藏书 380 万册、7:00-22:00 开放。",
     ["doc09", "doc11", "doc13"]),
]

# ------------------------------------------------------------
# 6.1 实体归一化（指南 Q4）：词面变体合并到规范实体
# ------------------------------------------------------------
ALIASES = {
    "云大": "云山大学",
    "云山大学校": "云山大学",
    "AI 协会": "AI协会",
    "ai协会": "AI协会",
    "人工智能学院院长": "人工智能学院",
    "南洋理工": "新加坡南洋理工大学",
}


def canon_ent(name):
    """实体名归一化：去掉书名号/引号与空白，再查别名表。"""
    if name is None:
        return None
    s = re.sub(r"[《》「」\s]", "", str(name))
    return ALIASES.get(s, s)


def canon_text(text):
    """查询串归一化：把别名替换成规范实体，便于实体匹配（含「云大」→「云山大学」）。"""
    s = re.sub(r"[《》「」\s]", "", text)
    for a, c in sorted(ALIASES.items(), key=lambda kv: -len(kv[0])):
        if a in s and c not in s:
            s = s.replace(a, c)
    return s


class GraphRAG:
    name = "GraphRAG"

    def __init__(self, emb, triples=None, summaries=None, normalize=True, corpus=None):
        import networkx as nx
        self.emb = emb
        self.normalize = normalize
        self.corpus = corpus or CORPUS
        self.triples = list(triples if triples is not None else PRE_TRIPLES)
        self.summaries = list(summaries if summaries is not None else COMMUNITY_SUMMARIES)
        t0 = time.time()
        self.G = nx.DiGraph()
        for h, r, t, src in self.triples:
            if normalize:
                h, t = canon_ent(h), canon_ent(t)
            self.G.add_edge(h, t, rel=r, src=src)
        try:
            # 教材 8.7 节用 Louvain；此处用 networkx 内置贪心模块度社区发现
            comms = nx.community.greedy_modularity_communities(self.G.to_undirected())
        except Exception:
            comms = [set(self.G.nodes)]
        self.comms = comms
        self.summ_M = emb.encode([s for s, _ in self.summaries])
        self.build_time = time.time() - t0
        print(f"[graph] 实体{self.G.number_of_nodes()} 个 / 三元组{len(self.triples)} 条 / 社区{len(comms)} 个")

    # ---- 实体匹配：查询里出现的最长实体（归一化后比较） ----
    def _entities(self, query, topn=2):
        qn = canon_text(query) if self.normalize else re.sub(r"[《》「」\s]", "", query)
        ents = sorted([n for n in self.G.nodes if n in qn], key=len, reverse=True)
        return ents[:topn]

    # ---- Local 查询：沿边最多走三跳，收集邻域三元组 ----
    def _local_scored(self, query):
        ents = self._entities(query)
        seen, scored = set(), []
        frontier = ents[:]
        for _ in range(3):
            nxt = []
            for e in frontier:
                edges = list(self.G.out_edges(e, data=True)) + list(self.G.in_edges(e, data=True))
                for h, t, d in edges:
                    key = (h, d["rel"], t)
                    if key not in seen:
                        seen.add(key)
                        scored.append((d["src"], f"{h} —{d['rel']}→ {t}"))
                    other = t if h == e else h
                    if other not in frontier:
                        nxt.append(other)
            nxt.sort(key=lambda x: 0 if self.G.out_degree(x) > 0 else 1)  # 实体优先展开
            frontier = nxt[:6]
        return ents, scored

    def _local(self, query):
        ents, scored = self._local_scored(query)
        if not scored:
            return None, [], None  # 未命中实体 → 退化为全局查询
        qv = self.emb.encode([query])[0]
        sims = self.emb.encode([t for _, t in scored]) @ qv
        best = {}
        for (doc, txt), s in zip(scored, sims):
            if doc not in best or s > best[doc]:
                best[doc] = float(s)
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        return ents, scored, ranked

    def _global_ranked(self, query):
        qv = self.emb.encode([query])[0]
        scores = self.summ_M @ qv
        top = scores.argsort()[::-1][:2]  # 取最相关的 2 个社区
        docs, hits = [], []
        for ci in top:
            hits.append({"community": ci, "score": float(scores[ci]), "summary": self.summaries[ci][0][:40]})
            for d in self.summaries[ci][1]:
                if d not in docs:
                    docs.append(d)
        return docs, hits

    def retrieve_ranked(self, query, scope="local"):
        if scope == "global":
            docs, _ = self._global_ranked(query)
            return docs
        ents, scored, ranked = self._local(query)
        if ranked is None:
            docs, _ = self._global_ranked(query)
            return docs
        return _dedup_rank(ranked)

    def retrieve(self, query, k=5, scope="local"):
        return self.retrieve_ranked(query, scope)[:k]

    def explain(self, query, scope="local"):
        """打印一次检索的中间过程（报告 2.2 的「沿边走」命中过程截图用）。"""
        out = [f"实体匹配：{self._entities(query) or '（无，退化为全局查询）'}"]
        if scope == "local":
            ents, scored, ranked = self._local(query)
            if scored:
                out.append("沿边收集到的三元组（最多三跳）：")
                for doc, txt in scored[:12]:
                    out.append(f"  [{doc}] {txt}")
                out.append("邻域三元组按问题相似度排序后的文档：")
                for doc, s in ranked:
                    out.append(f"  {doc}  sim={s:.4f}")
        docs, hits = self._global_ranked(query)
        out.append("社区摘要匹配（Global）：")
        for h in hits:
            out.append(f"  社区{h['community']} sim={h['score']:.4f} :: {h['summary']}…")
        return "\n".join(out)


# ============================================================
# 7. WikiRAG（教材 8.8 节）：条目化知识库，检索单位是「条目」
#    条目 = 标题 + 聚合正文（重写而非复制原文），映射回来源文档评测
# ============================================================
PRE_ENTRIES = [
    ("云山大学", "云山大学位于岭南省云州市，建于1958年，省属重点大学，本科生约2.1万人，校训「格物致知」。"
                 "地铁3号线「云大站」距东门200米，校巴5条线路。校历：春季学期3月2日开学。",
     ["doc01", "doc12", "doc14"]),
    ("人工智能学院", "人工智能学院成立于2018年，院长王启明教授，下设机器学习系、智能科学系、认知计算系。"
                     "认知计算全国重点实验室依托本院建设。与新加坡南洋理工大学有本科联合培养项目（每年15人）。",
     ["doc02", "doc10", "doc15"]),
    ("计算机学院", "计算机学院前身是1985年成立的计算机系，院长陈国峰教授，设计算机科学与技术、软件工程两个专业。"
                   "参与岭南超算中心运行管理。",
     ["doc03", "doc10"]),
    ("人工智能学院教师", "李文瀚：教授，2019年入职，研究检索增强生成，主讲《大模型通识课》，指导AI协会。"
                         "赵婉晴：副教授，研究知识图谱，主讲《知识图谱导论》。孙浩然：讲师，研究强化学习，指导机器人战队。",
     ["doc04", "doc11"]),
    ("计算机学院教师", "周天宇：教授，研究分布式系统。林小雨：副教授，研究数据库系统。",
     ["doc05"]),
    ("大模型通识课", "《大模型通识课》（AI2101）：3学分32学时，春季学期开设，李文瀚主讲，面向全校本科生，"
                     "无先修要求，教材《人工智能——深度学习大模型智能体》。",
     ["doc06"]),
    ("知识图谱导论", "《知识图谱导论》（AI3305）：2学分，秋季学期，赵婉晴主讲，面向人工智能学院研究生，"
                     "需先修《数据结构》。",
     ["doc07", "doc08"]),
    ("奖学金体系", "国家奖学金8000元/年（约2%）；校长奖学金20000元/年（全校10人）；"
                   "云山一等奖学金3000元/年（约5%）。",
     ["doc09"]),
    ("学生社团与校园生活", "AI协会：指导教师李文瀚，每周三晚论文研读。机器人战队：指导教师孙浩然，"
                           "参加全国机器人大赛。图书馆藏书380万册，每天7:00-22:00开放，人工智能分馆在理科楼B座。",
     ["doc11", "doc13"]),
]

# 增量更新（任务五 b）：doc16 带来的条目改动（追加一句，不新建条目）
ENTRY_DOC16 = ("人工智能学院教师", "李文瀚：教授，2019年入职，研究检索增强生成，主讲《大模型通识课》，指导AI协会。"
                                  "赵婉晴：副教授，研究知识图谱，主讲《知识图谱导论》。孙浩然：讲师，研究强化学习，指导机器人战队。"
                                  "张明远：副教授，2026年入职，研究多模态大模型，主讲《多模态智能》，联合指导AI协会。",
               ["doc04", "doc11", "doc16"])


class WikiRAG:
    name = "WikiRAG"

    def __init__(self, emb, entries=None, corpus=None):
        self.emb = emb
        self.entries = list(entries if entries is not None else PRE_ENTRIES)
        self.corpus = corpus or CORPUS
        t0 = time.time()
        self.M = emb.encode([b for _, b, _ in self.entries])
        self.build_time = time.time() - t0

    def _rank_entries(self, query):
        qv = self.emb.encode([query])[0]
        scores = self.M @ qv
        return [(int(i), float(scores[i])) for i in scores.argsort()[::-1]]

    def retrieve_ranked(self, query, scope=None):
        docs = []
        for i, s in self._rank_entries(query)[:3]:  # 条目级 Top-3
            for d in self.entries[i][2]:
                if d not in docs:
                    docs.append(d)
        return docs

    def retrieve(self, query, k=5, scope=None):
        return self.retrieve_ranked(query)[:k]

    def top_entries(self, query, n=3):
        return [(self.entries[i][0], s) for i, s in self._rank_entries(query)[:n]]


class ForcedScope:
    """把某个检索器固定在 local / global 作用域（任务二：GraphRAG 两类查询对比）。"""

    def __init__(self, system, scope):
        self.s, self.scope, self.name = system, scope, system.name

    def retrieve(self, query, k=5, scope=None):
        return self.s.retrieve(query, k, self.scope)

    def retrieve_ranked(self, query, scope=None):
        return self.s.retrieve_ranked(query, self.scope)


# ============================================================
# 8. 评测（教材式 8-9 / 8-10）：检索单位统一映射回「文档」再算指标
# ============================================================
def evaluate(systems, questions=None, k=5, detail=False, tag=""):
    questions = questions or ALL_QUESTIONS
    result = {"k": k, "tag": tag, "n_questions": len(questions), "systems": {}}
    print(f"\n===== 检索评测（文档级，Top-{k}，共 {len(questions)} 题）=====")
    print(f"{'范式':<14}{'问题类型':<10}{'Recall@' + str(k):>11}{'MRR':>9}   完整命中")
    for name, s in systems.items():
        result["systems"][name] = {"by_type": {}, "per_question": {}}
        for tp in ("fact", "multi", "global"):
            qs = [q for q in questions if q["type"] == tp]
            if not qs:
                continue
            recs, rrs, full = [], [], 0
            for q in qs:
                docs = s.retrieve(q["q"], k)
                ev = set(q["evidence"])
                got = ev & set(docs[:k])
                recs.append(len(got) / len(ev))
                rank = next((i + 1 for i, d in enumerate(docs) if d in ev), 0)
                rrs.append(1.0 / rank if rank else 0.0)
                full += (got == ev)
                result["systems"][name]["per_question"][q["id"]] = {
                    "type": tp, "query": q["q"], "evidence": sorted(ev),
                    "retrieved": docs[:k], "hit": sorted(got), "miss": sorted(ev - got),
                    "first_hit_rank": rank, "recall": len(got) / len(ev),
                }
            result["systems"][name]["by_type"][tp] = {
                "recall": sum(recs) / len(recs), "mrr": sum(rrs) / len(rrs),
                "full_hit": full, "n": len(qs),
            }
            print(f"{name:<16}{tp:<12}{sum(recs)/len(recs):>9.3f}{sum(rrs)/len(rrs):>9.3f}   {full}/{len(qs)}")
        allq = list(result["systems"][name]["per_question"].values())
        result["systems"][name]["overall"] = {
            "recall": sum(x["recall"] for x in allq) / len(allq),
            "mrr": sum((1.0 / x["first_hit_rank"] if x["first_hit_rank"] else 0.0) for x in allq) / len(allq),
        }
        print(f"{name:<16}{'全部':<12}{result['systems'][name]['overall']['recall']:>9.3f}"
              f"{result['systems'][name]['overall']['mrr']:>9.3f}")
    if detail:
        for name, s in result["systems"].items():
            print(f"\n--- {name} 逐题 ---")
            for qid, info in s["per_question"].items():
                flag = "OK " if not info["miss"] else "MISS"
                print(f"  {qid} {info['type']:<7}{flag} rank={info['first_hit_rank']} "
                      f"检索={info['retrieved']} 漏={info['miss']}")
    return result


# ============================================================
# 9. LLM 生成环节（可选）：任一 OpenAI 兼容接口
#    环境变量：LLM_API_BASE / LLM_API_KEY / LLM_MODEL
# ============================================================
EXTRACT_PROMPT = """从下面这篇大学介绍文档中抽取「头实体-关系-尾实体」三元组，只输出 JSON 数组，
不要输出任何其他文字，格式如[["李文瀚","任职于","人工智能学院"]]。
关系用 2~4 字短语（任职于/研究方向/主讲/成立于等），一篇文档抽 5~10 条。
文档：
{doc}"""

ANSWER_PROMPT = """你是问答助手。只依据下面资料回答问题；资料不足以回答时明确说「资料中未提及」，禁止编造。
问题：{q}
资料：
{ctx}
用 1~3 句话回答，句末用括号注明用到的资料编号。"""


def llm_chat(prompt, temperature=0.0):
    from openai import OpenAI
    base = os.environ.get("LLM_API_BASE")
    key = os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL")
    if not (base and key and model):
        sys.exit("未配置 LLM：请设置环境变量 LLM_API_BASE / LLM_API_KEY / LLM_MODEL")
    client = OpenAI(base_url=base, api_key=key)
    resp = client.chat.completions.create(
        model=model, temperature=temperature,
        messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content


def parse_triples(raw):
    """把 LLM 返回的文本解析成三元组列表；对常见脏输出做容错。
    处理：markdown 代码围栏、前后解说文字、中文标点、尾逗号、单引号 JSON。"""
    if not raw:
        return []
    s = raw.strip()
    s = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", s).strip()          # 去代码围栏
    s = s.replace("“", '"').replace("”", '"').replace("'", '"')
    s = s.replace("，", ",").replace("：", ":")                       # 中文标点
    l = s.find("[")
    if l < 0:
        return []
    r = s.rfind("]")
    s = s[l:r + 1] if r > l else s[l:]
    s = re.sub(r",\s*([\]}])", r"\1", s)                             # 尾逗号

    def _try(text):
        try:
            v = json.loads(text)
        except Exception:
            return None
        return v if isinstance(v, list) else None

    arr = _try(s)
    if arr is None:
        # 修复阶梯：截断的输出补上缺失的引号/中括号后再试
        for suffix in ('"]', '"]]', ']', ']]', '"}]', '"]}]'):
            arr = _try(s + suffix)
            if arr is not None:
                break
    if arr is None:
        # 最后一招：逐个抓取已闭合的 [ ... ] 片段
        salvaged = []
        for m in re.finditer(r"\[[^\[\]]*\]", s):
            piece = _try(m.group(0))
            if piece:
                salvaged += piece
        arr = salvaged or None
    if arr is None:
        return []
    out = []
    for item in arr if isinstance(arr, list) else []:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            h, rel, t = [str(x).strip() for x in item[:3]]
            if h and rel and t:
                out.append((h, rel, t))
    return out


def extract_triples(corpus=None):
    corpus = corpus or CORPUS
    t0, out, bad = time.time(), [], 0
    for doc_id, text in corpus.items():
        raw = llm_chat(EXTRACT_PROMPT.replace("{doc}", text))
        triples = parse_triples(raw)
        if not triples:
            bad += 1
            print(f"[warn] {doc_id} 解析失败：{raw[:60]}...")
            continue
        out += [(h, r, t, doc_id) for h, r, t in triples]
        print(f"{doc_id}: 抽到 {len(triples)} 条")
    print(f"\n共 {len(out)} 条 / 解析失败 {bad} 篇，耗时 {time.time() - t0:.0f}s。")
    print("请人工抽检后，把高质量三元组替换进脚本 PRE_TRIPLES。")
    return out


# ----------------------------------------------
# 9.1 无 API 环境下的抽取式作答（忠实度评分用）
# ----------------------------------------------
STOPWORDS = set("的是什么了在和有与及分别哪些哪一年多少谁请介绍学校大学问答题")

# 关系锚词：问句与句子同时命中时给加成，缓解「只靠字面重合选错句」
REL_ANCHORS = ("校训", "金额", "开放时间", "开学", "放假", "成立于", "成立", "院长", "教师", "研究方向",
               "主讲", "课程代码", "学分", "学时", "指导教师", "社团", "平台", "依托", "共建", "比例",
               "人数", "项目", "协议", "藏书", "专业", "系", "年")
# 答案类型偏好：问数字/时间类问题时优先含数字的句子
NUMERIC_HINTS = ("多少", "几人", "哪一年", "几点", "什么时候", "时间", "日期", "金额", "比例", "册", "人")


def _sentences(text):
    return [s for s in re.split(r"[。；；]", text) if s.strip()]


def extractive_answer(question, docs, corpus=None, max_sent=2):
    """不调用 LLM 的抽取式作答：在「检索到的文档」里选最相关的 1~2 句并标注来源。
    它是受控、可复现的降级方案——正因为不会凭空生成，它出错时暴露的正是
    「检索到错误/无关资料被带偏」，而这恰好是忠实度要度量的问题。"""
    corpus = corpus or CORPUS
    cand, metas = [], []
    for d in docs:
        for s in _sentences(corpus.get(d, "")):
            cand.append(s)
            metas.append(d)
    if not cand:
        return "资料中未提及。", []
    lex = LexicalIndex(cand)
    base = lex.scores(question)
    qn = canon_text(question)
    q_anchors = [a for a in REL_ANCHORS if a in qn]
    asks_number = any(h in question for h in NUMERIC_HINTS)
    ents = [n for n in _all_entities() if n in qn]
    scores = []
    for i, s in enumerate(cand):
        sc = base[i]
        if any(a in s for a in q_anchors):
            sc += 1.5                              # 关系锚词命中
        if asks_number and re.search(r"[0-9０-９]", s):
            sc += 0.8                              # 问数量/时间时偏好含数字的句子
        if ents and any(e in s for e in ents):
            sc += 0.6                              # 含问句中的实体
        scores.append(sc)
    ranked = sorted(range(len(cand)), key=lambda i: -scores[i])
    picked, cited = [], []
    for i in ranked:
        if len(picked) >= max_sent:
            break
        if scores[i] <= 0:
            break
        if picked and any(cand[i] in p for p in picked):
            continue
        picked.append(cand[i])
        if metas[i] not in cited:
            cited.append(metas[i])
    if not picked:
        picked, cited = [cand[ranked[0]]], [metas[ranked[0]]]
    ans = "；".join(picked) + "（依据 " + "、".join(cited) + "）"
    return ans, cited


def _all_entities():
    ents = set()
    for h, r, t, _ in PRE_TRIPLES + TRIPLES_DOC16:
        ents.add(h)
        ents.add(t)
    for title, _, _ in PRE_ENTRIES:
        ents.add(title)
    return sorted(ents, key=len, reverse=True)


# ============================================================
# 10. 主入口
# ============================================================
def build_systems(emb, args):
    """按模式构建检索器；评测/扫描类模式默认四套（三范式 + 任务五 a 的混合检索）。"""
    want = {
        "eval": ["vector", "graph", "wiki", "hybrid"],
        "sweep": ["vector", "graph", "wiki", "hybrid"],
        "failures": ["vector", "graph", "wiki", "hybrid"],
        "answer": ["vector", "graph", "wiki", "hybrid"],
        "vector": ["vector"], "graph": ["graph"], "wiki": ["wiki"], "hybrid": ["hybrid"],
    }.get(args.mode, [])
    systems = {}
    for n in want:
        if n == "vector":
            systems["Vector RAG"] = VectorRAG(emb)
        elif n == "hybrid":
            systems["Hybrid RRF"] = HybridRAG(emb)
        elif n == "graph":
            systems["GraphRAG"] = GraphRAG(emb)
        elif n == "wiki":
            systems["WikiRAG"] = WikiRAG(emb)
    return systems


def dump(path, obj):
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
        print(f"[dump] 已写入 {path}")


def main():
    ap = argparse.ArgumentParser(description="实验作业四：三种 RAG 对比实验")
    ap.add_argument("--mode", default="eval",
                    choices=["eval", "vector", "graph", "wiki", "hybrid", "extract",
                             "sweep", "graph_scope", "failures", "answer", "update", "alias_test",
                             "extract_test", "figures"])
    ap.add_argument("--q", default="Q01", help="问题编号，如 Q01")
    ap.add_argument("--sys", default="vector", choices=["vector", "graph", "wiki", "hybrid"],
                    help="--mode answer 时指定范式")
    ap.add_argument("--scope", choices=["local", "global"], default="local")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--emb", default=os.environ.get("LAB04_EMB", "BAAI/bge-small-zh-v1.5"))
    ap.add_argument("--mrr_check", default=None, help="如 3,1,2：打印 MRR 手算过程")
    ap.add_argument("--detail", action="store_true", help="逐题打印检索与漏检")
    ap.add_argument("--dump", default=None, help="结果落盘 JSON 路径")
    args = ap.parse_args()

    # ---- MRR 手算复核（式 8-10），不需要模型 ----
    if args.mrr_check:
        ranks = [int(x) for x in args.mrr_check.split(",")]
        m = sum(1.0 / r for r in ranks) / len(ranks)
        print("MRR = (" + " + ".join(f"1/{r}" for r in ranks) + f") / {len(ranks)} = {m:.3f}")
        return

    # ---- 三元组解析器离线鲁棒性测试，不需要模型 ----
    if args.mode == "extract_test":
        cases = [
            ("标准 JSON", '[["李文瀚","任职于","人工智能学院"],["李文瀚","主讲","大模型通识课"]]'),
            ("带代码围栏", '```json\n[["人工智能学院","院长","王启明"]]\n```'),
            ("中文标点+尾逗号", '[["计算机学院","院长","陈国峰"]，]'),
            ("前后带解说", '好的，抽取结果如下：\n[["云山大学","校训","格物致知"]]\n以上。'),
            ("单引号/花引号", "[['图书馆','藏书量','380万册']]"),
            ("截断的数组", '[["春季学期","开学日期","3月2日"'),
            ("非数组输出", "抱歉，我无法完成该任务。"),
        ]
        print("===== 三元组解析器鲁棒性测试（模拟 LLM 脏输出）=====")
        ok = 0
        for name, raw in cases:
            got = parse_triples(raw)
            print(f"{name:<16} 解析出 {len(got):>1} 条   {got if got else '（判为失败，计入解析失败率）'}")
            ok += (len(got) > 0)
        print(f"\n7 例中 {ok} 例成功解析、{7 - ok} 例判为失败——这就是 GraphRAG 抽取环节的真实工程成本（教材 8.7）。")
        return

    if args.mode == "extract":
        extract_triples()
        return

    emb = Embedder(args.emb)

    # ---- K 敏感性扫描（任务四⑤）：内置 15 题 ----
    if args.mode == "sweep":
        out = {}
        for k in (1, 2, 3, 5):
            out[k] = evaluate(build_systems(emb, args), QUESTIONS, k)
        print("\n===== 窗口敏感性汇总（内置 15 题，Recall@K）=====")
        print(f"{'范式':<14}{'类型':<10}{'K=1':>8}{'K=2':>8}{'K=3':>8}{'K=5':>8}")
        for name in out[1]["systems"]:
            for tp in ("fact", "multi", "global"):
                row = [out[k]["systems"][name]["by_type"].get(tp, {}).get("recall", 0.0) for k in (1, 2, 3, 5)]
                print(f"{name:<16}{tp:<12}" + "".join(f"{v:>8.3f}" for v in row))
        dump(args.dump, {str(k): v for k, v in out.items()})
        return

    # ---- GraphRAG 的 local / global 两类查询对比（指南任务二）----
    if args.mode == "graph_scope":
        g = GraphRAG(emb)
        systems = {"GraphRAG(local)": ForcedScope(g, "local"),
                   "GraphRAG(global)": ForcedScope(g, "global")}
        res = evaluate(systems, ALL_QUESTIONS, args.k, detail=args.detail)
        dump(args.dump, res)
        return

    # ---- 失败案例归因（含完整排名，区分「排太后」与「没检到」）----
    if args.mode == "failures":
        systems = build_systems(emb, args)
        rows = []
        print("\n===== 失败案例归因（Top-5，20 题）=====")
        for name, s in systems.items():
            print(f"\n--- {name} ---")
            for q in ALL_QUESTIONS:
                docs = s.retrieve(q["q"], 5)
                ev = set(q["evidence"])
                miss = ev - set(docs)
                if not miss:
                    continue
                full = s.retrieve_ranked(q["q"])
                detail = []
                for d in sorted(miss):
                    pos = full.index(d) + 1 if d in full else None
                    detail.append(f"{d}({('排第' + str(pos)) if pos else '未进入候选集'})")
                kind = "检到但排太后" if all(d in full for d in miss) else "根本没检到"
                print(f"  {q['id']} {q['type']:<7} 漏掉 {'、'.join(detail)} → {kind}")
                rows.append({"system": name, "qid": q["id"], "type": q["type"], "query": q["q"],
                             "miss": sorted(miss), "detail": detail, "kind": kind,
                             "retrieved": docs[:5], "candidate_size": len(full)})
        dump(args.dump, rows)
        return

    # ---- 实体归一化前后对比 ----
    if args.mode == "alias_test":
        probe = "云大的校训是什么？"
        g_raw = GraphRAG(emb, normalize=False)
        g_norm = GraphRAG(emb, normalize=True)
        print(f"\n(1) 查询侧别名：探针问题「{probe}」\n")
        for tag, g in (("归一化前", g_raw), ("归一化后", g_norm)):
            ents = g._entities(probe)
            docs = g.retrieve(probe, 5, "local")
            print(f"[{tag}] 命中实体={ents or '无（退化为全局查询）'}  检索结果={docs}")
        print(f"节点数：归一化前 {g_raw.G.number_of_nodes()} → 归一化后 {g_norm.G.number_of_nodes()}"
              f"（预置三元组本身已是规范写法，故此处不变；归一化的作用在查询侧的变体匹配上）")

        # (2) 建图侧归一化：模拟 LLM 抽取出的同义变体（云大 / AI 协会 空格 / 《大模型通识课》书名号）
        noisy = PRE_TRIPLES + [("云大", "校训", "格物致知", "doc01"),
                               ("AI 协会", "指导教师", "李文瀚", "doc11"),
                               ("《大模型通识课》", "学分", "3学分", "doc06")]
        g_noisy_raw = GraphRAG(emb, triples=noisy, normalize=False)
        g_noisy_norm = GraphRAG(emb, triples=noisy, normalize=True)
        print(f"\n(2) 建图侧归一化：模拟抽取产物混入同义变体（云大 / 「AI 协会」/《大模型通识课》）")
        print(f"节点数：未归一化 {g_noisy_raw.G.number_of_nodes()} → 归一化后 "
              f"{g_noisy_norm.G.number_of_nodes()}；边数：{g_noisy_raw.G.number_of_edges()} → "
              f"{g_noisy_norm.G.number_of_edges()}")
        print("未归一化时同义变体各自成为只带 1 条边的孤立节点，该实体的知识被稀释到多个节点上；"
              "归一化后它们合并回主干节点，重复边被合并。")
        probe2 = "云山大学的校训是什么？"
        print(f"探针「{probe2}」检索结果：未归一化 {g_noisy_raw.retrieve(probe2, 5)} → "
              f"归一化后 {g_noisy_norm.retrieve(probe2, 5)}")
        dump(args.dump, {"probe": probe,
                         "raw_entities": g_raw._entities(probe), "norm_entities": g_norm._entities(probe),
                         "raw_docs": g_raw.retrieve(probe, 5), "norm_docs": g_norm.retrieve(probe, 5),
                         "nodes_raw": g_raw.G.number_of_nodes(), "nodes_norm": g_norm.G.number_of_nodes(),
                         "nodes_noisy_raw": g_noisy_raw.G.number_of_nodes(),
                         "nodes_noisy_norm": g_noisy_norm.G.number_of_nodes(),
                         "edges_noisy_raw": g_noisy_raw.G.number_of_edges(),
                         "edges_noisy_norm": g_noisy_norm.G.number_of_edges(),
                         "probe2": probe2, "probe2_raw": g_noisy_raw.retrieve(probe2, 5),
                         "probe2_norm": g_noisy_norm.retrieve(probe2, 5)})
        return

    # ---- 增量更新实验（任务五 b）：真实计时 ----
    if args.mode == "update":
        corpus2 = dict(CORPUS)
        corpus2.update(DOC16)
        print("===== 增量更新实验：新增 doc16（人工智能学院新教师张明远）=====")
        report = {"doc16": DOC16["doc16"]}
        # Vector：重新分块+重新编码
        t0 = time.time()
        v = VectorRAG(emb, corpus=corpus2)
        dt_all = time.time() - t0
        t0 = time.time()
        emb.encode([DOC16["doc16"]])
        dt_one = time.time() - t0
        print(f"\n[Vector RAG] 全库重建（{len(corpus2)} 篇/{len(v.chunks)} 块）耗时 {dt_all:.2f}s；"
              f"只编码新增 1 篇耗时 {dt_one:.2f}s（块级追加即可，无需改动旧块）")
        report["vector"] = {"rebuild_all": dt_all, "encode_new": dt_one, "chunks": len(v.chunks)}
        # Graph：补三元组 + 重建图 + 重算社区
        t0 = time.time()
        g = GraphRAG(emb, triples=PRE_TRIPLES + TRIPLES_DOC16)
        dt_g = time.time() - t0
        print(f"[GraphRAG] 需补 {len(TRIPLES_DOC16)} 条三元组，重建图与社区发现耗时 {dt_g:.2f}s；"
              f"新增实体 {sorted({h for h, _, _, _ in TRIPLES_DOC16} | {t for _, _, t, _ in TRIPLES_DOC16})}；"
              f"受影响的社区摘要需重新生成（本例中 doc16 落在社区 C1）")
        report["graph"] = {"add_triples": len(TRIPLES_DOC16), "rebuild": dt_g}
        # Wiki：条目内追加一句
        entries2 = [(t, b, s) for t, b, s in PRE_ENTRIES]
        entries2[3] = ENTRY_DOC16
        t0 = time.time()
        w = WikiRAG(emb, entries=entries2)
        dt_w = time.time() - t0
        print(f"[WikiRAG] 「人工智能学院教师」条目追加 1 句并重编码该条目，耗时 {dt_w:.2f}s；"
              f"其余 {len(entries2) - 1} 个条目不动（条目独立可增改）")
        report["wiki"] = {"reencode": dt_w, "entries": len(entries2)}
        # 验证新增内容可被检索
        q = "张明远的研究方向是什么？"
        print(f"\n用新问题验证：{q}")
        for name, s in (("Vector RAG", v), ("GraphRAG", g), ("WikiRAG", w)):
            docs = s.retrieve(q, 5)
            pos = docs.index("doc16") + 1 if "doc16" in docs else None
            print(f"  {name:<12} 检索={docs}  doc16 排名={pos or '未命中'}")
            report[name] = {"docs": docs, "doc16_rank": pos}
        dump(args.dump, report)
        return

    # ---- 单题问答（抽取式作答降级方案）----
    if args.mode == "answer":
        systems = build_systems(emb, args)
        name = {"vector": "Vector RAG", "graph": "GraphRAG", "wiki": "WikiRAG", "hybrid": "Hybrid RRF"}[args.sys]
        q = next(x for x in ALL_QUESTIONS if x["id"].upper() == args.q.upper())
        docs = systems[name].retrieve(q["q"], args.k, args.scope)
        ans, cited = extractive_answer(q["q"], docs)
        print(f"\n[{name} / {q['id']}·{q['type']}] {q['q']}")
        print(f"标准证据：{q['evidence']}")
        print(f"检索到的文档：{docs}")
        print(f"上下文：\n" + "\n".join(f"  [{d}] {CORPUS[d]}" for d in docs))
        print(f"\n抽取式回答：{ans}")
        print(f"引用文档：{cited}")
        dump(args.dump, {"system": name, "qid": q["id"], "type": q["type"], "query": q["q"],
                         "evidence": q["evidence"], "retrieved": docs, "answer": ans, "cited": cited})
        return

    # ---- 单题：检索（+ 生成；无 API 时用抽取式作答）----
    if args.mode in ("vector", "graph", "wiki", "hybrid"):
        systems = build_systems(emb, args)
        name = {"vector": "Vector RAG", "graph": "GraphRAG", "wiki": "WikiRAG", "hybrid": "Hybrid RRF"}[args.mode]
        s = systems[name]
        q = next(x for x in ALL_QUESTIONS if x["id"].upper() == args.q.upper())
        docs = s.retrieve(q["q"], args.k, args.scope)
        print(f"\n[{name} / {q['id']}·{q['type']}] {q['q']}")
        print(f"标准证据：{q['evidence']}   命中：{sorted(set(q['evidence']) & set(docs[:args.k]))}")
        if args.mode in ("vector", "hybrid"):
            print("\n块级 Top-5（块号 | 余弦相似度 | 文档）：")
            for doc, sc in s.top_chunks(q["q"], 5):
                print(f"  {sc:.4f}  [{doc}] {next(c for c in s.chunks if c[0] == doc)[1][:40]}…")
        if args.mode == "graph":
            print("\n" + s.explain(q["q"], args.scope))
        if args.mode == "wiki":
            print("\n条目级 Top-3（条目 | 相似度）：")
            for t, sc in s.top_entries(q["q"], 3):
                print(f"  {sc:.4f}  {t}")
        print(f"\n检索到的文档：{docs}")
        ctx = "\n".join(f"[{d}] {CORPUS[d]}" for d in docs)
        try:
            print("\nLLM 回答：", llm_chat(ANSWER_PROMPT.replace("{q}", q["q"]).replace("{ctx}", ctx)))
        except SystemExit:
            ans, cited = extractive_answer(q["q"], docs)
            print("\n（未配置 LLM API，输出检索结果、上下文与抽取式作答）")
            print(ctx)
            print(f"\n抽取式回答：{ans}")
        dump(args.dump, {"system": name, "qid": q["id"], "type": q["type"], "query": q["q"],
                         "evidence": q["evidence"], "retrieved": docs})
        return

    # ---- 画面板 / 图谱 / 条目库截图（报告图 1~3）----
    if args.mode == "figures":
        from make_figures import make_all
        make_all(emb)
        return

    # ---- 默认：三范式 × 20 题检索评测 ----
    systems = build_systems(emb, args)
    res = evaluate(systems, ALL_QUESTIONS, args.k, detail=args.detail)
    print("\n提示：把本表抄进报告表 3；另抽 3 题手算 MRR，用 --mrr_check 复核（式 8-10）。")
    dump(args.dump, res)


if __name__ == "__main__":
    main()
