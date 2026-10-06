# -*- coding: utf-8 -*-
"""报告图 1~3 的生成脚本（被 hw4_rag.py --mode figures 调用）。

图 1  Vector RAG 单题检索面板（块级相似度 + 文档命中情况）
图 2  云山大学知识图谱（networkx spring 布局，实体=节点，三元组=有向边）
图 3  WikiRAG 条目化知识库（9 条条目的标题 / 聚合正文 / 来源文档）
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from hw4_rag import (CORPUS, PRE_ENTRIES, PRE_TRIPLES, QUESTIONS, ALL_QUESTIONS,
                     VectorRAG, WikiRAG, canon_ent)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def _panel(lines, path, title, figsize=(9.2, 6.4), fontsize=10.5):
    """把若干行文本渲染成图片（模拟终端面板截图）。行距按行数自适应，避免溢出画布。"""
    step = 0.88 / max(len(lines) + 1, 1)
    h = min(figsize[1], max(2.2, 0.34 * (len(lines) + 2)))
    fig = plt.figure(figsize=(figsize[0], h))
    fig.patch.set_facecolor("#1e1e1e")
    fig.text(0.02, 0.975, title, color="#4ec9b0", fontsize=fontsize + 1.5,
             family="SimHei", va="top")
    y = 0.93
    for ln in lines:
        color = "#d4d4d4"
        if ln.startswith("★"):
            color = "#ce9178"
        elif ln.startswith("—"):
            color = "#569cd6"
        fig.text(0.02, y, ln, color=color, fontsize=fontsize, family="SimHei", va="top")
        y -= step
    plt.axis("off")
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[fig] {path}（{len(lines)} 行）")


def fig1_vector_panel(emb):
    v = VectorRAG(emb)
    q = next(x for x in ALL_QUESTIONS if x["id"] == "Q01")
    lines = [f"$ python hw4_rag.py --mode vector --q Q01", ""]
    lines.append(f"[{q['id']}·{q['type']}] {q['q']}")
    lines.append(f"标准证据：{q['evidence']}")
    lines.append("")
    lines.append("块级 Top-5（余弦相似度 | 文档 | 块首 40 字）：")
    for doc, sc in v.top_chunks(q["q"], 5):
        chunk = next(c[1] for c in v.chunks if c[0] == doc)
        tag = "★命中证据" if doc in q["evidence"] else ""
        lines.append(f"  {sc:.4f}  [{doc}] {chunk[:38]}… {tag}")
    docs = v.retrieve(q["q"], 5)
    lines.append("")
    lines.append(f"检索到的文档：{docs}")
    lines.append(f"命中证据：{sorted(set(q['evidence']) & set(docs))}"
                 f"    漏检：{sorted(set(q['evidence']) - set(docs))}")
    lines.append("")
    lines.append("— 现象：桥接文档 doc02（学院院长）进入 Top-5 但排在末位，")
    lines.append("— 3 跳链上的 doc06/doc04/doc02 分别排在第 1、2、4 位。")
    _panel(lines, "fig1_vector_retrieval.png", "图 1  Vector RAG 单题检索面板（Q01 多跳题）")


def fig2_kg(emb):
    G = nx.DiGraph()
    for h, r, t, src in PRE_TRIPLES:
        G.add_edge(canon_ent(h), canon_ent(t), rel=r, src=src)
    pos = nx.spring_layout(G, k=0.55, seed=2024, iterations=200)
    deg = dict(G.degree)
    fig, ax = plt.subplots(figsize=(12.5, 8.6))
    widths = [0.8 + 0.5 * deg[u] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#9aa4b2", width=widths,
                           arrows=True, arrowsize=9, alpha=0.85)
    sizes = [420 + 260 * deg[n] for n in G.nodes()]
    colors = ["#4c8bf5" if deg[n] >= 3 else "#9ecbff" for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=colors,
                           edgecolors="white", linewidths=1.2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8.5, font_family="SimHei")
    ax.set_title(f"图 2  云山大学知识图谱：{G.number_of_nodes()} 个实体 / {G.number_of_edges()} 条三元组"
                 f"（节点大小=度数，箭头=关系方向）", fontsize=12)
    sample = ["李文瀚 —任职于→ 人工智能学院", "人工智能学院 —院长→ 王启明",
              "认知计算全国重点实验室 —依托→ 人工智能学院", "知识图谱导论 —先修课程→ 数据结构"]
    ax.text(0.005, 0.02, "三元组样例：\n" + "\n".join("  " + s for s in sample),
            transform=ax.transAxes, fontsize=8.5, family="SimHei", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f6f8fa", ec="#c8d0d8"))
    ax.axis("off")
    fig.tight_layout()
    fig.savefig("fig2_knowledge_graph.png", dpi=175, bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig2_knowledge_graph.png")


def fig3_entries():
    lines = ["$ python -c \"from hw4_rag import PRE_ENTRIES; ...\"", ""]
    lines.append(f"条目库共 {len(PRE_ENTRIES)} 条（标题 | 聚合正文 | 来源文档）：")
    lines.append("")
    for i, (title, body, srcs) in enumerate(PRE_ENTRIES, 1):
        lines.append(f"[{i}] 《{title}》  长度 {len(body)} 字  来源 {srcs}")
        lines.append(f"    {body[:56]}…")
        lines.append("")
    lines.append("— 条目是「重写聚合」：如第 4 条把 doc04 的三位教师与 doc11 的两个社团")
    lines.append("— 合并成一段，原文里的「；」变成了条目内的并列句。")
    _panel(lines, "fig3_wiki_entries.png", "图 3  WikiRAG 条目化知识库（裁剪示意）",
           figsize=(10.4, 8.2), fontsize=10)


def make_all(emb):
    fig1_vector_panel(emb)
    fig2_kg(emb)
    fig3_entries()


if __name__ == "__main__":
    from hw4_rag import Embedder
    import os
    make_all(Embedder(os.environ.get("LAB04_EMB", "BAAI/bge-small-zh-v1.5")))
