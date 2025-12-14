# Final-project
import re
import math
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path("/Users/aayushsapkota/Downloads")

DATA_PATH  = BASE / "CISI.ALL"   # documents
QRY_PATH   = BASE / "CISI.QRY"   # queries
REL_PATH   = BASE / "CISI.REL"   # relevance judgments


# -----------------------------
# Parsing CISI files
# -----------------------------
def parse_cisi_all(path):
    """
    Parses CISI.ALL (documents).
    Returns:
      docs: dict[int, str] doc_id -> concatenated text (T + A + W)
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().replace("\r\n", "\n")

    docs = {}
    current_id = None
    current_field = None
    buf = {"T": [], "A": [], "W": []}

    for line in text.split("\n"):
        if line.startswith(".I "):
            if current_id is not None:
                docs[current_id] = " ".join(buf["T"] + buf["A"] + buf["W"]).strip()
            current_id = int(line.split()[1])
            current_field = None
            buf = {"T": [], "A": [], "W": []}
        elif line.startswith(".T"):
            current_field = "T"
        elif line.startswith(".A"):
            current_field = "A"
        elif line.startswith(".W"):
            current_field = "W"
        elif line.startswith("."):
            # ignore other sections .B .X etc
            current_field = None
        else:
            if current_field in buf:
                buf[current_field].append(line.strip())

    if current_id is not None:
        docs[current_id] = " ".join(buf["T"] + buf["A"] + buf["W"]).strip()

    return docs


def parse_cisi_qry(path):
    """
    Parses CISI.QRY (queries).
    Returns:
      queries: dict[int, str] qid -> query text from .W
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().replace("\r\n", "\n")

    queries = {}
    current_id = None
    in_w = False
    buf = []

    for line in text.split("\n"):
        if line.startswith(".I "):
            if current_id is not None:
                queries[current_id] = " ".join(buf).strip()
            current_id = int(line.split()[1])
            in_w = False
            buf = []
        elif line.startswith(".W"):
            in_w = True
        elif line.startswith("."):
            in_w = False
        else:
            if in_w:
                buf.append(line.strip())

    if current_id is not None:
        queries[current_id] = " ".join(buf).strip()

    return queries


def parse_cisi_rel(path):
    """
    Parses CISI.REL (qrels).
    Typical lines: qid docid 0 0.000000
    Returns:
      qrels: dict[int, set[int]] qid -> relevant docids
    """
    qrels = defaultdict(set)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            qid = int(parts[0])
            docid = int(parts[1])
            qrels[qid].add(docid)
    return dict(qrels)

# -----------------------------
# Text processing
# -----------------------------
TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")

def tokenize(text, stopwords=None):
    tokens = [t.lower() for t in TOKEN_RE.findall(text)]
    if stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    # Optional: remove very short tokens
    tokens = [t for t in tokens if len(t) > 1]
    return tokens

def load_stopwords():
    # Minimal list; replace with a larger list if you want.
    return {
        "a","an","the","and","or","of","to","in","on","for","with","by","is","are",
        "was","were","be","been","as","at","from","that","this","it","its","into",
        "their","they","them","can","may","might","will","would","should","could"
    }

# -----------------------------
# Indexing
# -----------------------------
def build_inverted_index(docs_tokens):
    """
    docs_tokens: dict[docid, list[str]]
    Returns:
      inv: dict[term, dict[docid, tf]]
      doc_len: dict[docid, int]
      df: dict[term, int]
      N: int
      avgdl: float
    """
    inv = defaultdict(dict)
    doc_len = {}
    for docid, toks in docs_tokens.items():
        tf = Counter(toks)
        doc_len[docid] = len(toks)
        for term, freq in tf.items():
            inv[term][docid] = freq

    df = {term: len(postings) for term, postings in inv.items()}
    N = len(docs_tokens)
    avgdl = sum(doc_len.values()) / N if N else 0.0
    return dict(inv), doc_len, df, N, avgdl

# -----------------------------
# Retrieval models
# -----------------------------
def bm25_rank(query_tokens, inv, doc_len, df, N, avgdl, k1=1.5, b=0.75, topk=1000):
    scores = defaultdict(float)
    for term in query_tokens:
        if term not in inv:
            continue
        n = df.get(term, 0)
        # BM25 idf variant (common)
        idf = math.log(1 + (N - n + 0.5) / (n + 0.5)) if n > 0 else 0.0
        postings = inv[term]
        for docid, f in postings.items():
            dl = doc_len[docid]
            denom = f + k1 * (1 - b + b * (dl / avgdl))
            score = idf * (f * (k1 + 1)) / denom
            scores[docid] += score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:topk]

def tfidf_cosine_rank(query_tokens, inv, doc_len, df, N, topk=1000):
    # Build query tf-idf vector
    qtf = Counter(query_tokens)
    qvec = {}
    for t, f in qtf.items():
        if t in df:
            idf = math.log((N + 1) / (df[t] + 1)) + 1.0
            qvec[t] = (1 + math.log(f)) * idf

    qnorm = math.sqrt(sum(v*v for v in qvec.values()))
    if qnorm == 0:
        return []

    # Accumulate dot-products
    dot = defaultdict(float)
    dnorm = defaultdict(float)

    for t, qv in qvec.items():
        postings = inv.get(t, {})
        idf = math.log((N + 1) / (df[t] + 1)) + 1.0
        for docid, tf in postings.items():
            dv = (1 + math.log(tf)) * idf
            dot[docid] += qv * dv
            dnorm[docid] += dv * dv

    scores = {}
    for docid, dp in dot.items():
        denom = qnorm * math.sqrt(dnorm[docid])
        if denom > 0:
            scores[docid] = dp / denom

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:topk]

# -----------------------------
# Metrics
# -----------------------------
def precision_at_k(ranked_docids, relevant_set, k):
    if k <= 0:
        return 0.0
    topk = ranked_docids[:k]
    if not topk:
        return 0.0
    rel = sum(1 for d in topk if d in relevant_set)
    return rel / k

def recall_at_k(ranked_docids, relevant_set, k):
    if not relevant_set:
        return 0.0
    topk = ranked_docids[:k]
    rel = sum(1 for d in topk if d in relevant_set)
    return rel / len(relevant_set)

def average_precision(ranked_docids, relevant_set):
    if not relevant_set:
        return 0.0
    hit = 0
    s = 0.0
    for i, docid in enumerate(ranked_docids, start=1):
        if docid in relevant_set:
            hit += 1
            s += hit / i
    return s / len(relevant_set)

def ndcg_at_k(ranked_docids, relevant_set, k):
    # binary relevance
    def dcg(docs):
        s = 0.0
        for i, d in enumerate(docs, start=1):
            rel = 1.0 if d in relevant_set else 0.0
            s += (2**rel - 1) / math.log2(i + 1)
        return s

    topk = ranked_docids[:k]
    dcg_k = dcg(topk)
    # ideal: all relevant first
    ideal = [1] * min(len(relevant_set), k)
    idcg = sum((2**1 - 1) / math.log2(i + 1) for i in range(1, len(ideal) + 1))
    return (dcg_k / idcg) if idcg > 0 else 0.0

def evaluate_run(run, qrels, ks=(5, 10, 20)):
    """
    run: dict[qid, list[docid]]  ranked docs per query
    qrels: dict[qid, set[docid]]
    Returns: dict of averaged metrics
    """
    qids = sorted(set(run.keys()) & set(qrels.keys()))
    if not qids:
        return {}

    metrics = defaultdict(float)
    for qid in qids:
        ranked = run[qid]
        relset = qrels[qid]

        for k in ks:
            metrics[f"P@{k}"] += precision_at_k(ranked, relset, k)
            metrics[f"R@{k}"] += recall_at_k(ranked, relset, k)
            metrics[f"nDCG@{k}"] += ndcg_at_k(ranked, relset, k)

        metrics["AP"] += average_precision(ranked, relset)

    n = len(qids)
    metrics["MAP"] = metrics["AP"] / n
    del metrics["AP"]

    for key in list(metrics.keys()):
        metrics[key] /= n

    metrics["num_queries"] = n
    return dict(metrics)

# -----------------------------
# Main
# -----------------------------
def main(cisi_all_path, cisi_qry_path, cisi_rel_path, topk=1000):
    stopwords = load_stopwords()

    docs = parse_cisi_all(cisi_all_path)
    queries = parse_cisi_qry(cisi_qry_path)
    qrels = parse_cisi_rel(cisi_rel_path)

    docs_tokens = {docid: tokenize(text, stopwords) for docid, text in docs.items()}
    inv, doc_len, df, N, avgdl = build_inverted_index(docs_tokens)

    bm25_run = {}
    tfidf_run = {}

    for qid, qtext in queries.items():
        qtoks = tokenize(qtext, stopwords)

        bm25_ranked = bm25_rank(qtoks, inv, doc_len, df, N, avgdl, topk=topk)
        tfidf_ranked = tfidf_cosine_rank(qtoks, inv, doc_len, df, N, topk=topk)

        bm25_run[qid] = [d for d, s in bm25_ranked]
        tfidf_run[qid] = [d for d, s in tfidf_ranked]

    ks = (5, 10, 20)
    bm25_metrics = evaluate_run(bm25_run, qrels, ks=ks)
    tfidf_metrics = evaluate_run(tfidf_run, qrels, ks=ks)

    print("BM25 metrics:", bm25_metrics)
    print("TF-IDF cosine metrics:", tfidf_metrics)
   

if __name__ == "__main__":
    main(DATA_PATH, QRY_PATH, REL_PATH, topk=1000)




