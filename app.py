# server/app.py
import numpy as np, pandas as pd, pickle, joblib
from scipy import sparse
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from flask import Flask, render_template, request

# 放在檔案頂部某處，統一管理需要輸出的欄位
NEED_COLS = ["book_id","title","image_url","authors","original_publication_year","language_code"]

# ===== 載入已訓練資產 =====
books = pd.read_csv("artifacts/books.csv")
# 補齊展示用欄位，避免後續 merge/選欄 KeyError 或 NaN
books["image_url"] = books.get("image_url", "").fillna("")
books["authors"] = books.get("authors", "").fillna("")
books["language_code"] = books.get("language_code", "").fillna("")
if "original_publication_year" in books.columns:
    books["original_publication_year"] = books["original_publication_year"].astype("Int64")
# 繼續作業（向量器、矩陣、映射等）
tfidf = joblib.load("artifacts/tfidf_vectorizer.joblib")
tfidf_matrix = sparse.load_npz("artifacts/tfidf_matrix.npz")
with open("artifacts/id_mapping.pkl","rb") as f:
    maps = pickle.load(f)
user_id_to_code = maps["user_id_to_code"]
book_id_to_code = maps["book_id_to_code"]
code_to_book_id = maps["code_to_book_id"]
user_f = np.load("artifacts/als_user_factors.npy")
item_f = np.load("artifacts/als_item_factors.npy")

title_to_idx = pd.Series(books.index, index=books["title"]).drop_duplicates()

def search_by_keyword(q, topn=10):
    q_vec = tfidf.transform([q])
    sims = linear_kernel(q_vec, tfidf_matrix).ravel()
    top_idx = sims.argsort()[::-1][:topn]

    out = books.iloc[top_idx][NEED_COLS].assign(score=sims[top_idx])
    return out

def collaborative_recommend(user_id, topn=20):
    if user_id not in user_id_to_code:
        return pd.DataFrame(columns=["book_id","title","score"])
    u = user_id_to_code[user_id]
    # 計算對所有 item 的相似分數（user_f[u] 與 item_f 做內積）
    scores = item_f @ user_f[u]
    top_idx = np.argpartition(-scores, topn)[:topn]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    book_ids_rec = [code_to_book_id[i] for i in top_idx]
    score_map = {bid: scores[book_id_to_code[bid]] for bid in book_ids_rec}
    
    out = books[books["book_id"].isin(book_ids_rec)][["book_id","title"]].copy()
    out["score"] = out["book_id"].map(score_map)
    return out.sort_values("score", ascending=False)

def content_neighbors(book_title, topn=50):
    if book_title not in title_to_idx:
        return pd.DataFrame(columns=NEED_COLS + ["score"])
    idx = title_to_idx[book_title]
    sims = linear_kernel(tfidf_matrix[idx], tfidf_matrix).ravel()
    order = sims.argsort()[::-1]
    order = order[order != idx][:topn]

    out = books.iloc[order][NEED_COLS].copy()
    out["score"] = sims[order]
    return out

def hybrid_mmr(user_id, book_title, topn=10, alpha=0.5, lambda_div=0.7, disliked=None):
    disliked = set(disliked or [])
    # 內容相似
    cont = content_neighbors(book_title, topn=200)
    # 協同分數
    collab = collaborative_recommend(user_id, topn=200)
    collab_map = dict(zip(collab["book_id"], collab["score"]))

    # 合併並融合分數
    cand = []
    for _, row in cont.iterrows():
        bid = row["book_id"]
        if bid in disliked:
            continue
        c_score = collab_map.get(bid, 0.0)
        final = alpha * c_score + (1 - alpha) * row["score"]
        cand.append((bid, final))
    if not cand:
        return pd.DataFrame(columns=["book_id","title","score"])

    cand_df = pd.DataFrame(cand, columns=["book_id","score"]).merge(
        books[NEED_COLS], on="book_id", how="left"
    )

    # 為 MMR 預先算候選內兩兩相似（以書名向量近似）
    tfidf_mat = tfidf.transform(cand_df["title"].fillna(""))
    sim_mat = cosine_similarity(tfidf_mat)

    selected, remaining = [], set(range(len(cand_df)))
    while len(selected) < topn and remaining:
        if not selected:
            # 先選最相關
            best = max(remaining, key=lambda i: cand_df.iloc[i]["score"])
            selected.append(best); remaining.remove(best); 
            continue
        # 計算每個候選的 MMR 分數
        def mmr(i):
            rel = cand_df.iloc[i]["score"]
            # 與已選集合的最大相似度
            div = max(sim_mat[i, j] for j in selected) if selected else 0.0
            return lambda_div * rel - (1 - lambda_div) * div
        best = max(remaining, key=mmr)
        selected.append(best); remaining.remove(best)
    return cand_df.iloc[selected][["image_url", "title", "book_id", "authors", "original_publication_year", "language_code","score"]].reset_index(drop=True)

app = Flask(__name__)

"""@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")"""

@app.route("/", methods=["GET"])
def index():
    # 提供下拉用的種子書清單，可取前 500 熱門或按字母排序 #.head(500)
    seed_titles = books["title"].dropna().drop_duplicates().sort_values().tolist()
    # 可選：提供一些存在於矩陣中的使用者做下拉
    user_choices = list(user_id_to_code.keys())[:200] if user_id_to_code else []
    return render_template("index.html", seed_titles=seed_titles, user_choices=user_choices)

@app.route("/ui_recommend", methods=["POST"])
def ui_recommend():
    method = request.form.get("method", "hybrid")
    seed_title = request.form.get("seed_title", "").strip()
    topn = int(request.form.get("topn", 10))
    alpha = float(request.form.get("alpha", 0.6))
    lambda_div = float(request.form.get("lambda_div", 0.7))
    user_id = request.form.get("user_id", "").strip()

    # 若未提供 user_id，視方法決定退化策略
    uid_cast = None
    if user_id:
        try: uid_cast = int(user_id)
        except: uid_cast = user_id

    if method == "content":
        rows = content_neighbors(seed_title, topn=topn)
        mode = "content"
        q = f"seed={seed_title}"
    elif method == "collab":
        if uid_cast is None:
            # 無 user 時，提供提示或做簡單回退：以 seed 的內容相似做替代
            rows = content_neighbors(seed_title, topn=topn)
            mode = "collab(fallback:content)"
        else:
            rows = collaborative_recommend(uid_cast, topn=topn)
            mode = "collab"
        q = f"user={user_id or 'auto'}, seed={seed_title}"
    else:  # hybrid
        if uid_cast is None:
            # 沒 user 時讓 alpha=0，相當於純內容；或你也可固定 alpha=0.2
            rows = hybrid_mmr(user_id="", book_title=seed_title, topn=topn, alpha=0.0, lambda_div=lambda_div, disliked=[])
        else:
            rows = hybrid_mmr(uid_cast, seed_title, topn=topn, alpha=alpha, lambda_div=lambda_div, disliked=[])
        mode = "hybrid"
        q = f"user={user_id or 'auto'}, seed={seed_title}"

    return render_template("results.html", mode=mode, q=q, rows=rows.to_dict(orient="records"))

@app.route("/search", methods=["POST"])
def search():
    q = request.form.get("q","").strip()
    results = search_by_keyword(q, topn=20) if q else pd.DataFrame(columns=["image_url", "title", "book_id", "authors", "original_publication_year", "language_code", "score"])
    return render_template("results.html", mode="search", q=q, rows=results.to_dict(orient="records"))
#"book_id","title","score"

@app.route("/recommend_collab", methods=["POST"])
def recommend_collab():
    user_id = request.form.get("user_id","").strip()
    topn = int(request.form.get("topn", 10))
    try:
        uid_cast = int(user_id)
    except:
        uid_cast = user_id
    rows = collaborative_recommend(uid_cast, topn=topn)
    return render_template(
        "results.html",
        mode="collab",
        q=f"user={user_id}",
        rows=rows.to_dict(orient="records")
    )

@app.route("/recommend", methods=["POST"])
def recommend():
    user_id = request.form.get("user_id","").strip()
    book_title = request.form.get("book_title","").strip()
    topn = int(request.form.get("topn",10))
    alpha = float(request.form.get("alpha",0.5))
    lambda_div = float(request.form.get("lambda_div",0.7))
    disliked = request.form.get("disliked","").strip()
    disliked_list = [int(x) for x in disliked.split(",") if x.strip().isdigit()] if disliked else []
    try: uid_cast = int(user_id)
    except: uid_cast = user_id
    rows = hybrid_mmr(uid_cast, book_title, topn=topn, alpha=alpha, lambda_div=lambda_div, disliked=disliked_list)
    return render_template("results.html", mode="recommend", q=f"user={user_id}, seed={book_title}",
                           rows=rows.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7001, debug=True)
