# server/app.py
import numpy as np, pandas as pd, pickle, joblib
from scipy import sparse
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from flask import Flask, render_template, request, redirect, url_for, session
import random

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
with open("artifacts/id_mapping.pkl", "rb") as f:
    maps = pickle.load(f)
user_id_to_code = maps["user_id_to_code"]
book_id_to_code = maps["book_id_to_code"]
code_to_book_id = maps["code_to_book_id"]
user_f = np.load("artifacts/als_user_factors.npy")
item_f = np.load("artifacts/als_item_factors.npy")

# Defensive check: sometimes training was done on transposed matrix
# (e.g. fitting ALS on item-user instead of user-item) which causes
# `als.user_factors` and `als.item_factors` to be saved in the
# opposite order relative to our `user_id_to_code` / `book_id_to_code` maps.
# If the loaded shapes don't match the mapping sizes, swap them.
if user_f.shape[0] != len(user_id_to_code) or item_f.shape[0] != len(book_id_to_code):
    # If they are swapped, swap back
    if user_f.shape[0] == len(book_id_to_code) and item_f.shape[0] == len(user_id_to_code):
        print("Warning: detected swapped ALS factor arrays — correcting by swapping user/item factors")
        user_f, item_f = item_f, user_f
    else:
        # Shapes are unexpected; print sizes to help debugging but continue (may still error later)
        print(f"Warning: unexpected ALS factor shapes: user_f={user_f.shape}, item_f={item_f.shape}")
        print(f"Mappings sizes: users={len(user_id_to_code)}, books={len(book_id_to_code)}")

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
    
    out = books[books["book_id"].isin(book_ids_rec)][NEED_COLS].copy()#[]"book_id","title"]
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
            continue # 排除dislike
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
app.secret_key = "your-strong-secret-key"   # 用真正的隨機值

@app.route("/", methods=["GET"])
def index():
    # 提供下拉用的種子書清單
    seed_titles = books["title"].dropna().drop_duplicates().sort_values().tolist()
    # 提供一些存在於矩陣中的使用者做下拉
    user_choices = list(user_id_to_code.keys()) if user_id_to_code else []
    if user_choices:
        user_min = min(user_choices)
        user_max = max(user_choices)
        user_default = user_min  # 或其他你想預設的 id
    else:
        user_min = 0
        user_max = 0
        user_default = 0

    # 從 session 拿出已被 dislike 的 book_id
    disliked_ids = session.get("disliked_ids", [])
    if disliked_ids:
        disliked_books = books[books["book_id"].isin(disliked_ids)][["book_id", "title"]]
        disliked_books = disliked_books.to_dict(orient="records")
    else:
        disliked_books = []
    last_seed_title = session.get("last_seed_title", "")
    return render_template(
        "index.html",
        seed_titles=seed_titles,
        user_choices=user_choices,
        disliked_books=disliked_books,
        last_seed_title=last_seed_title,
        user_min=user_min,
        user_max=user_max,
        user_default=user_default,
    )

@app.route("/ui_recommend", methods=["POST"])
def ui_recommend():
    #method = request.form.get("method", "hybrid")
    method = request.form.get("filter_type", "hybrid")
    seed_title = request.form.get("seed_title", "").strip()
    session["last_seed_title"] = seed_title  # 記錄上次輸入的書名
    topn = int(request.form.get("topn", 10))
    alpha = float(request.form.get("alpha", 0.6))
    lambda_div = float(request.form.get("lambda_div", 0.7))
    user_id_raw = request.form.get("user_id", "").strip()

    disliked_ids = session.get("disliked_ids", []) # 讀取所有被 dislike 的書，作為 MMR 排除名單

    uid_cast = None
    auto_chosen = False   # 用來標記這次是不是自動抽 user

    if user_id_raw and user_id_raw != "auto":
        # 正常情況：使用者直接選某個 user id
        try:
            uid_cast = int(user_id_raw)
        except:
            uid_cast = user_id_raw
    elif user_id_raw == "auto":
        # Auto：從 user list 隨機抽一個 id
        if user_id_to_code:
            uid_cast = random.choice(list(user_id_to_code.keys()))
            auto_chosen = True
            user_id_raw = str(uid_cast)  # 之後顯示在 q 裡
        else:
            uid_cast = None  # 沒有 user 可用時，後面會 fallback # user_id_raw 為空字串時維持 uid_cast = None（表示不用 CF）

    if method == "content":
        rows = content_neighbors(seed_title, topn=topn)
        mode = "content"
        q = {"user": "N/A", "seed": seed_title}
    elif method == "collaborative":
        rows = collaborative_recommend(uid_cast, topn=topn)
        mode = "collaborative(auto)" if auto_chosen else "collaborative"
        q = {"user": user_id_raw or "auto", "seed": seed_title}
    else:  # hybrid
        rows = hybrid_mmr(uid_cast, seed_title, topn=topn, alpha=alpha, lambda_div=lambda_div, disliked=disliked_ids,)
        mode = "hybrid(auto)" if auto_chosen else "hybrid"
        q = {"user": user_id_raw or "auto", "seed": seed_title}
    return render_template("results.html", mode=mode, q=q, rows=rows.to_dict(orient="records"))

@app.route("/dislike", methods=["POST"])
def dislike():
    book_id = request.form.get("book_id", "").strip()

    if not book_id:
        return redirect(url_for("index"))
    
    try:
        book_id_int = int(book_id)
    except:
        return redirect(url_for("index"))
    
    disliked_ids = session.get("disliked_ids", [])
    if book_id_int not in disliked_ids:
        disliked_ids.append(book_id_int)
    session["disliked_ids"] = disliked_ids
    return redirect(url_for("index"))

@app.route("/remove_disliked", methods=["POST"])
def remove_disliked():
    book_id = request.form.get("book_id", "").strip()

    try:
        book_id_int = int(book_id)
    except:
        return redirect(url_for("index"))

    disliked_ids = session.get("disliked_ids", [])
    disliked_ids = [bid for bid in disliked_ids if bid != book_id_int]
    session["disliked_ids"] = disliked_ids

    return redirect(url_for("index"))

@app.route("/clear_disliked", methods=["POST"])
def clear_disliked():
    session["disliked_ids"] = [] # 清空整個 dislike 清單
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7001, debug=True)
