# train.py
import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle
import joblib
import implicit
from sklearn.feature_extraction.text import TfidfVectorizer

# 1) 載入資料
print("Loading data...")
books = pd.read_csv("books.csv")  # 這是您的 10k 書籍清單
ratings = pd.read_csv("ratings.csv")

# === 關鍵修正：只保留存在於 books.csv 中的評分 ===
print(f"Original ratings count: {len(ratings)}")
valid_book_ids = set(books["book_id"].unique())
ratings = ratings[ratings["book_id"].isin(valid_book_ids)]
print(f"Filtered ratings count: {len(ratings)}")
# ==============================================

# 2) TF-IDF 訓練與矩陣 (內容過濾用)
print("Training TF-IDF...")
tfidf = TfidfVectorizer(stop_words="english")
# 確保 books 照順序處理，並處理 NaN
tfidf_matrix = tfidf.fit_transform(books["title"].fillna(""))

# 3) ALS 訓練 (協同過濾用)
print("Preparing ALS data...")
# 將 user_id 和 book_id 轉為類別型態 (Category) 以取得整數索引
user_ids = ratings["user_id"].astype("category")
book_ids = ratings["book_id"].astype("category")

user_index = user_ids.cat.codes.values
item_index = book_ids.cat.codes.values
data = ratings["rating"].astype(float).values

# 建立稀疏矩陣
# shape = (使用者數量, 書籍數量)
user_item = sp.coo_matrix((data, (user_index, item_index)),
                          shape=(len(user_ids.cat.categories), len(book_ids.cat.categories))).tocsr()

# ALS 需要 Item-User 矩陣
item_user = user_item.T.tocsr()

print("Training ALS model...")
als = implicit.als.AlternatingLeastSquares(factors=50, regularization=0.1, iterations=20)
# Fit on the user-item matrix so that `als.user_factors` corresponds to users
# (rows = users) and `als.item_factors` corresponds to items/books.
als.fit(user_item)

# 4) 構建映射 (Mapping)
# 這些字典將幫助我們在「原始 ID」和「矩陣索引」之間轉換，確保索引 (0, 1, 2...) 對應的是排序後的唯一 ID
print("Building mappings...")
user_id_to_code = dict(zip(user_ids.cat.categories, range(len(user_ids.cat.categories))))
book_id_to_code = dict(zip(book_ids.cat.categories, range(len(book_ids.cat.categories))))
code_to_book_id = {v: k for k, v in book_id_to_code.items()}

# 5) 導出資產 (Artifacts)
print("Saving artifacts...")
# 匯出整理後的 books (確保欄位一致)
books[["image_url", "title", "book_id", "authors", "original_publication_year", "language_code"]].to_csv("artifacts/books.csv", index=False)

joblib.dump(tfidf, "artifacts/tfidf_vectorizer.joblib")
sp.save_npz("artifacts/tfidf_matrix.npz", tfidf_matrix)
np.save("artifacts/als_user_factors.npy", als.user_factors)
np.save("artifacts/als_item_factors.npy", als.item_factors)

print(f"Saved ALS factors: user_factors.shape={als.user_factors.shape}, item_factors.shape={als.item_factors.shape}")
print(f"Mappings sizes: users={len(user_id_to_code)}, books={len(book_id_to_code)}")

with open("artifacts/id_mapping.pkl", "wb") as f:
    pickle.dump({
        "user_id_to_code": user_id_to_code,
        "book_id_to_code": book_id_to_code,
        "code_to_book_id": code_to_book_id
    }, f)

print("Done! Please restart your Flask app.")

