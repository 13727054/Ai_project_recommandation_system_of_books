# train.py
import pandas as pd, numpy as np, scipy.sparse as sp, pickle, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import implicit

# 1) 載資料
books = pd.read_csv("books.csv")  # 需含: book_id, title 等
ratings = pd.read_csv("ratings.csv")  # 需含: user_id, book_id, rating

# 2) TF-IDF 訓練與矩陣
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(books["title"].fillna(""))

# 3) ALS 訓練（implicit）
user_ids = ratings["user_id"].astype("category")
book_ids = ratings["book_id"].astype("category")
user_index = user_ids.cat.codes.values
item_index = book_ids.cat.codes.values
data = ratings["rating"].astype(float).values
user_item = sp.coo_matrix((data, (user_index, item_index)),
                          shape=(len(user_ids.cat.categories), len(book_ids.cat.categories))).tocsr()
item_user = user_item.T.tocsr()
als = implicit.als.AlternatingLeastSquares(factors=50, regularization=0.1, iterations=20)
als.fit(item_user)

# 4) 構建映射
user_id_to_code = dict(zip(user_ids, range(len(user_ids.cat.categories))))
book_id_to_code = dict(zip(book_ids, range(len(book_ids.cat.categories))))
code_to_book_id = {v: k for k, v in book_id_to_code.items()}

# 5) 導出資產"book_id","title"
books[["image_url", "title", "book_id", "authors", "original_publication_year", "language_code"]].to_csv("artifacts/books.csv", index=False)
joblib.dump(tfidf, "artifacts/tfidf_vectorizer.joblib")
sp.save_npz("artifacts/tfidf_matrix.npz", tfidf_matrix)
np.save("artifacts/als_user_factors.npy", als.user_factors)
np.save("artifacts/als_item_factors.npy", als.item_factors)
with open("artifacts/id_mapping.pkl","wb") as f:
    pickle.dump({
        "user_id_to_code": user_id_to_code,
        "book_id_to_code": book_id_to_code,
        "code_to_book_id": code_to_book_id
    }, f)
