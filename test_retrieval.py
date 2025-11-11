# test_retrieval.py のようなスクリプトのイメージ

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import ... # vectorize_references.py で使ったのと同じモデル
import os

# --- 設定 ---
CHROMA_PATH = "chroma_references" # 参考文献DBのパス
EMBEDDING_MODEL_NAME = "..." # 使用した埋め込みモデル名
QUERY = "親権停止の要件について教えてください。"
# ------------

print(f"質問: {QUERY}\n")

# 1. 埋め込みモデルをロード
# (HuggingFaceEmbeddingsなど、vectorize_... と同じものを指定)
embedding_function = ... # (例: HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, ...)) 

# 2. 既存のDBをロード
vectordb = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_function
)

print(f"DB '{CHROMA_PATH}' をロードしました。\n")

# 3. 類似検索を実行 (k=5件取得してみる)
try:
    retrieved_docs = vectordb.similarity_search(QUERY, k=5)
    
    print(f"--- 検索結果 (上位 {len(retrieved_docs)} 件) ---")
    
    if not retrieved_docs:
        print("!!! 関連するチャンクが見つかりませんでした。!!!")
    
    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- チャンク {i+1} ---")
        print(f"【メタデータ】: {doc.metadata}")
        print("【内容（先頭150文字）】:")
        print(doc.page_content[:150] + "...") # 長すぎるので先頭だけ表示
        print("-" * 20)

except Exception as e:
    print(f"!!! 検索中にエラーが発生しました: {e} !!!")