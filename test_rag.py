from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# -----------------------------------------------------------------
# ▼ ユーザー設定 (ご自身の環境に合わせて変更してください)
# -----------------------------------------------------------------

# 1. ベクトルDB(Chroma)の保存先ディレクトリ
# (vectorize_references.py で指定したものと同じ)
# 14行目を修正
CHROMA_PATH = r"C:\Projects\PR\DB"

# 2. 埋め込みモデルの指定
# (vectorize_references.py で指定したものと同じ)
EMBED_MODEL_NAME = "all-mpnet-base-v2"

# 3. LM Studio の API エンドポイント
# (通常は "http://localhost:1234/v1" です)
LOCAL_LLM_URL = "http://localhost:1234/v1"

# 4. LLMに試したい質問
# (参考文献の内容に基づいて質問してみてください)
TEST_QUESTION = "親権停止の要件について教えてください。"
# TEST_QUESTION = "家事審判法とは何ですか？"

# -----------------------------------------------------------------

def main():
    print(f"質問: {TEST_QUESTION}\n")

    # 1. 埋め込みモデルのロード
    print("STEP 1: 埋め込みモデルをロード中...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    # 2. 既存のChromaDBを読み込む
    print("STEP 2: 既存のベクトルDBをロード中...")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    # DBをリトリーバー(検索機)として設定
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # 上位3件を検索

    # 3. LM Studio (ローカルLLM) への接続設定
    print("STEP 3: ローカルLLM (LM Studio) への接続準備中...")
    llm = ChatOpenAI(
        base_url=LOCAL_LLM_URL,
        api_key="lm-studio", # LM Studioの場合、api_keyは何でも良い
        model="local-model", # モデル名も何でも良い
        temperature=0 # 創造性より正確性を優先
    )

    # 4. RAGプロンプトの定義
    # LLMに対し、「以下のコンテキスト(文脈)に基づいて回答してください」と指示
    prompt_template = """
    以下の「コンテキスト」情報のみに基づいて、質問に回答してください。
    コンテキストに答えがない場合は、「コンテキストからでは分かりません」と回答してください。

    【コンテキスト】
    {context}

    【質問】
    {input}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # 5. RAGチェーンの構築 (LCEL)
    # (検索 -> コンテキストを整形 -> LLMに渡す) という一連の流れを定義
    
    # 検索したチャンク(documents)を {context} にまとめるチェーン
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # ユーザーの質問(input)をまずリトリーバーに渡し、
    # その結果(context)と元の質問(input)を document_chain に渡すチェーン
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # 6. RAGチェーンの実行
    print("STEP 4: RAGを実行中... (LLMが回答を生成しています)")
    response = retrieval_chain.invoke({"input": TEST_QUESTION})

    print("\n-------------------------------------------------")
    print("✅ LLMによる回答:")
    print("-------------------------------------------------")
    print(response.get("answer", "エラー: 回答がありませんでした。"))
    print("\n-------------------------------------------------")

    # (参考) どのチャンクが参照されたかを表示
    print("（参考: 回答の根拠として使用されたチャンク）")
    for i, doc in enumerate(response.get("context", [])):
        print(f"\n--- [チャンク {i+1} (出典: {doc.metadata.get('source', '不明')})]---")
        print(doc.page_content[:200] + "...") # 冒頭200文字だけ表示
    print("-------------------------------------------------")

if __name__ == "__main__":
    main()