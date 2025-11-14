# --- 修正版 app.py (OpenAI互換APIを使用) ---
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# ▼▼▼ 修正点 1: langchain-openai から ChatOpenAI をインポート ▼▼▼
from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

# --- 定数設定 ---
CHROMA_PATH = "./DB"
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"

# ▼▼▼ 修正点 2: LM Studio の「OpenAI互換」エンドポイントを指定 ▼▼▼
# デフォルトは http://localhost:1234/v1 です
llm_url = st.secrets["LLM_BASE_URL"]
LLM_MODEL_NAME = "local-model" # OpenAI APIでは "api_key" が必要だが、ローカルLLMなので "dummy" で良い

# --- 1. モデルとDBのロード (初回のみ実行) ---
@st.cache_resource
def get_models_and_retriever():
    print("AIモデルとDBをロード中...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    
    # ▼▼▼ 修正点 3: Ollama クラスの代わりに ChatOpenAI クラスを使用 ▼▼▼
    llm = ChatOpenAI(
        base_url=llm_url,
        api_key="dummy-key", # 必須だが値は何でもよい
        model=LLM_MODEL_NAME # LM Studio側でロードしているモデル
    )
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    
    # 既存のChromaDBをロード
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embeddings
    )
    
    # ChromaDBを「Retriever」として設定
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) 
    print("ロード完了。")
    return llm, retriever

# --- 2. RAGチェーンの定義 (修正なし) ---
def create_rag_chain(llm, retriever):
    # Define the prompt template
    prompt_template = ChatPromptTemplate.from_template(
        """
        あなたは法律分野のアシスタントです。
        提供された「コンテキスト情報」のみに基づいて、質問に答えてください。
        コンテキスト情報に答えが無い場合は、「コンテキストからでは分かりません」と回答してください。
        
        【コンテキスト情報】
        {context}
        
        【質問】
        {question}
        
        【回答】
        """
    )

    # Define the chain that retrieves context and prepares the input for the LLM
    # This part will return a dictionary like {"context": [...], "question": "..."}
    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )

    # Define the chain that generates the answer using the prompt and LLM
    answer_generation_chain = prompt_template | llm | StrOutputParser()

    # Combine them: first get context and question, then use that to generate the answer.
    # We want to return both the generated answer and the original context.
    rag_chain_with_sources = RunnableParallel(
        answer=setup_and_retrieval | answer_generation_chain,
        context=setup_and_retrieval | itemgetter("context"), # Explicitly get context from the setup_and_retrieval output
    )
    return rag_chain_with_sources

# --- 3. Streamlit アプリのUI定義 (修正なし) ---
st.title("⚖️ 親権喪失・親権停止 RAGチャット (参考文献Ver.)")
st.caption(f"使用LLM: {LLM_MODEL_NAME} | 埋め込み: {EMBED_MODEL_NAME}")

try:
    llm, retriever = get_models_and_retriever()
    rag_chain = create_rag_chain(llm, retriever)
except Exception as e:
    st.error(f"モデルまたはDBのロードに失敗しました: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("質問を入力してください (例: 親権停止の要件は？)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("AIが回答を生成中です..."):
            result = rag_chain.invoke(prompt)
            
            st.markdown(result["answer"]) # Display the answer
            
            # Display sources in an expander
            if result.get("context"):
                with st.expander("参照元ドキュメント"):
                    for i, doc in enumerate(result["context"]):
                        st.write(f"**ドキュメント {i+1} (出典: {doc.metadata.get('source', '不明')})**")
                        st.write(doc.page_content)
                        st.markdown("---")
            else:
                st.info("参照元ドキュメントは見つかりませんでした。")
    
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
