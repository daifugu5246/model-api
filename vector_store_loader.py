import os
from pydantic import BaseModel, Field
from typing import List
from langchain_community.vectorstores import FAISS
from langchain.schema import Document, BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# โหลด embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-xlm-r-multilingual-v1")
q1_report = 28.52694625
q1_news = 20.24707

#โหลด_vector_stores
def load_report_vector_stores(base_path, symbols, quarters, embeddings):
    vector_stores = {}
    failed_files = {}
    for symbol in symbols:
        for quarter in quarters:
            file_path = f"{base_path}/{symbol}/vector_store_{quarter}"
            if not os.path.exists(file_path):
                failed_files[f"{symbol}_{quarter}"] = "File not found"
                continue
            try:
                vector_stores[f"vector_store_{symbol}_{quarter}"] = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
            except Exception as e :
                failed_files[f"{symbol}_{quarter}"] = f"Error loading file: {str(e)}"
    return vector_stores,failed_files

#โหลด news_vector_stores
def load_news_vector_stores(base_path, symbols, embeddings):
    vector_stores = {}
    failed_files = {}
    for symbol in symbols:
        file_path = f"{base_path}/{symbol}/vector_store_{symbol}_news"
        if not os.path.exists(file_path):
            failed_files[f"vector_store_{symbol}_news"] = "File not found"
            continue
        try:
            vector_stores[f"vector_store_{symbol}_news"] = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
        except Exception as e :
            failed_files[f"vector_store_{symbol}_news"] = f"Error loading file: {str(e)}"
    return vector_stores,failed_files


class CombinedRetriever(BaseRetriever, BaseModel):
    retrievers: List[BaseRetriever] = Field(...)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        combined_results = []
        for retriever in self.retrievers:
            results = retriever.get_relevant_documents(query)
            combined_results.extend(results)
        return combined_results

def select_and_combine_retrievers(symbol:str, quarter:str, report_vectorstore, news_vectorstore, instruction: str) -> CombinedRetriever:
    retrievers = []
    docs_with_scores_all = []

    docs_with_scores = report_vectorstore.similarity_search_with_score(instruction, k=5)
    filter_docs = [doc for doc, score in docs_with_scores if score > 0]
    retrievers.append(filter_docs)
    
    # รวม retrievers สำหรับข่าว
    news_with_scores = news_vectorstore.similarity_search_with_score(instruction, k=5)
    filter_news = [doc for doc, score in news_with_scores if score > 0]
    retrievers.append(filter_news)

    if not retrievers:
        raise ValueError("No valid retrievers found for the given symbols, quarters, or news.")

    return CombinedRetriever(retrievers=retrievers), docs_with_scores_all

def load_all_vector_stores():
    report_base_path = "./data/VECTOR_STORE_1600"
    news_base_path = "./data/VECTOR_STORE_NEWS_1600"

    symbols = ["PTT"]
    quarters = ["Q1_67", "Q2_67", "Q1_66", "Q2_66", "Q3_66", "Q4_66"]

    # โหลดข้อมูล
    report_vector_stores, report_failed_files = load_report_vector_stores(report_base_path, symbols, quarters, embeddings)
    news_vector_stores, news_failed_files = load_news_vector_stores(news_base_path, symbols, embeddings)

    # print(report_vector_stores)
    return {
        "report_vector_stores": report_vector_stores,
        "news_vector_stores": news_vector_stores,
        "report_failed_files": report_failed_files,
        "news_failed_files": news_failed_files
    }

# print(load_all_vector_stores())

# ตรวจสอบ vectorstore
# report_vector_data = load_all_vector_stores()
# report_vector_stores = report_vector_data["report_vector_stores"]
# print("Report Vector Stores:", list(report_vector_stores.keys()))



