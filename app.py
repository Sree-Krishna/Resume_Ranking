import streamlit as st
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
import faiss
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
    Settings,
    Document,
    QueryBundle
)
from llama_index.vector_stores.faiss import FaissVectorStore
from hybrid_retriever import HybridRetriever
from clean_data import TextCleaner, clean_job
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import SentenceTransformerRerank
import asyncio

# set the embedding
# checkout Voyage embedding, lightweight and high performance
# model_name = 'BAAI/bge-small-en-v1.5'
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
Settings.embed_model = HuggingFaceEmbedding(model_name)
embed_model = HuggingFaceEmbedding(model_name)

# Function to preprocess and vectorize text (replace with more advanced NLP)
def preprocess_text(text):
    return text.lower()

@st.cache_resource(show_spinner=False)
def create_nodes(path='data/resumes'):
    # create the pipeline with transformations
    pipeline = IngestionPipeline(
        transformations=[
            TextCleaner(),
            SentenceSplitter(chunk_size=512, chunk_overlap=10),
            embed_model,
        ]
    )
    with st.spinner(text="Loading the Streamlit docs – hang tight! This should take 1-2 minutes."):
        documents = SimpleDirectoryReader(path).load_data()
        nodes = pipeline.run(documents=documents)
    # index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    return nodes

def set_storage_context(vetcorDb='faiss'):
  
  if vetcorDb == 'faiss':
    d = len(embed_model.get_text_embedding("hello world"))
    faiss_index = faiss.IndexFlatL2(d)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return storage_context

def create_index(nodes, storage_context, documents=False):
    with st.spinner(text="Loading the Streamlit Index – hang tight! This should take 1-2 minutes."):
        if documents:
            index = VectorStoreIndex.from_documents(documents=documents, storage_context=storage_context)
        else:
            index = index = VectorStoreIndex(nodes, storage_context=storage_context)
        return index

def set_retriever(nodes, index):
    # retireve the top 20 most similar nodes using embeddings
    vector_retriever = index.as_retriever(similarity_top_k=20)

    # retireve the top 20 most similar nodes using bm25
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=20)
    hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)
    return hybrid_retriever

def display_results(data):
    final_list = {}
    count = 0
    for i in range(len(data)):
        if data[i].metadata['file_name'] not in final_list:
            final_list[data[i].metadata['file_name']] = data[i].score
            count += 1
        if count == 10:
            break
    for key in final_list:
        st.write(f"{key} Resume (Score: {final_list[key]:.2f})")

def main():
    """Streamlit app for resume ranking"""

    st.title("Resume Ranking")

    job_description = st.text_area("Enter Job Description:", height=100)
    job_description = clean_job(job_description)

    nodes = create_nodes()
    storage_context = set_storage_context()
    index = create_index(nodes, storage_context)

    # save index to disk
    index.storage_context.persist()

    # load index from disk
    vector_store = FaissVectorStore.from_persist_dir("./storage")
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir="./storage"
    )
    index = load_index_from_storage(storage_context=storage_context)

    # set retriever
    hybrid_retriever = set_retriever(nodes, index)

    # set reranker
    rerank_model = 'cross-encoder/ms-marco-MiniLM-L-2-v2'
    # rerank_model = 'BAAI/bge-reranker-base'
    reranker = SentenceTransformerRerank(top_n=20, model=rerank_model)

    if st.button("Rank Resumes"):
        # Preprocess and vectorize the job description
        # job_description_vector = vectorizer.transform([job_description])[0]
        with st.spinner("processing..."):
            retrieved_nodes = hybrid_retriever.retrieve(job_description)
            st.subheader("Top 10 Matching Resumes:")
            display_results(retrieved_nodes)

            reranked_nodes = reranker.postprocess_nodes(
                retrieved_nodes,
                query_bundle=QueryBundle(job_description),
            )
            st.subheader("Top 10 Matching Resumes Reranked:")
            display_results(reranked_nodes)

if __name__ == "__main__":
  main()
