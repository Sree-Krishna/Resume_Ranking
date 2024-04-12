import asyncio
import nest_asyncio
import faiss
import os
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
    Settings,
    Document,
    QueryBundle
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from hybrid_retriever import HybridRetriever
from utils import TextCleaner, MyFileReader, clean_text
import streamlit as st

def create_text_pipeline():
    """
    Creates a pipeline for text preprocessing and embedding.
    """
    return IngestionPipeline(
        transformations=[TextCleaner(),
                         SentenceSplitter(chunk_size=512, chunk_overlap=10),
                         embed_model]
    )

# Define embedding model (update with desired model name)
# checkout Voyage embedding, lightweight and high performance
# model_name = 'BAAI/bge-small-en-v1.5'
model_name = "sentence-transformers/all-MiniLM-L6-v2"
# Global setting for embedding model
embed_model = HuggingFaceEmbedding(model_name)
Settings.embed_model = embed_model

@st.cache_resource(show_spinner=False)
def load_or_create_index(path="data/resumes", storage_path=''):
    """
    Loads the index from disk if it exists, otherwise creates a new one.
    """
    if os.path.exists(storage_path):
        vector_store = FaissVectorStore.from_persist_dir(storage_path)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=storage_path
        )
        index = load_index_from_storage(storage_context=storage_context)
    else:
        documents = SimpleDirectoryReader(input_dir=path, file_extractor={".csv": MyFileReader()}).load_data()
        pipeline = create_text_pipeline()
        nodes = pipeline.run(documents=documents)
        # StorageContext can be created to handle multiple storage type like doc store, vector store
        d = len(embed_model.get_text_embedding("Test Embedding"))
        faiss_index = faiss.IndexFlatL2(d)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # you can also pre-chunk the nodes, and pass those in. This will not apply chunking, it will just embed
        index = VectorStoreIndex(nodes, storage_context=storage_context)
        index.storage_context.persist(persist_dir=storage_path)
        index.storage_context.persist()
    return index


def create_retrievers(index):
    """
    Creates the hybrid retriever for ranking resumes.
    """
    # retireve the top 20 most similar nodes using embeddings
    vector_retriever = index.as_retriever(similarity_top_k=20)
    # retireve the top 20 most similar nodes using bm25
    # bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=20)
    bm25_retriever = vector_retriever
    hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)
    return hybrid_retriever

def create_reranker(model_name="cross-encoder/ms-marco-MiniLM-L-2-v2"):
    """
    Creates a sentence transformer reranker (optional).
    Ref: https://www.sbert.net/docs/pretrained_models.html#model-overview
    https://www.sbert.net/examples/applications/retrieve_rerank/README.html
    """
    return SentenceTransformerRerank(top_n=20, model=model_name) if model_name else None

def display_results(data):
    final_list = {}
    final_text = {}
    count = 0
    for i in range(len(data)):
        if data[i].metadata["file_name"] not in final_list:
            # TODO: to get unique document -> data[i].node.ref_doc_id
            final_list[data[i].metadata["file_name"]+' '+data[i].id_[-5:]] = data[i].score
            final_text[data[i].metadata["file_name"]+' '+data[i].id_[-5:]] = data[i].text
            count += 1
        if count == 10:
            break
    for key in final_list:
        st.write(f"{key} (Score: {final_list[key]:.2f})")
        st.code(final_text[key], language="markdown")
        # st.write(final_text[key])

def resume_ranking_page():
    """Streamlit app for resume ranking"""

    st.title("Resume Ranking")
    st.subheader("Get highly relevant resumes for your job description")
    st.write("Top 10 matching resumes will be displaed")

    job_description = st.text_area("Enter Job Description:", height=100)
    job_description = clean_text(job_description)

    with st.spinner("Indexing Resumes..."):
        resume_index = load_or_create_index(path='data/resumes', storage_path='./storage_resume')

    resume_hybrid_retriever = create_retrievers(resume_index)

    reranker = create_reranker()

    if st.button("Rank Resumes"):
        with st.spinner("processing..."):
            retrieved_nodes = resume_hybrid_retriever.retrieve(job_description)
            if reranker:
                reranked_nodes = reranker.postprocess_nodes(
                    retrieved_nodes, query_bundle=QueryBundle(job_description)
                )
            else:
                reranked_nodes = retrieved_nodes

        # st.subheader("Top 10 Matching Resumes:")
        # display_results(retrieved_nodes)
        if reranker:
            st.subheader("The top matched resumes in descending order of scores:")
            display_results(reranked_nodes)

def job_matching_page():
    """
    Job matching page functionalities (placeholder for future implementation).
    """
    st.title("Job Matching")
    st.subheader("Get highly relevant job matches for your resume")
    st.write("Top 10 matching jobs will be displayed")
    resume_text = st.text_area("Enter your resume as text:", height=100)
    resume_text = clean_text(resume_text)

    with st.spinner("Indexing Jobs..."):
        job_index = load_or_create_index(path='data/job_descriptions', storage_path='./storage_jobs')

    job_hybrid_retriever = create_retrievers(job_index)

    reranker = create_reranker()

    if st.button("Fetch Jobs"):
        with st.spinner("processing..."):
            retrieved_nodes = job_hybrid_retriever.retrieve(resume_text)
            if reranker:
                reranked_nodes = reranker.postprocess_nodes(
                    retrieved_nodes, query_bundle=QueryBundle(resume_text)
                )
            else:
                reranked_nodes = retrieved_nodes

        # st.subheader("Top 10 Matching Resumes:")
        # display_results(retrieved_nodes)
        if reranker:
            st.subheader("The top matching jobs in descending order of scores:")
            display_results(reranked_nodes)

def main():
    """
    Streamlit app with sidebar and page navigation.
    """
    st.set_page_config(page_title="Nirvana Resume Ranking and Job Matching")

    page = st.sidebar.selectbox("Select the app", ["Resume Ranking", "Job Matching"])

    if page == "Resume Ranking":
        resume_ranking_page()
    elif page == "Job Matching":
        job_matching_page()


if __name__ == "__main__":
    # Create a new event loop
    loop = asyncio.new_event_loop()

    # Set the event loop as the current event loop
    asyncio.set_event_loop(loop)
    nest_asyncio.apply()

    main()
