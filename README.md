---

# Llama-Powered Resume Ranking

This repository contains the code for a Streamlit application that utilizes a hybrid retrieval approach to rank resumes based on a provided job description.

## Features

- **Hybrid Retrieval**: Combines vector embeddings and BM25 similarity for robust search.
- **Sentence Transformers**: Leverages pre-trained sentence transformers for efficient text representation.
- **Streamlit Integration**: Provides a user-friendly interface for job description input and ranked resume display.
- **Optional Reranking**: Implements a sentence transformer-based reranking step for potentially improved results (requires additional model configuration).
- **Offline Persistence**: Saves the built index to disk for faster loading on subsequent application runs.

## Requirements

- Python 3.7+
- Streamlit
- llama-index
- faiss
- sentence-transformers
- (Optional) Additional libraries for specific reranking models

## Installation

1. Clone this repository.
2. Create a new virtual environment (recommended).
3. Navigate to the project directory.
4. Install required dependencies using `pip install -r requirements.txt`.

## Running the Application

1. Ensure you have the necessary data (resumes) in the `data/resumes` directory.
2. Run the application using `streamlit run main.py`.
3. Enter a job description in the text area provided.
4. Click the "Rank Resumes" button.
5. The application will display the top 10 most relevant resumes based on the combined retrieval strategy.
6. (Optional) The application will also display the top 10 reranked results if a reranking model is specified.

## Additional Notes

- The code utilizes pre-trained models for sentence embedding and reranking (if enabled). Download the desired models and update the configuration accordingly.
- The current implementation uses a basic text cleaning step. You can customize this process for more advanced text pre-processing based on your data characteristics.
- The application is designed to be a starting point for building a resume ranking system. You can further customize and extend it based on your specific needs.

--- 