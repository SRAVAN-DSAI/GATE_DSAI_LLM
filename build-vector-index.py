import os
import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def build_and_save_index(docs_path, output_path):
    """
    Loads PDFs from a directory, creates a FAISS vector index, and saves it.

    Args:
        docs_path (str): The path to the directory containing PDF files.
        output_path (str): The path to save the created FAISS index.
    """
    print(f"--- Starting Index Build Process ---")

    # --- Step 1: Load Documents ---
    print(f"1. Loading documents from '{docs_path}'...")
    all_docs = []
    if not os.path.exists(docs_path):
        print(f"ERROR: Document path not found: {docs_path}")
        return

    for root, _, files in os.walk(docs_path):
        for file in files:
            if file.endswith(".pdf"):
                try:
                    file_path = os.path.join(root, file)
                    loader = PyPDFLoader(file_path)
                    loaded_docs = loader.load_and_split()
                    all_docs.extend(loaded_docs)
                except Exception as e:
                    print(f"   -> ERROR loading {file}: {e}")

    if not all_docs:
        print("ERROR: No documents were loaded. Exiting.")
        return
    print(f"   -> Total document chunks loaded: {len(all_docs)}")

    # --- Step 2: Initialize Embedding Model from Kaggle Input ---
    print("2. Initializing BGE embedding model from local Kaggle path...")
    
    # --- UPDATED MODEL PATH ---
    # This now points to the model you added to your Kaggle notebook,
    # which avoids a slow download from the internet.
    embedding_model_name = "/kaggle/input/baaibge-en-icl/pytorch/default/1"
    
    # Check if the model path exists
    if not os.path.exists(embedding_model_name):
        print(f"ERROR: Kaggle model not found at '{embedding_model_name}'.")
        print("Please ensure you have added the 'bge-large-en-v1-5' model to your notebook.")
        return
        
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    print("   -> Model initialized successfully.")

    # --- Step 3: Create and Save Index ---
    print("3. Creating FAISS vector index... (This may take several minutes)")
    vector_store = FAISS.from_documents(all_docs, embeddings)
    print("   -> Index created successfully.")
    
    print(f"4. Saving index to '{output_path}'...")
    vector_store.save_local(output_path)
    
    if os.path.exists(os.path.join(output_path, "index.faiss")):
        print("   -> Verification successful. Index saved.")
    else:
        print("   -> ERROR: Index saving failed.")

    print("--- Index Build Process Complete ---")

if __name__ == "__main__":
    # Define the default input and output paths for the Kaggle environment
    KAGGLE_INPUT_DIR = "/kaggle/input/gate-dsai-llm"
    KAGGLE_OUTPUT_DIR = "/kaggle/working/my_vector_index"
    
    build_and_save_index(KAGGLE_INPUT_DIR, KAGGLE_OUTPUT_DIR)