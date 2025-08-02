import os
import argparse
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
# --- UPDATED IMPORT ---
# We now use HuggingFaceInstructEmbeddings for the new model.
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

def build_and_save_index(docs_path, output_path):
    """
    Loads PDFs from a directory, creates a FAISS vector index, and saves it.
    """
    print(f"--- Starting Index Build Process ---")

    # --- Step 1: Load Documents ---
    print(f"1. Loading documents from '{docs_path}' using PyMuPDF...")
    all_docs = []
    if not os.path.exists(docs_path):
        print(f"ERROR: Document path not found: {docs_path}")
        return

    for root, _, files in os.walk(docs_path):
        for file in files:
            if file.endswith(".pdf"):
                try:
                    file_path = os.path.join(root, file)
                    loader = PyMuPDFLoader(file_path)
                    loaded_docs = loader.load()
                    all_docs.extend(loaded_docs)
                except Exception as e:
                    print(f"   -> ERROR loading {file}: {e}")

    if not all_docs:
        print("ERROR: No documents were loaded. Exiting.")
        return
    print(f"   -> Total document chunks loaded: {len(all_docs)}")

    # --- Step 2: Initialize Instruction-Tuned Embedding Model ---
    print("2. Initializing hkunlp/instructor-xl embedding model...")
    print("   (The first run will download the model, which may take some time.)")
    
    # --- UPDATED MODEL ---
    # This model is specifically designed for understanding technical text.
    embedding_model_name = "hkunlp/instructor-xl"
    
    # We provide specific instructions to the model to optimize it for our task.
    # This tells the model how to handle the documents for the best retrieval performance.
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=embedding_model_name,
        embed_instruction="Represent the scientific document for retrieval: "
    )
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