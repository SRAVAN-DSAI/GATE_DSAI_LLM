import os
import argparse
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- Argument Parsing ---
parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str, help="Path to the input PDF data.")
parser.add_argument("--output_index", type=str, help="Path to save the output index.")
args = parser.parse_args()

print(f"--- Starting Index Build Job ---")
print(f"Input data path: {args.input_data}")
print(f"Output index path: {args.output_index}")

# --- Step 1: Load Documents ---
print(f"1. Loading documents from '{args.input_data}' using PyMuPDF...")
all_docs = []
for root, _, files in os.walk(args.input_data):
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
    raise ValueError("No documents were loaded. Exiting.")
print(f"   -> Total document chunks loaded: {len(all_docs)}")

# --- Step 2: Initialize Embedding Model ---
print("2. Initializing BAAI/bge-m3 embedding model...")
embedding_model_name = "BAAI/bge-m3"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
print("   -> Model initialized.")

# --- Step 3: Create Index in Batches for Memory Efficiency ---
print("3. Creating FAISS vector index in batches...")

batch_size = 500  # Process 500 documents at a time
vector_store = None

for i in range(0, len(all_docs), batch_size):
    batch = all_docs[i:i + batch_size]
    print(f"   -> Processing batch {i//batch_size + 1}...")
    
    if vector_store is None:
        # Create the index from the first batch
        vector_store = FAISS.from_documents(batch, embeddings)
    else:
        # Add subsequent batches to the existing index
        vector_store.add_documents(batch)

print("   -> Index created successfully.")

# --- Step 4: Save the Final Index ---
print(f"4. Saving index to '{args.output_index}'...")
vector_store.save_local(args.output_index)

# Verification
index_file_path = os.path.join(args.output_index, "index.faiss")
if os.path.exists(index_file_path):
    print("   -> Verification successful. Index saved.")
else:
    raise RuntimeError("ERROR: Index saving failed.")

print("--- Index Build Job Complete ---")