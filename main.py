# main.py

# 1. Import the functions from your other files
import os
import glob
import logging
from data_loader import extract_text_from_pdf
from chunker import chunk_text
from embedder import embed_chunks
from vector_db import upload_to_qdrant, close_connection

def run_rag_pipeline():
    print("Starting RAG Pipeline...")
    logging.info("Starting Multi-Document RAG Pipeline...")
        
        # 1. Find ALL PDFs in the current folder
    pdf_files = glob.glob("fintech-docs/*.pdf")
        
    if not pdf_files:
        logging.error("No PDF files found in this folder!")
        return
        
    logging.info(f"Found {len(pdf_files)} PDFs to process: {pdf_files}\n")

    # 2. Loop through each PDF
    for pdf_path in pdf_files:
        logging.info(f"{'='*60}")
        logging.info(f" Processing: {pdf_path}")
        logging.info(f"{'='*60}")

        # --- STEP 1: EXTRACTION ---
        print("\n[1/3] Extracting text from PDF...")
        raw_text = extract_text_from_pdf(pdf_path)
        print(f"Success! Extracted {len(raw_text)} characters.")

        # --- STEP 2: CHUNKING ---
        print("\n[2/3] Chunking the text...")
        # We pass the 'raw_text' variable directly into the chunker!
        chunks = chunk_text(raw_text, chunk_size=800, overlap=100)
        print(f"Success! Created {len(chunks)} chunks.")

        # --- STEP 3: EMBEDDING ---
        print("\n[3/3] Generating embeddings via Cohere...")
        # We pass the 'chunks' variable directly into the embedder!
        embeddings = embed_chunks(chunks)
        print(f"Success! Generated {len(embeddings)} vectors.")

        # --- FINAL OUTPUT ---
        print(f"Your data is ready to be saved to a Vector Database.")

        upload_to_qdrant(chunks, embeddings, source_filename=pdf_path)
        
        print("Pipeline Complete! Your data is ready to be searched.")
            
    # --- CRITICAL FIX: Close the database gracefully ---
    close_connection()
    # Optional: Return the data if you want to use it later in this script
    return chunks, embeddings

# Run the pipeline
if __name__ == "__main__":
    run_rag_pipeline()