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
        
    logging.info(f"Found {len(pdf_files)} PDFs to process.\n")
    
    success_count = 0
    failed_count = 0

    # 2. Loop through each PDF
    for pdf_path in pdf_files:
        logging.info(f"{'='*60}")
        logging.info(f" Processing: {pdf_path}")
        logging.info(f"{'='*60}")

        try:
            # --- STEP 1: EXTRACTION ---
            print("\n[1/3] Extracting text from PDF...")
            result = extract_text_from_pdf(pdf_path)
            
            # FIX 1: Unpack the result FIRST, before printing
            if isinstance(result, tuple):
                raw_text, metadata = result
                source_name = metadata.get("filename", pdf_path)
            else:
                raw_text = result
                source_name = pdf_path
                
            print(f"Success! Extracted {len(raw_text)} characters.")

            # FIX 2: Safety check to skip empty/scanned PDFs
            if len(raw_text.strip()) < 100:
                print(f"SKIPPING: Extracted too little text. File might be scanned or encrypted.\n")
                failed_count += 1
                continue

            # --- STEP 2: CHUNKING ---
            print("\n[2/3] Chunking the text...")
            chunks = chunk_text(raw_text, chunk_size=800, overlap=100)
            print(f"Success! Created {len(chunks)} chunks.")

            # --- STEP 3: EMBEDDING ---
            print("\n[3/3] Generating embeddings...")
            embeddings = embed_chunks(chunks)
            print(f"Success! Generated {len(embeddings)} vectors.")

            # --- FINAL OUTPUT ---
            print(f"Your data is ready to be saved to a Vector Database.")
            upload_to_qdrant(chunks, embeddings, source_filename=source_name)
            
            print("🎉 Pipeline Complete for this document!\n")
            success_count += 1
            
        except Exception as e:
            # FIX 3: If a file crashes, log it and keep going!
            print(f" FAILED to process {pdf_path}. Error: {e}\n")
            failed_count += 1
            continue
            
    print(f"\n{'='*60}")
    print(f"PIPELINE FINISHED")
    print(f"Successfully processed: {success_count} documents")
    print(f"Failed/Skipped: {failed_count} documents")
    print(f"{'='*60}")
            
    close_connection()

# Run the pipeline
if __name__ == "__main__":
    run_rag_pipeline()