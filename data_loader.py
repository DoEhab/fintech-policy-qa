# pdf_loader.py
import os
import logging
import pdfplumber # <-- Upgraded extractor

def extract_text_from_pdf(pdf_path):
    """
    Extracts text and metadata from a PDF using pdfplumber.
    Returns a tuple: (text, metadata_dict)
    """
    try:
        metadata = {
            "filename": os.path.basename(pdf_path),
            "title": os.path.basename(pdf_path).replace(".pdf", "").replace("_", " ")
        }
        
        full_text = ""
        
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # extract_text() handles complex layouts much better than pypdf
                page_text = page.extract_text()
                if page_text:
                    full_text += f"\n --- Page {i + 1} ---\n" + page_text
                    
        if len(full_text.strip()) < 50:
            logging.warning(f"Very little text extracted from {metadata['filename']}. It may be a scanned image.")
            
        return full_text, metadata
        
    except Exception as e:
        logging.error(f" Failed to extract text from {pdf_path}. Error: {e}")
        return "", {"filename": os.path.basename(pdf_path), "title": os.path.basename(pdf_path)}

    
"""pypdf fails to extract text from some PDFs, so we use this as a fallback."""
# """"""
# from pypdf import PdfReader

# def extract_text_from_pdf(pdf_path):
#     """
#     Extracts text from a PDF file.

#     Args:
#         pdf_path (str): The path to the PDF file.

#     Returns:
#         str: The extracted text from the PDF file.
#     """
#     reader = PdfReader(pdf_path)
#     text = ""
#     for i, page in enumerate(reader.pages):
#         temp_text = page.extract_text()
#         text += f"\n --- Page {i + 1} ---\n" + temp_text

#     return text

