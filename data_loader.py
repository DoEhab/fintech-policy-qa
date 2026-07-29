from pypdf import PdfReader

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text from the PDF file.
    """
    reader = PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        temp_text = page.extract_text()
        text += f"\n --- Page {i + 1} ---\n" + temp_text

    return text

