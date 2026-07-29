import re

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits the input text into chunks of specified size.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The maximum size of each chunk.
        overlap (int): The number of overlapping characters between consecutive chunks.
    Returns:
        list: A list of text chunks.
    """
    sentences = re.split(r'(?<=[.!?]) +', text)  # Split text into sentences
    chunks = []
    current_chunk = []  # Fixed typo (was current_chunck)
    current_length = 0

    for sentence in sentences:  # Fixed indentation (removed extra space)
        # If adding this sentence exceeds chunk_size, save the chunk
        if current_length + len(sentence) > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            
            # Handle Overlap: Keep the last few sentences/words for the next chunk
            overlap_text = " ".join(current_chunk)
            # Simple overlap logic: take the last 'overlap' characters
            overlap_words = overlap_text[-overlap:].split()
            current_chunk = overlap_words 
            current_length = len(" ".join(current_chunk))
            
        current_chunk.append(sentence)
        current_length += len(sentence) + 1
        
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

