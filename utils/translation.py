# utils/translation.py
from deep_translator import MyMemoryTranslator
import html
import os
import time

def translate_text(text, target_language='si'):
    """
    Translate text to Sinhala (or other target language) using MyMemory Translator
    
    Args:
        text (str): Text to translate
        target_language (str): Target language code (default: 'si' for Sinhala)
    
    Returns:
        str: Translated text
    """
    try:
        # Split text into smaller chunks to avoid length limitations
        chunks = split_into_chunks(text, 500)  # MyMemory has a character limit
        translated_chunks = []
        
        for chunk in chunks:
            if chunk.strip():
                # Add delay to respect rate limits
                time.sleep(1)
                
                # Translate chunk
                translator = MyMemoryTranslator(source='en', target=target_language)
                translated_text = translator.translate(chunk)
                translated_chunks.append(translated_text)
            else:
                translated_chunks.append('')
                
        # Combine all translated chunks
        return '\n'.join(translated_chunks)
        
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return f"{text}\n\n[Translation error: {str(e)}]"

def split_into_chunks(text, chunk_size):
    """Split text into chunks while preserving paragraphs"""
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para_length = len(para)
        if current_length + para_length > chunk_size and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = [para]
            current_length = para_length
        else:
            current_chunk.append(para)
            current_length += para_length
            
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
        
    return chunks