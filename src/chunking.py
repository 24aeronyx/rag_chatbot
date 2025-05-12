import json
import os
import nltk
from nltk.tokenize import sent_tokenize

# Download tokenizer untuk pertama kali
nltk.download('punkt_tab')

# Konfigurasi
INPUT_FILE = './data/penyakit-data-processed.json'
OUTPUT_FILE = './data/penyakit-data-chunked.json'
MAX_CHARS = 800
OVERLAP_SENTENCES = 1

def chunk_paragraph(paragraph, max_chars=MAX_CHARS, overlap_sentences=OVERLAP_SENTENCES):
    sentences = sent_tokenize(paragraph)
    chunks = []
    current_chunk = []

    for sentence in sentences:
        temp_chunk = " ".join(current_chunk + [sentence])
        if len(temp_chunk) <= max_chars:
            current_chunk.append(sentence)
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            # Mulai chunk baru dengan overlap kalimat sebelumnya
            current_chunk = current_chunk[-overlap_sentences:] + [sentence]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def chunk_data(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    chunked_data = []

    for entry in data:
        name = entry["name"]
        href = entry["href"]
        paragraphs = entry["paragraphs"]
        
        chunked_paragraphs = []
        for paragraph in paragraphs:
            chunks = chunk_paragraph(paragraph)
            chunked_paragraphs.extend(chunks)

        chunked_data.append({
            "name": name,
            "href": href,
            "chunks": chunked_paragraphs
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunked_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Chunking selesai. Data disimpan di: {output_path}")

if __name__ == "__main__":
    chunk_data(INPUT_FILE, OUTPUT_FILE)
