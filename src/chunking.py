import json

# Konfigurasi
INPUT_FILE = './data/penyakit-data-processed.json'
OUTPUT_FILE = './data/penyakit-data-chunked.json'
MIN_CHARS = 100
MAX_CHARS = 800

def chunk_paragraphs(paragraphs, min_chars=MIN_CHARS, max_chars=MAX_CHARS):
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 1 <= max_chars:
            current_chunk += " " + para if current_chunk else para
        else:
            if len(current_chunk) >= min_chars:
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                # Jika current_chunk terlalu pendek, tetap tambahkan para
                current_chunk += " " + para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

def chunk_data(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    chunked_data = []

    for entry in data:
        name = entry["name"]
        href = entry["href"]
        paragraphs = entry["paragraphs"]

        # Hapus paragraf terakhir (biasanya referensi)
        if paragraphs:
            paragraphs = paragraphs[:-1]

        # Gabungkan paragraf ke dalam chunk berdasarkan panjang karakter
        chunks = chunk_paragraphs(paragraphs)

        chunked_data.append({
            "name": name,
            "href": href,
            "chunks": chunks
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunked_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Chunking selesai. Data disimpan di: {output_path}")

if __name__ == "__main__":
    chunk_data(INPUT_FILE, OUTPUT_FILE)
