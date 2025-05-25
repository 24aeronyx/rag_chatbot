import json

# Path input dan output
INPUT_FILE = 'Data/penyakit-data-processed.json'
OUTPUT_FILE = 'Data/penyakit-data-chunked.json'

def chunk_data(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    chunked_data = []

    for entry in data:
        name = entry["name"]
        href = entry["href"]
        paragraphs = entry["paragraphs"]

        # Optional: hapus paragraf terakhir jika ingin
        if paragraphs:
            paragraphs = paragraphs[:-1]

        # Setiap paragraf menjadi satu chunk
        chunks = [p.strip() for p in paragraphs if p.strip()]

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
