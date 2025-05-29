import json

INPUT_FILE = 'Data/penyakit-data-raw.json'
OUTPUT_FILE = 'Data/penyakit-data-processed.json'

def preprocess_data(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_data = []

    for entry in data:
        name = entry["name"]
        href = entry["href"]
        paragraphs = entry["paragraphs"]

        # Hapus paragraf terakhir (referensi)
        if paragraphs:
            paragraphs = paragraphs[:-1]

        # Ganti ; dengan , di setiap paragraf
        cleaned_paragraphs = [p.replace(';', ',').strip() for p in paragraphs if p.strip()]

        processed_data.append({
            "name": name,
            "href": href,
            "paragraphs": cleaned_paragraphs
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Preprocessing selesai. Data disimpan di: {output_path}")

if __name__ == "__main__":
    preprocess_data(INPUT_FILE, OUTPUT_FILE)
