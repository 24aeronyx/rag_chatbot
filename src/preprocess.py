import json

def preprocess_data(input_file, output_file):
    # Load data JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Proses setiap entri penyakit
    for entry in data:
        # Bersihkan newline dan hapus paragraf terakhir
        cleaned_paragraphs = [para.replace("\n", " ").strip() for para in entry['paragraphs']]
        if cleaned_paragraphs:
            cleaned_paragraphs.pop()  # Hapus paragraf terakhir
        entry['paragraphs'] = cleaned_paragraphs

    # Simpan data yang telah diproses
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    preprocess_data('./data/penyakit-data-final.json', './data/penyakit-data-processed.json')
