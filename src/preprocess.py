import json

def preprocess_data(input_file, output_file):
    # Load data JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Proses setiap paragraf untuk membersihkan newline
    for entry in data:
        entry['paragraphs'] = [para.replace("\n", " ").strip() for para in entry['paragraphs']]

    # Simpan data yang telah diproses
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    preprocess_data('./data/penyakit-data.json', './data/penyakit-data-processed.json')
