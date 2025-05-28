from chromadb import PersistentClient

# Konfigurasi
PERSIST_DIR = './embeddings'
COLLECTION_NAME = 'penyakit_embeddings'

def count_chunks():
    # Hubungkan ke ChromaDB
    client = PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Ambil semua dokumen (tanpa batasan jumlah jika memungkinkan)
    items = collection.peek()  # Ambil sebagian data untuk dicek format

    if not items or len(items['ids']) == 0:
        print("📭 Collection kosong atau tidak ditemukan.")
        return

    # Gunakan count berdasarkan jumlah 'ids'
    all_data = collection.get()
    total = len(all_data['ids'])

    print(f"📦 Total chunk yang tersedia di koleksi '{COLLECTION_NAME}': {total}")

if __name__ == "__main__":
    count_chunks()
