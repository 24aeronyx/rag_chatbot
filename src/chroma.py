from chromadb import PersistentClient

# Inisialisasi client dengan path direktori penyimpanan
client = PersistentClient(path="./embeddings")  # Sesuaikan dengan path kamu

# Hapus collection berdasarkan nama
collection_name = "penyakit_embeddings"
client.delete_collection(name=collection_name)

print(f"✅ Collection '{collection_name}' berhasil dihapus.")
