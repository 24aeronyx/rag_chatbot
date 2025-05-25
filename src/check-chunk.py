from chromadb import PersistentClient

# Inisialisasi client dan collection
client = PersistentClient(path="./embeddings")  # Sesuaikan dengan path kamu
collection = client.get_collection(name="penyakit_embeddings")

# Cari dokumen yang mengandung frasa tertentu (misalnya "Kanker Pembuluh Darah")
# atau kamu bisa cari berdasarkan metadata 'name' jika kamu menyimpannya di sana
results = collection.query(
    query_texts=["Diare"],
    n_results=5
)

# Tampilkan hasil
for i, doc in enumerate(results['documents'][0]):
    print(f"\n📄 Dokumen {i+1}:\n{doc}")
