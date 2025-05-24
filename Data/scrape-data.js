const { chromium } = require("playwright");
const fs = require("fs");

const links = JSON.parse(fs.readFileSync("./data/penyakit-links.json"));
const checkpointFile = "./data/checkpoint.json";
let checkpoint = { lastIndex: 0, penyakitData: [] };
if (fs.existsSync(checkpointFile)) {
  checkpoint = JSON.parse(fs.readFileSync(checkpointFile));
  console.log(`Melanjutkan dari indeks: ${checkpoint.lastIndex}`);
}

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();

  const delay = (ms) => new Promise((res) => setTimeout(res, ms));
  const penyakitData = checkpoint.penyakitData;

  // Daftar kata blacklist, bisa disesuaikan
  const blacklistKeywords = [
    // ... isi blacklist ...
    "Beranda", "Chat Bersama Dokter", "Penyakit A-Z", "Obat A-Z",
    "Tentang Kami", "Karier", "Hubungi Kami", "Tim Editorial", 
    "Langganan", "Syarat & Ketentuan", "Privasi", "Iklan", 
    "Gabung di Tim Dokter", "Daftarkan Rumah Sakit Anda", "alomedika.com",
    "Tanya Dokter", "Pilih", "Mohon tunggu", "Pengiriman SMS",
    "Hindari menggunakan kombinasi", "Info Kesehatan", "Cari Dokter",
    "Alodokter Shop", "Virus", "Kanker", "Jantung", "Otak",
    "Psikologi", "Defisiensi", "Infeksi", "Mata", "Pencernaan", "Semua Penyakit",
  ];

  function isBlacklisted(text) {
    return blacklistKeywords.some(b => text.includes(b));
  }

  for (let i = checkpoint.lastIndex; i < links.length; i++) {
    const link = links[i];
    console.log(`Scraping ${link.name}: ${link.href}`);

    try {
      await page.goto(link.href, { waitUntil: "networkidle" });

      // Ambil semua <p> dan <li> dalam urutan muncul di DOM
      const elements = await page.$$eval("p, li", (els) =>
        els.map(el => ({
          tag: el.tagName.toLowerCase(),
          text: el.innerText.trim(),
        }))
      );

      // Filter blacklist
      const filtered = elements.filter(el => el.text && !isBlacklisted(el.text));

      // Sekarang gabungkan sesuai aturan:
      // Jika <p> berakhir dengan ':', gabung semua <li> berikutnya ke paragraf tsb sampai ketemu <p> baru
      const paragraphs = [];
      let iEl = 0;
      while (iEl < filtered.length) {
        const el = filtered[iEl];
        if (el.tag === "p") {
          let paragraph = el.text;
          iEl++;

          // Jika p berakhir dengan ':', gabungkan semua li setelahnya
          if (paragraph.endsWith(":")) {
            while (iEl < filtered.length) {
              const nextEl = filtered[iEl];
              if (nextEl.tag === "li") {
                paragraph += " " + nextEl.text;
                iEl++;
              } else if (nextEl.tag === "p") {
                // ketemu p baru, berhenti gabung
                break;
              } else {
                // jika ada tag lain, skip saja atau berhenti
                iEl++;
              }
            }
          }
          paragraphs.push(paragraph);
        } else {
          // kalau li berdiri sendiri tanpa p sebelumnya (jarang), bisa dimasukkan juga
          paragraphs.push(el.text);
          iEl++;
        }
      }

      penyakitData.push({
        name: link.name,
        href: link.href,
        paragraphs,
      });

      checkpoint = {
        lastIndex: i + 1,
        penyakitData,
      };

      fs.writeFileSync(checkpointFile, JSON.stringify(checkpoint, null, 2));

      await delay(2000);
    } catch (err) {
      console.error(`Error saat scraping ${link.href}:`, err);
    }
  }

  fs.writeFileSync("./data/penyakit-data-final.json", JSON.stringify(penyakitData, null, 2));
  await browser.close();
})();
