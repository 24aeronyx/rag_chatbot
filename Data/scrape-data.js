const { chromium } = require("playwright");
const fs = require("fs");

const links = JSON.parse(fs.readFileSync("Data/penyakit-links.json"));
const checkpointFile = "Data/checkpoint.json";
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

  const blacklistKeywords = [
    "Beranda",
    "Chat Bersama Dokter",
    "Penyakit A-Z",
    "Obat A-Z",
    "Tentang Kami",
    "Karier",
    "Hubungi Kami",
    "Tim Editorial",
    "Langganan",
    "Syarat & Ketentuan",
    "Privasi",
    "Iklan",
    "Gabung di Tim Dokter",
    "Daftarkan Rumah Sakit Anda",
    "alomedika.com",
    "Tanya Dokter",
    "Pilih",
    "Mohon tunggu",
    "Pengiriman SMS",
    "Hindari menggunakan kombinasi",
    "Info Kesehatan",
    "Cari Dokter",
    "Alodokter Shop",
    "Virus",
    "Kanker",
    "Jantung",
    "Otak",
    "Psikologi",
    "Defisiensi",
    "Infeksi",
    "Mata",
    "Pencernaan",
    "Semua Penyakit",
  ];

  function isBlacklisted(text) {
    return blacklistKeywords.some((b) => text.includes(b));
  }

  for (let i = checkpoint.lastIndex; i < links.length; i++) {
    const link = links[i];
    console.log(`Scraping ${link.name}: ${link.href}`);

    try {
      await page.goto(link.href, { waitUntil: "networkidle" });

      // Ambil semua <p> dan <li> sebagai objek {tag, text}
      const elements = await page.$$eval("p, li", (els) =>
        els
          .map((el) => ({
            tag: el.tagName.toLowerCase(),
            text: el.innerText.trim(),
          }))
          .filter((el) => el.text.length > 0)
      );

      // Filter blacklist
      const filtered = elements.filter((el) => !isBlacklisted(el.text));

      const paragraphs = [];
      let lastParagraph = "";
      let listBuffer = [];

      for (const el of filtered) {
        if (el.tag === "p") {
          // Jika sebelumnya ada <li>, gabungkan ke lastParagraph
          if (listBuffer.length > 0) {
            if (lastParagraph.endsWith(":")) {
              paragraphs[
                paragraphs.length - 1
              ] = `${lastParagraph} ${listBuffer.join("; ")}`;
            } else {
              paragraphs.push(listBuffer.join("; "));
            }
            listBuffer = [];
          }
          lastParagraph = el.text;
          paragraphs.push(lastParagraph);
        } else if (el.tag === "li") {
          listBuffer.push(el.text);
        }
      }

      // Tangani sisa <li> di akhir
      if (listBuffer.length > 0) {
        if (lastParagraph.endsWith(":")) {
          paragraphs[
            paragraphs.length - 1
          ] = `${lastParagraph} ${listBuffer.join("; ")}`;
        } else {
          paragraphs.push(listBuffer.join("; "));
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

  fs.writeFileSync(
    "Data/penyakit-data-raw.json",
    JSON.stringify(penyakitData, null, 2)
  );
  await browser.close();
})();
