const { chromium } = require('playwright');
const fs = require('fs');

const links = JSON.parse(fs.readFileSync('penyakit-links.json'));

const checkpointFile = 'checkpoint.json';
let checkpoint = { lastIndex: 0, penyakitData: [] };

if (fs.existsSync(checkpointFile)) {
  checkpoint = JSON.parse(fs.readFileSync(checkpointFile));
  console.log(`Melanjutkan dari indeks: ${checkpoint.lastIndex}`);
}

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  const penyakitData = checkpoint.penyakitData; 
  for (let i = checkpoint.lastIndex; i < links.length; i++) {
    const link = links[i];
    console.log(`Scraping ${link.name}: ${link.href}`);
    const blacklist = [
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
      "alomedika.com"
    ];
    try {
      await page.goto(link.href, { waitUntil: 'networkidle' });

      const paragraphs = await page.$$eval(
        'p',
        (elements, blacklist) =>
          elements
            .map((el) => el.innerText.trim())
            .filter((text) => text.length > 0 && !blacklist.includes(text)),
        blacklist 
      );
    
      penyakitData.push({
        name: link.name,
        href: link.href,
        paragraphs: paragraphs,
      });

      checkpoint = {
        lastIndex: i + 1,  
        penyakitData: penyakitData,
      };
      fs.writeFileSync(checkpointFile, JSON.stringify(checkpoint, null, 2));

      await delay(5000);
    } catch (error) {
      console.log(`Error pada ${link.href}: ${error.message}`);
    }
  }

  fs.writeFileSync('penyakit-data.json', JSON.stringify(penyakitData, null, 2));

  await browser.close();
})();
