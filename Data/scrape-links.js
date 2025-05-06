const { chromium } = require('playwright');
const fs = require('fs');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('https://www.alodokter.com/penyakit-a-z', { waitUntil: 'networkidle' });

  // Ambil semua href dari list
  const links = await page.$$eval('ul > li > a', (elements) =>
    elements.map((el) => ({
      name: el.innerText,  
      href: el.href,       
    }))
  );

  console.log(links);  

  fs.writeFileSync('penyakit-links.json', JSON.stringify(links, null, 2));

  await browser.close();
})();
