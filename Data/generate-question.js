const fs = require('fs');
const path = require('path');
const seedrandom = require('seedrandom');

// ===== KONFIGURASI =====
const inputPath = path.join(__dirname, 'penyakit-links.json');
const outputPath = path.join(__dirname, 'generated-questions.json');
const jumlahPertanyaan = 25;
const randomSeed = 42; // seed tetap untuk hasil acak yang konsisten

// ===== TEMPLATE PERTANYAAN =====
const templates = [
  "Apa penyebab dari {name}?",
  "Bagaimana cara mengobati {name}?",
  "Apa saja gejala {name}?",
  "Apakah {name} berbahaya?",
  "Bagaimana cara mendiagnosis penyakit {name}?",
  "Apa komplikasi yang mungkin terjadi akibat {name}?",
  "Apa itu {name}?"
];

// ===== LOGIKA UTAMA =====
function generateQuestions(data, seed) {
  const rng = seedrandom(seed);
  const selected = data.slice().sort(() => rng() - 0.5).slice(0, jumlahPertanyaan);
  
  return selected.map(entry => {
    const template = templates[Math.floor(rng() * templates.length)];
    const question = template.replace("{name}", entry.name);
    return {
      question
    };
  });
}


// ===== JALANKAN =====
const raw = fs.readFileSync(inputPath, 'utf8');
const penyakitList = JSON.parse(raw);

const result = generateQuestions(penyakitList, randomSeed);
fs.writeFileSync(outputPath, JSON.stringify(result, null, 2));

console.log(`✅ Berhasil generate ${jumlahPertanyaan} pertanyaan ke: ${outputPath}`);
