const fs = require('fs');

// Baca data JSON
const data = JSON.parse(fs.readFileSync('./data/penyakit-links.json', 'utf-8'));

// Template pertanyaan
const templates = [
  "Apa itu [NAME]?",
  "Apa penyebab dari [NAME]?",
  "Apa saja gejala [NAME]?",
  "Bagaimana cara mengobati [NAME]?",
  "Apakah [NAME] berbahaya?",
  "Bagaimana diagnosis untuk [NAME]?",
  "Apa komplikasi yang mungkin terjadi akibat [NAME]?"
];

// Fungsi untuk merandom elemen dari array
function getRandomElement(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

// Fungsi untuk membuat pertanyaan dari name
function generateQuestion(name) {
  const template = getRandomElement(templates);
  return template.replace('[NAME]', name);
}

// Ambil 25 penyakit unik secara acak
const shuffled = data.sort(() => 0.5 - Math.random());
const selected = shuffled.slice(0, 25);

// Generate pertanyaan
const questions = selected.map(item => {
  return {
    question: generateQuestion(item.name),
    ground_truth: item.name
  };
});

// Simpan ke file JSON
fs.writeFileSync('./data/generated-questions.json', JSON.stringify(questions, null, 2), 'utf-8');

console.log("✅ 25 pertanyaan berhasil disimpan ke generated-questions.json");
