// GroundZero AI Dashboard - Compact

// Theme
function toggleTheme() {
    document.body.classList.toggle('light-mode');
    localStorage.setItem('theme', document.body.classList.contains('light-mode') ? 'light' : 'dark');
}
if (localStorage.getItem('theme') === 'light') document.body.classList.add('light-mode');

// Modals
function openModal(id) {
    document.getElementById('modal-' + id).classList.add('active');
}
function closeModal(id) {
    document.getElementById('modal-' + id).classList.remove('active');
}
document.querySelectorAll('.modal').forEach(m => {
    m.addEventListener('click', e => { if (e.target === m) m.classList.remove('active'); });
});
document.addEventListener('keydown', e => {
    if (e.key === 'Escape') document.querySelectorAll('.modal.active').forEach(m => m.classList.remove('active'));
});

// Milestones
const MILESTONES = [
    { level: 1, name: "Pattern Recognition", facts: 100, causal: 10 },
    { level: 2, name: "Knowledge Building", facts: 1000, causal: 100 },
    { level: 3, name: "Causal Understanding", facts: 5000, causal: 500 },
    { level: 4, name: "Reasoning Chains", facts: 20000, causal: 2000 },
    { level: 5, name: "Deep Understanding", facts: 100000, causal: 10000 },
    { level: 6, name: "Human-Like", facts: 500000, causal: 50000 }
];

// Stats (from localStorage or default)
let stats = JSON.parse(localStorage.getItem('gz_stats') || '{"facts":0,"causal":0,"questions":0,"confidence":70}');

function getCurrentLevel() {
    for (let i = MILESTONES.length - 1; i >= 0; i--) {
        if (stats.facts >= MILESTONES[i].facts && stats.causal >= MILESTONES[i].causal) return i + 1;
    }
    return 0;
}

function getProgress() {
    const lvl = getCurrentLevel();
    if (lvl >= MILESTONES.length) return 100;
    const next = MILESTONES[lvl];
    const fp = Math.min(stats.facts / next.facts, 1);
    const cp = Math.min(stats.causal / next.causal, 1);
    return Math.round((fp + cp) / 2 * 100);
}

function update() {
    const lvl = getCurrentLevel();
    const prog = getProgress();
    
    document.getElementById('level').textContent = lvl;
    document.getElementById('facts').textContent = stats.facts.toLocaleString();
    document.getElementById('causal').textContent = stats.causal.toLocaleString();
    document.getElementById('questions').textContent = stats.questions.toLocaleString();
    document.getElementById('confidence').textContent = stats.confidence + '%';
    
    document.getElementById('level-badge').textContent = 'LEVEL ' + lvl;
    document.getElementById('level-name').textContent = lvl > 0 ? MILESTONES[lvl-1].name : 'Starting Out';
    document.getElementById('progress-bar').style.width = prog + '%';
    document.getElementById('progress-pct').textContent = prog + '%';
    
    // Timeline
    const tl = document.getElementById('timeline');
    tl.innerHTML = MILESTONES.map((m, i) => {
        const status = (i + 1) < lvl ? 'done' : (i + 1) === lvl ? 'current' : '';
        return `<div class="tl-item ${status}">
            <div class="tl-dot"></div>
            <div class="tl-info">
                <div class="tl-name">L${m.level}: ${m.name}</div>
                <div class="tl-req">${m.facts.toLocaleString()} facts</div>
            </div>
        </div>`;
    }).join('');
}

// Simulate learning (for demo)
function simulate() {
    if (Math.random() > 0.6) stats.facts += Math.floor(Math.random() * 3) + 1;
    if (Math.random() > 0.8) stats.causal += 1;
    if (Math.random() > 0.7) stats.questions += 1;
    stats.confidence = 65 + Math.floor(Math.random() * 25);
    localStorage.setItem('gz_stats', JSON.stringify(stats));
    update();
}

update();
setInterval(simulate, 3000);