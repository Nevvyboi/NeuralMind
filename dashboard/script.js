// GroundZero AI Dashboard - Fixed Version

const API_BASE = '/api';
const POLL_INTERVAL = 2000;

let lossChart = null;
let lastUpdateTime = null;

// =============================================================================
// THEME SYSTEM
// =============================================================================

function toggleMode() {
    const isLight = document.body.classList.toggle('light-mode');
    document.getElementById('modeToggle').textContent = isLight ? '☀️' : '🌙';
    localStorage.setItem('gz-mode', isLight ? 'light' : 'dark');
    updateChartColors();
}

function setTheme(theme) {
    document.body.classList.remove('theme-blue', 'theme-green', 'theme-purple', 'theme-pink', 'theme-orange', 'theme-red');
    document.body.classList.add('theme-' + theme);
    document.getElementById('themeDropdown').value = theme;
    localStorage.setItem('gz-theme', theme);
    updateChartColors();
}

function loadSavedTheme() {
    const mode = localStorage.getItem('gz-mode') || 'dark';
    if (mode === 'light') {
        document.body.classList.add('light-mode');
        document.getElementById('modeToggle').textContent = '☀️';
    }
    
    const theme = localStorage.getItem('gz-theme') || 'blue';
    setTheme(theme);
}

function updateChartColors() {
    if (!lossChart) return;
    const style = getComputedStyle(document.documentElement);
    lossChart.data.datasets[0].borderColor = style.getPropertyValue('--accent').trim();
    lossChart.update('none');
}

// =============================================================================
// MODALS
// =============================================================================

function openModal(id) {
    document.getElementById('modal-' + id).classList.add('active');
    if (id === 'neural-graph') setTimeout(drawNeuralGraph, 100);
}

function closeModal(id) {
    document.getElementById('modal-' + id).classList.remove('active');
}

document.addEventListener('click', e => {
    if (e.target.classList.contains('modal')) e.target.classList.remove('active');
});

document.addEventListener('keydown', e => {
    if (e.key === 'Escape') document.querySelectorAll('.modal.active').forEach(m => m.classList.remove('active'));
});

// =============================================================================
// MILESTONES & STATS
// =============================================================================

const MILESTONES = [
    { level: 1, name: "Pattern Recognition", facts: 100 },
    { level: 2, name: "Knowledge Building", facts: 1000 },
    { level: 3, name: "Causal Understanding", facts: 5000 },
    { level: 4, name: "Reasoning Chains", facts: 20000 },
    { level: 5, name: "Deep Understanding", facts: 100000 },
    { level: 6, name: "Human-Like", facts: 500000 }
];

let stats = {
    facts: 0, entities: 0, causal: 0, confidence: 70,
    neural: { epochs: 0, loss: null, embedDim: 0, predictions: 0, hypotheses: 0, isTrained: false, entities: 0, relations: 0, triples: 0 },
    lossHistory: []
};

function getCurrentLevel() {
    for (let i = MILESTONES.length - 1; i >= 0; i--) {
        if (stats.facts >= MILESTONES[i].facts) return i + 1;
    }
    return 0;
}

function getProgress() {
    const lvl = getCurrentLevel();
    if (lvl >= MILESTONES.length) return 100;
    const next = MILESTONES[lvl];
    return Math.round((stats.facts / next.facts) * 100);
}

// =============================================================================
// UI UPDATE
// =============================================================================

function updateUI() {
    const lvl = getCurrentLevel();
    const prog = getProgress();
    
    setText('level', lvl);
    setText('facts', formatNum(stats.facts));
    setText('entities', formatNum(stats.entities));
    setText('causal', formatNum(stats.causal));
    setText('epochs', stats.neural.epochs);
    setText('loss', stats.neural.loss ? stats.neural.loss.toFixed(4) : '-');
    setText('confidence', Math.round(stats.confidence) + '%');
    
    setText('level-badge', 'LEVEL ' + lvl);
    setText('level-name', lvl > 0 ? MILESTONES[lvl-1].name : 'Starting Out');
    document.getElementById('progress-bar').style.width = prog + '%';
    setText('progress-pct', prog + '%');
    
    setText('embedDim', stats.neural.embedDim);
    setText('trainEpochs', stats.neural.epochs);
    setText('lossVal', stats.neural.loss ? stats.neural.loss.toFixed(4) : '-');
    setText('predictions', stats.neural.predictions);
    setText('hypotheses', stats.neural.hypotheses);
    
    setBar('embedBar', stats.neural.embedDim, 200);
    setBar('trainBar', stats.neural.epochs, 500);
    setBar('lossBar', stats.neural.loss ? 100 - (stats.neural.loss * 100) : 0, 100);
    
    const badge = document.getElementById('neuralBadge');
    if (badge) {
        if (stats.neural.isTrained) { badge.textContent = 'Trained ✓'; badge.className = 'badge trained'; }
        else if (stats.neural.epochs > 0) { badge.textContent = 'Training...'; badge.className = 'badge training'; }
        else { badge.textContent = 'Not Trained'; badge.className = 'badge'; }
    }
    
    updateCapabilities();
    updateTimeline(lvl);
    updateLossChart();
    
    setText('gsEntities', stats.neural.entities);
    setText('gsRelations', stats.neural.relations);
    setText('gsTriples', stats.neural.triples);
    setText('gsEmbedDim', stats.neural.embedDim);
}

function formatNum(n) {
    if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
    if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
    return n.toLocaleString();
}

function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}

function setBar(id, val, max) {
    const el = document.getElementById(id);
    if (el) el.style.width = Math.min(100, (val / max) * 100) + '%';
}

function updateCapabilities() {
    [['capUnderstand', stats.neural.entities > 0],
     ['capGeneralize', stats.neural.isTrained],
     ['capLearn', stats.neural.epochs > 0],
     ['capReason', stats.neural.predictions > 0],
     ['capCreate', stats.neural.hypotheses > 0]
    ].forEach(([id, active]) => {
        const el = document.getElementById(id);
        if (el) el.classList.toggle('active', active);
    });
}

function updateTimeline(lvl) {
    const tl = document.getElementById('timeline');
    if (!tl) return;
    tl.innerHTML = MILESTONES.map((m, i) => {
        const status = (i + 1) < lvl ? 'done' : (i + 1) === lvl ? 'current' : '';
        return `<div class="tl-item ${status}">
            <div class="tl-dot"></div>
            <div class="tl-info">
                <div class="tl-name">L${m.level}: ${m.name}</div>
                <div class="tl-req">${formatNum(m.facts)} facts</div>
            </div>
        </div>`;
    }).join('');
}

// =============================================================================
// LAST UPDATED
// =============================================================================

function updateLastUpdated() {
    lastUpdateTime = new Date();
    setText('lastUpdated', formatTimeAgo(lastUpdateTime));
}

function formatTimeAgo(date) {
    const s = Math.floor((new Date() - date) / 1000);
    if (s < 5) return 'just now';
    if (s < 60) return s + 's ago';
    if (s < 3600) return Math.floor(s / 60) + 'm ago';
    return Math.floor(s / 3600) + 'h ago';
}

setInterval(() => {
    if (lastUpdateTime) setText('lastUpdated', formatTimeAgo(lastUpdateTime));
}, 1000);

// =============================================================================
// LOSS CHART
// =============================================================================

function initLossChart() {
    const ctx = document.getElementById('lossChart');
    if (!ctx) return;
    
    const style = getComputedStyle(document.documentElement);
    
    lossChart = new Chart(ctx.getContext('2d'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                data: [],
                borderColor: style.getPropertyValue('--accent').trim(),
                backgroundColor: 'transparent',
                borderWidth: 2,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: { 
                    display: true,
                    grid: { color: 'rgba(128,128,128,0.1)' },
                    ticks: { font: { size: 11 } }
                }
            }
        }
    });
}

function updateLossChart() {
    if (!lossChart || !stats.lossHistory.length) return;
    lossChart.data.labels = stats.lossHistory.map((_, i) => i);
    lossChart.data.datasets[0].data = stats.lossHistory;
    lossChart.update('none');
}

// =============================================================================
// NEURAL GRAPH
// =============================================================================

function drawNeuralGraph() {
    const canvas = document.getElementById('neuralGraphCanvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    
    const style = getComputedStyle(document.documentElement);
    const accent = style.getPropertyValue('--accent').trim();
    const accent2 = style.getPropertyValue('--accent2').trim();
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const count = Math.min(stats.neural.entities || 20, 40);
    const nodes = [];
    const cx = canvas.width / 2, cy = canvas.height / 2;
    const r = Math.min(canvas.width, canvas.height) * 0.35;
    
    for (let i = 0; i < count; i++) {
        const angle = (i / count) * Math.PI * 2;
        const dist = r * (0.6 + Math.random() * 0.4);
        nodes.push({ x: cx + Math.cos(angle) * dist, y: cy + Math.sin(angle) * dist, type: Math.random() > 0.75 });
    }
    
    ctx.strokeStyle = 'rgba(128,128,128,0.15)';
    ctx.lineWidth = 1;
    nodes.forEach((n, i) => {
        for (let j = 0; j < 2; j++) {
            const t = Math.floor(Math.random() * nodes.length);
            if (t !== i) { ctx.beginPath(); ctx.moveTo(n.x, n.y); ctx.lineTo(nodes[t].x, nodes[t].y); ctx.stroke(); }
        }
    });
    
    nodes.forEach(n => {
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.type ? 5 : 7, 0, Math.PI * 2);
        ctx.fillStyle = n.type ? accent2 : accent;
        ctx.fill();
    });
    
    ctx.beginPath();
    ctx.arc(cx, cy, 12, 0, Math.PI * 2);
    ctx.fillStyle = accent;
    ctx.fill();
}

function refreshGraph() { drawNeuralGraph(); }

// =============================================================================
// API FUNCTIONS
// =============================================================================

async function trainNeural() {
    const btn = event.target;
    btn.textContent = '⏳ Training...';
    btn.disabled = true;
    try {
        await fetch(`${API_BASE}/neural/train`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ epochs: 50 }) });
        btn.textContent = '✓ Started!';
        setTimeout(() => { btn.textContent = '🧠 Train Now'; btn.disabled = false; }, 2000);
    } catch (e) {
        btn.textContent = '🧠 Train Now';
        btn.disabled = false;
        alert('Start server: python main.py dashboard');
    }
}

async function generateHypotheses() {
    const list = document.getElementById('hypothesesList');
    list.innerHTML = '<div class="hypothesis">Loading...</div>';
    try {
        const res = await fetch(`${API_BASE}/neural/hypotheses`);
        const data = await res.json();
        list.innerHTML = data.hypotheses?.length 
            ? data.hypotheses.map(h => `<div class="hypothesis"><span>${h.Head} → ${h.Relation} → ${h.Tail}</span><span class="conf">${Math.round(h.Confidence * 100)}%</span></div>`).join('')
            : '<div class="hypothesis">Train the network first!</div>';
    } catch (e) { list.innerHTML = '<div class="hypothesis">Server offline</div>'; }
}

// =============================================================================
// CHAT
// =============================================================================

function initChat() {
    const input = document.getElementById('chatInput');
    const btn = document.getElementById('sendBtn');
    if (!input || !btn) return;
    btn.addEventListener('click', sendMessage);
    input.addEventListener('keypress', e => { if (e.key === 'Enter') sendMessage(); });
}

async function sendMessage() {
    const input = document.getElementById('chatInput');
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    addMessage(text, 'user');
    try {
        const res = await fetch(`${API_BASE}/chat`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: text }) });
        const data = await res.json();
        addMessage(data.answer, 'bot');
        stats.confidence = data.confidence * 100;
    } catch (e) { addMessage("Offline. Start: python main.py dashboard", 'bot'); }
}

function addMessage(text, sender) {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = `msg ${sender}`;
    div.textContent = sender === 'bot' ? '🧠 ' + text : text;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

// =============================================================================
// API POLLING
// =============================================================================

async function fetchStats() {
    try {
        const res = await fetch(`${API_BASE}/stats`);
        const data = await res.json();
        
        stats.facts = data.Knowledge?.TotalFacts || stats.facts;
        stats.causal = data.Causal?.TotalRelations || stats.causal;
        stats.confidence = (data.Chat?.AverageConfidence || 0.7) * 100;
        
        if (data.Neural) {
            stats.neural = {
                epochs: data.Neural.TrainingEpochs || 0,
                loss: data.Neural.LastLoss,
                embedDim: data.Neural.EmbeddingDim || 0,
                predictions: data.Neural.Predictions || 0,
                hypotheses: data.Neural.Hypotheses || 0,
                isTrained: data.Neural.IsTrained || false,
                entities: data.Neural.TotalEntities || 0,
                relations: data.Neural.TotalRelations || 0,
                triples: data.Neural.TotalTriples || 0
            };
            stats.entities = data.Neural.TotalEntities || 0;
        }
        
        if (data.LossHistory) stats.lossHistory = data.LossHistory;
        
        setStatus('connected', 'Live');
        updateLastUpdated();
    } catch (e) {
        setStatus('error', 'Offline');
    }
    updateUI();
}

function setStatus(state, text) {
    const dot = document.getElementById('statusDot');
    const txt = document.getElementById('statusText');
    if (dot) dot.className = 'live-dot ' + state;
    if (txt) txt.textContent = text;
}

// =============================================================================
// INIT
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    loadSavedTheme();
    initLossChart();
    initChat();
    updateUI();
    updateLastUpdated();
    fetchStats();
    setInterval(fetchStats, POLL_INTERVAL);
    console.log('🧠 GroundZero AI Dashboard loaded');
});