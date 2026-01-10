/**
 * GroundZero AI - User Interface v2.6
 * Complete with Learning Dashboard, Voice Recording, and all features
 */

const API = {
    chat: '/api/chat',
    stats: '/api/stats',
    search: '/api/search',
    timeline: '/api/timeline',
    learningStart: '/api/learning/start',
    learningStop: '/api/learning/stop',
    learningStats: '/api/learning/stats',
    learningRecent: '/api/learning/recent',
    learningSessions: '/api/learning/sessions',
    teach: '/api/teach',
    voiceTranscribe: '/api/voice/transcribe'
};

// ============================================================
// Theme Manager
// ============================================================
const ThemeManager = {
    init() {
        const saved = localStorage.getItem('theme') || 'dark';
        this.applyTheme(saved);
        
        const toggle = document.getElementById('theme-toggle');
        if (toggle) {
            toggle.checked = (saved === 'dark');
            toggle.addEventListener('change', (e) => {
                this.applyTheme(e.target.checked ? 'dark' : 'light');
            });
        }
    },
    
    applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        document.body.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);
    }
};

// ============================================================
// Voice Recorder
// ============================================================
class VoiceRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.stream = null;
        this.isRecording = false;
        this.startTime = null;
        this.timerInterval = null;
    }
    
    init() {
        const voiceBtn = document.getElementById('voice-btn');
        if (!voiceBtn) return;
        
        voiceBtn.addEventListener('mousedown', (e) => {
            e.preventDefault();
            this.startRecording();
        });
        
        document.addEventListener('mouseup', () => {
            if (this.isRecording) this.stopRecording();
        });
        
        voiceBtn.addEventListener('touchstart', (e) => {
            e.preventDefault();
            this.startRecording();
        });
        
        document.addEventListener('touchend', () => {
            if (this.isRecording) this.stopRecording();
        });
        
        document.getElementById('voice-cancel')?.addEventListener('click', () => this.cancelRecording());
    }
    
    async startRecording() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.audioChunks = [];
            const mimeType = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/ogg';
            this.mediaRecorder = new MediaRecorder(this.stream, { mimeType });
            
            this.mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) this.audioChunks.push(e.data);
            };
            
            this.mediaRecorder.start(100);
            this.isRecording = true;
            this.startTime = Date.now();
            
            document.getElementById('voice-btn')?.classList.add('recording');
            document.getElementById('voice-status-bar')?.classList.remove('hidden');
            
            this.timerInterval = setInterval(() => {
                const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
                const mins = Math.floor(elapsed / 60);
                const secs = elapsed % 60;
                const timerEl = document.getElementById('voice-timer');
                if (timerEl) timerEl.textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
            }, 100);
        } catch (err) {
            showToast('Microphone access denied', 'error');
        }
    }
    
    async stopRecording() {
        if (!this.isRecording) return;
        
        return new Promise((resolve) => {
            this.mediaRecorder.onstop = async () => {
                const blob = new Blob(this.audioChunks, { type: this.mediaRecorder.mimeType });
                this.cleanup();
                
                if (blob.size > 1000) {
                    const msgEl = document.getElementById('voice-message');
                    if (msgEl) msgEl.textContent = '⏳ Transcribing...';
                    const text = await this.transcribe(blob);
                    document.getElementById('voice-status-bar')?.classList.add('hidden');
                    
                    if (text && text.trim()) {
                        const input = document.getElementById('chat-input');
                        if (input) {
                            input.value = text;
                            window.chatInterface?.sendMessage();
                        }
                    }
                } else {
                    document.getElementById('voice-status-bar')?.classList.add('hidden');
                }
                resolve();
            };
            this.mediaRecorder.stop();
        });
    }
    
    cancelRecording() {
        this.cleanup();
        document.getElementById('voice-status-bar')?.classList.add('hidden');
    }
    
    cleanup() {
        if (this.stream) this.stream.getTracks().forEach(t => t.stop());
        this.isRecording = false;
        document.getElementById('voice-btn')?.classList.remove('recording');
        if (this.timerInterval) clearInterval(this.timerInterval);
    }
    
    async transcribe(blob) {
        const formData = new FormData();
        formData.append('audio', blob, 'recording.webm');
        try {
            const res = await fetch(API.voiceTranscribe, { method: 'POST', body: formData });
            const data = await res.json();
            return data.text || '';
        } catch { return ''; }
    }
}

// ============================================================
// Learning Dashboard - DETAILED REAL-TIME PROGRESS
// ============================================================
class LearningDashboard {
    constructor() {
        this.isLearning = false;
        this.pollInterval = null;
        this.timerInterval = null;
        this.startTime = null;
        this.progressFrame = null;
        this.serverStartTime = null;
    }
    
    init() {
        document.getElementById('btn-start-learning')?.addEventListener('click', () => this.start());
        document.getElementById('btn-stop-learning')?.addEventListener('click', () => this.stop());
        document.getElementById('btn-refresh')?.addEventListener('click', () => this.refresh());
        document.getElementById('btn-teach')?.addEventListener('click', () => this.teach());
        
        this.loadStats();
        this.loadArticles();
        this.loadSessions();
        this.checkStatus();
    }
    
    async start() {
        const startBtn = document.getElementById('btn-start-learning');
        if (!startBtn) return;
        
        startBtn.disabled = true;
        startBtn.innerHTML = '⏳ Starting...';
        
        try {
            const res = await fetch(API.learningStart, { method: 'POST' });
            if (res.ok) {
                this.setState(true);
                showToast('Learning started!', 'success');
            } else {
                throw new Error('Failed');
            }
        } catch (err) {
            showToast('Failed to start learning', 'error');
            startBtn.disabled = false;
            startBtn.innerHTML = '▶️ Start Learning';
        }
    }
    
    async stop() {
        try {
            await fetch(API.learningStop, { method: 'POST' });
            this.setState(false);
            showToast('Learning stopped', 'info');
            this.refresh();
        } catch {
            showToast('Failed to stop', 'error');
        }
    }
    
    setState(learning) {
        this.isLearning = learning;
        
        const startBtn = document.getElementById('btn-start-learning');
        const stopBtn = document.getElementById('btn-stop-learning');
        const progress = document.getElementById('learning-progress');
        const badge = document.getElementById('learning-status-badge');
        const indicator = document.getElementById('learning-indicator');
        
        if (learning) {
            startBtn?.classList.add('hidden');
            stopBtn?.classList.remove('hidden');
            progress?.classList.remove('hidden');
            if (badge) { badge.textContent = '🟢 Learning'; badge.classList.add('active'); }
            indicator?.classList.remove('hidden');
            
            this.startTime = Date.now();
            this.timerInterval = setInterval(() => this.updateTimer(), 1000);
            this.animateProgress();
            this.pollInterval = setInterval(() => this.poll(), 1500); // Poll faster for responsiveness
        } else {
            startBtn?.classList.remove('hidden');
            if (startBtn) { startBtn.disabled = false; startBtn.innerHTML = '▶️ Start Learning'; }
            stopBtn?.classList.add('hidden');
            progress?.classList.add('hidden');
            if (badge) { badge.textContent = 'Idle'; badge.classList.remove('active'); }
            indicator?.classList.add('hidden');
            
            // Clear current article display
            const currentEl = document.getElementById('current-article');
            if (currentEl) currentEl.textContent = 'Stopped';
            
            if (this.timerInterval) clearInterval(this.timerInterval);
            if (this.pollInterval) clearInterval(this.pollInterval);
            if (this.progressFrame) cancelAnimationFrame(this.progressFrame);
            this.timerInterval = null;
            this.pollInterval = null;
            this.progressFrame = null;
        }
    }
    
    updateTimer() {
        // Use server time if available, otherwise local
        let elapsed;
        if (this.serverStartTime) {
            elapsed = Math.floor((Date.now() - this.serverStartTime) / 1000);
        } else if (this.startTime) {
            elapsed = Math.floor((Date.now() - this.startTime) / 1000);
        } else {
            return;
        }
        
        const hrs = Math.floor(elapsed / 3600);
        const mins = Math.floor((elapsed % 3600) / 60);
        const secs = elapsed % 60;
        
        let timeStr;
        if (hrs > 0) {
            timeStr = `${hrs}:${mins.toString().padStart(2,'0')}:${secs.toString().padStart(2,'0')}`;
        } else {
            timeStr = `${mins.toString().padStart(2,'0')}:${secs.toString().padStart(2,'0')}`;
        }
        
        const el = document.getElementById('learning-timer');
        if (el) el.textContent = timeStr;
    }
    
    animateProgress() {
        let width = 0;
        const bar = document.getElementById('progress-bar');
        const animate = () => {
            if (!this.isLearning) return;
            width = (width + 0.3) % 100;
            if (bar) bar.style.width = width + '%';
            this.progressFrame = requestAnimationFrame(animate);
        };
        animate();
    }
    
    async poll() {
        try {
            const res = await fetch(API.learningStats);
            const data = await res.json();
            
            // Update current article with better display
            const currentEl = document.getElementById('current-article');
            if (currentEl) {
                if (data.current_article) {
                    currentEl.textContent = data.current_article;
                    currentEl.title = data.current_url || data.current_article;
                } else if (data.is_running) {
                    currentEl.textContent = 'Fetching next article...';
                }
            }
            
            // Update session stats display
            const sessionStats = document.getElementById('session-stats');
            if (sessionStats && data.is_running) {
                const arts = data.articles_this_session || 0;
                const words = data.words_this_session || 0;
                sessionStats.innerHTML = `
                    <span>📖 ${arts} articles this session</span>
                    <span>📝 ${this.fmt(words)} words this session</span>
                `;
            }
            
            // Sync timer with server
            if (data.start_time) {
                this.serverStartTime = new Date(data.start_time).getTime();
            } else if (data.elapsed_seconds && data.is_running) {
                this.serverStartTime = Date.now() - (data.elapsed_seconds * 1000);
            }
            
            // Update all stats
            this.updateDisplay(data);
            
            // Check if stopped externally
            if (data.is_running === false && this.isLearning) {
                this.setState(false);
                this.refresh();
            }
        } catch (err) {
            console.error('Poll error:', err);
        }
    }
    
    async checkStatus() {
        try {
            const res = await fetch(API.learningStats);
            const data = await res.json();
            
            if (data.is_running) {
                // Sync timer with server
                if (data.start_time) {
                    this.serverStartTime = new Date(data.start_time).getTime();
                } else if (data.elapsed_seconds) {
                    this.serverStartTime = Date.now() - (data.elapsed_seconds * 1000);
                }
                this.setState(true);
            }
            this.updateDisplay(data);
        } catch {}
    }
    
    updateDisplay(data) {
        this.setNum('stat-articles', data.articles_learned || 0);
        this.setNum('stat-words', data.tokens_trained || 0);
        this.setNum('stat-facts', data.facts || 0);
        this.setNum('stat-vectors', data.vectors || 0);
        
        // Also update the big stats on stats page
        this.setNum('big-articles', data.articles_learned || 0);
        this.setNum('big-facts', data.facts || 0);
        this.setNum('big-vectors', data.vectors || 0);
    }
    
    setNum(id, val) {
        const el = document.getElementById(id);
        if (el) el.textContent = this.fmt(val);
    }
    
    fmt(n) {
        if (n >= 1e6) return (n/1e6).toFixed(1) + 'M';
        if (n >= 1e3) return (n/1e3).toFixed(1) + 'K';
        return n.toLocaleString();
    }
    
    async loadStats() {
        try {
            const res = await fetch(API.stats);
            const data = await res.json();
            this.setNum('stat-articles', data.articles_learned || 0);
            this.setNum('stat-words', data.tokens_trained || 0);
            this.setNum('stat-facts', data.facts || data.knowledge || 0);
            this.setNum('stat-vectors', data.vectors || 0);
            this.setNum('big-articles', data.articles_learned || 0);
            this.setNum('big-facts', data.facts || data.knowledge || 0);
            this.setNum('big-vectors', data.vectors || 0);
        } catch {}
    }
    
    async loadArticles() {
        const container = document.getElementById('recent-articles');
        if (!container) return;
        
        try {
            const res = await fetch(API.learningRecent + '?limit=10');
            const data = await res.json();
            
            if (data.articles?.length > 0) {
                container.innerHTML = data.articles.map(a => `
                    <div class="article-item">
                        <div class="article-title" title="${this.esc(a.url || '')}">${this.esc(a.title || 'Untitled')}</div>
                        <div class="article-meta">
                            <span>${(a.word_count||0).toLocaleString()} words</span>
                            <span>${this.fmtTime(a.learned_at)}</span>
                        </div>
                    </div>
                `).join('');
            } else {
                container.innerHTML = '<div class="empty-state"><span class="empty-icon">📚</span><p>No articles yet</p></div>';
            }
        } catch {
            container.innerHTML = '<p>Could not load</p>';
        }
    }
    
    async loadSessions() {
        const container = document.getElementById('learning-sessions');
        if (!container) return;
        
        try {
            const res = await fetch(API.learningSessions + '?limit=5');
            const data = await res.json();
            
            if (data.sessions?.length > 0) {
                container.innerHTML = data.sessions.map(s => `
                    <div class="session-item">
                        <div class="session-header">
                            <span class="session-date">${this.fmtDate(s.started_at)}</span>
                            <span class="session-status ${s.status||'completed'}">${(s.status||'COMPLETED').toUpperCase()}</span>
                        </div>
                        <div class="session-stats">
                            📖 ${s.articles_learned||0} articles · 📝 ${this.fmt(s.words_learned||0)} words · ⏱️ ${this.fmtDur(s.duration_seconds)}
                        </div>
                    </div>
                `).join('');
            } else {
                container.innerHTML = '<div class="empty-state"><span class="empty-icon">📈</span><p>No sessions</p></div>';
            }
        } catch {
            container.innerHTML = '<p>Could not load</p>';
        }
    }
    
    async teach() {
        const input = document.getElementById('teach-input');
        const text = input?.value?.trim();
        if (!text) { showToast('Enter knowledge to teach', 'error'); return; }
        
        try {
            const res = await fetch(API.teach, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ knowledge: text, source: 'user' })
            });
            if (res.ok) {
                showToast('Knowledge saved!', 'success');
                input.value = '';
                this.loadStats();
            }
        } catch { showToast('Failed', 'error'); }
    }
    
    refresh() { this.loadStats(); this.loadArticles(); this.loadSessions(); }
    fmtTime(s) { return s ? new Date(s).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'}) : ''; }
    fmtDate(s) { return s ? new Date(s).toLocaleDateString([],{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}) : ''; }
    fmtDur(s) { if(!s)return'0s'; if(s<60)return s+'s'; const m=Math.floor(s/60),sec=s%60; return m<60?`${m}m ${sec}s`:`${Math.floor(m/60)}h ${m%60}m`; }
    esc(s) { const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }
}

// ============================================================
// Chat Interface
// ============================================================
class ChatInterface {
    constructor() {
        this.messages = document.getElementById('chat-messages');
        this.input = document.getElementById('chat-input');
        
        document.getElementById('send-btn')?.addEventListener('click', () => this.sendMessage());
        document.getElementById('btn-clear-chat')?.addEventListener('click', () => this.clear());
        document.getElementById('btn-clear-all')?.addEventListener('click', () => this.clear());
        
        this.input?.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); this.sendMessage(); }
        });
    }
    
    async sendMessage() {
        const text = this.input?.value?.trim();
        if (!text) return;
        
        this.input.value = '';
        this.addMsg(text, 'user');
        const typing = this.addTyping();
        
        try {
            const res = await fetch(API.chat, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text })
            });
            const data = await res.json();
            typing.remove();
            this.addMsg(data.response || 'No response', 'assistant');
        } catch {
            typing.remove();
            this.addMsg('Error connecting', 'error');
        }
    }
    
    addMsg(content, type) {
        const div = document.createElement('div');
        div.className = `message ${type}-message`;
        const avatar = type==='user'?'👤':type==='error'?'❌':'🤖';
        div.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content"><p>${this.fmt(content)}</p></div>
        `;
        this.messages?.appendChild(div);
        if (this.messages) this.messages.scrollTop = this.messages.scrollHeight;
        return div;
    }
    
    addTyping() {
        const div = document.createElement('div');
        div.className = 'message assistant-message';
        div.innerHTML = '<div class="message-avatar">🤖</div><div class="message-content"><div class="typing-dots"><span></span><span></span><span></span></div></div>';
        this.messages?.appendChild(div);
        if (this.messages) this.messages.scrollTop = this.messages.scrollHeight;
        return div;
    }
    
    fmt(t) { return t.replace(/```([\s\S]*?)```/g,'<pre><code>$1</code></pre>').replace(/`([^`]+)`/g,'<code>$1</code>').replace(/\n/g,'<br>'); }
    clear() { if(this.messages) this.messages.innerHTML = '<div class="message system"><div class="message-content"><p>💬 Cleared</p></div></div>'; }
}

// ============================================================
// Knowledge Explorer
// ============================================================
class KnowledgeExplorer {
    constructor() {
        this.container = document.getElementById('knowledge-container') || document.getElementById('search-results');
        this.input = document.getElementById('knowledge-search') || document.getElementById('search-input');
        
        document.getElementById('search-btn')?.addEventListener('click', () => this.search());
        document.getElementById('btn-search')?.addEventListener('click', () => this.search());
        this.input?.addEventListener('keypress', (e) => { if(e.key==='Enter') this.search(); });
    }
    
    async search() {
        const q = this.input?.value?.trim();
        if (!q) return;
        
        this.container.innerHTML = '<p class="searching">🔍 Searching...</p>';
        
        try {
            const res = await fetch(`${API.search}?q=${encodeURIComponent(q)}&k=10`);
            const data = await res.json();
            this.render(data.results || [], q);
        } catch {
            this.container.innerHTML = '<p class="error-text">Search failed</p>';
        }
    }
    
    render(results, query) {
        if (!results.length) {
            this.container.innerHTML = `<div class="no-results"><p>No results for "${query}"</p></div>`;
            return;
        }
        
        const labels = { definition:'📖 Definition', fact:'💡 Fact', knowledge:'📚 Knowledge', source:'📄 Source' };
        this.container.innerHTML = `
            <p class="results-count">Found ${results.length} results for "${query}"</p>
            ${results.map(r => `
                <div class="search-result ${r.type}">
                    <span class="result-type">${labels[r.type]||r.type}</span>
                    <p class="result-content">${this.highlight(r.content||'', query)}</p>
                </div>
            `).join('')}
        `;
    }
    
    highlight(text, q) {
        const words = q.toLowerCase().split(/\s+/);
        let result = text;
        words.forEach(w => { if(w.length>2) result = result.replace(new RegExp(`(${w})`,'gi'),'<mark>$1</mark>'); });
        return result;
    }
}

// ============================================================
// Stats Dashboard
// ============================================================
class StatsDashboard {
    constructor() { this.load(); setInterval(() => this.load(), 30000); }
    
    async load() {
        try {
            const res = await fetch(API.stats);
            const s = await res.json();
            
            const fmt = n => n >= 1e6 ? (n/1e6).toFixed(1)+'M' : n.toLocaleString();
            
            // Big stats
            const big = id => document.getElementById(id);
            if (big('big-articles')) big('big-articles').textContent = fmt(s.articles_learned||0);
            if (big('big-facts')) big('big-facts').textContent = fmt(s.facts||s.knowledge||0);
            if (big('big-vectors')) big('big-vectors').textContent = fmt(s.vectors||0);
            
            // Table
            const tbody = document.getElementById('stats-table-body');
            if (tbody) {
                tbody.innerHTML = `
                    <tr><td>Articles</td><td>${fmt(s.articles_learned||0)}</td></tr>
                    <tr><td>Words</td><td>${fmt(s.tokens_trained||0)}</td></tr>
                    <tr><td>Facts</td><td>${fmt(s.facts||s.knowledge||0)}</td></tr>
                    <tr><td>Vectors</td><td>${fmt(s.vectors||0)}</td></tr>
                    <tr><td>Vocabulary</td><td>${fmt(s.vocabulary||0)}</td></tr>
                `;
            }
            
            const db = document.getElementById('db-status');
            if (db) db.innerHTML = `<p>✅ Databases: Online</p><p>✅ Vectors: ${fmt(s.vectors||0)}</p>`;
        } catch {}
    }
}

// ============================================================
// Tabs & Toast
// ============================================================
function setupTabs() {
    document.querySelectorAll('.nav-item').forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
            btn.classList.add('active');
            document.querySelectorAll('.tab-content').forEach(t => t.classList.toggle('active', t.id===`tab-${tab}`));
            if (tab==='learning') window.learningDashboard?.refresh();
            if (tab==='stats') window.statsDashboard?.load();
        });
    });
}

function showToast(msg, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = msg;
    container.appendChild(toast);
    setTimeout(() => { toast.classList.add('fade-out'); setTimeout(() => toast.remove(), 300); }, 3000);
}

// ============================================================
// Initialize
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    console.log('GroundZero AI v2.6 initializing...');
    
    ThemeManager.init();
    setupTabs();
    
    window.chatInterface = new ChatInterface();
    window.voiceRecorder = new VoiceRecorder();
    window.voiceRecorder.init();
    window.statsDashboard = new StatsDashboard();
    window.knowledgeExplorer = new KnowledgeExplorer();
    window.learningDashboard = new LearningDashboard();
    window.learningDashboard.init();
    
    console.log('GroundZero AI v2.6 ready!');
});