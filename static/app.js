class GroundZeroApp {
    constructor() {
        this.isLearning = false;
        this.isProcessing = false;
        this.pollInterval = null;
        
        // Learning timer
        this.timerInterval = null;
        this.timerStartTime = null;
        this.timerElapsed = 0;
        
        // Content storage for zoom view
        this.currentFullContent = '';
        this.currentTitle = '';
        this.currentUrl = '';
        
        // Knowledge Explorer
        this.knowledgeEntries = [];
        this.selectedEntryId = null;
        
        // Track source count
        this.lastSourceCount = 0;
        
        // Prevent concurrent status calls
        this.isLoadingStatus = false;
        
        // Manual learning timers (for multiple concurrent)
        this.learningTimers = {};
        
        this.init();
    }
    
    init() {
        this.connectSocket();
        this.bindEvents();
        this.loadStatus();
        this.loadRecentSources();
        this.initSessionTracking();
    }
    
    connectSocket() {
        try {
            const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
            this.socket = new WebSocket(wsUrl);
            this.socket.onopen = () => console.log('WebSocket connected');
            this.socket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'article_complete') {
                        this.loadStatus();
                        this.loadRecentSources();
                    }
                } catch (e) {}
            };
            this.socket.onerror = () => console.log('WebSocket error, using polling');
        } catch (e) {
            console.log('WebSocket not available');
        }
    }
    
    bindEvents() {
        document.getElementById('start-btn').onclick = () => this.startLearning();
        document.getElementById('stop-btn').onclick = () => this.stopLearning();
        
        document.getElementById('send-btn').onclick = () => this.sendMessage();
        document.getElementById('chat-input').onkeydown = e => {
            if (e.key === 'Enter' && !e.shiftKey) { 
                e.preventDefault(); 
                this.sendMessage(); 
            }
        };
        
        document.getElementById('chat-input').oninput = e => {
            e.target.style.height = 'auto';
            e.target.style.height = Math.min(e.target.scrollHeight, 100) + 'px';
        };
        
        document.getElementById('teach-btn').onclick = () => this.teach();
        
        document.getElementById('learn-url-btn').onclick = () => this.learnFromUrl();
        document.getElementById('learn-url').onkeydown = e => {
            if (e.key === 'Enter') {
                e.preventDefault();
                this.learnFromUrl();
            }
        };
        
        document.getElementById('zoom-btn').onclick = () => this.openZoomModal();
        document.getElementById('zoom-close').onclick = () => this.closeZoomModal();
        document.getElementById('zoom-modal').onclick = (e) => {
            if (e.target.id === 'zoom-modal') this.closeZoomModal();
        };
        
        document.getElementById('fullscreen-btn').onclick = () => this.toggleFullscreen();
        document.addEventListener('fullscreenchange', () => this.updateFullscreenButton());
        
        document.getElementById('knowledge-explorer-btn').onclick = () => this.openKnowledgeExplorer();
        document.getElementById('ke-close').onclick = () => this.closeKnowledgeExplorer();
        document.getElementById('knowledge-explorer-modal').onclick = (e) => {
            if (e.target.id === 'knowledge-explorer-modal') this.closeKnowledgeExplorer();
        };
        document.getElementById('ke-search-input').oninput = (e) => this.filterKnowledge(e.target.value);
        
        document.getElementById('clear-chat-btn').onclick = () => this.clearChatContext();
        
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeZoomModal();
                this.closeKnowledgeExplorer();
            }
        });
    }
    
    async clearChatContext() {
        try {
            await fetch('/api/chat/clear-context', { method: 'POST' });
            const container = document.getElementById('chat-messages');
            container.innerHTML = `
                <div class="message ai">
                    <div class="avatar">🧠</div>
                    <div class="content">
                        <p>Chat context cleared! I've forgotten our previous conversation.</p>
                        <p>Ask me anything - I'll search my knowledge to find answers.</p>
                        <span class="time">Just now</span>
                    </div>
                </div>
            `;
            this.toast('Chat context cleared', 'success');
        } catch (e) {
            this.toast('Failed to clear context', 'error');
        }
    }
    
    toggleFullscreen() {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen().catch(err => {
                this.toast('Fullscreen not available', 'error');
            });
        } else {
            document.exitFullscreen();
        }
    }
    
    updateFullscreenButton() {
        const btn = document.getElementById('fullscreen-btn');
        const icon = document.getElementById('fs-icon');
        const text = btn.querySelector('.fs-text');
        
        if (document.fullscreenElement) {
            btn.classList.add('is-fullscreen');
            icon.textContent = '⛶';
            text.textContent = 'Exit';
        } else {
            btn.classList.remove('is-fullscreen');
            icon.textContent = '⛶';
            text.textContent = 'Fullscreen';
        }
    }
    
    openZoomModal() {
        const modal = document.getElementById('zoom-modal');
        const content = document.getElementById('zoom-content');
        const title = document.getElementById('zoom-title');
        const stats = document.getElementById('zoom-stats');
        const source = document.getElementById('zoom-source');
        
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
        
        title.textContent = this.currentTitle || 'Full Content';
        source.href = this.currentUrl || '#';
        source.textContent = this.currentUrl ? 'View Source' : '';
        
        if (this.currentFullContent) {
            const words = this.currentFullContent.split(/\s+/).filter(w => w.length > 0);
            stats.textContent = `${words.length.toLocaleString()} words`;
            content.textContent = this.currentFullContent;
        } else {
            content.innerHTML = '<p class="empty-text">No content loaded yet.</p>';
            stats.textContent = '0 words';
        }
    }
    
    closeZoomModal() {
        document.getElementById('zoom-modal').classList.remove('active');
        document.body.style.overflow = '';
    }
    
    startTimer() {
        this.timerElapsed = 0;
        this.timerStartTime = Date.now();
        document.getElementById('learning-timer').classList.add('active');
        document.getElementById('timer-status').textContent = 'RUNNING';
        if (this.timerInterval) clearInterval(this.timerInterval);
        this.timerInterval = setInterval(() => this.updateTimer(), 1000);
        this.updateTimer();
    }
    
    stopTimer() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
        document.getElementById('learning-timer').classList.remove('active');
        document.getElementById('timer-status').textContent = 'STOPPED';
    }
    
    updateTimer() {
        if (!this.timerStartTime) return;
        const elapsed = Date.now() - this.timerStartTime;
        document.getElementById('timer-value').textContent = this.formatTime(elapsed);
    }
    
    formatTime(ms) {
        const totalSeconds = Math.floor(ms / 1000);
        const hours = Math.floor(totalSeconds / 3600);
        const minutes = Math.floor((totalSeconds % 3600) / 60);
        const seconds = totalSeconds % 60;
        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }
    
    formatSeconds(ms) {
        const seconds = Math.floor(ms / 1000);
        const tenths = Math.floor((ms % 1000) / 100);
        return `${seconds}.${tenths}s`;
    }
    
    async loadStatus() {
        if (this.isLoadingStatus) return;
        this.isLoadingStatus = true;
        
        try {
            const res = await fetch('/api/status');
            if (!res.ok) throw new Error('Status fetch failed');
            
            const data = await res.json();
            
            if (data.stats) {
                const stats = data.stats;
                
                if (stats.current_article) {
                    this.safeSetText('current-url', stats.current_article);
                    this.currentTitle = stats.current_article;
                }
                if (stats.current_url) {
                    const urlEl = document.getElementById('current-url');
                    if (urlEl) urlEl.href = stats.current_url;
                    this.currentUrl = stats.current_url;
                }
                if (stats.current_content) {
                    this.currentFullContent = stats.current_content;
                    const previewEl = document.getElementById('content-preview');
                    if (previewEl) previewEl.textContent = stats.current_content.substring(0, 500) + '...';
                }
                
                if (stats.current_session) {
                    this.safeSetText('current-articles', stats.current_session.articles_read || 0);
                    this.safeSetText('current-words', this.formatNum(stats.current_session.words_learned || 0));
                    this.safeSetText('current-knowledge', stats.current_session.knowledge_added || 0);
                }
                
                if (stats.is_running && !this.isLearning) {
                    this.isLearning = true;
                    this.updateLearningUI(true);
                    this.startPolling();
                    this.startTimer();
                } else if (!stats.is_running && this.isLearning) {
                    this.isLearning = false;
                    this.updateLearningUI(false);
                    this.stopPolling();
                    this.stopTimer();
                    this.loadRecentSources();
                }
                
                if (stats.total) {
                    const total = stats.total;
                    this.safeSetText('vocab-count', this.formatNum(total.vocabulary_size || 0));
                    this.safeSetText('vocab-stat', this.formatNum(total.vocabulary_size || 0));
                    this.safeSetText('knowledge-count', this.formatNum(total.total_knowledge || 0));
                    this.safeSetText('knowledge-stat', this.formatNum(total.total_knowledge || 0));
                    this.safeSetText('sources-count', this.formatNum(total.total_sources || 0));
                    this.safeSetText('sources-stat', this.formatNum(total.total_sources || 0));
                    this.safeSetText('tokens-count', this.formatNum(total.total_words || 0));
                    
                    if (total.vectors) {
                        this.safeSetText('vectors-count', this.formatNum(total.vectors.total_vectors || 0));
                    }
                    if (total.total_learning_time !== undefined) {
                        this.safeSetText('total-learning-time', this.formatDuration(total.total_learning_time));
                    }
                    if (total.total_sessions !== undefined) {
                        this.safeSetText('sessions-count', this.formatNum(total.total_sessions));
                    }
                    
                    this.safeSetText('knowledge-badge', this.formatNum(total.total_knowledge || 0));
                    
                    const newSourceCount = total.total_sources || 0;
                    if (newSourceCount >= this.lastSourceCount + 30) {
                        this.lastSourceCount = newSourceCount;
                        this.loadRecentSources();
                    }
                }
            }
        } catch (e) {
            console.error('Failed to load status:', e);
        } finally {
            this.isLoadingStatus = false;
        }
    }
    
    async loadRecentSources() {
        try {
            const res = await fetch('/api/knowledge/recent?limit=20');
            if (!res.ok) return;
            
            const data = await res.json();
            const sources = data.recent || [];
            const sourcesEl = document.getElementById('learned-sources');
            
            if (!sourcesEl) return;
            
            if (sources.length > 0) {
                sourcesEl.innerHTML = sources.map(s => 
                    `<div class="source-item">
                        <a href="${s.source_url || '#'}" target="_blank">${this.escapeHtml(s.source_title || 'Source')}</a>
                    </div>`
                ).join('');
            } else {
                sourcesEl.innerHTML = '<p class="empty-text">No sources yet</p>';
            }
        } catch (e) {
            console.error('Failed to load recent sources:', e);
        }
    }
    
    safeSetText(id, value) {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    }
    
    async startLearning() {
        try {
            document.getElementById('current-url').textContent = 'Finding content...';
            document.getElementById('content-preview').textContent = 'Searching for knowledge sources...';
            
            const res = await fetch('/api/learn/start', { method: 'POST' });
            const data = await res.json();
            
            if (data.status === 'started') {
                this.isLearning = true;
                this.updateLearningUI(true);
                this.startPolling();
                this.startTimer();
                this.toast('Learning started!', 'success');
            }
        } catch (e) {
            this.toast('Failed to start learning', 'error');
        }
    }
    
    async stopLearning() {
        try {
            await fetch('/api/learn/stop', { method: 'POST' });
            this.isLearning = false;
            this.updateLearningUI(false);
            this.stopPolling();
            this.stopTimer();
            this.toast('Learning stopped', 'info');
            this.loadRecentSources();
            this.loadStatus();
            this.loadSessions();
            
            document.getElementById('current-articles').textContent = '0';
            document.getElementById('current-words').textContent = '0';
            document.getElementById('current-knowledge').textContent = '0';
        } catch (e) {
            this.toast('Failed to stop learning', 'error');
        }
    }
    
    startPolling() {
        if (this.pollInterval) return;
        this.pollInterval = setInterval(() => this.loadStatus(), 2000);
    }
    
    stopPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
    }
    
    updateLearningUI(isLearning) {
        document.getElementById('start-btn').disabled = isLearning;
        document.getElementById('stop-btn').disabled = !isLearning;
        document.getElementById('start-btn').textContent = isLearning ? '⏳ Learning...' : '▶ Start';
        
        const status = document.getElementById('status-indicator');
        status.className = isLearning ? 'stat-pill status learning' : 'stat-pill status';
        status.innerHTML = `<span class="pulse"></span>${isLearning ? 'Learning' : 'Ready'}`;
    }
    
    async sendMessage() {
        const input = document.getElementById('chat-input');
        const message = input.value.trim();
        if (!message || this.isProcessing) return;
        
        this.isProcessing = true;
        input.value = '';
        input.style.height = 'auto';
        
        this.addUserMessage(message);
        const thinkingId = this.addThinkingMessage();
        
        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message, auto_search: true })
            });
            
            const data = await res.json();
            this.removeThinkingMessage(thinkingId);
            this.addAIMessage(data);
            this.loadStatus();
            this.loadRecentSources();
        } catch (e) {
            this.removeThinkingMessage(thinkingId);
            this.addAIMessage({ response: "Sorry, I encountered an error. Please try again." });
        }
        
        this.isProcessing = false;
    }
    
    addUserMessage(text) {
        const container = document.getElementById('chat-messages');
        const div = document.createElement('div');
        div.className = 'message user';
        div.innerHTML = `<div class="content"><p>${this.escapeHtml(text)}</p><span class="time">Just now</span></div>`;
        container.appendChild(div);
        container.scrollTop = container.scrollHeight;
    }
    
    addThinkingMessage() {
        const container = document.getElementById('chat-messages');
        const id = 'thinking-' + Date.now();
        const div = document.createElement('div');
        div.className = 'message ai thinking';
        div.id = id;
        div.innerHTML = `<div class="avatar">🧠</div><div class="content"><div class="thinking-dots"><span></span><span></span><span></span></div></div>`;
        container.appendChild(div);
        container.scrollTop = container.scrollHeight;
        return id;
    }
    
    removeThinkingMessage(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }
    
    addAIMessage(data) {
        const container = document.getElementById('chat-messages');
        const div = document.createElement('div');
        div.className = 'message ai';
        
        let html = `<div class="avatar">🧠</div><div class="content">`;
        
        const responseHtml = this.parseMarkdown(data.response || data.answer || "I don't have an answer for that.");
        html += responseHtml;
        
        if (data.sources && data.sources.length > 0) {
            html += `<div class="sources"><strong>Sources:</strong>`;
            data.sources.forEach(s => {
                html += `<a href="${s.url}" target="_blank">${this.escapeHtml(s.title || 'Source')}</a>`;
            });
            html += `</div>`;
        }
        
        if (data.learned_from && data.learned_from.length > 0) {
            html += `<div class="learned-from"><strong>🎓 Learned from:</strong>`;
            data.learned_from.forEach(s => {
                html += `<a href="${s.url}" target="_blank">${this.escapeHtml(s.title)}</a>`;
            });
            html += `</div>`;
        }
        
        html += `<span class="time">Just now</span></div>`;
        div.innerHTML = html;
        container.appendChild(div);
        container.scrollTop = container.scrollHeight;
    }
    
    parseMarkdown(text) {
        return text
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.+?)\*/g, '<em>$1</em>')
            .replace(/`(.+?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>')
            .split('<br><br>').map(p => `<p>${p}</p>`).join('');
    }
    
    async teach() {
        const content = document.getElementById('teach-content').value.trim();
        if (!content) {
            this.toast('Enter some knowledge to teach', 'error');
            return;
        }
        
        try {
            const res = await fetch('/api/teach', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content, title: 'User taught knowledge' })
            });
            
            const data = await res.json();
            if (data.success) {
                this.toast('Knowledge taught successfully!', 'success');
                document.getElementById('teach-content').value = '';
                this.loadStatus();
                this.loadRecentSources();
            } else {
                this.toast(data.error || 'Failed to teach', 'error');
            }
        } catch (e) {
            this.toast('Failed to teach', 'error');
        }
    }
    
    // ========== LEARN FROM URL - BACKGROUND PROCESSING ==========
    
    async learnFromUrl() {
        const urlInput = document.getElementById('learn-url');
        const btn = document.getElementById('learn-url-btn');
        const url = urlInput.value.trim();
        
        if (!url) {
            this.toast('Enter a URL', 'error');
            return;
        }
        if (!url.startsWith('http')) {
            this.toast('URL must start with http:// or https://', 'error');
            return;
        }
        
        // Create inline progress indicator
        const progressId = this.showInlineProgress(url);
        
        // Clear input immediately so user can paste another URL
        urlInput.value = '';
        
        // Update button to show activity
        const originalBtnText = btn.innerHTML;
        btn.innerHTML = '<span class="spinner-small"></span>';
        btn.classList.add('loading');
        
        // Start timer for this specific learning task
        const startTime = Date.now();
        this.learningTimers[progressId] = setInterval(() => {
            this.updateProgressTimer(progressId, startTime);
        }, 100);
        
        try {
            // Stage 1: Sending request
            this.updateInlineProgress(progressId, 'searching', `Sending ${this.extractDomain(url)}`, startTime);
            
            const res = await fetch('/api/learn/url', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url })
            });
            
            const data = await res.json();
            
            if (data.success) {
                // Check if it's background processing or immediate completion
                if (data.status === 'processing') {
                    // Backend started processing in background
                    this.updateInlineProgress(progressId, 'processing', 'Processing in background...', startTime);
                    
                    // Stop timer
                    if (this.learningTimers[progressId]) {
                        clearInterval(this.learningTimers[progressId]);
                        delete this.learningTimers[progressId];
                    }
                    
                    // Show completion after brief delay
                    await this.sleep(500);
                    this.completeInlineProgress(progressId, 'Queued ✓', startTime, 'success');
                    
                    // Poll for actual completion
                    this.pollForNewSource(url, progressId);
                    
                } else {
                    // Immediate completion (shouldn't happen with new backend but handle it)
                    this.updateInlineProgress(progressId, 'storing', 'Storing knowledge', startTime);
                    await this.sleep(200);
                    this.completeInlineProgress(progressId, data.title || 'Learned!', startTime);
                    this.addLearningSuccessMessage(data, url);
                    this.refreshAllStats();
                }
                
            } else if (data.duplicate) {
                this.completeInlineProgress(progressId, 'Already known', startTime, 'info');
            } else {
                this.failInlineProgress(progressId, data.message || data.error || 'Failed', startTime);
            }
            
        } catch (e) {
            console.error('Learn URL error:', e);
            this.failInlineProgress(progressId, 'Network error', startTime);
        } finally {
            // Stop timer if still running
            if (this.learningTimers[progressId]) {
                clearInterval(this.learningTimers[progressId]);
                delete this.learningTimers[progressId];
            }
            
            // Reset button
            btn.innerHTML = originalBtnText || '🌐 Learn';
            btn.classList.remove('loading');
        }
    }
    
    async pollForNewSource(url, progressId) {
        // Poll for completion in background - check if source was added
        const domain = this.extractDomain(url);
        let attempts = 0;
        const maxAttempts = 30; // Poll for up to 60 seconds (30 * 2s)
        
        const poll = async () => {
            attempts++;
            try {
                await this.loadRecentSources();
                await this.loadStatus();
                
                // Check if we've exceeded max attempts
                if (attempts >= maxAttempts) {
                    this.toast(`Learning ${domain} - check Recent Sources`, 'info');
                    return;
                }
                
                // Continue polling
                setTimeout(poll, 2000);
            } catch (e) {
                // Stop polling on error
            }
        };
        
        // Start polling after 3 seconds
        setTimeout(poll, 3000);
    }
    
    showInlineProgress(url) {
        const container = document.getElementById('learning-progress-container');
        if (!container) return null;
        
        const progressId = 'progress-' + Date.now();
        const domain = this.extractDomain(url);
        
        const el = document.createElement('div');
        el.className = 'inline-progress';
        el.id = progressId;
        el.innerHTML = `
            <span class="progress-dot searching"></span>
            <span class="progress-text">Searching ${domain}</span>
            <span class="progress-timer">0.0s</span>
        `;
        
        container.appendChild(el);
        return progressId;
    }
    
    updateInlineProgress(progressId, stage, text, startTime) {
        if (!progressId) return;
        const el = document.getElementById(progressId);
        if (!el) return;
        
        const dot = el.querySelector('.progress-dot');
        const textEl = el.querySelector('.progress-text');
        
        // Update dot color based on stage
        dot.className = `progress-dot ${stage}`;
        textEl.textContent = text;
        
        this.updateProgressTimer(progressId, startTime);
    }
    
    updateProgressTimer(progressId, startTime) {
        if (!progressId) return;
        const el = document.getElementById(progressId);
        if (!el) return;
        
        const timerEl = el.querySelector('.progress-timer');
        if (timerEl) {
            timerEl.textContent = this.formatSeconds(Date.now() - startTime);
        }
    }
    
    completeInlineProgress(progressId, text, startTime, type = 'success') {
        if (!progressId) return;
        const el = document.getElementById(progressId);
        if (!el) return;
        
        // Stop timer
        if (this.learningTimers[progressId]) {
            clearInterval(this.learningTimers[progressId]);
            delete this.learningTimers[progressId];
        }
        
        const dot = el.querySelector('.progress-dot');
        const textEl = el.querySelector('.progress-text');
        const timerEl = el.querySelector('.progress-timer');
        
        dot.className = `progress-dot ${type}`;
        textEl.textContent = type === 'success' ? `✓ ${text}` : `ℹ ${text}`;
        timerEl.textContent = this.formatSeconds(Date.now() - startTime);
        
        el.classList.add('complete');
        
        // Auto-remove after 3 seconds
        setTimeout(() => {
            el.classList.add('fade-out');
            setTimeout(() => el.remove(), 400);
        }, 3000);
    }
    
    failInlineProgress(progressId, reason, startTime) {
        if (!progressId) return;
        const el = document.getElementById(progressId);
        if (!el) return;
        
        // Stop timer
        if (this.learningTimers[progressId]) {
            clearInterval(this.learningTimers[progressId]);
            delete this.learningTimers[progressId];
        }
        
        const dot = el.querySelector('.progress-dot');
        const textEl = el.querySelector('.progress-text');
        const timerEl = el.querySelector('.progress-timer');
        
        dot.className = 'progress-dot error';
        textEl.textContent = `✗ ${reason}`;
        timerEl.textContent = this.formatSeconds(Date.now() - startTime);
        
        el.classList.add('failed');
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            el.classList.add('fade-out');
            setTimeout(() => el.remove(), 400);
        }, 5000);
    }
    
    addLearningSuccessMessage(data, url) {
        const container = document.getElementById('chat-messages');
        const div = document.createElement('div');
        div.className = 'message ai learning-notification';
        
        const wordCount = data.word_count || data.words_added || 'new';
        
        div.innerHTML = `
            <div class="avatar">🎓</div>
            <div class="content">
                <div class="learning-success-badge">✨ New Knowledge Added</div>
                <p><strong>${this.escapeHtml(data.title || 'Web page')}</strong></p>
                <p>I learned ${typeof wordCount === 'number' ? wordCount.toLocaleString() + ' words' : 'content'} from this source.</p>
                <div class="sources">
                    <a href="${url}" target="_blank">View source →</a>
                </div>
                <span class="time">Just now</span>
            </div>
        `;
        
        container.appendChild(div);
        container.scrollTop = container.scrollHeight;
    }
    
    async refreshAllStats() {
        await Promise.all([
            this.loadStatus(),
            this.loadRecentSources(),
            this.updateNeuralBadge()
        ]);
    }
    
    async updateNeuralBadge() {
        try {
            const res = await fetch('/api/stats');
            const data = await res.json();
            
            if (data.neural && data.neural.available) {
                const badge = document.getElementById('neural-badge');
                if (badge) badge.textContent = this.formatNum(data.neural.model_params || 0);
                
                const tokensEl = document.getElementById('current-neural-tokens');
                if (tokensEl && data.neural.total_tokens_trained) {
                    tokensEl.textContent = this.formatNum(data.neural.total_tokens_trained);
                }
            }
        } catch (e) {}
    }
    
    extractDomain(url) {
        try {
            const parsed = new URL(url);
            return parsed.hostname.replace('www.', '');
        } catch (e) {
            return url.substring(0, 30) + '...';
        }
    }
    
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    // ========== SESSIONS ==========
    
    initSessionTracking() {
        const expandBtn = document.getElementById('expand-sessions-btn');
        if (expandBtn) {
            expandBtn.onclick = () => this.toggleSessionHistory();
        }
        this.loadSessions();
    }
    
    toggleSessionHistory() {
        const btn = document.getElementById('expand-sessions-btn');
        const history = document.getElementById('session-history');
        
        if (history.classList.contains('expanded')) {
            history.classList.remove('expanded');
            btn.classList.remove('expanded');
        } else {
            history.classList.add('expanded');
            btn.classList.add('expanded');
            this.loadSessions();
        }
    }
    
    async loadSessions() {
        try {
            const res = await fetch('/api/sessions');
            const data = await res.json();
            
            if (data.summary) {
                this.safeSetText('total-sessions', this.formatNum(data.summary.total_sessions || 0));
                this.safeSetText('total-articles', this.formatNum(data.summary.total_articles || 0));
                this.safeSetText('total-words-learned', this.formatNum(data.summary.total_words || 0));
                this.safeSetText('total-knowledge-added', this.formatNum(data.summary.total_knowledge || 0));
                this.safeSetText('total-learning-time', this.formatDuration(data.summary.total_time_seconds || 0));
                this.safeSetText('sessions-count', this.formatNum(data.summary.total_sessions || 0));
            }
            
            const listEl = document.getElementById('session-list');
            if (data.sessions && data.sessions.length > 0) {
                listEl.innerHTML = data.sessions.map(s => `
                    <div class="session-item">
                        <div class="session-header">
                            <span class="session-id">Session #${s.id}</span>
                            <span class="session-status ${s.status === 'active' ? 'active' : ''}">${s.status === 'active' ? 'Active' : 'Completed'}</span>
                        </div>
                        <div class="session-stats-row">
                            <span>⏱️ ${this.formatDuration(s.duration_seconds || 0)}</span>
                            <span>📄 ${s.articles_learned || 0}</span>
                            <span>📝 ${this.formatNum(s.words_learned || 0)}</span>
                        </div>
                    </div>
                `).join('');
            } else {
                listEl.innerHTML = '<div class="no-sessions">No sessions yet</div>';
            }
        } catch (e) {}
    }
    
    // ========== KNOWLEDGE EXPLORER ==========
    
    async openKnowledgeExplorer() {
        const modal = document.getElementById('knowledge-explorer-modal');
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
        
        this.knowledgeEntries = [];
        this.selectedEntryId = null;
        document.getElementById('ke-search-input').value = '';
        document.getElementById('ke-detail-title').textContent = 'Select a topic';
        document.getElementById('ke-detail-confidence').textContent = '';
        document.getElementById('ke-detail-confidence').style.display = 'none';
        document.getElementById('ke-related-list').innerHTML = '<p class="ke-hint">Click on a topic to see related concepts</p>';
        
        await this.loadKnowledgeEntries();
    }
    
    closeKnowledgeExplorer() {
        document.getElementById('knowledge-explorer-modal').classList.remove('active');
        document.body.style.overflow = '';
    }
    
    async loadKnowledgeEntries() {
        const listEl = document.getElementById('ke-list');
        listEl.innerHTML = '<div class="ke-loading">Loading knowledge base...</div>';
        
        try {
            const res = await fetch('/api/knowledge/all?limit=200');
            const data = await res.json();
            
            this.knowledgeEntries = data.entries || [];
            this.renderKnowledgeList(this.knowledgeEntries);
        } catch (e) {
            listEl.innerHTML = '<div class="ke-loading">Failed to load</div>';
        }
    }
    
    renderKnowledgeList(entries) {
        const listEl = document.getElementById('ke-list');
        
        if (entries.length === 0) {
            listEl.innerHTML = '<div class="ke-empty">No knowledge entries yet.</div>';
            return;
        }
        
        listEl.innerHTML = entries.map(entry => `
            <div class="ke-item ${this.selectedEntryId === entry.id ? 'selected' : ''}" data-id="${entry.id}">
                <span class="ke-item-title">${this.escapeHtml(entry.title || 'Untitled')}</span>
                <span class="ke-item-confidence">${Math.round((entry.confidence || 0.5) * 100)}%</span>
            </div>
        `).join('');
        
        listEl.querySelectorAll('.ke-item').forEach(item => {
            item.onclick = () => this.selectKnowledgeEntry(parseInt(item.dataset.id));
        });
    }
    
    filterKnowledge(query) {
        if (!this.knowledgeEntries) return;
        
        const filtered = query.trim() === '' 
            ? this.knowledgeEntries 
            : this.knowledgeEntries.filter(e => 
                (e.title || '').toLowerCase().includes(query.toLowerCase())
            );
        
        this.renderKnowledgeList(filtered);
    }
    
    async selectKnowledgeEntry(entryId) {
        this.selectedEntryId = entryId;
        
        document.querySelectorAll('.ke-item').forEach(item => {
            item.classList.toggle('selected', parseInt(item.dataset.id) === entryId);
        });
        
        const entry = this.knowledgeEntries.find(e => e.id === entryId);
        if (!entry) return;
        
        document.getElementById('ke-detail-title').textContent = entry.title || 'Untitled';
        const confEl = document.getElementById('ke-detail-confidence');
        confEl.textContent = `${Math.round((entry.confidence || 0.5) * 100)}% confidence`;
        confEl.style.display = 'inline-block';
        
        const relatedEl = document.getElementById('ke-related-list');
        relatedEl.innerHTML = '<div class="ke-loading-small">Finding related...</div>';
        
        try {
            const res = await fetch(`/api/knowledge/${entryId}/related?limit=10`);
            const data = await res.json();
            
            if (data.related && data.related.length > 0) {
                relatedEl.innerHTML = data.related.map(r => `
                    <div class="ke-related-item" data-id="${r.id}">
                        <span class="ke-related-title">${this.escapeHtml(r.title || 'Untitled')}</span>
                        <span class="ke-related-similarity">${r.similarity}%</span>
                    </div>
                `).join('');
                
                relatedEl.querySelectorAll('.ke-related-item').forEach(item => {
                    item.onclick = () => this.selectKnowledgeEntry(parseInt(item.dataset.id));
                });
            } else {
                relatedEl.innerHTML = '<p class="ke-hint">No related topics found</p>';
            }
        } catch (e) {
            relatedEl.innerHTML = '<p class="ke-hint">Failed to load</p>';
        }
    }
    
    // ========== UTILITIES ==========
    
    toast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        container.appendChild(toast);
        setTimeout(() => toast.remove(), 3000);
    }
    
    formatNum(n) {
        if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
        if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
        return String(n);
    }
    
    formatDuration(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;
        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

document.addEventListener('DOMContentLoaded', () => new GroundZeroApp());