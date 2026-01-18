/**
 * GroundZero AI - Dashboard JavaScript
 * With Themes, Link Detection, Fixed Layout
 */

// ============================================================================
// STATE
// ============================================================================

let conversationId = null;
let messages = [];
let isLoading = false;
let currentReasoning = null;
let userName = 'User';
let pendingFiles = [];
let webSearchEnabled = false;

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    loadUserProfile();
    loadStats();
    initDragAndDrop();
    loadThemeSettings();
});

// ============================================================================
// THEME SYSTEM
// ============================================================================

function loadThemeSettings() {
    const savedTheme = localStorage.getItem('gz-theme') || 'default';
    const savedMode = localStorage.getItem('gz-mode') || 'dark';
    
    setTheme(savedTheme, false);
    setMode(savedMode, false);
}

function setTheme(theme, save = true) {
    document.documentElement.setAttribute('data-theme', theme);
    
    // Update buttons
    document.querySelectorAll('.theme-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.classList.contains(`theme-${theme}`)) {
            btn.classList.add('active');
        }
    });
    
    if (save) {
        localStorage.setItem('gz-theme', theme);
        showToast(`Theme changed to ${theme}`);
    }
}

function setMode(mode, save = true) {
    document.documentElement.setAttribute('data-mode', mode);
    
    // Update buttons
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.mode === mode) {
            btn.classList.add('active');
        }
    });
    
    if (save) {
        localStorage.setItem('gz-mode', mode);
        showToast(`${mode === 'dark' ? 'Dark' : 'Light'} mode enabled`);
    }
}

// ============================================================================
// DRAG AND DROP
// ============================================================================

function initDragAndDrop() {
    const messagesContainer = document.getElementById('messagesContainer');
    const chatSection = document.getElementById('chat-section');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        document.body.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        chatSection.addEventListener(eventName, () => {
            messagesContainer.classList.add('drag-over');
        }, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        chatSection.addEventListener(eventName, () => {
            messagesContainer.classList.remove('drag-over');
        }, false);
    });
    
    chatSection.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            for (let i = 0; i < files.length; i++) {
                addPendingFile(files[i]);
            }
        }
    }, false);
}

// ============================================================================
// WEB SEARCH TOGGLE
// ============================================================================

function toggleSearch() {
    webSearchEnabled = !webSearchEnabled;
    
    const btn = document.getElementById('searchToggle');
    const indicator = document.getElementById('searchIndicator');
    
    if (webSearchEnabled) {
        btn.classList.add('active');
        indicator.classList.add('visible');
    } else {
        btn.classList.remove('active');
        indicator.classList.remove('visible');
    }
}

// ============================================================================
// FILE ATTACHMENT
// ============================================================================

function triggerFileUpload() {
    document.getElementById('chatFileInput').click();
}

function handleFileSelect(event) {
    const files = event.target.files;
    for (let i = 0; i < files.length; i++) {
        addPendingFile(files[i]);
    }
    event.target.value = '';
}

function addPendingFile(file) {
    if (pendingFiles.some(f => f.name === file.name && f.size === file.size)) {
        showToast('File already attached', 'info');
        return;
    }
    
    pendingFiles.push(file);
    renderPendingFiles();
    document.getElementById('messageInput').focus();
}

function removePendingFile(index) {
    pendingFiles.splice(index, 1);
    renderPendingFiles();
}

function renderPendingFiles() {
    const container = document.getElementById('attachedFilesPreview');
    
    if (pendingFiles.length === 0) {
        container.innerHTML = '';
        container.classList.remove('has-files');
        document.getElementById('messageInput').placeholder = 'Message GroundZero...';
        return;
    }
    
    container.classList.add('has-files');
    container.innerHTML = pendingFiles.map((file, index) => `
        <div class="attached-file">
            <span class="file-icon">${getFileIcon(file.name)}</span>
            <span class="file-name">${truncateFilename(file.name, 15)}</span>
            <span class="file-size">${formatFileSize(file.size)}</span>
            <button class="remove-file-btn" onclick="removePendingFile(${index})" title="Remove">×</button>
        </div>
    `).join('');
    
    document.getElementById('messageInput').placeholder = `Ask about your file...`;
}

function getFileIcon(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    const icons = {
        'csv': '📊', 'xlsx': '📊', 'xls': '📊',
        'pdf': '📕',
        'txt': '📄', 'md': '📄',
        'json': '📋',
        'docx': '📝', 'doc': '📝',
        'py': '🐍',
        'js': '📜',
        'html': '🌐',
        'xml': '📰',
        'yaml': '⚙️', 'yml': '⚙️',
        'png': '🖼️', 'jpg': '🖼️', 'jpeg': '🖼️', 'gif': '🖼️',
    };
    return icons[ext] || '📎';
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function truncateFilename(name, maxLen) {
    if (name.length <= maxLen) return name;
    const ext = name.includes('.') ? '.' + name.split('.').pop() : '';
    const base = name.slice(0, name.length - ext.length);
    const truncatedBase = base.slice(0, maxLen - ext.length - 2);
    return truncatedBase + '..' + ext;
}

// ============================================================================
// SEND MESSAGE
// ============================================================================

async function sendMessage() {
    const input = document.getElementById('messageInput');
    const message = input.value.trim();
    const files = [...pendingFiles];
    
    if (!message && files.length === 0) return;
    if (isLoading) return;
    
    // Build display message
    let userMessageDisplay = message;
    if (files.length > 0) {
        const fileList = files.map(f => `📎 ${f.name}`).join('\n');
        userMessageDisplay = files.length > 0 && message 
            ? `${fileList}\n\n${message}`
            : fileList + (message ? `\n\n${message}` : '\n\nAnalyze this file');
    }
    
    addMessage('user', userMessageDisplay);
    
    // Clear input
    input.value = '';
    autoResize(input);
    pendingFiles = [];
    renderPendingFiles();
    
    // Show loading
    isLoading = true;
    document.getElementById('sendBtn').disabled = true;
    const loadingId = addMessage('assistant', '...', true);
    
    try {
        let response;
        
        if (files.length > 0) {
            const formData = new FormData();
            formData.append('message', message || 'Analyze this file and tell me what it contains');
            formData.append('user_id', 'default');
            formData.append('search_web', webSearchEnabled);
            if (conversationId) {
                formData.append('conversation_id', conversationId);
            }
            
            files.forEach((file) => {
                formData.append('file', file);
            });
            
            response = await fetch('/api/chat', {
                method: 'POST',
                body: formData
            });
        } else {
            response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: message,
                    conversation_id: conversationId,
                    user_id: 'default',
                    search_web: webSearchEnabled
                })
            });
        }
        
        const data = await response.json();
        
        removeMessage(loadingId);
        addMessage('assistant', data.response, false, data.reasoning);
        
        if (data.reasoning) {
            currentReasoning = data.reasoning;
        }
        
        if (data.conversation_id) {
            conversationId = data.conversation_id;
        }
        
    } catch (error) {
        removeMessage(loadingId);
        addMessage('assistant', 'Sorry, there was an error processing your message.');
        console.error('Chat error:', error);
    }
    
    isLoading = false;
    document.getElementById('sendBtn').disabled = false;
}

// ============================================================================
// MESSAGE DISPLAY
// ============================================================================

function addMessage(role, content, isLoading = false, reasoning = null) {
    const container = document.getElementById('messagesContainer');
    const welcome = container.querySelector('.welcome-message');
    if (welcome) welcome.remove();
    
    const messageId = 'msg-' + Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message ${isLoading ? 'loading' : ''}`;
    messageDiv.id = messageId;
    
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const name = role === 'user' ? userName : 'GroundZero';
    const avatar = role === 'user' ? '👤' : '🧠';
    
    let reasoningBtn = '';
    if (reasoning && reasoning.steps && reasoning.steps.length > 0) {
        reasoningBtn = `<button class="reasoning-btn" onclick='showReasoning(${JSON.stringify(reasoning).replace(/'/g, "\\'")})'>
            <span class="reasoning-dot"></span> Show reasoning (${reasoning.steps.length} steps)
        </button>`;
    }
    
    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <div class="message-header">
                <span class="message-name">${name}</span>
                <span class="message-time">${time}</span>
            </div>
            <div class="message-text">${isLoading ? '<span class="typing-indicator">Thinking...</span>' : formatMessage(content)}</div>
            ${reasoningBtn}
            ${!isLoading && role === 'assistant' ? `
            <div class="message-actions">
                <button onclick="giveFeedback(1)" title="Good response">👍</button>
                <button onclick="giveFeedback(-1)" title="Bad response">👎</button>
                <button onclick="copyMessage('${messageId}')" title="Copy">📋</button>
            </div>` : ''}
        </div>
    `;
    
    container.appendChild(messageDiv);
    container.scrollTop = container.scrollHeight;
    
    messages.push({ id: messageId, role, content });
    
    return messageId;
}

function removeMessage(messageId) {
    const msg = document.getElementById(messageId);
    if (msg) msg.remove();
    messages = messages.filter(m => m.id !== messageId);
}

function formatMessage(text) {
    if (!text) return '';
    
    // Escape HTML
    text = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    
    // Code blocks
    text = text.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
        return `<pre><code class="language-${lang || ''}">${code}</code></pre>`;
    });
    
    // Inline code
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Bold
    text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    
    // Italic
    text = text.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    
    // Links - Markdown style [text](url)
    text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    
    // Auto-detect URLs and make them clickable
    text = text.replace(/(^|[^"'])(https?:\/\/[^\s<]+)/g, '$1<a href="$2" target="_blank" rel="noopener">$2</a>');
    
    // Horizontal rule
    text = text.replace(/^---$/gm, '<hr>');
    
    // Line breaks
    text = text.replace(/\n/g, '<br>');
    
    return text;
}

function newChat() {
    conversationId = null;
    messages = [];
    pendingFiles = [];
    webSearchEnabled = false;
    
    const searchBtn = document.getElementById('searchToggle');
    const searchIndicator = document.getElementById('searchIndicator');
    if (searchBtn) searchBtn.classList.remove('active');
    if (searchIndicator) searchIndicator.classList.remove('visible');
    
    renderPendingFiles();
    
    document.getElementById('messagesContainer').innerHTML = `
        <div class="welcome-message">
            <h2>Welcome to GroundZero AI! 🧠</h2>
            <p>I'm your personal AI assistant that learns and grows with you.</p>
            <div class="capabilities">
                <div class="capability"><span>💬</span><span>Chat & Learn</span></div>
                <div class="capability"><span>📄</span><span>Analyze Files</span></div>
                <div class="capability"><span>🌐</span><span>Web Search</span></div>
                <div class="capability"><span>💻</span><span>Run Code</span></div>
            </div>
            <p class="hint-text">📎 Attach files • 🌐 Toggle web search • Just ask anything!</p>
        </div>
    `;
    showSection('chat');
}

// ============================================================================
// KNOWLEDGE
// ============================================================================

async function searchKnowledge() {
    const query = document.getElementById('knowledgeSearch').value.trim();
    if (!query) return;
    
    try {
        const response = await fetch(`/api/knowledge/search?q=${encodeURIComponent(query)}`);
        const data = await response.json();
        
        const container = document.getElementById('knowledgeResults');
        
        if (data.results && data.results.length > 0) {
            container.innerHTML = data.results.map(r => `
                <div class="knowledge-item">
                    <h4>${r.name || 'Unknown'}</h4>
                    <p>${r.content || ''}</p>
                    <span class="confidence">Confidence: ${((r.confidence || 0) * 100).toFixed(0)}%</span>
                </div>
            `).join('');
        } else {
            container.innerHTML = '<p class="empty-state">No results found</p>';
        }
    } catch (error) {
        console.error('Error searching knowledge:', error);
    }
}

async function teachKnowledge() {
    const subject = document.getElementById('teachSubject').value.trim();
    const content = document.getElementById('teachContent').value.trim();
    
    if (!subject || !content) {
        showToast('Enter subject and content', 'error');
        return;
    }
    
    try {
        const response = await fetch('/api/knowledge/teach', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ subject, content })
        });
        
        const data = await response.json();
        
        if (data.stored) {
            showToast('Knowledge saved!');
            document.getElementById('teachSubject').value = '';
            document.getElementById('teachContent').value = '';
        } else {
            showToast('Failed to save', 'error');
        }
    } catch (error) {
        showToast('Error saving knowledge', 'error');
    }
}

// ============================================================================
// STATS
// ============================================================================

async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        const stats = await response.json();
        
        const content = document.getElementById('statsContent');
        if (content) {
            content.innerHTML = `
                <div class="stat-card">
                    <h3>🤖 Model</h3>
                    <p class="stat-value">${stats.model?.name || 'Unknown'}</p>
                    <p class="stat-label">Status: ${stats.model?.loaded ? 'Loaded' : 'Mock Mode'}</p>
                </div>
                <div class="stat-card">
                    <h3>📚 Knowledge</h3>
                    <p class="stat-value">${stats.knowledge?.total_nodes || 0}</p>
                    <p class="stat-label">Nodes</p>
                </div>
                <div class="stat-card">
                    <h3>💾 Memory</h3>
                    <p class="stat-value">${stats.memory?.conversations || 0}</p>
                    <p class="stat-label">Conversations</p>
                </div>
                <div class="stat-card">
                    <h3>📈 Learning</h3>
                    <p class="stat-value">${stats.learning?.queue_size || 0}</p>
                    <p class="stat-label">Items in Queue</p>
                </div>
            `;
        }
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// ============================================================================
// USER & SETTINGS
// ============================================================================

async function loadUserProfile() {
    try {
        const response = await fetch('/api/user');
        const user = await response.json();
        
        userName = user.name || 'User';
        const userNameEl = document.getElementById('userName');
        if (userNameEl) userNameEl.textContent = userName;
        
        const settingsName = document.getElementById('settingsName');
        if (settingsName) settingsName.value = userName;
    } catch (error) {
        console.error('Error loading user:', error);
    }
}

async function saveName() {
    const name = document.getElementById('settingsName').value.trim();
    if (!name) return;
    
    try {
        await fetch('/api/user', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });
        
        userName = name;
        document.getElementById('userName').textContent = name;
        showToast('Name saved!');
    } catch (error) {
        showToast('Failed to save name', 'error');
    }
}

// ============================================================================
// REASONING PANEL
// ============================================================================

function showReasoning(reasoning) {
    const panel = document.getElementById('reasoningPanel');
    const content = document.getElementById('reasoningContent');
    
    if (!reasoning || !reasoning.steps) {
        content.innerHTML = '<p>No reasoning available</p>';
    } else {
        content.innerHTML = `
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${(reasoning.confidence || 0.5) * 100}%"></div>
                <span>${Math.round((reasoning.confidence || 0.5) * 100)}% confident</span>
            </div>
            <div class="reasoning-steps">
                ${reasoning.steps.map((step, i) => `
                    <div class="reasoning-step">
                        <span class="step-number">${i + 1}</span>
                        <div class="step-content">
                            <p class="step-thought">${step.thought || ''}</p>
                            ${step.action ? `<p class="step-action">Action: ${step.action}</p>` : ''}
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }
    
    panel.classList.add('open');
}

function closeReasoning() {
    document.getElementById('reasoningPanel').classList.remove('open');
}

// ============================================================================
// NAVIGATION
// ============================================================================

function showSection(sectionName) {
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    
    const section = document.getElementById(`${sectionName}-section`);
    if (section) section.classList.add('active');
    
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.toggle('active', item.dataset.section === sectionName);
    });
    
    if (sectionName === 'stats') loadStats();
}

// ============================================================================
// FEEDBACK
// ============================================================================

async function giveFeedback(rating) {
    if (messages.length < 2) return;
    
    const lastUserMsg = [...messages].reverse().find(m => m.role === 'user');
    const lastAssistantMsg = [...messages].reverse().find(m => m.role === 'assistant');
    
    if (!lastUserMsg || !lastAssistantMsg) return;
    
    try {
        await fetch('/api/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: lastUserMsg.content,
                response: lastAssistantMsg.content,
                rating: rating > 0 ? 5 : 1
            })
        });
        
        showToast(rating > 0 ? 'Thanks for the feedback!' : 'Feedback recorded');
    } catch (error) {
        console.error('Feedback error:', error);
    }
}

function copyMessage(messageId) {
    const msg = messages.find(m => m.id === messageId);
    if (msg) {
        navigator.clipboard.writeText(msg.content);
        showToast('Copied to clipboard');
    }
}

// ============================================================================
// UTILITIES
// ============================================================================

function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
}

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    if (toast) {
        toast.textContent = message;
        toast.className = `toast show ${type}`;
        
        setTimeout(() => {
            toast.classList.remove('show');
        }, 3000);
    }
}