/**
 * GroundZero AI - Dashboard JavaScript
 * With File Upload in Chat Support (UPGRADED!)
 */

// ============================================================================
// STATE
// ============================================================================

let conversationId = null;
let messages = [];
let isLoading = false;
let currentReasoning = null;
let userName = 'User';
let loadedDocuments = [];
let pendingFile = null;  // NEW: Track file waiting to be sent

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    loadUserProfile();
    loadStats();
    refreshDocuments();
    refreshFiles();
    
    // Set up upload area click
    const uploadArea = document.getElementById('uploadArea');
    if (uploadArea) {
        uploadArea.addEventListener('click', () => {
            document.getElementById('fileUpload2').click();
        });
    }
    
    // NEW: Initialize chat file upload
    initChatFileUpload();
});

// ============================================================================
// CHAT FILE UPLOAD - NEW!
// ============================================================================

function initChatFileUpload() {
    const messagesContainer = document.getElementById('messagesContainer');
    if (!messagesContainer) return;
    
    // Drag and drop for chat
    messagesContainer.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        messagesContainer.classList.add('drag-over');
    });
    
    messagesContainer.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        messagesContainer.classList.remove('drag-over');
    });
    
    messagesContainer.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        messagesContainer.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleChatFile(files[0]);
        }
    });
}

function handleChatFile(file) {
    if (!file) return;
    
    pendingFile = file;
    showFilePreview(file);
    
    // Focus on input
    const input = document.getElementById('messageInput');
    if (input) {
        input.placeholder = `Ask about ${file.name}...`;
        input.focus();
    }
}

function showFilePreview(file) {
    // Remove existing preview
    clearFilePreview();
    
    // Create preview element
    const preview = document.createElement('div');
    preview.id = 'chatFilePreview';
    preview.className = 'chat-file-preview';
    preview.innerHTML = `
        <span class="file-icon">${getFileIcon(file.name)}</span>
        <span class="file-name">${file.name}</span>
        <span class="file-size">(${formatFileSize(file.size)})</span>
        <button onclick="clearFilePreview()" class="clear-file" title="Remove file">✕</button>
    `;
    
    // Insert before input container
    const inputContainer = document.querySelector('.chat-input-wrapper') || 
                          document.querySelector('.chat-input-container') ||
                          document.getElementById('messageInput').parentElement;
    
    if (inputContainer) {
        inputContainer.insertBefore(preview, inputContainer.firstChild);
    }
}

function clearFilePreview() {
    pendingFile = null;
    
    const preview = document.getElementById('chatFilePreview');
    if (preview) preview.remove();
    
    const input = document.getElementById('messageInput');
    if (input) {
        input.placeholder = 'Type a message... (or upload a file to analyze)';
    }
    
    const fileInput = document.getElementById('chatFileInput');
    if (fileInput) fileInput.value = '';
}

function triggerFileUpload() {
    let fileInput = document.getElementById('chatFileInput');
    
    if (!fileInput) {
        // Create hidden file input if it doesn't exist
        fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.id = 'chatFileInput';
        fileInput.style.display = 'none';
        fileInput.accept = '.csv,.xlsx,.xls,.pdf,.txt,.json,.docx,.py,.js,.html,.md';
        fileInput.onchange = (e) => {
            if (e.target.files.length > 0) {
                handleChatFile(e.target.files[0]);
            }
        };
        document.body.appendChild(fileInput);
    }
    
    fileInput.click();
}

function getFileIcon(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    const icons = {
        'csv': '📊',
        'xlsx': '📊',
        'xls': '📊',
        'pdf': '📕',
        'txt': '📄',
        'json': '📋',
        'docx': '📝',
        'doc': '📝',
        'py': '🐍',
        'js': '📜',
        'html': '🌐',
        'md': '📑',
    };
    return icons[ext] || '📎';
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// ============================================================================
// CHAT FUNCTIONS - UPGRADED WITH FILE SUPPORT
// ============================================================================

async function sendMessage() {
    const input = document.getElementById('messageInput');
    const message = input.value.trim();
    const file = pendingFile;
    
    if (!message && !file) return;
    if (isLoading) return;
    
    // Build user message display
    let userMessageText = message || 'Analyze this file';
    if (file) {
        userMessageText = `📎 ${file.name}\n\n${message || 'Analyze this file and tell me what it contains'}`;
    }
    
    // Add user message to UI
    addMessage('user', userMessageText);
    input.value = '';
    autoResize(input);
    
    // Clear file preview
    const hadFile = !!file;
    const fileName = file ? file.name : null;
    clearFilePreview();
    
    // Show loading
    isLoading = true;
    document.getElementById('sendBtn').disabled = true;
    const loadingId = addMessage('assistant', '...', true);
    
    try {
        let response;
        
        if (hadFile && file) {
            // Send with file (multipart form)
            const formData = new FormData();
            formData.append('message', message || 'Analyze this file and tell me what it contains');
            formData.append('file', file);
            formData.append('user_id', 'default');
            if (conversationId) {
                formData.append('conversation_id', conversationId);
            }
            
            response = await fetch('/api/chat', {
                method: 'POST',
                body: formData
            });
        } else {
            // Regular chat (JSON)
            response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: message,
                    conversation_id: conversationId,
                    user_id: 'default'
                })
            });
        }
        
        const data = await response.json();
        
        // Remove loading message
        removeMessage(loadingId);
        
        // Build response with file indicator
        let responseText = data.response;
        if (data.file_analyzed) {
            responseText = `📄 Analyzed: **${data.file_analyzed}**\n\n${responseText}`;
        }
        
        // Add response
        addMessage('assistant', responseText, false, data.reasoning);
        
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
        reasoningBtn = `<button class="reasoning-btn" onclick="showReasoning(${JSON.stringify(reasoning).replace(/"/g, '&quot;')})">
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
    
    // Code blocks
    text = text.replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code class="language-$1">$2</code></pre>');
    
    // Inline code
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Bold
    text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    
    // Line breaks
    text = text.replace(/\n/g, '<br>');
    
    return text;
}

function newChat() {
    conversationId = null;
    messages = [];
    pendingFile = null;
    clearFilePreview();
    
    document.getElementById('messagesContainer').innerHTML = `
        <div class="welcome-message">
            <h2>Welcome to GroundZero AI! 🧠</h2>
            <p>I'm your personal AI assistant that learns and grows with you.</p>
            <div class="capabilities">
                <div class="capability"><span>💬</span><span>Chat & Learn</span></div>
                <div class="capability"><span>📄</span><span>Read Documents</span></div>
                <div class="capability"><span>💻</span><span>Run Code</span></div>
                <div class="capability"><span>📝</span><span>Create Files</span></div>
            </div>
            <p class="upload-hint">📎 <strong>NEW!</strong> Drag & drop files here or click the 📎 button to analyze documents!</p>
        </div>
    `;
    showSection('chat');
}

// ============================================================================
// FILE UPLOAD & DOCUMENTS
// ============================================================================

async function uploadFile(file) {
    if (!file) return;
    
    showToast(`Uploading ${file.name}...`);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            showToast(`Loaded: ${file.name}`);
            refreshDocuments();
            
            // If in chat, show message about loaded doc
            if (document.getElementById('chat-section').classList.contains('active')) {
                addMessage('assistant', `📄 Loaded document: **${file.name}**\n\nYou can now ask me questions about this document!`);
            }
        } else {
            showToast(`Error: ${data.error}`, 'error');
        }
    } catch (error) {
        showToast('Upload failed', 'error');
        console.error('Upload error:', error);
    }
}

function handleDrop(e) {
    e.preventDefault();
    e.target.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    for (let file of files) {
        uploadFile(file);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    e.target.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.target.classList.remove('drag-over');
}

async function refreshDocuments() {
    try {
        const response = await fetch('/api/documents');
        const data = await response.json();
        
        loadedDocuments = data.documents || [];
        
        const container = document.getElementById('loadedDocuments');
        if (container) {
            if (loadedDocuments.length === 0) {
                container.innerHTML = '<p class="no-docs">No documents loaded yet</p>';
            } else {
                container.innerHTML = loadedDocuments.map(doc => `
                    <div class="loaded-doc" onclick="selectDocument('${doc.id}')">
                        <span class="doc-icon">${getFileIcon(doc.filename)}</span>
                        <span class="doc-name">${doc.filename}</span>
                        <span class="doc-info">${doc.word_count || 0} words</span>
                    </div>
                `).join('');
            }
        }
    } catch (error) {
        console.error('Error refreshing documents:', error);
    }
}

async function askDocuments() {
    const input = document.getElementById('docQuestion');
    const question = input.value.trim();
    
    if (!question) return;
    
    // Send to chat with document context
    addMessage('user', `📄 Question about documents: ${question}`);
    input.value = '';
    
    isLoading = true;
    const loadingId = addMessage('assistant', '...', true);
    
    try {
        const response = await fetch('/api/documents/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });
        
        const data = await response.json();
        
        removeMessage(loadingId);
        addMessage('assistant', data.answer || data.error || 'No answer found');
        
    } catch (error) {
        removeMessage(loadingId);
        addMessage('assistant', 'Error asking about documents');
    }
    
    isLoading = false;
    showSection('chat');
}

// ============================================================================
// CODE EXECUTION
// ============================================================================

async function runCode() {
    const codeInput = document.getElementById('codeInput');
    const code = codeInput.value.trim();
    const language = document.getElementById('codeLanguage')?.value || 'python';
    
    if (!code) return;
    
    const outputDiv = document.getElementById('codeOutput');
    outputDiv.innerHTML = '<p class="running">Running...</p>';
    
    try {
        const response = await fetch('/api/code/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code, language })
        });
        
        const data = await response.json();
        
        if (data.success) {
            outputDiv.innerHTML = `<pre class="success">${escapeHtml(data.output)}</pre>`;
        } else {
            outputDiv.innerHTML = `<pre class="error">${escapeHtml(data.error || 'Unknown error')}</pre>`;
        }
    } catch (error) {
        outputDiv.innerHTML = `<pre class="error">Error: ${error.message}</pre>`;
    }
}

function clearCode() {
    document.getElementById('codeInput').value = '';
    document.getElementById('codeOutput').innerHTML = '<p class="placeholder">Output will appear here...</p>';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================================================
// FILE CREATION
// ============================================================================

async function createFile() {
    const filename = document.getElementById('createFilename').value.trim();
    const content = document.getElementById('createContent').value.trim();
    const fileType = document.getElementById('createFileType').value;
    
    if (!filename) {
        showToast('Enter a filename', 'error');
        return;
    }
    
    // Ensure proper extension
    let finalFilename = filename;
    if (!filename.includes('.')) {
        finalFilename = filename + '.' + fileType;
    }
    
    try {
        const response = await fetch('/api/files/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filename: finalFilename,
                content: content,
                type: fileType
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showToast(`Created: ${finalFilename}`);
            document.getElementById('createFilename').value = '';
            document.getElementById('createContent').value = '';
            refreshFiles();
        } else {
            showToast(`Error: ${data.error}`, 'error');
        }
    } catch (error) {
        showToast('Failed to create file', 'error');
    }
}

async function refreshFiles() {
    try {
        const response = await fetch('/api/files/list');
        const data = await response.json();
        
        const container = document.getElementById('createdFiles');
        if (container) {
            if (!data.files || data.files.length === 0) {
                container.innerHTML = '<p class="no-files">No files created yet</p>';
            } else {
                container.innerHTML = data.files.map(f => `
                    <div class="created-file">
                        <span class="file-icon">${getFileIcon(f.name)}</span>
                        <span class="file-name">${f.name}</span>
                        <a href="/api/files/download/${f.name}" class="download-btn" download>⬇️</a>
                    </div>
                `).join('');
            }
        }
    } catch (error) {
        console.error('Error refreshing files:', error);
    }
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
                    <span class="confidence">Confidence: ${(r.confidence * 100).toFixed(0)}%</span>
                </div>
            `).join('');
        } else {
            container.innerHTML = '<p>No results found</p>';
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
                    <p class="stat-label">Version: ${stats.model?.version || 'N/A'}</p>
                    <p class="stat-label">Status: ${stats.model?.loaded ? 'Loaded' : 'Mock Mode'}</p>
                </div>
                <div class="stat-card">
                    <h3>📚 Knowledge</h3>
                    <p class="stat-value">${stats.knowledge?.total_nodes || 0}</p>
                    <p class="stat-label">Nodes</p>
                    <p class="stat-label">Edges: ${stats.knowledge?.total_edges || 0}</p>
                </div>
                <div class="stat-card">
                    <h3>💾 Memory</h3>
                    <p class="stat-value">${stats.memory?.conversations || 0}</p>
                    <p class="stat-label">Conversations</p>
                    <p class="stat-label">User: ${stats.memory?.current_user || 'default'}</p>
                </div>
                <div class="stat-card">
                    <h3>📈 Learning</h3>
                    <p class="stat-value">${stats.learning?.queue_size || 0}</p>
                    <p class="stat-label">Items in Queue</p>
                </div>
                <div class="stat-card">
                    <h3>🛠️ Tools</h3>
                    <p class="stat-value">${stats.tools?.total || 0}</p>
                    <p class="stat-label">Executions</p>
                    <p class="stat-label">Documents: ${loadedDocuments.length}</p>
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
// REASONING
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
                            ${step.result ? `<p class="step-result">Result: ${step.result}</p>` : ''}
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
    // Hide all sections
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    
    // Show selected section
    const section = document.getElementById(`${sectionName}-section`);
    if (section) section.classList.add('active');
    
    // Update nav
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.toggle('active', item.dataset.section === sectionName);
    });
    
    // Load section-specific data
    if (sectionName === 'stats') loadStats();
    if (sectionName === 'files') {
        refreshDocuments();
        refreshFiles();
    }
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
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
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