"""
GroundZero AI - Dashboard Backend
=================================

Flask API with file tools support:
- Chat WITH FILE UPLOAD (NEW!)
- Chat WITH WEB SEARCH (NEW!)
- File upload & document understanding
- Code execution
- File creation
- Knowledge graph
- Stats
"""

import os
import json
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

try:
    from ..utils import logger, timestamp, generate_id, get_data_path, ensure_dir
except ImportError:
    from utils import logger, timestamp, generate_id, get_data_path, ensure_dir


# Upload config
UPLOAD_FOLDER = get_data_path("uploads")
ALLOWED_EXTENSIONS = {
    'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'csv', 'xlsx', 'xls', 
    'docx', 'doc', 'json', 'md', 'py', 'js', 'html', 'xml', 'yaml', 'yml'
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_app(groundzero_ai=None):
    """Create Flask app with GroundZero AI integration."""
    
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'groundzero-secret-key'
    app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
    CORS(app)
    
    ensure_dir(UPLOAD_FOLDER)
    
    # Store reference to AI
    app.groundzero = groundzero_ai
    
    # ========================================================================
    # PAGES
    # ========================================================================
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    # ========================================================================
    # CHAT API - UPGRADED WITH FILE UPLOAD + WEB SEARCH SUPPORT
    # ========================================================================
    
    @app.route('/api/chat', methods=['POST'])
    def chat():
        """
        Chat with GroundZero AI.
        
        UPGRADED: Now supports file uploads AND web search!
        
        Accepts either:
        - JSON body: {"message": "...", "user_id": "...", "search_web": true/false}
        - Multipart form with file: message + file + search_web
        """
        message = None
        user_id = 'default'
        conversation_id = None
        file_content = None
        file_name = None
        search_web = False  # NEW: Web search flag
        
        # Check if this is a file upload (multipart/form-data)
        if request.content_type and 'multipart/form-data' in request.content_type:
            message = request.form.get('message', '')
            user_id = request.form.get('user_id', 'default')
            conversation_id = request.form.get('conversation_id')
            # NEW: Get search_web from form data
            search_web = request.form.get('search_web', 'false').lower() == 'true'
            
            # Handle file upload
            if 'file' in request.files:
                file = request.files['file']
                if file and file.filename and allowed_file(file.filename):
                    file_name = secure_filename(file.filename)
                    
                    # Save the file
                    name, ext = os.path.splitext(file_name)
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    unique_filename = f"{name}_{ts}{ext}"
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                    file.save(filepath)
                    
                    # Read file content
                    try:
                        if app.groundzero and app.groundzero.tools:
                            result = app.groundzero.tools.read_file(filepath)
                            if result.success:
                                doc_id = result.result.get("id")
                                doc = app.groundzero.tools.loaded_docs.get(doc_id)
                                if doc and hasattr(doc, 'raw_content'):
                                    file_content = doc.raw_content
                        
                        if not file_content:
                            # Fallback: read directly
                            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                                file_content = f.read()
                    except Exception as e:
                        logger.error(f"Error reading uploaded file: {e}")
                        try:
                            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                                file_content = f.read()
                        except:
                            file_content = f"[Could not read file: {file_name}]"
        else:
            # Regular JSON request
            data = request.json or {}
            message = data.get('message', '')
            user_id = data.get('user_id', 'default')
            conversation_id = data.get('conversation_id')
            file_content = data.get('file_content')
            file_name = data.get('file_name')
            # NEW: Get search_web from JSON data
            search_web = data.get('search_web', False)
        
        # Validate
        if not message and not file_content:
            return jsonify({'error': 'No message provided'}), 400
        
        if not message and file_content:
            message = "Analyze this file and tell me what it contains"
        
        if app.groundzero:
            try:
                # Call chat with file AND search support
                response, trace = app.groundzero.chat(
                    message,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    return_reasoning=True,
                    file_content=file_content,
                    file_name=file_name,
                    search_web=search_web,  # NEW: Pass search flag
                )
                
                # Build reasoning steps from trace object
                reasoning_steps = []
                if trace and hasattr(trace, 'steps'):
                    reasoning_steps = [
                        {
                            "step": getattr(s, 'step_number', i+1),
                            "thought": getattr(s, 'thought', ''),
                            "type": getattr(s, 'reasoning_type', 'think'),
                            "action": getattr(s, 'action', ''),
                        }
                        for i, s in enumerate(trace.steps)
                    ]
                
                # Format reasoning for frontend
                reasoning_data = None
                if reasoning_steps:
                    reasoning_data = {
                        'steps': [{'step': s.get('step', i+1),
                                   'thought': s.get('thought', ''),
                                   'action': s.get('action') or s.get('type', 'think'),
                                   'result': ''}
                                  for i, s in enumerate(reasoning_steps)],
                        'confidence': getattr(trace, 'confidence', 0.8) if trace else 0.8,
                        'verified': False,
                    }
                
                return jsonify({
                    'response': response,
                    'reasoning': reasoning_data,
                    'timestamp': timestamp(),
                    'file_analyzed': file_name if file_content else None,
                    'web_search_used': search_web,  # NEW: Tell frontend if search was used
                })
                
            except Exception as e:
                logger.error(f"Chat error: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    'response': f"Error: {str(e)}",
                    'reasoning': None,
                    'timestamp': timestamp(),
                })
        else:
            return jsonify({
                'response': "GroundZero AI is not initialized.",
                'reasoning': None,
                'timestamp': timestamp(),
            })
    
    # ========================================================================
    # FILE UPLOAD & DOCUMENT API
    # ========================================================================
    
    @app.route('/api/upload', methods=['POST'])
    def upload_file():
        """Upload and process a file."""
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            
            # Create unique filename with timestamp
            name, ext = os.path.splitext(filename)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_filename = f"{name}_{ts}{ext}"
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            # Process with GroundZero's document understanding
            result = {
                'success': True,
                'filename': unique_filename,
                'filepath': filepath,
                'size': os.path.getsize(filepath),
            }
            
            # Try to read and analyze with tools
            if app.groundzero and app.groundzero.tools:
                try:
                    read_result = app.groundzero.tools.read_file(filepath)
                    if read_result.success:
                        result['document'] = read_result.result
                except Exception as e:
                    logger.warning(f"Could not analyze file: {e}")
            
            return jsonify(result)
        
        return jsonify({'error': 'File type not allowed'}), 400
    
    @app.route('/api/documents', methods=['GET'])
    def list_documents():
        """List loaded documents."""
        if app.groundzero and app.groundzero.tools:
            docs = app.groundzero.get_loaded_documents()
            return jsonify({'documents': docs})
        return jsonify({'documents': []})
    
    @app.route('/api/documents/<doc_id>', methods=['GET', 'DELETE'])
    def document_actions(doc_id):
        """Get or delete a document."""
        if request.method == 'DELETE':
            if app.groundzero and app.groundzero.tools:
                if doc_id in app.groundzero.tools.loaded_docs:
                    del app.groundzero.tools.loaded_docs[doc_id]
                    return jsonify({'status': 'deleted'})
            return jsonify({'error': 'Document not found'}), 404
        
        # GET
        if app.groundzero and app.groundzero.tools:
            doc = app.groundzero.tools.loaded_docs.get(doc_id)
            if doc:
                return jsonify({
                    'id': doc_id,
                    'filename': doc.filename,
                    'type': doc.file_type,
                    'preview': doc.raw_content[:1000] if doc.raw_content else '',
                })
        return jsonify({'error': 'Document not found'}), 404
    
    # ========================================================================
    # CODE EXECUTION API
    # ========================================================================
    
    @app.route('/api/code/execute', methods=['POST'])
    def execute_code():
        """Execute Python code."""
        data = request.json
        code = data.get('code', '')
        
        if not code:
            return jsonify({'error': 'No code provided'}), 400
        
        if app.groundzero:
            result = app.groundzero.run_code(code)
            return jsonify(result)
        
        return jsonify({'error': 'AI not initialized'}), 500
    
    # ========================================================================
    # FILE CREATION API
    # ========================================================================
    
    @app.route('/api/files/create', methods=['POST'])
    def create_file_api():
        """Create a document file."""
        data = request.json
        filename = data.get('filename', '')
        content = data.get('content', '')
        file_type = data.get('type', 'auto')
        
        if not filename or not content:
            return jsonify({'error': 'Filename and content required'}), 400
        
        if app.groundzero:
            try:
                ext = Path(filename).suffix.lower()
                
                if ext == '.docx':
                    filepath = app.groundzero.create_word(filename, content, title=data.get('title'))
                elif ext == '.xlsx':
                    filepath = app.groundzero.create_excel(filename, content, sheet_name=data.get('sheet', 'Sheet1'))
                elif ext == '.pdf':
                    filepath = app.groundzero.create_pdf(filename, content, title=data.get('title'))
                elif ext == '.pptx':
                    filepath = app.groundzero.create_powerpoint(filename, content, title=data.get('title'))
                else:
                    # Text file
                    result = app.groundzero.tools.create_file(filename, content)
                    filepath = result.result
                
                return jsonify({
                    'success': True,
                    'filepath': str(filepath),
                    'filename': filename,
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        return jsonify({'error': 'AI not initialized'}), 500
    
    @app.route('/api/files/list', methods=['GET'])
    def list_files():
        """List files in workspace."""
        if app.groundzero and app.groundzero.tools:
            workspace = Path(app.groundzero.tools.workspace)
            outputs_dir = workspace / "outputs"
            
            files = []
            if outputs_dir.exists():
                for f in outputs_dir.glob("*"):
                    if f.is_file():
                        files.append({
                            'name': f.name,
                            'size': f.stat().st_size,
                            'modified': datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                            'type': f.suffix[1:] if f.suffix else 'unknown',
                        })
            
            return jsonify({'files': files})
        
        return jsonify({'files': []})
    
    @app.route('/api/files/download/<filename>', methods=['GET'])
    def download_file(filename):
        """Download a created file."""
        if app.groundzero and app.groundzero.tools:
            workspace = Path(app.groundzero.tools.workspace)
            filepath = workspace / "outputs" / secure_filename(filename)
            
            if filepath.exists():
                return send_file(filepath, as_attachment=True)
        
        return jsonify({'error': 'File not found'}), 404
    
    # ========================================================================
    # DATA ANALYSIS API
    # ========================================================================
    
    @app.route('/api/analyze', methods=['POST'])
    def analyze_data():
        """Analyze uploaded data."""
        data = request.json
        filepath = data.get('filepath', '')
        code = data.get('code', '')
        
        if not filepath:
            return jsonify({'error': 'No file specified'}), 400
        
        if app.groundzero:
            result = app.groundzero.run_code(code or f"import pandas as pd; df = pd.read_csv('{filepath}'); print(df.describe())")
            return jsonify(result)
        
        return jsonify({'error': 'AI not initialized'}), 500
    
    # ========================================================================
    # KNOWLEDGE API
    # ========================================================================
    
    @app.route('/api/knowledge/search', methods=['GET'])
    def search_knowledge():
        query = request.args.get('q', '')
        
        if app.groundzero:
            results = app.groundzero.knowledge_graph.search(query)
            return jsonify({'results': results})
        
        return jsonify({'results': []})
    
    @app.route('/api/knowledge/graph', methods=['GET'])
    def get_knowledge_graph():
        """Get knowledge graph for visualization."""
        if app.groundzero:
            stats = app.groundzero.knowledge_graph.get_stats()
            nodes = []
            edges = []
            
            # Get nodes
            for node_id, node in list(app.groundzero.knowledge_graph.nodes.items())[:100]:
                nodes.append({
                    'id': node_id,
                    'name': node.name,
                    'type': node.node_type,
                    'confidence': node.confidence,
                })
            
            # Get edges
            for edge in list(app.groundzero.knowledge_graph.edges)[:200]:
                edges.append({
                    'source': edge.source,
                    'target': edge.target,
                    'relation': edge.relation,
                })
            
            return jsonify({
                'nodes': nodes,
                'edges': edges,
                'stats': stats,
            })
        
        return jsonify({'nodes': [], 'edges': [], 'stats': {}})
    
    @app.route('/api/knowledge/teach', methods=['POST'])
    def teach_knowledge():
        """Teach new knowledge."""
        data = request.json
        subject = data.get('subject', '')
        content = data.get('content', '')
        
        if not subject or not content:
            return jsonify({'error': 'Subject and content required'}), 400
        
        if app.groundzero:
            result = app.groundzero.teach(subject, content)
            return jsonify(result)
        
        return jsonify({'error': 'AI not initialized'}), 500
    
    # ========================================================================
    # STATS API
    # ========================================================================
    
    @app.route('/api/stats', methods=['GET'])
    def get_stats():
        if app.groundzero:
            stats = app.groundzero.get_stats()
            return jsonify(stats)
        
        return jsonify({
            'model': {'name': 'Not initialized'},
            'knowledge': {'total_nodes': 0},
            'memory': {'conversations': 0},
        })
    
    # ========================================================================
    # FEEDBACK API
    # ========================================================================
    
    @app.route('/api/feedback', methods=['POST'])
    def feedback():
        data = request.json
        prompt = data.get('prompt', '')
        response = data.get('response', '')
        rating = data.get('rating', 3)
        correction = data.get('correction')
        
        if app.groundzero:
            if correction:
                app.groundzero.handle_correction(response, correction)
                return jsonify({'status': 'correction_recorded'})
            else:
                # Store feedback for learning
                return jsonify({'status': 'feedback_recorded'})
        
        return jsonify({'status': 'no_ai'})
    
    # ========================================================================
    # USER API
    # ========================================================================
    
    @app.route('/api/user', methods=['GET', 'POST'])
    def user_profile():
        if request.method == 'GET':
            if app.groundzero:
                user = app.groundzero.memory.get_current_user()
                if user:
                    return jsonify({
                        'id': user.id if hasattr(user, 'id') else 'default',
                        'name': user.name,
                        'preferences': user.preferences.__dict__ if user.preferences else {},
                    })
            return jsonify({'id': 'default', 'name': 'User'})
        
        else:  # POST
            data = request.json
            if app.groundzero:
                user = app.groundzero.memory.get_current_user()
                if user and data.get('name'):
                    user.name = data['name']
                    app.groundzero.memory.users.save_all()
                    return jsonify({'status': 'updated'})
            return jsonify({'status': 'no_change'})
    
    return app


def run_dashboard(groundzero_ai=None, host='0.0.0.0', port=8080):
    """Run the dashboard server."""
    app = create_app(groundzero_ai)
    logger.info(f"Starting GroundZero Dashboard at http://{host}:{port}")
    app.run(host=host, port=port, debug=False)


__all__ = ['create_app', 'run_dashboard']