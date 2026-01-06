"""
API Routes with Neural Network Stats
====================================
REST API endpoints using FastAPI - includes neural network endpoints.
"""

from datetime import datetime
from typing import Optional, List
from pathlib import Path

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter()


# ==================== MODELS ====================

class ChatRequest(BaseModel):
    message: str
    auto_search: bool = True


class SearchRequest(BaseModel):
    query: str


class LearnURLRequest(BaseModel):
    url: str


class TeachRequest(BaseModel):
    content: str
    title: str = "User taught"


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100


# ==================== STATIC FILES ====================

@router.get("/")
async def index():
    """Serve main page"""
    static_path = Path(__file__).parent.parent / "static" / "index.html"
    if static_path.exists():
        return FileResponse(str(static_path))
    raise HTTPException(status_code=404, detail="index.html not found")


# ==================== STATUS ====================

@router.get("/api/status")
async def status():
    """Get system status including Knowledge Graph and Neural Network"""
    from .server import get_components
    c = get_components()
    
    response = {
        'status': 'online',
        'timestamp': datetime.now().isoformat(),
        'stats': c['learner'].get_stats() if c['learner'] else {}
    }
    
    # Add Knowledge Graph stats if available
    if c.get('graph_reasoner'):
        response['knowledge_graph'] = c['graph_reasoner'].get_stats()
    
    # Add Neural Network stats if available
    if c.get('neural_brain'):
        response['neural'] = c['neural_brain'].get_stats()
        response['neural']['available'] = True
    else:
        response['neural'] = {'available': False}
    
    return response


# ==================== CHAT ====================

@router.post("/api/chat")
async def chat(request: ChatRequest):
    """Main chat endpoint - semantic search + response with context"""
    from .server import get_components
    c = get_components()
    
    message = request.message.strip()
    
    if not message:
        raise HTTPException(status_code=400, detail="No message provided")
    
    try:
        # Generate with conversation context
        result = c['response_generator'].generate(message, session_id="default")
        
        if request.auto_search and result.get('needs_search', False):
            return {
                'needs_search': True,
                'response': result.get('response', ''),
                'query': message,
                'stats': c['learner'].get_stats()
            }
        
        return {
            **result,
            'needs_search': False,
            'stats': c['learner'].get_stats()
        }
        
    except Exception as e:
        print(f"Chat error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'response': "I encountered an issue. Please try again.",
            'needs_search': False,
            'confidence': 0,
            'sources': [],
            'error': str(e)
        }


@router.post("/api/chat/clear-context")
async def clear_chat_context():
    """Clear conversation context"""
    from .server import get_components
    c = get_components()
    
    c['response_generator'].clear_context("default")
    return {'status': 'cleared'}


@router.post("/api/chat/search-and-respond")
async def search_and_respond(request: SearchRequest):
    """Search Wikipedia, learn, then respond"""
    from .server import get_components
    c = get_components()
    
    query = request.query.strip()
    
    if not query:
        raise HTTPException(status_code=400, detail="No query provided")
    
    try:
        # Search and learn
        learn_result = c['learner'].search_and_learn(query, max_articles=3)
        
        # Generate response with new knowledge
        result = c['response_generator'].generate_after_learning(
            query,
            learn_result.get('learned_from', [])
        )
        
        return {
            **result,
            'searched': True,
            'sources_count': learn_result.get('count', 0),
            'stats': c['learner'].get_stats()
        }
        
    except Exception as e:
        print(f"Search error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ==================== LEARNING ====================

@router.post("/api/learn/start")
async def start_learning():
    """Start continuous Wikipedia learning"""
    from .server import get_components
    c = get_components()
    
    result = c['learner'].start()
    return result


@router.post("/api/learn/stop")
async def stop_learning():
    """Stop continuous learning and save"""
    from .server import get_components
    c = get_components()
    
    result = c['learner'].stop()
    
    # Ensure data is saved
    c['kb'].save()
    
    return result


@router.post("/api/learn/pause")
async def pause_learning():
    """Pause learning"""
    from .server import get_components
    c = get_components()
    
    return c['learner'].pause()


@router.post("/api/learn/resume")
async def resume_learning():
    """Resume learning"""
    from .server import get_components
    c = get_components()
    
    return c['learner'].resume()


@router.get("/api/learn/status")
async def learning_status():
    """Get learning status"""
    from .server import get_components
    c = get_components()
    
    return c['learner'].get_stats()


@router.post("/api/learn/url")
async def learn_from_url(request: LearnURLRequest):
    """Learn from a specific URL"""
    from .server import get_components
    c = get_components()
    
    url = request.url.strip()
    
    if not url:
        raise HTTPException(status_code=400, detail="No URL provided")
    
    result = c['learner'].learn_from_url(url)
    result['stats'] = c['learner'].get_stats()
    
    # Save after learning
    c['kb'].save()
    
    return result


@router.post("/api/teach")
async def teach(request: TeachRequest):
    """Teach the AI directly"""
    from .server import get_components
    c = get_components()
    
    content = request.content.strip()
    title = request.title
    
    if not content:
        raise HTTPException(status_code=400, detail="No content provided")
    
    if len(content) < 20:
        raise HTTPException(status_code=400, detail="Content too short (min 20 chars)")
    
    knowledge_id, is_new = c['kb'].add_knowledge(
        content=content,
        source_title=title,
        source_url='',
        confidence=0.8
    )
    
    # Also teach the neural network
    if c.get('neural_brain'):
        try:
            c['neural_brain'].learn(content, source=title)
        except Exception as e:
            print(f"Neural teach error: {e}")
    
    # Rebuild embeddings and save
    c['kb'].initialize_embeddings()
    c['kb'].save()
    
    return {
        'success': True,
        'is_new': is_new,
        'knowledge_id': knowledge_id,
        'stats': c['learner'].get_stats()
    }


# ==================== STATS ====================

@router.get("/api/stats")
async def stats():
    """Get all statistics including neural network"""
    from .server import get_components
    c = get_components()
    
    result = c['learner'].get_stats()
    
    # Add neural stats
    if c.get('neural_brain'):
        result['neural'] = c['neural_brain'].get_stats()
        result['neural']['available'] = True
    else:
        result['neural'] = {'available': False}
    
    return result


@router.get("/api/knowledge/recent")
async def recent_knowledge(limit: int = 10):
    """Get recently learned knowledge"""
    from .server import get_components
    c = get_components()
    
    recent = c['kb'].get_recent_knowledge(limit)
    return {'recent': recent}


# ==================== NEURAL NETWORK ====================

@router.get("/api/neural/stats")
async def neural_stats():
    """Get detailed neural network statistics"""
    from .server import get_components
    c = get_components()
    
    if not c.get('neural_brain'):
        return {
            'available': False,
            'message': 'Neural network not available. Install PyTorch: pip install torch'
        }
    
    stats = c['neural_brain'].get_stats()
    stats['available'] = True
    return stats


@router.post("/api/neural/generate")
async def neural_generate(request: GenerateRequest):
    """Generate text using neural network"""
    from .server import get_components
    c = get_components()
    
    if not c.get('neural_brain'):
        raise HTTPException(
            status_code=503, 
            detail="Neural network not available. Install PyTorch: pip install torch"
        )
    
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="No prompt provided")
    
    try:
        result = c['neural_brain'].generate(prompt, max_tokens=request.max_tokens)
        return {
            'prompt': prompt,
            'generated': result,
            'stats': c['neural_brain'].get_stats()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/neural/train")
async def neural_train():
    """Trigger manual neural network training on buffered texts"""
    from .server import get_components
    c = get_components()
    
    if not c.get('neural_brain'):
        raise HTTPException(
            status_code=503, 
            detail="Neural network not available"
        )
    
    try:
        result = c['neural_brain'].train_batch()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== SESSIONS ====================

@router.get("/api/sessions")
async def get_sessions(limit: int = 20):
    """Get learning session history"""
    from .server import get_components
    c = get_components()
    
    sessions = c['learner'].get_session_history(limit)
    summary = c['kb'].get_session_summary()
    
    return {
        'sessions': sessions,
        'summary': summary
    }


@router.get("/api/sessions/summary")
async def get_session_summary():
    """Get session summary statistics"""
    from .server import get_components
    c = get_components()
    
    return c['kb'].get_session_summary()


# ==================== KNOWLEDGE EXPLORER ====================

@router.get("/api/knowledge/all")
async def get_all_knowledge(limit: int = 100):
    """Get all knowledge entries for explorer"""
    from .server import get_components
    c = get_components()
    
    entries = c['kb'].vectors.get_all_knowledge(limit)
    return {'entries': entries, 'count': len(entries)}


@router.get("/api/knowledge/{entry_id}/related")
async def get_related_knowledge(entry_id: int, limit: int = 5):
    """Get knowledge entries related to a specific entry"""
    from .server import get_components
    c = get_components()
    
    related = c['kb'].vectors.get_related(entry_id, limit)
    return {'related': related}


# ==================== WEBSOCKET ====================

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time learning updates"""
    from .server import active_connections
    
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Keep connection alive, receive any messages
            data = await websocket.receive_text()
            # Echo back for ping/pong
            await websocket.send_json({'type': 'pong', 'data': data})
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)
    except Exception:
        if websocket in active_connections:
            active_connections.remove(websocket)
