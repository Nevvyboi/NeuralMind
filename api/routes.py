"""
API Routes with Neural Network Stats
====================================
REST API endpoints using FastAPI - includes neural network endpoints.
"""

import asyncio
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
        # Generate with conversation context (run in thread pool)
        result = await asyncio.to_thread(
            c['response_generator'].generate, 
            message, 
            "default"  # session_id
        )
        
        if request.auto_search and result.get('needs_search', False):
            stats = await asyncio.to_thread(c['learner'].get_stats)
            return {
                'needs_search': True,
                'response': result.get('response', ''),
                'query': message,
                'stats': stats
            }
        
        stats = await asyncio.to_thread(c['learner'].get_stats)
        return {
            **result,
            'needs_search': False,
            'stats': stats
        }
        
    except asyncio.CancelledError:
        # Re-raise to let framework handle gracefully during shutdown
        raise
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
        # Search and learn (run in thread pool)
        learn_result = await asyncio.to_thread(
            c['learner'].search_and_learn, 
            query, 
            3  # max_articles
        )
        
        # Generate response with new knowledge
        result = await asyncio.to_thread(
            c['response_generator'].generate_after_learning,
            query,
            learn_result.get('learned_from', [])
        )
        
        stats = await asyncio.to_thread(c['learner'].get_stats)
        
        return {
            **result,
            'searched': True,
            'sources_count': learn_result.get('count', 0),
            'stats': stats
        }
        
    except asyncio.CancelledError:
        raise
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
    """Learn from a specific URL - runs in background so it won't be cancelled"""
    from .server import get_components
    import concurrent.futures
    import threading
    
    c = get_components()
    
    url = request.url.strip()
    
    if not url:
        raise HTTPException(status_code=400, detail="No URL provided")
    
    # Use a separate thread pool that won't be cancelled with the request
    def do_learning():
        """Run learning in background thread"""
        try:
            result = c['learner'].learn_from_url(url)
            # Save after learning
            c['kb'].save()
            print(f"✅ Background learning complete: {url[:50]}...")
            return result
        except Exception as e:
            print(f"❌ Background learning error for {url[:50]}: {e}")
            return {'success': False, 'error': str(e)}
    
    # Start background thread (fire and forget)
    thread = threading.Thread(target=do_learning, daemon=True)
    thread.start()
    
    # Return immediately - don't wait for completion
    return {
        'success': True,
        'status': 'processing',
        'message': 'Learning started in background',
        'url': url
    }


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
    
    try:
        knowledge_id, is_new = await asyncio.to_thread(
            c['kb'].add_knowledge,
            content,
            title,
            '',  # source_url
            0.8  # confidence
        )
        
        # Also teach the neural network
        if c.get('neural_brain'):
            try:
                await asyncio.to_thread(c['neural_brain'].learn, content, title)
            except Exception as e:
                print(f"Neural teach error: {e}")
        
        # Rebuild embeddings and save
        await asyncio.to_thread(c['kb'].initialize_embeddings)
        await asyncio.to_thread(c['kb'].save)
        
        stats = await asyncio.to_thread(c['learner'].get_stats)
        
        return {
            'success': True,
            'is_new': is_new,
            'knowledge_id': knowledge_id,
            'stats': stats
        }
        
    except asyncio.CancelledError:
        raise
    except Exception as e:
        print(f"Error teaching: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
        result = await asyncio.to_thread(c['neural_brain'].generate, prompt, request.max_tokens)
        return {
            'prompt': prompt,
            'generated': result,
            'stats': c['neural_brain'].get_stats()
        }
    except asyncio.CancelledError:
        raise
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
        result = await asyncio.to_thread(c['neural_brain'].train_batch)
        return result
    except asyncio.CancelledError:
        raise
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


# ==================== TOPIC SEARCH ====================

class TopicSearchRequest(BaseModel):
    query: str
    limit: int = 20


@router.post("/api/knowledge/search-topics")
async def search_topics(request: TopicSearchRequest):
    """
    Search knowledge base for topics with detailed statistics.
    Returns matching entries with relevance scores, word counts, and metadata.
    """
    from .server import get_components
    c = get_components()
    
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="No query provided")
    
    try:
        # Search using the knowledge base
        results = await asyncio.to_thread(c['kb'].search, query, request.limit)
        
        # Get overall stats
        kb_stats = c['kb'].get_statistics()
        
        # Calculate search statistics
        total_relevance = sum(r.get('relevance', 0) for r in results)
        avg_relevance = total_relevance / len(results) if results else 0
        
        # Get word counts for results
        total_words_in_results = 0
        enriched_results = []
        
        for r in results:
            content = r.get('content', '')
            word_count = len(content.split()) if content else 0
            total_words_in_results += word_count
            
            enriched_results.append({
                'id': r.get('id'),
                'title': r.get('source_title', 'Unknown'),
                'url': r.get('source_url', ''),
                'relevance': round(r.get('relevance', 0) * 100, 1),
                'confidence': round(r.get('confidence', 0) * 100, 1),
                'word_count': word_count,
                'preview': content[:300] + '...' if len(content) > 300 else content,
                'created_at': r.get('created_at', '')
            })
        
        return {
            'query': query,
            'results': enriched_results,
            'stats': {
                'total_results': len(results),
                'avg_relevance': round(avg_relevance * 100, 1),
                'total_words_in_results': total_words_in_results,
                'knowledge_base': {
                    'total_entries': kb_stats.get('total_knowledge', 0),
                    'total_sources': kb_stats.get('total_sources', 0),
                    'total_words': kb_stats.get('total_words', 0),
                    'vocabulary_size': kb_stats.get('vocabulary_size', 0)
                }
            }
        }
        
    except Exception as e:
        print(f"Topic search error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ==================== SYSTEM MONITOR ====================

@router.get("/api/system/monitor")
async def system_monitor():
    """
    Get real-time system resource usage for the application.
    Includes CPU, memory, and component-specific stats.
    """
    import os
    import sys
    
    from .server import get_components
    c = get_components()
    
    # Try to import psutil for system stats
    try:
        import psutil
        PSUTIL_AVAILABLE = True
    except ImportError:
        PSUTIL_AVAILABLE = False
    
    result = {
        'timestamp': datetime.now().isoformat(),
        'psutil_available': PSUTIL_AVAILABLE
    }
    
    # System-wide stats
    if PSUTIL_AVAILABLE:
        # Current process
        process = psutil.Process(os.getpid())
        
        # Memory info
        mem_info = process.memory_info()
        result['process'] = {
            'pid': os.getpid(),
            'memory_mb': round(mem_info.rss / 1024 / 1024, 1),
            'memory_percent': round(process.memory_percent(), 1),
            'cpu_percent': round(process.cpu_percent(interval=0.1), 1),
            'threads': process.num_threads(),
            'open_files': len(process.open_files()) if hasattr(process, 'open_files') else 0
        }
        
        # System-wide
        result['system'] = {
            'cpu_percent': round(psutil.cpu_percent(interval=0.1), 1),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': round(psutil.virtual_memory().total / 1024 / 1024 / 1024, 1),
            'memory_available_gb': round(psutil.virtual_memory().available / 1024 / 1024 / 1024, 1),
            'memory_percent': round(psutil.virtual_memory().percent, 1),
            'disk_percent': round(psutil.disk_usage('/').percent, 1) if os.name != 'nt' else round(psutil.disk_usage('C:\\').percent, 1)
        }
    else:
        result['process'] = {'error': 'psutil not installed - run: pip install psutil'}
        result['system'] = {'error': 'psutil not installed'}
    
    # Python info
    result['python'] = {
        'version': sys.version.split()[0],
        'platform': sys.platform
    }
    
    # Knowledge Base stats
    try:
        kb_stats = c['kb'].get_statistics()
        vector_stats = c['kb'].vectors.get_stats()
        embed_stats = c['kb'].embeddings.get_stats()
        
        result['knowledge_base'] = {
            'total_entries': kb_stats.get('total_knowledge', 0),
            'total_sources': kb_stats.get('total_sources', 0),
            'total_words': kb_stats.get('total_words', 0),
            'vocabulary_size': kb_stats.get('vocabulary_size', 0),
            'vector_dimension': vector_stats.get('dimension', 0),
            'index_type': vector_stats.get('index_type', 'Unknown'),
            'faiss_available': vector_stats.get('faiss_available', False)
        }
        
        result['embeddings'] = {
            'vocabulary_size': embed_stats.get('vocabulary_size', 0),
            'dimension': embed_stats.get('dimension', 0),
            'total_documents': embed_stats.get('total_documents', 0)
        }
    except Exception as e:
        result['knowledge_base'] = {'error': str(e)}
    
    # Neural Network stats
    if c.get('neural_brain'):
        try:
            neural_stats = c['neural_brain'].get_stats()
            
            # Estimate memory usage
            params = neural_stats.get('parameters', 0)
            # Each parameter is typically 4 bytes (float32)
            estimated_model_mb = (params * 4) / 1024 / 1024
            
            result['neural_network'] = {
                'available': True,
                'parameters': params,
                'parameters_formatted': f"{params:,}",
                'estimated_memory_mb': round(estimated_model_mb, 1),
                'tokens_trained': neural_stats.get('tokens_trained', 0),
                'vocabulary_size': neural_stats.get('vocab_size', 0),
                'model_type': neural_stats.get('model_type', 'Transformer'),
                'embedding_dim': neural_stats.get('embedding_dim', 0),
                'num_heads': neural_stats.get('num_heads', 0),
                'num_layers': neural_stats.get('num_layers', 0),
                'batch_size': neural_stats.get('batch_size', 0),
                'learning_rate': neural_stats.get('learning_rate', 0),
                'buffer_size': neural_stats.get('buffer_size', 0),
                'last_loss': neural_stats.get('last_loss', 0)
            }
        except Exception as e:
            result['neural_network'] = {'available': True, 'error': str(e)}
    else:
        result['neural_network'] = {'available': False}
    
    # Knowledge Graph stats
    if c.get('graph_reasoner'):
        try:
            graph_stats = c['graph_reasoner'].get_stats()
            result['knowledge_graph'] = {
                'available': True,
                'total_facts': graph_stats.get('total_facts', 0),
                'total_entities': graph_stats.get('total_entities', 0),
                'total_relations': graph_stats.get('total_relations', 0)
            }
        except Exception as e:
            result['knowledge_graph'] = {'available': True, 'error': str(e)}
    else:
        result['knowledge_graph'] = {'available': False}
    
    # Strategic Learning stats
    if c.get('learner') and hasattr(c['learner'], 'strategic') and c['learner'].strategic:
        try:
            strategic_stats = c['learner'].strategic.get_stats()
            result['strategic_learning'] = {
                'available': True,
                'learned_titles': strategic_stats.get('learned_titles', 0),
                'vital_progress': strategic_stats.get('vital_progress', {}),
                'categories_queued': strategic_stats.get('categories_queued', 0),
                'next_source': strategic_stats.get('next_source', 'unknown')
            }
        except Exception as e:
            result['strategic_learning'] = {'available': True, 'error': str(e)}
    else:
        result['strategic_learning'] = {'available': False}
    
    # GPU/CUDA info if available
    try:
        import torch
        result['gpu'] = {
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        if torch.cuda.is_available():
            result['gpu']['device_name'] = torch.cuda.get_device_name(0)
            result['gpu']['memory_allocated_mb'] = round(torch.cuda.memory_allocated(0) / 1024 / 1024, 1)
            result['gpu']['memory_reserved_mb'] = round(torch.cuda.memory_reserved(0) / 1024 / 1024, 1)
    except ImportError:
        result['gpu'] = {'available': False, 'message': 'PyTorch not installed'}
    except Exception as e:
        result['gpu'] = {'available': False, 'error': str(e)}
    
    # Recommendations based on usage
    recommendations = []
    
    if PSUTIL_AVAILABLE:
        mem_percent = result['process'].get('memory_percent', 0)
        if mem_percent > 50:
            recommendations.append({
                'type': 'warning',
                'message': f'High memory usage ({mem_percent}%). Consider using smaller model.'
            })
        elif mem_percent < 20:
            recommendations.append({
                'type': 'success',
                'message': f'Memory usage is low ({mem_percent}%). You could use a larger model.'
            })
    
    if c.get('neural_brain'):
        params = result.get('neural_network', {}).get('parameters', 0)
        if params > 10_000_000:
            recommendations.append({
                'type': 'info',
                'message': f'Large model ({params:,} params). Good for quality, may be slow.'
            })
        elif params < 1_000_000:
            recommendations.append({
                'type': 'info',
                'message': f'Small model ({params:,} params). Fast but limited capacity.'
            })
    
    result['recommendations'] = recommendations
    
    return result


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