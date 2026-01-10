"""
GroundZero API Routes v2.7
Enhanced Learning Dashboard with real-time stats
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from pathlib import Path
import asyncio
import sqlite3
from datetime import datetime

router = APIRouter()


# ============================================================
# HELPERS
# ============================================================

def get_components():
    try:
        from . import server
        return getattr(server, '_components', {})
    except:
        return {}

def get_data_dir() -> Path:
    try:
        from config import Settings
        return Path(Settings().data_dir)
    except:
        return Path("./data")


# ============================================================
# MODELS
# ============================================================

class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    confidence: float = 0.0

class TeachRequest(BaseModel):
    knowledge: str
    source: str = "user"


# ============================================================
# SEARCH FUNCTIONS
# ============================================================

def search_vectors_db(query: str, limit: int = 10) -> List[Dict]:
    results = []
    db_path = get_data_dir() / "vectors.db"
    if not db_path.exists():
        return results
    
    words = query.lower().split()[:5]
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        where = ' OR '.join(["LOWER(content) LIKE ?" for _ in words])
        params = [f"%{w}%" for w in words] + [limit]
        
        cursor.execute(f"SELECT content, source_title FROM vectors WHERE {where} LIMIT ?", params)
        for row in cursor.fetchall():
            if row['content'] and len(row['content']) > 20:
                results.append({"type": "knowledge", "content": row['content'][:450], "source": "learned"})
        conn.close()
    except Exception as e:
        print(f"Search error: {e}")
    return results


def search_knowledge_graph(query: str, limit: int = 20) -> List[Dict]:
    results = []
    db_path = get_data_dir() / "knowledge_graph.db"
    if not db_path.exists():
        return results
    
    words = query.lower().split()[:5]
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Facts
        where = ' OR '.join(["(LOWER(subject) LIKE ? OR LOWER(object) LIKE ?)" for _ in words])
        params = []
        for w in words:
            params.extend([f"%{w}%", f"%{w}%"])
        params.append(limit)
        
        cursor.execute(f"SELECT subject, relation, object FROM facts WHERE {where} LIMIT ?", params)
        for row in cursor.fetchall():
            results.append({"type": "fact", "content": f"{row['subject']} → {row['relation']} → {row['object']}", "source": "graph"})
        
        # Definitions
        where = ' OR '.join(["(LOWER(term) LIKE ? OR LOWER(definition) LIKE ?)" for _ in words])
        params = []
        for w in words:
            params.extend([f"%{w}%", f"%{w}%"])
        params.append(limit)
        
        cursor.execute(f"SELECT term, definition FROM definitions WHERE {where} LIMIT ?", params)
        for row in cursor.fetchall():
            if row['term'] and row['definition']:
                results.append({"type": "definition", "content": f"📖 {row['term']}: {row['definition'][:300]}", "source": "definitions"})
        
        conn.close()
    except Exception as e:
        print(f"KG search error: {e}")
    return results


# ============================================================
# COMPREHENSIVE STATS ENDPOINT
# ============================================================

@router.get("/api/stats")
async def get_stats():
    data_dir = get_data_dir()
    components = get_components()
    
    stats = {
        "vectors": 0,
        "facts": 0,
        "definitions": 0,
        "articles_learned": 0,
        "tokens_trained": 0,
        "vocabulary": 0,
        "params": 0
    }
    
    # Vectors
    try:
        conn = sqlite3.connect(str(data_dir / "vectors.db"))
        stats["vectors"] = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
        conn.close()
    except: pass
    
    # Knowledge Graph
    try:
        conn = sqlite3.connect(str(data_dir / "knowledge_graph.db"))
        stats["facts"] = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        stats["definitions"] = conn.execute("SELECT COUNT(*) FROM definitions").fetchone()[0]
        conn.close()
    except: pass
    
    # Knowledge.db
    try:
        conn = sqlite3.connect(str(data_dir / "knowledge.db"))
        stats["articles_learned"] = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        row = conn.execute("SELECT total_words FROM stats LIMIT 1").fetchone()
        if row:
            stats["tokens_trained"] = row[0]
        vocab = conn.execute("SELECT COUNT(*) FROM vocabulary").fetchone()
        if vocab:
            stats["vocabulary"] = vocab[0]
        conn.close()
    except: pass
    
    # Neural params
    if components.get('neural_brain'):
        try:
            nb_stats = components['neural_brain'].get_stats()
            stats["params"] = nb_stats.get('model_params', 0)
        except: pass
    
    stats["knowledge"] = stats["facts"] + stats["definitions"]
    return stats


# ============================================================
# LEARNING STATS - DETAILED REAL-TIME INFO
# ============================================================

@router.get("/api/learning/stats")
async def get_learning_stats():
    """Get detailed learning engine statistics including current article"""
    components = get_components()
    learner = components.get('learner')
    data_dir = get_data_dir()
    
    result = {
        "is_running": False,
        "current_article": None,
        "current_url": None,
        "articles_this_session": 0,
        "words_this_session": 0,
        "articles_learned": 0,
        "tokens_trained": 0,
        "facts": 0,
        "vectors": 0,
        "start_time": None,
        "elapsed_seconds": 0
    }
    
    if learner:
        result["is_running"] = getattr(learner, 'is_running', False)
        result["current_article"] = getattr(learner, 'current_article', None)
        result["current_url"] = getattr(learner, 'current_url', None)
        result["articles_this_session"] = getattr(learner, 'articles_this_session', 0)
        result["words_this_session"] = getattr(learner, 'words_this_session', 0)
        
        # Get start time if running
        start_time = getattr(learner, 'start_time', None)
        if start_time and result["is_running"]:
            result["start_time"] = start_time.isoformat() if hasattr(start_time, 'isoformat') else str(start_time)
            try:
                from datetime import datetime
                if isinstance(start_time, datetime):
                    result["elapsed_seconds"] = int((datetime.now() - start_time).total_seconds())
            except: pass
        
        # Get stats from learner if available
        if hasattr(learner, 'get_stats'):
            try:
                lstats = learner.get_stats()
                result["articles_this_session"] = lstats.get('articles_this_session', result["articles_this_session"])
                result["words_this_session"] = lstats.get('words_this_session', result["words_this_session"])
            except: pass
    
    # Get totals from database
    try:
        conn = sqlite3.connect(str(data_dir / "knowledge.db"))
        result["articles_learned"] = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        row = conn.execute("SELECT total_words FROM stats LIMIT 1").fetchone()
        if row:
            result["tokens_trained"] = row[0]
        conn.close()
    except: pass
    
    try:
        conn = sqlite3.connect(str(data_dir / "knowledge_graph.db"))
        result["facts"] = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        conn.close()
    except: pass
    
    try:
        conn = sqlite3.connect(str(data_dir / "vectors.db"))
        result["vectors"] = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
        conn.close()
    except: pass
    
    return result


@router.get("/api/learning/recent")
async def get_recent_articles(limit: int = 10):
    data_dir = get_data_dir()
    articles = []
    
    try:
        conn = sqlite3.connect(str(data_dir / "knowledge.db"))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT title, url, word_count, learned_at FROM sources ORDER BY learned_at DESC LIMIT ?", (limit,))
        for row in cursor.fetchall():
            articles.append({
                "title": row['title'] or 'Untitled',
                "url": row['url'],
                "word_count": row['word_count'] or 0,
                "learned_at": row['learned_at']
            })
        conn.close()
    except Exception as e:
        print(f"Recent articles error: {e}")
    
    return {"articles": articles}


@router.get("/api/learning/sessions")
async def get_learning_sessions(limit: int = 10):
    data_dir = get_data_dir()
    sessions = []
    
    try:
        conn = sqlite3.connect(str(data_dir / "knowledge.db"))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT started_at, ended_at, duration_seconds, articles_learned, words_learned, status
            FROM learning_sessions ORDER BY started_at DESC LIMIT ?
        """, (limit,))
        
        for row in cursor.fetchall():
            sessions.append({
                "started_at": row['started_at'],
                "ended_at": row['ended_at'],
                "duration_seconds": row['duration_seconds'] or 0,
                "articles_learned": row['articles_learned'] or 0,
                "words_learned": row['words_learned'] or 0,
                "status": row['status'] or 'completed'
            })
        conn.close()
    except Exception as e:
        print(f"Sessions error: {e}")
    
    return {"sessions": sessions}


@router.post("/api/learning/start")
async def start_learning():
    components = get_components()
    learner = components.get('learner')
    
    if not learner:
        raise HTTPException(status_code=503, detail="Learning engine not available")
    
    try:
        # Reset session counters
        if hasattr(learner, 'articles_this_session'):
            learner.articles_this_session = 0
        if hasattr(learner, 'words_this_session'):
            learner.words_this_session = 0
        if hasattr(learner, 'start_time'):
            learner.start_time = datetime.now()
        
        if hasattr(learner, 'start'):
            await asyncio.to_thread(learner.start)
        elif hasattr(learner, 'start_learning'):
            await asyncio.to_thread(learner.start_learning)
        return {"status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/learning/stop")
async def stop_learning():
    components = get_components()
    learner = components.get('learner')
    
    if learner and hasattr(learner, 'stop'):
        learner.stop()
    return {"status": "stopped"}


@router.post("/api/teach")
async def teach(request: TeachRequest):
    components = get_components()
    learner = components.get('learner')
    graph = components.get('graph_reasoner')
    
    if not request.knowledge:
        raise HTTPException(status_code=400, detail="Knowledge required")
    
    added = []
    
    if graph and hasattr(graph, 'add_fact'):
        try:
            graph.add_fact("user_knowledge", "contains", request.knowledge[:200])
            added.append("graph")
        except: pass
    
    if learner and hasattr(learner, 'learn_text'):
        try:
            learner.learn_text(request.knowledge, source=request.source)
            added.append("learner")
        except: pass
    
    return {"status": "success", "added": added}


# ============================================================
# CHAT ENDPOINT
# ============================================================

@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatMessage):
    components = get_components()
    response_gen = components.get('response_generator')
    neural_brain = components.get('neural_brain')
    query = request.message
    
    def is_good(text):
        if not text or len(text) < 5:
            return False
        if text.count('<|unk|>') > 3:
            return False
        return True
    
    # Try response generator
    if response_gen and hasattr(response_gen, 'generate'):
        try:
            result = await asyncio.to_thread(response_gen.generate, query)
            text = result.get('response', '') if isinstance(result, dict) else str(result)
            if is_good(text):
                return ChatResponse(response=text, confidence=0.8)
        except: pass
    
    # Try neural
    if neural_brain and hasattr(neural_brain, 'generate'):
        try:
            result = await asyncio.to_thread(neural_brain.generate, query, max_tokens=100)
            text = str(result) if result else ''
            if is_good(text):
                return ChatResponse(response=text, confidence=0.6)
        except: pass
    
    # Search knowledge
    defs = search_knowledge_graph(query, limit=3)
    definitions = [d for d in defs if d['type'] == 'definition']
    if definitions:
        return ChatResponse(response=definitions[0]['content'], confidence=0.7)
    
    facts = [d for d in defs if d['type'] == 'fact']
    if facts:
        return ChatResponse(response="Here's what I know:\n• " + "\n• ".join(f['content'] for f in facts[:5]), confidence=0.5)
    
    content = search_vectors_db(query, limit=2)
    if content:
        return ChatResponse(response=content[0]['content'][:500], confidence=0.4)
    
    # Fallbacks
    q = query.lower()
    if any(w in q for w in ['hello', 'hi', 'hey']):
        return ChatResponse(response="Hello! I'm GroundZero AI. Ask me anything!", confidence=0.9)
    
    return ChatResponse(response=f"I couldn't find info about '{query}'. Try the Knowledge tab!", confidence=0.2)


# ============================================================
# SEARCH ENDPOINT
# ============================================================

@router.get("/api/search")
async def search(q: str = Query(...), k: int = 10):
    results = []
    results.extend(search_vectors_db(q, limit=k))
    results.extend(search_knowledge_graph(q, limit=k))
    
    # Dedupe
    seen = set()
    unique = []
    for r in results:
        key = r['content'][:80].lower()
        if key not in seen:
            seen.add(key)
            unique.append(r)
    
    return {"query": q, "results": unique[:k], "total": len(unique)}


# ============================================================
# OTHER ENDPOINTS
# ============================================================

@router.get("/")
async def index():
    static_path = Path(__file__).parent.parent / "static" / "index.html"
    if static_path.exists():
        return FileResponse(static_path)
    return {"message": "GroundZero API"}

@router.get("/health")
async def health():
    return {"status": "healthy"}