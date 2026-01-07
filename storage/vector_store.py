"""
Vector Store - FIXED VERSION with proper subject extraction
===========================================================
"""

import numpy as np
import sqlite3
import threading
import json
import hashlib
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS not available. Install with: pip install faiss-cpu")


class VectorStore:
    """Vector database with FAISS and smart title matching."""
    
    def __init__(self, data_dir: Path, dimension: int = 256):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.dimension = dimension
        self.db_path = self.data_dir / "vectors.db"
        self.index_path = self.data_dir / "vectors.faiss"
        
        self._lock = threading.RLock()
        self.index = None
        self._vector_cache: Dict[int, np.ndarray] = {}
        
        self._init_database()
        self._load_index()
    
    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
            isolation_level=None
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn
    
    def _init_database(self) -> None:
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                content_hash TEXT UNIQUE,
                source_url TEXT DEFAULT '',
                source_title TEXT DEFAULT '',
                confidence REAL DEFAULT 0.5,
                vector_data BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT DEFAULT '{}'
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_source_url ON vectors(source_url)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_source_title ON vectors(source_title)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON vectors(content_hash)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON vectors(created_at DESC)")
        conn.close()
    
    def _load_index(self) -> None:
        if not FAISS_AVAILABLE:
            self._load_vectors_to_cache()
            return
        
        with self._lock:
            if self.index_path.exists():
                try:
                    self.index = faiss.read_index(
                        str(self.index_path),
                        faiss.IO_FLAG_MMAP
                    )
                    print(f"✅ Loaded FAISS index: {self.index.ntotal} vectors")
                    return
                except Exception as e:
                    print(f"⚠️ Could not load FAISS index: {e}")
            
            self._rebuild_faiss_index()
    
    def _rebuild_faiss_index(self) -> None:
        if not FAISS_AVAILABLE:
            return
        
        print("🔄 Building FAISS index from database...")
        
        base_index = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIDMap(base_index)
        
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT id, vector_data FROM vectors WHERE vector_data IS NOT NULL"
        ).fetchall()
        conn.close()
        
        if not rows:
            print("✨ Created empty FAISS index")
            return
        
        ids = []
        vectors = []
        
        for row_id, vector_blob in rows:
            if vector_blob:
                try:
                    vector = np.frombuffer(vector_blob, dtype=np.float32)
                    if len(vector) == self.dimension:
                        ids.append(row_id)
                        vectors.append(vector)
                except:
                    pass
        
        if vectors:
            vectors_np = np.array(vectors, dtype=np.float32)
            ids_np = np.array(ids, dtype=np.int64)
            faiss.normalize_L2(vectors_np)
            self.index.add_with_ids(vectors_np, ids_np)
            self._save_index()
            print(f"✅ Built FAISS index: {len(vectors)} vectors")
    
    def _load_vectors_to_cache(self) -> None:
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT id, vector_data FROM vectors WHERE vector_data IS NOT NULL"
        ).fetchall()
        conn.close()
        
        for row_id, vector_blob in rows:
            if vector_blob:
                try:
                    vector = np.frombuffer(vector_blob, dtype=np.float32)
                    if len(vector) == self.dimension:
                        self._vector_cache[row_id] = vector
                except:
                    pass
        
        if self._vector_cache:
            print(f"✅ Loaded {len(self._vector_cache)} vectors into memory")
    
    def _save_index(self) -> None:
        if FAISS_AVAILABLE and self.index is not None:
            try:
                faiss.write_index(self.index, str(self.index_path))
            except Exception as e:
                print(f"⚠️ Could not save FAISS index: {e}")
    
    def _content_hash(self, content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()
    
    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        vector = np.array(vector, dtype=np.float32).flatten()
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector
    
    def _extract_subject(self, query: str) -> str:
        """
        Extract the SUBJECT from a query.
        "Tell me about France" -> "France"
        "What is the capital of France" -> "capital of France"
        "Who is Albert Einstein" -> "Albert Einstein"
        """
        if not query:
            return ""
        
        query = query.strip()
        query_lower = query.lower()
        
        # Remove common question prefixes
        prefixes = [
            r'^tell me about\s+',
            r'^what is\s+',
            r'^what are\s+',
            r'^who is\s+',
            r'^who are\s+',
            r'^where is\s+',
            r'^when is\s+',
            r'^when was\s+',
            r'^how is\s+',
            r'^describe\s+',
            r'^explain\s+',
            r'^define\s+',
            r'^what do you know about\s+',
            r'^can you tell me about\s+',
            r'^i want to know about\s+',
            r'^information about\s+',
            r'^info on\s+',
        ]
        
        subject = query
        for prefix in prefixes:
            match = re.match(prefix, query_lower)
            if match:
                # Extract the part after the prefix, preserving original case
                subject = query[match.end():].strip()
                break
        
        # Remove trailing punctuation
        subject = re.sub(r'[?.!]+$', '', subject).strip()
        
        # Remove articles at the start
        subject = re.sub(r'^(the|a|an)\s+', '', subject, flags=re.IGNORECASE).strip()
        
        return subject
    
    def add(self, vector: np.ndarray, content: str,
            source_url: str = '', source_title: str = '',
            confidence: float = 0.5, metadata: Dict = None) -> int:
        if metadata is None:
            metadata = {}
        
        vector = self._normalize(vector)
        vector_blob = vector.tobytes()
        content_hash = self._content_hash(content)
        
        with self._lock:
            conn = self._get_connection()
            
            existing = conn.execute(
                "SELECT id FROM vectors WHERE content_hash = ?",
                (content_hash,)
            ).fetchone()
            
            if existing:
                conn.close()
                return 0
            
            cursor = conn.execute(
                """INSERT INTO vectors 
                   (content, content_hash, source_url, source_title, confidence, vector_data, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (content, content_hash, source_url, source_title, confidence, 
                 vector_blob, json.dumps(metadata))
            )
            vector_id = cursor.lastrowid
            conn.close()
            
            if FAISS_AVAILABLE and self.index is not None:
                vector_2d = vector.reshape(1, -1)
                ids = np.array([vector_id], dtype=np.int64)
                self.index.add_with_ids(vector_2d, ids)
                
                if vector_id % 100 == 0:
                    self._save_index()
            else:
                self._vector_cache[vector_id] = vector
            
            return vector_id
    
    def search(self, query_vector: np.ndarray, top_k: int = 10,
               min_score: float = 0.0, query_text: str = "") -> List[Dict[str, Any]]:
        """
        Search with SUBJECT EXTRACTION and TITLE MATCHING.
        """
        query_vector = self._normalize(query_vector).reshape(1, -1)
        results = []
        seen_ids = set()
        
        # Extract the SUBJECT from the query
        subject = self._extract_subject(query_text) if query_text else ""
        subject_lower = subject.lower().strip()
        
        with self._lock:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            
            # === DIRECT TITLE LOOKUP using extracted SUBJECT ===
            if subject_lower and len(subject_lower) > 1:
                # 1. Exact title match
                exact_row = conn.execute(
                    "SELECT * FROM vectors WHERE LOWER(source_title) = ? LIMIT 1",
                    (subject_lower,)
                ).fetchone()
                
                if exact_row:
                    results.append({
                        'id': exact_row['id'],
                        'content': exact_row['content'],
                        'source_url': exact_row['source_url'],
                        'source_title': exact_row['source_title'],
                        'confidence': exact_row['confidence'],
                        'relevance': 1.0,  # Perfect match!
                        'metadata': json.loads(exact_row['metadata'] or '{}')
                    })
                    seen_ids.add(exact_row['id'])
                
                # 2. Title starts with subject
                if len(results) < top_k:
                    starts_rows = conn.execute(
                        "SELECT * FROM vectors WHERE LOWER(source_title) LIKE ? ORDER BY LENGTH(source_title) LIMIT 5",
                        (subject_lower + '%',)
                    ).fetchall()
                    
                    for row in starts_rows:
                        if row['id'] not in seen_ids:
                            results.append({
                                'id': row['id'],
                                'content': row['content'],
                                'source_url': row['source_url'],
                                'source_title': row['source_title'],
                                'confidence': row['confidence'],
                                'relevance': 0.95,
                                'metadata': json.loads(row['metadata'] or '{}')
                            })
                            seen_ids.add(row['id'])
                
                # 3. Subject appears anywhere in title
                if len(results) < top_k and len(subject_lower) > 2:
                    contains_rows = conn.execute(
                        "SELECT * FROM vectors WHERE LOWER(source_title) LIKE ? ORDER BY LENGTH(source_title) LIMIT 5",
                        ('%' + subject_lower + '%',)
                    ).fetchall()
                    
                    for row in contains_rows:
                        if row['id'] not in seen_ids:
                            results.append({
                                'id': row['id'],
                                'content': row['content'],
                                'source_url': row['source_url'],
                                'source_title': row['source_title'],
                                'confidence': row['confidence'],
                                'relevance': 0.85,
                                'metadata': json.loads(row['metadata'] or '{}')
                            })
                            seen_ids.add(row['id'])
            
            # === FAISS VECTOR SEARCH ===
            if FAISS_AVAILABLE and self.index is not None and self.index.ntotal > 0:
                k = min(top_k * 3, self.index.ntotal)
                scores, ids = self.index.search(query_vector, k)
                
                for score, vec_id in zip(scores[0], ids[0]):
                    if vec_id < 0 or int(vec_id) in seen_ids:
                        continue
                    
                    similarity = float(score)
                    if similarity < min_score:
                        continue
                    
                    row = conn.execute(
                        "SELECT * FROM vectors WHERE id = ?", (int(vec_id),)
                    ).fetchone()
                    
                    if row:
                        title = row['source_title'] or ""
                        title_lower = title.lower()
                        
                        # Title boosting
                        title_boost = 0.0
                        if subject_lower:
                            if title_lower == subject_lower:
                                title_boost = 0.5
                            elif title_lower.startswith(subject_lower):
                                title_boost = 0.4
                            elif subject_lower in title_lower:
                                title_boost = 0.25
                        
                        boosted_score = min(1.0, similarity + title_boost)
                        
                        results.append({
                            'id': row['id'],
                            'content': row['content'],
                            'source_url': row['source_url'],
                            'source_title': row['source_title'],
                            'confidence': row['confidence'],
                            'relevance': boosted_score,
                            'metadata': json.loads(row['metadata'] or '{}')
                        })
                        seen_ids.add(row['id'])
            
            elif self._vector_cache:
                # Brute force fallback
                similarities = []
                for vec_id, vec in self._vector_cache.items():
                    if vec_id in seen_ids:
                        continue
                    sim = float(np.dot(query_vector.flatten(), vec))
                    if sim >= min_score:
                        similarities.append((vec_id, sim))
                
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                for vec_id, sim in similarities[:top_k * 2]:
                    row = conn.execute(
                        "SELECT * FROM vectors WHERE id = ?", (vec_id,)
                    ).fetchone()
                    
                    if row:
                        title = row['source_title'] or ""
                        title_lower = title.lower()
                        
                        title_boost = 0.0
                        if subject_lower:
                            if title_lower == subject_lower:
                                title_boost = 0.5
                            elif subject_lower in title_lower:
                                title_boost = 0.2
                        
                        boosted_score = min(1.0, sim + title_boost)
                        
                        results.append({
                            'id': row['id'],
                            'content': row['content'],
                            'source_url': row['source_url'],
                            'source_title': row['source_title'],
                            'confidence': row['confidence'],
                            'relevance': boosted_score,
                            'metadata': json.loads(row['metadata'] or '{}')
                        })
            
            conn.close()
        
        # Sort by relevance
        results.sort(key=lambda x: x['relevance'], reverse=True)
        return results[:top_k]
    
    def get_by_id(self, vector_id: int) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM vectors WHERE id = ?", (vector_id,)).fetchone()
        conn.close()
        
        if row:
            return {
                'id': row['id'],
                'content': row['content'],
                'source_url': row['source_url'],
                'source_title': row['source_title'],
                'confidence': row['confidence'],
                'metadata': json.loads(row['metadata'] or '{}')
            }
        return None
    
    def exists(self, source_url: str) -> bool:
        conn = self._get_connection()
        row = conn.execute(
            "SELECT id FROM vectors WHERE source_url = ? LIMIT 1", (source_url,)
        ).fetchone()
        conn.close()
        return row is not None
    
    def count(self) -> int:
        if FAISS_AVAILABLE and self.index is not None:
            return self.index.ntotal
        return len(self._vector_cache)
    
    def save(self) -> None:
        self._save_index()
        print(f"💾 Vector store saved: {self.count()} vectors")
    
    def get_stats(self) -> Dict[str, Any]:
        try:
            conn = self._get_connection()
            total = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
            sources = conn.execute(
                "SELECT COUNT(DISTINCT source_url) FROM vectors WHERE source_url != ''"
            ).fetchone()[0]
            conn.close()
        except:
            total = self.count()
            sources = 0
        
        return {
            'total_vectors': self.count(),
            'total_entries': total,
            'unique_sources': sources,
            'dimension': self.dimension,
            'index_type': 'FAISS' if FAISS_AVAILABLE else 'BruteForce',
            'faiss_available': FAISS_AVAILABLE
        }
    
    def get_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, source_title, source_url, created_at FROM vectors ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            conn.close()
            
            return [{
                'id': row['id'],
                'source_title': row['source_title'],
                'source_url': row['source_url'],
                'created_at': row['created_at']
            } for row in rows]
        except:
            return []
    
    def get_all_knowledge(self, limit: int = 100) -> List[Dict[str, Any]]:
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, source_title, source_url, confidence, created_at FROM vectors ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            conn.close()
            
            return [{
                'id': row['id'],
                'title': row['source_title'],
                'url': row['source_url'],
                'confidence': row['confidence'],
                'created_at': row['created_at']
            } for row in rows]
        except:
            return []
    
    def get_all_with_content(self) -> List[Dict[str, Any]]:
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT id, content FROM vectors").fetchall()
            conn.close()
            return [{'id': row['id'], 'content': row['content']} for row in rows]
        except:
            return []
    
    def update_vector(self, entry_id: int, new_vector: np.ndarray) -> None:
        new_vector = self._normalize(new_vector)
        vector_blob = new_vector.tobytes()
        
        with self._lock:
            try:
                conn = self._get_connection()
                conn.execute(
                    "UPDATE vectors SET vector_data = ? WHERE id = ?",
                    (vector_blob, entry_id)
                )
                conn.close()
                
                if not FAISS_AVAILABLE:
                    self._vector_cache[entry_id] = new_vector
            except Exception as e:
                print(f"⚠️ Could not update vector: {e}")
    
    def get_related(self, entry_id: int, top_k: int = 5) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            row = conn.execute(
                "SELECT vector_data FROM vectors WHERE id = ?", (entry_id,)
            ).fetchone()
            conn.close()
            
            if not row or not row[0]:
                return []
            
            query_vector = np.frombuffer(row[0], dtype=np.float32)
            results = self.search(query_vector, top_k=top_k + 1, min_score=0.05)
            
            related = []
            for r in results:
                if r['id'] != entry_id:
                    related.append({
                        'id': r['id'],
                        'title': r['source_title'],
                        'confidence': r['confidence'],
                        'similarity': round(r['relevance'] * 100, 1)
                    })
            
            return related[:top_k]
    
    def rebuild_index(self) -> None:
        if FAISS_AVAILABLE:
            self._rebuild_faiss_index()
        else:
            self._load_vectors_to_cache()