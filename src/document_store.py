"""
Document Store for GroundZero AI (RAG Preparation)
===================================================
Stores Wikipedia articles with semantic embeddings for future LLM integration.

Usage:
    from src.document_store import DocumentStore
    
    store = DocumentStore("data/documents")
    store.AddArticle("Albert Einstein", "Albert Einstein was a physicist...")
    
    # Semantic search
    results = store.Search("theory of relativity")

Requirements:
    pip install chromadb
"""

from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
import hashlib
from datetime import datetime

# Try to import ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("⚠️ ChromaDB not installed. Run: pip install chromadb")


def _simple_hash_embedding(text: str, dim: int = 384) -> List[float]:
    """
    Create a simple deterministic embedding from text.
    This is a fallback when network-based embeddings aren't available.
    Not as good as real embeddings but works offline.
    """
    import math
    
    # Hash the text
    text_lower = text.lower()
    words = text_lower.split()
    
    # Create embedding by hashing word combinations
    embedding = [0.0] * dim
    
    for i, word in enumerate(words):
        # Hash each word
        word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16)
        
        # Distribute across embedding dimensions
        for j in range(dim):
            idx = (word_hash + j * 31) % dim
            val = ((word_hash >> (j % 32)) & 0xFF) / 255.0 - 0.5
            embedding[idx] += val / (1 + math.log(1 + i))
    
    # Normalize
    norm = math.sqrt(sum(x * x for x in embedding))
    if norm > 0:
        embedding = [x / norm for x in embedding]
    
    return embedding


# Custom embedding function that works without network
class SimpleEmbeddingFunction:
    """Custom embedding function that works without network access"""
    
    def __init__(self, dim: int = 384):
        self.dim = dim
        # ChromaDB requires these attributes
        self._name = "simple_hash_embedding"
    
    @property
    def name(self) -> str:
        return self._name
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for input texts"""
        return [_simple_hash_embedding(text, self.dim) for text in input]
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query"""
        return _simple_hash_embedding(query, self.dim)


class DocumentStore:
    """
    Document storage for RAG integration with future LLM.
    Uses ChromaDB for semantic search over learned content.
    """
    
    def __init__(self, DataDir: str):
        """
        Initialize the document store.
        
        Args:
            DataDir: Directory to store ChromaDB data
        """
        self.DataDir = Path(DataDir)
        self.DataDir.mkdir(parents=True, exist_ok=True)
        
        self.Client = None
        self.Articles = None
        self.Facts = None
        self.Conversations = None
        self.EmbedDim = 384
        
        if CHROMA_AVAILABLE:
            self._Initialize()
        else:
            print("⚠️ DocumentStore disabled - ChromaDB not available")
    
    def _Embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        return [_simple_hash_embedding(t, self.EmbedDim) for t in texts]
    
    def _Initialize(self):
        """Initialize ChromaDB client and collections"""
        try:
            self.Client = chromadb.PersistentClient(
                path=str(self.DataDir),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Collection for Wikipedia articles - NO embedding function, we'll provide embeddings
            self.Articles = self.Client.get_or_create_collection(
                name="wikipedia_articles",
                metadata={"description": "Learned Wikipedia content for RAG", "hnsw:space": "cosine"}
            )
            
            # Collection for extracted facts
            self.Facts = self.Client.get_or_create_collection(
                name="knowledge_facts",
                metadata={"description": "Extracted facts from knowledge graph", "hnsw:space": "cosine"}
            )
            
            # Collection for conversation history
            self.Conversations = self.Client.get_or_create_collection(
                name="conversations",
                metadata={"description": "Chat history for context", "hnsw:space": "cosine"}
            )
            
            print(f"✅ DocumentStore: {self.Articles.count()} articles, "
                  f"{self.Facts.count()} facts")
                  
        except Exception as e:
            print(f"⚠️ Error initializing ChromaDB: {e}")
            self.Client = None
    
    def AddArticle(self, Title: str, Content: str, Source: str = "wikipedia",
                   Metadata: Dict = None) -> bool:
        """
        Store a Wikipedia article with auto-embedding.
        
        Args:
            Title: Article title
            Content: Article text
            Source: Source identifier
            Metadata: Additional metadata
            
        Returns:
            True if successful
        """
        if not CHROMA_AVAILABLE or self.Articles is None:
            return False
        
        try:
            # Create unique ID
            doc_id = f"{Source}_{abs(hash(Title)) % 10**9}"
            
            # Prepare metadata
            meta = {
                "title": str(Title)[:500],  # Limit length
                "source": str(Source),
                "timestamp": datetime.now().isoformat(),
                "length": len(Content)
            }
            if Metadata:
                # Ensure all metadata values are strings
                for k, v in Metadata.items():
                    meta[str(k)] = str(v) if v is not None else ""
            
            # Generate embedding
            content_text = Content[:50000]
            embeddings = self._Embed([content_text])
            
            # Try to delete existing first (for upsert behavior)
            try:
                self.Articles.delete(ids=[doc_id])
            except:
                pass
            
            # Add new with explicit embeddings
            self.Articles.add(
                documents=[content_text],
                embeddings=embeddings,
                metadatas=[meta],
                ids=[doc_id]
            )
            return True
            
        except Exception as e:
            print(f"Error adding article: {e}")
            return False
    
    def AddFact(self, Subject: str, Predicate: str, Object: str,
                Source: str = "knowledge_graph") -> bool:
        """
        Store a knowledge graph triple as a searchable fact.
        
        Args:
            Subject, Predicate, Object: Triple components
            Source: Source of the fact
        """
        if not CHROMA_AVAILABLE or self.Facts is None:
            return False
        
        try:
            # Convert triple to natural language
            fact_text = f"{Subject} {Predicate.replace('_', ' ')} {Object}"
            doc_id = f"fact_{abs(hash(fact_text)) % 10**9}"
            
            # Generate embedding
            embeddings = self._Embed([fact_text[:5000]])
            
            # Try delete first for upsert behavior
            try:
                self.Facts.delete(ids=[doc_id])
            except:
                pass
            
            self.Facts.add(
                documents=[fact_text[:5000]],  # Limit length
                embeddings=embeddings,
                metadatas=[{
                    "subject": str(Subject)[:200],
                    "predicate": str(Predicate)[:200],
                    "object": str(Object)[:200],
                    "source": str(Source)
                }],
                ids=[doc_id]
            )
            return True
            
        except Exception as e:
            print(f"Error adding fact: {e}")
            return False
    
    def AddConversation(self, Role: str, Content: str, SessionID: str) -> bool:
        """
        Store a conversation turn for context memory.
        
        Args:
            Role: "user" or "assistant"
            Content: Message content
            SessionID: Conversation session identifier
        """
        if not CHROMA_AVAILABLE or self.Conversations is None:
            return False
        
        try:
            doc_id = f"conv_{SessionID}_{datetime.now().timestamp()}"
            
            # Generate embedding
            embeddings = self._Embed([Content])
            
            self.Conversations.add(
                documents=[Content],
                embeddings=embeddings,
                metadatas=[{
                    "role": Role,
                    "session": SessionID,
                    "timestamp": datetime.now().isoformat()
                }],
                ids=[doc_id]
            )
            return True
            
        except Exception as e:
            print(f"Error adding conversation: {e}")
            return False
    
    def Search(self, Query: str, Collection: str = "articles",
               TopK: int = 5) -> List[Tuple[str, Dict, float]]:
        """
        Semantic search over stored documents.
        
        Args:
            Query: Search query
            Collection: "articles", "facts", or "conversations"
            TopK: Number of results
            
        Returns:
            List of (document, metadata, distance) tuples
        """
        if not CHROMA_AVAILABLE:
            return []
        
        try:
            coll = {
                "articles": self.Articles,
                "facts": self.Facts,
                "conversations": self.Conversations
            }.get(Collection, self.Articles)
            
            if coll is None or coll.count() == 0:
                return []
            
            # Generate query embedding
            query_embeddings = self._Embed([Query])
            
            results = coll.query(
                query_embeddings=query_embeddings,
                n_results=min(TopK, coll.count())
            )
            
            if not results['documents'] or not results['documents'][0]:
                return []
            
            return list(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0] if results['distances'] else [0] * len(results['documents'][0])
            ))
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def SearchArticles(self, Query: str, TopK: int = 5) -> List[Tuple[str, Dict, float]]:
        """Convenience method for article search"""
        return self.Search(Query, "articles", TopK)
    
    def SearchFacts(self, Query: str, TopK: int = 10) -> List[Tuple[str, Dict, float]]:
        """Convenience method for fact search"""
        return self.Search(Query, "facts", TopK)
    
    def GetRAGContext(self, Query: str, MaxArticles: int = 3,
                      MaxFacts: int = 5) -> str:
        """
        Get combined context for RAG (Retrieval-Augmented Generation).
        
        This method retrieves relevant articles and facts to provide
        context for LLM generation.
        
        Args:
            Query: User query
            MaxArticles: Max article snippets to include
            MaxFacts: Max facts to include
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Get relevant articles
        articles = self.SearchArticles(Query, MaxArticles)
        if articles:
            context_parts.append("=== Relevant Articles ===")
            for doc, meta, dist in articles:
                title = meta.get('title', 'Unknown')
                # Truncate long articles
                snippet = doc[:500] + "..." if len(doc) > 500 else doc
                context_parts.append(f"\n[{title}]:\n{snippet}")
        
        # Get relevant facts
        facts = self.SearchFacts(Query, MaxFacts)
        if facts:
            context_parts.append("\n\n=== Related Facts ===")
            for doc, meta, dist in facts:
                context_parts.append(f"• {doc}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def SyncFromKnowledgeGraph(self, KnowledgeGraph, BatchSize: int = 100):
        """
        Sync facts from KnowledgeGraph to enable semantic fact search.
        
        Args:
            KnowledgeGraph: Your KnowledgeGraph instance
            BatchSize: Number of facts to process at once
        """
        if not CHROMA_AVAILABLE or self.Facts is None:
            return
        
        if not hasattr(KnowledgeGraph, 'AllTriples'):
            print("⚠️ KnowledgeGraph has no AllTriples")
            return
        
        triples = list(KnowledgeGraph.AllTriples)
        total = len(triples)
        added = 0
        
        print(f"📥 Syncing {total} facts to DocumentStore...")
        
        for i in range(0, total, BatchSize):
            batch = triples[i:i + BatchSize]
            
            documents = []
            metadatas = []
            ids = []
            
            for subj, pred, obj in batch:
                fact_text = f"{subj} {pred.replace('_', ' ')} {obj}"
                doc_id = f"fact_{abs(hash(fact_text)) % 10**9}"
                
                documents.append(fact_text[:5000])
                metadatas.append({
                    "subject": str(subj)[:200],
                    "predicate": str(pred)[:200],
                    "object": str(obj)[:200],
                    "source": "knowledge_graph"
                })
                ids.append(doc_id)
            
            try:
                # Generate embeddings for batch
                embeddings = self._Embed(documents)
                
                # Delete existing first for upsert behavior
                try:
                    self.Facts.delete(ids=ids)
                except:
                    pass
                
                self.Facts.add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                added += len(batch)
            except Exception as e:
                print(f"Batch error: {e}")
        
        print(f"✅ Synced {added} facts to DocumentStore")
    
    def GetStats(self) -> Dict:
        """Get document store statistics"""
        return {
            "TotalArticles": self.Articles.count() if self.Articles else 0,
            "TotalFacts": self.Facts.count() if self.Facts else 0,
            "TotalConversations": self.Conversations.count() if self.Conversations else 0,
            "ChromaDBAvailable": CHROMA_AVAILABLE
        }
    
    def Clear(self, Collection: str = None):
        """
        Clear documents from collections.
        
        Args:
            Collection: "articles", "facts", "conversations", or None for all
        """
        if not CHROMA_AVAILABLE or self.Client is None:
            return
        
        try:
            if Collection is None or Collection == "articles":
                if self.Articles:
                    self.Client.delete_collection("wikipedia_articles")
                    self.Articles = self.Client.create_collection("wikipedia_articles")
            
            if Collection is None or Collection == "facts":
                if self.Facts:
                    self.Client.delete_collection("knowledge_facts")
                    self.Facts = self.Client.create_collection("knowledge_facts")
            
            if Collection is None or Collection == "conversations":
                if self.Conversations:
                    self.Client.delete_collection("conversations")
                    self.Conversations = self.Client.create_collection("conversations")
            
            print(f"✅ Cleared {Collection or 'all collections'}")
            
        except Exception as e:
            print(f"Error clearing: {e}")


# Convenience function
def CheckChromaDB() -> bool:
    """Check if ChromaDB is installed and working"""
    if not CHROMA_AVAILABLE:
        print("=" * 50)
        print("ChromaDB is not installed!")
        print("To install:")
        print("  pip install chromadb")
        print("=" * 50)
        return False
    return True


if __name__ == "__main__":
    # Test the document store
    CheckChromaDB()
    
    if CHROMA_AVAILABLE:
        # Create test store
        store = DocumentStore("test_documents")
        
        # Add test article
        store.AddArticle(
            "Albert Einstein",
            "Albert Einstein was a German-born theoretical physicist who is widely "
            "held to be one of the greatest scientists of all time. He developed the "
            "theory of relativity, one of the two pillars of modern physics.",
            "wikipedia"
        )
        
        # Add test facts
        store.AddFact("Einstein", "born_in", "Germany")
        store.AddFact("Einstein", "developed", "theory of relativity")
        store.AddFact("Einstein", "profession", "physicist")
        
        # Test search
        print("\n=== Article Search ===")
        results = store.SearchArticles("theory of relativity")
        for doc, meta, dist in results:
            print(f"[{meta.get('title')}] (dist: {dist:.3f})")
            print(f"  {doc[:100]}...")
        
        print("\n=== Fact Search ===")
        results = store.SearchFacts("Einstein profession")
        for doc, meta, dist in results:
            print(f"• {doc} (dist: {dist:.3f})")
        
        print("\n=== RAG Context ===")
        context = store.GetRAGContext("What did Einstein discover?")
        print(context[:500])
        
        print(f"\n=== Stats ===")
        print(store.GetStats())
