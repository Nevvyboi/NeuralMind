"""
GroundZero AI v4.0 - Main Entry Point
=====================================
"""

import os
import sys
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import core modules from src/
CORE_AVAILABLE = False
try:
    from src.knowledge_graph import KnowledgeGraph
    from src.causal_graph import CausalGraph
    from src.chat_engine import SmartChatEngine
    from src.reasoning import ChainOfThought
    from src.metacognition import MetacognitionEngine
    from src.constitutional import ConstitutionalAI
    from src.question_detector import QuestionDetector
    from src.auto_learner import WikipediaLearner
    from src.world_model import WorldModel
    from src.progress_tracker import ProgressTracker
    CORE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Core modules not fully available: {e}")

# Import neural pipeline modules from src/
NEURAL_PIPELINE_AVAILABLE = False
NeuralPipeline = None
ChatEngineEnhanced = None

try:
    from src.neural_pipeline import NeuralPipeline
    from src.chat_engine_enhanced import ChatEngineEnhanced
    NEURAL_PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Neural pipeline not available: {e}")

# Import vector store (FAISS) - optional
VECTOR_STORE_AVAILABLE = False
VectorStore = None
try:
    from src.vector_store import VectorStore, FAISS_AVAILABLE
    VECTOR_STORE_AVAILABLE = FAISS_AVAILABLE
    if not FAISS_AVAILABLE:
        print("💡 FAISS not installed. Run: pip install faiss-cpu")
except ImportError:
    print("💡 VectorStore not available. Copy vector_store.py to src/")

# Import document store (ChromaDB) - optional
DOCUMENT_STORE_AVAILABLE = False
DocumentStore = None
try:
    from src.document_store import DocumentStore, CHROMA_AVAILABLE
    DOCUMENT_STORE_AVAILABLE = CHROMA_AVAILABLE
    if not CHROMA_AVAILABLE:
        print("💡 ChromaDB not installed. Run: pip install chromadb")
except ImportError:
    print("💡 DocumentStore not available. Copy document_store.py to src/")


# =============================================================================
# GLOBAL STATE
# =============================================================================

@dataclass
class GlobalState:
    """Global application state"""
    # Core engines
    KnowledgeGraph: Optional[Any] = None
    CausalGraph: Optional[Any] = None
    ChatEngine: Optional[Any] = None
    Reasoning: Optional[Any] = None
    Metacognition: Optional[Any] = None
    Constitutional: Optional[Any] = None
    QuestionDetector: Optional[Any] = None
    WorldModel: Optional[Any] = None
    ProgressTracker: Optional[Any] = None
    AutoLearner: Optional[Any] = None
    
    # Neural components
    NeuralEngine: Optional[Any] = None
    NeuralPipeline: Optional[Any] = None
    EnhancedChat: Optional[Any] = None
    
    # NEW: Vector and Document stores
    VectorStore: Optional[Any] = None      # FAISS for TransE embeddings
    DocumentStore: Optional[Any] = None    # ChromaDB for RAG
    
    # State
    IsInitialized: bool = False
    IsLearning: bool = False
    CurrentArticle: str = ""
    ArticlesLearned: int = 0
    FactsThisSession: int = 0
    
    # Loss history for chart
    LossHistory: List[float] = field(default_factory=list)
    
    @classmethod
    def Initialize(cls) -> 'GlobalState':
        """Initialize all engines"""
        state = cls()
        
        print("\n" + "=" * 60)
        print("🧠 GroundZero AI v4.0 - Initializing...")
        print("=" * 60)
        
        # Initialize core components
        if CORE_AVAILABLE:
            print("\n📚 Loading core modules...")
            
            # Create data directory
            data_dir = Path(__file__).parent / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            
            state.KnowledgeGraph = KnowledgeGraph(str(data_dir / "knowledge.db"))
            print(f"  ✅ KnowledgeGraph: {len(state.KnowledgeGraph.Triples)} triples")
            
            state.CausalGraph = CausalGraph(str(data_dir))
            print(f"  ✅ CausalGraph")
            
            state.ChatEngine = SmartChatEngine(state.KnowledgeGraph, state.CausalGraph)
            print(f"  ✅ ChatEngine")
            
            state.Reasoning = ChainOfThought()
            print(f"  ✅ Reasoning")
            
            state.Metacognition = MetacognitionEngine()
            print(f"  ✅ Metacognition")
            
            state.Constitutional = ConstitutionalAI()
            print(f"  ✅ Constitutional")
            
            state.QuestionDetector = QuestionDetector()
            print(f"  ✅ QuestionDetector")
            
            state.WorldModel = WorldModel()
            print(f"  ✅ WorldModel")
            
            state.ProgressTracker = ProgressTracker()
            print(f"  ✅ ProgressTracker")
            
            # Check for neural engine
            if hasattr(state.ChatEngine, 'NeuralEngine'):
                state.NeuralEngine = state.ChatEngine.NeuralEngine
                print(f"  ✅ NeuralEngine (TransE)")
        
        # Initialize neural pipeline
        if NEURAL_PIPELINE_AVAILABLE and state.NeuralEngine:
            print("\n🔮 Loading neural pipeline...")
            state._InitializeNeuralPipeline()
        else:
            print("\n⚠️ Neural pipeline not available")
        
        # Initialize vector store (FAISS) - for semantic search over embeddings
        if VECTOR_STORE_AVAILABLE and VectorStore is not None:
            print("\n🔍 Loading vector store (FAISS)...")
            try:
                data_dir = Path(__file__).parent / "data"
                state.VectorStore = VectorStore(str(data_dir), Dimension=100)
                
                # Sync existing embeddings if neural engine has them
                if state.NeuralEngine and hasattr(state.NeuralEngine, 'EntityEmbeddings'):
                    if state.NeuralEngine.EntityEmbeddings:
                        state.VectorStore.SyncFromNeuralEngine(state.NeuralEngine)
                
                stats = state.VectorStore.GetStats()
                print(f"  ✅ VectorStore: {stats.get('TotalEntities', 0)} entities indexed")
            except Exception as e:
                print(f"  ⚠️ VectorStore error: {e}")
                state.VectorStore = None
        
        # Initialize document store (ChromaDB) - for RAG
        if DOCUMENT_STORE_AVAILABLE and DocumentStore is not None:
            print("\n📄 Loading document store (ChromaDB)...")
            try:
                data_dir = Path(__file__).parent / "data" / "documents"
                state.DocumentStore = DocumentStore(str(data_dir))
                stats = state.DocumentStore.GetStats()
                print(f"  ✅ DocumentStore: {stats.get('TotalArticles', 0)} articles, "
                      f"{stats.get('TotalFacts', 0)} facts")
            except Exception as e:
                print(f"  ⚠️ DocumentStore error: {e}")
                state.DocumentStore = None
        
        state.IsInitialized = True
        print("\n✅ Initialization complete!")
        print("=" * 60 + "\n")
        
        return state
    
    def _InitializeNeuralPipeline(self):
        """Initialize the neural pipeline with current embeddings"""
        if not self.NeuralEngine:
            return
        
        try:
            # Get embeddings from neural engine
            entity_embs = {}
            rel_embs = {}
            
            if hasattr(self.NeuralEngine, 'EntityEmbeddings'):
                entity_embs = self.NeuralEngine.EntityEmbeddings
            elif hasattr(self.NeuralEngine, 'Embeddings'):
                entity_embs = self.NeuralEngine.Embeddings
            
            if hasattr(self.NeuralEngine, 'RelationEmbeddings'):
                rel_embs = self.NeuralEngine.RelationEmbeddings
            
            # Build entity triples mapping
            entity_triples = {}
            if self.KnowledgeGraph and hasattr(self.KnowledgeGraph, 'Triples'):
                for subj, rel, obj in self.KnowledgeGraph.Triples:
                    subj_lower = subj.lower()
                    obj_lower = obj.lower()
                    
                    if subj_lower not in entity_triples:
                        entity_triples[subj_lower] = []
                    entity_triples[subj_lower].append((subj, rel, obj))
                    
                    if obj_lower not in entity_triples:
                        entity_triples[obj_lower] = []
                    entity_triples[obj_lower].append((subj, rel, obj))
            
            # Build graph adjacency
            graph = {}
            if self.KnowledgeGraph and hasattr(self.KnowledgeGraph, 'Triples'):
                for subj, rel, obj in self.KnowledgeGraph.Triples:
                    subj_lower = subj.lower()
                    obj_lower = obj.lower()
                    
                    if subj_lower not in graph:
                        graph[subj_lower] = []
                    graph[subj_lower].append((obj_lower, rel))
                    
                    if obj_lower not in graph:
                        graph[obj_lower] = []
                    graph[obj_lower].append((subj_lower, rel))
            
            # Get embedding dimension
            embed_dim = 100
            if entity_embs:
                first_emb = next(iter(entity_embs.values()))
                embed_dim = len(first_emb)
            
            # Create neural pipeline
            self.NeuralPipeline = NeuralPipeline(EmbedDim=embed_dim)
            self.NeuralPipeline.Initialize(
                entity_embs,
                rel_embs,
                entity_triples,
                graph
            )
            
            # Create enhanced chat engine
            self.EnhancedChat = ChatEngineEnhanced(
                SymbolicEngine=self.ChatEngine,
                EmbedDim=embed_dim
            )
            self.EnhancedChat.InitializeNeuralPipeline(
                entity_embs,
                rel_embs,
                entity_triples,
                graph
            )
            
            print(f"  ✅ NeuralPipeline: {len(entity_embs)} entities, {len(rel_embs)} relations")
            print(f"  ✅ EnhancedChat: Ready")
            
        except Exception as e:
            print(f"  ❌ Neural pipeline error: {e}")
    
    def RefreshNeuralPipeline(self):
        """Refresh neural pipeline after training"""
        if NEURAL_PIPELINE_AVAILABLE and self.NeuralEngine:
            self._InitializeNeuralPipeline()
        
        # Also refresh vector store with new embeddings
        if self.VectorStore and self.NeuralEngine:
            try:
                self.VectorStore.SyncFromNeuralEngine(self.NeuralEngine)
            except Exception as e:
                print(f"⚠️ VectorStore sync error: {e}")
    
    def SyncKnowledgeToNeural(self):
        """Sync knowledge graph facts to neural network"""
        if not self.NeuralEngine or not self.KnowledgeGraph:
            return
        
        if hasattr(self.NeuralEngine, 'AddFacts'):
            triples = list(self.KnowledgeGraph.Triples)
            self.NeuralEngine.AddFacts(triples)
    
    def SyncToDocumentStore(self, Title: str = None, Content: str = None):
        """
        Sync content to document store for RAG.
        If Title and Content provided, adds that article.
        Otherwise syncs all facts from knowledge graph.
        """
        if not self.DocumentStore:
            return
        
        try:
            if Title and Content:
                # Add specific article
                self.DocumentStore.AddArticle(Title, Content, "wikipedia")
            else:
                # Sync all facts from knowledge graph
                if self.KnowledgeGraph and hasattr(self.KnowledgeGraph, 'AllTriples'):
                    self.DocumentStore.SyncFromKnowledgeGraph(self.KnowledgeGraph)
        except Exception as e:
            print(f"⚠️ DocumentStore sync error: {e}")
    
    def FindSimilarEntities(self, Entity: str, TopK: int = 10) -> List[Tuple[str, float]]:
        """Find entities similar to given entity using vector search"""
        if not self.VectorStore:
            return []
        try:
            return self.VectorStore.FindSimilar(Entity, TopK)
        except Exception as e:
            print(f"⚠️ Similarity search error: {e}")
            return []
    
    def GetRAGContext(self, Query: str) -> str:
        """Get relevant context for RAG from document store"""
        if not self.DocumentStore:
            return ""
        try:
            return self.DocumentStore.GetRAGContext(Query)
        except Exception as e:
            print(f"⚠️ RAG context error: {e}")
            return ""


# Global instance
STATE: Optional[GlobalState] = None


# =============================================================================
# HTTP HANDLER
# =============================================================================

class GroundZeroHandler(SimpleHTTPRequestHandler):
    """HTTP handler for dashboard and API"""
    
    def __init__(self, *args, **kwargs):
        # Serve from dashboard folder
        dashboard_dir = Path(__file__).parent / 'dashboard'
        if dashboard_dir.exists():
            super().__init__(*args, directory=str(dashboard_dir), **kwargs)
        else:
            super().__init__(*args, directory=str(Path(__file__).parent), **kwargs)
    
    def log_message(self, format, *args):
        """Log only important API requests (not stats polling)"""
        if '/api/' in str(args):
            path = str(args[0])
            # Skip noisy stats polling
            if '/api/stats' not in path:
                print(f"[API] {args[0]}")
    
    def do_GET(self):
        """Handle GET requests"""
        path = urllib.parse.urlparse(self.path).path
        
        if path.startswith('/api/'):
            self._HandleAPI(path)
        else:
            super().do_GET()
    
    def do_POST(self):
        """Handle POST requests"""
        path = urllib.parse.urlparse(self.path).path
        
        if path.startswith('/api/'):
            self._HandleAPI(path)
        else:
            self.send_error(404)
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def _HandleAPI(self, path: str):
        """Handle API requests"""
        global STATE
        
        if STATE is None or not STATE.IsInitialized:
            self._SendJSON({"error": "System not initialized"}, 503)
            return
        
        try:
            if self.command == 'GET':
                if path == '/api/stats':
                    self._SendStats()
                elif path == '/api/knowledge':
                    self._SendKnowledge()
                elif path == '/api/neural':
                    self._SendNeuralStats()
                elif path == '/api/neural/hypotheses':
                    self._SendHypotheses()
                elif path == '/api/pipeline/stats':
                    self._SendPipelineStats()
                elif path == '/api/storage/stats':
                    self._SendStorageStats()
                else:
                    self.send_error(404)
            
            elif self.command == 'POST':
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length).decode('utf-8')
                data = json.loads(body) if body else {}
                
                if path == '/api/chat':
                    self._HandleChat(data)
                elif path == '/api/learn':
                    self._HandleLearn(data)
                elif path == '/api/train' or path == '/api/neural/train':
                    self._HandleTrain(data)
                elif path == '/api/stop':
                    self._HandleStop()
                elif path == '/api/similar':
                    self._HandleSimilarSearch(data)
                elif path == '/api/rag':
                    self._HandleRAGContext(data)
                elif path == '/api/migrate':
                    self._HandleMigration(data)
                else:
                    self.send_error(404)
        
        except ConnectionAbortedError:
            pass  # Client closed connection - ignore
        except ConnectionResetError:
            pass  # Client reset connection - ignore
        except BrokenPipeError:
            pass  # Pipe broken - ignore
        except Exception as e:
            try:
                self._SendJSON({"error": str(e)}, 500)
            except:
                pass  # Can't send error response, connection already closed
    
    def _SendJSON(self, data: dict, status: int = 200):
        """Send JSON response"""
        try:
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode('utf-8'))
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass  # Client closed connection - ignore
        except Exception:
            pass  # Silently ignore write errors
    
    def _SendStats(self):
        """Send system stats in DASHBOARD-COMPATIBLE FORMAT"""
        global STATE
        
        try:
            # Calculate counts from KnowledgeGraph
            total_facts = 0
            total_entities = 0
            total_relations = 0
            
            if STATE.KnowledgeGraph:
                # Use AllTriples which is the actual set
                if hasattr(STATE.KnowledgeGraph, 'AllTriples'):
                    triples = STATE.KnowledgeGraph.AllTriples
                    total_facts = len(triples)
                    entities = set()
                    relations = set()
                    for t in triples:
                        entities.add(t[0])
                        entities.add(t[2])
                        relations.add(t[1])
                    total_entities = len(entities)
                    total_relations = len(relations)
                elif hasattr(STATE.KnowledgeGraph, 'Size'):
                    total_facts = STATE.KnowledgeGraph.Size()
            
            # Get neural stats
            neural_epochs = 0
            neural_loss = None
            neural_predictions = 0
            neural_hypotheses = 0
            neural_embed_dim = 100
            neural_is_trained = False
            
            if STATE.NeuralEngine:
                # Access Stats object directly
                if hasattr(STATE.NeuralEngine, 'Stats'):
                    ns = STATE.NeuralEngine.Stats
                    neural_epochs = ns.TrainingEpochs if hasattr(ns, 'TrainingEpochs') else 0
                    neural_loss = ns.LastLoss if hasattr(ns, 'LastLoss') else None
                    neural_embed_dim = ns.EmbeddingDim if hasattr(ns, 'EmbeddingDim') else 100
                    neural_is_trained = ns.IsTrained if hasattr(ns, 'IsTrained') else False
                    neural_predictions = ns.Predictions if hasattr(ns, 'Predictions') else 0
                    neural_hypotheses = ns.Hypotheses if hasattr(ns, 'Hypotheses') else 0
                
                # Override with direct attributes if available
                if hasattr(STATE.NeuralEngine, 'Dim'):
                    neural_embed_dim = STATE.NeuralEngine.Dim
                if hasattr(STATE.NeuralEngine, 'EntityEmbeddings'):
                    total_entities = max(total_entities, len(STATE.NeuralEngine.EntityEmbeddings))
                if hasattr(STATE.NeuralEngine, 'RelationEmbeddings'):
                    total_relations = max(total_relations, len(STATE.NeuralEngine.RelationEmbeddings))
            
            # Get causal count
            causal_count = 0
            if STATE.CausalGraph:
                if hasattr(STATE.CausalGraph, 'Stats') and isinstance(STATE.CausalGraph.Stats, dict):
                    causal_count = STATE.CausalGraph.Stats.get("TotalRelations", 0)
                elif hasattr(STATE.CausalGraph, 'Size'):
                    causal_count = STATE.CausalGraph.Size()
            
            # Build response in DASHBOARD FORMAT
            stats = {
                "Knowledge": {
                    "TotalFacts": total_facts
                },
                "Causal": {
                    "TotalRelations": causal_count
                },
                "Chat": {
                    "AverageConfidence": 0.7
                },
                "Neural": {
                    "TotalEntities": total_entities,
                    "TotalRelations": total_relations,
                    "TotalTriples": total_facts,
                    "EmbeddingDim": neural_embed_dim,
                    "TrainingEpochs": neural_epochs,
                    "LastLoss": neural_loss,
                    "IsTrained": neural_is_trained or neural_epochs > 0,
                    "Predictions": neural_predictions,
                    "Hypotheses": neural_hypotheses
                },
                "LossHistory": list(STATE.LossHistory) if STATE.LossHistory else [],
                "isLearning": STATE.IsLearning,
                "currentArticle": STATE.CurrentArticle,
                "articlesLearned": STATE.ArticlesLearned,
                "factsThisSession": STATE.FactsThisSession,
                "pipelineReady": STATE.NeuralPipeline is not None,
                "Storage": {
                    "VectorStoreReady": STATE.VectorStore is not None,
                    "VectorEntities": STATE.VectorStore.GetStats().get("TotalEntities", 0) if STATE.VectorStore else 0,
                    "DocumentStoreReady": STATE.DocumentStore is not None,
                    "StoredArticles": STATE.DocumentStore.GetStats().get("TotalArticles", 0) if STATE.DocumentStore else 0,
                    "StoredFacts": STATE.DocumentStore.GetStats().get("TotalFacts", 0) if STATE.DocumentStore else 0
                }
            }
            
            self._SendJSON(stats)
            
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass  # Client closed connection - ignore silently
        except Exception as e:
            # Only log non-connection errors
            if "10053" not in str(e) and "10054" not in str(e):
                print(f"Error in _SendStats: {e}")
            try:
                self._SendJSON({"error": str(e)}, 500)
            except:
                pass  # Can't send, connection closed
    
    def _SendKnowledge(self):
        """Send knowledge graph data"""
        global STATE
        
        triples = []
        if STATE.KnowledgeGraph:
            for subj, rel, obj in list(STATE.KnowledgeGraph.Triples)[:100]:
                triples.append({
                    "subject": subj,
                    "relation": rel,
                    "object": obj
                })
        
        self._SendJSON({"triples": triples})
    
    def _SendNeuralStats(self):
        """Send neural network stats"""
        global STATE
        
        if STATE.NeuralEngine and hasattr(STATE.NeuralEngine, 'GetStats'):
            stats = STATE.NeuralEngine.GetStats()
            self._SendJSON(stats)
        else:
            self._SendJSON({"error": "Neural engine not available"}, 404)
    
    def _SendHypotheses(self):
        """Send generated hypotheses"""
        global STATE
        
        hypotheses = []
        if STATE.NeuralEngine and hasattr(STATE.NeuralEngine, 'GenerateHypotheses'):
            try:
                hypos = STATE.NeuralEngine.GenerateHypotheses(limit=10)
                hypotheses = hypos if hypos else []
            except:
                pass
        
        self._SendJSON({"hypotheses": hypotheses})
    
    def _SendPipelineStats(self):
        """Send neural pipeline stats"""
        global STATE
        
        if STATE.NeuralPipeline:
            stats = STATE.NeuralPipeline.GetStats()
            self._SendJSON(stats)
        else:
            self._SendJSON({"error": "Pipeline not available"}, 404)
    
    def _HandleChat(self, data: dict):
        """Handle chat request"""
        global STATE
        
        # Accept both 'query' and 'message' keys
        query = data.get('query', '') or data.get('message', '')
        if not query:
            self._SendJSON({"error": "No query provided"}, 400)
            return
        
        if STATE.EnhancedChat:
            response = STATE.EnhancedChat.Chat(query)
            self._SendJSON({
                "answer": response.Answer,
                "confidence": response.Confidence,
                "mode": response.Mode.value,
                "neural": response.NeuralUsed,
                "path": [
                    {
                        "from": h.FromEntity,
                        "relation": h.Relation,
                        "to": h.ToEntity,
                        "weight": h.AttentionWeight
                    }
                    for h in response.ReasoningPath
                ] if response.ReasoningPath else [],
                "timeMs": response.ProcessingTimeMs
            })
        elif STATE.ChatEngine:
            response = STATE.ChatEngine.Chat(query)
            answer = response if isinstance(response, str) else response.get('answer', str(response))
            confidence = 0.5
            if isinstance(response, dict):
                confidence = response.get('confidence', 0.5)
            self._SendJSON({
                "answer": answer,
                "confidence": confidence,
                "mode": "symbolic",
                "neural": False
            })
        else:
            self._SendJSON({"error": "Chat engine not available"}, 503)
    
    def _HandleLearn(self, data: dict):
        """Handle learn request - CONTINUOUS learning until stopped"""
        global STATE
        
        if STATE.IsLearning:
            self._SendJSON({"status": "Already learning"})
            return
        
        def LearnTask():
            global STATE
            STATE.IsLearning = True
            STATE.CurrentArticle = "Starting..."
            STATE.ArticlesLearned = 0
            STATE.FactsThisSession = 0
            
            try:
                # Create learner connected to chat engine
                learner = WikipediaLearner(STATE.ChatEngine)
                
                # CONTINUOUS learning - runs until IsLearning is set to False
                while STATE.IsLearning:
                    try:
                        # Update status BEFORE fetching
                        STATE.CurrentArticle = f"Fetching article #{STATE.ArticlesLearned + 1}..."
                        
                        # Fetch random Wikipedia article
                        title, text = learner.FetchRandomWikipedia(Simple=True)
                        
                        if title and text:
                            # Update status to show current article
                            STATE.CurrentArticle = title
                            
                            # Learn from the text
                            facts_before = len(STATE.KnowledgeGraph.AllTriples) if STATE.KnowledgeGraph else 0
                            results = learner.LearnFromText(text, "wikipedia")
                            facts_after = len(STATE.KnowledgeGraph.AllTriples) if STATE.KnowledgeGraph else 0
                            
                            # Calculate actual new facts added
                            new_facts = facts_after - facts_before
                            
                            # Update counters IMMEDIATELY
                            STATE.ArticlesLearned += 1
                            STATE.FactsThisSession += new_facts
                            
                            # Store article in DocumentStore for RAG (if available)
                            if STATE.DocumentStore:
                                try:
                                    STATE.DocumentStore.AddArticle(title, text, "wikipedia", {
                                        "facts_extracted": new_facts
                                    })
                                except:
                                    pass  # Don't fail learning if document store fails
                            
                            # Add new triples to neural engine
                            if STATE.NeuralEngine and STATE.KnowledgeGraph:
                                for subj, pred, obj in list(STATE.KnowledgeGraph.AllTriples)[-new_facts:] if new_facts > 0 else []:
                                    STATE.NeuralEngine.AddTriple(subj, pred, obj)
                            
                            # Train neural network every 100 articles
                            if STATE.ArticlesLearned % 100 == 0 and STATE.ArticlesLearned > 0:
                                if STATE.NeuralEngine and hasattr(STATE.NeuralEngine, 'Train'):
                                    STATE.CurrentArticle = f"🧠 Training neural network..."
                                    STATE.NeuralEngine.Train(Epochs=50)  # More epochs since less frequent
                                    
                                    # Record loss history
                                    if hasattr(STATE.NeuralEngine, 'Stats') and STATE.NeuralEngine.Stats.LastLoss:
                                        STATE.LossHistory.append(STATE.NeuralEngine.Stats.LastLoss)
                                        if len(STATE.LossHistory) > 50:
                                            STATE.LossHistory = STATE.LossHistory[-50:]
                                    
                                    STATE.RefreshNeuralPipeline()
                        else:
                            STATE.CurrentArticle = "Retrying..."
                        
                        # Small delay to be nice to Wikipedia
                        time.sleep(0.5)
                        
                    except Exception as e:
                        STATE.CurrentArticle = f"Error: {str(e)[:30]}..."
                        print(f"Article error: {e}")
                        time.sleep(1)
                        continue
            
            except Exception as e:
                STATE.CurrentArticle = f"Fatal error: {str(e)}"
                print(f"Learning fatal error: {e}")
            
            finally:
                # Final training when stopped
                if STATE.ArticlesLearned > 0:
                    STATE.CurrentArticle = "Final training..."
                    if STATE.NeuralEngine and hasattr(STATE.NeuralEngine, 'Train'):
                        STATE.NeuralEngine.Train(Epochs=100)  # More thorough final training
                        if hasattr(STATE.NeuralEngine, 'Stats') and STATE.NeuralEngine.Stats.LastLoss:
                            STATE.LossHistory.append(STATE.NeuralEngine.Stats.LastLoss)
                    STATE.RefreshNeuralPipeline()
                
                STATE.IsLearning = False
                STATE.CurrentArticle = f"Stopped. +{STATE.ArticlesLearned} articles, +{STATE.FactsThisSession} facts"
        
        thread = threading.Thread(target=LearnTask, daemon=True)
        thread.start()
        
        self._SendJSON({"status": "Learning started (continuous)"})
    
    def _HandleTrain(self, data: dict):
        """Handle train request"""
        global STATE
        
        epochs = data.get('epochs', 100)
        
        if STATE.NeuralEngine and hasattr(STATE.NeuralEngine, 'Train'):
            STATE.NeuralEngine.Train(Epochs=epochs)
            
            # Record loss
            if hasattr(STATE.NeuralEngine, 'LastLoss') and STATE.NeuralEngine.LastLoss:
                STATE.LossHistory.append(STATE.NeuralEngine.LastLoss)
                if len(STATE.LossHistory) > 50:
                    STATE.LossHistory = STATE.LossHistory[-50:]
            
            STATE.RefreshNeuralPipeline()
            self._SendJSON({"status": f"Trained for {epochs} epochs"})
        else:
            self._SendJSON({"error": "Neural engine not available"}, 503)
    
    def _HandleStop(self):
        """Handle stop request"""
        global STATE
        STATE.IsLearning = False
        self._SendJSON({"status": "Stopped"})
    
    def _SendStorageStats(self):
        """Send storage statistics (VectorStore + DocumentStore)"""
        global STATE
        
        stats = {
            "vectorStore": None,
            "documentStore": None
        }
        
        if STATE.VectorStore:
            stats["vectorStore"] = STATE.VectorStore.GetStats()
        
        if STATE.DocumentStore:
            stats["documentStore"] = STATE.DocumentStore.GetStats()
        
        self._SendJSON(stats)
    
    def _HandleSimilarSearch(self, data: dict):
        """Handle similarity search request"""
        global STATE
        
        entity = data.get('entity', '')
        top_k = data.get('topK', 10)
        
        if not entity:
            self._SendJSON({"error": "Entity required"}, 400)
            return
        
        if not STATE.VectorStore:
            self._SendJSON({
                "error": "VectorStore not available. Install FAISS: pip install faiss-cpu",
                "similar": []
            })
            return
        
        similar = STATE.FindSimilarEntities(entity, top_k)
        self._SendJSON({
            "entity": entity,
            "similar": [{"entity": e, "score": round(s, 4)} for e, s in similar]
        })
    
    def _HandleRAGContext(self, data: dict):
        """Handle RAG context request"""
        global STATE
        
        query = data.get('query', '')
        
        if not query:
            self._SendJSON({"error": "Query required"}, 400)
            return
        
        if not STATE.DocumentStore:
            self._SendJSON({
                "error": "DocumentStore not available. Install ChromaDB: pip install chromadb",
                "context": ""
            })
            return
        
        context = STATE.GetRAGContext(query)
        self._SendJSON({
            "query": query,
            "context": context
        })
    
    def _HandleMigration(self, data: dict):
        """Handle data migration to new storage systems"""
        global STATE
        
        action = data.get('action', 'all')  # 'vector', 'document', or 'all'
        
        results = {"status": "Migration started", "results": {}}
        
        # Migrate to VectorStore
        if action in ['vector', 'all'] and STATE.VectorStore and STATE.NeuralEngine:
            try:
                STATE.VectorStore.SyncFromNeuralEngine(STATE.NeuralEngine)
                stats = STATE.VectorStore.GetStats()
                results["results"]["vectorStore"] = {
                    "status": "success",
                    "entities": stats.get("TotalEntities", 0)
                }
            except Exception as e:
                results["results"]["vectorStore"] = {"status": "error", "message": str(e)}
        
        # Migrate to DocumentStore
        if action in ['document', 'all'] and STATE.DocumentStore and STATE.KnowledgeGraph:
            try:
                STATE.SyncToDocumentStore()
                stats = STATE.DocumentStore.GetStats()
                results["results"]["documentStore"] = {
                    "status": "success",
                    "facts": stats.get("TotalFacts", 0)
                }
            except Exception as e:
                results["results"]["documentStore"] = {"status": "error", "message": str(e)}
        
        results["status"] = "Migration complete"
        self._SendJSON(results)


# =============================================================================
# CLI FUNCTIONS
# =============================================================================

def RunDashboard(port: int = 8080):
    """Run the web dashboard"""
    global STATE
    STATE = GlobalState.Initialize()
    
    print(f"\n🌐 Starting dashboard on http://localhost:{port}")
    print("   Press Ctrl+C to stop\n")
    
    server = HTTPServer(('localhost', port), GroundZeroHandler)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        server.shutdown()


def RunChat():
    """Run interactive chat"""
    global STATE
    STATE = GlobalState.Initialize()
    
    print("\n" + "=" * 60)
    print("💬 GroundZero AI Chat")
    print("=" * 60)
    print("Type 'quit' to exit, 'stats' for statistics\n")
    
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if query.lower() == 'quit':
            break
        
        if query.lower() == 'stats':
            if STATE.EnhancedChat:
                print(f"\n📊 Stats: {STATE.EnhancedChat.GetStats()}\n")
            continue
        
        if not query:
            continue
        
        if STATE.EnhancedChat:
            response = STATE.EnhancedChat.Chat(query)
            print(f"\n🧠 GroundZero: {response.Answer}")
            print(f"   [Mode: {response.Mode.value}, Confidence: {response.Confidence:.0%}]\n")
        elif STATE.ChatEngine:
            response = STATE.ChatEngine.Chat(query)
            print(f"\n🧠 GroundZero: {response}\n")
        else:
            print("\n❌ Chat engine not available\n")
    
    print("\n👋 Goodbye!\n")


def RunTests():
    """Run system tests"""
    global STATE
    STATE = GlobalState.Initialize()
    
    print("\n" + "=" * 60)
    print("🧪 Running Tests")
    print("=" * 60 + "\n")
    
    tests_passed = 0
    tests_failed = 0
    
    print("1. Testing Knowledge Graph...")
    if STATE.KnowledgeGraph and len(STATE.KnowledgeGraph.Triples) > 0:
        print(f"   ✅ {len(STATE.KnowledgeGraph.Triples)} triples loaded")
        tests_passed += 1
    else:
        print("   ❌ No triples")
        tests_failed += 1
    
    print("2. Testing Chat Engine...")
    if STATE.ChatEngine:
        try:
            response = STATE.ChatEngine.Chat("What is a test?")
            print(f"   ✅ Response: {str(response)[:50]}...")
            tests_passed += 1
        except Exception as e:
            print(f"   ❌ Error: {e}")
            tests_failed += 1
    else:
        print("   ❌ Not available")
        tests_failed += 1
    
    print("3. Testing Neural Engine...")
    if STATE.NeuralEngine:
        print(f"   ✅ Available")
        tests_passed += 1
    else:
        print("   ⚠️ Not available")
        tests_failed += 1
    
    print("4. Testing Neural Pipeline...")
    if STATE.NeuralPipeline:
        try:
            response = STATE.NeuralPipeline.Process("What is a dog?")
            print(f"   ✅ Response: {response.Answer[:50]}...")
            tests_passed += 1
        except Exception as e:
            print(f"   ❌ Error: {e}")
            tests_failed += 1
    else:
        print("   ⚠️ Not available")
        tests_failed += 1
    
    print("5. Testing Enhanced Chat...")
    if STATE.EnhancedChat:
        try:
            response = STATE.EnhancedChat.Chat("Why do dogs bark?")
            print(f"   ✅ Mode: {response.Mode.value}, Answer: {response.Answer[:40]}...")
            tests_passed += 1
        except Exception as e:
            print(f"   ❌ Error: {e}")
            tests_failed += 1
    else:
        print("   ⚠️ Not available")
        tests_failed += 1
    
    print("\n" + "=" * 60)
    print(f"📋 Results: {tests_passed} passed, {tests_failed} failed")
    print("=" * 60 + "\n")
    
    return tests_failed == 0


def RunLearn():
    """Run auto-learning"""
    global STATE
    STATE = GlobalState.Initialize()
    
    print("\n" + "=" * 60)
    print("📚 Starting Auto-Learning")
    print("=" * 60)
    print("Press Ctrl+C to stop\n")
    
    learner = WikipediaLearner(STATE.KnowledgeGraph)
    articles = 0
    
    try:
        while True:
            result = learner.LearnRandomArticle()
            articles += 1
            
            print(f"[{articles}] Learned: {result.get('title', 'Unknown')[:40]}... "
                  f"(+{result.get('facts', 0)} facts)")
            
            if articles % 100 == 0:
                STATE.SyncKnowledgeToNeural()
                if STATE.NeuralEngine and hasattr(STATE.NeuralEngine, 'Train'):
                    print(f"\n🔄 Training neural network...")
                    STATE.NeuralEngine.Train(Epochs=50)
                    
                    if hasattr(STATE.NeuralEngine, 'LastLoss') and STATE.NeuralEngine.LastLoss:
                        STATE.LossHistory.append(STATE.NeuralEngine.LastLoss)
                    
                    STATE.RefreshNeuralPipeline()
                    print(f"   Done! Total triples: {len(STATE.KnowledgeGraph.Triples)}\n")
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print(f"\n\n📊 Learned {articles} articles")
        print("🔄 Final training...")
        
        if STATE.NeuralEngine and hasattr(STATE.NeuralEngine, 'Train'):
            STATE.NeuralEngine.Train(Epochs=100)
            STATE.RefreshNeuralPipeline()
        
        print("✅ Done!\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        RunDashboard()
    else:
        command = sys.argv[1].lower()
        
        if command == 'test':
            success = RunTests()
            sys.exit(0 if success else 1)
        elif command == 'chat':
            RunChat()
        elif command == 'learn':
            RunLearn()
        elif command == 'dashboard':
            port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
            RunDashboard(port)
        else:
            print(f"Unknown command: {command}")
            print("Usage: python main.py [test|chat|learn|dashboard]")
            sys.exit(1)


if __name__ == "__main__":
    main()