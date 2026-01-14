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
    
    # State
    IsInitialized: bool = False
    IsLearning: bool = False
    
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
            
            state.KnowledgeGraph = KnowledgeGraph()
            print(f"  ✅ KnowledgeGraph: {len(state.KnowledgeGraph.Triples)} triples")
            
            state.CausalGraph = CausalGraph()
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
    
    def SyncKnowledgeToNeural(self):
        """Sync knowledge graph facts to neural network"""
        if not self.NeuralEngine or not self.KnowledgeGraph:
            return
        
        if hasattr(self.NeuralEngine, 'AddFacts'):
            triples = list(self.KnowledgeGraph.Triples)
            self.NeuralEngine.AddFacts(triples)


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
        """Suppress default logging"""
        pass
    
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
                else:
                    self.send_error(404)
        
        except Exception as e:
            self._SendJSON({"error": str(e)}, 500)
    
    def _SendJSON(self, data: dict, status: int = 200):
        """Send JSON response"""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def _SendStats(self):
        """Send system stats in DASHBOARD-COMPATIBLE FORMAT"""
        global STATE
        
        # Calculate counts
        total_facts = len(STATE.KnowledgeGraph.Triples) if STATE.KnowledgeGraph else 0
        total_entities = len(set(
            e for t in STATE.KnowledgeGraph.Triples 
            for e in [t[0], t[2]]
        )) if STATE.KnowledgeGraph else 0
        total_relations = len(set(
            t[1] for t in STATE.KnowledgeGraph.Triples
        )) if STATE.KnowledgeGraph else 0
        
        # Get neural stats
        neural_epochs = 0
        neural_loss = None
        neural_predictions = 0
        neural_hypotheses = 0
        neural_embed_dim = 100
        neural_is_trained = False
        
        if STATE.NeuralEngine:
            if hasattr(STATE.NeuralEngine, 'Epochs'):
                neural_epochs = STATE.NeuralEngine.Epochs
            if hasattr(STATE.NeuralEngine, 'LastLoss'):
                neural_loss = STATE.NeuralEngine.LastLoss
            if hasattr(STATE.NeuralEngine, 'EmbeddingDim'):
                neural_embed_dim = STATE.NeuralEngine.EmbeddingDim
            if hasattr(STATE.NeuralEngine, 'IsTrained'):
                neural_is_trained = STATE.NeuralEngine.IsTrained
            if hasattr(STATE.NeuralEngine, 'Predictions'):
                neural_predictions = len(STATE.NeuralEngine.Predictions) if STATE.NeuralEngine.Predictions else 0
            if hasattr(STATE.NeuralEngine, 'Hypotheses'):
                neural_hypotheses = len(STATE.NeuralEngine.Hypotheses) if STATE.NeuralEngine.Hypotheses else 0
            
            # Try GetStats() for more info
            if hasattr(STATE.NeuralEngine, 'GetStats'):
                ns = STATE.NeuralEngine.GetStats()
                neural_epochs = ns.get('TrainingEpochs', ns.get('Epochs', neural_epochs))
                neural_loss = ns.get('LastLoss', ns.get('Loss', neural_loss))
                neural_embed_dim = ns.get('EmbeddingDim', ns.get('EmbedDim', neural_embed_dim))
                neural_is_trained = ns.get('IsTrained', neural_epochs > 0)
                neural_predictions = ns.get('Predictions', neural_predictions)
                neural_hypotheses = ns.get('Hypotheses', neural_hypotheses)
        
        # Get causal count
        causal_count = 0
        if STATE.CausalGraph and hasattr(STATE.CausalGraph, 'Relations'):
            causal_count = len(STATE.CausalGraph.Relations)
        elif STATE.CausalGraph and hasattr(STATE.CausalGraph, 'Edges'):
            causal_count = len(STATE.CausalGraph.Edges)
        
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
            "pipelineReady": STATE.NeuralPipeline is not None
        }
        
        self._SendJSON(stats)
    
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
        """Handle learn request"""
        global STATE
        
        if STATE.IsLearning:
            self._SendJSON({"status": "Already learning"})
            return
        
        def LearnTask():
            global STATE
            STATE.IsLearning = True
            
            try:
                learner = WikipediaLearner(STATE.KnowledgeGraph)
                
                articles_processed = 0
                while STATE.IsLearning and articles_processed < 100:
                    result = learner.LearnRandomArticle()
                    articles_processed += 1
                    
                    if articles_processed % 10 == 0:
                        STATE.SyncKnowledgeToNeural()
                        
                        if STATE.NeuralEngine and hasattr(STATE.NeuralEngine, 'Train'):
                            STATE.NeuralEngine.Train(epochs=20)
                            
                            # Record loss history
                            if hasattr(STATE.NeuralEngine, 'LastLoss') and STATE.NeuralEngine.LastLoss:
                                STATE.LossHistory.append(STATE.NeuralEngine.LastLoss)
                                if len(STATE.LossHistory) > 50:
                                    STATE.LossHistory = STATE.LossHistory[-50:]
                        
                        STATE.RefreshNeuralPipeline()
                    
                    time.sleep(0.5)
            
            finally:
                STATE.IsLearning = False
                
                if STATE.NeuralEngine and hasattr(STATE.NeuralEngine, 'Train'):
                    STATE.NeuralEngine.Train(epochs=50)
                    if hasattr(STATE.NeuralEngine, 'LastLoss') and STATE.NeuralEngine.LastLoss:
                        STATE.LossHistory.append(STATE.NeuralEngine.LastLoss)
                STATE.RefreshNeuralPipeline()
        
        thread = threading.Thread(target=LearnTask, daemon=True)
        thread.start()
        
        self._SendJSON({"status": "Learning started"})
    
    def _HandleTrain(self, data: dict):
        """Handle train request"""
        global STATE
        
        epochs = data.get('epochs', 100)
        
        if STATE.NeuralEngine and hasattr(STATE.NeuralEngine, 'Train'):
            STATE.NeuralEngine.Train(epochs=epochs)
            
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
            
            if articles % 10 == 0:
                STATE.SyncKnowledgeToNeural()
                if STATE.NeuralEngine and hasattr(STATE.NeuralEngine, 'Train'):
                    print(f"\n🔄 Training neural network...")
                    STATE.NeuralEngine.Train(epochs=20)
                    
                    if hasattr(STATE.NeuralEngine, 'LastLoss') and STATE.NeuralEngine.LastLoss:
                        STATE.LossHistory.append(STATE.NeuralEngine.LastLoss)
                    
                    STATE.RefreshNeuralPipeline()
                    print(f"   Done! Total triples: {len(STATE.KnowledgeGraph.Triples)}\n")
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print(f"\n\n📊 Learned {articles} articles")
        print("🔄 Final training...")
        
        if STATE.NeuralEngine and hasattr(STATE.NeuralEngine, 'Train'):
            STATE.NeuralEngine.Train(epochs=50)
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