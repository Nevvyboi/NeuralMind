"""
ChatEngineEnhanced - Neural Pipeline Integrated Chat Engine
Part of GroundZero AI Neural Pipeline

Features:
- Integrates NeuralPipeline with existing SmartChatEngine
- Auto-detects when to use neural pipeline
- Falls back to symbolic reasoning if neural fails
- Tracks neural query statistics
"""

import re
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Import neural components
try:
    from .neural_pipeline import NeuralPipeline, PipelineResponse
    from .tiny_lm import QuestionType
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False


class ProcessingMode(Enum):
    """How the query was processed"""
    NEURAL = "neural"
    SYMBOLIC = "symbolic"
    HYBRID = "hybrid"
    FALLBACK = "fallback"


@dataclass
class EnhancedResponse:
    """Enhanced response with processing metadata"""
    Answer: str
    Confidence: float
    Mode: ProcessingMode
    ReasoningPath: List[Any]
    ProcessingTimeMs: float
    SourceFacts: List[str]
    NeuralUsed: bool


class ChatEngineEnhanced:
    """
    Enhanced chat engine that integrates neural pipeline with symbolic reasoning.
    
    Automatically routes queries to the most appropriate processing path:
    - Neural pipeline for complex questions (causal, counterfactual, comparative)
    - Symbolic for simple factual lookups
    - Hybrid for best results
    """
    
    def __init__(self, SymbolicEngine: Any = None, EmbedDim: int = 100):
        """
        Initialize enhanced chat engine.
        
        Args:
            SymbolicEngine: Existing symbolic chat engine (SmartChatEngine)
            EmbedDim: Embedding dimension for neural pipeline
        """
        self.SymbolicEngine = SymbolicEngine
        self.EmbedDim = EmbedDim
        
        # Neural pipeline (lazy initialized)
        self.NeuralPipeline: Optional[NeuralPipeline] = None
        self.NeuralInitialized = False
        
        # Processing statistics
        self.Stats = {
            "TotalQueries": 0,
            "NeuralQueries": 0,
            "SymbolicQueries": 0,
            "HybridQueries": 0,
            "FallbackQueries": 0,
            "AvgNeuralTimeMs": 0,
            "AvgSymbolicTimeMs": 0
        }
        
        # Neural trigger patterns (when to use neural pipeline)
        self.NeuralTriggerPatterns = [
            r'\bwhy\b',
            r'\bhow\b',
            r'\bcause[sd]?\b',
            r'\bcompare\b',
            r'\bdifference\b',
            r'\bsimilar\b',
            r'\bwhat if\b',
            r'\bwould\b',
            r'\bshould\b',
            r'\bcould\b',
            r'\brelat(ed|ion)\b',
            r'\bexplain\b',
            r'\bdescribe\b'
        ]
        
        # Word count threshold for neural
        self.MinWordsForNeural = 5
    
    def InitializeNeuralPipeline(
        self,
        EntityEmbeddings: Dict[str, List[float]],
        RelationEmbeddings: Dict[str, List[float]],
        EntityTriples: Dict[str, List[Tuple[str, str, str]]],
        Graph: Dict[str, List[Tuple[str, str]]]
    ):
        """
        Initialize the neural pipeline with knowledge graph data.
        
        Args:
            EntityEmbeddings: Entity embeddings from TransE
            RelationEmbeddings: Relation embeddings from TransE
            EntityTriples: Entity to triples mapping
            Graph: Adjacency list for GNN
        """
        if not NEURAL_AVAILABLE:
            print("⚠️ Neural components not available")
            return
        
        self.NeuralPipeline = NeuralPipeline(EmbedDim=self.EmbedDim)
        self.NeuralPipeline.Initialize(
            EntityEmbeddings,
            RelationEmbeddings,
            EntityTriples,
            Graph
        )
        self.NeuralInitialized = True
        print("✅ Neural pipeline initialized")
    
    def ShouldUseNeural(self, Query: str) -> bool:
        """
        Determine if a query should use the neural pipeline.
        
        Args:
            Query: User query
            
        Returns:
            True if neural pipeline should be used
        """
        if not self.NeuralInitialized:
            return False
        
        query_lower = Query.lower()
        
        # Check trigger patterns
        for pattern in self.NeuralTriggerPatterns:
            if re.search(pattern, query_lower):
                return True
        
        # Check word count
        words = query_lower.split()
        if len(words) >= self.MinWordsForNeural:
            return True
        
        # Check for question types that benefit from neural
        if self.NeuralPipeline:
            analysis = self.NeuralPipeline.TinyLM.Analyze(Query)
            neural_types = {
                QuestionType.CAUSAL,
                QuestionType.COUNTERFACTUAL,
                QuestionType.COMPARATIVE,
                QuestionType.PROCEDURAL
            }
            if analysis.QuestionType in neural_types:
                return True
        
        return False
    
    def _ProcessNeural(self, Query: str) -> Optional[PipelineResponse]:
        """Process query with neural pipeline"""
        if not self.NeuralInitialized or not self.NeuralPipeline:
            return None
        
        try:
            return self.NeuralPipeline.Process(Query)
        except Exception as e:
            print(f"Neural processing error: {e}")
            return None
    
    def _ProcessSymbolic(self, Query: str) -> Tuple[str, float, List[str]]:
        """Process query with symbolic engine"""
        if not self.SymbolicEngine:
            return "Symbolic engine not available.", 0.0, []
        
        try:
            # Call existing chat engine
            if hasattr(self.SymbolicEngine, 'Chat'):
                response = self.SymbolicEngine.Chat(Query)
                if isinstance(response, dict):
                    return response.get('answer', str(response)), response.get('confidence', 0.5), []
                return str(response), 0.5, []
            elif hasattr(self.SymbolicEngine, 'Process'):
                response = self.SymbolicEngine.Process(Query)
                return str(response), 0.5, []
            else:
                return "Unable to process query.", 0.0, []
        except Exception as e:
            print(f"Symbolic processing error: {e}")
            return f"Error: {e}", 0.0, []
    
    def Chat(self, Query: str) -> EnhancedResponse:
        """
        Process a chat query with intelligent routing.
        
        Args:
            Query: User query
            
        Returns:
            EnhancedResponse with answer and metadata
        """
        start_time = time.time()
        self.Stats["TotalQueries"] += 1
        
        # Determine processing mode
        use_neural = self.ShouldUseNeural(Query)
        
        if use_neural:
            # Try neural first
            neural_response = self._ProcessNeural(Query)
            
            if neural_response and neural_response.Confidence > 0.3:
                # Neural succeeded
                self.Stats["NeuralQueries"] += 1
                processing_time = (time.time() - start_time) * 1000
                self._UpdateAvgTime("AvgNeuralTimeMs", processing_time, self.Stats["NeuralQueries"])
                
                return EnhancedResponse(
                    Answer=neural_response.Answer,
                    Confidence=neural_response.Confidence,
                    Mode=ProcessingMode.NEURAL,
                    ReasoningPath=neural_response.ReasoningPath,
                    ProcessingTimeMs=processing_time,
                    SourceFacts=[f"{h.FromEntity} → {h.Relation} → {h.ToEntity}" 
                                for h in neural_response.ReasoningPath[:3]],
                    NeuralUsed=True
                )
            else:
                # Neural failed, try hybrid
                symbolic_answer, symbolic_conf, facts = self._ProcessSymbolic(Query)
                
                if neural_response and symbolic_answer:
                    # Combine results
                    self.Stats["HybridQueries"] += 1
                    
                    if neural_response.Confidence > symbolic_conf:
                        answer = neural_response.Answer
                        confidence = neural_response.Confidence
                        reasoning = neural_response.ReasoningPath
                    else:
                        answer = symbolic_answer
                        confidence = symbolic_conf
                        reasoning = []
                    
                    processing_time = (time.time() - start_time) * 1000
                    
                    return EnhancedResponse(
                        Answer=answer,
                        Confidence=confidence,
                        Mode=ProcessingMode.HYBRID,
                        ReasoningPath=reasoning,
                        ProcessingTimeMs=processing_time,
                        SourceFacts=facts,
                        NeuralUsed=True
                    )
                else:
                    # Pure fallback
                    self.Stats["FallbackQueries"] += 1
                    processing_time = (time.time() - start_time) * 1000
                    
                    return EnhancedResponse(
                        Answer=symbolic_answer or "I couldn't find an answer.",
                        Confidence=symbolic_conf,
                        Mode=ProcessingMode.FALLBACK,
                        ReasoningPath=[],
                        ProcessingTimeMs=processing_time,
                        SourceFacts=facts,
                        NeuralUsed=neural_response is not None
                    )
        else:
            # Use symbolic only
            symbolic_answer, symbolic_conf, facts = self._ProcessSymbolic(Query)
            self.Stats["SymbolicQueries"] += 1
            processing_time = (time.time() - start_time) * 1000
            self._UpdateAvgTime("AvgSymbolicTimeMs", processing_time, self.Stats["SymbolicQueries"])
            
            return EnhancedResponse(
                Answer=symbolic_answer,
                Confidence=symbolic_conf,
                Mode=ProcessingMode.SYMBOLIC,
                ReasoningPath=[],
                ProcessingTimeMs=processing_time,
                SourceFacts=facts,
                NeuralUsed=False
            )
    
    def _UpdateAvgTime(self, Key: str, NewTime: float, Count: int):
        """Update rolling average time"""
        if Count == 1:
            self.Stats[Key] = NewTime
        else:
            self.Stats[Key] = (self.Stats[Key] * (Count - 1) + NewTime) / Count
    
    def QuickAnswer(self, Query: str) -> str:
        """
        Get just the answer text (simple interface).
        
        Args:
            Query: User query
            
        Returns:
            Answer string
        """
        response = self.Chat(Query)
        return response.Answer
    
    def GetStats(self) -> Dict:
        """Get engine statistics"""
        stats = dict(self.Stats)
        stats["NeuralAvailable"] = NEURAL_AVAILABLE
        stats["NeuralInitialized"] = self.NeuralInitialized
        
        if self.NeuralPipeline:
            stats["PipelineStats"] = self.NeuralPipeline.GetStats()
        
        return stats
    
    def RefreshNeuralPipeline(
        self,
        EntityEmbeddings: Dict[str, List[float]],
        RelationEmbeddings: Dict[str, List[float]],
        EntityTriples: Dict[str, List[Tuple[str, str, str]]],
        Graph: Dict[str, List[Tuple[str, str]]]
    ):
        """
        Refresh neural pipeline with updated embeddings.
        
        Call this after retraining the neural network.
        """
        self.InitializeNeuralPipeline(
            EntityEmbeddings,
            RelationEmbeddings,
            EntityTriples,
            Graph
        )


# =============================================================================
# STANDALONE CHAT FUNCTION
# =============================================================================

def CreateEnhancedChatEngine(
    KnowledgeGraph: Any = None,
    NeuralEngine: Any = None,
    EmbedDim: int = 100
) -> ChatEngineEnhanced:
    """
    Factory function to create enhanced chat engine.
    
    Args:
        KnowledgeGraph: Knowledge graph instance
        NeuralEngine: Neural engine with TransE embeddings
        EmbedDim: Embedding dimension
        
    Returns:
        Configured ChatEngineEnhanced
    """
    engine = ChatEngineEnhanced(
        SymbolicEngine=KnowledgeGraph,
        EmbedDim=EmbedDim
    )
    
    if NeuralEngine and hasattr(NeuralEngine, 'EntityEmbeddings'):
        # Build required data structures
        entity_embs = getattr(NeuralEngine, 'EntityEmbeddings', {})
        rel_embs = getattr(NeuralEngine, 'RelationEmbeddings', {})
        
        # Build entity triples from knowledge graph
        entity_triples = {}
        if KnowledgeGraph and hasattr(KnowledgeGraph, 'Triples'):
            for subj, rel, obj in KnowledgeGraph.Triples:
                subj_lower = subj.lower()
                if subj_lower not in entity_triples:
                    entity_triples[subj_lower] = []
                entity_triples[subj_lower].append((subj, rel, obj))
                
                obj_lower = obj.lower()
                if obj_lower not in entity_triples:
                    entity_triples[obj_lower] = []
                entity_triples[obj_lower].append((subj, rel, obj))
        
        # Build graph adjacency list
        graph = {}
        if KnowledgeGraph and hasattr(KnowledgeGraph, 'Triples'):
            for subj, rel, obj in KnowledgeGraph.Triples:
                subj_lower = subj.lower()
                if subj_lower not in graph:
                    graph[subj_lower] = []
                graph[subj_lower].append((obj.lower(), rel))
                
                obj_lower = obj.lower()
                if obj_lower not in graph:
                    graph[obj_lower] = []
                graph[obj_lower].append((subj.lower(), rel))
        
        engine.InitializeNeuralPipeline(
            entity_embs,
            rel_embs,
            entity_triples,
            graph
        )
    
    return engine


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing ChatEngineEnhanced...")
    
    # Create without symbolic engine for testing
    engine = ChatEngineEnhanced(SymbolicEngine=None, EmbedDim=100)
    
    # Test without neural (should fall back gracefully)
    test_queries = [
        "What is a dog?",
        "Why do birds fly?",
        "Hello"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = engine.Chat(query)
        print(f"  Answer: {response.Answer}")
        print(f"  Mode: {response.Mode.value}")
        print(f"  Neural Used: {response.NeuralUsed}")
    
    print(f"\nStats: {engine.GetStats()}")