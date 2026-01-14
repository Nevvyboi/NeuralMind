"""
GroundZero AI - Neural Pipeline Module
======================================

This module contains the neural components for the GroundZero AI system:

- TinyLM: Lightweight language model for query understanding
- AttentionReasoner: Multi-hop reasoning with attention tracking
- ContextGNN: Graph Neural Network for context propagation
- NeuralPipeline: Full pipeline orchestration
- ChatEngineEnhanced: Integrated chat engine

Usage:
    from src import NeuralPipeline, ChatEngineEnhanced
    
    pipeline = NeuralPipeline(EmbedDim=100)
    pipeline.Initialize(entity_embs, rel_embs, triples, graph)
    response = pipeline.Process("What is a dog?")
"""

# Version
__version__ = "2.0.0"
__author__ = "GroundZero AI"

# Check if neural components are available
NEURAL_PIPELINE_AVAILABLE = False

try:
    from .tiny_lm import TinyLM, QueryAnalysis, QuestionType
    from .attention_reasoner import (
        AttentionReasoner, 
        ReasoningHop, 
        ReasoningPath, 
        CandidateAnswer
    )
    from .context_gnn import ContextGNN, GNNNode, GNNOutput
    from .neural_pipeline import NeuralPipeline, PipelineResponse, RelevantSubgraph
    from .chat_engine_enhanced import (
        ChatEngineEnhanced, 
        EnhancedResponse, 
        ProcessingMode,
        CreateEnhancedChatEngine
    )
    
    NEURAL_PIPELINE_AVAILABLE = True
    
except ImportError as e:
    print(f"⚠️ Warning: Some neural components could not be imported: {e}")
    
    # Provide stub classes for graceful degradation
    class TinyLM:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("TinyLM not available")
    
    class AttentionReasoner:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("AttentionReasoner not available")
    
    class ContextGNN:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("ContextGNN not available")
    
    class NeuralPipeline:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("NeuralPipeline not available")
    
    class ChatEngineEnhanced:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("ChatEngineEnhanced not available")


# Export all public classes
__all__ = [
    # Version info
    '__version__',
    '__author__',
    'NEURAL_PIPELINE_AVAILABLE',
    
    # TinyLM
    'TinyLM',
    'QueryAnalysis',
    'QuestionType',
    
    # AttentionReasoner
    'AttentionReasoner',
    'ReasoningHop',
    'ReasoningPath',
    'CandidateAnswer',
    
    # ContextGNN
    'ContextGNN',
    'GNNNode',
    'GNNOutput',
    
    # NeuralPipeline
    'NeuralPipeline',
    'PipelineResponse',
    'RelevantSubgraph',
    
    # ChatEngineEnhanced
    'ChatEngineEnhanced',
    'EnhancedResponse',
    'ProcessingMode',
    'CreateEnhancedChatEngine',
]


def GetPipelineInfo() -> dict:
    """Get information about the neural pipeline module"""
    return {
        "version": __version__,
        "author": __author__,
        "available": NEURAL_PIPELINE_AVAILABLE,
        "components": [
            "TinyLM",
            "AttentionReasoner", 
            "ContextGNN",
            "NeuralPipeline",
            "ChatEngineEnhanced"
        ],
        "features": [
            "Multi-hop reasoning (up to 3 hops)",
            "Graph attention network (4 heads, 2 layers)",
            "Question type classification (6 types)",
            "GreaseLM-style bidirectional fusion",
            "TransE embedding alignment"
        ]
    }


# Quick test function
def TestPipeline():
    """Quick test of pipeline components"""
    if not NEURAL_PIPELINE_AVAILABLE:
        print("❌ Neural pipeline not available")
        return False
    
    print("Testing neural pipeline components...")
    
    try:
        # Test TinyLM
        lm = TinyLM(EmbedDim=100)
        analysis = lm.Analyze("What is a dog?")
        print(f"✅ TinyLM: {analysis.QuestionType.value}")
        
        # Test AttentionReasoner
        reasoner = AttentionReasoner(EmbedDim=100)
        print(f"✅ AttentionReasoner: {reasoner.NumHeads} heads, {reasoner.MaxHops} hops")
        
        # Test ContextGNN
        gnn = ContextGNN(EmbedDim=100)
        print(f"✅ ContextGNN: {gnn.NumLayers} layers, {gnn.NumHeads} heads")
        
        # Test NeuralPipeline
        pipeline = NeuralPipeline(EmbedDim=100)
        print(f"✅ NeuralPipeline: {pipeline.FusionIterations} fusion iterations")
        
        print("\n✅ All components working!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    print(f"GroundZero AI Neural Pipeline v{__version__}")
    print("=" * 50)
    info = GetPipelineInfo()
    print(f"Available: {info['available']}")
    print(f"Components: {', '.join(info['components'])}")
    print("\nRunning tests...")
    TestPipeline()