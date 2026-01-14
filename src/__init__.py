"""
GroundZero AI v4.0 - Source Package
===================================

A neurosymbolic AI system combining knowledge graphs with neural embeddings.
"""

# Core modules
from .knowledge_graph import KnowledgeGraph
from .causal_graph import CausalGraph
from .chat_engine import SmartChatEngine
from .reasoning import ChainOfThoughtReasoner, ChainOfThought
from .metacognition import MetacognitiveController, MetacognitionEngine
from .constitutional import Constitution, ConstitutionalAI
from .question_detector import QuestionTypeDetector, QuestionDetector
from .auto_learner import AutoLearner, WikipediaLearner
from .world_model import WorldModel
from .progress_tracker import ProgressTracker

# Neural modules (optional)
try:
    from .neural_engine import NeuralEngine
    from .neural_pipeline import NeuralPipeline
    from .chat_engine_enhanced import ChatEngineEnhanced
    from .tiny_lm import TinyLM
    from .attention_reasoner import AttentionReasoner
    from .context_gnn import ContextGNN
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False

# NLP modules (optional)
try:
    from .nlp_extractor import NLPExtractor
    from .nlp_integration import NLPIntegration
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

__version__ = "4.0.0"
__all__ = [
    # Core
    "KnowledgeGraph",
    "CausalGraph", 
    "SmartChatEngine",
    "ChainOfThoughtReasoner",
    "ChainOfThought",
    "MetacognitiveController",
    "MetacognitionEngine",
    "Constitution",
    "ConstitutionalAI",
    "QuestionTypeDetector",
    "QuestionDetector",
    "AutoLearner",
    "WikipediaLearner",
    "WorldModel",
    "ProgressTracker",
    # Neural (if available)
    "NeuralEngine",
    "NeuralPipeline",
    "ChatEngineEnhanced",
    "TinyLM",
    "AttentionReasoner",
    "ContextGNN",
]
