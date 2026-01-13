"""
GroundZero AI - Neurosymbolic Intelligence System
================================================

A system that achieves genuine understanding beyond pattern matching.

Components:
- KnowledgeGraph: Explicit fact storage
- CausalGraph: Cause-effect reasoning
- QuestionTypeDetector: Auto-detect question types
- MetacognitiveController: Self-awareness
- ChainOfThoughtReasoner: Step-by-step reasoning
- Constitution: Safety & alignment
- SmartChatEngine: Unified interface
- ProgressTracker: Learning progress
"""

from .knowledge_graph import KnowledgeGraph, KnowledgeTriple
from .causal_graph import CausalGraph, CausalRelation
from .question_detector import QuestionTypeDetector, QuestionType, ThinkingMode
from .metacognition import MetacognitiveController, MetacognitiveState
from .reasoning import ChainOfThoughtReasoner, ReasoningStep
from .constitutional import Constitution
from .chat_engine import SmartChatEngine, ChatResponse
from .progress_tracker import ProgressTracker

__version__ = "2.0.0"
__author__ = "GroundZero AI"

__all__ = [
    # Knowledge
    "KnowledgeGraph",
    "KnowledgeTriple",
    
    # Causal
    "CausalGraph", 
    "CausalRelation",
    
    # Detection
    "QuestionTypeDetector",
    "QuestionType",
    "ThinkingMode",
    
    # Metacognition
    "MetacognitiveController",
    "MetacognitiveState",
    
    # Reasoning
    "ChainOfThoughtReasoner",
    "ReasoningStep",
    
    # Constitutional
    "Constitution",
    
    # Chat
    "SmartChatEngine",
    "ChatResponse",
    
    # Progress
    "ProgressTracker",
]
