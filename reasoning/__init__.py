from .engine import ReasoningEngine, ReasoningType, ReasoningResult, ReasoningStep
from .logic import LogicReasoner
from .math_solver import MathSolver
from .code_analyzer import CodeAnalyzer
from .metacognition import Metacognition
from .advanced_reasoning import (
    AdvancedReasoningEngine,
    ChainOfThought,
    TreeOfThoughts,
    SelfConsistency,
    MetacognitiveMonitor,
    SelfVerifier,
    Thought,
    ThoughtType,
    ReasoningPath
)
from .cognitive_architecture import CognitiveArchitecture, ThinkingMode, CognitiveState

# Next-gen reasoning (Claude/GPT-4/DeepSeek/Qwen inspired)
try:
    from .nextgen_reasoning import (
        NextGenReasoningEngine,
        ReasoningStrategy,
        ReasoningResult as NextGenReasoningResult,
        ReasoningStep as NextGenReasoningStep,
        QueryAnalyzer,
        ChainOfThoughtReasoner,
        SelfVerifier as NextGenSelfVerifier,
        HypothesisGenerator
    )
    NEXTGEN_AVAILABLE = True
except ImportError:
    NEXTGEN_AVAILABLE = False

__all__ = [
    "ReasoningEngine",
    "ReasoningType",
    "ReasoningResult",
    "ReasoningStep",
    "LogicReasoner",
    "MathSolver",
    "CodeAnalyzer",
    "Metacognition",
    # Advanced reasoning
    "AdvancedReasoningEngine",
    "ChainOfThought",
    "TreeOfThoughts",
    "SelfConsistency",
    "MetacognitiveMonitor",
    "SelfVerifier",
    "Thought",
    "ThoughtType",
    "ReasoningPath",
    # Cognitive architecture
    "CognitiveArchitecture",
    "ThinkingMode",
    "CognitiveState",
    # Next-gen reasoning
    "NextGenReasoningEngine",
    "ReasoningStrategy",
    "QueryAnalyzer",
    "ChainOfThoughtReasoner",
    "HypothesisGenerator",
    "NEXTGEN_AVAILABLE"
]