from .engine import ResponseGenerator, QuestionType, ReasoningResult

# Context Brain replaces the old context module
try:
    from .context_brain import ContextBrain, SmartSearcher, QueryIntent
    CONTEXT_BRAIN_AVAILABLE = True
except ImportError:
    CONTEXT_BRAIN_AVAILABLE = False

__all__ = [
    "ResponseGenerator", 
    "QuestionType", 
    "ReasoningResult",
    "ContextBrain",
    "SmartSearcher", 
    "QueryIntent"
]