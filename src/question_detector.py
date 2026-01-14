"""
Question Type Detector Module
=============================

Automatically detects the type of question being asked.
This allows routing to the appropriate reasoning system
without requiring explicit commands.

Supported Types:
- FACTUAL: "What is X?"
- CAUSAL: "Why does X cause Y?"
- COUNTERFACTUAL: "What if X happened?"
- PROCEDURAL: "How do I do X?"
- COMPARATIVE: "Difference between X and Y?"
- DEFINITIONAL: "Define X"
- And more...
"""

import re
from enum import Enum, auto
from typing import Tuple, List, Dict, Pattern


class QuestionType(Enum):
    """Types of questions the system can detect and handle"""
    FACTUAL = auto()          # "What is X?" "Who is Y?"
    CAUSAL = auto()           # "Why does X cause Y?" "What causes Z?"
    COUNTERFACTUAL = auto()   # "What if X happened?" "What would happen if?"
    PROCEDURAL = auto()       # "How do I do X?" "What are the steps?"
    COMPARATIVE = auto()      # "What's the difference between X and Y?"
    DEFINITIONAL = auto()     # "What does X mean?" "Define Y"
    TEMPORAL = auto()         # "When did X happen?" "What happened after Y?"
    SPATIAL = auto()          # "Where is X?" "What's near Y?"
    QUANTITATIVE = auto()     # "How many?" "How much?"
    OPINION = auto()          # "What do you think about X?"
    CLARIFICATION = auto()    # "Can you explain more?" "What do you mean?"
    GREETING = auto()         # "Hello" "Hi" "How are you?"
    UNKNOWN = auto()          # Fallback


class ThinkingMode(Enum):
    """Adaptive thinking modes (like o3-mini's Low/Medium/High)"""
    FAST = auto()      # System 1 - quick, intuitive (1-3 steps)
    MEDIUM = auto()    # Balanced (3-7 steps)
    DEEP = auto()      # System 2 - slow, deliberate (7-15 steps)


class QuestionTypeDetector:
    """
    Automatically classify user questions.
    
    This allows the chat system to:
    1. Route factual questions → Knowledge Graph
    2. Route causal questions → Causal Graph
    3. Route counterfactual questions → Counterfactual Engine
    4. Adjust thinking depth appropriately
    """
    
    # Pattern definitions for each question type
    PATTERNS: Dict[QuestionType, List[str]] = {
        QuestionType.GREETING: [
            r'^(hi|hello|hey|greetings)(\s|!|\.|\?|$)',
            r'^good\s*(morning|afternoon|evening)',
            r'^how\s+are\s+you',
            r"^what'?s\s+up",
            r'^yo\b',
            r'^hi\s+there',
        ],
        QuestionType.COUNTERFACTUAL: [
            r'what\s+(would|could|might)\s+(happen|occur|result)',
            r'what\s+if\b',
            r'if\s+.+\s+(what|how|would)',
            r'imagine\s+if',
            r'suppose\s+(that\s+)?',
            r'hypothetically',
            r'in\s+a\s+world\s+where',
            r'had\s+.+\s+been',
        ],
        QuestionType.CAUSAL: [
            r'^why\s+(does|do|did|is|are|was|were)\b',
            r'^why\b',
            r'what\s+(causes?|leads?\s+to|results?\s+in)',
            r'how\s+(does|do|did)\s+.+\s+(cause|lead|result|affect)',
            r'because\s+of\s+what',
            r'what\s+is\s+the\s+reason',
            r'explain\s+why',
            r'what\s+made\s+',
            r'cause\s+.+\?$',
        ],
        QuestionType.PROCEDURAL: [
            r'how\s+(do|can|should|would)\s+(i|you|we|one)',
            r'what\s+are\s+the\s+steps',
            r'step[\s-]by[\s-]step',
            r'how\s+to\b',
            r'instructions?\s+for',
            r'guide\s+(me|to)',
            r'teach\s+me',
            r'walk\s+me\s+through',
        ],
        QuestionType.COMPARATIVE: [
            r'(what|how)\s+is\s+the\s+difference',
            r'compare\b',
            r'versus|vs\.?\b',
            r'better\s+than|worse\s+than',
            r'similar\s+to|different\s+from',
            r'which\s+(is|are)\s+(better|worse|faster|slower)',
            r'pros\s+and\s+cons',
        ],
        QuestionType.DEFINITIONAL: [
            r'^what\s+(is|are)\s+(a|an|the)?\s*\w+',
            r'define\b',
            r'definition\s+of',
            r'what\s+does\s+.+\s+mean',
            r'meaning\s+of',
            r'explain\s+(the\s+)?(concept|term|word)',
            r'^what\s+is\s+\w+\??$',
        ],
        QuestionType.TEMPORAL: [
            r'when\s+(did|does|will|was|were|is)',
            r'what\s+(time|date|year|day)',
            r'how\s+long\s+(ago|until|did)',
            r'before|after\s+.+\s+(happen|occur)',
            r'timeline\s+of',
            r'history\s+of',
        ],
        QuestionType.SPATIAL: [
            r'where\s+(is|are|was|were|did)',
            r'location\s+of',
            r'(near|close\s+to|far\s+from)\s+what',
            r'in\s+which\s+(place|country|city)',
            r'located\s+',
        ],
        QuestionType.QUANTITATIVE: [
            r'how\s+(many|much|often|long)',
            r'what\s+(number|amount|quantity)',
            r'count\s+of',
            r'\d+\s*(times|percent|%)',
            r'total\s+',
        ],
        QuestionType.OPINION: [
            r'what\s+do\s+you\s+(think|believe|feel)',
            r'your\s+opinion',
            r'(do|would)\s+you\s+(recommend|suggest|prefer)',
            r'is\s+it\s+(good|bad|worth)',
            r'should\s+i',
            r'what\s+would\s+you\s+do',
        ],
        QuestionType.CLARIFICATION: [
            r'(can|could)\s+you\s+(explain|elaborate|clarify)',
            r'what\s+do\s+you\s+mean',
            r'i\s+don\'?t\s+understand',
            r'more\s+(details?|information)',
            r'tell\s+me\s+more',
            r'expand\s+on',
        ],
        QuestionType.FACTUAL: [
            r'^(what|who|which|whose)\s+(is|are|was|were)',
            r'^tell\s+me\s+about',
            r'^(do|does|did|is|are|was|were)\s+\w+\s+\w+\?$',
            r'fact\s+about',
            r'true\s+that',
            r'is\s+it\s+true',
        ],
    }
    
    def __init__(self):
        """Initialize with compiled patterns for efficiency"""
        self.CompiledPatterns: Dict[QuestionType, List[Pattern]] = {
            QType: [re.compile(P, re.IGNORECASE) for P in Patterns]
            for QType, Patterns in self.PATTERNS.items()
        }
    
    def Detect(self, Question: str) -> Tuple[QuestionType, float]:
        """
        Detect the type of question.
        
        Args:
            Question: The user's question
        
        Returns:
            Tuple of (QuestionType, confidence score 0-1)
        """
        Question = Question.strip()
        
        if not Question:
            return QuestionType.UNKNOWN, 0.0
        
        # Check each question type
        Scores: Dict[QuestionType, float] = {}
        
        for QType, Patterns in self.CompiledPatterns.items():
            MatchCount = 0
            for Pattern in Patterns:
                if Pattern.search(Question):
                    MatchCount += 1
            
            if MatchCount > 0:
                # Score based on number of matching patterns
                # Base score of 0.3, plus match ratio
                Scores[QType] = min(MatchCount / len(Patterns) + 0.3, 1.0)
        
        if not Scores:
            return QuestionType.UNKNOWN, 0.3
        
        # Return highest scoring type
        BestType = max(Scores, key=Scores.get)
        return BestType, Scores[BestType]
    
    def GetThinkingMode(self, QType: QuestionType) -> ThinkingMode:
        """
        Determine appropriate thinking depth for question type.
        
        Complex questions get deeper thinking:
        - CAUSAL, COUNTERFACTUAL → DEEP (System 2)
        - OPINION, PROCEDURAL → MEDIUM
        - GREETING, FACTUAL → FAST (System 1)
        
        Args:
            QType: The detected question type
        
        Returns:
            Appropriate ThinkingMode
        """
        # Questions requiring deep reasoning
        if QType in [QuestionType.CAUSAL, QuestionType.COUNTERFACTUAL, 
                     QuestionType.COMPARATIVE]:
            return ThinkingMode.DEEP
        
        # Questions needing medium effort
        if QType in [QuestionType.OPINION, QuestionType.CLARIFICATION,
                     QuestionType.PROCEDURAL]:
            return ThinkingMode.MEDIUM
        
        # Quick responses for simple questions
        return ThinkingMode.FAST
    
    def GetDescription(self, QType: QuestionType) -> str:
        """Get human-readable description of question type"""
        Descriptions = {
            QuestionType.FACTUAL: "Factual question seeking information",
            QuestionType.CAUSAL: "Causal question asking about cause-effect",
            QuestionType.COUNTERFACTUAL: "Counterfactual/hypothetical question",
            QuestionType.PROCEDURAL: "Procedural question asking how to do something",
            QuestionType.COMPARATIVE: "Comparative question comparing things",
            QuestionType.DEFINITIONAL: "Definitional question seeking meaning",
            QuestionType.TEMPORAL: "Temporal question about time/sequence",
            QuestionType.SPATIAL: "Spatial question about location",
            QuestionType.QUANTITATIVE: "Quantitative question about amounts",
            QuestionType.OPINION: "Opinion question seeking perspective",
            QuestionType.CLARIFICATION: "Clarification request",
            QuestionType.GREETING: "Greeting/salutation",
            QuestionType.UNKNOWN: "Unknown question type",
        }
        return Descriptions.get(QType, "Unknown")

# Alias for main.py compatibility
QuestionDetector = QuestionTypeDetector