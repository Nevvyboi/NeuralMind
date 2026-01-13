"""
Metacognition Module
====================

The system's ability to think about its own thinking.
Knows what it knows and doesn't know.

Based on research:
- SOFAI Architecture (Booch et al. 2021)
- Metacognition in LLMs (2025)
- Kahneman's dual-process theory

Features:
- Confidence calibration
- Knowledge gap detection
- System 1/2 selection
- Uncertainty acknowledgment
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .knowledge_graph import KnowledgeGraph
    from .question_detector import QuestionType, ThinkingMode


@dataclass
class MetacognitiveState:
    """Current metacognitive state of the system"""
    ConfidenceInAnswer: float = 0.5
    UncertaintyAreas: List[str] = field(default_factory=list)
    ReasoningStepsTaken: int = 0
    SelfCorrections: int = 0
    KnowledgeGaps: List[str] = field(default_factory=list)
    ThinkingMode: 'ThinkingMode' = None
    
    def __post_init__(self):
        # Import here to avoid circular imports
        from .question_detector import ThinkingMode
        if self.ThinkingMode is None:
            self.ThinkingMode = ThinkingMode.MEDIUM


class MetacognitiveController:
    """
    Metacognition: The system's ability to think about its own thinking.
    
    Key capabilities:
    1. Know what it knows and doesn't know (epistemic awareness)
    2. Monitor reasoning quality during processing
    3. Decide when to use System 1 vs System 2 thinking
    4. Self-correct errors when detected
    5. Calibrate confidence appropriately
    """
    
    def __init__(self):
        self.State = MetacognitiveState()
        self.ReasoningHistory: List[Dict] = []
        self.PerformanceMetrics = {
            "TotalQuestions": 0,
            "CorrectPredictions": 0,
            "Overconfident": 0,
            "Underconfident": 0,
        }
    
    def AssessQuestion(self, Question: str, QType: 'QuestionType', 
                       Knowledge: Optional['KnowledgeGraph'] = None) -> MetacognitiveState:
        """
        Assess a question before answering.
        Determines confidence and appropriate thinking mode.
        
        Args:
            Question: The user's question
            QType: Detected question type
            Knowledge: Knowledge graph to check coverage
        
        Returns:
            MetacognitiveState with confidence and recommendations
        """
        from .question_detector import QuestionType, ThinkingMode, QuestionTypeDetector
        
        State = MetacognitiveState()
        
        # Extract key concepts from question
        Words = set(Question.lower().split())
        
        # Check knowledge coverage
        KnownConcepts = 0
        UnknownConcepts = []
        
        for Word in Words:
            if len(Word) > 3:  # Skip short words
                if Knowledge:
                    Facts = Knowledge.Query(Subject=Word)
                    if Facts:
                        KnownConcepts += 1
                    else:
                        UnknownConcepts.append(Word)
                else:
                    UnknownConcepts.append(Word)
        
        # Calculate initial confidence based on knowledge coverage
        TotalConcepts = len([W for W in Words if len(W) > 3])
        if TotalConcepts > 0:
            KnowledgeCoverage = KnownConcepts / TotalConcepts
        else:
            KnowledgeCoverage = 0.3
        
        State.ConfidenceInAnswer = min(0.3 + KnowledgeCoverage * 0.5, 0.9)
        State.KnowledgeGaps = UnknownConcepts[:5]  # Top 5 unknown concepts
        
        # Determine thinking mode based on question type
        Detector = QuestionTypeDetector()
        State.ThinkingMode = Detector.GetThinkingMode(QType)
        
        # Identify uncertainty areas based on question type
        if QType == QuestionType.COUNTERFACTUAL:
            State.UncertaintyAreas.append("Counterfactual outcomes are inherently uncertain")
            State.ConfidenceInAnswer *= 0.8
        
        if QType == QuestionType.OPINION:
            State.UncertaintyAreas.append("Opinions vary; this is my perspective")
            State.ConfidenceInAnswer *= 0.7
        
        if QType == QuestionType.UNKNOWN:
            State.UncertaintyAreas.append("Question type unclear")
            State.ConfidenceInAnswer *= 0.8
        
        self.State = State
        return State
    
    def MonitorReasoning(self, Step: str, Confidence: float):
        """
        Monitor reasoning during processing.
        Detects potential errors and updates confidence.
        
        Args:
            Step: Description of reasoning step
            Confidence: Confidence in this step
        """
        self.State.ReasoningStepsTaken += 1
        
        # Check for potential errors
        ErrorIndicators = [
            "contradicts",
            "impossible",
            "doesn't make sense",
            "error",
            "mistake",
            "incorrect",
            "wrong",
        ]
        
        StepLower = Step.lower()
        for Indicator in ErrorIndicators:
            if Indicator in StepLower:
                self.State.SelfCorrections += 1
                self.State.ConfidenceInAnswer *= 0.9
                break
        
        # Update confidence based on reasoning progress
        # Weighted average of current and step confidence
        self.State.ConfidenceInAnswer = (
            self.State.ConfidenceInAnswer * 0.7 + Confidence * 0.3
        )
        
        # Record in history
        self.ReasoningHistory.append({
            "Step": Step,
            "Confidence": Confidence,
            "Timestamp": time.time()
        })
    
    def ShouldUseSystem2(self, QType: 'QuestionType', Complexity: float) -> bool:
        """
        Decide whether to use System 2 (deep) reasoning.
        
        System 1: Fast, intuitive, pattern-based
        System 2: Slow, deliberate, logical
        
        Args:
            QType: Question type
            Complexity: Estimated complexity (0-1)
        
        Returns:
            True if System 2 should be used
        """
        from .question_detector import QuestionType
        
        # Always use System 2 for complex question types
        if QType in [QuestionType.CAUSAL, QuestionType.COUNTERFACTUAL]:
            return True
        
        # Use System 2 if complexity is high
        if Complexity > 0.7:
            return True
        
        # Use System 2 if confidence is low
        if self.State.ConfidenceInAnswer < 0.5:
            return True
        
        return False  # Use System 1 (fast)
    
    def GenerateConfidenceStatement(self) -> str:
        """
        Generate honest statement about confidence level.
        
        Returns:
            Human-readable confidence statement
        """
        Conf = self.State.ConfidenceInAnswer
        
        if Conf >= 0.9:
            return "I'm very confident in this answer."
        elif Conf >= 0.7:
            return "I'm fairly confident, though there may be nuances I'm missing."
        elif Conf >= 0.5:
            return "I'm moderately confident, but please verify this information."
        elif Conf >= 0.3:
            return "I'm not very confident about this. Please take it with caution."
        else:
            return "I'm quite uncertain. This is my best guess based on limited information."
    
    def GetKnowledgeGapStatement(self) -> Optional[str]:
        """
        Acknowledge what the system doesn't know.
        
        Returns:
            Statement about knowledge gaps, or None if no significant gaps
        """
        if self.State.KnowledgeGaps:
            Gaps = ", ".join(self.State.KnowledgeGaps[:3])
            return f"I have limited knowledge about: {Gaps}"
        return None
    
    def Reset(self):
        """Reset state for new question"""
        self.State = MetacognitiveState()
        self.ReasoningHistory = []
