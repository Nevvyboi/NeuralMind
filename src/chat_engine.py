"""
Smart Chat Engine Module
========================

Integrates all components into a unified chat interface.

Features:
- Auto-detects question type
- Routes to appropriate reasoning system
- Uses System 1/2 thinking appropriately
- Applies constitutional checks
- Provides metacognitive awareness
- PERSISTENT storage for both facts AND causal relations
"""

import time
import random
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import Counter

from .knowledge_graph import KnowledgeGraph
from .causal_graph import CausalGraph
from .question_detector import QuestionTypeDetector, QuestionType, ThinkingMode
from .metacognition import MetacognitiveController, MetacognitiveState
from .reasoning import ChainOfThoughtReasoner, ReasoningStep
from .constitutional import Constitution


@dataclass
class ChatResponse:
    """Complete response with all metadata"""
    Answer: str
    QuestionType: QuestionType
    ThinkingMode: ThinkingMode
    Confidence: float
    ReasoningSteps: List[ReasoningStep]
    ConstitutionalCheck: Dict
    MetacognitiveState: MetacognitiveState
    ProcessingTime: float


class SmartChatEngine:
    """
    Smart chat engine that:
    1. Auto-detects question type
    2. Routes to appropriate reasoning system
    3. Uses System 1/2 thinking appropriately
    4. Applies constitutional checks
    5. Provides metacognitive awareness
    6. PERSISTS both facts and causal relations!
    """
    
    def __init__(self, Knowledge: 'KnowledgeGraph' = None, 
                 Causal: 'CausalGraph' = None, DataDir: str = "./data"):
        """
        Initialize chat engine.
        
        Args:
            Knowledge: KnowledgeGraph instance (optional, creates new if None)
            Causal: CausalGraph instance (optional, creates new if None)
            DataDir: Directory for data storage
        """
        self.DataDir = Path(DataDir)
        self.DataDir.mkdir(parents=True, exist_ok=True)
        
        # Use provided instances or create new ones
        self.Knowledge = Knowledge if Knowledge else KnowledgeGraph(str(self.DataDir / "knowledge.db"))
        self.Causal = Causal if Causal else CausalGraph(str(self.DataDir))
        
        self.QuestionDetector = QuestionTypeDetector()
        self.Metacognition = MetacognitiveController()
        self.Reasoner = ChainOfThoughtReasoner(self.Knowledge, self.Causal)
        
        # Conversation history
        self.History: List[Dict] = []
        
        # Statistics
        self.Stats = {
            "TotalQuestions": 0,
            "QuestionTypes": Counter(),
            "AverageConfidence": 0.0,
            "ThinkingModes": Counter(),
        }
    
    def Process(self, UserInput: str) -> ChatResponse:
        """
        Process user input and generate response.
        
        Args:
            UserInput: The user's message
        
        Returns:
            ChatResponse with answer and metadata
        """
        StartTime = time.time()
        
        # 1. Detect question type
        QType, TypeConfidence = self.QuestionDetector.Detect(UserInput)
        self.Stats["QuestionTypes"][QType.name] += 1
        
        # 2. Assess metacognitively
        MetaState = self.Metacognition.AssessQuestion(
            UserInput, QType, self.Knowledge
        )
        
        # 3. Determine thinking mode
        Mode = MetaState.ThinkingMode
        self.Stats["ThinkingModes"][Mode.name] += 1
        
        # 4. Generate response based on question type
        Answer, ReasoningSteps = self._GenerateResponse(UserInput, QType, Mode)
        
        # 5. Apply constitutional check
        ConstitutionalCheck = Constitution.Evaluate(Answer, UserInput)
        
        # 6. Update statistics
        self.Stats["TotalQuestions"] += 1
        RunningAvg = self.Stats["AverageConfidence"]
        N = self.Stats["TotalQuestions"]
        self.Stats["AverageConfidence"] = (RunningAvg * (N - 1) + MetaState.ConfidenceInAnswer) / N
        
        # 7. Build response
        ProcessingTime = time.time() - StartTime
        
        Response = ChatResponse(
            Answer=Answer,
            QuestionType=QType,
            ThinkingMode=Mode,
            Confidence=MetaState.ConfidenceInAnswer,
            ReasoningSteps=ReasoningSteps,
            ConstitutionalCheck=ConstitutionalCheck,
            MetacognitiveState=MetaState,
            ProcessingTime=ProcessingTime
        )
        
        # 8. Save to history
        self.History.append({
            "Input": UserInput,
            "Response": Response,
            "Timestamp": time.time()
        })
        
        return Response
    
    def _GenerateResponse(self, Question: str, QType: QuestionType, 
                          Mode: ThinkingMode) -> Tuple[str, List[ReasoningStep]]:
        """Generate response based on question type"""
        
        # Generate reasoning chain
        Steps = self.Reasoner.Think(Question, Mode)
        
        # Handle different question types
        if QType == QuestionType.GREETING:
            return self._HandleGreeting(Question), Steps
        
        elif QType == QuestionType.FACTUAL:
            return self._HandleFactual(Question), Steps
        
        elif QType == QuestionType.CAUSAL:
            return self._HandleCausal(Question), Steps
        
        elif QType == QuestionType.COUNTERFACTUAL:
            return self._HandleCounterfactual(Question), Steps
        
        elif QType == QuestionType.PROCEDURAL:
            return self._HandleProcedural(Question), Steps
        
        elif QType == QuestionType.COMPARATIVE:
            return self._HandleComparative(Question), Steps
        
        elif QType == QuestionType.DEFINITIONAL:
            return self._HandleDefinitional(Question), Steps
        
        elif QType == QuestionType.OPINION:
            return self._HandleOpinion(Question), Steps
        
        else:
            return self._HandleGeneral(Question), Steps
    
    def _HandleGreeting(self, Question: str) -> str:
        Greetings = [
            "Hello! I'm GroundZero AI, designed to understand and reason. How can I help you today?",
            "Hi there! I'm ready to help you with questions, explanations, and reasoning. What would you like to know?",
            "Greetings! I'm here to assist with information, causal reasoning, and explanations. What's on your mind?",
        ]
        return random.choice(Greetings)
    
    def _HandleFactual(self, Question: str) -> str:
        """Handle factual questions using knowledge graph"""
        Words = Question.lower().split()
        
        # Search for relevant facts
        AllFacts = []
        for Word in Words:
            if len(Word) > 3:
                Facts = self.Knowledge.Query(Subject=Word)
                AllFacts.extend(Facts)
                Facts = self.Knowledge.Query(Object=Word)
                AllFacts.extend(Facts)
        
        if AllFacts:
            # Format facts into answer
            FactStrings = list(set(F.ToNaturalLanguage() for F in AllFacts[:5]))
            return "Based on my knowledge:\n\n" + "\n".join(f"• {F}" for F in FactStrings)
        
        return "I don't have specific factual information about this in my knowledge base yet. Would you like to teach me?"
    
    def _HandleCausal(self, Question: str) -> str:
        """Handle causal questions"""
        Words = Question.lower().split()
        
        Explanations = []
        for Word in Words:
            if len(Word) > 3:
                Effects = self.Causal.GetEffects(Word)
                for E in Effects[:2]:
                    Explanations.append(f"• {E.Cause} causes {E.Effect} ({E.Strength:.0%} strength)")
                
                Causes = self.Causal.GetCauses(Word)
                for C in Causes[:2]:
                    Explanations.append(f"• {C.Cause} causes {C.Effect} ({C.Strength:.0%} strength)")
        
        if Explanations:
            return "Here's my causal analysis:\n\n" + "\n".join(set(Explanations))
        
        return "I don't have causal information about this yet. As I learn more, I'll be able to explain cause-effect relationships better."
    
    def _HandleCounterfactual(self, Question: str) -> str:
        """Handle counterfactual/hypothetical questions"""
        Match = re.search(r'(?:what\s+if|if)\s+(.+?)(?:\?|$)', Question.lower())
        
        if Match:
            Event = Match.group(1).strip()
            Words = Event.split()
            
            for Word in Words:
                if len(Word) > 3:
                    Effects = self.Causal.Counterfactual(Word, Intervention=True)
                    if Effects:
                        Response = f"If '{Word}' happened, here are the likely effects:\n\n"
                        for Effect, Prob in sorted(Effects.items(), key=lambda X: -X[1])[:5]:
                            Change = "more" if Prob > 0 else "less"
                            Response += f"• {Effect} would become {abs(Prob):.0%} {Change} likely\n"
                        return Response
        
        return "That's an interesting hypothetical! While I can reason about it, I don't have enough causal knowledge to predict specific outcomes yet."
    
    def _HandleProcedural(self, Question: str) -> str:
        """Handle how-to questions"""
        return ("Here's my general approach:\n\n"
                "1. First, understand the goal clearly\n"
                "2. Break it into smaller, manageable steps\n"
                "3. Execute each step carefully\n"
                "4. Verify the results at each stage\n\n"
                "Would you like me to elaborate on any specific step?")
    
    def _HandleComparative(self, Question: str) -> str:
        """Handle comparison questions"""
        Match = re.search(r'between\s+(\w+)\s+and\s+(\w+)', Question.lower())
        if Match:
            Item1, Item2 = Match.groups()
            
            Facts1 = self.Knowledge.Query(Subject=Item1)
            Facts2 = self.Knowledge.Query(Subject=Item2)
            
            Response = f"Comparing {Item1} and {Item2}:\n\n"
            
            if Facts1:
                Response += f"**{Item1.title()}:**\n"
                for F in Facts1[:3]:
                    Response += f"  • {F.ToNaturalLanguage()}\n"
            
            if Facts2:
                Response += f"\n**{Item2.title()}:**\n"
                for F in Facts2[:3]:
                    Response += f"  • {F.ToNaturalLanguage()}\n"
            
            if not Facts1 and not Facts2:
                Response = "I don't have enough information to compare these concepts yet."
            
            return Response
        
        return "I'd be happy to compare these concepts. Could you specify which aspects you'd like me to focus on?"
    
    def _HandleDefinitional(self, Question: str) -> str:
        """Handle definition questions"""
        Match = re.search(r'(?:what\s+is|define|meaning\s+of)\s+(?:a|an|the)?\s*(\w+)', Question.lower())
        
        if Match:
            Term = Match.group(1)
            Facts = self.Knowledge.Query(Subject=Term)
            
            if Facts:
                Response = f"**{Term.title()}**\n\n"
                for F in Facts[:5]:
                    Response += f"• {F.ToNaturalLanguage()}\n"
                return Response
        
        return "I don't have a specific definition for this term in my knowledge base yet. Would you like to teach me?"
    
    def _HandleOpinion(self, Question: str) -> str:
        """Handle opinion questions with appropriate uncertainty"""
        return ("In my perspective (acknowledging that this is an AI viewpoint):\n\n"
                "This is a nuanced topic with multiple valid perspectives. "
                "I'd encourage you to weigh different viewpoints before forming your own conclusion.\n\n"
                "I'm happy to explore specific aspects if you'd like to discuss further.")
    
    def _HandleGeneral(self, Question: str) -> str:
        """Handle general questions"""
        Words = Question.lower().split()
        AllFacts = []
        
        for Word in Words:
            if len(Word) > 3:
                AllFacts.extend(self.Knowledge.Query(Subject=Word)[:2])
                AllFacts.extend(self.Knowledge.GetRelated(Word, MaxDepth=1)[:2])
        
        if AllFacts:
            Response = "Based on what I know:\n\n"
            Seen = set()
            for F in AllFacts[:5]:
                NL = F.ToNaturalLanguage()
                if NL not in Seen:
                    Response += f"• {NL}\n"
                    Seen.add(NL)
            return Response
        
        return "I'm still learning about this topic. Would you like to teach me some facts about it?"
    
    def Learn(self, Text: str) -> Dict[str, int]:
        """
        Learn from text input.
        
        Args:
            Text: Natural language text to learn from
        
        Returns:
            Dict with counts of facts and causal relations learned
        """
        Results = {
            "Facts": 0,
            "CausalRelations": 0,
        }
        
        # Extract and store knowledge
        Triples = self.Knowledge.ExtractFromText(Text)
        for T in Triples:
            if self.Knowledge.Add(T.Subject, T.Predicate, T.Object):
                Results["Facts"] += 1
        
        # Extract causal relations
        Results["CausalRelations"] = self.Causal.LearnFromText(Text)
        
        return Results
    
    def GetStats(self) -> Dict:
        """Get comprehensive statistics"""
        return {
            "Chat": {
                "TotalQuestions": self.Stats["TotalQuestions"],
                "QuestionTypes": dict(self.Stats["QuestionTypes"]),
                "ThinkingModes": dict(self.Stats["ThinkingModes"]),
                "AverageConfidence": self.Stats["AverageConfidence"],
            },
            "Knowledge": self.Knowledge.GetStats(),
            "Causal": self.Causal.GetStats(),
            "History": len(self.History),
        }
    
    def Save(self):
        """Save all data - BOTH knowledge AND causal!"""
        self.Knowledge.Save()
        self.Causal.Save()  # ← NOW SAVES CAUSAL TOO!
    
    def Close(self):
        """Close connections"""
        self.Knowledge.Close()