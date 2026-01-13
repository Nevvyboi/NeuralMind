"""
Causal Graph Module
===================

Stores and reasons about cause-effect relationships.
Unlike correlation, this is TRUE causal understanding.

Features:
- Causal relations with strength
- Causal chain discovery
- Counterfactual reasoning ("What if X happened?")
- Intervention modeling ("What if I MAKE X happen?")
- Causal explanations
"""

import re
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Dict, Set, Optional


@dataclass
class CausalRelation:
    """A causal relationship between two concepts"""
    Cause: str
    Effect: str
    Strength: float  # 0.0 to 1.0 - probability of effect given cause
    Mechanism: str = ""  # How the causation works
    Conditions: List[str] = field(default_factory=list)  # When this applies
    Evidence: List[str] = field(default_factory=list)  # Supporting observations
    
    def __hash__(self):
        return hash((self.Cause, self.Effect))
    
    def __eq__(self, other):
        if isinstance(other, CausalRelation):
            return self.Cause == other.Cause and self.Effect == other.Effect
        return False


class CausalGraph:
    """
    Causal reasoning engine.
    
    Key difference from correlation:
    - Correlation: "Rain and wet ground often appear together"
    - Causation: "Rain CAUSES wet ground with 90% probability"
    
    This enables:
    - Counterfactual reasoning: "What if it hadn't rained?"
    - Intervention: "What if I make it rain?"
    - Explanation: "Why is the ground wet?"
    """
    
    # Patterns for extracting causal relations from text
    CAUSAL_PATTERNS = [
        r'(\w+)\s+causes?\s+(\w+)',
        r'(\w+)\s+leads?\s+to\s+(\w+)',
        r'(\w+)\s+results?\s+in\s+(\w+)',
        r'because\s+(?:of\s+)?(\w+),?\s+(\w+)',
        r'(\w+)\s+makes?\s+(\w+)',
        r'if\s+(\w+)\s+then\s+(\w+)',
        r'(\w+)\s+produces?\s+(\w+)',
        r'(\w+)\s+triggers?\s+(\w+)',
        r'(\w+)\s+creates?\s+(\w+)',
        r'(\w+)\s+prevents?\s+(\w+)',
    ]
    
    def __init__(self):
        # Forward relations: cause → list of effects
        self.Relations: Dict[str, List[CausalRelation]] = defaultdict(list)
        # Reverse relations: effect → list of causes
        self.ReverseRelations: Dict[str, List[CausalRelation]] = defaultdict(list)
        
        # Statistics
        self.Stats = {
            "TotalRelations": 0,
            "ChainsFound": 0,
            "CounterfactualsComputed": 0,
        }
    
    def AddCause(self, Cause: str, Effect: str, Strength: float = 0.8,
                 Mechanism: str = "", Conditions: List[str] = None,
                 Evidence: List[str] = None) -> bool:
        """
        Add a causal relationship.
        
        Args:
            Cause: The cause (e.g., "rain")
            Effect: The effect (e.g., "wet_ground")
            Strength: Probability of effect given cause (0-1)
            Mechanism: How the causation works
            Conditions: When this relationship applies
            Evidence: Supporting observations
        
        Returns:
            True if added, False if already exists
        """
        Cause = Cause.lower().strip()
        Effect = Effect.lower().strip()
        
        # Check if already exists
        for Existing in self.Relations.get(Cause, []):
            if Existing.Effect == Effect:
                return False
        
        Relation = CausalRelation(
            Cause=Cause,
            Effect=Effect,
            Strength=Strength,
            Mechanism=Mechanism,
            Conditions=Conditions or [],
            Evidence=Evidence or []
        )
        
        self.Relations[Cause].append(Relation)
        self.ReverseRelations[Effect].append(Relation)
        self.Stats["TotalRelations"] += 1
        
        return True
    
    def GetEffects(self, Cause: str) -> List[CausalRelation]:
        """Get all effects of a cause"""
        return self.Relations.get(Cause.lower().strip(), [])
    
    def GetCauses(self, Effect: str) -> List[CausalRelation]:
        """Get all causes of an effect"""
        return self.ReverseRelations.get(Effect.lower().strip(), [])
    
    def CausalChain(self, Start: str, End: str, MaxDepth: int = 5) -> List[List[CausalRelation]]:
        """
        Find all causal chains from Start to End.
        
        Example:
            CausalChain("rain", "accident")
            → [[rain→wet, wet→slippery, slippery→accident]]
        
        Args:
            Start: Starting cause
            End: Final effect to reach
            MaxDepth: Maximum chain length
        
        Returns:
            List of causal chains (each chain is a list of relations)
        """
        Start = Start.lower().strip()
        End = End.lower().strip()
        
        Chains = []
        
        def Search(Current: str, Path: List[CausalRelation], Depth: int):
            if Depth > MaxDepth:
                return
            if Current == End and Path:
                Chains.append(Path.copy())
                self.Stats["ChainsFound"] += 1
                return
            
            for Relation in self.Relations.get(Current, []):
                if Relation not in Path:  # Avoid cycles
                    Path.append(Relation)
                    Search(Relation.Effect, Path, Depth + 1)
                    Path.pop()
        
        Search(Start, [], 0)
        return Chains
    
    def Counterfactual(self, Event: str, Intervention: bool = False) -> Dict[str, float]:
        """
        Counterfactual reasoning: What if Event happened/didn't happen?
        
        This is the KEY capability that pattern matching cannot do.
        
        Args:
            Event: The hypothetical event
            Intervention: True = "make it happen", False = "didn't happen"
        
        Returns:
            Dict mapping effects to probability changes
            Positive = more likely, Negative = less likely
        
        Example:
            Counterfactual("rain", Intervention=True)
            → {"wet_ground": 0.9, "slippery": 0.72, "accident": 0.43}
        """
        Event = Event.lower().strip()
        Effects = {}
        self.Stats["CounterfactualsComputed"] += 1
        
        def PropagateEffect(Current: str, CurrentStrength: float, Visited: Set[str]):
            if Current in Visited or CurrentStrength < 0.1:
                return
            Visited.add(Current)
            
            for Relation in self.Relations.get(Current, []):
                # Multiply strengths along the causal chain
                EffectChange = Relation.Strength * CurrentStrength
                
                if Intervention:
                    Effects[Relation.Effect] = EffectChange  # Positive change
                else:
                    Effects[Relation.Effect] = -EffectChange  # Negative change
                
                # Recursively propagate
                PropagateEffect(Relation.Effect, EffectChange, Visited)
        
        PropagateEffect(Event, 1.0, set())
        return Effects
    
    def Explain(self, Effect: str) -> str:
        """
        Generate causal explanation for an effect.
        
        Args:
            Effect: The effect to explain
        
        Returns:
            Natural language explanation
        """
        Effect = Effect.lower().strip()
        Causes = self.ReverseRelations.get(Effect, [])
        
        if not Causes:
            return f"I don't have causal knowledge about what causes {Effect}."
        
        Explanations = []
        for R in sorted(Causes, key=lambda X: -X.Strength):
            Exp = f"{R.Cause} causes {Effect}"
            if R.Mechanism:
                Exp += f" (mechanism: {R.Mechanism})"
            if R.Strength < 1.0:
                Exp += f" [{R.Strength:.0%} confidence]"
            Explanations.append(Exp)
        
        return "This happens because:\n• " + "\n• ".join(Explanations)
    
    def LearnFromText(self, Text: str) -> int:
        """
        Extract causal relationships from natural language text.
        
        Args:
            Text: Natural language text
        
        Returns:
            Number of relations extracted
        """
        Text = Text.lower()
        Count = 0
        
        for Pattern in self.CAUSAL_PATTERNS:
            for Match in re.finditer(Pattern, Text):
                Groups = Match.groups()
                if len(Groups) >= 2:
                    Cause, Effect = Groups[0], Groups[1]
                    if len(Cause) > 2 and len(Effect) > 2:
                        if self.AddCause(Cause, Effect, Strength=0.7):
                            Count += 1
        
        return Count
    
    def GetStats(self) -> Dict:
        """Get statistics"""
        return self.Stats.copy()
    
    def Size(self) -> int:
        """Get total number of relations"""
        return self.Stats["TotalRelations"]
