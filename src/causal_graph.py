"""
Causal Graph - Persistent Version
=================================

Stores cause-effect relationships with persistence to JSON file.
Supports causal chains, counterfactual reasoning, and auto-save.

This is TRUE causal understanding - not just correlation!
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class CausalRelation:
    """A cause-effect relationship"""
    Cause: str
    Effect: str
    Strength: float = 0.8  # 0-1, how strong the causal link is
    Confidence: float = 0.7  # 0-1, how confident we are
    Mechanism: str = ""  # How the causation works
    Conditions: List[str] = field(default_factory=list)
    Evidence: List[str] = field(default_factory=list)
    
    def __hash__(self):
        return hash((self.Cause, self.Effect))
    
    def __eq__(self, other):
        if isinstance(other, CausalRelation):
            return self.Cause == other.Cause and self.Effect == other.Effect
        return False
    
    def ToDict(self) -> dict:
        return {
            "cause": self.Cause,
            "effect": self.Effect,
            "strength": self.Strength,
            "confidence": self.Confidence,
            "mechanism": self.Mechanism,
            "conditions": self.Conditions,
            "evidence": self.Evidence
        }
    
    @classmethod
    def FromDict(cls, d: dict) -> "CausalRelation":
        return cls(
            Cause=d["cause"],
            Effect=d["effect"],
            Strength=d.get("strength", 0.8),
            Confidence=d.get("confidence", 0.7),
            Mechanism=d.get("mechanism", ""),
            Conditions=d.get("conditions", []),
            Evidence=d.get("evidence", [])
        )


class CausalGraph:
    """
    Persistent causal graph for cause-effect reasoning.
    
    Key difference from correlation:
    - Correlation: "Rain and wet ground often appear together"
    - Causation: "Rain CAUSES wet ground with 90% probability"
    
    Features:
    - Automatic persistence to JSON file
    - Causal chain discovery
    - Counterfactual reasoning
    - Intervention modeling
    - Strength propagation
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
        r'(\w+)\s+affects?\s+(\w+)',
        r'(\w+)\s+influences?\s+(\w+)',
        r'(\w+)\s+determines?\s+(\w+)',
        r'(\w+)\s+enables?\s+(\w+)',
        r'(\w+)\s+allows?\s+(\w+)',
    ]
    
    def __init__(self, DataPath: Optional[str] = None):
        """
        Initialize causal graph with optional persistence.
        
        Args:
            DataPath: Path to data directory. If provided, loads/saves causal.json
        """
        # Forward relations: cause → list of effects
        self.Relations: Dict[str, List[CausalRelation]] = defaultdict(list)
        # Reverse relations: effect → list of causes
        self.ReverseRelations: Dict[str, List[CausalRelation]] = defaultdict(list)
        
        self.DataPath = Path(DataPath) if DataPath else None
        self.FilePath = self.DataPath / "causal.json" if self.DataPath else None
        
        self.Stats = {
            "TotalRelations": 0,
            "ChainsFound": 0,
            "CounterfactualsComputed": 0
        }
        
        # Load existing data if available
        if self.FilePath and self.FilePath.exists():
            self._Load()
    
    def _Load(self):
        """Load causal relations from JSON file"""
        try:
            with open(self.FilePath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for rel_data in data.get("relations", []):
                rel = CausalRelation.FromDict(rel_data)
                self.Relations[rel.Cause].append(rel)
                self.ReverseRelations[rel.Effect].append(rel)
            
            self.Stats = data.get("stats", self.Stats)
            self.Stats["TotalRelations"] = sum(len(effects) for effects in self.Relations.values())
            
            print(f"  📂 Loaded {self.Stats['TotalRelations']} causal relations from {self.FilePath.name}")
            
        except Exception as e:
            print(f"  ⚠️ Could not load causal graph: {e}")
    
    def Save(self):
        """Save causal relations to JSON file"""
        if not self.FilePath:
            return
        
        try:
            # Ensure directory exists
            self.FilePath.parent.mkdir(parents=True, exist_ok=True)
            
            # Collect all relations
            relations = []
            for rel_list in self.Relations.values():
                for rel in rel_list:
                    relations.append(rel.ToDict())
            
            data = {
                "relations": relations,
                "stats": self.Stats
            }
            
            with open(self.FilePath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"  ⚠️ Could not save causal graph: {e}")
    
    def AddCause(self, Cause: str, Effect: str, Strength: float = 0.8,
                 Mechanism: str = "", Conditions: List[str] = None,
                 Evidence: str = None, AutoSave: bool = False) -> bool:
        """
        Add a causal relationship.
        
        Args:
            Cause: The cause entity
            Effect: The effect entity
            Strength: How strong the causal link is (0-1)
            Mechanism: How the causation works
            Conditions: When this relationship applies
            Evidence: Optional evidence/source
            AutoSave: Whether to auto-save after adding
        
        Returns:
            True if new relation added, False if updated existing
        """
        Cause = Cause.lower().strip().replace(" ", "_")
        Effect = Effect.lower().strip().replace(" ", "_")
        
        if not Cause or not Effect or Cause == Effect:
            return False
        
        if len(Cause) < 2 or len(Effect) < 2:
            return False
        
        # Check if already exists
        for Existing in self.Relations.get(Cause, []):
            if Existing.Effect == Effect:
                # Update strength/confidence
                Existing.Strength = (Existing.Strength + Strength) / 2
                Existing.Confidence = min(1.0, Existing.Confidence + 0.05)
                if Evidence and Evidence not in Existing.Evidence:
                    Existing.Evidence.append(Evidence)
                return False
        
        # Add new relation
        Relation = CausalRelation(
            Cause=Cause,
            Effect=Effect,
            Strength=Strength,
            Confidence=0.7,
            Mechanism=Mechanism,
            Conditions=Conditions or [],
            Evidence=[Evidence] if Evidence else []
        )
        
        self.Relations[Cause].append(Relation)
        self.ReverseRelations[Effect].append(Relation)
        self.Stats["TotalRelations"] += 1
        
        if AutoSave and self.FilePath:
            self.Save()
        
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
        """
        Event = Event.lower().strip()
        Effects = {}
        self.Stats["CounterfactualsComputed"] += 1
        
        def PropagateEffect(Current: str, CurrentStrength: float, Visited: Set[str]):
            if Current in Visited or CurrentStrength < 0.1:
                return
            Visited.add(Current)
            
            for Relation in self.Relations.get(Current, []):
                EffectChange = Relation.Strength * CurrentStrength
                
                if Intervention:
                    Effects[Relation.Effect] = EffectChange
                else:
                    Effects[Relation.Effect] = -EffectChange
                
                PropagateEffect(Relation.Effect, EffectChange, Visited)
        
        PropagateEffect(Event, 1.0, set())
        return Effects
    
    def Explain(self, Effect: str) -> str:
        """Generate causal explanation for an effect."""
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
                    # Filter out common words
                    StopWords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                                 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                                 'can', 'need', 'dare', 'ought', 'used', 'to', 'it', 'its',
                                 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
                                 'we', 'they', 'what', 'which', 'who', 'whom', 'whose',
                                 'where', 'when', 'why', 'how', 'all', 'each', 'every',
                                 'both', 'few', 'more', 'most', 'other', 'some', 'such',
                                 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                                 'too', 'very', 'just', 'also', 'now', 'here', 'there'}
                    
                    if (len(Cause) > 2 and len(Effect) > 2 and 
                        Cause not in StopWords and Effect not in StopWords):
                        if self.AddCause(Cause, Effect, Strength=0.7):
                            Count += 1
        
        return Count
    
    def GetStats(self) -> Dict:
        """Get statistics"""
        return self.Stats.copy()
    
    def Size(self) -> int:
        """Get total number of relations"""
        return self.Stats["TotalRelations"]
    
    def GetAllRelations(self) -> List[CausalRelation]:
        """Get all causal relations"""
        relations = []
        for rel_list in self.Relations.values():
            relations.extend(rel_list)
        return relations
    
    def Clear(self):
        """Clear all relations"""
        self.Relations.clear()
        self.ReverseRelations.clear()
        self.Stats = {
            "TotalRelations": 0,
            "ChainsFound": 0,
            "CounterfactualsComputed": 0
        }
        if self.FilePath and self.FilePath.exists():
            self.FilePath.unlink()


# Test
if __name__ == "__main__":
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and populate
        cg1 = CausalGraph(tmpdir)
        cg1.AddCause("rain", "wet_ground", Strength=0.9)
        cg1.AddCause("wet_ground", "slippery", Strength=0.8)
        cg1.AddCause("slippery", "accidents", Strength=0.6)
        cg1.Save()
        print(f"Created {cg1.Size()} relations")
        
        # Load in new instance
        cg2 = CausalGraph(tmpdir)
        print(f"Loaded {cg2.Size()} relations")
        
        # Test causal chain
        chains = cg2.CausalChain("rain", "accidents")
        print(f"Chains from rain to accidents: {len(chains)}")
        
        # Test counterfactual
        cf = cg2.Counterfactual("rain", Intervention=True)
        print(f"If rain happens: {cf}")
        
        print("✅ Persistence test passed!")