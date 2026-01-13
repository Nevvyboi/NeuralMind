#!/usr/bin/env python3
"""
GroundZero AI - NLP Integration Module
======================================

Integrates the state-of-the-art NLP extractor with the knowledge graph system.

This module provides:
- Automatic fact extraction from raw text
- Causal relation extraction  
- Entity linking
- Quality filtering

Author: GroundZero AI
Version: 1.0.0
"""

from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
import re

# Import NLP extractor
try:
    from .nlp_extractor import NLPProcessor, ExtractionResult, Triple, CausalRelation
    NLP_AVAILABLE = True
except ImportError:
    try:
        from nlp_extractor import NLPProcessor, ExtractionResult, Triple, CausalRelation
        NLP_AVAILABLE = True
    except ImportError:
        NLP_AVAILABLE = False
        print("⚠️ NLP extractor not found, using basic extraction")


@dataclass
class LearningResult:
    """Result of learning from text"""
    FactsAdded: int = 0
    CausalAdded: int = 0
    EntitiesFound: int = 0
    FactsSkipped: int = 0  # Duplicates
    ProcessingTimeMs: int = 0
    Source: str = ""


class SmartTextLearner:
    """
    Integrates NLP extraction with knowledge storage
    
    Features:
    - Uses spaCy when available for high-quality extraction
    - Falls back to regex patterns when spaCy unavailable
    - Deduplicates facts
    - Filters low-quality extractions
    - Tracks extraction quality metrics
    """
    
    # Minimum confidence for facts
    MIN_FACT_CONFIDENCE = 0.5
    
    # Minimum entity length
    MIN_ENTITY_LENGTH = 2
    
    # Maximum entity length (words)
    MAX_ENTITY_WORDS = 6
    
    # Blacklist of low-quality predicates
    PREDICATE_BLACKLIST = {
        'is', 'are', 'was', 'were', 'be', 'been',
        'do', 'does', 'did', 'done',
        'have', 'has', 'had',
        'get', 'got', 'gets',
        'go', 'goes', 'went', 'gone',
        'come', 'comes', 'came',
        'say', 'says', 'said',
        'make', 'makes', 'made',
        'take', 'takes', 'took', 'taken',
        'see', 'sees', 'saw', 'seen',
        'know', 'knows', 'knew', 'known',
        'think', 'thinks', 'thought',
        'want', 'wants', 'wanted',
        'use', 'uses', 'used',
        'find', 'finds', 'found',
        'give', 'gives', 'gave', 'given',
        'tell', 'tells', 'told',
        'may', 'might', 'can', 'could', 'would', 'should',
        'must', 'shall', 'will',
    }
    
    # Good predicates that indicate real relationships
    GOOD_PREDICATES = {
        'is_a', 'is_an', 'are_a', 'are_an',
        'is_type_of', 'is_kind_of', 'is_part_of',
        'has', 'has_a', 'has_an', 'have', 'have_a',
        'contains', 'includes', 'consists_of', 'comprises',
        'belongs_to', 'part_of', 'member_of',
        'located_in', 'lives_in', 'works_at', 'born_in',
        'created_by', 'invented_by', 'discovered_by', 'founded_by',
        'made_of', 'composed_of', 'built_from',
        'used_for', 'used_by', 'used_in',
        'causes', 'leads_to', 'results_in', 'produces',
        'enables', 'allows', 'prevents', 'requires',
        'connects', 'links', 'relates_to', 'associated_with',
        'similar_to', 'different_from', 'opposite_of',
        'larger_than', 'smaller_than', 'faster_than', 'slower_than',
        'before', 'after', 'during', 'while',
        'above', 'below', 'inside', 'outside', 'near', 'far_from',
        'capital_of', 'president_of', 'ceo_of', 'founder_of',
        'developed', 'developed_by', 'written_by', 'directed_by',
        'studies', 'researches', 'investigates', 'examines',
        'defines', 'describes', 'explains', 'represents',
    }
    
    def __init__(self, KnowledgeGraph=None, CausalGraph=None, UseSpacy: bool = True):
        """
        Initialize the smart learner
        
        Args:
            KnowledgeGraph: Knowledge graph instance to add facts to
            CausalGraph: Causal graph instance to add relations to
            UseSpacy: Whether to use spaCy for extraction
        """
        self.KG = KnowledgeGraph
        self.CG = CausalGraph
        self.UseSpacy = UseSpacy
        
        # Initialize NLP processor
        self.NLP = None
        if NLP_AVAILABLE and UseSpacy:
            try:
                self.NLP = NLPProcessor(UseSpacy=True)
                print("✓ Advanced NLP extraction enabled (spaCy)")
            except Exception as E:
                print(f"⚠️ Could not initialize NLP: {E}")
        
        if not self.NLP:
            print("ℹ️ Using basic regex extraction")
        
        # Track seen facts to avoid duplicates
        self.SeenFacts: Set[Tuple[str, str, str]] = set()
        self.SeenCausal: Set[Tuple[str, str]] = set()
        
        # Quality metrics
        self.TotalExtracted = 0
        self.TotalFiltered = 0
        self.TotalAdded = 0
    
    def Learn(self, Text: str, Source: str = "") -> LearningResult:
        """
        Learn facts and causal relations from text
        
        Args:
            Text: Input text to learn from
            Source: Source identifier (e.g., "Wikipedia:Physics")
            
        Returns:
            LearningResult with statistics
        """
        import time
        StartTime = time.time()
        
        Result = LearningResult(Source=Source)
        
        if self.NLP and self.NLP.IsSpacyAvailable():
            # Use advanced NLP extraction
            Extraction = self.NLP.ExtractAll(Text)
            Result.EntitiesFound = len(Extraction.Entities)
            
            # Process triples
            for Triple_ in Extraction.Triples:
                if self._IsQualityFact(Triple_):
                    Added = self._AddFact(Triple_.Subject, Triple_.Predicate, Triple_.Object)
                    if Added:
                        Result.FactsAdded += 1
                    else:
                        Result.FactsSkipped += 1
            
            # Process causal relations
            for Causal in Extraction.CausalRelations:
                if self._IsQualityCausal(Causal):
                    Added = self._AddCausal(Causal.Cause, Causal.Effect, Causal.Strength)
                    if Added:
                        Result.CausalAdded += 1
        else:
            # Use basic extraction
            Facts = self._BasicExtractFacts(Text)
            Causals = self._BasicExtractCausal(Text)
            
            for Subj, Pred, Obj in Facts:
                Added = self._AddFact(Subj, Pred, Obj)
                if Added:
                    Result.FactsAdded += 1
                else:
                    Result.FactsSkipped += 1
            
            for Cause, Effect, Strength in Causals:
                Added = self._AddCausal(Cause, Effect, Strength)
                if Added:
                    Result.CausalAdded += 1
        
        Result.ProcessingTimeMs = int((time.time() - StartTime) * 1000)
        
        return Result
    
    def _IsQualityFact(self, Triple_: 'Triple') -> bool:
        """Check if a fact meets quality standards"""
        # Check confidence
        if Triple_.Confidence < self.MIN_FACT_CONFIDENCE:
            return False
        
        # Check entity lengths
        if len(Triple_.Subject) < self.MIN_ENTITY_LENGTH:
            return False
        if len(Triple_.Object) < self.MIN_ENTITY_LENGTH:
            return False
        
        # Check word counts
        if len(Triple_.Subject.split()) > self.MAX_ENTITY_WORDS:
            return False
        if len(Triple_.Object.split()) > self.MAX_ENTITY_WORDS:
            return False
        
        # Check predicate
        Pred = Triple_.Predicate.lower()
        
        # Skip blacklisted predicates (too generic)
        if Pred in self.PREDICATE_BLACKLIST:
            return False
        
        # Prefer good predicates
        # (Still accept others but with lower priority)
        
        # Check for numeric-only entities
        if Triple_.Subject.replace(' ', '').isdigit():
            return False
        if Triple_.Object.replace(' ', '').isdigit():
            return False
        
        # Check subject != object
        if Triple_.Subject.lower() == Triple_.Object.lower():
            return False
        
        return True
    
    def _IsQualityCausal(self, Causal: 'CausalRelation') -> bool:
        """Check if a causal relation meets quality standards"""
        # Check entity lengths
        if len(Causal.Cause) < self.MIN_ENTITY_LENGTH:
            return False
        if len(Causal.Effect) < self.MIN_ENTITY_LENGTH:
            return False
        
        # Check word counts
        if len(Causal.Cause.split()) > self.MAX_ENTITY_WORDS:
            return False
        if len(Causal.Effect.split()) > self.MAX_ENTITY_WORDS:
            return False
        
        # Check cause != effect
        if Causal.Cause.lower() == Causal.Effect.lower():
            return False
        
        return True
    
    def _AddFact(self, Subject: str, Predicate: str, Object: str) -> bool:
        """Add a fact to the knowledge graph if not duplicate"""
        # Normalize
        Subject = Subject.strip().lower()
        Predicate = Predicate.strip().lower().replace(' ', '_')
        Object = Object.strip().lower()
        
        # Check for duplicate
        Key = (Subject, Predicate, Object)
        if Key in self.SeenFacts:
            return False
        
        self.SeenFacts.add(Key)
        
        # Add to knowledge graph
        if self.KG:
            self.KG.Add(Subject, Predicate, Object)
        
        self.TotalAdded += 1
        return True
    
    def _AddCausal(self, Cause: str, Effect: str, Strength: float) -> bool:
        """Add a causal relation if not duplicate"""
        # Normalize
        Cause = Cause.strip().lower()
        Effect = Effect.strip().lower()
        
        # Check for duplicate
        Key = (Cause, Effect)
        if Key in self.SeenCausal:
            return False
        
        self.SeenCausal.add(Key)
        
        # Add to causal graph
        if self.CG:
            self.CG.AddCause(Cause, Effect, Strength=Strength)
        
        return True
    
    # =========================================================================
    # BASIC EXTRACTION (Fallback)
    # =========================================================================
    
    def _BasicExtractFacts(self, Text: str) -> List[Tuple[str, str, str]]:
        """Basic regex-based fact extraction"""
        Facts = []
        
        Patterns = [
            # X is a/an Y
            (r'([A-Z][a-z]+(?:\s+[a-z]+)*)\s+is\s+(?:a|an)\s+([a-z]+(?:\s+[a-z]+)?)', 'is_a'),
            # X is the Y of Z
            (r'([A-Z][a-z]+(?:\s+[a-z]+)*)\s+is\s+the\s+([a-z]+)\s+of\s+([A-Z][a-z]+)', 'is_the'),
            # X are Y
            (r'([A-Z][a-z]+s?)\s+are\s+([a-z]+(?:\s+[a-z]+)?)', 'are'),
            # X has Y
            (r'([A-Z][a-z]+(?:\s+[a-z]+)*)\s+(?:has|have)\s+(?:a|an|the)?\s*([a-z]+(?:\s+[a-z]+)?)', 'has'),
            # X contains Y
            (r'([A-Z][a-z]+(?:\s+[a-z]+)*)\s+contains?\s+([a-z]+(?:\s+[a-z]+)?)', 'contains'),
            # X consists of Y
            (r'([A-Z][a-z]+(?:\s+[a-z]+)*)\s+consists?\s+of\s+([a-z]+(?:\s+[a-z]+)?)', 'consists_of'),
            # X is located in Y
            (r'([A-Z][a-z]+(?:\s+[a-z]+)*)\s+is\s+located\s+in\s+([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)', 'located_in'),
            # X was founded in Y
            (r'([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)\s+was\s+founded\s+in\s+(\d{4})', 'founded_in'),
            # X invented Y
            (r'([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)\s+invented\s+(?:the\s+)?([a-z]+(?:\s+[a-z]+)?)', 'invented'),
            # X discovered Y
            (r'([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)\s+discovered\s+(?:the\s+)?([a-z]+(?:\s+[a-z]+)?)', 'discovered'),
        ]
        
        for Pattern, Pred in Patterns:
            for Match in re.finditer(Pattern, Text, re.IGNORECASE):
                Groups = Match.groups()
                if len(Groups) >= 2:
                    Subj = Groups[0].strip()
                    Obj = Groups[-1].strip()
                    
                    if len(Subj) >= 2 and len(Obj) >= 2:
                        Facts.append((Subj, Pred, Obj))
        
        return Facts
    
    def _BasicExtractCausal(self, Text: str) -> List[Tuple[str, str, float]]:
        """Basic regex-based causal extraction"""
        Relations = []
        
        Patterns = [
            (r'([A-Za-z]+(?:\s+[a-z]+)?)\s+causes?\s+([a-z]+(?:\s+[a-z]+)?)', 0.85),
            (r'([A-Za-z]+(?:\s+[a-z]+)?)\s+leads?\s+to\s+([a-z]+(?:\s+[a-z]+)?)', 0.80),
            (r'([A-Za-z]+(?:\s+[a-z]+)?)\s+results?\s+in\s+([a-z]+(?:\s+[a-z]+)?)', 0.80),
            (r'due\s+to\s+([a-z]+(?:\s+[a-z]+)?),?\s+([a-z]+(?:\s+[a-z]+)?)', 0.75),
            (r'because\s+of\s+([a-z]+(?:\s+[a-z]+)?),?\s+([a-z]+(?:\s+[a-z]+)?)', 0.75),
            (r'([A-Za-z]+(?:\s+[a-z]+)?)\s+(?:produces?|creates?)\s+([a-z]+(?:\s+[a-z]+)?)', 0.75),
            (r'([A-Za-z]+(?:\s+[a-z]+)?)\s+(?:triggers?|induces?)\s+([a-z]+(?:\s+[a-z]+)?)', 0.80),
        ]
        
        for Pattern, Strength in Patterns:
            for Match in re.finditer(Pattern, Text, re.IGNORECASE):
                Cause = Match.group(1).strip()
                Effect = Match.group(2).strip()
                
                if len(Cause) >= 2 and len(Effect) >= 2 and Cause.lower() != Effect.lower():
                    Relations.append((Cause, Effect, Strength))
        
        return Relations
    
    def GetStats(self) -> Dict:
        """Get extraction statistics"""
        return {
            "TotalExtracted": self.TotalExtracted,
            "TotalFiltered": self.TotalFiltered,
            "TotalAdded": self.TotalAdded,
            "UniqueFactsSeen": len(self.SeenFacts),
            "UniqueCausalSeen": len(self.SeenCausal),
            "SpacyAvailable": self.NLP.IsSpacyAvailable() if self.NLP else False
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def ExtractFactsFromText(Text: str) -> List[Tuple[str, str, str]]:
    """
    Quick function to extract facts from text
    
    Args:
        Text: Input text
        
    Returns:
        List of (subject, predicate, object) tuples
    """
    Learner = SmartTextLearner(UseSpacy=True)
    
    if Learner.NLP and Learner.NLP.IsSpacyAvailable():
        return Learner.NLP.ExtractFacts(Text)
    else:
        return Learner._BasicExtractFacts(Text)


def ExtractCausalFromText(Text: str) -> List[Tuple[str, str, float]]:
    """
    Quick function to extract causal relations from text
    
    Args:
        Text: Input text
        
    Returns:
        List of (cause, effect, strength) tuples
    """
    Learner = SmartTextLearner(UseSpacy=True)
    
    if Learner.NLP and Learner.NLP.IsSpacyAvailable():
        return Learner.NLP.ExtractCausal(Text)
    else:
        return Learner._BasicExtractCausal(Text)


# =============================================================================
# TEST
# =============================================================================

def Test():
    """Test the integration module"""
    print("\n" + "=" * 70)
    print("🧪 Testing NLP Integration")
    print("=" * 70 + "\n")
    
    Learner = SmartTextLearner(UseSpacy=True)
    
    TestTexts = [
        "The Sun is a star. Stars are massive celestial bodies. The Sun provides light and heat to Earth.",
        "Deforestation causes soil erosion. Soil erosion leads to flooding. Flooding destroys crops.",
        "Albert Einstein developed the theory of relativity. Relativity is a fundamental theory in physics.",
        "Water consists of hydrogen and oxygen. Water is essential for life. Plants need water to grow.",
    ]
    
    for i, Text in enumerate(TestTexts, 1):
        print(f"Test {i}: {Text[:60]}...")
        Result = Learner.Learn(Text, Source=f"Test_{i}")
        print(f"  → Facts: {Result.FactsAdded}, Causal: {Result.CausalAdded}")
        print()
    
    Stats = Learner.GetStats()
    print(f"Total unique facts: {Stats['UniqueFactsSeen']}")
    print(f"Total unique causal: {Stats['UniqueCausalSeen']}")
    print(f"spaCy available: {Stats['SpacyAvailable']}")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    Test()