"""
Knowledge Graph Module
======================

Stores facts as explicit (subject, predicate, object) triples.
Unlike neural networks where knowledge is hidden in weights,
every fact is inspectable and queryable.

Features:
- Add/query facts
- Transitive inference (A→B, B→C, therefore A→C)
- Multi-hop reasoning
- Natural language extraction
- SQLite persistence
"""

import re
import time
import sqlite3
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Set


@dataclass
class KnowledgeTriple:
    """A fact represented as (subject, predicate, object)"""
    Subject: str
    Predicate: str
    Object: str
    Confidence: float = 1.0
    Source: str = "learned"
    Timestamp: float = field(default_factory=time.time)
    
    def ToTuple(self) -> Tuple[str, str, str]:
        return (self.Subject, self.Predicate, self.Object)
    
    def ToNaturalLanguage(self) -> str:
        """Convert triple to readable sentence"""
        PredicateMap = {
            "is_a": "is a",
            "has_property": "has the property",
            "has": "has",
            "part_of": "is part of",
            "causes": "causes",
            "located_in": "is located in",
            "created_by": "was created by",
            "synonym": "is also known as",
            "antonym": "is the opposite of",
            "related_to": "is related to",
            "contains": "contains",
            "needs": "needs",
            "produces": "produces",
        }
        Pred = PredicateMap.get(self.Predicate, self.Predicate.replace("_", " "))
        return f"{self.Subject} {Pred} {self.Object}"
    
    def __hash__(self):
        return hash(self.ToTuple())
    
    def __eq__(self, other):
        if isinstance(other, KnowledgeTriple):
            return self.ToTuple() == other.ToTuple()
        return False


class KnowledgeGraph:
    """
    Knowledge Graph with explicit fact storage.
    
    Unlike neural networks:
    - Facts are explicit, not hidden in weights
    - Every fact can be inspected and queried
    - Reasoning is transparent and verifiable
    """
    
    # Patterns for extracting facts from natural language
    EXTRACTION_PATTERNS = [
        (r'(\w+)\s+is\s+(?:a|an)\s+(\w+)', "is_a"),
        (r'(\w+)\s+are\s+(\w+)', "is_a"),
        (r'(\w+)\s+has\s+(?:a|an)?\s*(\w+)', "has"),
        (r'(\w+)\s+contains?\s+(\w+)', "contains"),
        (r'(\w+)\s+causes?\s+(\w+)', "causes"),
        (r'(\w+)\s+leads?\s+to\s+(\w+)', "causes"),
        (r'(\w+)\s+is\s+(?:located\s+)?in\s+(\w+)', "located_in"),
        (r'(\w+)\s+was\s+created\s+by\s+(\w+)', "created_by"),
        (r'(\w+)\s+invented\s+(\w+)', "created_by"),
        (r'(\w+)\s+is\s+part\s+of\s+(\w+)', "part_of"),
        (r'(\w+)\s+belongs?\s+to\s+(\w+)', "part_of"),
        (r'(\w+)\s+needs?\s+(\w+)', "needs"),
        (r'(\w+)\s+produces?\s+(\w+)', "produces"),
    ]
    
    def __init__(self, DbPath: Optional[str] = None):
        """
        Initialize Knowledge Graph.
        
        Args:
            DbPath: Path to SQLite database. If None, uses in-memory only.
        """
        self.DbPath = DbPath
        self.Db = None
        
        # In-memory indices for fast lookup
        self.BySubject: Dict[str, List[KnowledgeTriple]] = defaultdict(list)
        self.ByPredicate: Dict[str, List[KnowledgeTriple]] = defaultdict(list)
        self.ByObject: Dict[str, List[KnowledgeTriple]] = defaultdict(list)
        self.AllTriples: Set[Tuple[str, str, str]] = set()
        
        # Statistics
        self.Stats = {
            "TotalFacts": 0,
            "FactsAdded": 0,
            "FactsInferred": 0,
            "QueriesAnswered": 0,
        }
        
        if DbPath:
            self._InitDatabase()
            self._LoadFromDatabase()
    
    def _InitDatabase(self):
        """Initialize SQLite database"""
        Path(self.DbPath).parent.mkdir(parents=True, exist_ok=True)
        self.Db = sqlite3.connect(self.DbPath)
        self.Db.execute('''
            CREATE TABLE IF NOT EXISTS Triples (
                Id INTEGER PRIMARY KEY AUTOINCREMENT,
                Subject TEXT NOT NULL,
                Predicate TEXT NOT NULL,
                Object TEXT NOT NULL,
                Confidence REAL DEFAULT 1.0,
                Source TEXT DEFAULT 'learned',
                Timestamp REAL,
                UNIQUE(Subject, Predicate, Object)
            )
        ''')
        self.Db.execute('CREATE INDEX IF NOT EXISTS IdxSubject ON Triples(Subject)')
        self.Db.execute('CREATE INDEX IF NOT EXISTS IdxPredicate ON Triples(Predicate)')
        self.Db.execute('CREATE INDEX IF NOT EXISTS IdxObject ON Triples(Object)')
        self.Db.commit()
    
    def _LoadFromDatabase(self):
        """Load existing triples from database"""
        if not self.Db:
            return
        
        Cursor = self.Db.execute(
            'SELECT Subject, Predicate, Object, Confidence, Source, Timestamp FROM Triples'
        )
        for Row in Cursor:
            Triple = KnowledgeTriple(*Row)
            self._AddToIndices(Triple)
        
        self.Stats["TotalFacts"] = len(self.AllTriples)
    
    def _AddToIndices(self, Triple: KnowledgeTriple) -> bool:
        """Add triple to in-memory indices"""
        Key = Triple.ToTuple()
        if Key in self.AllTriples:
            return False
        
        self.AllTriples.add(Key)
        self.BySubject[Triple.Subject].append(Triple)
        self.ByPredicate[Triple.Predicate].append(Triple)
        self.ByObject[Triple.Object].append(Triple)
        return True
    
    def Add(self, Subject: str, Predicate: str, Object: str,
            Confidence: float = 1.0, Source: str = "learned") -> bool:
        """
        Add a new fact to the knowledge graph.
        
        Args:
            Subject: The subject of the triple (e.g., "dog")
            Predicate: The relationship (e.g., "is_a")
            Object: The object (e.g., "animal")
            Confidence: Confidence score 0-1
            Source: Where this fact came from
        
        Returns:
            True if fact was added, False if already exists
        """
        Triple = KnowledgeTriple(
            Subject=Subject.lower().strip(),
            Predicate=Predicate.lower().strip(),
            Object=Object.lower().strip(),
            Confidence=Confidence,
            Source=Source
        )
        
        if not self._AddToIndices(Triple):
            return False
        
        self.Stats["FactsAdded"] += 1
        self.Stats["TotalFacts"] += 1
        
        # Persist to database
        if self.Db:
            try:
                self.Db.execute(
                    'INSERT OR IGNORE INTO Triples VALUES (NULL,?,?,?,?,?,?)',
                    (Triple.Subject, Triple.Predicate, Triple.Object,
                     Triple.Confidence, Triple.Source, Triple.Timestamp)
                )
                self.Db.commit()
            except Exception:
                pass
        
        return True
    
    def Query(self, Subject: Optional[str] = None,
              Predicate: Optional[str] = None,
              Object: Optional[str] = None) -> List[KnowledgeTriple]:
        """
        Query the knowledge graph.
        
        Args:
            Subject: Filter by subject (optional)
            Predicate: Filter by predicate (optional)
            Object: Filter by object (optional)
        
        Returns:
            List of matching triples
        """
        self.Stats["QueriesAnswered"] += 1
        Results = None
        
        if Subject:
            Subject = Subject.lower().strip()
            Results = set(T.ToTuple() for T in self.BySubject.get(Subject, []))
        
        if Predicate:
            Predicate = Predicate.lower().strip()
            PredResults = set(T.ToTuple() for T in self.ByPredicate.get(Predicate, []))
            Results = PredResults if Results is None else Results & PredResults
        
        if Object:
            Object = Object.lower().strip()
            ObjResults = set(T.ToTuple() for T in self.ByObject.get(Object, []))
            Results = ObjResults if Results is None else Results & ObjResults
        
        if Results is None:
            return []
        
        # Convert back to triples
        return [
            next(T for T in self.BySubject[S] if T.ToTuple() == (S, P, O))
            for S, P, O in Results
        ]
    
    def GetRelated(self, Entity: str, MaxDepth: int = 2) -> List[KnowledgeTriple]:
        """
        Get all facts related to an entity within N hops.
        
        Args:
            Entity: The entity to search from
            MaxDepth: Maximum number of hops
        
        Returns:
            List of related triples
        """
        Entity = Entity.lower().strip()
        Visited = set()
        Results = []
        Queue = [(Entity, 0)]
        
        while Queue:
            Current, Depth = Queue.pop(0)
            if Current in Visited or Depth > MaxDepth:
                continue
            Visited.add(Current)
            
            # Facts where entity is subject
            for Triple in self.BySubject.get(Current, []):
                Results.append(Triple)
                if Depth < MaxDepth:
                    Queue.append((Triple.Object, Depth + 1))
            
            # Facts where entity is object
            for Triple in self.ByObject.get(Current, []):
                Results.append(Triple)
                if Depth < MaxDepth:
                    Queue.append((Triple.Subject, Depth + 1))
        
        return Results
    
    def InferTransitive(self, Predicate: str) -> List[KnowledgeTriple]:
        """
        Infer new facts through transitive reasoning.
        
        If A is_a B and B is_a C, then A is_a C.
        
        Args:
            Predicate: The predicate to apply transitivity to
        
        Returns:
            List of newly inferred triples
        """
        Predicate = Predicate.lower().strip()
        Existing = self.ByPredicate.get(Predicate, [])
        
        # Build graph: subject → set of objects
        Graph = defaultdict(set)
        for Triple in Existing:
            Graph[Triple.Subject].add(Triple.Object)
        
        Inferred = []
        for Start in Graph:
            Visited = set()
            Queue = [Start]
            
            while Queue:
                Current = Queue.pop(0)
                for Next in Graph.get(Current, []):
                    if Next not in Visited:
                        Visited.add(Next)
                        Queue.append(Next)
                        
                        # Check if this is a new inference
                        if Next != Start and (Start, Predicate, Next) not in self.AllTriples:
                            NewTriple = KnowledgeTriple(
                                Subject=Start,
                                Predicate=Predicate,
                                Object=Next,
                                Confidence=0.9,
                                Source="inferred_transitive"
                            )
                            Inferred.append(NewTriple)
                            self.Stats["FactsInferred"] += 1
        
        return Inferred
    
    def ExtractFromText(self, Text: str) -> List[KnowledgeTriple]:
        """
        Extract knowledge triples from natural language text.
        
        Args:
            Text: Natural language text
        
        Returns:
            List of extracted triples
        """
        Extracted = []
        Text = Text.lower()
        
        for Pattern, Predicate in self.EXTRACTION_PATTERNS:
            for Match in re.finditer(Pattern, Text):
                Subject, Object = Match.groups()
                if len(Subject) > 2 and len(Object) > 2:
                    Extracted.append(KnowledgeTriple(
                        Subject=Subject,
                        Predicate=Predicate,
                        Object=Object,
                        Source="extracted"
                    ))
        
        return Extracted
    
    def Size(self) -> int:
        """Get total number of facts"""
        return len(self.AllTriples)
    
    def GetStats(self) -> Dict:
        """Get statistics"""
        return self.Stats.copy()
    
    def Save(self):
        """Save to database"""
        if self.Db:
            self.Db.commit()
    
    def Close(self):
        """Close database connection"""
        if self.Db:
            self.Db.close()
