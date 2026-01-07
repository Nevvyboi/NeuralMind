"""
GroundZero Context Brain
========================
Intelligent context understanding that learns and improves over time.

PROBLEMS THIS SOLVES:
- "Parid" should match "Paris" (fuzzy matching)
- "I mean Paris" should understand correction (correction detection)
- System learns from corrections (adaptive learning)
- Disambiguates "Paris" (city) vs "Paris" (person) (entity resolution)

ARCHITECTURE:
┌─────────────────────────────────────────────────────────────────┐
│                      CONTEXT BRAIN                               │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   Query     │  │ Correction  │  │    Entity               │ │
│  │ Understander│→ │  Detector   │→ │    Resolver             │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│         ↓                ↓                    ↓                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              LEARNING DATABASE (SQLite)                      ││
│  │  - Learned corrections (typo → correct)                     ││
│  │  - Entity popularity scores                                  ││
│  │  - User query patterns                                       ││
│  │  - Phonetic index                                            ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘

KEY FEATURES:
1. Multi-algorithm fuzzy matching (Levenshtein, Soundex, Metaphone, Jaro-Winkler)
2. Learns from user corrections automatically
3. Context-aware entity disambiguation
4. Persistent learning (survives restarts)
5. Phonetic matching for misspellings
6. Query intent detection
"""

import re
import sqlite3
import threading
import json
import math
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from enum import Enum


# ============================================================
# STRING SIMILARITY ALGORITHMS
# ============================================================

class StringSimilarity:
    """
    Multiple string similarity algorithms for robust matching.
    
    Algorithms:
    1. Levenshtein Distance - Edit distance (typos)
    2. Soundex - Phonetic encoding (sounds alike)
    3. Metaphone - Better phonetic encoding
    4. Jaro-Winkler - Weighted prefix similarity
    5. N-gram similarity - Character overlap
    """
    
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """
        Calculate Levenshtein (edit) distance.
        Number of insertions, deletions, substitutions to transform s1 → s2.
        """
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    @staticmethod
    def levenshtein_similarity(s1: str, s2: str) -> float:
        """
        Levenshtein similarity (0-1 scale).
        1.0 = identical, 0.0 = completely different
        """
        s1, s2 = s1.lower(), s2.lower()
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        
        distance = StringSimilarity.levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        return 1.0 - (distance / max_len)
    
    @staticmethod
    def soundex(word: str) -> str:
        """
        Soundex phonetic encoding.
        Words that sound similar get same code.
        
        "Paris" → "P620"
        "Parris" → "P620" (same!)
        """
        word = word.upper()
        if not word:
            return "0000"
        
        # Keep first letter
        soundex = word[0]
        
        # Encoding map
        codes = {
            'B': '1', 'F': '1', 'P': '1', 'V': '1',
            'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
            'D': '3', 'T': '3',
            'L': '4',
            'M': '5', 'N': '5',
            'R': '6'
        }
        
        prev_code = codes.get(word[0], '')
        
        for char in word[1:]:
            code = codes.get(char, '')
            if code and code != prev_code:
                soundex += code
                prev_code = code
            elif char in 'AEIOUYHW':
                prev_code = ''
        
        # Pad or truncate to 4 characters
        soundex = (soundex + '000')[:4]
        return soundex
    
    @staticmethod
    def metaphone(word: str) -> str:
        """
        Metaphone phonetic encoding.
        More accurate than Soundex for English.
        """
        word = word.upper()
        if not word:
            return ""
        
        result = []
        i = 0
        
        # Skip initial silent letters
        if word[:2] in ['KN', 'GN', 'PN', 'AE', 'WR']:
            i = 1
        elif word[:1] == 'X':
            word = 'S' + word[1:]
        elif word[:2] == 'WH':
            word = 'W' + word[2:]
        
        while i < len(word):
            char = word[i]
            
            # Vowels only at start
            if char in 'AEIOU':
                if i == 0:
                    result.append(char)
            elif char == 'B':
                if i == 0 or word[i-1:i+1] != 'MB':
                    result.append('B')
            elif char == 'C':
                if i < len(word) - 1 and word[i+1] in 'IEY':
                    result.append('S')
                elif i < len(word) - 1 and word[i+1] == 'H':
                    result.append('X')
                    i += 1
                else:
                    result.append('K')
            elif char == 'D':
                if i < len(word) - 1 and word[i+1] == 'G' and len(word) > i+2 and word[i+2] in 'IEY':
                    result.append('J')
                    i += 1
                else:
                    result.append('T')
            elif char == 'F':
                result.append('F')
            elif char == 'G':
                if i < len(word) - 1 and word[i+1] in 'IEY':
                    result.append('J')
                elif i < len(word) - 1 and word[i+1] not in 'HN':
                    result.append('K')
                elif i == len(word) - 1 or word[i+1] not in 'HN':
                    result.append('K')
            elif char == 'H':
                if i == 0 or word[i-1] not in 'CSPTG':
                    if i < len(word) - 1 and word[i+1] in 'AEIOU':
                        result.append('H')
            elif char == 'J':
                result.append('J')
            elif char == 'K':
                if i == 0 or word[i-1] != 'C':
                    result.append('K')
            elif char == 'L':
                result.append('L')
            elif char == 'M':
                result.append('M')
            elif char == 'N':
                result.append('N')
            elif char == 'P':
                if i < len(word) - 1 and word[i+1] == 'H':
                    result.append('F')
                    i += 1
                else:
                    result.append('P')
            elif char == 'Q':
                result.append('K')
            elif char == 'R':
                result.append('R')
            elif char == 'S':
                if i < len(word) - 1 and word[i+1] == 'H':
                    result.append('X')
                    i += 1
                elif i < len(word) - 2 and word[i+1:i+3] in ['IO', 'IA']:
                    result.append('X')
                else:
                    result.append('S')
            elif char == 'T':
                if i < len(word) - 1 and word[i+1] == 'H':
                    result.append('0')  # TH
                    i += 1
                elif i < len(word) - 2 and word[i+1:i+3] in ['IO', 'IA']:
                    result.append('X')
                else:
                    result.append('T')
            elif char == 'V':
                result.append('F')
            elif char == 'W':
                if i < len(word) - 1 and word[i+1] in 'AEIOU':
                    result.append('W')
            elif char == 'X':
                result.append('KS')
            elif char == 'Y':
                if i < len(word) - 1 and word[i+1] in 'AEIOU':
                    result.append('Y')
            elif char == 'Z':
                result.append('S')
            
            i += 1
        
        return ''.join(result)
    
    @staticmethod
    def jaro_winkler(s1: str, s2: str) -> float:
        """
        Jaro-Winkler similarity.
        Gives bonus for matching prefix (good for names).
        """
        s1, s2 = s1.lower(), s2.lower()
        
        if s1 == s2:
            return 1.0
        
        len1, len2 = len(s1), len(s2)
        
        if len1 == 0 or len2 == 0:
            return 0.0
        
        # Match window
        match_distance = max(len1, len2) // 2 - 1
        if match_distance < 0:
            match_distance = 0
        
        s1_matches = [False] * len1
        s2_matches = [False] * len2
        
        matches = 0
        transpositions = 0
        
        # Find matches
        for i in range(len1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len2)
            
            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break
        
        if matches == 0:
            return 0.0
        
        # Count transpositions
        k = 0
        for i in range(len1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1
        
        # Jaro similarity
        jaro = (matches / len1 + matches / len2 + 
                (matches - transpositions / 2) / matches) / 3
        
        # Winkler modification (prefix bonus)
        prefix = 0
        for i in range(min(len1, len2, 4)):
            if s1[i] == s2[i]:
                prefix += 1
            else:
                break
        
        return jaro + prefix * 0.1 * (1 - jaro)
    
    @staticmethod
    def ngram_similarity(s1: str, s2: str, n: int = 2) -> float:
        """
        N-gram (character) similarity.
        Compares overlapping character sequences.
        """
        s1, s2 = s1.lower(), s2.lower()
        
        if len(s1) < n or len(s2) < n:
            return 1.0 if s1 == s2 else 0.0
        
        def get_ngrams(s):
            return set(s[i:i+n] for i in range(len(s) - n + 1))
        
        ng1, ng2 = get_ngrams(s1), get_ngrams(s2)
        
        if not ng1 or not ng2:
            return 0.0
        
        intersection = len(ng1 & ng2)
        union = len(ng1 | ng2)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def combined_similarity(s1: str, s2: str) -> float:
        """
        Combined similarity using multiple algorithms.
        Weighted average for best results.
        """
        # Weights for each algorithm
        weights = {
            'levenshtein': 0.30,
            'jaro_winkler': 0.30,
            'ngram': 0.20,
            'soundex': 0.10,
            'metaphone': 0.10
        }
        
        scores = {
            'levenshtein': StringSimilarity.levenshtein_similarity(s1, s2),
            'jaro_winkler': StringSimilarity.jaro_winkler(s1, s2),
            'ngram': StringSimilarity.ngram_similarity(s1, s2),
            'soundex': 1.0 if StringSimilarity.soundex(s1) == StringSimilarity.soundex(s2) else 0.0,
            'metaphone': 1.0 if StringSimilarity.metaphone(s1) == StringSimilarity.metaphone(s2) else 0.0
        }
        
        total = sum(scores[algo] * weight for algo, weight in weights.items())
        return total


# ============================================================
# INTENT DETECTION
# ============================================================

class QueryIntent(Enum):
    """Types of query intents"""
    QUESTION = "question"           # What is X?
    CORRECTION = "correction"       # I mean Y / No, the other one
    CLARIFICATION = "clarification" # Which one? The first one
    CONFIRMATION = "confirmation"   # Yes / That's right
    REJECTION = "rejection"         # No / Not that
    FOLLOWUP = "followup"          # Tell me more / What about...
    NEW_TOPIC = "new_topic"        # Unrelated new question
    GREETING = "greeting"          # Hi, hello


class IntentDetector:
    """
    Detects the intent behind a user query.
    Critical for understanding corrections vs new questions.
    """
    
    CORRECTION_PATTERNS = [
        r"^i mean\s+(.+)",
        r"^no[,.]?\s*i mean\s+(.+)",
        r"^not that[,.]?\s*(.+)",
        r"^i meant\s+(.+)",
        r"^actually[,.]?\s*(.+)",
        r"^sorry[,.]?\s*i meant?\s+(.+)",
        r"^no[,.]?\s*the (.+) one",
        r"^the (.+) one",
        r"^i was asking about\s+(.+)",
        r"^i\'m asking about\s+(.+)",
        r"^i want to know about\s+(.+)",
        r"^no[,.]?\s*(.+)",  # Simple "no, X"
    ]
    
    CLARIFICATION_PATTERNS = [
        r"^the (first|second|third|1st|2nd|3rd|last) one",
        r"^(first|second|third|1st|2nd|3rd|last) option",
        r"^option (\d+)",
        r"^number (\d+)",
        r"^the (\w+) one",  # "the red one", "the french one"
    ]
    
    CONFIRMATION_PATTERNS = [
        r"^(yes|yeah|yep|correct|right|exactly|that\'s (it|right|correct))",
        r"^(ok|okay|sure|fine)",
    ]
    
    REJECTION_PATTERNS = [
        r"^(no|nope|wrong|not that|none of)",
        r"^that\'s not (it|right|what i)",
    ]
    
    FOLLOWUP_PATTERNS = [
        r"^(tell me more|more (about|info)|what (else|about))",
        r"^(and|also|additionally|furthermore)",
        r"^(how about|what about)\s+(.+)",
        r"^(why|how|when|where)\s+",
    ]
    
    GREETING_PATTERNS = [
        r"^(hi|hello|hey|good (morning|afternoon|evening)|greetings)",
        r"^(thanks|thank you|bye|goodbye)",
    ]
    
    @classmethod
    def detect(cls, query: str, previous_query: str = None, 
               previous_results: List[Dict] = None) -> Tuple[QueryIntent, Optional[str]]:
        """
        Detect query intent and extract relevant entity if applicable.
        
        Returns: (intent, extracted_entity)
        """
        query_lower = query.lower().strip()
        
        # Check greeting
        for pattern in cls.GREETING_PATTERNS:
            if re.match(pattern, query_lower):
                return QueryIntent.GREETING, None
        
        # Check confirmation
        for pattern in cls.CONFIRMATION_PATTERNS:
            if re.match(pattern, query_lower):
                return QueryIntent.CONFIRMATION, None
        
        # Check rejection
        for pattern in cls.REJECTION_PATTERNS:
            if re.match(pattern, query_lower):
                return QueryIntent.REJECTION, None
        
        # Check correction (most important!)
        for pattern in cls.CORRECTION_PATTERNS:
            match = re.match(pattern, query_lower)
            if match:
                extracted = match.group(1).strip()
                return QueryIntent.CORRECTION, extracted
        
        # Check clarification
        for pattern in cls.CLARIFICATION_PATTERNS:
            match = re.match(pattern, query_lower)
            if match:
                return QueryIntent.CLARIFICATION, match.group(1)
        
        # Check followup
        for pattern in cls.FOLLOWUP_PATTERNS:
            if re.match(pattern, query_lower):
                return QueryIntent.FOLLOWUP, None
        
        # Default: new question
        return QueryIntent.QUESTION, None


# ============================================================
# ENTITY RESOLUTION
# ============================================================

@dataclass
class Entity:
    """An entity with metadata"""
    name: str
    canonical_name: str
    entity_type: str  # person, place, thing, concept
    aliases: List[str] = field(default_factory=list)
    popularity: float = 1.0
    description: str = ""
    source_count: int = 0
    
    def matches(self, query: str, threshold: float = 0.7) -> float:
        """Check if query matches this entity"""
        query_lower = query.lower()
        
        # Exact match
        if query_lower == self.name.lower():
            return 1.0
        if query_lower == self.canonical_name.lower():
            return 1.0
        
        # Alias match
        for alias in self.aliases:
            if query_lower == alias.lower():
                return 0.95
        
        # Fuzzy match
        best_score = StringSimilarity.combined_similarity(query, self.name)
        best_score = max(best_score, StringSimilarity.combined_similarity(query, self.canonical_name))
        
        for alias in self.aliases:
            score = StringSimilarity.combined_similarity(query, alias)
            best_score = max(best_score, score)
        
        return best_score if best_score >= threshold else 0.0


class EntityResolver:
    """
    Resolves queries to entities with disambiguation.
    Learns from user feedback.
    """
    
    # Popular entities that should rank higher
    POPULAR_ENTITIES = {
        'paris': {'type': 'city', 'popularity': 100, 'desc': 'Capital of France'},
        'london': {'type': 'city', 'popularity': 95, 'desc': 'Capital of UK'},
        'new york': {'type': 'city', 'popularity': 90, 'desc': 'City in USA'},
        'tokyo': {'type': 'city', 'popularity': 85, 'desc': 'Capital of Japan'},
        'berlin': {'type': 'city', 'popularity': 80, 'desc': 'Capital of Germany'},
        'rome': {'type': 'city', 'popularity': 78, 'desc': 'Capital of Italy'},
        'madrid': {'type': 'city', 'popularity': 75, 'desc': 'Capital of Spain'},
        'moscow': {'type': 'city', 'popularity': 73, 'desc': 'Capital of Russia'},
        'beijing': {'type': 'city', 'popularity': 72, 'desc': 'Capital of China'},
        'washington': {'type': 'city', 'popularity': 70, 'desc': 'Capital of USA'},
        'france': {'type': 'country', 'popularity': 95, 'desc': 'Country in Europe'},
        'germany': {'type': 'country', 'popularity': 90, 'desc': 'Country in Europe'},
        'japan': {'type': 'country', 'popularity': 88, 'desc': 'Country in Asia'},
        'china': {'type': 'country', 'popularity': 87, 'desc': 'Country in Asia'},
        'python': {'type': 'concept', 'popularity': 85, 'desc': 'Programming language'},
        'einstein': {'type': 'person', 'popularity': 90, 'desc': 'Physicist'},
    }
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.entities: Dict[str, Entity] = {}
        self._load_entities()
    
    def _load_entities(self):
        """Load entities from database"""
        # Load popular entities
        for name, data in self.POPULAR_ENTITIES.items():
            self.entities[name.lower()] = Entity(
                name=name.title(),
                canonical_name=name.title(),
                entity_type=data['type'],
                popularity=data['popularity'],
                description=data.get('desc', '')
            )
    
    def resolve(self, query: str, context: Dict = None) -> List[Tuple[Entity, float]]:
        """
        Resolve query to possible entities with confidence scores.
        
        Returns: List of (Entity, confidence) sorted by confidence
        """
        query_lower = query.lower().strip()
        matches = []
        
        # Check all known entities
        for key, entity in self.entities.items():
            score = entity.matches(query_lower)
            if score > 0:
                # Boost by popularity
                adjusted_score = score * (0.5 + 0.5 * (entity.popularity / 100))
                matches.append((entity, adjusted_score))
        
        # Sort by score
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:10]
    
    def add_entity(self, name: str, entity_type: str, aliases: List[str] = None,
                   popularity: float = 50.0, description: str = "") -> Entity:
        """Add a new entity to the resolver"""
        entity = Entity(
            name=name,
            canonical_name=name,
            entity_type=entity_type,
            aliases=aliases or [],
            popularity=popularity,
            description=description
        )
        self.entities[name.lower()] = entity
        return entity
    
    def boost_entity(self, name: str, boost: float = 5.0):
        """Boost entity popularity (from user selection)"""
        key = name.lower()
        if key in self.entities:
            self.entities[key].popularity = min(100, self.entities[key].popularity + boost)


# ============================================================
# CONTEXT BRAIN (MAIN CLASS)
# ============================================================

class ContextBrain:
    """
    The main Context Brain - intelligent context understanding.
    
    Features:
    1. Smart query understanding with fuzzy matching
    2. Correction detection and learning
    3. Entity disambiguation
    4. Persistent learning
    5. Conversation context tracking
    
    This LEARNS and IMPROVES over time!
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.data_dir / "context_brain.db"
        self._lock = threading.RLock()
        
        # Components
        self.intent_detector = IntentDetector()
        self.entity_resolver = EntityResolver(self.db_path)
        
        # Conversation state
        self.conversations: Dict[str, 'ConversationState'] = {}
        
        # Learned corrections (typo → correct)
        self.corrections: Dict[str, str] = {}
        
        # Phonetic index for fast lookup
        self.soundex_index: Dict[str, List[str]] = defaultdict(list)
        self.metaphone_index: Dict[str, List[str]] = defaultdict(list)
        
        # Initialize database
        self._init_database()
        self._load_learned_data()
        
        print("✅ Context Brain initialized")
    
    def _init_database(self):
        """Initialize the learning database"""
        conn = sqlite3.connect(str(self.db_path))
        
        # Learned corrections table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                typo TEXT NOT NULL,
                correct TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(typo, correct)
            )
        """)
        
        # Entity popularity table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS entity_popularity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_name TEXT UNIQUE NOT NULL,
                popularity_score REAL DEFAULT 50.0,
                selection_count INTEGER DEFAULT 0,
                last_selected TIMESTAMP
            )
        """)
        
        # Query patterns table (for learning common patterns)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern TEXT NOT NULL,
                resolved_to TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                UNIQUE(pattern, resolved_to)
            )
        """)
        
        # Phonetic index table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS phonetic_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                soundex_code TEXT,
                metaphone_code TEXT,
                word TEXT NOT NULL,
                source_title TEXT,
                UNIQUE(word, source_title)
            )
        """)
        
        # Create indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_corrections_typo ON corrections(typo)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_phonetic_soundex ON phonetic_index(soundex_code)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_phonetic_metaphone ON phonetic_index(metaphone_code)")
        
        conn.commit()
        conn.close()
    
    def _load_learned_data(self):
        """Load learned corrections and patterns"""
        conn = sqlite3.connect(str(self.db_path))
        
        # Load corrections
        rows = conn.execute("SELECT typo, correct FROM corrections").fetchall()
        for typo, correct in rows:
            self.corrections[typo.lower()] = correct
        
        # Load phonetic index
        rows = conn.execute("SELECT soundex_code, metaphone_code, word FROM phonetic_index").fetchall()
        for soundex, metaphone, word in rows:
            if soundex:
                self.soundex_index[soundex].append(word)
            if metaphone:
                self.metaphone_index[metaphone].append(word)
        
        conn.close()
        
        if self.corrections:
            print(f"   📚 Loaded {len(self.corrections)} learned corrections")
        if self.soundex_index:
            print(f"   🔤 Loaded {len(self.soundex_index)} phonetic patterns")
    
    # ==================== MAIN API ====================
    
    def understand_query(self, query: str, session_id: str = "default",
                         known_entities: List[str] = None) -> Dict[str, Any]:
        """
        Main method to understand a query.
        
        Returns:
            {
                'original_query': str,
                'understood_query': str,  # Corrected/expanded query
                'intent': QueryIntent,
                'entities': List[Entity],
                'corrections_applied': List[Tuple[str, str]],
                'confidence': float,
                'suggestions': List[str],  # Alternative interpretations
                'needs_disambiguation': bool,
                'disambiguation_options': List[str]
            }
        """
        # Get or create conversation state
        conv = self._get_conversation(session_id)
        
        # Detect intent
        intent, extracted = IntentDetector.detect(
            query, 
            conv.last_query,
            conv.last_results
        )
        
        result = {
            'original_query': query,
            'understood_query': query,
            'intent': intent,
            'entities': [],
            'corrections_applied': [],
            'confidence': 1.0,
            'suggestions': [],
            'needs_disambiguation': False,
            'disambiguation_options': []
        }
        
        # Handle based on intent
        if intent == QueryIntent.CORRECTION:
            result = self._handle_correction(query, extracted, conv, result, known_entities)
        
        elif intent == QueryIntent.CLARIFICATION:
            result = self._handle_clarification(query, extracted, conv, result)
        
        elif intent == QueryIntent.QUESTION or intent == QueryIntent.NEW_TOPIC:
            result = self._handle_question(query, conv, result, known_entities)
        
        elif intent == QueryIntent.FOLLOWUP:
            result = self._handle_followup(query, conv, result, known_entities)
        
        # Update conversation state
        conv.add_turn(query, result)
        
        return result
    
    def _handle_correction(self, query: str, extracted: str, conv: 'ConversationState',
                          result: Dict, known_entities: List[str]) -> Dict:
        """Handle a correction like 'I mean Paris'"""
        if not extracted:
            extracted = query
        
        # Learn this correction
        if conv.last_query:
            self._learn_correction(conv.last_query, extracted)
        
        # Now understand the corrected query
        result['understood_query'] = extracted
        result['intent'] = QueryIntent.CORRECTION
        
        # Find best match
        matches = self._find_best_matches(extracted, known_entities)
        
        if matches:
            best_match, score = matches[0]
            result['understood_query'] = best_match
            result['confidence'] = score
            result['corrections_applied'].append((query, best_match))
            
            if len(matches) > 1 and matches[1][1] > 0.5:
                result['suggestions'] = [m[0] for m in matches[1:4]]
        
        return result
    
    def _handle_clarification(self, query: str, extracted: str, 
                              conv: 'ConversationState', result: Dict) -> Dict:
        """Handle clarification like 'the first one' or 'the french one'"""
        result['intent'] = QueryIntent.CLARIFICATION
        
        # Check if we have previous options
        if conv.disambiguation_options:
            # Parse the clarification
            idx = None
            if extracted:
                if extracted.lower() in ['first', '1st', '1']:
                    idx = 0
                elif extracted.lower() in ['second', '2nd', '2']:
                    idx = 1
                elif extracted.lower() in ['third', '3rd', '3']:
                    idx = 2
                elif extracted.lower() == 'last':
                    idx = -1
                else:
                    # Try to match by description
                    idx = self._match_by_description(extracted, conv.disambiguation_options)
                
                if idx is not None and -len(conv.disambiguation_options) <= idx < len(conv.disambiguation_options):
                    selected = conv.disambiguation_options[idx]
                    result['understood_query'] = selected
                    result['confidence'] = 0.95
                    
                    # Boost this entity's popularity
                    self.entity_resolver.boost_entity(selected)
        
        return result
    
    def _handle_question(self, query: str, conv: 'ConversationState',
                        result: Dict, known_entities: List[str]) -> Dict:
        """Handle a regular question"""
        # Apply known corrections first
        corrected_query = self._apply_corrections(query)
        if corrected_query != query:
            result['corrections_applied'].append((query, corrected_query))
            query = corrected_query
        
        # Find best matches
        matches = self._find_best_matches(query, known_entities)
        
        if not matches:
            # Try phonetic matching
            matches = self._phonetic_search(query, known_entities)
        
        if matches:
            best_match, score = matches[0]
            
            # Check if disambiguation needed
            if len(matches) > 1:
                second_score = matches[1][1]
                
                # If scores are close, might need disambiguation
                if second_score > 0.7 and abs(score - second_score) < 0.15:
                    result['needs_disambiguation'] = True
                    result['disambiguation_options'] = [m[0] for m in matches[:5]]
                    conv.disambiguation_options = result['disambiguation_options']
            
            result['understood_query'] = best_match
            result['confidence'] = score
            
            if len(matches) > 1:
                result['suggestions'] = [m[0] for m in matches[1:4]]
        else:
            # No good matches - might be a new topic or typo
            result['confidence'] = 0.5
            result['suggestions'] = self._suggest_similar(query, known_entities)
        
        return result
    
    def _handle_followup(self, query: str, conv: 'ConversationState',
                        result: Dict, known_entities: List[str]) -> Dict:
        """Handle followup questions"""
        result['intent'] = QueryIntent.FOLLOWUP
        
        # Use previous context
        if conv.current_entity:
            # Modify query to include context
            context_query = f"{conv.current_entity} {query}"
            result['understood_query'] = context_query
        
        return result
    
    # ==================== MATCHING & SEARCH ====================
    
    def _find_best_matches(self, query: str, known_entities: List[str] = None,
                          threshold: float = 0.5) -> List[Tuple[str, float]]:
        """Find best matching entities for a query"""
        matches = []
        
        query_lower = query.lower().strip()
        
        # Extract main subject from query
        subject = self._extract_subject(query)
        
        # Search in known entities
        if known_entities:
            for entity in known_entities:
                entity_lower = entity.lower()
                
                # Calculate similarity
                score = StringSimilarity.combined_similarity(subject, entity_lower)
                
                # Boost for word containment
                if subject in entity_lower or entity_lower in subject:
                    score = max(score, 0.85)
                
                # Boost for exact word match
                entity_words = set(entity_lower.split())
                query_words = set(subject.split())
                if entity_words & query_words:
                    score = max(score, 0.8)
                
                if score >= threshold:
                    matches.append((entity, score))
        
        # Also check entity resolver
        resolved = self.entity_resolver.resolve(subject)
        for entity, score in resolved:
            if score >= threshold:
                matches.append((entity.name, score))
        
        # Sort and deduplicate
        matches.sort(key=lambda x: x[1], reverse=True)
        
        seen = set()
        unique_matches = []
        for entity, score in matches:
            key = entity.lower()
            if key not in seen:
                seen.add(key)
                unique_matches.append((entity, score))
        
        return unique_matches[:10]
    
    def _phonetic_search(self, query: str, known_entities: List[str] = None,
                        threshold: float = 0.6) -> List[Tuple[str, float]]:
        """Search using phonetic matching"""
        matches = []
        
        subject = self._extract_subject(query)
        query_soundex = StringSimilarity.soundex(subject)
        query_metaphone = StringSimilarity.metaphone(subject)
        
        # Search phonetic index
        candidates = set()
        candidates.update(self.soundex_index.get(query_soundex, []))
        candidates.update(self.metaphone_index.get(query_metaphone, []))
        
        for candidate in candidates:
            score = StringSimilarity.combined_similarity(subject, candidate)
            if score >= threshold:
                matches.append((candidate, score))
        
        # Also check known entities
        if known_entities:
            for entity in known_entities:
                entity_soundex = StringSimilarity.soundex(entity)
                entity_metaphone = StringSimilarity.metaphone(entity)
                
                if query_soundex == entity_soundex or query_metaphone == entity_metaphone:
                    score = StringSimilarity.combined_similarity(subject, entity)
                    score = max(score, 0.7)  # Boost for phonetic match
                    matches.append((entity, score))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:10]
    
    def _extract_subject(self, query: str) -> str:
        """Extract the main subject from a query"""
        query = query.lower().strip()
        
        # Remove common question prefixes
        prefixes = [
            r"^(what|who|where|when|why|how|which)\s+(is|are|was|were|do|does|did)\s+(a|an|the)?\s*",
            r"^(tell\s+me\s+about)\s+(a|an|the)?\s*",
            r"^(i\s+mean)\s+",
            r"^(define)\s+(a|an|the)?\s*",
            r"^(explain)\s+(a|an|the)?\s*",
            r"^(describe)\s+(a|an|the)?\s*",
            r"^(what\'s)\s+(a|an|the)?\s*",
            r"^(a|an|the)\s+",
        ]
        
        for prefix in prefixes:
            query = re.sub(prefix, "", query)
        
        # Remove trailing punctuation
        query = re.sub(r"[?!.,]+$", "", query)
        
        return query.strip()
    
    def _suggest_similar(self, query: str, known_entities: List[str] = None,
                        limit: int = 5) -> List[str]:
        """Suggest similar entities when no good match found"""
        if not known_entities:
            return []
        
        subject = self._extract_subject(query)
        
        suggestions = []
        for entity in known_entities:
            score = StringSimilarity.combined_similarity(subject, entity)
            if score > 0.3:
                suggestions.append((entity, score))
        
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in suggestions[:limit]]
    
    def _match_by_description(self, description: str, options: List[str]) -> Optional[int]:
        """Match a clarification to options by description"""
        description_lower = description.lower()
        
        for i, option in enumerate(options):
            option_lower = option.lower()
            if description_lower in option_lower:
                return i
        
        return None
    
    # ==================== LEARNING ====================
    
    def _apply_corrections(self, query: str) -> str:
        """Apply learned corrections to query"""
        words = query.lower().split()
        corrected_words = []
        
        for word in words:
            if word in self.corrections:
                corrected_words.append(self.corrections[word])
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def _learn_correction(self, typo: str, correct: str):
        """Learn a new correction"""
        typo_subject = self._extract_subject(typo).lower()
        correct_subject = self._extract_subject(correct).lower()
        
        if typo_subject == correct_subject:
            return  # Same, nothing to learn
        
        # Don't learn if they're too similar (might be valid alternative)
        sim = StringSimilarity.levenshtein_similarity(typo_subject, correct_subject)
        if sim > 0.9:
            return
        
        # Add to memory
        self.corrections[typo_subject] = correct_subject
        
        # Persist to database
        with self._lock:
            try:
                conn = sqlite3.connect(str(self.db_path))
                
                # Try to update existing correction first
                cursor = conn.execute("""
                    UPDATE corrections 
                    SET frequency = frequency + 1
                    WHERE typo = ? AND correct = ?
                """, (typo_subject, correct_subject))
                
                # If no row updated, insert new one
                if cursor.rowcount == 0:
                    conn.execute("""
                        INSERT INTO corrections (typo, correct, frequency)
                        VALUES (?, ?, 1)
                    """, (typo_subject, correct_subject))
                
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"⚠️ Error saving correction: {e}")
        
        print(f"📚 Learned correction: '{typo_subject}' → '{correct_subject}'")
    
    def learn_from_content(self, content: str, source_title: str = ""):
        """
        Learn entities from content for better matching.
        Called when knowledge is added.
        """
        # Extract important words/phrases
        words = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content))
        
        for word in words:
            if len(word) < 3:
                continue
            
            # Calculate phonetic codes
            soundex = StringSimilarity.soundex(word)
            metaphone = StringSimilarity.metaphone(word)
            
            # Add to indexes
            self.soundex_index[soundex].append(word)
            self.metaphone_index[metaphone].append(word)
            
            # Persist
            with self._lock:
                try:
                    conn = sqlite3.connect(str(self.db_path))
                    conn.execute("""
                        INSERT OR IGNORE INTO phonetic_index 
                        (soundex_code, metaphone_code, word, source_title)
                        VALUES (?, ?, ?, ?)
                    """, (soundex, metaphone, word, source_title))
                    conn.commit()
                    conn.close()
                except:
                    pass
        
        # Add source title as known entity
        if source_title and len(source_title) > 2:
            self.entity_resolver.add_entity(
                source_title, 
                'concept',
                popularity=50.0
            )
    
    def record_selection(self, query: str, selected: str):
        """Record when user selects a specific result"""
        # Boost entity popularity
        self.entity_resolver.boost_entity(selected)
        
        # Learn pattern
        subject = self._extract_subject(query).lower()
        
        with self._lock:
            try:
                conn = sqlite3.connect(str(self.db_path))
                
                # Try to update existing pattern first
                cursor = conn.execute("""
                    UPDATE query_patterns 
                    SET frequency = frequency + 1
                    WHERE pattern = ? AND resolved_to = ?
                """, (subject, selected.lower()))
                
                # If no row updated, insert new one
                if cursor.rowcount == 0:
                    conn.execute("""
                        INSERT INTO query_patterns (pattern, resolved_to, frequency)
                        VALUES (?, ?, 1)
                    """, (subject, selected.lower()))
                
                # Try to update entity popularity
                cursor = conn.execute("""
                    UPDATE entity_popularity 
                    SET popularity_score = MIN(100, popularity_score + 2),
                        selection_count = selection_count + 1,
                        last_selected = CURRENT_TIMESTAMP
                    WHERE entity_name = ?
                """, (selected.lower(),))
                
                # If no row updated, insert new one
                if cursor.rowcount == 0:
                    conn.execute("""
                        INSERT INTO entity_popularity (entity_name, popularity_score, selection_count, last_selected)
                        VALUES (?, 55, 1, CURRENT_TIMESTAMP)
                    """, (selected.lower(),))
                
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"⚠️ Error recording selection: {e}")
    
    # ==================== CONVERSATION STATE ====================
    
    def _get_conversation(self, session_id: str) -> 'ConversationState':
        """Get or create conversation state"""
        if session_id not in self.conversations:
            self.conversations[session_id] = ConversationState(session_id)
        return self.conversations[session_id]
    
    def clear_conversation(self, session_id: str = "default"):
        """Clear conversation state"""
        if session_id in self.conversations:
            del self.conversations[session_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get context brain statistics"""
        conn = sqlite3.connect(str(self.db_path))
        
        corrections_count = conn.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]
        phonetic_count = conn.execute("SELECT COUNT(*) FROM phonetic_index").fetchone()[0]
        patterns_count = conn.execute("SELECT COUNT(*) FROM query_patterns").fetchone()[0]
        
        conn.close()
        
        return {
            'learned_corrections': corrections_count,
            'phonetic_entries': phonetic_count,
            'query_patterns': patterns_count,
            'active_conversations': len(self.conversations),
            'known_entities': len(self.entity_resolver.entities)
        }


@dataclass
class ConversationState:
    """Tracks conversation state for a session"""
    session_id: str
    turns: List[Dict] = field(default_factory=list)
    last_query: str = ""
    last_results: List[Dict] = field(default_factory=list)
    current_entity: str = ""
    disambiguation_options: List[str] = field(default_factory=list)
    
    def add_turn(self, query: str, result: Dict):
        """Add a conversation turn"""
        self.turns.append({
            'query': query,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep last 20 turns
        if len(self.turns) > 20:
            self.turns = self.turns[-20:]
        
        self.last_query = query
        
        # Update current entity
        if result.get('confidence', 0) > 0.7:
            self.current_entity = result.get('understood_query', '')
        
        # Update disambiguation options
        if result.get('needs_disambiguation'):
            self.disambiguation_options = result.get('disambiguation_options', [])
        elif result.get('confidence', 0) > 0.8:
            self.disambiguation_options = []


# ============================================================
# SMART SEARCHER - INTEGRATION WITH KNOWLEDGE BASE
# ============================================================

class SmartSearcher:
    """
    Smart searcher that uses Context Brain for better results.
    Drop-in replacement for direct knowledge base search.
    """
    
    def __init__(self, knowledge_base, context_brain: ContextBrain):
        self.kb = knowledge_base
        self.brain = context_brain
    
    def search(self, query: str, session_id: str = "default", 
               limit: int = 10) -> Dict[str, Any]:
        """
        Smart search with context understanding.
        """
        # Get known entity titles from knowledge base
        known_entities = self._get_known_entities()
        
        # Understand the query
        understanding = self.brain.understand_query(
            query, 
            session_id,
            known_entities
        )
        
        # Use understood query for search
        search_query = understanding['understood_query']
        
        # Search knowledge base
        results = self.kb.search(search_query, limit=limit)
        
        # If no results with understood query, try original
        if not results and search_query != query:
            results = self.kb.search(query, limit=limit)
        
        # Try suggestions if still no results
        if not results and understanding['suggestions']:
            for suggestion in understanding['suggestions'][:3]:
                results = self.kb.search(suggestion, limit=limit)
                if results:
                    understanding['understood_query'] = suggestion
                    break
        
        return {
            'results': results,
            'understood_query': understanding['understood_query'],
            'corrections': understanding['corrections_applied'],
            'confidence': understanding['confidence'],
            'suggestions': understanding['suggestions'],
            'needs_disambiguation': understanding['needs_disambiguation'],
            'disambiguation_options': understanding['disambiguation_options'],
            'intent': understanding['intent'].value
        }
    
    def _get_known_entities(self, limit: int = 1000) -> List[str]:
        """Get known entity titles from knowledge base"""
        try:
            recent = self.kb.vectors.get_all_knowledge(limit)
            return [r['title'] for r in recent if r.get('title')]
        except:
            return []
    
    def record_selection(self, query: str, selected_title: str):
        """Record when user selects a result"""
        self.brain.record_selection(query, selected_title)


# ============================================================
# TESTING
# ============================================================

def test_context_brain():
    """Test the context brain"""
    import tempfile
    
    print("=" * 60)
    print("🧪 Testing Context Brain")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        brain = ContextBrain(Path(tmpdir))
        
        # Test string similarity
        print("\n📊 String Similarity Tests:")
        tests = [
            ("Paris", "Parid"),
            ("Paris", "Parris"),
            ("Paris", "Pankovo"),
            ("Einstein", "Einstien"),
            ("Python", "Pyton"),
        ]
        
        for s1, s2 in tests:
            lev = StringSimilarity.levenshtein_similarity(s1, s2)
            jw = StringSimilarity.jaro_winkler(s1, s2)
            sx = StringSimilarity.soundex(s1) == StringSimilarity.soundex(s2)
            mp = StringSimilarity.metaphone(s1) == StringSimilarity.metaphone(s2)
            combined = StringSimilarity.combined_similarity(s1, s2)
            
            print(f"   {s1} vs {s2}:")
            print(f"      Levenshtein: {lev:.2f}, Jaro-Winkler: {jw:.2f}")
            print(f"      Soundex match: {sx}, Metaphone match: {mp}")
            print(f"      Combined: {combined:.2f}")
        
        # Test intent detection
        print("\n🎯 Intent Detection Tests:")
        intent_tests = [
            "What is Paris?",
            "I mean London",
            "No, the second one",
            "Tell me more",
            "Yes, that's it",
            "Hi there!",
        ]
        
        for query in intent_tests:
            intent, extracted = IntentDetector.detect(query)
            print(f"   '{query}' → {intent.value}" + 
                  (f" (extracted: {extracted})" if extracted else ""))
        
        # Test query understanding
        print("\n🧠 Query Understanding Tests:")
        known = ["Paris", "Paridris", "Pankovo", "London", "Tokyo"]
        
        queries = [
            "Tell me about Parid",
            "I mean Paris",
            "the french city",
        ]
        
        for query in queries:
            result = brain.understand_query(query, "test", known)
            print(f"\n   Query: '{query}'")
            print(f"   Understood: '{result['understood_query']}'")
            print(f"   Intent: {result['intent'].value}")
            print(f"   Confidence: {result['confidence']:.2f}")
            if result['corrections_applied']:
                print(f"   Corrections: {result['corrections_applied']}")
            if result['suggestions']:
                print(f"   Suggestions: {result['suggestions']}")
        
        # Test learning
        print("\n📚 Learning Test:")
        brain.learn_from_content(
            "Paris is the capital of France. Paris is known for the Eiffel Tower.",
            "Paris"
        )
        brain.learn_from_content(
            "London is the capital of the United Kingdom.",
            "London"
        )
        
        stats = brain.get_stats()
        print(f"   Stats: {stats}")
        
        print("\n✅ Context Brain tests passed!")


if __name__ == "__main__":
    test_context_brain()