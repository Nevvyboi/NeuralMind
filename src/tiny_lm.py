"""
TinyLM - Lightweight Language Model for Query Understanding
Part of GroundZero AI Neural Pipeline

Features:
- Word-level tokenization with 10K vocabulary
- Self-attention pooling for query encoding
- Entity extraction (keywords, proper nouns, quoted terms)
- Question type classification (6 types)
- Alignment with TransE embeddings
"""

import math
import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import random


class QuestionType(Enum):
    """Types of questions the system can handle"""
    FACTUAL = "factual"           # What is X? Who is Y?
    CAUSAL = "causal"             # Why does X? What causes Y?
    COUNTERFACTUAL = "counterfactual"  # What if X? What would happen?
    PROCEDURAL = "procedural"     # How to X? Steps to Y?
    COMPARATIVE = "comparative"   # Compare X and Y? Difference between?
    DEFINITIONAL = "definitional" # Define X? Meaning of Y?


@dataclass
class QueryAnalysis:
    """Result of query analysis"""
    OriginalQuery: str
    NormalizedQuery: str
    Tokens: List[str]
    Entities: List[str]
    Keywords: List[str]
    QuestionType: QuestionType
    Embedding: List[float]
    Confidence: float


class TinyLM:
    """
    Lightweight Language Model for understanding queries.
    
    This is NOT a generative model - it's designed to:
    1. Tokenize and normalize queries
    2. Extract entities and keywords
    3. Classify question types
    4. Create query embeddings aligned with TransE
    """
    
    def __init__(self, EmbedDim: int = 100, VocabSize: int = 10000):
        """
        Initialize TinyLM.
        
        Args:
            EmbedDim: Dimension of embeddings (should match TransE)
            VocabSize: Maximum vocabulary size
        """
        self.EmbedDim = EmbedDim
        self.VocabSize = VocabSize
        
        # Vocabulary
        self.Word2ID: Dict[str, int] = {'<PAD>': 0, '<UNK>': 1}
        self.ID2Word: Dict[int, str] = {0: '<PAD>', 1: '<UNK>'}
        self.WordFreq: Dict[str, int] = {}
        
        # Word embeddings (initialized randomly, aligned with TransE later)
        self.Embeddings: Dict[str, List[float]] = {}
        
        # Attention weights for pooling
        self.AttentionWeights: List[float] = self._InitWeights(EmbedDim)
        self.AttentionBias: float = 0.0
        
        # Question type patterns
        self.QuestionPatterns = self._InitQuestionPatterns()
        
        # Stop words for keyword extraction
        self.StopWords = self._InitStopWords()
        
        # TransE alignment (set after linking)
        self.TransEEmbeddings: Dict[str, List[float]] = {}
        self.IsAligned = False
    
    def _InitWeights(self, Dim: int) -> List[float]:
        """Initialize weights with Xavier initialization"""
        scale = math.sqrt(2.0 / Dim)
        return [random.gauss(0, scale) for _ in range(Dim)]
    
    def _InitQuestionPatterns(self) -> Dict[QuestionType, List[str]]:
        """Initialize patterns for question type detection"""
        return {
            QuestionType.FACTUAL: [
                r'^what is\b', r'^who is\b', r'^where is\b', r'^when\b',
                r'^which\b', r'^what are\b', r'^who are\b', r'^name\b',
                r'^list\b', r'^tell me about\b'
            ],
            QuestionType.CAUSAL: [
                r'^why\b', r'^what causes?\b', r'^what leads? to\b',
                r'^reason for\b', r'^because of\b', r'cause of\b',
                r'result in\b', r'leads? to\b'
            ],
            QuestionType.COUNTERFACTUAL: [
                r'^what if\b', r'^what would\b', r'^suppose\b',
                r'^imagine\b', r'^hypothetically\b', r'^if .* then\b',
                r'would happen\b', r'could happen\b'
            ],
            QuestionType.PROCEDURAL: [
                r'^how to\b', r'^how do\b', r'^how can\b', r'^steps to\b',
                r'^process of\b', r'^method for\b', r'^way to\b',
                r'^procedure\b', r'^instructions?\b'
            ],
            QuestionType.COMPARATIVE: [
                r'^compare\b', r'^difference between\b', r'^versus\b',
                r'^vs\.?\b', r'^better\b', r'^worse\b', r'^similar\b',
                r'compared to\b', r'different from\b', r'same as\b'
            ],
            QuestionType.DEFINITIONAL: [
                r'^define\b', r'^meaning of\b', r'^definition\b',
                r'^what does .* mean\b', r'^explain\b', r'^describe\b'
            ]
        }
    
    def _InitStopWords(self) -> Set[str]:
        """Initialize common stop words"""
        return {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'between',
            'under', 'again', 'further', 'then', 'once', 'here', 'there',
            'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
            'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and',
            'but', 'if', 'or', 'because', 'until', 'while', 'although',
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
            'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
            'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
            'am', 'about', 'tell', 'please', 'know', 'like', 'want'
        }
    
    def Tokenize(self, Text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            Text: Input text
            
        Returns:
            List of tokens
        """
        # Normalize
        text = Text.lower().strip()
        
        # Handle contractions
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'s", " is", text)
        text = re.sub(r"'d", " would", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'m", " am", text)
        
        # Split on whitespace and punctuation, keeping words
        tokens = re.findall(r'\b\w+\b', text)
        
        return tokens
    
    def BuildVocab(self, Texts: List[str]):
        """
        Build vocabulary from texts.
        
        Args:
            Texts: List of training texts
        """
        # Count word frequencies
        for text in Texts:
            tokens = self.Tokenize(text)
            for token in tokens:
                self.WordFreq[token] = self.WordFreq.get(token, 0) + 1
        
        # Sort by frequency and take top VocabSize
        sorted_words = sorted(self.WordFreq.items(), key=lambda x: -x[1])
        
        for word, freq in sorted_words[:self.VocabSize - 2]:  # -2 for PAD, UNK
            idx = len(self.Word2ID)
            self.Word2ID[word] = idx
            self.ID2Word[idx] = word
            
            # Initialize embedding
            self.Embeddings[word] = self._InitWeights(self.EmbedDim)
    
    def GetEmbedding(self, Word: str) -> List[float]:
        """
        Get embedding for a word.
        
        Args:
            Word: Input word
            
        Returns:
            Embedding vector
        """
        word = Word.lower()
        
        # Check TransE alignment first
        if self.IsAligned and word in self.TransEEmbeddings:
            return self.TransEEmbeddings[word]
        
        # Check local embeddings
        if word in self.Embeddings:
            return self.Embeddings[word]
        
        # Return zero vector for unknown
        return [0.0] * self.EmbedDim
    
    def _SelfAttentionPool(self, TokenEmbeddings: List[List[float]]) -> List[float]:
        """
        Pool token embeddings using self-attention.
        
        Args:
            TokenEmbeddings: List of token embeddings
            
        Returns:
            Pooled embedding
        """
        if not TokenEmbeddings:
            return [0.0] * self.EmbedDim
        
        # Compute attention scores
        scores = []
        for emb in TokenEmbeddings:
            # Score = W . emb + b
            score = sum(w * e for w, e in zip(self.AttentionWeights, emb))
            score += self.AttentionBias
            scores.append(score)
        
        # Softmax
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        sum_exp = sum(exp_scores)
        weights = [e / sum_exp for e in exp_scores]
        
        # Weighted sum
        pooled = [0.0] * self.EmbedDim
        for weight, emb in zip(weights, TokenEmbeddings):
            for i in range(self.EmbedDim):
                pooled[i] += weight * emb[i]
        
        return pooled
    
    def ExtractEntities(self, Text: str) -> List[str]:
        """
        Extract entities from text.
        
        Entities include:
        - Proper nouns (capitalized words)
        - Quoted terms
        - Technical terms
        - Multi-word phrases
        
        Args:
            Text: Input text
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Extract quoted terms
        quoted = re.findall(r'"([^"]+)"', Text)
        entities.extend(quoted)
        quoted = re.findall(r"'([^']+)'", Text)
        entities.extend(quoted)
        
        # Extract proper nouns (capitalized words not at sentence start)
        words = Text.split()
        for i, word in enumerate(words):
            # Skip first word and common words
            if i == 0:
                continue
            
            # Check if capitalized and not all caps
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word and clean_word[0].isupper() and not clean_word.isupper():
                entities.append(clean_word.lower())
        
        # Extract potential multi-word entities (consecutive capitalized words)
        multi_word_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
        multi_words = re.findall(multi_word_pattern, Text)
        entities.extend([mw.lower() for mw in multi_words])
        
        # Deduplicate while preserving order
        seen = set()
        unique_entities = []
        for e in entities:
            e_lower = e.lower().strip()
            if e_lower and e_lower not in seen:
                seen.add(e_lower)
                unique_entities.append(e_lower)
        
        return unique_entities
    
    def ExtractKeywords(self, Text: str) -> List[str]:
        """
        Extract keywords from text.
        
        Keywords are important words that are:
        - Not stop words
        - Nouns, verbs, adjectives
        - Relevant to the query
        
        Args:
            Text: Input text
            
        Returns:
            List of keywords
        """
        tokens = self.Tokenize(Text)
        
        keywords = []
        for token in tokens:
            if token not in self.StopWords and len(token) > 2:
                keywords.append(token)
        
        return keywords
    
    def ClassifyQuestionType(self, Text: str) -> Tuple[QuestionType, float]:
        """
        Classify the type of question.
        
        Args:
            Text: Input query
            
        Returns:
            Tuple of (QuestionType, confidence)
        """
        text_lower = Text.lower().strip()
        
        # Check patterns for each type
        type_scores = {}
        
        for qtype, patterns in self.QuestionPatterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    score += 1
            type_scores[qtype] = score
        
        # Find best match
        best_type = QuestionType.FACTUAL  # Default
        best_score = 0
        
        for qtype, score in type_scores.items():
            if score > best_score:
                best_score = score
                best_type = qtype
        
        # Calculate confidence
        total_score = sum(type_scores.values())
        if total_score > 0:
            confidence = best_score / total_score
        else:
            confidence = 0.5  # Default confidence for no pattern match
        
        return best_type, confidence
    
    def Encode(self, Text: str) -> List[float]:
        """
        Encode text into a fixed-size embedding.
        
        Args:
            Text: Input text
            
        Returns:
            Embedding vector
        """
        tokens = self.Tokenize(Text)
        
        if not tokens:
            return [0.0] * self.EmbedDim
        
        # Get embeddings for all tokens
        token_embeddings = []
        for token in tokens:
            emb = self.GetEmbedding(token)
            token_embeddings.append(emb)
        
        # Pool using self-attention
        return self._SelfAttentionPool(token_embeddings)
    
    def AlignWithTransE(self, TransEEmbeddings: Dict[str, List[float]], BlendRatio: float = 0.7):
        """
        Align TinyLM embeddings with TransE embeddings.
        
        This ensures consistency between query embeddings and
        knowledge graph entity embeddings.
        
        Args:
            TransEEmbeddings: Entity embeddings from TransE
            BlendRatio: How much to weight TransE (0.7 = 70% TransE)
        """
        self.TransEEmbeddings = {}
        
        for entity, transe_emb in TransEEmbeddings.items():
            # Normalize entity name
            entity_lower = entity.lower().replace('_', ' ')
            
            if entity_lower in self.Embeddings:
                # Blend existing embedding with TransE
                local_emb = self.Embeddings[entity_lower]
                blended = []
                for t, l in zip(transe_emb, local_emb):
                    blended.append(BlendRatio * t + (1 - BlendRatio) * l)
                self.TransEEmbeddings[entity_lower] = blended
            else:
                # Use TransE embedding directly
                self.TransEEmbeddings[entity_lower] = transe_emb
        
        self.IsAligned = True
    
    def Analyze(self, Query: str) -> QueryAnalysis:
        """
        Perform complete analysis of a query.
        
        Args:
            Query: Input query
            
        Returns:
            QueryAnalysis with all extracted information
        """
        # Normalize
        normalized = Query.strip()
        
        # Tokenize
        tokens = self.Tokenize(Query)
        
        # Extract entities and keywords
        entities = self.ExtractEntities(Query)
        keywords = self.ExtractKeywords(Query)
        
        # Classify question type
        qtype, confidence = self.ClassifyQuestionType(Query)
        
        # Get embedding
        embedding = self.Encode(Query)
        
        return QueryAnalysis(
            OriginalQuery=Query,
            NormalizedQuery=normalized.lower(),
            Tokens=tokens,
            Entities=entities,
            Keywords=keywords,
            QuestionType=qtype,
            Embedding=embedding,
            Confidence=confidence
        )
    
    def GetStats(self) -> Dict:
        """Get statistics about the model"""
        return {
            "VocabSize": len(self.Word2ID),
            "EmbedDim": self.EmbedDim,
            "IsAligned": self.IsAligned,
            "TransEEntities": len(self.TransEEmbeddings),
            "StopWords": len(self.StopWords)
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing TinyLM...")
    
    # Create instance
    lm = TinyLM(EmbedDim=100)
    
    # Test queries
    test_queries = [
        "What is a dog?",
        "Why do birds fly?",
        "Compare cats and dogs",
        "How to make coffee?",
        "What if the sun disappeared?",
        "Define photosynthesis"
    ]
    
    for query in test_queries:
        analysis = lm.Analyze(query)
        print(f"\nQuery: {query}")
        print(f"  Type: {analysis.QuestionType.value}")
        print(f"  Entities: {analysis.Entities}")
        print(f"  Keywords: {analysis.Keywords}")
        print(f"  Confidence: {analysis.Confidence:.2f}")