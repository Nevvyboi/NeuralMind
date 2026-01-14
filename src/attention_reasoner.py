"""
AttentionReasoner - Multi-Hop Reasoning with Attention Tracking
Part of GroundZero AI Neural Pipeline

Features:
- Multi-head attention (4 heads) over graph nodes
- 3-hop maximum reasoning depth
- Explicit path tracking with attention weights
- Candidate answer scoring
- Natural language explanation generation
"""

import math
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class ReasoningHop:
    """Single hop in reasoning chain"""
    FromEntity: str
    Relation: str
    ToEntity: str
    AttentionWeight: float
    HopNumber: int


@dataclass 
class ReasoningPath:
    """Complete reasoning path from query to answer"""
    Hops: List[ReasoningHop] = field(default_factory=list)
    TotalConfidence: float = 0.0
    
    def AddHop(self, hop: ReasoningHop):
        self.Hops.append(hop)
        # Confidence decays with each hop
        if len(self.Hops) == 1:
            self.TotalConfidence = hop.AttentionWeight
        else:
            self.TotalConfidence *= hop.AttentionWeight
    
    def GetEntities(self) -> List[str]:
        """Get all entities in the path"""
        entities = []
        for hop in self.Hops:
            if hop.FromEntity not in entities:
                entities.append(hop.FromEntity)
            if hop.ToEntity not in entities:
                entities.append(hop.ToEntity)
        return entities


@dataclass
class CandidateAnswer:
    """Candidate answer with scoring"""
    Entity: str
    Score: float
    Path: ReasoningPath
    Explanation: str


class AttentionReasoner:
    """
    Multi-hop reasoner using attention mechanism.
    
    Performs reasoning over knowledge graph by:
    1. Starting from query entities
    2. Using attention to select relevant relations/entities
    3. Following paths up to N hops
    4. Scoring candidate answers
    """
    
    def __init__(
        self, 
        EmbedDim: int = 100, 
        NumHeads: int = 4, 
        MaxHops: int = 3
    ):
        """
        Initialize AttentionReasoner.
        
        Args:
            EmbedDim: Dimension of embeddings
            NumHeads: Number of attention heads
            MaxHops: Maximum reasoning hops
        """
        self.EmbedDim = EmbedDim
        self.NumHeads = NumHeads
        self.HeadDim = EmbedDim // NumHeads
        self.MaxHops = MaxHops
        
        # Attention weights for each head
        self.QueryWeights: List[List[float]] = [
            self._InitWeights(self.HeadDim, EmbedDim)
            for _ in range(NumHeads)
        ]
        self.KeyWeights: List[List[float]] = [
            self._InitWeights(self.HeadDim, EmbedDim)
            for _ in range(NumHeads)
        ]
        self.ValueWeights: List[List[float]] = [
            self._InitWeights(self.HeadDim, EmbedDim)
            for _ in range(NumHeads)
        ]
        
        # Output projection
        self.OutputWeights = self._InitWeights(EmbedDim, EmbedDim)
        
        # Knowledge graph connections (set externally)
        self.EntityTriples: Dict[str, List[Tuple[str, str, str]]] = {}
        self.EntityEmbeddings: Dict[str, List[float]] = {}
        self.RelationEmbeddings: Dict[str, List[float]] = {}
    
    def _InitWeights(self, OutDim: int, InDim: int) -> List[List[float]]:
        """Initialize weight matrix with Xavier initialization"""
        import random
        scale = math.sqrt(2.0 / (InDim + OutDim))
        return [
            [random.gauss(0, scale) for _ in range(InDim)]
            for _ in range(OutDim)
        ]
    
    def _MatVecMul(self, M: List[List[float]], V: List[float]) -> List[float]:
        """Matrix-vector multiplication"""
        return [sum(m * v for m, v in zip(row, V)) for row in M]
    
    def _DotProduct(self, A: List[float], B: List[float]) -> float:
        """Dot product of two vectors"""
        return sum(a * b for a, b in zip(A, B))
    
    def _Softmax(self, X: List[float]) -> List[float]:
        """Softmax function"""
        if not X:
            return []
        max_val = max(X)
        exp_x = [math.exp(x - max_val) for x in X]
        sum_exp = sum(exp_x)
        return [e / sum_exp for e in exp_x]
    
    def _CosineSimilarity(self, A: List[float], B: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        if not A or not B:
            return 0.0
        
        dot = sum(a * b for a, b in zip(A, B))
        norm_a = math.sqrt(sum(a * a for a in A))
        norm_b = math.sqrt(sum(b * b for b in B))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot / (norm_a * norm_b)
    
    def SetKnowledge(
        self,
        EntityTriples: Dict[str, List[Tuple[str, str, str]]],
        EntityEmbeddings: Dict[str, List[float]],
        RelationEmbeddings: Dict[str, List[float]]
    ):
        """
        Set knowledge graph data.
        
        Args:
            EntityTriples: Mapping entity -> list of (subj, rel, obj) triples
            EntityEmbeddings: Entity embeddings from TransE
            RelationEmbeddings: Relation embeddings from TransE
        """
        self.EntityTriples = EntityTriples
        self.EntityEmbeddings = EntityEmbeddings
        self.RelationEmbeddings = RelationEmbeddings
    
    def _ComputeAttention(
        self,
        QueryEmbed: List[float],
        CandidateEmbeds: List[List[float]],
        Head: int
    ) -> List[float]:
        """
        Compute attention weights for one head.
        
        Args:
            QueryEmbed: Query embedding
            CandidateEmbeds: List of candidate embeddings
            Head: Which attention head
            
        Returns:
            Attention weights
        """
        if not CandidateEmbeds:
            return []
        
        # Project query
        Q = self._MatVecMul(self.QueryWeights[Head], QueryEmbed)
        
        # Project candidates as keys
        scores = []
        for cand_emb in CandidateEmbeds:
            K = self._MatVecMul(self.KeyWeights[Head], cand_emb)
            # Scaled dot-product attention
            score = self._DotProduct(Q, K) / math.sqrt(self.HeadDim)
            scores.append(score)
        
        # Softmax
        return self._Softmax(scores)
    
    def _MultiHeadAttention(
        self,
        QueryEmbed: List[float],
        CandidateEmbeds: List[List[float]]
    ) -> Tuple[List[float], List[float]]:
        """
        Compute multi-head attention.
        
        Args:
            QueryEmbed: Query embedding
            CandidateEmbeds: List of candidate embeddings
            
        Returns:
            Tuple of (output embedding, averaged attention weights)
        """
        if not CandidateEmbeds:
            return [0.0] * self.EmbedDim, []
        
        # Compute attention for each head
        all_head_weights = []
        all_head_outputs = []
        
        for h in range(self.NumHeads):
            weights = self._ComputeAttention(QueryEmbed, CandidateEmbeds, h)
            all_head_weights.append(weights)
            
            # Compute weighted sum of values
            output = [0.0] * self.HeadDim
            for w, cand_emb in zip(weights, CandidateEmbeds):
                V = self._MatVecMul(self.ValueWeights[h], cand_emb)
                for i in range(self.HeadDim):
                    output[i] += w * V[i]
            
            all_head_outputs.append(output)
        
        # Concatenate head outputs
        concat = []
        for head_out in all_head_outputs:
            concat.extend(head_out)
        
        # Project to output dimension
        output = self._MatVecMul(self.OutputWeights, concat)
        
        # Average attention weights across heads
        avg_weights = []
        for i in range(len(CandidateEmbeds)):
            avg = sum(all_head_weights[h][i] for h in range(self.NumHeads)) / self.NumHeads
            avg_weights.append(avg)
        
        return output, avg_weights
    
    def _GetNeighbors(self, Entity: str) -> List[Tuple[str, str, str]]:
        """Get all triples connected to an entity"""
        entity_lower = Entity.lower()
        
        # Check direct mapping
        if entity_lower in self.EntityTriples:
            return self.EntityTriples[entity_lower]
        
        # Check with underscores
        entity_underscore = entity_lower.replace(' ', '_')
        if entity_underscore in self.EntityTriples:
            return self.EntityTriples[entity_underscore]
        
        return []
    
    def _GetEntityEmbedding(self, Entity: str) -> List[float]:
        """Get embedding for an entity"""
        entity_lower = Entity.lower()
        
        if entity_lower in self.EntityEmbeddings:
            return self.EntityEmbeddings[entity_lower]
        
        entity_underscore = entity_lower.replace(' ', '_')
        if entity_underscore in self.EntityEmbeddings:
            return self.EntityEmbeddings[entity_underscore]
        
        return [0.0] * self.EmbedDim
    
    def Reason(
        self,
        QueryEmbedding: List[float],
        StartEntities: List[str],
        MaxCandidates: int = 10
    ) -> Tuple[List[CandidateAnswer], List[ReasoningPath]]:
        """
        Perform multi-hop reasoning.
        
        Args:
            QueryEmbedding: Embedding of the query
            StartEntities: Entities to start reasoning from
            MaxCandidates: Maximum candidate answers to return
            
        Returns:
            Tuple of (candidate answers, all reasoning paths)
        """
        all_paths: List[ReasoningPath] = []
        visited: Set[str] = set()
        candidates: Dict[str, CandidateAnswer] = {}
        
        # Initialize with start entities
        current_entities = [(e, ReasoningPath(), 1.0) for e in StartEntities]
        
        for hop in range(self.MaxHops):
            next_entities = []
            
            for entity, path, score in current_entities:
                entity_lower = entity.lower()
                
                if entity_lower in visited:
                    continue
                visited.add(entity_lower)
                
                # Get neighbors
                neighbors = self._GetNeighbors(entity)
                
                if not neighbors:
                    continue
                
                # Get embeddings for all neighbor entities
                neighbor_entities = []
                neighbor_embeddings = []
                neighbor_relations = []
                
                for subj, rel, obj in neighbors:
                    # Determine the "other" entity
                    other = obj if subj.lower() == entity_lower else subj
                    other_emb = self._GetEntityEmbedding(other)
                    
                    if other_emb and sum(abs(x) for x in other_emb) > 0:
                        neighbor_entities.append((other, rel, subj, obj))
                        neighbor_embeddings.append(other_emb)
                        neighbor_relations.append(rel)
                
                if not neighbor_embeddings:
                    continue
                
                # Compute attention over neighbors
                _, attention_weights = self._MultiHeadAttention(
                    QueryEmbedding, neighbor_embeddings
                )
                
                # Select top neighbors by attention
                scored_neighbors = list(zip(
                    neighbor_entities, attention_weights, neighbor_embeddings
                ))
                scored_neighbors.sort(key=lambda x: -x[1])
                
                # Take top neighbors
                for (other, rel, subj, obj), attn_weight, _ in scored_neighbors[:5]:
                    if attn_weight < 0.05:  # Skip very low attention
                        continue
                    
                    # Create new path
                    new_path = ReasoningPath(Hops=list(path.Hops))
                    new_path.AddHop(ReasoningHop(
                        FromEntity=entity,
                        Relation=rel,
                        ToEntity=other,
                        AttentionWeight=attn_weight,
                        HopNumber=hop + 1
                    ))
                    
                    new_score = score * attn_weight
                    
                    # Add to candidates
                    other_lower = other.lower()
                    if other_lower not in candidates or new_score > candidates[other_lower].Score:
                        candidates[other_lower] = CandidateAnswer(
                            Entity=other,
                            Score=new_score,
                            Path=new_path,
                            Explanation=""
                        )
                    
                    all_paths.append(new_path)
                    next_entities.append((other, new_path, new_score))
            
            current_entities = next_entities
            
            if not current_entities:
                break
        
        # Generate explanations and sort candidates
        result_candidates = list(candidates.values())
        
        for cand in result_candidates:
            cand.Explanation = self._GenerateExplanation(cand.Path)
        
        result_candidates.sort(key=lambda x: -x.Score)
        
        return result_candidates[:MaxCandidates], all_paths
    
    def _GenerateExplanation(self, Path: ReasoningPath) -> str:
        """
        Generate natural language explanation for reasoning path.
        
        Args:
            Path: Reasoning path
            
        Returns:
            Natural language explanation
        """
        if not Path.Hops:
            return "No reasoning path available."
        
        parts = []
        
        for i, hop in enumerate(Path.Hops):
            relation_readable = hop.Relation.replace('_', ' ')
            
            if i == 0:
                parts.append(f"{hop.FromEntity} {relation_readable} {hop.ToEntity}")
            else:
                parts.append(f"which {relation_readable} {hop.ToEntity}")
        
        explanation = ", ".join(parts)
        confidence = f" (confidence: {Path.TotalConfidence:.2%})"
        
        return explanation + confidence
    
    def ScoreAnswer(
        self,
        QueryEmbedding: List[float],
        AnswerEntity: str,
        VisitedEntities: Set[str]
    ) -> float:
        """
        Score a candidate answer.
        
        Args:
            QueryEmbedding: Query embedding
            AnswerEntity: Candidate answer entity
            VisitedEntities: Entities visited during reasoning
            
        Returns:
            Score for the answer
        """
        answer_emb = self._GetEntityEmbedding(AnswerEntity)
        
        if not answer_emb or sum(abs(x) for x in answer_emb) == 0:
            return 0.0
        
        # Base score: similarity to query
        similarity = self._CosineSimilarity(QueryEmbedding, answer_emb)
        
        # Boost if visited during reasoning
        visited_boost = 1.5 if AnswerEntity.lower() in VisitedEntities else 1.0
        
        # Compute relevance from attention
        _, attention = self._MultiHeadAttention(QueryEmbedding, [answer_emb])
        relevance = attention[0] if attention else 0.5
        
        # Combined score
        score = (similarity * 0.4 + relevance * 0.6) * visited_boost
        
        return max(0.0, min(1.0, score))
    
    def GetStats(self) -> Dict:
        """Get statistics about the reasoner"""
        return {
            "EmbedDim": self.EmbedDim,
            "NumHeads": self.NumHeads,
            "MaxHops": self.MaxHops,
            "KnownEntities": len(self.EntityEmbeddings),
            "KnownRelations": len(self.RelationEmbeddings),
            "EntityTriples": len(self.EntityTriples)
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing AttentionReasoner...")
    
    reasoner = AttentionReasoner(EmbedDim=100, NumHeads=4, MaxHops=3)
    
    # Set up mock knowledge
    import random
    
    entities = ['dog', 'mammal', 'animal', 'cat', 'pet', 'fur', 'bark', 'wolf']
    relations = ['is_a', 'has_property', 'related_to']
    
    # Create mock embeddings
    entity_embs = {e: [random.gauss(0, 0.1) for _ in range(100)] for e in entities}
    rel_embs = {r: [random.gauss(0, 0.1) for _ in range(100)] for r in relations}
    
    # Create mock triples
    triples = {
        'dog': [
            ('dog', 'is_a', 'mammal'),
            ('dog', 'is_a', 'pet'),
            ('dog', 'has_property', 'bark'),
            ('dog', 'related_to', 'wolf')
        ],
        'mammal': [
            ('mammal', 'is_a', 'animal'),
            ('dog', 'is_a', 'mammal'),
            ('cat', 'is_a', 'mammal')
        ],
        'cat': [
            ('cat', 'is_a', 'mammal'),
            ('cat', 'is_a', 'pet'),
            ('cat', 'has_property', 'fur')
        ]
    }
    
    reasoner.SetKnowledge(triples, entity_embs, rel_embs)
    
    # Test reasoning
    query_emb = entity_embs['dog']
    candidates, paths = reasoner.Reason(query_emb, ['dog'], MaxCandidates=5)
    
    print(f"\nFound {len(candidates)} candidates and {len(paths)} paths")
    
    for cand in candidates[:3]:
        print(f"\n  Candidate: {cand.Entity}")
        print(f"  Score: {cand.Score:.3f}")
        print(f"  Explanation: {cand.Explanation}")