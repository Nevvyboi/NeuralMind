"""
NeuralPipeline - Full Neurosymbolic Pipeline Integration
Part of GroundZero AI Neural Pipeline

Features:
- RelevanceScorer: BFS subgraph extraction (2 hops, 50 nodes max)
- Full pipeline orchestration: TinyLM → Scorer → [Reasoner ↔ GNN] × 2 iterations
- Type-specific answer generation
- Initialize() connects to TransE embeddings from NeuralEngine
"""

import math
import time
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import deque

from .tiny_lm import TinyLM, QueryAnalysis, QuestionType
from .attention_reasoner import AttentionReasoner, CandidateAnswer, ReasoningPath, ReasoningHop
from .context_gnn import ContextGNN, GNNOutput


@dataclass
class PipelineResponse:
    """Response from the neural pipeline"""
    Query: str
    Answer: str
    Confidence: float
    QuestionType: str
    ReasoningPath: List[ReasoningHop]
    CandidateAnswers: List[Tuple[str, float]]
    ProcessingTimeMs: float
    Explanation: str


@dataclass
class RelevantSubgraph:
    """Subgraph extracted for a query"""
    Entities: List[str]
    Triples: List[Tuple[str, str, str]]
    RelevanceScores: Dict[str, float]
    HopsFromQuery: Dict[str, int]


class RelevanceScorer:
    """
    Scores and extracts relevant subgraph for a query.
    
    Uses BFS from query entities to find connected nodes,
    then scores by cosine similarity to query embedding.
    """
    
    def __init__(self, MaxHops: int = 2, MaxNodes: int = 50):
        """
        Initialize RelevanceScorer.
        
        Args:
            MaxHops: Maximum BFS hops from query entities
            MaxNodes: Maximum nodes in subgraph
        """
        self.MaxHops = MaxHops
        self.MaxNodes = MaxNodes
        
        # Knowledge graph data (set externally)
        self.EntityTriples: Dict[str, List[Tuple[str, str, str]]] = {}
        self.EntityEmbeddings: Dict[str, List[float]] = {}
    
    def SetKnowledge(
        self,
        EntityTriples: Dict[str, List[Tuple[str, str, str]]],
        EntityEmbeddings: Dict[str, List[float]]
    ):
        """Set knowledge graph data"""
        self.EntityTriples = EntityTriples
        self.EntityEmbeddings = EntityEmbeddings
    
    def _CosineSimilarity(self, A: List[float], B: List[float]) -> float:
        """Compute cosine similarity"""
        if not A or not B:
            return 0.0
        
        dot = sum(a * b for a, b in zip(A, B))
        norm_a = math.sqrt(sum(a * a for a in A))
        norm_b = math.sqrt(sum(b * b for b in B))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot / (norm_a * norm_b)
    
    def _GetNeighbors(self, Entity: str) -> List[Tuple[str, str, str]]:
        """Get triples for an entity"""
        entity_lower = Entity.lower()
        
        if entity_lower in self.EntityTriples:
            return self.EntityTriples[entity_lower]
        
        entity_underscore = entity_lower.replace(' ', '_')
        if entity_underscore in self.EntityTriples:
            return self.EntityTriples[entity_underscore]
        
        return []
    
    def _GetEmbedding(self, Entity: str) -> List[float]:
        """Get embedding for entity"""
        entity_lower = Entity.lower()
        
        if entity_lower in self.EntityEmbeddings:
            return self.EntityEmbeddings[entity_lower]
        
        entity_underscore = entity_lower.replace(' ', '_')
        if entity_underscore in self.EntityEmbeddings:
            return self.EntityEmbeddings[entity_underscore]
        
        return []
    
    def ExtractSubgraph(
        self,
        QueryEntities: List[str],
        QueryEmbedding: List[float]
    ) -> RelevantSubgraph:
        """
        Extract relevant subgraph using BFS and relevance scoring.
        
        Args:
            QueryEntities: Starting entities from query
            QueryEmbedding: Query embedding for relevance scoring
            
        Returns:
            RelevantSubgraph with entities, triples, and scores
        """
        visited: Set[str] = set()
        entities: List[str] = []
        triples: List[Tuple[str, str, str]] = []
        relevance_scores: Dict[str, float] = {}
        hops_from_query: Dict[str, int] = {}
        
        # BFS queue: (entity, hop_count)
        queue = deque()
        
        for entity in QueryEntities:
            entity_lower = entity.lower()
            if entity_lower not in visited:
                queue.append((entity_lower, 0))
                visited.add(entity_lower)
                hops_from_query[entity_lower] = 0
        
        while queue and len(entities) < self.MaxNodes:
            entity, hop = queue.popleft()
            
            # Add entity
            entities.append(entity)
            
            # Score relevance
            emb = self._GetEmbedding(entity)
            if emb:
                relevance_scores[entity] = self._CosineSimilarity(QueryEmbedding, emb)
            else:
                relevance_scores[entity] = 0.0
            
            # Get neighbors if within hop limit
            if hop < self.MaxHops:
                neighbors = self._GetNeighbors(entity)
                
                for subj, rel, obj in neighbors:
                    triples.append((subj, rel, obj))
                    
                    # Add unvisited neighbors to queue
                    for neighbor in [subj.lower(), obj.lower()]:
                        if neighbor not in visited and len(entities) + len(queue) < self.MaxNodes:
                            visited.add(neighbor)
                            queue.append((neighbor, hop + 1))
                            hops_from_query[neighbor] = hop + 1
        
        # Deduplicate triples
        unique_triples = list(set(triples))
        
        return RelevantSubgraph(
            Entities=entities,
            Triples=unique_triples,
            RelevanceScores=relevance_scores,
            HopsFromQuery=hops_from_query
        )


class NeuralPipeline:
    """
    Complete neural pipeline for question answering.
    
    Pipeline flow:
    Query → TinyLM (entities, embedding, type) 
          → RelevanceScorer (subgraph from TransE)
          → [Fusion Loop 2x: AttentionReasoner ↔ ContextGNN]
          → Answer Generator
          → Response + Path
    """
    
    def __init__(self, EmbedDim: int = 100):
        """
        Initialize NeuralPipeline.
        
        Args:
            EmbedDim: Dimension of embeddings (should match TransE)
        """
        self.EmbedDim = EmbedDim
        
        # Components
        self.TinyLM = TinyLM(EmbedDim=EmbedDim)
        self.Reasoner = AttentionReasoner(EmbedDim=EmbedDim, NumHeads=4, MaxHops=3)
        self.GNN = ContextGNN(EmbedDim=EmbedDim, NumHeads=4, NumLayers=2)
        self.Scorer = RelevanceScorer(MaxHops=2, MaxNodes=50)
        
        # Configuration
        self.FusionIterations = 2
        self.MinConfidence = 0.1
        
        # State
        self.IsInitialized = False
        self.Stats = {
            "QueriesProcessed": 0,
            "AvgProcessingTimeMs": 0,
            "CacheHits": 0
        }
    
    def Initialize(
        self,
        EntityEmbeddings: Dict[str, List[float]],
        RelationEmbeddings: Dict[str, List[float]],
        EntityTriples: Dict[str, List[Tuple[str, str, str]]],
        Graph: Dict[str, List[Tuple[str, str]]]
    ):
        """
        Initialize pipeline with knowledge graph data.
        
        Args:
            EntityEmbeddings: Entity embeddings from TransE
            RelationEmbeddings: Relation embeddings from TransE
            EntityTriples: Mapping entity -> list of triples
            Graph: Adjacency list for GNN
        """
        # Align TinyLM with TransE
        self.TinyLM.AlignWithTransE(EntityEmbeddings)
        
        # Set up Reasoner
        self.Reasoner.SetKnowledge(EntityTriples, EntityEmbeddings, RelationEmbeddings)
        
        # Set up GNN
        self.GNN.SetKnowledge(EntityEmbeddings, RelationEmbeddings, Graph)
        
        # Set up Scorer
        self.Scorer.SetKnowledge(EntityTriples, EntityEmbeddings)
        
        self.IsInitialized = True
    
    def _GenerateAnswer(
        self,
        Analysis: QueryAnalysis,
        Candidates: List[CandidateAnswer],
        Subgraph: RelevantSubgraph
    ) -> Tuple[str, str]:
        """
        Generate answer based on question type and candidates.
        
        Args:
            Analysis: Query analysis
            Candidates: Candidate answers from reasoning
            Subgraph: Relevant subgraph
            
        Returns:
            Tuple of (answer, explanation)
        """
        if not Candidates:
            return "I couldn't find relevant information.", "No reasoning path found."
        
        top_candidate = Candidates[0]
        qtype = Analysis.QuestionType
        
        # Type-specific answer generation
        if qtype == QuestionType.FACTUAL:
            answer = self._GenerateFactualAnswer(Analysis, top_candidate)
        elif qtype == QuestionType.CAUSAL:
            answer = self._GenerateCausalAnswer(Analysis, top_candidate, Subgraph)
        elif qtype == QuestionType.DEFINITIONAL:
            answer = self._GenerateDefinitionalAnswer(Analysis, top_candidate, Subgraph)
        elif qtype == QuestionType.COMPARATIVE:
            answer = self._GenerateComparativeAnswer(Analysis, Candidates, Subgraph)
        elif qtype == QuestionType.COUNTERFACTUAL:
            answer = self._GenerateCounterfactualAnswer(Analysis, top_candidate)
        elif qtype == QuestionType.PROCEDURAL:
            answer = self._GenerateProceduralAnswer(Analysis, Candidates, Subgraph)
        else:
            answer = f"Based on my knowledge, the answer relates to: {top_candidate.Entity}"
        
        return answer, top_candidate.Explanation
    
    def _GenerateFactualAnswer(
        self,
        Analysis: QueryAnalysis,
        Candidate: CandidateAnswer
    ) -> str:
        """Generate answer for factual questions"""
        if Candidate.Path.Hops:
            first_hop = Candidate.Path.Hops[0]
            relation = first_hop.Relation.replace('_', ' ')
            
            if 'is_a' in relation:
                return f"{first_hop.FromEntity.title()} is a type of {first_hop.ToEntity}."
            elif 'has' in relation:
                return f"{first_hop.FromEntity.title()} {relation} {first_hop.ToEntity}."
            else:
                return f"{first_hop.FromEntity.title()} {relation} {first_hop.ToEntity}."
        
        return f"The answer is: {Candidate.Entity}"
    
    def _GenerateCausalAnswer(
        self,
        Analysis: QueryAnalysis,
        Candidate: CandidateAnswer,
        Subgraph: RelevantSubgraph
    ) -> str:
        """Generate answer for causal questions"""
        if Candidate.Path.Hops:
            # Build causal chain
            chain_parts = []
            for hop in Candidate.Path.Hops:
                chain_parts.append(f"{hop.FromEntity} → {hop.ToEntity}")
            
            chain = " → ".join(chain_parts)
            
            return f"The causal relationship is: {chain}. This happens because {Candidate.Path.Hops[0].FromEntity} {Candidate.Path.Hops[0].Relation.replace('_', ' ')} {Candidate.Path.Hops[0].ToEntity}."
        
        return f"This is caused by: {Candidate.Entity}"
    
    def _GenerateDefinitionalAnswer(
        self,
        Analysis: QueryAnalysis,
        Candidate: CandidateAnswer,
        Subgraph: RelevantSubgraph
    ) -> str:
        """Generate answer for definitional questions"""
        entity = Analysis.Entities[0] if Analysis.Entities else Analysis.Keywords[0] if Analysis.Keywords else "the concept"
        
        # Collect defining attributes
        attributes = []
        for subj, rel, obj in Subgraph.Triples[:5]:
            if subj.lower() == entity.lower():
                attributes.append(f"{rel.replace('_', ' ')} {obj}")
        
        if attributes:
            attr_str = ", ".join(attributes[:3])
            return f"{entity.title()} is defined as something that {attr_str}."
        
        if Candidate.Path.Hops:
            hop = Candidate.Path.Hops[0]
            return f"{entity.title()} {hop.Relation.replace('_', ' ')} {hop.ToEntity}."
        
        return f"{entity.title()} refers to: {Candidate.Entity}"
    
    def _GenerateComparativeAnswer(
        self,
        Analysis: QueryAnalysis,
        Candidates: List[CandidateAnswer],
        Subgraph: RelevantSubgraph
    ) -> str:
        """Generate answer for comparative questions"""
        entities = Analysis.Entities[:2] if len(Analysis.Entities) >= 2 else Analysis.Keywords[:2]
        
        if len(entities) < 2:
            return "I need two entities to compare."
        
        # Find common and different properties
        entity1, entity2 = entities[0], entities[1]
        props1 = set()
        props2 = set()
        
        for subj, rel, obj in Subgraph.Triples:
            if subj.lower() == entity1.lower():
                props1.add((rel, obj))
            if subj.lower() == entity2.lower():
                props2.add((rel, obj))
        
        common = props1 & props2
        diff1 = props1 - props2
        diff2 = props2 - props1
        
        answer_parts = []
        
        if common:
            common_str = ", ".join([f"{r.replace('_', ' ')} {o}" for r, o in list(common)[:2]])
            answer_parts.append(f"Both {entity1} and {entity2} share: {common_str}")
        
        if diff1:
            diff1_str = ", ".join([f"{r.replace('_', ' ')} {o}" for r, o in list(diff1)[:2]])
            answer_parts.append(f"{entity1.title()} specifically: {diff1_str}")
        
        if diff2:
            diff2_str = ", ".join([f"{r.replace('_', ' ')} {o}" for r, o in list(diff2)[:2]])
            answer_parts.append(f"{entity2.title()} specifically: {diff2_str}")
        
        if answer_parts:
            return " ".join(answer_parts)
        
        return f"I found limited comparison data between {entity1} and {entity2}."
    
    def _GenerateCounterfactualAnswer(
        self,
        Analysis: QueryAnalysis,
        Candidate: CandidateAnswer
    ) -> str:
        """Generate answer for counterfactual questions"""
        return f"In that hypothetical scenario, the most likely outcome would involve: {Candidate.Entity}. {Candidate.Explanation}"
    
    def _GenerateProceduralAnswer(
        self,
        Analysis: QueryAnalysis,
        Candidates: List[CandidateAnswer],
        Subgraph: RelevantSubgraph
    ) -> str:
        """Generate answer for procedural questions"""
        if Candidates:
            steps = []
            for i, cand in enumerate(Candidates[:3], 1):
                steps.append(f"{i}. {cand.Entity}")
            
            return "The process involves: " + " → ".join(steps)
        
        return "I don't have enough information to describe the procedure."
    
    def Process(self, Query: str) -> PipelineResponse:
        """
        Process a query through the full pipeline.
        
        Args:
            Query: User's question
            
        Returns:
            PipelineResponse with answer and metadata
        """
        start_time = time.time()
        
        if not self.IsInitialized:
            return PipelineResponse(
                Query=Query,
                Answer="Pipeline not initialized. Please wait for system startup.",
                Confidence=0.0,
                QuestionType="unknown",
                ReasoningPath=[],
                CandidateAnswers=[],
                ProcessingTimeMs=0,
                Explanation="System not ready"
            )
        
        # Step 1: Analyze query with TinyLM
        analysis = self.TinyLM.Analyze(Query)
        
        # Get query entities and keywords as starting points
        start_entities = analysis.Entities + analysis.Keywords
        
        if not start_entities:
            # Fall back to all words
            start_entities = [w for w in analysis.Tokens if len(w) > 2]
        
        # Step 2: Extract relevant subgraph
        subgraph = self.Scorer.ExtractSubgraph(start_entities, analysis.Embedding)
        
        # Step 3: Run reasoning with fusion
        all_candidates = []
        reasoner_embeddings: Dict[str, List[float]] = {}
        
        for iteration in range(self.FusionIterations):
            # Run AttentionReasoner
            candidates, paths = self.Reasoner.Reason(
                analysis.Embedding,
                start_entities,
                MaxCandidates=10
            )
            
            # Get GNN context
            gnn_output = self.GNN.Forward(subgraph.Entities, Training=False)
            
            # Fuse embeddings (GreaseLM-style)
            for cand in candidates:
                entity_lower = cand.Entity.lower()
                reasoner_embeddings[entity_lower] = self.Reasoner._GetEntityEmbedding(cand.Entity)
            
            fused_embeddings = self.GNN.FuseWithReasoner(
                gnn_output.NodeEmbeddings,
                reasoner_embeddings,
                FusionWeight=0.5
            )
            
            all_candidates.extend(candidates)
        
        # Deduplicate and sort candidates
        seen = set()
        unique_candidates = []
        for cand in all_candidates:
            if cand.Entity.lower() not in seen:
                seen.add(cand.Entity.lower())
                unique_candidates.append(cand)
        
        unique_candidates.sort(key=lambda x: -x.Score)
        
        # Step 4: Generate answer
        answer, explanation = self._GenerateAnswer(analysis, unique_candidates, subgraph)
        
        # Calculate confidence
        confidence = unique_candidates[0].Score if unique_candidates else 0.0
        confidence = min(confidence * analysis.Confidence, 1.0)
        
        # Build response
        processing_time = (time.time() - start_time) * 1000
        
        reasoning_path = []
        if unique_candidates and unique_candidates[0].Path.Hops:
            reasoning_path = unique_candidates[0].Path.Hops
        
        candidate_tuples = [(c.Entity, c.Score) for c in unique_candidates[:5]]
        
        # Update stats
        self.Stats["QueriesProcessed"] += 1
        self.Stats["AvgProcessingTimeMs"] = (
            (self.Stats["AvgProcessingTimeMs"] * (self.Stats["QueriesProcessed"] - 1) + processing_time)
            / self.Stats["QueriesProcessed"]
        )
        
        return PipelineResponse(
            Query=Query,
            Answer=answer,
            Confidence=confidence,
            QuestionType=analysis.QuestionType.value,
            ReasoningPath=reasoning_path,
            CandidateAnswers=candidate_tuples,
            ProcessingTimeMs=processing_time,
            Explanation=explanation
        )
    
    def GetStats(self) -> Dict:
        """Get pipeline statistics"""
        return {
            "IsInitialized": self.IsInitialized,
            "FusionIterations": self.FusionIterations,
            "TinyLM": self.TinyLM.GetStats(),
            "Reasoner": self.Reasoner.GetStats(),
            "GNN": self.GNN.GetStats(),
            "Processing": self.Stats
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import random
    
    print("Testing NeuralPipeline...")
    
    pipeline = NeuralPipeline(EmbedDim=100)
    
    # Create mock knowledge graph
    entities = ['dog', 'mammal', 'animal', 'cat', 'pet', 'bark', 'fur', 'wolf', 'bird', 'fly']
    relations = ['is_a', 'has_property', 'related_to', 'causes']
    
    entity_embs = {e: [random.gauss(0, 0.1) for _ in range(100)] for e in entities}
    rel_embs = {r: [random.gauss(0, 0.1) for _ in range(100)] for r in relations}
    
    entity_triples = {
        'dog': [('dog', 'is_a', 'mammal'), ('dog', 'is_a', 'pet'), ('dog', 'has_property', 'bark')],
        'cat': [('cat', 'is_a', 'mammal'), ('cat', 'is_a', 'pet'), ('cat', 'has_property', 'fur')],
        'mammal': [('mammal', 'is_a', 'animal'), ('dog', 'is_a', 'mammal'), ('cat', 'is_a', 'mammal')],
        'bird': [('bird', 'is_a', 'animal'), ('bird', 'has_property', 'fly')]
    }
    
    graph = {
        'dog': [('mammal', 'is_a'), ('pet', 'is_a'), ('bark', 'has_property')],
        'cat': [('mammal', 'is_a'), ('pet', 'is_a'), ('fur', 'has_property')],
        'mammal': [('animal', 'is_a'), ('dog', 'is_a'), ('cat', 'is_a')],
        'bird': [('animal', 'is_a'), ('fly', 'has_property')]
    }
    
    # Initialize pipeline
    pipeline.Initialize(entity_embs, rel_embs, entity_triples, graph)
    
    # Test queries
    test_queries = [
        "What is a dog?",
        "Why do dogs bark?",
        "Compare cats and dogs"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = pipeline.Process(query)
        print(f"  Answer: {response.Answer}")
        print(f"  Confidence: {response.Confidence:.2%}")
        print(f"  Type: {response.QuestionType}")
        print(f"  Time: {response.ProcessingTimeMs:.1f}ms")