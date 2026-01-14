"""
ContextGNN - Graph Neural Network for Context Propagation
Part of GroundZero AI Neural Pipeline

Features:
- 2-layer Graph Attention Network (GAT)
- 4 attention heads per layer
- Edge-typed attention (relation embeddings)
- Residual connections + layer normalization
- Bidirectional fusion with AttentionReasoner (GreaseLM-style)
"""

import math
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import random


@dataclass
class GNNNode:
    """Node in the graph neural network"""
    EntityID: str
    Embedding: List[float]
    UpdatedEmbedding: List[float]
    Neighbors: List[str]
    EdgeTypes: List[str]  # Relation for each neighbor


@dataclass
class GNNOutput:
    """Output from GNN processing"""
    NodeEmbeddings: Dict[str, List[float]]
    AttentionWeights: Dict[str, List[float]]
    Iterations: int


class ContextGNN:
    """
    Graph Attention Network for propagating context through knowledge graph.
    
    Uses GAT architecture with:
    - Multi-head attention (4 heads)
    - Edge-type aware attention (uses relation embeddings)
    - 2 message passing layers
    - Residual connections
    - Layer normalization
    """
    
    def __init__(
        self,
        EmbedDim: int = 100,
        NumHeads: int = 4,
        NumLayers: int = 2,
        DropoutRate: float = 0.1
    ):
        """
        Initialize ContextGNN.
        
        Args:
            EmbedDim: Dimension of embeddings
            NumHeads: Number of attention heads per layer
            NumLayers: Number of message passing layers
            DropoutRate: Dropout rate for regularization
        """
        self.EmbedDim = EmbedDim
        self.NumHeads = NumHeads
        self.HeadDim = EmbedDim // NumHeads
        self.NumLayers = NumLayers
        self.DropoutRate = DropoutRate
        
        # Initialize weights for each layer
        self.Layers = []
        for layer in range(NumLayers):
            layer_weights = {
                'W_query': [self._InitWeights(self.HeadDim, EmbedDim) for _ in range(NumHeads)],
                'W_key': [self._InitWeights(self.HeadDim, EmbedDim) for _ in range(NumHeads)],
                'W_value': [self._InitWeights(self.HeadDim, EmbedDim) for _ in range(NumHeads)],
                'W_edge': [self._InitWeights(self.HeadDim, EmbedDim) for _ in range(NumHeads)],
                'W_out': self._InitWeights(EmbedDim, EmbedDim),
                'LayerNorm_gamma': [1.0] * EmbedDim,
                'LayerNorm_beta': [0.0] * EmbedDim
            }
            self.Layers.append(layer_weights)
        
        # Final projection
        self.FinalProjection = self._InitWeights(EmbedDim, EmbedDim)
        
        # Knowledge graph data (set externally)
        self.EntityEmbeddings: Dict[str, List[float]] = {}
        self.RelationEmbeddings: Dict[str, List[float]] = {}
        self.Graph: Dict[str, List[Tuple[str, str]]] = {}  # entity -> [(neighbor, relation)]
    
    def _InitWeights(self, OutDim: int, InDim: int) -> List[List[float]]:
        """Initialize weight matrix with Xavier initialization"""
        scale = math.sqrt(2.0 / (InDim + OutDim))
        return [
            [random.gauss(0, scale) for _ in range(InDim)]
            for _ in range(OutDim)
        ]
    
    def _MatVecMul(self, M: List[List[float]], V: List[float]) -> List[float]:
        """Matrix-vector multiplication"""
        return [sum(m * v for m, v in zip(row, V)) for row in M]
    
    def _VecAdd(self, A: List[float], B: List[float]) -> List[float]:
        """Vector addition"""
        return [a + b for a, b in zip(A, B)]
    
    def _VecScale(self, V: List[float], Scale: float) -> List[float]:
        """Scale a vector"""
        return [v * Scale for v in V]
    
    def _LeakyReLU(self, X: float, Alpha: float = 0.2) -> float:
        """Leaky ReLU activation"""
        return X if X > 0 else Alpha * X
    
    def _Softmax(self, X: List[float]) -> List[float]:
        """Softmax function"""
        if not X:
            return []
        max_val = max(X)
        exp_x = [math.exp(x - max_val) for x in X]
        sum_exp = sum(exp_x)
        return [e / sum_exp for e in exp_x]
    
    def _LayerNorm(
        self,
        X: List[float],
        Gamma: List[float],
        Beta: List[float],
        Eps: float = 1e-6
    ) -> List[float]:
        """Layer normalization"""
        mean = sum(X) / len(X)
        var = sum((x - mean) ** 2 for x in X) / len(X)
        std = math.sqrt(var + Eps)
        
        normalized = [(x - mean) / std for x in X]
        return [g * n + b for g, n, b in zip(Gamma, normalized, Beta)]
    
    def _Dropout(self, X: List[float], Training: bool = True) -> List[float]:
        """Apply dropout during training"""
        if not Training or self.DropoutRate == 0:
            return X
        
        scale = 1.0 / (1.0 - self.DropoutRate)
        return [
            x * scale if random.random() > self.DropoutRate else 0.0
            for x in X
        ]
    
    def SetKnowledge(
        self,
        EntityEmbeddings: Dict[str, List[float]],
        RelationEmbeddings: Dict[str, List[float]],
        Graph: Dict[str, List[Tuple[str, str]]]
    ):
        """
        Set knowledge graph data.
        
        Args:
            EntityEmbeddings: Entity embeddings from TransE
            RelationEmbeddings: Relation embeddings from TransE
            Graph: Adjacency list - entity -> [(neighbor, relation), ...]
        """
        self.EntityEmbeddings = EntityEmbeddings
        self.RelationEmbeddings = RelationEmbeddings
        self.Graph = Graph
    
    def _GetEntityEmbedding(self, Entity: str) -> List[float]:
        """Get embedding for an entity"""
        entity_lower = Entity.lower()
        
        if entity_lower in self.EntityEmbeddings:
            return list(self.EntityEmbeddings[entity_lower])
        
        entity_underscore = entity_lower.replace(' ', '_')
        if entity_underscore in self.EntityEmbeddings:
            return list(self.EntityEmbeddings[entity_underscore])
        
        return [0.0] * self.EmbedDim
    
    def _GetRelationEmbedding(self, Relation: str) -> List[float]:
        """Get embedding for a relation"""
        rel_lower = Relation.lower()
        
        if rel_lower in self.RelationEmbeddings:
            return list(self.RelationEmbeddings[rel_lower])
        
        return [0.0] * self.EmbedDim
    
    def _GetNeighbors(self, Entity: str) -> List[Tuple[str, str]]:
        """Get neighbors for an entity"""
        entity_lower = Entity.lower()
        
        if entity_lower in self.Graph:
            return self.Graph[entity_lower]
        
        entity_underscore = entity_lower.replace(' ', '_')
        if entity_underscore in self.Graph:
            return self.Graph[entity_underscore]
        
        return []
    
    def _GATLayer(
        self,
        NodeEmbeddings: Dict[str, List[float]],
        LayerIdx: int,
        Training: bool = True
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        """
        Single GAT layer with edge-typed attention.
        
        Args:
            NodeEmbeddings: Current node embeddings
            LayerIdx: Which layer (for weights)
            Training: Whether in training mode
            
        Returns:
            Tuple of (updated embeddings, attention weights per node)
        """
        layer = self.Layers[LayerIdx]
        new_embeddings = {}
        attention_weights = {}
        
        for entity, emb in NodeEmbeddings.items():
            neighbors = self._GetNeighbors(entity)
            
            if not neighbors:
                # No neighbors - keep embedding with residual
                new_embeddings[entity] = emb
                attention_weights[entity] = []
                continue
            
            # Multi-head attention aggregation
            head_outputs = []
            head_attention = []
            
            for h in range(self.NumHeads):
                # Project node as query
                Q = self._MatVecMul(layer['W_query'][h], emb)
                
                # Compute attention over neighbors
                scores = []
                neighbor_values = []
                
                for neighbor, relation in neighbors:
                    neighbor_emb = NodeEmbeddings.get(
                        neighbor.lower(), 
                        self._GetEntityEmbedding(neighbor)
                    )
                    
                    if not neighbor_emb or sum(abs(x) for x in neighbor_emb) == 0:
                        continue
                    
                    # Project neighbor as key
                    K = self._MatVecMul(layer['W_key'][h], neighbor_emb)
                    
                    # Get edge (relation) embedding
                    edge_emb = self._GetRelationEmbedding(relation)
                    E = self._MatVecMul(layer['W_edge'][h], edge_emb)
                    
                    # Attention score = LeakyReLU(Q . (K + E))
                    combined = self._VecAdd(K, E)
                    score = sum(q * c for q, c in zip(Q, combined))
                    score = self._LeakyReLU(score / math.sqrt(self.HeadDim))
                    
                    scores.append(score)
                    
                    # Project neighbor as value
                    V = self._MatVecMul(layer['W_value'][h], neighbor_emb)
                    neighbor_values.append(V)
                
                if not scores:
                    head_outputs.append([0.0] * self.HeadDim)
                    head_attention.append([])
                    continue
                
                # Softmax attention weights
                attn = self._Softmax(scores)
                head_attention.append(attn)
                
                # Weighted sum of values
                head_out = [0.0] * self.HeadDim
                for weight, value in zip(attn, neighbor_values):
                    for i in range(self.HeadDim):
                        head_out[i] += weight * value[i]
                
                head_outputs.append(head_out)
            
            # Concatenate heads
            concat = []
            for head_out in head_outputs:
                concat.extend(head_out)
            
            # Output projection
            aggregated = self._MatVecMul(layer['W_out'], concat)
            
            # Residual connection
            aggregated = self._VecAdd(aggregated, emb)
            
            # Layer normalization
            aggregated = self._LayerNorm(
                aggregated,
                layer['LayerNorm_gamma'],
                layer['LayerNorm_beta']
            )
            
            # Dropout
            aggregated = self._Dropout(aggregated, Training)
            
            new_embeddings[entity] = aggregated
            
            # Average attention across heads
            avg_attention = []
            for i in range(len(neighbors)):
                head_vals = [ha[i] for ha in head_attention if len(ha) > i]
                if head_vals:
                    avg_attention.append(sum(head_vals) / len(head_vals))
            attention_weights[entity] = avg_attention
        
        return new_embeddings, attention_weights
    
    def Forward(
        self,
        SubgraphEntities: List[str],
        Training: bool = True
    ) -> GNNOutput:
        """
        Forward pass through the GNN.
        
        Args:
            SubgraphEntities: List of entities in the subgraph
            Training: Whether in training mode
            
        Returns:
            GNNOutput with updated embeddings
        """
        # Initialize node embeddings
        node_embeddings: Dict[str, List[float]] = {}
        
        for entity in SubgraphEntities:
            emb = self._GetEntityEmbedding(entity)
            if emb and sum(abs(x) for x in emb) > 0:
                node_embeddings[entity.lower()] = emb
        
        if not node_embeddings:
            return GNNOutput(
                NodeEmbeddings={},
                AttentionWeights={},
                Iterations=0
            )
        
        # Message passing layers
        all_attention = {}
        
        for layer_idx in range(self.NumLayers):
            node_embeddings, attention = self._GATLayer(
                node_embeddings, layer_idx, Training
            )
            all_attention.update(attention)
        
        # Final projection
        for entity in node_embeddings:
            node_embeddings[entity] = self._MatVecMul(
                self.FinalProjection,
                node_embeddings[entity]
            )
        
        return GNNOutput(
            NodeEmbeddings=node_embeddings,
            AttentionWeights=all_attention,
            Iterations=self.NumLayers
        )
    
    def FuseWithReasoner(
        self,
        GNNEmbeddings: Dict[str, List[float]],
        ReasonerEmbeddings: Dict[str, List[float]],
        FusionWeight: float = 0.5
    ) -> Dict[str, List[float]]:
        """
        Fuse GNN embeddings with AttentionReasoner embeddings.
        
        This implements GreaseLM-style bidirectional fusion where
        both GNN and Reasoner inform each other.
        
        Args:
            GNNEmbeddings: Embeddings from GNN
            ReasonerEmbeddings: Embeddings from AttentionReasoner
            FusionWeight: Weight for GNN (1 - weight for Reasoner)
            
        Returns:
            Fused embeddings
        """
        fused = {}
        
        # Get all entities from both
        all_entities = set(GNNEmbeddings.keys()) | set(ReasonerEmbeddings.keys())
        
        for entity in all_entities:
            gnn_emb = GNNEmbeddings.get(entity, [0.0] * self.EmbedDim)
            reasoner_emb = ReasonerEmbeddings.get(entity, [0.0] * self.EmbedDim)
            
            # Check if both have valid embeddings
            gnn_valid = sum(abs(x) for x in gnn_emb) > 0
            reasoner_valid = sum(abs(x) for x in reasoner_emb) > 0
            
            if gnn_valid and reasoner_valid:
                # Blend both
                fused[entity] = [
                    FusionWeight * g + (1 - FusionWeight) * r
                    for g, r in zip(gnn_emb, reasoner_emb)
                ]
            elif gnn_valid:
                fused[entity] = gnn_emb
            elif reasoner_valid:
                fused[entity] = reasoner_emb
        
        return fused
    
    def GetContextEmbedding(
        self,
        SubgraphEntities: List[str],
        QueryEmbedding: List[float]
    ) -> List[float]:
        """
        Get a single context embedding for a subgraph.
        
        Uses attention over GNN-processed nodes weighted by
        query similarity.
        
        Args:
            SubgraphEntities: Entities in the subgraph
            QueryEmbedding: Query embedding for weighting
            
        Returns:
            Context embedding
        """
        # Get GNN outputs
        output = self.Forward(SubgraphEntities, Training=False)
        
        if not output.NodeEmbeddings:
            return [0.0] * self.EmbedDim
        
        # Compute attention weights based on query similarity
        weights = []
        embeddings = []
        
        for entity, emb in output.NodeEmbeddings.items():
            # Cosine similarity with query
            dot = sum(q * e for q, e in zip(QueryEmbedding, emb))
            norm_q = math.sqrt(sum(q * q for q in QueryEmbedding))
            norm_e = math.sqrt(sum(e * e for e in emb))
            
            if norm_q > 0 and norm_e > 0:
                sim = dot / (norm_q * norm_e)
            else:
                sim = 0.0
            
            weights.append(sim)
            embeddings.append(emb)
        
        # Softmax weights
        if weights:
            weights = self._Softmax(weights)
        
        # Weighted sum
        context = [0.0] * self.EmbedDim
        for weight, emb in zip(weights, embeddings):
            for i in range(self.EmbedDim):
                context[i] += weight * emb[i]
        
        return context
    
    def GetStats(self) -> Dict:
        """Get statistics about the GNN"""
        return {
            "EmbedDim": self.EmbedDim,
            "NumHeads": self.NumHeads,
            "NumLayers": self.NumLayers,
            "DropoutRate": self.DropoutRate,
            "KnownEntities": len(self.EntityEmbeddings),
            "KnownRelations": len(self.RelationEmbeddings),
            "GraphNodes": len(self.Graph)
        }


# =============================================================================
# TESTING  
# =============================================================================

if __name__ == "__main__":
    print("Testing ContextGNN...")
    
    gnn = ContextGNN(EmbedDim=100, NumHeads=4, NumLayers=2)
    
    # Set up mock knowledge
    entities = ['dog', 'mammal', 'animal', 'cat', 'pet', 'fur', 'bark']
    relations = ['is_a', 'has_property', 'related_to']
    
    # Create mock embeddings
    entity_embs = {e: [random.gauss(0, 0.1) for _ in range(100)] for e in entities}
    rel_embs = {r: [random.gauss(0, 0.1) for _ in range(100)] for r in relations}
    
    # Create mock graph
    graph = {
        'dog': [('mammal', 'is_a'), ('pet', 'is_a'), ('bark', 'has_property')],
        'mammal': [('animal', 'is_a'), ('dog', 'is_a'), ('cat', 'is_a')],
        'cat': [('mammal', 'is_a'), ('pet', 'is_a'), ('fur', 'has_property')],
        'pet': [('dog', 'is_a'), ('cat', 'is_a')],
        'animal': [('mammal', 'is_a')]
    }
    
    gnn.SetKnowledge(entity_embs, rel_embs, graph)
    
    # Test forward pass
    output = gnn.Forward(['dog', 'mammal', 'cat', 'animal'], Training=False)
    
    print(f"\nProcessed {len(output.NodeEmbeddings)} nodes in {output.Iterations} iterations")
    
    for entity, emb in list(output.NodeEmbeddings.items())[:3]:
        norm = math.sqrt(sum(e * e for e in emb))
        print(f"  {entity}: embedding norm = {norm:.3f}")
    
    # Test context embedding
    query_emb = entity_embs['dog']
    context = gnn.GetContextEmbedding(['dog', 'mammal', 'animal'], query_emb)
    context_norm = math.sqrt(sum(c * c for c in context))
    print(f"\nContext embedding norm: {context_norm:.3f}")