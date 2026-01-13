"""
Neural Engine - Knowledge Graph Embeddings
==========================================

A neural network that learns from knowledge graphs using TransE embeddings.

Capabilities:
- UNDERSTAND: Learn semantic embeddings of concepts
- GENERALIZE: Predict facts about unseen entities
- LEARN: Continuously improve from new knowledge
- REASON: Infer missing relationships  
- CREATE: Generate novel hypotheses

Based on:
- TransE (Bordes et al., 2013)
- Neurosymbolic AI research (DeLong et al., 2024)

Author: GroundZero AI
Version: 3.0.0
"""

import math
import json
import random
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import time

# Try to use numpy for faster computation
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class NeuralStats:
    """Statistics about the neural network"""
    TotalEntities: int = 0
    TotalRelations: int = 0
    TotalTriples: int = 0
    EmbeddingDim: int = 100
    TrainingEpochs: int = 0
    LastLoss: float = 0.0
    Predictions: int = 0
    Hypotheses: int = 0
    IsTrained: bool = False
    TrainingTime: float = 0.0
    
    def ToDict(self) -> Dict:
        return {
            "TotalEntities": self.TotalEntities,
            "TotalRelations": self.TotalRelations,
            "TotalTriples": self.TotalTriples,
            "EmbeddingDim": self.EmbeddingDim,
            "TrainingEpochs": self.TrainingEpochs,
            "LastLoss": round(self.LastLoss, 4),
            "Predictions": self.Predictions,
            "Hypotheses": self.Hypotheses,
            "IsTrained": self.IsTrained,
            "TrainingTime": round(self.TrainingTime, 2)
        }


@dataclass
class Prediction:
    """A predicted fact with confidence"""
    Head: str
    Relation: str
    Tail: str
    Confidence: float
    
    def ToDict(self) -> Dict:
        return {
            "Head": self.Head,
            "Relation": self.Relation,
            "Tail": self.Tail,
            "Confidence": round(self.Confidence, 3)
        }


# =============================================================================
# VECTOR OPERATIONS
# =============================================================================

class Vector:
    """Simple vector class for when numpy isn't available"""
    
    def __init__(self, data: List[float]):
        self.data = list(data)
        self.dim = len(data)
    
    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])
    
    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])
    
    def __mul__(self, scalar):
        return Vector([a * scalar for a in self.data])
    
    def __truediv__(self, scalar):
        return Vector([a / scalar for a in self.data])
    
    def dot(self, other) -> float:
        return sum(a * b for a, b in zip(self.data, other.data))
    
    def norm(self) -> float:
        return math.sqrt(sum(a * a for a in self.data))
    
    def normalize(self):
        n = self.norm()
        if n > 0:
            self.data = [a / n for a in self.data]
        return self
    
    def distance(self, other) -> float:
        return (self - other).norm()
    
    def cosine_similarity(self, other) -> float:
        dot = self.dot(other)
        norm_product = self.norm() * other.norm()
        if norm_product == 0:
            return 0.0
        return dot / norm_product
    
    def to_list(self) -> List[float]:
        return self.data.copy()
    
    @staticmethod
    def random(dim: int, scale: float = 0.1):
        return Vector([random.gauss(0, scale) for _ in range(dim)])
    
    @staticmethod
    def zeros(dim: int):
        return Vector([0.0] * dim)


# =============================================================================
# NEURAL ENGINE (TransE-based)
# =============================================================================

class NeuralEngine:
    """
    Neural network for knowledge graph embeddings.
    
    Uses TransE: For (head, relation, tail), we learn embeddings where:
        head_vector + relation_vector ≈ tail_vector
    
    This allows:
    - Link prediction (predict missing facts)
    - Similarity search (find related concepts)
    - Generalization (reason about unseen entities)
    - Hypothesis generation (create new ideas)
    """
    
    def __init__(
        self,
        EmbeddingDim: int = 100,
        LearningRate: float = 0.01,
        Margin: float = 1.0,
        NegativeSamples: int = 5
    ):
        """
        Initialize the neural engine.
        
        Args:
            EmbeddingDim: Size of embedding vectors
            LearningRate: Learning rate for training
            Margin: Margin for ranking loss
            NegativeSamples: Number of negative samples per positive
        """
        self.Dim = EmbeddingDim
        self.LearningRate = LearningRate
        self.Margin = Margin
        self.NegativeSamples = NegativeSamples
        
        # Embeddings storage
        self.EntityEmbeddings: Dict[str, List[float]] = {}
        self.RelationEmbeddings: Dict[str, List[float]] = {}
        
        # Entity/Relation indices
        self.Entity2Idx: Dict[str, int] = {}
        self.Idx2Entity: Dict[int, str] = {}
        self.Relation2Idx: Dict[str, int] = {}
        self.Idx2Relation: Dict[int, str] = {}
        
        # Knowledge triples
        self.Triples: List[Tuple[str, str, str]] = []
        self.TripleSet: Set[Tuple[str, str, str]] = set()
        
        # Training history
        self.LossHistory: List[float] = []
        self.Stats = NeuralStats(EmbeddingDim=EmbeddingDim)
        
        # Numpy arrays (if available)
        self.EntityMatrix = None
        self.RelationMatrix = None
    
    def _GetOrCreateEntityEmbedding(self, entity: str) -> List[float]:
        """Get or create embedding for an entity"""
        if entity not in self.EntityEmbeddings:
            # Create random embedding
            if HAS_NUMPY:
                emb = (np.random.randn(self.Dim) * 0.1).tolist()
            else:
                emb = [random.gauss(0, 0.1) for _ in range(self.Dim)]
            
            # Normalize
            norm = math.sqrt(sum(x*x for x in emb))
            if norm > 0:
                emb = [x / norm for x in emb]
            
            self.EntityEmbeddings[entity] = emb
            
            # Update index
            idx = len(self.Entity2Idx)
            self.Entity2Idx[entity] = idx
            self.Idx2Entity[idx] = entity
        
        return self.EntityEmbeddings[entity]
    
    def _GetOrCreateRelationEmbedding(self, relation: str) -> List[float]:
        """Get or create embedding for a relation"""
        if relation not in self.RelationEmbeddings:
            if HAS_NUMPY:
                emb = (np.random.randn(self.Dim) * 0.1).tolist()
            else:
                emb = [random.gauss(0, 0.1) for _ in range(self.Dim)]
            
            self.RelationEmbeddings[relation] = emb
            
            # Update index
            idx = len(self.Relation2Idx)
            self.Relation2Idx[relation] = idx
            self.Idx2Relation[idx] = relation
        
        return self.RelationEmbeddings[relation]
    
    def AddTriple(self, head: str, relation: str, tail: str):
        """Add a knowledge triple to the engine"""
        triple = (head, relation, tail)
        
        if triple not in self.TripleSet:
            self.Triples.append(triple)
            self.TripleSet.add(triple)
            
            # Ensure embeddings exist
            self._GetOrCreateEntityEmbedding(head)
            self._GetOrCreateEntityEmbedding(tail)
            self._GetOrCreateRelationEmbedding(relation)
            
            # Update stats
            self.Stats.TotalTriples = len(self.Triples)
            self.Stats.TotalEntities = len(self.EntityEmbeddings)
            self.Stats.TotalRelations = len(self.RelationEmbeddings)
    
    def AddTriples(self, triples: List[Tuple[str, str, str]]):
        """Add multiple triples at once"""
        for h, r, t in triples:
            self.AddTriple(h, r, t)
    
    def _Score(self, head: str, relation: str, tail: str) -> float:
        """
        Score a triple using TransE.
        Lower score = more likely to be true.
        
        TransE: score = ||head + relation - tail||
        """
        h = self._GetOrCreateEntityEmbedding(head)
        r = self._GetOrCreateRelationEmbedding(relation)
        t = self._GetOrCreateEntityEmbedding(tail)
        
        # h + r - t
        diff = [h[i] + r[i] - t[i] for i in range(self.Dim)]
        
        # L1 norm (Manhattan distance)
        return sum(abs(x) for x in diff)
    
    def _GenerateNegativeSample(self, head: str, relation: str, tail: str) -> Tuple[str, str, str]:
        """Generate a negative sample by corrupting head or tail"""
        entities = list(self.EntityEmbeddings.keys())
        
        if random.random() < 0.5:
            # Corrupt head
            new_head = random.choice(entities)
            while (new_head, relation, tail) in self.TripleSet:
                new_head = random.choice(entities)
            return (new_head, relation, tail)
        else:
            # Corrupt tail
            new_tail = random.choice(entities)
            while (head, relation, new_tail) in self.TripleSet:
                new_tail = random.choice(entities)
            return (head, relation, new_tail)
    
    def Train(self, Epochs: int = 100, BatchSize: int = 32, Verbose: bool = True) -> List[float]:
        """
        Train the neural network.
        
        Uses margin-based ranking loss:
            L = max(0, margin + score_pos - score_neg)
        
        Args:
            Epochs: Number of training epochs
            BatchSize: Batch size for training
            Verbose: Print progress
        
        Returns:
            List of loss values per epoch
        """
        if len(self.Triples) == 0:
            if Verbose:
                print("⚠️ No triples to train on!")
            return []
        
        start_time = time.time()
        
        if Verbose:
            print(f"\n🧠 Training Neural Network...")
            print(f"   Entities: {len(self.EntityEmbeddings)}")
            print(f"   Relations: {len(self.RelationEmbeddings)}")
            print(f"   Triples: {len(self.Triples)}")
            print(f"   Dimension: {self.Dim}")
            print()
        
        losses = []
        
        for epoch in range(Epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Shuffle triples
            shuffled = self.Triples.copy()
            random.shuffle(shuffled)
            
            # Process in batches
            for i in range(0, len(shuffled), BatchSize):
                batch = shuffled[i:i+BatchSize]
                batch_loss = 0.0
                
                for (h, r, t) in batch:
                    # Positive score
                    pos_score = self._Score(h, r, t)
                    
                    # Generate negative samples
                    for _ in range(self.NegativeSamples):
                        nh, nr, nt = self._GenerateNegativeSample(h, r, t)
                        neg_score = self._Score(nh, nr, nt)
                        
                        # Margin ranking loss
                        loss = max(0, self.Margin + pos_score - neg_score)
                        batch_loss += loss
                        
                        # Gradient update (simplified SGD)
                        if loss > 0:
                            self._UpdateEmbeddings(h, r, t, nh, nr, nt)
                
                epoch_loss += batch_loss
                num_batches += 1
            
            # Normalize entities
            self._NormalizeEntities()
            
            avg_loss = epoch_loss / (len(shuffled) * self.NegativeSamples) if shuffled else 0
            losses.append(avg_loss)
            
            if Verbose and (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch + 1}/{Epochs}: Loss = {avg_loss:.4f}")
        
        training_time = time.time() - start_time
        
        # Update stats
        self.LossHistory.extend(losses)
        self.Stats.TrainingEpochs += Epochs
        self.Stats.LastLoss = losses[-1] if losses else 0
        self.Stats.IsTrained = True
        self.Stats.TrainingTime += training_time
        
        if Verbose:
            print(f"\n✅ Training complete!")
            print(f"   Final loss: {losses[-1]:.4f}")
            print(f"   Time: {training_time:.2f}s")
        
        return losses
    
    def _UpdateEmbeddings(self, h, r, t, nh, nr, nt):
        """Update embeddings using gradient descent"""
        lr = self.LearningRate
        
        # Get embeddings
        h_emb = self.EntityEmbeddings[h]
        r_emb = self.RelationEmbeddings[r]
        t_emb = self.EntityEmbeddings[t]
        nh_emb = self.EntityEmbeddings[nh]
        nt_emb = self.EntityEmbeddings[nt]
        
        # Gradient for positive triple: minimize h + r - t
        for i in range(self.Dim):
            grad_sign = 1 if (h_emb[i] + r_emb[i] - t_emb[i]) > 0 else -1
            
            h_emb[i] -= lr * grad_sign
            r_emb[i] -= lr * grad_sign
            t_emb[i] += lr * grad_sign
        
        # Gradient for negative triple: maximize nh + r - nt
        for i in range(self.Dim):
            grad_sign = 1 if (nh_emb[i] + r_emb[i] - nt_emb[i]) > 0 else -1
            
            nh_emb[i] += lr * grad_sign
            nt_emb[i] -= lr * grad_sign
    
    def _NormalizeEntities(self):
        """Normalize entity embeddings to unit length"""
        for entity in self.EntityEmbeddings:
            emb = self.EntityEmbeddings[entity]
            norm = math.sqrt(sum(x*x for x in emb))
            if norm > 0:
                self.EntityEmbeddings[entity] = [x / norm for x in emb]
    
    def PredictTail(self, head: str, relation: str, TopK: int = 5) -> List[Prediction]:
        """
        Predict the tail entity given head and relation.
        
        Returns top-K predictions ranked by confidence.
        """
        self.Stats.Predictions += 1
        
        if head not in self.EntityEmbeddings:
            self._GetOrCreateEntityEmbedding(head)
        if relation not in self.RelationEmbeddings:
            self._GetOrCreateRelationEmbedding(relation)
        
        scores = []
        for entity in self.EntityEmbeddings:
            if entity != head:
                score = self._Score(head, relation, entity)
                scores.append((entity, score))
        
        # Sort by score (lower is better)
        scores.sort(key=lambda x: x[1])
        
        # Convert to predictions with confidence
        predictions = []
        max_score = scores[-1][1] if scores else 1
        
        for entity, score in scores[:TopK]:
            # Convert score to confidence (0-1, higher is better)
            confidence = 1.0 - (score / (max_score + 0.001))
            predictions.append(Prediction(
                Head=head,
                Relation=relation,
                Tail=entity,
                Confidence=max(0, min(1, confidence))
            ))
        
        return predictions
    
    def PredictHead(self, relation: str, tail: str, TopK: int = 5) -> List[Prediction]:
        """Predict the head entity given relation and tail"""
        self.Stats.Predictions += 1
        
        if tail not in self.EntityEmbeddings:
            self._GetOrCreateEntityEmbedding(tail)
        if relation not in self.RelationEmbeddings:
            self._GetOrCreateRelationEmbedding(relation)
        
        scores = []
        for entity in self.EntityEmbeddings:
            if entity != tail:
                score = self._Score(entity, relation, tail)
                scores.append((entity, score))
        
        scores.sort(key=lambda x: x[1])
        
        predictions = []
        max_score = scores[-1][1] if scores else 1
        
        for entity, score in scores[:TopK]:
            confidence = 1.0 - (score / (max_score + 0.001))
            predictions.append(Prediction(
                Head=entity,
                Relation=relation,
                Tail=tail,
                Confidence=max(0, min(1, confidence))
            ))
        
        return predictions
    
    def FindSimilar(self, entity: str, TopK: int = 5) -> List[Tuple[str, float]]:
        """Find entities similar to the given entity"""
        if entity not in self.EntityEmbeddings:
            self._GetOrCreateEntityEmbedding(entity)
        
        target = self.EntityEmbeddings[entity]
        
        similarities = []
        for other, emb in self.EntityEmbeddings.items():
            if other != entity:
                # Cosine similarity
                dot = sum(a * b for a, b in zip(target, emb))
                norm1 = math.sqrt(sum(a*a for a in target))
                norm2 = math.sqrt(sum(a*a for a in emb))
                
                if norm1 > 0 and norm2 > 0:
                    similarity = dot / (norm1 * norm2)
                    similarities.append((other, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:TopK]
    
    def GenerateHypotheses(self, MinConfidence: float = 0.5, MaxHypotheses: int = 20) -> List[Prediction]:
        """
        Generate novel hypotheses (facts that might be true).
        
        Looks for missing links with high prediction confidence.
        """
        hypotheses = []
        
        # For each entity and relation, predict possible tails
        for entity in list(self.EntityEmbeddings.keys())[:50]:  # Limit for speed
            for relation in self.RelationEmbeddings:
                predictions = self.PredictTail(entity, relation, TopK=3)
                
                for pred in predictions:
                    # Check if this is a novel hypothesis (not in training data)
                    if (pred.Head, pred.Relation, pred.Tail) not in self.TripleSet:
                        if pred.Confidence >= MinConfidence:
                            hypotheses.append(pred)
                            self.Stats.Hypotheses += 1
                            
                            if len(hypotheses) >= MaxHypotheses:
                                return hypotheses
        
        # Sort by confidence
        hypotheses.sort(key=lambda x: x.Confidence, reverse=True)
        return hypotheses[:MaxHypotheses]
    
    def ReasonAboutUnknown(self, entity: str) -> Dict:
        """
        Reason about an entity that wasn't in training.
        
        Uses similarity to known entities to make inferences.
        """
        # Create embedding for unknown entity
        self._GetOrCreateEntityEmbedding(entity)
        
        # Find similar known entities
        similar = self.FindSimilar(entity, TopK=5)
        
        # Collect predictions based on similar entities
        predictions_by_relation = defaultdict(list)
        
        for similar_entity, similarity in similar:
            # Get facts about similar entity
            for (h, r, t) in self.Triples:
                if h == similar_entity:
                    predictions_by_relation[r].append({
                        "Tail": t,
                        "BasedOn": similar_entity,
                        "Similarity": similarity
                    })
        
        # Aggregate predictions
        inferences = []
        for relation, preds in predictions_by_relation.items():
            # Weight by similarity
            tail_scores = defaultdict(float)
            for p in preds:
                tail_scores[p["Tail"]] += p["Similarity"]
            
            # Best prediction
            if tail_scores:
                best_tail = max(tail_scores.items(), key=lambda x: x[1])
                inferences.append({
                    "Relation": relation,
                    "PredictedTail": best_tail[0],
                    "Confidence": min(1.0, best_tail[1])
                })
        
        return {
            "Entity": entity,
            "SimilarEntities": [{"Entity": e, "Similarity": round(s, 3)} for e, s in similar],
            "Inferences": sorted(inferences, key=lambda x: x["Confidence"], reverse=True)
        }
    
    def LearnOnline(self, head: str, relation: str, tail: str, Iterations: int = 10):
        """
        Online learning: update model with a single new fact.
        
        This allows continuous learning without full retraining.
        """
        self.AddTriple(head, relation, tail)
        
        # Quick local update
        for _ in range(Iterations):
            pos_score = self._Score(head, relation, tail)
            neg_h, neg_r, neg_t = self._GenerateNegativeSample(head, relation, tail)
            neg_score = self._Score(neg_h, neg_r, neg_t)
            
            loss = max(0, self.Margin + pos_score - neg_score)
            if loss > 0:
                self._UpdateEmbeddings(head, relation, tail, neg_h, neg_r, neg_t)
        
        self._NormalizeEntities()
    
    def GetStats(self) -> NeuralStats:
        """Get current statistics"""
        return self.Stats
    
    def GetStatsDict(self) -> Dict:
        """Get stats as dictionary (for API)"""
        return self.Stats.ToDict()
    
    def Save(self, filepath: str):
        """Save the model to a file"""
        data = {
            "Dim": self.Dim,
            "LearningRate": self.LearningRate,
            "Margin": self.Margin,
            "NegativeSamples": self.NegativeSamples,
            "EntityEmbeddings": self.EntityEmbeddings,
            "RelationEmbeddings": self.RelationEmbeddings,
            "Entity2Idx": self.Entity2Idx,
            "Idx2Entity": {str(k): v for k, v in self.Idx2Entity.items()},
            "Relation2Idx": self.Relation2Idx,
            "Idx2Relation": {str(k): v for k, v in self.Idx2Relation.items()},
            "Triples": self.Triples,
            "LossHistory": self.LossHistory,
            "Stats": self.Stats.ToDict()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def Load(self, filepath: str):
        """Load the model from a file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.Dim = data["Dim"]
        self.LearningRate = data["LearningRate"]
        self.Margin = data["Margin"]
        self.NegativeSamples = data["NegativeSamples"]
        self.EntityEmbeddings = data["EntityEmbeddings"]
        self.RelationEmbeddings = data["RelationEmbeddings"]
        self.Entity2Idx = data["Entity2Idx"]
        self.Idx2Entity = {int(k): v for k, v in data["Idx2Entity"].items()}
        self.Relation2Idx = data["Relation2Idx"]
        self.Idx2Relation = {int(k): v for k, v in data["Idx2Relation"].items()}
        self.Triples = data["Triples"]
        self.TripleSet = set(tuple(t) for t in self.Triples)
        self.LossHistory = data["LossHistory"]
        
        # Restore stats
        stats = data.get("Stats", {})
        self.Stats = NeuralStats(
            TotalEntities=stats.get("TotalEntities", len(self.EntityEmbeddings)),
            TotalRelations=stats.get("TotalRelations", len(self.RelationEmbeddings)),
            TotalTriples=stats.get("TotalTriples", len(self.Triples)),
            EmbeddingDim=self.Dim,
            TrainingEpochs=stats.get("TrainingEpochs", 0),
            LastLoss=stats.get("LastLoss", 0),
            Predictions=stats.get("Predictions", 0),
            Hypotheses=stats.get("Hypotheses", 0),
            IsTrained=stats.get("IsTrained", False),
            TrainingTime=stats.get("TrainingTime", 0)
        )


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("🧠 Neural Engine Test\n")
    
    # Create engine
    engine = NeuralEngine(EmbeddingDim=50)
    
    # Add knowledge
    facts = [
        ("dog", "is_a", "mammal"),
        ("cat", "is_a", "mammal"),
        ("mammal", "is_a", "animal"),
        ("bird", "is_a", "animal"),
        ("eagle", "is_a", "bird"),
        ("sparrow", "is_a", "bird"),
        ("whale", "is_a", "mammal"),
        ("dolphin", "is_a", "mammal"),
        ("dog", "has", "fur"),
        ("cat", "has", "fur"),
        ("bird", "has", "feathers"),
        ("dog", "can", "bark"),
        ("cat", "can", "meow"),
        ("bird", "can", "fly"),
    ]
    
    engine.AddTriples(facts)
    
    # Train
    losses = engine.Train(Epochs=50, Verbose=True)
    
    # Test prediction
    print("\n📊 Predictions:")
    print("\nWhat is a dog?")
    for pred in engine.PredictTail("dog", "is_a", TopK=3):
        print(f"  → {pred.Tail}: {pred.Confidence:.0%}")
    
    print("\nWhat is similar to dog?")
    for entity, sim in engine.FindSimilar("dog", TopK=3):
        print(f"  → {entity}: {sim:.0%} similar")
    
    print("\nGenerating hypotheses...")
    for hyp in engine.GenerateHypotheses(MinConfidence=0.3, MaxHypotheses=5):
        print(f"  → {hyp.Head} {hyp.Relation} {hyp.Tail} ({hyp.Confidence:.0%})")
    
    print("\n✅ Test complete!")
    print(f"\nStats: {engine.GetStatsDict()}")
