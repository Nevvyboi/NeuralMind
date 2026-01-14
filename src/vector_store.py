"""
FAISS Vector Store for GroundZero AI
=====================================
Provides fast semantic search over TransE entity embeddings.

Usage:
    from src.vector_store import VectorStore
    
    store = VectorStore("data", Dimension=100)
    store.SyncFromNeuralEngine(neural_engine)
    
    # Find similar entities
    similar = store.FindSimilar("dog", TopK=10)
    # Returns: [("cat", 0.92), ("wolf", 0.89), ...]

Requirements:
    pip install faiss-cpu  # or faiss-gpu for NVIDIA GPU
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Try to import faiss
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS not installed. Run: pip install faiss-cpu")


class VectorStore:
    """
    Fast vector storage for TransE embeddings using FAISS.
    Enables semantic search over knowledge graph entities.
    """
    
    def __init__(self, DataDir: str, Dimension: int = 100):
        """
        Initialize the vector store.
        
        Args:
            DataDir: Directory to store index files
            Dimension: Embedding dimension (must match TransE)
        """
        self.DataDir = Path(DataDir)
        self.Dimension = Dimension
        self.Index = None
        self.EntityToID: Dict[str, int] = {}
        self.IDToEntity: Dict[int, str] = {}
        self.RelationToID: Dict[str, int] = {}
        self.IDToRelation: Dict[int, str] = {}
        
        # Separate index for relations
        self.RelationIndex = None
        
        if FAISS_AVAILABLE:
            self._Load()
        else:
            print("⚠️ VectorStore disabled - FAISS not available")
    
    def _Load(self):
        """Load existing index or create new one"""
        index_path = self.DataDir / "faiss_entities.bin"
        meta_path = self.DataDir / "faiss_meta.pkl"
        rel_index_path = self.DataDir / "faiss_relations.bin"
        
        if index_path.exists() and meta_path.exists():
            try:
                self.Index = faiss.read_index(str(index_path))
                
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                    self.EntityToID = meta.get('EntityToID', {})
                    self.IDToEntity = meta.get('IDToEntity', {})
                    self.RelationToID = meta.get('RelationToID', {})
                    self.IDToRelation = meta.get('IDToRelation', {})
                
                if rel_index_path.exists():
                    self.RelationIndex = faiss.read_index(str(rel_index_path))
                
                print(f"✅ Loaded FAISS: {self.Index.ntotal} entities, "
                      f"{len(self.RelationToID)} relations")
            except Exception as e:
                print(f"⚠️ Error loading FAISS index: {e}")
                self._CreateNew()
        else:
            self._CreateNew()
    
    def _CreateNew(self):
        """Create new empty indexes"""
        # IndexFlatIP = Inner Product (cosine similarity for normalized vectors)
        self.Index = faiss.IndexFlatIP(self.Dimension)
        self.RelationIndex = faiss.IndexFlatIP(self.Dimension)
        self.EntityToID = {}
        self.IDToEntity = {}
        self.RelationToID = {}
        self.IDToRelation = {}
        print("✅ Created new FAISS indexes")
    
    def Save(self):
        """Persist indexes to disk"""
        if not FAISS_AVAILABLE or self.Index is None:
            return
        
        self.DataDir.mkdir(parents=True, exist_ok=True)
        
        # Save entity index
        faiss.write_index(self.Index, str(self.DataDir / "faiss_entities.bin"))
        
        # Save relation index
        if self.RelationIndex and self.RelationIndex.ntotal > 0:
            faiss.write_index(self.RelationIndex, str(self.DataDir / "faiss_relations.bin"))
        
        # Save metadata
        with open(self.DataDir / "faiss_meta.pkl", 'wb') as f:
            pickle.dump({
                'EntityToID': self.EntityToID,
                'IDToEntity': self.IDToEntity,
                'RelationToID': self.RelationToID,
                'IDToRelation': self.IDToRelation,
            }, f)
        
        print(f"💾 Saved FAISS: {self.Index.ntotal} entities, "
              f"{len(self.RelationToID)} relations")
    
    def AddEntities(self, Entities: List[str], Embeddings: np.ndarray):
        """
        Add entity embeddings to the index.
        
        Args:
            Entities: List of entity names
            Embeddings: numpy array of shape (n_entities, dimension)
        """
        if not FAISS_AVAILABLE:
            return
        
        embeddings = Embeddings.astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        start_id = self.Index.ntotal
        self.Index.add(embeddings)
        
        # Update mappings
        for i, entity in enumerate(Entities):
            idx = start_id + i
            self.EntityToID[entity] = idx
            self.IDToEntity[idx] = entity
    
    def AddRelations(self, Relations: List[str], Embeddings: np.ndarray):
        """Add relation embeddings to the relation index"""
        if not FAISS_AVAILABLE or self.RelationIndex is None:
            return
        
        embeddings = Embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        
        start_id = self.RelationIndex.ntotal
        self.RelationIndex.add(embeddings)
        
        for i, relation in enumerate(Relations):
            idx = start_id + i
            self.RelationToID[relation] = idx
            self.IDToRelation[idx] = relation
    
    def Search(self, QueryEmbedding: np.ndarray, TopK: int = 10) -> List[Tuple[str, float]]:
        """
        Find most similar entities to query embedding.
        
        Args:
            QueryEmbedding: Query vector
            TopK: Number of results
            
        Returns:
            List of (entity_name, similarity_score) tuples
        """
        if not FAISS_AVAILABLE or self.Index is None or self.Index.ntotal == 0:
            return []
        
        # Normalize query
        query = QueryEmbedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query)
        
        # Search
        scores, indices = self.Index.search(query, min(TopK, self.Index.ntotal))
        
        # Convert to entity names
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx in self.IDToEntity:
                results.append((self.IDToEntity[idx], float(score)))
        
        return results
    
    def FindSimilar(self, Entity: str, TopK: int = 10) -> List[Tuple[str, float]]:
        """
        Find entities most similar to a given entity.
        
        Args:
            Entity: Entity name to find similar entities for
            TopK: Number of results
            
        Returns:
            List of (entity_name, similarity_score) tuples
        """
        if not FAISS_AVAILABLE or Entity not in self.EntityToID:
            return []
        
        idx = self.EntityToID[Entity]
        
        # Reconstruct the embedding
        try:
            embedding = self.Index.reconstruct(idx).reshape(1, -1)
            results = self.Search(embedding, TopK + 1)
            # Exclude the entity itself
            return [(e, s) for e, s in results if e != Entity][:TopK]
        except Exception as e:
            print(f"Error finding similar: {e}")
            return []
    
    def SearchByText(self, Text: str, NeuralEngine, TopK: int = 10) -> List[Tuple[str, float]]:
        """
        Search for entities related to text using neural engine encoding.
        Requires NeuralEngine to have entity embedding lookup.
        """
        # This is a placeholder - actual implementation depends on 
        # how you want to encode text to vector
        # For now, return empty list
        return []
    
    def SyncFromNeuralEngine(self, NeuralEngine):
        """
        Synchronize embeddings from your TransE NeuralEngine.
        Call this after training to enable semantic search.
        
        Args:
            NeuralEngine: Your NeuralEngine instance with EntityEmbeddings
        """
        if not FAISS_AVAILABLE:
            print("⚠️ Cannot sync - FAISS not available")
            return
        
        # Check for entity embeddings
        if not hasattr(NeuralEngine, 'EntityEmbeddings'):
            print("⚠️ NeuralEngine has no EntityEmbeddings")
            return
        
        entity_embeddings = NeuralEngine.EntityEmbeddings
        if not entity_embeddings:
            print("⚠️ EntityEmbeddings is empty")
            return
        
        # Get dimension from first embedding
        first_key = next(iter(entity_embeddings.keys()))
        first_emb = entity_embeddings[first_key]
        
        if hasattr(first_emb, '__len__'):
            dim = len(first_emb)
        else:
            print("⚠️ Could not determine embedding dimension")
            return
        
        self.Dimension = dim
        
        # Rebuild indexes
        self._CreateNew()
        
        # Add entity embeddings
        entities = list(entity_embeddings.keys())
        embeddings = np.array([
            entity_embeddings[e] if isinstance(entity_embeddings[e], (list, np.ndarray))
            else [entity_embeddings[e]]
            for e in entities
        ]).astype('float32')
        
        self.AddEntities(entities, embeddings)
        
        # Add relation embeddings if available
        if hasattr(NeuralEngine, 'RelationEmbeddings') and NeuralEngine.RelationEmbeddings:
            rel_embeddings = NeuralEngine.RelationEmbeddings
            relations = list(rel_embeddings.keys())
            rel_vectors = np.array([
                rel_embeddings[r] if isinstance(rel_embeddings[r], (list, np.ndarray))
                else [rel_embeddings[r]]
                for r in relations
            ]).astype('float32')
            
            self.AddRelations(relations, rel_vectors)
        
        # Save to disk
        self.Save()
        
        print(f"✅ Synced from NeuralEngine: {len(entities)} entities, dim={dim}")
    
    def GetStats(self) -> Dict:
        """Get vector store statistics"""
        return {
            "TotalEntities": self.Index.ntotal if self.Index else 0,
            "TotalRelations": len(self.RelationToID),
            "Dimension": self.Dimension,
            "FAISSAvailable": FAISS_AVAILABLE
        }


# Convenience function to check if FAISS is available
def CheckFAISS() -> bool:
    """Check if FAISS is installed and working"""
    if not FAISS_AVAILABLE:
        print("=" * 50)
        print("FAISS is not installed!")
        print("To install:")
        print("  pip install faiss-cpu    # For CPU only")
        print("  pip install faiss-gpu    # For NVIDIA GPU")
        print("=" * 50)
        return False
    return True


if __name__ == "__main__":
    # Test the vector store
    CheckFAISS()
    
    if FAISS_AVAILABLE:
        # Create test store
        store = VectorStore("test_data", Dimension=100)
        
        # Add some test entities
        entities = ["dog", "cat", "wolf", "bird", "fish"]
        embeddings = np.random.randn(5, 100).astype('float32')
        
        store.AddEntities(entities, embeddings)
        store.Save()
        
        # Test similarity search
        similar = store.FindSimilar("dog", TopK=3)
        print(f"Similar to 'dog': {similar}")
        
        print(f"\nStats: {store.GetStats()}")
