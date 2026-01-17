"""
GroundZero AI - Knowledge Graph System
======================================

A comprehensive knowledge graph that stores, connects, and verifies information.
The AI uses this to:
1. Store learned knowledge
2. Connect related concepts
3. Verify facts
4. Reason over relationships
"""

import json
import re
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import hashlib

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from ..utils import (
        get_data_path, ensure_dir, load_json, save_json,
        logger, timestamp, generate_id
    )
except ImportError:
    from utils import (
        get_data_path, ensure_dir, load_json, save_json,
        logger, timestamp, generate_id
    )


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class KnowledgeNode:
    """A node in the knowledge graph representing a concept or entity."""
    id: str
    name: str
    node_type: str  # concept, entity, fact, skill, event
    content: str
    confidence: float = 1.0
    source: str = "learned"
    created_at: str = field(default_factory=timestamp)
    updated_at: str = field(default_factory=timestamp)
    access_count: int = 0
    verified: bool = False
    metadata: Dict = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "node_type": self.node_type,
            "content": self.content,
            "confidence": self.confidence,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "access_count": self.access_count,
            "verified": self.verified,
            "metadata": self.metadata,
            # Don't save embedding in main JSON (too large)
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'KnowledgeNode':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class KnowledgeEdge:
    """An edge connecting two knowledge nodes."""
    id: str
    source_id: str
    target_id: str
    relation: str  # is_a, part_of, related_to, causes, etc.
    weight: float = 1.0
    confidence: float = 1.0
    bidirectional: bool = False
    created_at: str = field(default_factory=timestamp)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'KnowledgeEdge':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================================
# KNOWLEDGE GRAPH
# ============================================================================

class KnowledgeGraph:
    """
    The main knowledge graph for storing and reasoning over knowledge.
    
    Features:
    - Add/update/delete nodes and edges
    - Query by various criteria
    - Find paths between concepts
    - Semantic similarity search (with embeddings)
    - Verification and confidence tracking
    - Automatic relationship inference
    """
    
    # Standard relation types
    RELATIONS = {
        "is_a": "categorical membership",
        "part_of": "compositional relationship",
        "has_property": "attribute relationship",
        "related_to": "general association",
        "causes": "causal relationship",
        "requires": "dependency relationship",
        "example_of": "instantiation",
        "opposite_of": "antonym relationship",
        "similar_to": "similarity",
        "leads_to": "sequential/consequential",
        "defined_as": "definitional",
        "used_for": "functional purpose",
        "located_in": "spatial relationship",
        "created_by": "authorship/creation",
        "instance_of": "type instantiation",
    }
    
    def __init__(self, path: str = None):
        self.path = Path(path) if path else get_data_path("knowledge", "knowledge_graph.json")
        self.embeddings_path = get_data_path("embeddings", "knowledge_embeddings.npy")
        
        # Storage
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: Dict[str, KnowledgeEdge] = {}
        
        # Indexes for fast lookup
        self._name_index: Dict[str, Set[str]] = defaultdict(set)  # name -> node_ids
        self._type_index: Dict[str, Set[str]] = defaultdict(set)  # type -> node_ids
        self._outgoing_edges: Dict[str, Set[str]] = defaultdict(set)  # node_id -> edge_ids
        self._incoming_edges: Dict[str, Set[str]] = defaultdict(set)  # node_id -> edge_ids
        self._relation_index: Dict[str, Set[str]] = defaultdict(set)  # relation -> edge_ids
        
        # Embeddings matrix (if using)
        self._embeddings_matrix = None
        self._id_to_embedding_idx: Dict[str, int] = {}
        
        # Stats
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "nodes_added": 0,
            "edges_added": 0,
        }
        
        # Load existing data
        self._load()
    
    def _load(self):
        """Load knowledge graph from disk."""
        if self.path.exists():
            try:
                data = load_json(self.path)
                
                # Load nodes
                for node_data in data.get("nodes", []):
                    node = KnowledgeNode.from_dict(node_data)
                    self.nodes[node.id] = node
                    self._index_node(node)
                
                # Load edges
                for edge_data in data.get("edges", []):
                    edge = KnowledgeEdge.from_dict(edge_data)
                    self.edges[edge.id] = edge
                    self._index_edge(edge)
                
                logger.info(f"Loaded knowledge graph: {len(self.nodes)} nodes, {len(self.edges)} edges")
            except Exception as e:
                logger.error(f"Error loading knowledge graph: {e}")
        
        # Load embeddings if available
        if HAS_NUMPY and self.embeddings_path.exists():
            try:
                data = np.load(self.embeddings_path, allow_pickle=True).item()
                self._embeddings_matrix = data.get("embeddings")
                self._id_to_embedding_idx = data.get("id_map", {})
                logger.info(f"Loaded {len(self._id_to_embedding_idx)} embeddings")
            except Exception as e:
                logger.warning(f"Could not load embeddings: {e}")
    
    def save(self):
        """Save knowledge graph to disk."""
        ensure_dir(self.path.parent)
        
        data = {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges.values()],
            "stats": self.stats,
            "saved_at": timestamp(),
        }
        
        save_json(self.path, data)
        
        # Save embeddings separately
        if HAS_NUMPY and self._embeddings_matrix is not None:
            ensure_dir(self.embeddings_path.parent)
            np.save(self.embeddings_path, {
                "embeddings": self._embeddings_matrix,
                "id_map": self._id_to_embedding_idx,
            })
        
        logger.info(f"Saved knowledge graph: {len(self.nodes)} nodes, {len(self.edges)} edges")
    
    def _index_node(self, node: KnowledgeNode):
        """Add node to indexes."""
        # Name index (lowercase for case-insensitive search)
        name_words = node.name.lower().split()
        for word in name_words:
            self._name_index[word].add(node.id)
        self._name_index[node.name.lower()].add(node.id)
        
        # Type index
        self._type_index[node.node_type].add(node.id)
    
    def _index_edge(self, edge: KnowledgeEdge):
        """Add edge to indexes."""
        self._outgoing_edges[edge.source_id].add(edge.id)
        self._incoming_edges[edge.target_id].add(edge.id)
        self._relation_index[edge.relation].add(edge.id)
        
        if edge.bidirectional:
            self._outgoing_edges[edge.target_id].add(edge.id)
            self._incoming_edges[edge.source_id].add(edge.id)
    
    def _remove_from_indexes(self, node: KnowledgeNode):
        """Remove node from indexes."""
        name_words = node.name.lower().split()
        for word in name_words:
            self._name_index[word].discard(node.id)
        self._name_index[node.name.lower()].discard(node.id)
        self._type_index[node.node_type].discard(node.id)
    
    # ========================================================================
    # NODE OPERATIONS
    # ========================================================================
    
    def add_node(
        self,
        name: str,
        content: str,
        node_type: str = "concept",
        confidence: float = 1.0,
        source: str = "learned",
        metadata: Dict = None,
        embedding: List[float] = None,
    ) -> KnowledgeNode:
        """Add a new knowledge node."""
        # Generate ID from name
        node_id = hashlib.md5(name.lower().encode()).hexdigest()[:12]
        
        # Check if exists - update instead
        if node_id in self.nodes:
            return self.update_node(node_id, content=content, confidence=confidence, 
                                   source=source, metadata=metadata)
        
        node = KnowledgeNode(
            id=node_id,
            name=name,
            node_type=node_type,
            content=content,
            confidence=confidence,
            source=source,
            metadata=metadata or {},
            embedding=embedding,
        )
        
        self.nodes[node_id] = node
        self._index_node(node)
        self.stats["nodes_added"] += 1
        
        logger.debug(f"Added node: {name} ({node_type})")
        return node
    
    def update_node(
        self,
        node_id: str,
        **updates
    ) -> Optional[KnowledgeNode]:
        """Update an existing node."""
        if node_id not in self.nodes:
            return None
        
        node = self.nodes[node_id]
        
        # Handle name change (reindex)
        if "name" in updates and updates["name"] != node.name:
            self._remove_from_indexes(node)
            node.name = updates["name"]
            self._index_node(node)
            del updates["name"]
        
        # Update other fields
        for key, value in updates.items():
            if hasattr(node, key):
                setattr(node, key, value)
        
        node.updated_at = timestamp()
        return node
    
    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Get a node by ID."""
        node = self.nodes.get(node_id)
        if node:
            node.access_count += 1
        return node
    
    def get_node_by_name(self, name: str) -> Optional[KnowledgeNode]:
        """Get a node by name."""
        node_ids = self._name_index.get(name.lower(), set())
        if node_ids:
            node_id = next(iter(node_ids))
            return self.get_node(node_id)
        return None
    
    def delete_node(self, node_id: str) -> bool:
        """Delete a node and its edges."""
        if node_id not in self.nodes:
            return False
        
        node = self.nodes[node_id]
        
        # Remove edges
        edge_ids = list(self._outgoing_edges.get(node_id, set()) | 
                       self._incoming_edges.get(node_id, set()))
        for edge_id in edge_ids:
            self.delete_edge(edge_id)
        
        # Remove from indexes
        self._remove_from_indexes(node)
        
        # Remove node
        del self.nodes[node_id]
        return True
    
    def search_nodes(
        self,
        query: str = None,
        node_type: str = None,
        min_confidence: float = 0,
        limit: int = 100,
    ) -> List[KnowledgeNode]:
        """Search for nodes matching criteria."""
        self.stats["total_queries"] += 1
        
        results = []
        
        # Start with type filter if provided
        if node_type:
            candidates = [self.nodes[nid] for nid in self._type_index.get(node_type, set()) if nid in self.nodes]
        else:
            candidates = list(self.nodes.values())
        
        # Filter by query
        if query:
            query_words = set(query.lower().split())
            matching_ids = set()
            for word in query_words:
                matching_ids.update(self._name_index.get(word, set()))
            candidates = [n for n in candidates if n.id in matching_ids or 
                         query.lower() in n.content.lower()]
        
        # Filter by confidence
        results = [n for n in candidates if n.confidence >= min_confidence]
        
        # Sort by relevance (access count + confidence)
        results.sort(key=lambda n: (n.access_count * 0.1 + n.confidence), reverse=True)
        
        return results[:limit]
    
    # ========================================================================
    # EDGE OPERATIONS
    # ========================================================================
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        weight: float = 1.0,
        confidence: float = 1.0,
        bidirectional: bool = False,
        metadata: Dict = None,
    ) -> Optional[KnowledgeEdge]:
        """Add an edge between two nodes."""
        if source_id not in self.nodes or target_id not in self.nodes:
            logger.warning(f"Cannot add edge: nodes not found ({source_id} -> {target_id})")
            return None
        
        # Generate edge ID
        edge_id = f"{source_id}_{relation}_{target_id}"
        
        # Check if exists
        if edge_id in self.edges:
            # Update weight/confidence
            self.edges[edge_id].weight = weight
            self.edges[edge_id].confidence = confidence
            return self.edges[edge_id]
        
        edge = KnowledgeEdge(
            id=edge_id,
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            weight=weight,
            confidence=confidence,
            bidirectional=bidirectional,
            metadata=metadata or {},
        )
        
        self.edges[edge_id] = edge
        self._index_edge(edge)
        self.stats["edges_added"] += 1
        
        return edge
    
    def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge."""
        if edge_id not in self.edges:
            return False
        
        edge = self.edges[edge_id]
        
        # Remove from indexes
        self._outgoing_edges[edge.source_id].discard(edge_id)
        self._incoming_edges[edge.target_id].discard(edge_id)
        self._relation_index[edge.relation].discard(edge_id)
        
        if edge.bidirectional:
            self._outgoing_edges[edge.target_id].discard(edge_id)
            self._incoming_edges[edge.source_id].discard(edge_id)
        
        del self.edges[edge_id]
        return True
    
    def get_edges(
        self,
        node_id: str,
        direction: str = "both",
        relation: str = None,
    ) -> List[KnowledgeEdge]:
        """Get edges connected to a node."""
        edge_ids = set()
        
        if direction in ["out", "both"]:
            edge_ids.update(self._outgoing_edges.get(node_id, set()))
        if direction in ["in", "both"]:
            edge_ids.update(self._incoming_edges.get(node_id, set()))
        
        edges = [self.edges[eid] for eid in edge_ids if eid in self.edges]
        
        if relation:
            edges = [e for e in edges if e.relation == relation]
        
        return edges
    
    def get_neighbors(
        self,
        node_id: str,
        relation: str = None,
        direction: str = "both",
    ) -> List[KnowledgeNode]:
        """Get neighboring nodes."""
        edges = self.get_edges(node_id, direction=direction, relation=relation)
        
        neighbor_ids = set()
        for edge in edges:
            if edge.source_id == node_id:
                neighbor_ids.add(edge.target_id)
            if edge.target_id == node_id:
                neighbor_ids.add(edge.source_id)
        
        return [self.nodes[nid] for nid in neighbor_ids if nid in self.nodes]
    
    # ========================================================================
    # KNOWLEDGE OPERATIONS
    # ========================================================================
    
    def add_knowledge(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        source: str = "learned",
    ) -> Tuple[KnowledgeNode, KnowledgeNode, KnowledgeEdge]:
        """
        Add a knowledge triple (subject, predicate, object).
        
        Example: add_knowledge("Python", "is_a", "Programming Language")
        """
        # Create/get subject node
        subj_node = self.get_node_by_name(subject)
        if not subj_node:
            subj_node = self.add_node(subject, f"{subject}", source=source, confidence=confidence)
        
        # Create/get object node
        obj_node = self.get_node_by_name(obj)
        if not obj_node:
            obj_node = self.add_node(obj, f"{obj}", source=source, confidence=confidence)
        
        # Create edge
        edge = self.add_edge(
            subj_node.id, obj_node.id,
            relation=predicate,
            confidence=confidence,
        )
        
        return subj_node, obj_node, edge
    
    def query_knowledge(
        self,
        subject: str = None,
        predicate: str = None,
        obj: str = None,
    ) -> List[Tuple[KnowledgeNode, KnowledgeEdge, KnowledgeNode]]:
        """
        Query knowledge triples.
        
        Examples:
            query_knowledge(subject="Python") - What do we know about Python?
            query_knowledge(predicate="is_a") - All is_a relationships
            query_knowledge(subject="Python", predicate="is_a") - What is Python?
        """
        results = []
        
        # Get candidate edges
        if predicate:
            edge_ids = self._relation_index.get(predicate, set())
        else:
            edge_ids = set(self.edges.keys())
        
        for edge_id in edge_ids:
            edge = self.edges.get(edge_id)
            if not edge:
                continue
            
            source_node = self.nodes.get(edge.source_id)
            target_node = self.nodes.get(edge.target_id)
            
            if not source_node or not target_node:
                continue
            
            # Filter by subject
            if subject and source_node.name.lower() != subject.lower():
                continue
            
            # Filter by object
            if obj and target_node.name.lower() != obj.lower():
                continue
            
            results.append((source_node, edge, target_node))
        
        return results
    
    def find_path(
        self,
        start: str,
        end: str,
        max_depth: int = 5,
    ) -> Optional[List[Tuple[KnowledgeNode, KnowledgeEdge]]]:
        """Find a path between two concepts using BFS."""
        start_node = self.get_node_by_name(start)
        end_node = self.get_node_by_name(end)
        
        if not start_node or not end_node:
            return None
        
        # BFS
        visited = {start_node.id}
        queue = [(start_node, [])]  # (node, path)
        
        while queue:
            current, path = queue.pop(0)
            
            if len(path) >= max_depth:
                continue
            
            for edge in self.get_edges(current.id, direction="out"):
                next_id = edge.target_id if edge.source_id == current.id else edge.source_id
                
                if next_id == end_node.id:
                    return path + [(current, edge), (end_node, None)]
                
                if next_id not in visited:
                    visited.add(next_id)
                    next_node = self.nodes[next_id]
                    queue.append((next_node, path + [(current, edge)]))
        
        return None
    
    def get_context(self, query: str, max_items: int = 10) -> str:
        """Get relevant knowledge context for a query."""
        nodes = self.search_nodes(query=query, limit=max_items)
        
        context_parts = []
        for node in nodes:
            # Get node info
            context_parts.append(f"- {node.name}: {node.content}")
            
            # Get related facts
            for edge in self.get_edges(node.id, direction="out")[:3]:
                target = self.nodes.get(edge.target_id)
                if target:
                    context_parts.append(f"  → {edge.relation} → {target.name}")
        
        return "\n".join(context_parts)
    
    # ========================================================================
    # VERIFICATION & CONFIDENCE
    # ========================================================================
    
    def verify_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
    ) -> Tuple[bool, float, str]:
        """
        Verify if a fact exists and is correct.
        
        Returns: (exists, confidence, explanation)
        """
        results = self.query_knowledge(subject=subject, predicate=predicate)
        
        for subj_node, edge, obj_node in results:
            if obj_node.name.lower() == obj.lower():
                return True, edge.confidence, f"Found: {subj_node.name} {edge.relation} {obj_node.name}"
        
        # Check for contradictions
        for subj_node, edge, obj_node in results:
            if edge.relation == predicate:
                return False, 0.0, f"Contradiction: {subj_node.name} {edge.relation} {obj_node.name}, not {obj}"
        
        return False, 0.0, "Fact not found in knowledge base"
    
    def update_confidence(
        self,
        node_id: str,
        delta: float,
        reason: str = None,
    ):
        """Update confidence for a node."""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.confidence = max(0, min(1, node.confidence + delta))
            node.updated_at = timestamp()
            if reason:
                node.metadata["confidence_history"] = node.metadata.get("confidence_history", [])
                node.metadata["confidence_history"].append({
                    "delta": delta,
                    "reason": reason,
                    "timestamp": timestamp(),
                })
    
    def mark_verified(self, node_id: str, verified: bool = True, source: str = None):
        """Mark a node as verified/unverified."""
        if node_id in self.nodes:
            self.nodes[node_id].verified = verified
            self.nodes[node_id].updated_at = timestamp()
            if source:
                self.nodes[node_id].metadata["verification_source"] = source
    
    # ========================================================================
    # STATISTICS & EXPORT
    # ========================================================================
    
    def get_stats(self) -> Dict:
        """Get knowledge graph statistics."""
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": {t: len(ids) for t, ids in self._type_index.items()},
            "relation_types": {r: len(ids) for r, ids in self._relation_index.items()},
            "verified_nodes": sum(1 for n in self.nodes.values() if n.verified),
            "avg_confidence": sum(n.confidence for n in self.nodes.values()) / max(1, len(self.nodes)),
            **self.stats,
        }
    
    def export_to_text(self) -> str:
        """Export knowledge graph as human-readable text."""
        lines = [
            f"# GroundZero Knowledge Graph",
            f"# Nodes: {len(self.nodes)}, Edges: {len(self.edges)}",
            "",
            "## Nodes",
        ]
        
        for node in self.nodes.values():
            lines.append(f"- [{node.node_type}] {node.name}: {node.content[:100]}...")
        
        lines.extend(["", "## Relationships"])
        
        for edge in self.edges.values():
            source = self.nodes.get(edge.source_id, KnowledgeNode("?", "?", "?", "?"))
            target = self.nodes.get(edge.target_id, KnowledgeNode("?", "?", "?", "?"))
            lines.append(f"- {source.name} --[{edge.relation}]--> {target.name}")
        
        return "\n".join(lines)


# ============================================================================
# KNOWLEDGE EXTRACTOR (Extract from text)
# ============================================================================

class KnowledgeExtractor:
    """Extract knowledge triples from text."""
    
    # Patterns for extracting relationships
    PATTERNS = [
        # "X is a Y"
        (r"(\w+(?:\s+\w+)*)\s+is\s+an?\s+(\w+(?:\s+\w+)*)", "is_a"),
        # "X are Y"
        (r"(\w+(?:\s+\w+)*)\s+are\s+(\w+(?:\s+\w+)*)", "is_a"),
        # "X is part of Y"
        (r"(\w+(?:\s+\w+)*)\s+is\s+part\s+of\s+(\w+(?:\s+\w+)*)", "part_of"),
        # "X contains Y"
        (r"(\w+(?:\s+\w+)*)\s+contains?\s+(\w+(?:\s+\w+)*)", "has_part"),
        # "X causes Y"
        (r"(\w+(?:\s+\w+)*)\s+causes?\s+(\w+(?:\s+\w+)*)", "causes"),
        # "X is used for Y"
        (r"(\w+(?:\s+\w+)*)\s+is\s+used\s+for\s+(\w+(?:\s+\w+)*)", "used_for"),
        # "X is defined as Y"
        (r"(\w+(?:\s+\w+)*)\s+is\s+defined\s+as\s+(.+?)(?:\.|$)", "defined_as"),
        # "X was created by Y"
        (r"(\w+(?:\s+\w+)*)\s+was\s+created\s+by\s+(\w+(?:\s+\w+)*)", "created_by"),
        # "X is located in Y"
        (r"(\w+(?:\s+\w+)*)\s+is\s+located\s+in\s+(\w+(?:\s+\w+)*)", "located_in"),
        # "X, also known as Y"
        (r"(\w+(?:\s+\w+)*),?\s+also\s+known\s+as\s+(\w+(?:\s+\w+)*)", "same_as"),
    ]
    
    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
        self.compiled_patterns = [(re.compile(p, re.IGNORECASE), r) for p, r in self.PATTERNS]
    
    def extract_from_text(self, text: str, source: str = "extraction") -> List[Tuple]:
        """Extract knowledge triples from text."""
        triples = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            for pattern, relation in self.compiled_patterns:
                matches = pattern.findall(sentence)
                for match in matches:
                    if len(match) >= 2:
                        subject = match[0].strip()
                        obj = match[1].strip()
                        
                        # Filter out very short or very long items
                        if 2 <= len(subject) <= 50 and 2 <= len(obj) <= 100:
                            triples.append((subject, relation, obj))
        
        return triples
    
    def extract_and_store(self, text: str, source: str = "extraction") -> int:
        """Extract triples from text and add to knowledge graph."""
        triples = self.extract_from_text(text, source)
        
        count = 0
        for subject, predicate, obj in triples:
            try:
                self.graph.add_knowledge(subject, predicate, obj, source=source)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to add triple: {e}")
        
        return count


# Export
__all__ = [
    'KnowledgeNode', 'KnowledgeEdge', 'KnowledgeGraph', 'KnowledgeExtractor',
]
