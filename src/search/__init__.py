"""
GroundZero AI - Web Search & Verification System
================================================

Search the web when knowledge is not available, verify information,
and store verified knowledge.

Features:
1. Multi-engine search (DuckDuckGo, Wikipedia, arXiv)
2. Source verification and reliability scoring
3. Content extraction and summarization
4. Fact verification against multiple sources
5. Deep learning mode for comprehensive research
"""

import re
import json
import hashlib
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from urllib.parse import quote_plus, urlparse
import concurrent.futures

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
class SearchResult:
    """A search result from any engine."""
    title: str
    url: str
    snippet: str
    source: str  # search engine name
    relevance_score: float = 0.0
    reliability_score: float = 0.5
    timestamp: str = field(default_factory=timestamp)
    full_content: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SearchResult':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class VerificationResult:
    """Result of fact verification."""
    claim: str
    verified: bool
    confidence: float
    supporting_sources: List[SearchResult]
    contradicting_sources: List[SearchResult]
    explanation: str
    timestamp: str = field(default_factory=timestamp)


# ============================================================================
# SEARCH ENGINES
# ============================================================================

class SearchEngine:
    """Base class for search engines."""
    
    name: str = "base"
    reliability_base: float = 0.5
    
    def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        raise NotImplementedError
    
    def fetch_content(self, url: str) -> Optional[str]:
        """Fetch full content from URL."""
        try:
            import requests
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            if response.status_code == 200:
                return response.text
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
        return None


class DuckDuckGoSearch(SearchEngine):
    """DuckDuckGo search engine."""
    
    name = "duckduckgo"
    reliability_base = 0.6
    
    def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        results = []
        
        try:
            from duckduckgo_search import DDGS
            
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(SearchResult(
                        title=r.get("title", ""),
                        url=r.get("href", ""),
                        snippet=r.get("body", ""),
                        source=self.name,
                        reliability_score=self._score_reliability(r.get("href", "")),
                    ))
        except ImportError:
            logger.warning("duckduckgo_search not installed. Install with: pip install duckduckgo-search")
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
        
        return results
    
    def _score_reliability(self, url: str) -> float:
        """Score URL reliability based on domain."""
        domain = urlparse(url).netloc.lower()
        
        # High reliability domains
        if any(d in domain for d in ['.edu', '.gov', 'wikipedia.org', 'arxiv.org']):
            return 0.9
        
        # Known reliable sources
        reliable = ['bbc.com', 'reuters.com', 'nature.com', 'science.org', 'github.com']
        if any(r in domain for r in reliable):
            return 0.85
        
        # Medium reliability
        if any(d in domain for d in ['.org', 'medium.com', 'stackoverflow.com']):
            return 0.7
        
        return self.reliability_base


class WikipediaSearch(SearchEngine):
    """Wikipedia search engine."""
    
    name = "wikipedia"
    reliability_base = 0.85
    
    def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        results = []
        
        try:
            import wikipedia
            
            # Search for pages
            search_results = wikipedia.search(query, results=max_results)
            
            for title in search_results[:max_results]:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    results.append(SearchResult(
                        title=page.title,
                        url=page.url,
                        snippet=page.summary[:500],
                        source=self.name,
                        reliability_score=self.reliability_base,
                        full_content=page.content[:5000],
                    ))
                except (wikipedia.DisambiguationError, wikipedia.PageError):
                    continue
        except ImportError:
            logger.warning("wikipedia not installed. Install with: pip install wikipedia")
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
        
        return results


class ArxivSearch(SearchEngine):
    """arXiv search for academic papers."""
    
    name = "arxiv"
    reliability_base = 0.9
    
    def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        results = []
        
        try:
            import arxiv
            
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            for paper in search.results():
                results.append(SearchResult(
                    title=paper.title,
                    url=paper.entry_id,
                    snippet=paper.summary[:500],
                    source=self.name,
                    reliability_score=self.reliability_base,
                    metadata={
                        "authors": [a.name for a in paper.authors],
                        "published": str(paper.published),
                        "categories": paper.categories,
                    }
                ))
        except ImportError:
            logger.warning("arxiv not installed. Install with: pip install arxiv")
        except Exception as e:
            logger.error(f"arXiv search error: {e}")
        
        return results


# ============================================================================
# SEARCH MANAGER
# ============================================================================

class SearchManager:
    """
    Unified search across multiple engines with caching and verification.
    """
    
    # Domain reliability scores
    DOMAIN_SCORES = {
        "wikipedia.org": 0.9,
        "arxiv.org": 0.9,
        "github.com": 0.85,
        "stackoverflow.com": 0.85,
        "python.org": 0.9,
        "docs.python.org": 0.9,
        "pytorch.org": 0.9,
        "tensorflow.org": 0.9,
        "huggingface.co": 0.85,
        "nature.com": 0.9,
        "science.org": 0.9,
        "bbc.com": 0.8,
        "reuters.com": 0.85,
        "medium.com": 0.6,
        "reddit.com": 0.5,
        "quora.com": 0.5,
    }
    
    def __init__(self, cache_path: str = None, cache_hours: int = 24):
        self.cache_path = Path(cache_path) if cache_path else get_data_path("cache", "search_cache.json")
        ensure_dir(self.cache_path.parent)
        
        self.cache_hours = cache_hours
        self.cache: Dict[str, Dict] = {}
        
        # Initialize search engines
        self.engines = {
            "duckduckgo": DuckDuckGoSearch(),
            "wikipedia": WikipediaSearch(),
            "arxiv": ArxivSearch(),
        }
        
        self._load_cache()
    
    def _load_cache(self):
        """Load search cache."""
        if self.cache_path.exists():
            self.cache = load_json(self.cache_path)
    
    def _save_cache(self):
        """Save search cache."""
        # Clean old entries
        cutoff = datetime.now() - timedelta(hours=self.cache_hours)
        self.cache = {
            k: v for k, v in self.cache.items()
            if datetime.fromisoformat(v.get("timestamp", "2000-01-01")) > cutoff
        }
        save_json(self.cache_path, self.cache)
    
    def _cache_key(self, query: str, engines: List[str]) -> str:
        """Generate cache key."""
        content = f"{query}:{','.join(sorted(engines))}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def search(
        self,
        query: str,
        engines: List[str] = None,
        max_results: int = 10,
        use_cache: bool = True,
    ) -> List[SearchResult]:
        """
        Search across multiple engines.
        """
        if engines is None:
            engines = ["duckduckgo", "wikipedia"]
        
        # Check cache
        cache_key = self._cache_key(query, engines)
        if use_cache and cache_key in self.cache:
            cached = self.cache[cache_key]
            if datetime.fromisoformat(cached["timestamp"]) > datetime.now() - timedelta(hours=self.cache_hours):
                logger.debug(f"Cache hit for query: {query[:50]}")
                return [SearchResult.from_dict(r) for r in cached["results"]]
        
        # Search each engine
        all_results = []
        
        for engine_name in engines:
            engine = self.engines.get(engine_name)
            if not engine:
                continue
            
            try:
                results = engine.search(query, max_results=max_results)
                all_results.extend(results)
                logger.debug(f"{engine_name}: {len(results)} results")
            except Exception as e:
                logger.error(f"Error searching {engine_name}: {e}")
        
        # Remove duplicates by URL
        seen_urls = set()
        unique_results = []
        for r in all_results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                unique_results.append(r)
        
        # Score relevance
        self._score_relevance(query, unique_results)
        
        # Sort by combined score
        unique_results.sort(
            key=lambda r: r.relevance_score * 0.4 + r.reliability_score * 0.6,
            reverse=True
        )
        
        # Cache results
        self.cache[cache_key] = {
            "query": query,
            "timestamp": timestamp(),
            "results": [r.to_dict() for r in unique_results],
        }
        self._save_cache()
        
        return unique_results[:max_results]
    
    def _score_relevance(self, query: str, results: List[SearchResult]):
        """Score relevance of results to query."""
        query_words = set(query.lower().split())
        
        for result in results:
            # Title match
            title_words = set(result.title.lower().split())
            title_overlap = len(query_words & title_words) / max(1, len(query_words))
            
            # Snippet match
            snippet_words = set(result.snippet.lower().split())
            snippet_overlap = len(query_words & snippet_words) / max(1, len(query_words))
            
            # Combined score
            result.relevance_score = title_overlap * 0.4 + snippet_overlap * 0.3 + 0.3
    
    def get_domain_reliability(self, url: str) -> float:
        """Get reliability score for a URL domain."""
        domain = urlparse(url).netloc.lower()
        
        # Check exact matches
        if domain in self.DOMAIN_SCORES:
            return self.DOMAIN_SCORES[domain]
        
        # Check partial matches
        for known_domain, score in self.DOMAIN_SCORES.items():
            if known_domain in domain:
                return score
        
        # TLD-based scoring
        if domain.endswith('.edu') or domain.endswith('.gov'):
            return 0.85
        if domain.endswith('.org'):
            return 0.7
        
        return 0.5


# ============================================================================
# FACT VERIFIER
# ============================================================================

class FactVerifier:
    """
    Verify facts using multiple sources and cross-referencing.
    """
    
    def __init__(self, search_manager: SearchManager):
        self.search = search_manager
    
    def verify(
        self,
        claim: str,
        min_sources: int = 3,
        min_confidence: float = 0.7,
    ) -> VerificationResult:
        """
        Verify a claim by searching for evidence.
        """
        logger.info(f"Verifying claim: {claim[:100]}")
        
        # Search for evidence
        results = self.search.search(claim, max_results=10)
        
        supporting = []
        contradicting = []
        
        # Analyze each result
        for result in results:
            support_score = self._analyze_support(claim, result)
            
            if support_score > 0.6:
                supporting.append(result)
            elif support_score < 0.4:
                contradicting.append(result)
        
        # Calculate confidence
        if not supporting and not contradicting:
            verified = False
            confidence = 0.0
            explanation = "No evidence found to verify or contradict this claim."
        elif len(supporting) >= min_sources and not contradicting:
            verified = True
            avg_reliability = sum(r.reliability_score for r in supporting) / len(supporting)
            confidence = min(0.95, avg_reliability * (len(supporting) / min_sources))
            explanation = f"Verified by {len(supporting)} reliable sources."
        elif contradicting and not supporting:
            verified = False
            confidence = sum(r.reliability_score for r in contradicting) / len(contradicting)
            explanation = f"Contradicted by {len(contradicting)} sources."
        else:
            # Mixed evidence
            support_weight = sum(r.reliability_score for r in supporting)
            contradict_weight = sum(r.reliability_score for r in contradicting)
            
            if support_weight > contradict_weight * 1.5:
                verified = True
                confidence = support_weight / (support_weight + contradict_weight)
                explanation = f"Mostly supported ({len(supporting)} for, {len(contradicting)} against)."
            else:
                verified = False
                confidence = 0.5
                explanation = f"Conflicting evidence ({len(supporting)} for, {len(contradicting)} against)."
        
        return VerificationResult(
            claim=claim,
            verified=verified,
            confidence=confidence,
            supporting_sources=supporting,
            contradicting_sources=contradicting,
            explanation=explanation,
        )
    
    def _analyze_support(self, claim: str, result: SearchResult) -> float:
        """Analyze if a search result supports the claim."""
        claim_words = set(claim.lower().split())
        content = f"{result.title} {result.snippet}".lower()
        content_words = set(content.split())
        
        # Word overlap
        overlap = len(claim_words & content_words) / max(1, len(claim_words))
        
        # Check for negation words near claim words
        negation_words = ["not", "no", "never", "false", "incorrect", "wrong", "untrue"]
        has_negation = any(neg in content for neg in negation_words)
        
        if has_negation:
            # Check if negation is close to claim keywords
            for neg in negation_words:
                if neg in content:
                    # Simple proximity check
                    neg_idx = content.find(neg)
                    for word in claim_words:
                        if word in content:
                            word_idx = content.find(word)
                            if abs(neg_idx - word_idx) < 50:  # Within ~10 words
                                return 0.2
        
        return overlap * 0.5 + 0.3 + (result.reliability_score * 0.2)


# ============================================================================
# DEEP RESEARCHER
# ============================================================================

class DeepResearcher:
    """
    Conduct deep research on a topic - find sources, learn how to implement, etc.
    """
    
    def __init__(self, search_manager: SearchManager, verifier: FactVerifier):
        self.search = search_manager
        self.verifier = verifier
    
    def research(
        self,
        topic: str,
        depth: str = "standard",  # quick, standard, deep, comprehensive
        focus: List[str] = None,  # e.g., ["implementation", "examples", "theory"]
    ) -> Dict[str, Any]:
        """
        Conduct comprehensive research on a topic.
        """
        logger.info(f"Starting {depth} research on: {topic}")
        
        research = {
            "topic": topic,
            "depth": depth,
            "timestamp": timestamp(),
            "overview": None,
            "key_concepts": [],
            "sources": [],
            "implementation_guide": None,
            "examples": [],
            "related_topics": [],
            "summary": None,
        }
        
        # Determine search queries based on depth
        queries = self._generate_queries(topic, depth, focus)
        
        # Execute searches
        all_results = []
        for query in queries:
            results = self.search.search(query, max_results=5)
            all_results.extend(results)
            time.sleep(0.5)  # Rate limiting
        
        # Remove duplicates
        seen = set()
        unique_results = []
        for r in all_results:
            if r.url not in seen:
                seen.add(r.url)
                unique_results.append(r)
        
        research["sources"] = [r.to_dict() for r in unique_results[:20]]
        
        # Extract overview from Wikipedia if available
        wiki_results = [r for r in unique_results if r.source == "wikipedia"]
        if wiki_results:
            research["overview"] = wiki_results[0].full_content or wiki_results[0].snippet
        
        # Extract key concepts
        research["key_concepts"] = self._extract_concepts(topic, unique_results)
        
        # Look for implementation guides
        if not focus or "implementation" in focus:
            impl_results = self.search.search(f"how to implement {topic}", max_results=5)
            if impl_results:
                research["implementation_guide"] = {
                    "sources": [r.to_dict() for r in impl_results],
                    "summary": self._summarize_implementation(impl_results),
                }
        
        # Look for examples
        if not focus or "examples" in focus:
            example_results = self.search.search(f"{topic} example code tutorial", max_results=5)
            research["examples"] = [r.to_dict() for r in example_results]
        
        # Find related topics
        research["related_topics"] = self._find_related(topic, unique_results)
        
        # Generate summary
        research["summary"] = self._generate_summary(topic, research)
        
        return research
    
    def _generate_queries(self, topic: str, depth: str, focus: List[str] = None) -> List[str]:
        """Generate search queries based on depth and focus."""
        queries = [topic]
        
        if depth in ["standard", "deep", "comprehensive"]:
            queries.extend([
                f"what is {topic}",
                f"{topic} explanation",
            ])
        
        if depth in ["deep", "comprehensive"]:
            queries.extend([
                f"{topic} how it works",
                f"{topic} tutorial",
                f"{topic} implementation",
            ])
        
        if depth == "comprehensive":
            queries.extend([
                f"{topic} best practices",
                f"{topic} advanced",
                f"{topic} research paper",
                f"{topic} vs alternatives",
            ])
        
        # Add focus-specific queries
        if focus:
            for f in focus:
                queries.append(f"{topic} {f}")
        
        return queries
    
    def _extract_concepts(self, topic: str, results: List[SearchResult]) -> List[str]:
        """Extract key concepts from search results."""
        concepts = set()
        
        # Common technical term patterns
        for result in results:
            text = f"{result.title} {result.snippet}"
            
            # Find capitalized terms
            caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            concepts.update(caps[:5])
            
            # Find quoted terms
            quoted = re.findall(r'"([^"]+)"', text)
            concepts.update(quoted[:3])
        
        # Remove topic itself and very common words
        concepts = {c for c in concepts if c.lower() != topic.lower() and len(c) > 2}
        
        return list(concepts)[:15]
    
    def _summarize_implementation(self, results: List[SearchResult]) -> str:
        """Summarize implementation steps from results."""
        steps = []
        
        for result in results[:3]:
            snippet = result.snippet
            
            # Look for numbered steps
            numbered = re.findall(r'\d+[.)]\s*([^.]+)', snippet)
            steps.extend(numbered[:3])
            
            # Look for bullet points or list items
            bullets = re.findall(r'[-•]\s*([^.]+)', snippet)
            steps.extend(bullets[:3])
        
        if not steps:
            return "See sources for implementation details."
        
        return "Key steps: " + "; ".join(steps[:5])
    
    def _find_related(self, topic: str, results: List[SearchResult]) -> List[str]:
        """Find related topics from search results."""
        related = set()
        
        # Extract from "see also", "related" sections
        for result in results:
            text = f"{result.title} {result.snippet}".lower()
            
            # Find "related to", "similar to", "see also" patterns
            patterns = [
                r'related to ([^,\.]+)',
                r'similar to ([^,\.]+)',
                r'see also:? ([^,\.]+)',
                r'also known as ([^,\.]+)',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text)
                related.update(matches[:2])
        
        # Remove topic itself
        related = {r.strip() for r in related if topic.lower() not in r.lower() and len(r) > 2}
        
        return list(related)[:10]
    
    def _generate_summary(self, topic: str, research: Dict) -> str:
        """Generate research summary."""
        parts = [f"Research on: {topic}"]
        
        if research["overview"]:
            parts.append(f"\nOverview: {research['overview'][:500]}...")
        
        if research["key_concepts"]:
            parts.append(f"\nKey concepts: {', '.join(research['key_concepts'][:10])}")
        
        if research["implementation_guide"]:
            parts.append(f"\nImplementation: {research['implementation_guide']['summary']}")
        
        parts.append(f"\nSources: {len(research['sources'])} found")
        
        return "\n".join(parts)


# ============================================================================
# WEB SEARCH INTERFACE
# ============================================================================

class WebSearch:
    """
    Main interface for web search functionality.
    """
    
    def __init__(self):
        self.search_manager = SearchManager()
        self.verifier = FactVerifier(self.search_manager)
        self.researcher = DeepResearcher(self.search_manager, self.verifier)
    
    def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Simple search."""
        return self.search_manager.search(query, max_results=max_results)
    
    def verify(self, claim: str) -> VerificationResult:
        """Verify a fact."""
        return self.verifier.verify(claim)
    
    def research(self, topic: str, depth: str = "standard") -> Dict[str, Any]:
        """Conduct deep research."""
        return self.researcher.research(topic, depth=depth)
    
    def learn_topic(
        self,
        topic: str,
        knowledge_graph=None,
    ) -> Dict[str, Any]:
        """
        Learn about a topic and optionally store in knowledge graph.
        """
        # Conduct deep research
        research = self.research(topic, depth="deep")
        
        # Store in knowledge graph if provided
        if knowledge_graph and research["key_concepts"]:
            # Add main topic node
            knowledge_graph.add_node(
                name=topic,
                content=research["overview"] or research["summary"],
                node_type="concept",
                source="web_research",
            )
            
            # Add concepts and connect
            for concept in research["key_concepts"]:
                knowledge_graph.add_knowledge(
                    topic, "related_to", concept,
                    source="web_research",
                    confidence=0.7,
                )
        
        return research


# Export
__all__ = [
    'SearchResult', 'VerificationResult',
    'SearchEngine', 'DuckDuckGoSearch', 'WikipediaSearch', 'ArxivSearch',
    'SearchManager', 'FactVerifier', 'DeepResearcher', 'WebSearch',
]
