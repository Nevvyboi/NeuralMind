"""
Continuous Learner
==================
Orchestrates background learning from multiple web sources.
Now learns from Wikipedia, DuckDuckGo results, news sites, blogs, and more.
"""

import threading
import time
import random
import re
from queue import Queue, Empty
from typing import Optional, Callable, List, Dict, Any
from datetime import datetime

from web import WikipediaCrawler, ContentExtractor, WikipediaSearch
from storage import MemoryStore

# Import advanced multi-source learning
try:
    from .advanced_scraper import UniversalScraper, MultiSourceSearch
    MULTI_SOURCE_AVAILABLE = True
except ImportError:
    MULTI_SOURCE_AVAILABLE = False


class ContinuousLearner:
    """
    Background learning orchestrator.
    Continuously fetches and processes web content from MULTIPLE sources:
    - Wikipedia (primary for educational content)
    - DuckDuckGo search results
    - News sites
    - Blogs and general websites
    - Any accessible URL
    """
    
    def __init__(
        self,
        memory_store: MemoryStore,
        neural_model,
        seed_urls: List[str],
        target_sites: int = 50,
        chunk_size: int = 500,
        request_delay: float = 1.0
    ):
        self.memory = memory_store
        self.model = neural_model
        self.seed_urls = seed_urls
        self.target_sites = target_sites
        self.chunk_size = chunk_size
        
        # Primary source - Wikipedia
        self.wiki_search = WikipediaSearch()
        self.extractor = ContentExtractor()
        
        # Multi-source learning (DuckDuckGo, news, blogs, etc.)
        if MULTI_SOURCE_AVAILABLE:
            self.web_scraper = UniversalScraper()
            self.multi_search = MultiSourceSearch()
        else:
            self.web_scraper = None
            self.multi_search = None
        
        # Queues for different source types
        self.topic_queue: Queue = Queue()     # Wikipedia topics
        self.url_queue: Queue = Queue()        # Direct URLs to learn from
        self.search_queue: Queue = Queue()     # Search queries
        
        # State
        self.is_running = False
        self.learning_thread: Optional[threading.Thread] = None
        
        # Learning mode: 'wiki', 'web', 'search' - rotate between them
        self.learning_mode = 'wiki'
        self.mode_counter = 0
        
        # Statistics
        self.sites_learned = 0
        self.sites_skipped = 0
        self.total_chunks = 0
        self.total_words = 0
        self.errors: List[Dict] = []
        self.sources_by_type = {'wikipedia': 0, 'web': 0, 'news': 0, 'blog': 0}
        
        # Current state
        self.current_url: Optional[str] = None
        self.current_title: Optional[str] = None
        self.current_preview: Optional[str] = None
        self.current_source_type: str = 'wikipedia'
        
        # Callbacks
        self.on_progress: Optional[Callable] = None
        self.on_content: Optional[Callable] = None
        self.on_complete: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # Initialize queues
        self._seed_queue()
        self._seed_search_queries()
    
    def _seed_queue(self) -> None:
        """Populate queue with seed topics from URLs"""
        topics = []
        for url in self.seed_urls:
            if '/wiki/' in url:
                topic = url.split('/wiki/')[-1].replace('_', ' ')
                topics.append(topic)
        
        random.shuffle(topics)
        for topic in topics:
            self.topic_queue.put(topic)
    
    def _seed_search_queries(self) -> None:
        """Populate search queue with diverse topics to learn about"""
        # Diverse educational topics for web search
        search_topics = [
            # Science
            "machine learning basics", "quantum computing explained", "climate change science",
            "human biology systems", "space exploration history", "renewable energy technologies",
            "genetic engineering", "artificial intelligence ethics", "neuroscience discoveries",
            
            # Technology
            "programming languages comparison", "cybersecurity best practices", "cloud computing",
            "blockchain technology", "internet history", "software development methodologies",
            "mobile app development", "data science techniques", "web development frameworks",
            
            # History
            "world war 2 events", "ancient civilizations", "industrial revolution",
            "civil rights movement", "cold war history", "renaissance period",
            
            # Business & Economics
            "startup entrepreneurship", "stock market basics", "global economics",
            "marketing strategies", "leadership principles", "personal finance",
            
            # Arts & Culture
            "music history genres", "art movements", "film history",
            "literature classics", "architecture styles", "photography techniques",
            
            # Health & Wellness
            "nutrition science", "exercise physiology", "mental health awareness",
            "medical breakthroughs", "public health", "sleep science",
            
            # Philosophy & Psychology
            "philosophy schools of thought", "cognitive psychology", "behavioral economics",
            "critical thinking", "emotional intelligence", "decision making",
        ]
        
        random.shuffle(search_topics)
        for topic in search_topics:
            self.search_queue.put(topic)
    
    def start(self) -> Dict[str, Any]:
        """Start background learning - truly continuous, no limits"""
        if self.is_running:
            return {'status': 'already_running'}
        
        # No target limit - learn continuously
        self.is_running = True
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learning_thread.start()
        
        return {'status': 'started'}
    
    def stop(self) -> Dict[str, Any]:
        """Stop background learning"""
        self.is_running = False
        if self.learning_thread:
            self.learning_thread.join(timeout=2)
        
        # Save model state
        self.model.save()
        
        return {
            'status': 'stopped',
            'sites_learned': self.sites_learned,
            'total_chunks': self.total_chunks
        }
    
    def reset(self) -> Dict[str, Any]:
        """Reset learning progress"""
        self.stop()
        
        self.sites_learned = 0
        self.sites_skipped = 0
        self.total_chunks = 0
        self.total_words = 0
        self.errors = []
        
        # Clear and reseed queue
        while not self.topic_queue.empty():
            try:
                self.topic_queue.get_nowait()
            except Empty:
                break
        
        self._seed_queue()
        
        return {'status': 'reset'}
    
    def _learning_loop(self) -> None:
        """Main learning loop - rotates between multiple sources"""
        while self.is_running:
            try:
                success = False
                
                # Rotate between learning modes
                self.mode_counter += 1
                
                # Every 3 iterations, switch mode
                if self.mode_counter % 3 == 0:
                    self.learning_mode = 'search' if MULTI_SOURCE_AVAILABLE else 'wiki'
                elif self.mode_counter % 3 == 1:
                    self.learning_mode = 'web' if MULTI_SOURCE_AVAILABLE else 'wiki'
                else:
                    self.learning_mode = 'wiki'
                
                # Try different learning approaches
                if self.learning_mode == 'wiki':
                    success = self._learn_from_wikipedia()
                elif self.learning_mode == 'search' and self.multi_search:
                    success = self._learn_from_web_search()
                elif self.learning_mode == 'web' and self.web_scraper:
                    success = self._learn_from_url_queue()
                else:
                    success = self._learn_from_wikipedia()
                
                if success:
                    self.sites_learned += 1
                
                # Emit progress
                if self.on_progress:
                    self.on_progress(self.get_stats())
                
                # Rate limiting - variable based on source
                delay = 1.5 if self.learning_mode == 'wiki' else 2.0
                time.sleep(delay)
                
            except Exception as e:
                self._handle_error(str(e))
                time.sleep(2.0)
    
    def _learn_from_wikipedia(self) -> bool:
        """Learn from Wikipedia - primary source"""
        topic = None
        try:
            topic = self.topic_queue.get(timeout=0.5)
        except Empty:
            self._add_random_topics()
            return False
        
        if topic:
            result = self._learn_from_topic(topic)
            return result.get('success', False)
        return False
    
    def _learn_from_web_search(self) -> bool:
        """Learn from DuckDuckGo search results"""
        if not self.multi_search:
            return False
        
        query = None
        try:
            query = self.search_queue.get(timeout=0.5)
        except Empty:
            # Reseed search queries
            self._seed_search_queries()
            return False
        
        if not query:
            return False
        
        try:
            # Search DuckDuckGo
            results = self.multi_search.search(query, max_results=3)
            
            if not results:
                return False
            
            # Learn from the top result
            for result in results[:2]:
                url = result.url
                
                # Skip already learned
                if self.memory.is_source_learned(url):
                    continue
                
                # Skip Wikipedia (we handle that separately)
                if 'wikipedia.org' in url:
                    continue
                
                # Scrape and learn
                learned = self._learn_from_url(url, query)
                if learned:
                    return True
            
            return False
            
        except Exception as e:
            self._handle_error(f"Web search error: {e}")
            return False
    
    def _learn_from_url_queue(self) -> bool:
        """Learn from URLs in the URL queue"""
        url = None
        try:
            url = self.url_queue.get(timeout=0.5)
        except Empty:
            return False
        
        if url:
            return self._learn_from_url(url)
        return False
    
    def _learn_from_url(self, url: str, topic: str = None) -> bool:
        """Learn from any URL using universal scraper"""
        if not self.web_scraper:
            return False
        
        # Skip if already learned
        if self.memory.is_source_learned(url):
            return False
        
        try:
            # Scrape the URL
            scraped = self.web_scraper.scrape(url)
            
            if not scraped or not scraped.success:
                return False
            
            if len(scraped.text) < 200:
                return False
            
            # Update current state
            self.current_url = url
            self.current_title = scraped.title or topic or url
            self.current_preview = scraped.text[:500]
            self.current_source_type = scraped.source_type
            
            # Emit content update
            if self.on_content:
                self.on_content({
                    'url': url,
                    'title': self.current_title,
                    'preview': self.current_preview,
                    'length': len(scraped.text),
                    'source_type': scraped.source_type
                })
            
            # Chunk and learn
            chunks = self.extractor.chunk_text(scraped.text, self.chunk_size)
            chunks_learned = 0
            words_learned = 0
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 50:
                    continue
                
                # Update preview with current chunk
                self.current_preview = chunk[:500]
                
                # Emit progress periodically
                if self.on_progress and i % 2 == 0:
                    self.on_progress(self.get_stats())
                
                # Learn from chunk
                learn_result = self.model.learn_from_text(chunk, source=url)
                chunks_learned += 1
                words_learned += learn_result.get('new_words', 0)
                
                # Store knowledge
                summary = chunk[:200] if len(chunk) > 200 else chunk
                self.memory.store_knowledge(
                    content=chunk,
                    summary=summary,
                    source_url=url,
                    source_title=self.current_title
                )
            
            # Mark as learned
            self.memory.mark_source_learned(
                url=url,
                title=self.current_title,
                content_length=len(scraped.text),
                chunks_learned=chunks_learned,
                words_learned=words_learned
            )
            
            self.total_chunks += chunks_learned
            self.total_words += words_learned
            self.sources_by_type[scraped.source_type] = self.sources_by_type.get(scraped.source_type, 0) + 1
            
            # Add related URLs to queue
            self._extract_urls_from_text(scraped.text, url)
            
            return True
            
        except Exception as e:
            self._handle_error(f"URL learning error ({url}): {e}")
            return False
    
    def _extract_urls_from_text(self, text: str, source_url: str) -> None:
        """Extract and queue interesting URLs from text"""
        # This is a placeholder - could be enhanced to find related links
        pass
    
    def _learn_from_topic(self, topic: str) -> Dict[str, Any]:
        """Learn from a topic using Wikipedia API"""
        url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
        
        # Skip if already learned
        if self.memory.is_source_learned(url):
            self.sites_skipped += 1
            return {'success': False, 'reason': 'already_learned'}
        
        # Don't set title until we have content - prevents mismatch
        self.current_url = url
        self.current_source_type = 'wikipedia'
        
        # Get content via API (more reliable than HTML scraping)
        content = self.wiki_search.get_content(topic)
        
        if not content or not content.get('content'):
            # Try searching for the topic
            search_results = self.wiki_search.search(topic, limit=1)
            if search_results:
                actual_title = search_results[0]['title']
                content = self.wiki_search.get_content(actual_title)
        
        if not content or not content.get('content'):
            return {'success': False, 'reason': 'no_content'}
        
        text = content['content']
        title = content.get('title', topic)
        
        if len(text) < 100:
            return {'success': False, 'reason': 'insufficient_content'}
        
        # Now set title and preview together (synced)
        self.current_title = title
        self.current_preview = text[:500]  # More preview text
        
        # Emit content update
        if self.on_content:
            self.on_content({
                'url': url,
                'title': title,
                'preview': self.current_preview,
                'length': len(text),
                'source_type': 'wikipedia'
            })
        
        # Chunk and learn
        chunks = self.extractor.chunk_text(text, self.chunk_size)
        chunks_learned = 0
        words_learned = 0
        
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:
                continue
            
            # Update preview to show current chunk being processed
            self.current_preview = chunk[:500]
            
            # Emit progress with current chunk
            if self.on_progress and i % 2 == 0:  # Every 2 chunks
                self.on_progress(self.get_stats())
            
            # Learn from chunk
            learn_result = self.model.learn_from_text(chunk, source=url)
            chunks_learned += 1
            words_learned += learn_result.get('new_words', 0)
            
            # Store knowledge
            summary = chunk[:200] if len(chunk) > 200 else chunk
            self.memory.store_knowledge(
                content=chunk,
                summary=summary,
                source_url=url,
                source_title=title
            )
        
        # Mark source as learned
        self.memory.mark_source_learned(
            url=url,
            title=title,
            content_length=len(text),
            chunks_learned=chunks_learned,
            words_learned=words_learned
        )
        
        self.total_chunks += chunks_learned
        self.total_words += words_learned
        self.sources_by_type['wikipedia'] = self.sources_by_type.get('wikipedia', 0) + 1
        
        # Extract related topics and add to queue
        self._add_related_topics(text)
        
        return {
            'success': True,
            'chunks': chunks_learned,
            'words': words_learned
        }
    
    def _add_related_topics(self, text: str) -> None:
        """Extract and add related topics from text"""
        # Find capitalized terms (potential Wikipedia topics)
        potential_topics = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        
        # Filter and add unique topics
        seen = set()
        count = 0
        for topic in potential_topics:
            if topic not in seen and len(topic) > 3 and self.topic_queue.qsize() < 100:
                seen.add(topic)
                self.topic_queue.put(topic)
                count += 1
                if count >= 5:  # Add up to 5 related topics
                    break
        
        # Also add interesting phrases as search queries for web search
        if self.search_queue.qsize() < 50:
            # Find interesting phrases (2-4 word combinations)
            phrases = re.findall(r'\b([A-Z][a-z]+(?:\s+[a-z]+){1,3})\b', text)
            for phrase in phrases[:3]:
                if len(phrase) > 10 and len(phrase) < 50:
                    self.search_queue.put(phrase)
    
    def _add_random_topics(self) -> None:
        """Add random Wikipedia articles to queue"""
        try:
            random_articles = self.wiki_search.get_random_articles(5)
            for article in random_articles:
                self.topic_queue.put(article['title'])
        except Exception as e:
            print(f"Error getting random articles: {e}")
    
    def _complete(self) -> None:
        """Handle learning completion"""
        self.is_running = False
        self.model.save()
        
        if self.on_complete:
            self.on_complete(self.get_stats())
    
    def _handle_error(self, error: str) -> None:
        """Handle and log errors"""
        self.errors.append({
            'url': self.current_url,
            'error': error,
            'time': datetime.now().isoformat()
        })
        
        if self.on_error:
            self.on_error(error)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        # No longer use target_sites for progress - show actual count
        return {
            'is_running': self.is_running,
            'progress': 0,  # Don't show progress bar anymore
            'is_complete': False,  # Never complete - continuous learning
            'sites_learned': self.sites_learned,
            'sites_skipped': self.sites_skipped,
            'total_chunks': self.total_chunks,
            'total_words': self.total_words,
            'queue_size': self.topic_queue.qsize() + self.search_queue.qsize(),
            'current_url': self.current_url,
            'current_title': self.current_title,
            'current_preview': self.current_preview,
            'current_source_type': getattr(self, 'current_source_type', 'wikipedia'),
            'learning_mode': self.learning_mode,
            'sources_by_type': self.sources_by_type,
            'error_count': len(self.errors)
        }
    
    def learn_from_url(self, url: str) -> Dict[str, Any]:
        """Learn from any URL (manual trigger) - now supports all URLs"""
        if self.memory.is_source_learned(url):
            return {'success': False, 'reason': 'already_learned'}
        
        result = None
        
        # Handle Wikipedia URLs
        if '/wiki/' in url:
            topic = url.split('/wiki/')[-1].replace('_', ' ')
            result = self._learn_from_topic(topic)
        # Handle any other URL with universal scraper
        elif self.web_scraper:
            success = self._learn_from_url(url)
            result = {'success': success}
        else:
            return {'success': False, 'reason': 'no_scraper_available'}
        
        if result.get('success'):
            self.sites_learned += 1
            self.model.save()
        
        return result
    
    def search_and_learn(self, query: str, max_articles: int = 3) -> Dict[str, Any]:
        """Search for a topic and learn from results"""
        results = self.wiki_search.search(query, limit=max_articles)
        learned_from = []
        
        if not results:
            return {
                'success': False,
                'query': query,
                'learned_from': [],
                'count': 0,
                'error': 'No search results found'
            }
        
        for result in results:
            title = result['title']
            url = result['url']
            
            if self.memory.is_source_learned(url):
                continue
            
            learn_result = self._learn_from_topic(title)
            
            if learn_result.get('success'):
                self.sites_learned += 1
                learned_from.append({
                    'title': title,
                    'url': url
                })
        
        if learned_from:
            self.model.save()
        
        return {
            'success': len(learned_from) > 0,
            'query': query,
            'learned_from': learned_from,
            'count': len(learned_from)
        }
    
    def learn_from_any_url(self, url: str) -> Dict[str, Any]:
        """
        Learn from ANY URL - not just Wikipedia.
        Uses universal scraper to extract content.
        """
        from web import UniversalScraper
        
        if self.memory.is_source_learned(url):
            return {'success': False, 'reason': 'already_learned'}
        
        scraper = UniversalScraper()
        result = scraper.fetch(url)
        
        if not result.get('success'):
            return {'success': False, 'reason': result.get('error', 'Failed to fetch')}
        
        content = result.get('content', '')
        title = result.get('title', 'Unknown')
        
        if len(content) < 100:
            return {'success': False, 'reason': 'insufficient_content'}
        
        # Chunk and learn
        chunks = self.extractor.chunk_text(content, self.chunk_size)
        chunks_learned = 0
        words_learned = 0
        
        for chunk in chunks:
            if len(chunk.strip()) < 50:
                continue
            
            learn_result = self.model.learn_from_text(chunk, source=url)
            chunks_learned += 1
            words_learned += learn_result.get('new_words', 0)
            
            summary = chunk[:200] if len(chunk) > 200 else chunk
            self.memory.store_knowledge(
                content=chunk,
                summary=summary,
                source_url=url,
                source_title=title
            )
        
        # Mark as learned
        self.memory.mark_source_learned(
            url=url,
            title=title,
            content_length=len(content),
            chunks_learned=chunks_learned,
            words_learned=words_learned
        )
        
        self.total_chunks += chunks_learned
        self.total_words += words_learned
        self.sites_learned += 1
        
        # Save
        self.model.save()
        
        return {
            'success': True,
            'title': title,
            'chunks': chunks_learned,
            'words': words_learned,
            'content_length': len(content)
        }
    
    def web_search_and_learn(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search the web and learn from top results.
        Uses DuckDuckGo for search (no API key needed).
        """
        from web import WebSearcher, UniversalScraper
        
        searcher = WebSearcher()
        scraper = UniversalScraper()
        
        # Search
        results = searcher.search(query, max_results=max_results)
        
        if not results:
            return {
                'success': False,
                'query': query,
                'error': 'No search results found',
                'learned_from': [],
                'count': 0
            }
        
        learned_from = []
        
        for result in results:
            url = result['url']
            
            # Skip if already learned
            if self.memory.is_source_learned(url):
                continue
            
            # Skip some domains that are problematic
            skip_domains = ['youtube.com', 'facebook.com', 'twitter.com', 'instagram.com', 'tiktok.com']
            if any(d in url.lower() for d in skip_domains):
                continue
            
            # Try to learn
            try:
                learn_result = self.learn_from_any_url(url)
                
                if learn_result.get('success'):
                    learned_from.append({
                        'title': learn_result.get('title', result.get('title', 'Unknown')),
                        'url': url,
                        'chunks': learn_result.get('chunks', 0)
                    })
            except Exception as e:
                print(f"Error learning from {url}: {e}")
                continue
            
            # Limit to avoid too many requests
            if len(learned_from) >= 3:
                break
        
        return {
            'success': len(learned_from) > 0,
            'query': query,
            'learned_from': learned_from,
            'count': len(learned_from),
            'results_found': len(results)
        }