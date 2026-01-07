"""
Strategic Learning Module
=========================
Intelligent Wikipedia learning that prioritizes important content:
1. Vital Articles - Wikipedia's most important ~50,000 articles
2. Category-Based - Learn all articles in specific categories
3. On-Demand - Learn when user asks about something unknown
4. Random - Fallback to random articles after priorities are done

This replaces random learning with strategic, comprehensive learning.
"""

import os
import json
import time
import random
import requests
import threading
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
from datetime import datetime
from urllib.parse import quote, unquote


class VitalArticlesProvider:
    """
    Provides Wikipedia's Vital Articles - the most important articles to learn.
    
    Wikipedia has tiered vital articles:
    - Level 1: 10 articles (most fundamental)
    - Level 2: 100 articles
    - Level 3: 1,000 articles
    - Level 4: 10,000 articles
    - Level 5: 50,000 articles
    
    We fetch and cache these lists for strategic learning.
    """
    
    API_URL = "https://en.wikipedia.org/w/api.php"
    
    # Vital article categories from Wikipedia
    VITAL_CATEGORIES = {
        'level1': 'Wikipedia:Vital_articles/Level/1',
        'level2': 'Wikipedia:Vital_articles/Level/2', 
        'level3': 'Wikipedia:Vital_articles/Level/3',
        'level4': 'Wikipedia:Vital_articles/Level/4',
        'level5': 'Wikipedia:Vital_articles/Level/5',
    }
    
    # Core topics that are always important
    CORE_TOPICS = [
        # Countries (all ~195)
        "List of sovereign states",
        # Sciences
        "Physics", "Chemistry", "Biology", "Mathematics", "Computer science",
        "Medicine", "Psychology", "Economics", "Philosophy", "History",
        # Technology
        "Artificial intelligence", "Machine learning", "Internet", "Computer",
        "Programming language", "Software", "Algorithm", "Database",
        # Major historical events
        "World War I", "World War II", "Cold War", "Industrial Revolution",
        "French Revolution", "American Revolution", "Renaissance",
        # Geography
        "Earth", "Continent", "Ocean", "Climate", "Mountain", "River",
        # People categories
        "List of Nobel laureates", "List of Presidents of the United States",
        # Arts & Culture
        "Literature", "Music", "Art", "Film", "Theatre", "Architecture",
        # Nature
        "Animal", "Plant", "Evolution", "Ecology", "Cell (biology)",
    ]
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.cache_file = self.data_dir / "vital_articles_cache.json"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GroundZero/2.0 (Learning AI; educational project)'
        })
        self._vital_articles: List[str] = []
        self._load_cache()
    
    def _load_cache(self):
        """Load cached vital articles list"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._vital_articles = data.get('articles', [])
                    cache_date = data.get('cached_at', '')
                    print(f"📚 Loaded {len(self._vital_articles)} vital articles from cache ({cache_date})")
            except Exception as e:
                print(f"Cache load error: {e}")
                self._vital_articles = []
    
    def _save_cache(self):
        """Save vital articles to cache"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'articles': self._vital_articles,
                    'cached_at': datetime.now().isoformat(),
                    'count': len(self._vital_articles)
                }, f)
            print(f"💾 Cached {len(self._vital_articles)} vital articles")
        except Exception as e:
            print(f"Cache save error: {e}")
    
    def fetch_vital_articles(self, max_level: int = 4) -> List[str]:
        """
        Fetch vital articles from Wikipedia.
        
        Args:
            max_level: Maximum level to fetch (1-5). Level 4 = ~10,000 articles.
        
        Returns:
            List of article titles
        """
        if self._vital_articles:
            return self._vital_articles
        
        print(f"🌐 Fetching vital articles (up to level {max_level})...")
        
        all_articles = set()
        
        # Add core topics first
        all_articles.update(self.CORE_TOPICS)
        
        # Fetch from vital article pages
        for level in range(1, max_level + 1):
            level_key = f'level{level}'
            if level_key in self.VITAL_CATEGORIES:
                articles = self._fetch_articles_from_page(self.VITAL_CATEGORIES[level_key])
                all_articles.update(articles)
                print(f"   Level {level}: {len(articles)} articles (total: {len(all_articles)})")
                time.sleep(0.5)  # Be nice to Wikipedia
        
        # Also fetch important lists
        important_lists = self._fetch_important_lists()
        all_articles.update(important_lists)
        
        self._vital_articles = list(all_articles)
        random.shuffle(self._vital_articles)  # Randomize order for variety
        
        self._save_cache()
        
        print(f"✅ Total vital articles: {len(self._vital_articles)}")
        return self._vital_articles
    
    def _fetch_articles_from_page(self, page_title: str) -> List[str]:
        """Fetch article links from a Wikipedia page"""
        try:
            params = {
                'action': 'query',
                'format': 'json',
                'titles': page_title,
                'prop': 'links',
                'pllimit': 'max',
                'plnamespace': 0  # Main namespace only
            }
            
            articles = []
            continue_token = None
            
            while True:
                if continue_token:
                    params['plcontinue'] = continue_token
                
                response = self.session.get(self.API_URL, params=params, timeout=30)
                data = response.json()
                
                pages = data.get('query', {}).get('pages', {})
                for page in pages.values():
                    for link in page.get('links', []):
                        title = link.get('title', '')
                        if title and not title.startswith(('Wikipedia:', 'Template:', 'Category:', 'File:')):
                            articles.append(title)
                
                # Check for continuation
                if 'continue' in data:
                    continue_token = data['continue'].get('plcontinue')
                else:
                    break
            
            return articles
            
        except Exception as e:
            print(f"Error fetching from {page_title}: {e}")
            return []
    
    def _fetch_important_lists(self) -> List[str]:
        """Fetch articles from important list pages"""
        important_lists = [
            "List of countries",
            "List of cities by population", 
            "List of programming languages",
            "List of elements",
            "List of Nobel laureates",
        ]
        
        articles = []
        for list_page in important_lists:
            try:
                list_articles = self._fetch_articles_from_page(list_page)
                articles.extend(list_articles[:100])  # Limit per list
            except:
                pass
        
        return articles
    
    def get_next_batch(self, batch_size: int = 10, exclude: Set[str] = None) -> List[str]:
        """Get next batch of vital articles to learn, excluding already learned"""
        if not self._vital_articles:
            self.fetch_vital_articles()
        
        exclude = exclude or set()
        available = [a for a in self._vital_articles if a not in exclude]
        
        if not available:
            return []
        
        return available[:batch_size]
    
    def get_progress(self, learned_titles: Set[str]) -> Dict[str, Any]:
        """Get progress on vital articles"""
        if not self._vital_articles:
            self.fetch_vital_articles()
        
        total = len(self._vital_articles)
        learned = len(learned_titles.intersection(set(self._vital_articles)))
        
        return {
            'total_vital': total,
            'learned_vital': learned,
            'remaining_vital': total - learned,
            'percentage': round((learned / total) * 100, 1) if total > 0 else 0
        }


class CategoryLearner:
    """
    Learn all articles from specific Wikipedia categories.
    
    Useful for focused learning on specific topics like:
    - All programming languages
    - All countries
    - All chemical elements
    - All animal species
    """
    
    API_URL = "https://en.wikipedia.org/w/api.php"
    
    # Pre-defined useful categories
    RECOMMENDED_CATEGORIES = {
        'countries': 'Category:Countries',
        'programming_languages': 'Category:Programming languages',
        'chemical_elements': 'Category:Chemical elements',
        'planets': 'Category:Planets of the Solar System',
        'capitals': 'Category:Capitals in Africa',  # Plus other continents
        'inventions': 'Category:Inventions',
        'diseases': 'Category:Diseases and disorders',
        'animals': 'Category:Animals',
        'plants': 'Category:Plants',
        'companies': 'Category:Companies',
        'universities': 'Category:Universities and colleges',
        'wars': 'Category:Wars',
        'religions': 'Category:Religions',
        'sports': 'Category:Sports',
        'foods': 'Category:Foods',
    }
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GroundZero/2.0 (Learning AI; educational project)'
        })
        self._category_cache: Dict[str, List[str]] = {}
    
    def get_category_articles(self, category: str, max_articles: int = 500) -> List[str]:
        """
        Get all articles from a Wikipedia category.
        
        Args:
            category: Category name (with or without 'Category:' prefix)
            max_articles: Maximum number of articles to fetch
        
        Returns:
            List of article titles
        """
        if not category.startswith('Category:'):
            category = f'Category:{category}'
        
        if category in self._category_cache:
            return self._category_cache[category]
        
        print(f"📂 Fetching articles from {category}...")
        
        try:
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'categorymembers',
                'cmtitle': category,
                'cmtype': 'page',
                'cmlimit': 'max'
            }
            
            articles = []
            continue_token = None
            
            while len(articles) < max_articles:
                if continue_token:
                    params['cmcontinue'] = continue_token
                
                response = self.session.get(self.API_URL, params=params, timeout=30)
                data = response.json()
                
                members = data.get('query', {}).get('categorymembers', [])
                for member in members:
                    title = member.get('title', '')
                    if title and not title.startswith(('Wikipedia:', 'Template:', 'Category:')):
                        articles.append(title)
                
                if 'continue' in data:
                    continue_token = data['continue'].get('cmcontinue')
                else:
                    break
                
                time.sleep(0.3)  # Rate limiting
            
            self._category_cache[category] = articles[:max_articles]
            print(f"   Found {len(articles)} articles in {category}")
            return articles[:max_articles]
            
        except Exception as e:
            print(f"Error fetching category {category}: {e}")
            return []
    
    def get_subcategories(self, category: str) -> List[str]:
        """Get subcategories of a category"""
        if not category.startswith('Category:'):
            category = f'Category:{category}'
        
        try:
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'categorymembers',
                'cmtitle': category,
                'cmtype': 'subcat',
                'cmlimit': 'max'
            }
            
            response = self.session.get(self.API_URL, params=params, timeout=30)
            data = response.json()
            
            return [m.get('title', '') for m in data.get('query', {}).get('categorymembers', [])]
            
        except Exception as e:
            print(f"Error fetching subcategories: {e}")
            return []


class StrategicLearner:
    """
    Orchestrates strategic learning with priorities:
    
    1. VITAL: Wikipedia's vital articles (most important content)
    2. CATEGORIES: User-specified category-based learning
    3. ON-DEMAND: Learn when asked about unknown topics
    4. RANDOM: Fallback to random articles after priorities done
    
    Tracks what's been learned to avoid duplicates.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.vital_provider = VitalArticlesProvider(data_dir)
        self.category_learner = CategoryLearner(data_dir)
        
        # Track learned articles
        self.learned_file = self.data_dir / "learned_articles.json"
        self.learned_titles: Set[str] = set()
        self._load_learned()
        
        # Learning queue and priorities
        self.priority_queue: List[Dict[str, Any]] = []
        self.category_queue: List[str] = []
        
        # Stats
        self.stats = {
            'vital_learned': 0,
            'category_learned': 0,
            'ondemand_learned': 0,
            'random_learned': 0,
            'total_learned': 0
        }
        
        # Session for Wikipedia API
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GroundZero/2.0 (Learning AI; educational project)'
        })
        
        print(f"📚 Strategic Learner initialized with {len(self.learned_titles)} previously learned articles")
    
    def _load_learned(self):
        """Load set of already learned article titles"""
        if self.learned_file.exists():
            try:
                with open(self.learned_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.learned_titles = set(data.get('titles', []))
                    self.stats = data.get('stats', self.stats)
            except:
                self.learned_titles = set()
    
    def _save_learned(self):
        """Save learned articles to disk"""
        try:
            with open(self.learned_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'titles': list(self.learned_titles),
                    'stats': self.stats,
                    'saved_at': datetime.now().isoformat()
                }, f)
        except Exception as e:
            print(f"Error saving learned articles: {e}")
    
    def mark_as_learned(self, title: str, source: str = 'unknown'):
        """Mark an article as learned"""
        if title not in self.learned_titles:
            self.learned_titles.add(title)
            self.stats['total_learned'] += 1
            
            if source == 'vital':
                self.stats['vital_learned'] += 1
            elif source == 'category':
                self.stats['category_learned'] += 1
            elif source == 'ondemand':
                self.stats['ondemand_learned'] += 1
            else:
                self.stats['random_learned'] += 1
            
            # Save periodically (every 10 articles)
            if self.stats['total_learned'] % 10 == 0:
                self._save_learned()
    
    def is_learned(self, title: str) -> bool:
        """Check if article was already learned"""
        return title in self.learned_titles
    
    def add_category_to_queue(self, category: str):
        """Add a category to the learning queue"""
        if category not in self.category_queue:
            self.category_queue.append(category)
            print(f"📂 Added category to queue: {category}")
    
    def add_priority_article(self, title: str, reason: str = "user request"):
        """Add a specific article to priority queue"""
        if not self.is_learned(title):
            self.priority_queue.append({
                'title': title,
                'reason': reason,
                'added_at': datetime.now().isoformat()
            })
    
    def get_next_articles(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get next articles to learn based on priority:
        1. Priority queue (user-requested)
        2. Vital articles
        3. Category articles
        4. Random articles (fallback)
        
        Returns list of {'title': str, 'source': str, 'url': str}
        """
        articles = []
        
        # 1. Priority queue first
        while self.priority_queue and len(articles) < count:
            item = self.priority_queue.pop(0)
            title = item['title']
            if not self.is_learned(title):
                articles.append({
                    'title': title,
                    'source': 'priority',
                    'url': f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
                })
        
        # 2. Vital articles
        if len(articles) < count:
            vital = self.vital_provider.get_next_batch(
                count - len(articles), 
                exclude=self.learned_titles
            )
            for title in vital:
                if not self.is_learned(title):
                    articles.append({
                        'title': title,
                        'source': 'vital',
                        'url': f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
                    })
        
        # 3. Category articles
        if len(articles) < count and self.category_queue:
            category = self.category_queue[0]
            cat_articles = self.category_learner.get_category_articles(category)
            
            for title in cat_articles:
                if len(articles) >= count:
                    break
                if not self.is_learned(title):
                    articles.append({
                        'title': title,
                        'source': 'category',
                        'url': f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
                    })
            
            # Check if category is exhausted
            remaining = [t for t in cat_articles if not self.is_learned(t)]
            if not remaining:
                self.category_queue.pop(0)
                print(f"✅ Completed category: {category}")
        
        # 4. Random articles as fallback
        if len(articles) < count:
            random_articles = self._get_random_unlearned(count - len(articles))
            articles.extend(random_articles)
        
        return articles
    
    def _get_random_unlearned(self, count: int) -> List[Dict[str, Any]]:
        """Get random Wikipedia articles that haven't been learned"""
        try:
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'random',
                'rnnamespace': 0,
                'rnlimit': count * 3  # Fetch extra to filter learned ones
            }
            
            response = self.session.get(
                "https://en.wikipedia.org/w/api.php", 
                params=params, 
                timeout=10
            )
            data = response.json()
            
            articles = []
            for item in data.get('query', {}).get('random', []):
                title = item.get('title', '')
                if title and not self.is_learned(title):
                    articles.append({
                        'title': title,
                        'source': 'random',
                        'url': f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
                    })
                    if len(articles) >= count:
                        break
            
            return articles
            
        except Exception as e:
            print(f"Error getting random articles: {e}")
            return []
    
    def get_article_content(self, title: str) -> Optional[Dict[str, Any]]:
        """Fetch full article content from Wikipedia"""
        try:
            params = {
                'action': 'query',
                'format': 'json',
                'titles': title,
                'prop': 'extracts|info',
                'explaintext': True,
                'exsectionformat': 'plain',
                'inprop': 'url'
            }
            
            response = self.session.get(
                "https://en.wikipedia.org/w/api.php",
                params=params,
                timeout=15
            )
            data = response.json()
            
            pages = data.get('query', {}).get('pages', {})
            
            for page_id, page in pages.items():
                if page_id == '-1':
                    continue
                
                content = page.get('extract', '')
                
                if content and len(content) > 100:
                    return {
                        'title': page.get('title', title),
                        'content': content,
                        'url': page.get('fullurl', f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"),
                        'word_count': len(content.split())
                    }
            
            return None
            
        except Exception as e:
            print(f"Error fetching '{title}': {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get strategic learning statistics"""
        vital_progress = self.vital_provider.get_progress(self.learned_titles)
        
        return {
            'learned_titles': len(self.learned_titles),
            'vital_progress': vital_progress,
            'categories_queued': len(self.category_queue),
            'priority_queued': len(self.priority_queue),
            'breakdown': self.stats,
            'next_source': self._get_next_source()
        }
    
    def _get_next_source(self) -> str:
        """Determine what the next learning source will be"""
        if self.priority_queue:
            return 'priority'
        
        vital_remaining = len([
            a for a in self.vital_provider._vital_articles 
            if a not in self.learned_titles
        ])
        if vital_remaining > 0:
            return f'vital ({vital_remaining} remaining)'
        
        if self.category_queue:
            return f'category: {self.category_queue[0]}'
        
        return 'random'
    
    def save(self):
        """Save all state to disk"""
        self._save_learned()
        print(f"💾 Saved {len(self.learned_titles)} learned articles")


# Convenience function to create strategic learner
def create_strategic_learner(data_dir: str = "data") -> StrategicLearner:
    """Create and initialize a strategic learner"""
    learner = StrategicLearner(data_dir)
    
    # Pre-fetch vital articles in background
    def prefetch():
        learner.vital_provider.fetch_vital_articles(max_level=4)
    
    thread = threading.Thread(target=prefetch, daemon=True)
    thread.start()
    
    return learner