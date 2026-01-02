"""
Advanced Web Scraper & Search System
=====================================
Universal text extraction from any website using multiple methods:
1. Trafilatura (primary - high quality article extraction)
2. Readability (Mozilla's algorithm)
3. Selenium (for JS-rendered pages)
4. BeautifulSoup (fallback)

Extracts FULL TEXT, not just first paragraph.
"""

import re
import time
import json
import random
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse, urljoin, quote_plus
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Web scraping imports
try:
    import trafilatura
    from trafilatura.settings import use_config
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    from readability import Document
    READABILITY_AVAILABLE = True
except ImportError:
    READABILITY_AVAILABLE = False

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

import requests


@dataclass
class ScrapedContent:
    """Scraped content from a URL"""
    url: str
    title: str
    text: str
    summary: str
    word_count: int
    success: bool
    source_type: str  # wikipedia, news, blog, general, academic
    metadata: Dict[str, Any]
    extraction_method: str = "unknown"  # trafilatura, readability, selenium, beautifulsoup


@dataclass
class SearchResult:
    """Search result from any search engine"""
    title: str
    url: str
    snippet: str
    source: str  # duckduckgo, brave, wikipedia, etc.
    relevance: float


class SeleniumDriver:
    """Manages Selenium WebDriver for JS-rendered pages"""
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_driver(cls):
        """Get or create Selenium driver (singleton)"""
        if not SELENIUM_AVAILABLE:
            return None
        
        with cls._lock:
            if cls._instance is None:
                try:
                    options = Options()
                    options.add_argument('--headless=new')
                    options.add_argument('--no-sandbox')
                    options.add_argument('--disable-dev-shm-usage')
                    options.add_argument('--disable-gpu')
                    options.add_argument('--window-size=1920,1080')
                    options.add_argument('--disable-blink-features=AutomationControlled')
                    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
                    
                    # Disable images for faster loading
                    prefs = {'profile.managed_default_content_settings.images': 2}
                    options.add_experimental_option('prefs', prefs)
                    
                    cls._instance = webdriver.Chrome(options=options)
                    cls._instance.set_page_load_timeout(30)
                except Exception as e:
                    print(f"Could not initialize Selenium: {e}")
                    return None
            
            return cls._instance
    
    @classmethod
    def close(cls):
        """Close the driver"""
        with cls._lock:
            if cls._instance:
                try:
                    cls._instance.quit()
                except:
                    pass
                cls._instance = None


class UniversalScraper:
    """
    Universal web scraper that extracts FULL TEXT from any website.
    Uses multiple extraction methods in order of quality:
    1. Trafilatura (best for articles)
    2. Readability (Mozilla's algorithm)
    3. Selenium (for JS-rendered pages)
    4. BeautifulSoup (fallback)
    """
    
    # JS-heavy domains that need Selenium
    JS_DOMAINS = [
        'twitter.com', 'x.com', 'facebook.com', 'instagram.com',
        'linkedin.com', 'bloomberg.com', 'wsj.com', 'ft.com',
    ]
    
    # Skip domains (not scrapable)
    SKIP_DOMAINS = [
        'youtube.com', 'youtu.be', 'vimeo.com', 'spotify.com',
        'soundcloud.com', 'amazon.com', 'ebay.com', 'reddit.com'
    ]
    
    def __init__(self, use_selenium: bool = True):
        self.use_selenium = use_selenium and SELENIUM_AVAILABLE
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Configure trafilatura for MAXIMUM RECALL (get all content)
        if TRAFILATURA_AVAILABLE:
            self.traf_config = use_config()
            self.traf_config.set("DEFAULT", "EXTRACTION_TIMEOUT", "30")
            self.traf_config.set("DEFAULT", "MIN_OUTPUT_SIZE", "100")
            # Set for maximum content extraction
            self.traf_config.set("DEFAULT", "MIN_EXTRACTED_SIZE", "100")
    
    def scrape(self, url: str, timeout: int = 20) -> ScrapedContent:
        """
        Scrape FULL CONTENT from any URL using multiple extraction methods.
        Tries: Trafilatura → Readability → Selenium → BeautifulSoup
        """
        domain = urlparse(url).netloc.lower()
        
        # Skip problematic domains
        if any(skip in domain for skip in self.SKIP_DOMAINS):
            return ScrapedContent(
                url=url, title="", text="", summary="",
                word_count=0, success=False, source_type="skip",
                metadata={"error": f"Skipped domain: {domain}"},
                extraction_method="none"
            )
        
        source_type = self._detect_source_type(url)
        
        # Check if JS rendering needed
        needs_js = any(js in domain for js in self.JS_DOMAINS)
        
        # Build extraction pipeline
        methods = []
        
        if needs_js and self.use_selenium:
            methods.append(('selenium', self._extract_selenium))
        
        methods.extend([
            ('trafilatura', self._extract_trafilatura),
            ('readability', self._extract_readability),
            ('beautifulsoup', self._extract_beautifulsoup_full),
        ])
        
        # Add Selenium as final fallback if not already tried
        if not needs_js and self.use_selenium:
            methods.append(('selenium', self._extract_selenium))
        
        # Download HTML first (used by non-Selenium methods)
        html = self._download(url, timeout)
        
        # Try each method
        for method_name, method_func in methods:
            try:
                if method_name == 'selenium':
                    result = method_func(url)
                else:
                    if not html:
                        continue
                    result = method_func(html, url)
                
                if result and result[0] and len(result[0]) >= 200:
                    text, title, metadata = result
                    text = self._clean_text(text)
                    
                    # Only accept if we have substantial content
                    if len(text) >= 200:
                        summary = self._generate_summary(text)
                        word_count = len(text.split())
                        
                        return ScrapedContent(
                            url=url,
                            title=title or self._extract_title_from_url(url),
                            text=text,
                            summary=summary,
                            word_count=word_count,
                            success=True,
                            source_type=source_type,
                            metadata=metadata,
                            extraction_method=method_name
                        )
            except Exception as e:
                print(f"{method_name} failed for {url}: {e}")
                continue
        
        return ScrapedContent(
            url=url, title="", text="", summary="",
            word_count=0, success=False, source_type=source_type,
            metadata={"error": "All extraction methods failed"},
            extraction_method="none"
        )
    
    def _download(self, url: str, timeout: int) -> Optional[str]:
        """Download HTML from URL"""
        try:
            response = self.session.get(url, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Download error for {url}: {e}")
            return None
    
    def _extract_selenium(self, url: str) -> Optional[Tuple[str, str, dict]]:
        """Extract using Selenium for JS-rendered pages"""
        if not self.use_selenium:
            return None
        
        driver = SeleniumDriver.get_driver()
        if not driver:
            return None
        
        try:
            driver.get(url)
            
            # Wait for body to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Extra wait for JS rendering
            time.sleep(2)
            
            # Scroll to load lazy content
            total_height = driver.execute_script("return document.body.scrollHeight")
            for i in range(0, min(total_height, 5000), 500):
                driver.execute_script(f"window.scrollTo(0, {i});")
                time.sleep(0.1)
            driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(0.5)
            
            # Get rendered HTML and extract
            html = driver.page_source
            title = driver.title
            
            # Use trafilatura on rendered HTML if available
            if TRAFILATURA_AVAILABLE:
                text = trafilatura.extract(
                    html,
                    include_tables=True,
                    favor_recall=True,
                    config=self.traf_config
                )
                if text and len(text) >= 200:
                    return text, title, {"rendered": True, "method": "selenium+trafilatura"}
            
            # Fallback: extract from rendered soup
            return self._extract_beautifulsoup_full(html, url, title_override=title)
            
        except Exception as e:
            print(f"Selenium error for {url}: {e}")
            return None
    
    def _extract_trafilatura(self, html: str, url: str) -> Optional[Tuple[str, str, dict]]:
        """Extract using trafilatura - FULL TEXT extraction"""
        if not TRAFILATURA_AVAILABLE:
            return None
        
        try:
            # Extract with maximum recall (get ALL content)
            text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
                favor_precision=False,
                favor_recall=True,  # Get more content
                config=self.traf_config
            )
            
            if not text or len(text) < 100:
                return None
            
            # Extract metadata
            metadata_obj = trafilatura.extract_metadata(html)
            title = ""
            metadata = {"method": "trafilatura"}
            
            if metadata_obj:
                title = metadata_obj.title or ""
                metadata.update({
                    "author": metadata_obj.author,
                    "date": str(metadata_obj.date) if metadata_obj.date else None,
                    "sitename": metadata_obj.sitename,
                    "description": metadata_obj.description,
                })
            
            return text, title, metadata
            
        except Exception as e:
            print(f"Trafilatura error: {e}")
            return None
    
    def _extract_readability(self, html: str, url: str) -> Optional[Tuple[str, str, dict]]:
        """Extract using Readability (Mozilla's algorithm)"""
        if not READABILITY_AVAILABLE:
            return None
        
        try:
            doc = Document(html)
            title = doc.title()
            content_html = doc.summary()
            
            # Convert HTML to text
            soup = BeautifulSoup(content_html, 'html.parser')
            
            # Get all paragraphs and text elements
            text_parts = []
            for elem in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li', 'td', 'th', 'div', 'span']):
                text = elem.get_text(separator=' ', strip=True)
                if len(text) > 30:
                    text_parts.append(text)
            
            text = '\n\n'.join(text_parts)
            text = re.sub(r'\s+', ' ', text)
            
            if len(text) < 100:
                return None
            
            return text, title, {"method": "readability"}
            
        except Exception as e:
            print(f"Readability error: {e}")
            return None
    
    def _extract_beautifulsoup_full(self, html: str, url: str, title_override: str = None) -> Optional[Tuple[str, str, dict]]:
        """Full content extraction using BeautifulSoup - gets ALL paragraphs"""
        if not BS4_AVAILABLE:
            return None
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove unwanted elements
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 
                           'aside', 'form', 'noscript', 'iframe', 'svg',
                           'button', 'input', 'select', 'textarea']):
                tag.decompose()
            
            # Remove by class patterns
            remove_patterns = ['sidebar', 'nav', 'menu', 'footer', 'header', 'ad', 
                              'advertisement', 'social', 'share', 'comment', 'related',
                              'cookie', 'popup', 'modal', 'banner', 'promo']
            for pattern in remove_patterns:
                for elem in soup.find_all(class_=re.compile(pattern, re.I)):
                    elem.decompose()
                for elem in soup.find_all(id=re.compile(pattern, re.I)):
                    elem.decompose()
            
            # Get title
            title = title_override
            if not title:
                og_title = soup.find('meta', property='og:title')
                if og_title and og_title.get('content'):
                    title = og_title['content'].strip()
                elif soup.title and soup.title.string:
                    title = soup.title.string.strip()
                else:
                    h1 = soup.find('h1')
                    if h1:
                        title = h1.get_text(strip=True)
            
            # Find main content area
            main_content = self._find_main_content(soup, url)
            
            # Extract ALL text elements
            text_parts = []
            seen = set()  # Avoid duplicates
            
            for elem in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'td', 'th', 'blockquote', 'pre']):
                text = elem.get_text(separator=' ', strip=True)
                
                # Skip short or duplicate content
                if len(text) < 25:
                    continue
                
                text_hash = hash(text[:50])
                if text_hash in seen:
                    continue
                seen.add(text_hash)
                
                # Add headers with formatting
                if elem.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    text_parts.append(f"\n## {text}\n")
                else:
                    text_parts.append(text)
            
            # Join all content
            full_text = '\n'.join(text_parts)
            full_text = re.sub(r'\s+', ' ', full_text)
            full_text = re.sub(r'\n\s*\n', '\n\n', full_text)
            
            if len(full_text) < 100:
                return None
            
            return full_text.strip(), title or "", {"method": "beautifulsoup_full"}
            
        except Exception as e:
            print(f"BeautifulSoup error: {e}")
            return None
    
    def _find_main_content(self, soup: BeautifulSoup, url: str) -> BeautifulSoup:
        """Find main content area of the page"""
        domain = urlparse(url).netloc.lower()
        
        # Site-specific selectors
        site_selectors = {
            'wikipedia.org': [('div', {'id': 'mw-content-text'}), ('div', {'id': 'bodyContent'})],
            'medium.com': [('article', {}), ('div', {'class': 'postArticle-content'})],
            'github.com': [('article', {}), ('div', {'class': 'markdown-body'})],
            'bbc.com': [('article', {}), ('div', {'class': 'story-body'})],
            'nytimes.com': [('article', {}), ('section', {'name': 'articleBody'})],
        }
        
        for site, selectors in site_selectors.items():
            if site in domain:
                for tag, attrs in selectors:
                    elem = soup.find(tag, attrs)
                    if elem and len(elem.get_text(strip=True)) > 200:
                        return elem
        
        # Generic content selectors
        generic_selectors = [
            ('article', {}),
            ('main', {}),
            ('div', {'role': 'main'}),
            ('div', {'id': 'content'}),
            ('div', {'id': 'main-content'}),
            ('div', {'id': 'main'}),
            ('div', {'class': 'content'}),
            ('div', {'class': 'article'}),
            ('div', {'class': 'post-content'}),
            ('div', {'class': 'entry-content'}),
            ('div', {'class': 'article-content'}),
            ('div', {'class': 'post'}),
            ('div', {'class': 'story'}),
        ]
        
        for tag, attrs in generic_selectors:
            elem = soup.find(tag, attrs)
            if elem and len(elem.get_text(strip=True)) > 200:
                return elem
        
        # Fallback: find largest text block
        best = soup.body or soup
        best_length = 0
        
        for div in soup.find_all(['div', 'article', 'section']):
            text_length = len(div.get_text(strip=True))
            paragraphs = div.find_all('p')
            
            # Prefer elements with multiple paragraphs
            if text_length > best_length and len(paragraphs) >= 3:
                best = div
                best_length = text_length
        
        return best
    
    def _extract_beautifulsoup(self, html: str, url: str) -> Optional[Tuple[str, str, dict]]:
        """Fallback extraction using BeautifulSoup"""
        return self._extract_beautifulsoup_full(html, url)
    
    def _extract_basic(self, html: str, url: str) -> Optional[Tuple[str, str, dict]]:
        """Basic regex extraction as last resort"""
        try:
            # Remove scripts and styles
            html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.I)
            html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.I)
            
            # Get title
            title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.I)
            title = title_match.group(1).strip() if title_match else ""
            
            # Remove all tags
            text = re.sub(r'<[^>]+>', ' ', html)
            text = self._clean_text(text)
            
            if len(text) > 100:
                return text, title, {}
            
            return None
            
        except Exception as e:
            print(f"Basic extraction error: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common noise patterns
        noise_patterns = [
            r'Subscribe to our newsletter.*?(?=\.|$)',
            r'Share on (Facebook|Twitter|LinkedIn).*?(?=\.|$)',
            r'Follow us on.*?(?=\.|$)',
            r'Cookie (policy|notice|consent).*?(?=\.|$)',
            r'Privacy policy.*?(?=\.|$)',
            r'Terms (of|and) (service|use|conditions).*?(?=\.|$)',
            r'©\s*\d{4}.*?(?=\.|$)',
            r'All rights reserved.*?(?=\.|$)',
        ]
        
        for pattern in noise_patterns:
            text = re.sub(pattern, '', text, flags=re.I)
        
        # Normalize quotes and dashes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('–', '-').replace('—', '-')
        
        # Remove multiple spaces again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _generate_summary(self, text: str, max_words: int = 100) -> str:
        """Generate a summary from text"""
        sentences = re.split(r'[.!?]+', text)
        summary_sentences = []
        word_count = 0
        
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 30:  # Skip very short sentences
                words = sent.split()
                if word_count + len(words) <= max_words:
                    summary_sentences.append(sent)
                    word_count += len(words)
                else:
                    break
        
        return '. '.join(summary_sentences) + '.' if summary_sentences else text[:500]
    
    def _detect_source_type(self, url: str) -> str:
        """Detect the type of source from URL"""
        domain = urlparse(url).netloc.lower()
        
        if 'wikipedia' in domain:
            return 'wikipedia'
        elif any(x in domain for x in ['news', 'cnn', 'bbc', 'reuters', 'nytimes', 'guardian']):
            return 'news'
        elif any(x in domain for x in ['medium', 'blog', 'wordpress', 'substack']):
            return 'blog'
        elif any(x in domain for x in ['edu', 'gov', 'org']):
            return 'academic'
        else:
            return 'general'
    
    def _extract_title_from_url(self, url: str) -> str:
        """Extract a title from the URL path"""
        path = urlparse(url).path
        # Get last segment
        segments = [s for s in path.split('/') if s]
        if segments:
            title = segments[-1].replace('-', ' ').replace('_', ' ')
            title = re.sub(r'\.\w+$', '', title)  # Remove extension
            return title.title()
        return urlparse(url).netloc


class MultiSourceSearch:
    """
    Multi-source search engine combining DuckDuckGo, Wikipedia, and other sources.
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.scraper = UniversalScraper()
        self._cache = {}
        self._cache_lock = threading.Lock()
    
    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        Search across multiple sources and return combined results.
        """
        # Check cache
        cache_key = hashlib.md5(query.lower().encode()).hexdigest()
        with self._cache_lock:
            if cache_key in self._cache:
                cached_time, cached_results = self._cache[cache_key]
                if time.time() - cached_time < 3600:  # 1 hour cache
                    return cached_results
        
        all_results = []
        
        # Search DuckDuckGo
        ddg_results = self._search_duckduckgo(query, max_results)
        all_results.extend(ddg_results)
        
        # Search Wikipedia directly for educational queries
        if self._is_educational_query(query):
            wiki_results = self._search_wikipedia(query, max_results=2)
            all_results.extend(wiki_results)
        
        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for r in all_results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                unique_results.append(r)
        
        # Sort by relevance
        unique_results.sort(key=lambda x: x.relevance, reverse=True)
        
        # Cache results
        with self._cache_lock:
            self._cache[cache_key] = (time.time(), unique_results[:max_results])
        
        return unique_results[:max_results]
    
    def _search_duckduckgo(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Search using DuckDuckGo HTML"""
        results = []
        
        try:
            # Use DuckDuckGo HTML search
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            if BS4_AVAILABLE:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find results
                for result in soup.select('.result')[:max_results]:
                    title_elem = result.select_one('.result__title')
                    snippet_elem = result.select_one('.result__snippet')
                    link_elem = result.select_one('.result__url')
                    
                    if title_elem and link_elem:
                        # Get actual URL
                        href = title_elem.find('a')
                        if href and href.get('href'):
                            url = href.get('href')
                            # Clean DuckDuckGo redirect URL
                            if 'uddg=' in url:
                                url_match = re.search(r'uddg=([^&]+)', url)
                                if url_match:
                                    from urllib.parse import unquote
                                    url = unquote(url_match.group(1))
                            
                            title = title_elem.get_text(strip=True)
                            snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                            
                            results.append(SearchResult(
                                title=title,
                                url=url,
                                snippet=snippet,
                                source='duckduckgo',
                                relevance=0.8
                            ))
            
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
        
        return results
    
    def _search_wikipedia(self, query: str, max_results: int = 2) -> List[SearchResult]:
        """Search Wikipedia API directly"""
        results = []
        
        try:
            # Use Wikipedia API
            api_url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'list': 'search',
                'srsearch': query,
                'srlimit': max_results,
                'format': 'json',
                'srprop': 'snippet|titlesnippet'
            }
            
            response = self.session.get(api_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            for item in data.get('query', {}).get('search', []):
                title = item.get('title', '')
                snippet = re.sub(r'<[^>]+>', '', item.get('snippet', ''))
                url = f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}"
                
                results.append(SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    source='wikipedia',
                    relevance=0.9  # Wikipedia is highly relevant for educational content
                ))
        
        except Exception as e:
            print(f"Wikipedia search error: {e}")
        
        return results
    
    def _is_educational_query(self, query: str) -> bool:
        """Check if query is likely educational/factual"""
        educational_patterns = [
            r'\bwhat is\b', r'\bwhat are\b', r'\bwho is\b', r'\bwho was\b',
            r'\bdefine\b', r'\bexplain\b', r'\bhow does\b', r'\bhow do\b',
            r'\bhistory of\b', r'\bmeaning of\b', r'\bwhy is\b', r'\bwhy do\b'
        ]
        
        query_lower = query.lower()
        return any(re.search(p, query_lower) for p in educational_patterns)
    
    def search_and_scrape(self, query: str, max_sources: int = 3) -> List[ScrapedContent]:
        """
        Search for query and scrape top results.
        Returns list of scraped content ready for learning.
        """
        results = self.search(query, max_results=max_sources + 2)  # Get extra in case some fail
        
        scraped = []
        for result in results:
            if len(scraped) >= max_sources:
                break
            
            content = self.scraper.scrape(result.url)
            if content.success and content.word_count >= 50:
                # Add search context to metadata
                content.metadata['search_query'] = query
                content.metadata['search_snippet'] = result.snippet
                scraped.append(content)
        
        return scraped


class IntelligentLearner:
    """
    Intelligent learning system that knows what it knows and what to learn.
    """
    
    def __init__(self, memory_store):
        self.memory = memory_store
        self.search = MultiSourceSearch()
        self.scraper = UniversalScraper()
        
        # Track knowledge gaps
        self.knowledge_gaps = []
        self.learning_queue = []
        
        # Topics to explore
        self.seed_topics = [
            "Science", "Technology", "History", "Mathematics", "Physics",
            "Biology", "Chemistry", "Geography", "Literature", "Philosophy",
            "Psychology", "Economics", "Computer Science", "Artificial Intelligence",
            "Medicine", "Art", "Music", "Politics", "Environment", "Space"
        ]
    
    def assess_knowledge(self, topic: str) -> Dict[str, Any]:
        """
        Assess how much we know about a topic.
        Returns knowledge coverage and confidence.
        """
        # Search internal knowledge
        results = self.memory.search_knowledge(topic, limit=10)
        
        if not results:
            return {
                'topic': topic,
                'coverage': 0.0,
                'confidence': 0.0,
                'known_facts': 0,
                'recommendation': 'learn'
            }
        
        # Count relevant facts
        known_facts = len(results)
        
        # Calculate confidence based on source diversity and recency
        sources = set()
        total_confidence = 0
        
        for r in results:
            sources.add(r.get('source_url', ''))
            total_confidence += r.get('confidence', 0.5)
        
        avg_confidence = total_confidence / known_facts if known_facts else 0
        source_diversity = min(1.0, len(sources) / 3)  # Up to 3 sources = full diversity
        
        coverage = min(1.0, known_facts / 10)  # 10 facts = full coverage
        overall_confidence = (avg_confidence + source_diversity + coverage) / 3
        
        return {
            'topic': topic,
            'coverage': coverage,
            'confidence': overall_confidence,
            'known_facts': known_facts,
            'source_count': len(sources),
            'recommendation': 'sufficient' if overall_confidence > 0.5 else 'learn'
        }
    
    def identify_knowledge_gaps(self, query: str) -> List[str]:
        """
        Identify what topics we need to learn to answer a query.
        """
        # Extract concepts from query
        concepts = self._extract_concepts(query)
        
        gaps = []
        for concept in concepts:
            assessment = self.assess_knowledge(concept)
            if assessment['recommendation'] == 'learn':
                gaps.append(concept)
        
        return gaps
    
    def learn_about(self, topic: str, callback=None) -> Dict[str, Any]:
        """
        Learn about a specific topic by searching and scraping.
        callback(status, progress, current_url) - for progress updates
        """
        if callback:
            callback('searching', 0, f"Searching for: {topic}")
        
        # Search and scrape
        contents = self.search.search_and_scrape(topic, max_sources=3)
        
        if not contents:
            return {
                'success': False,
                'topic': topic,
                'sources_learned': 0,
                'error': 'No content found'
            }
        
        learned_sources = []
        total_tokens = 0
        
        for i, content in enumerate(contents):
            progress = (i + 1) / len(contents)
            if callback:
                callback('learning', progress, content.url)
            
            # Process and store the content
            result = self._process_and_store(content, topic)
            if result['success']:
                learned_sources.append({
                    'url': content.url,
                    'title': content.title,
                    'words': content.word_count
                })
                total_tokens += content.word_count
        
        if callback:
            callback('complete', 1.0, f"Learned from {len(learned_sources)} sources")
        
        return {
            'success': True,
            'topic': topic,
            'sources_learned': len(learned_sources),
            'sources': learned_sources,
            'total_tokens': total_tokens
        }
    
    def _process_and_store(self, content: ScrapedContent, topic: str) -> Dict[str, Any]:
        """Process scraped content and store in memory"""
        try:
            # Chunk the text for better retrieval
            chunks = self._chunk_text(content.text, chunk_size=500, overlap=50)
            
            stored_count = 0
            for chunk in chunks:
                if len(chunk.strip()) > 50:
                    self.memory.store_knowledge(
                        content=chunk,
                        source_url=content.url,
                        source_title=content.title,
                        topic=topic,
                        confidence=0.8,
                        metadata={
                            'source_type': content.source_type,
                            'learned_at': time.time()
                        }
                    )
                    stored_count += 1
            
            return {'success': True, 'chunks_stored': stored_count}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks for better retrieval"""
        words = text.split()
        chunks = []
        
        i = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunks.append(' '.join(chunk_words))
            i += chunk_size - overlap
        
        return chunks
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Remove stopwords and get meaningful terms
        stopwords = {
            'what', 'is', 'are', 'the', 'a', 'an', 'how', 'why', 'when', 'where',
            'who', 'which', 'does', 'do', 'can', 'could', 'would', 'should',
            'will', 'about', 'tell', 'me', 'please', 'explain', 'describe',
            'this', 'that', 'these', 'those', 'it', 'they', 'them', 'i', 'you'
        }
        
        words = re.findall(r'\b\w+\b', text.lower())
        concepts = [w for w in words if w not in stopwords and len(w) > 2]
        
        # Return unique concepts, preserving order
        seen = set()
        unique = []
        for c in concepts:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        
        return unique[:5]  # Top 5 concepts
    
    def suggest_topics_to_learn(self, n: int = 5) -> List[str]:
        """Suggest topics to expand knowledge"""
        suggestions = []
        
        for topic in self.seed_topics:
            assessment = self.assess_knowledge(topic)
            if assessment['recommendation'] == 'learn':
                suggestions.append((topic, assessment['confidence']))
        
        # Sort by lowest confidence first (biggest gaps)
        suggestions.sort(key=lambda x: x[1])
        
        return [s[0] for s in suggestions[:n]]