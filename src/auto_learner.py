"""
Auto Learner Module
===================

Automatically fetches and learns from free knowledge sources:
- Wikipedia (articles)
- Simple Wikipedia (cleaner text)
- Project Gutenberg (books)
- Wiktionary (definitions)

No API keys required - all free and open!
"""

import re
import json
import time
import random
import urllib.request
import urllib.parse
import urllib.error
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LearnedContent:
    """Content fetched from a source"""
    Source: str
    Title: str
    Text: str
    Facts: int = 0
    Causal: int = 0


class AutoLearner:
    """
    Automatically learn from free online sources.
    
    Sources:
    - Wikipedia: General knowledge articles
    - Simple Wikipedia: Easier to parse
    - Project Gutenberg: Classic books
    - Wiktionary: Word definitions
    """
    
    # Wikipedia API endpoints
    WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
    SIMPLE_WIKI_API = "https://simple.wikipedia.org/w/api.php"
    
    # Topics to learn about (diverse knowledge)
    LEARNING_TOPICS = [
        # Science
        "Physics", "Chemistry", "Biology", "Astronomy", "Geology",
        "Mathematics", "Computer science", "Medicine", "Psychology",
        # Nature
        "Animal", "Plant", "Ocean", "Mountain", "River", "Forest",
        "Weather", "Climate", "Ecosystem", "Evolution",
        # Technology
        "Computer", "Internet", "Artificial intelligence", "Robot",
        "Electricity", "Engine", "Airplane", "Telephone",
        # History
        "Ancient Egypt", "Roman Empire", "World War", "Renaissance",
        "Industrial Revolution", "Space exploration",
        # Geography
        "Africa", "Europe", "Asia", "Americas", "Australia",
        "Country", "City", "Island", "Desert",
        # Culture
        "Music", "Art", "Literature", "Philosophy", "Religion",
        "Language", "Food", "Sport", "Film",
        # Daily life
        "Family", "School", "Work", "Health", "Money",
        "Transportation", "Communication", "Energy",
        # Concepts
        "Time", "Space", "Cause and effect", "Logic", "Knowledge",
        "Learning", "Memory", "Emotion", "Consciousness",
    ]
    
    def __init__(self, Engine=None):
        """
        Initialize auto-learner.
        
        Args:
            Engine: SmartChatEngine to learn into (optional)
        """
        self.Engine = Engine
        self.LearnedTopics: List[str] = []
        self.Stats = {
            "ArticlesFetched": 0,
            "TotalFacts": 0,
            "TotalCausal": 0,
            "Errors": 0,
        }
    
    def FetchWikipedia(self, Topic: str, Simple: bool = False) -> Optional[str]:
        """
        Fetch article text from Wikipedia.
        
        Args:
            Topic: Article title to fetch
            Simple: Use Simple Wikipedia (easier text)
        
        Returns:
            Article text or None if failed
        """
        Api = self.SIMPLE_WIKI_API if Simple else self.WIKIPEDIA_API
        
        Params = {
            "action": "query",
            "format": "json",
            "titles": Topic,
            "prop": "extracts",
            "explaintext": "true",
            "exsectionformat": "plain",
        }
        
        Url = f"{Api}?{urllib.parse.urlencode(Params)}"
        
        try:
            Req = urllib.request.Request(
                Url,
                headers={"User-Agent": "GroundZeroAI/2.0 (Educational Project)"}
            )
            with urllib.request.urlopen(Req, timeout=15) as Response:
                Data = json.loads(Response.read().decode())
                
            Pages = Data.get("query", {}).get("pages", {})
            for PageId, Page in Pages.items():
                if PageId != "-1":
                    return Page.get("extract", "")
            
            return None
            
        except urllib.error.URLError as E:
            # Network blocked or unavailable
            if "403" in str(E) or "Forbidden" in str(E):
                print("\n⚠️  Network access blocked. Using offline mode.")
                print("   Run on your local machine for Wikipedia access.\n")
            self.Stats["Errors"] += 1
            return None
        except Exception as E:
            self.Stats["Errors"] += 1
            return None
    
    def FetchRandomWikipedia(self, Simple: bool = False) -> Tuple[Optional[str], Optional[str]]:
        """
        Fetch a random Wikipedia article.
        
        Returns:
            Tuple of (title, text) or (None, None)
        """
        Api = self.SIMPLE_WIKI_API if Simple else self.WIKIPEDIA_API
        
        Params = {
            "action": "query",
            "format": "json",
            "list": "random",
            "rnnamespace": "0",
            "rnlimit": "1",
        }
        
        Url = f"{Api}?{urllib.parse.urlencode(Params)}"
        
        try:
            Req = urllib.request.Request(
                Url,
                headers={"User-Agent": "GroundZeroAI/2.0 (Educational Project)"}
            )
            with urllib.request.urlopen(Req, timeout=10) as Response:
                Data = json.loads(Response.read().decode())
            
            RandomPages = Data.get("query", {}).get("random", [])
            if RandomPages:
                Title = RandomPages[0].get("title")
                Text = self.FetchWikipedia(Title, Simple)
                return Title, Text
            
            return None, None
            
        except Exception as E:
            self.Stats["Errors"] += 1
            return None, None
    
    def ExtractSentences(self, Text: str, MaxSentences: int = 50) -> List[str]:
        """
        Extract clean sentences from text.
        
        Args:
            Text: Raw text
            MaxSentences: Maximum sentences to return
        
        Returns:
            List of clean sentences
        """
        if not Text:
            return []
        
        # Clean the text
        Text = re.sub(r'\s+', ' ', Text)
        Text = re.sub(r'\([^)]*\)', '', Text)  # Remove parenthetical
        Text = re.sub(r'\[[^\]]*\]', '', Text)  # Remove brackets
        
        # Split into sentences
        Sentences = re.split(r'(?<=[.!?])\s+', Text)
        
        # Filter and clean
        Clean = []
        for S in Sentences:
            S = S.strip()
            # Keep sentences that are informative
            if (len(S) > 20 and len(S) < 300 and
                not S.startswith("==") and
                not S.startswith("*") and
                S[0].isupper()):
                Clean.append(S)
                if len(Clean) >= MaxSentences:
                    break
        
        return Clean
    
    def LearnFromText(self, Text: str, Source: str = "wikipedia") -> Dict[str, int]:
        """
        Learn facts and causal relations from text.
        
        Args:
            Text: Text to learn from
            Source: Source name for attribution
        
        Returns:
            Dict with counts of facts and causal relations learned
        """
        if not self.Engine:
            return {"Facts": 0, "Causal": 0}
        
        Sentences = self.ExtractSentences(Text)
        Results = {"Facts": 0, "Causal": 0}
        
        for Sentence in Sentences:
            Learned = self.Engine.Learn(Sentence)
            Results["Facts"] += Learned.get("Facts", 0)
            Results["Causal"] += Learned.get("CausalRelations", 0)
        
        self.Stats["TotalFacts"] += Results["Facts"]
        self.Stats["TotalCausal"] += Results["Causal"]
        
        return Results
    
    def LearnTopic(self, Topic: str, Simple: bool = True) -> LearnedContent:
        """
        Learn about a specific topic from Wikipedia.
        
        Args:
            Topic: Topic to learn about
            Simple: Use Simple Wikipedia
        
        Returns:
            LearnedContent with results
        """
        print(f"  📚 Fetching: {Topic}...", end=" ", flush=True)
        
        Text = self.FetchWikipedia(Topic, Simple)
        
        if not Text:
            print("❌ Not found")
            return LearnedContent(
                Source="wikipedia",
                Title=Topic,
                Text="",
                Facts=0,
                Causal=0
            )
        
        Results = self.LearnFromText(Text, "wikipedia")
        self.Stats["ArticlesFetched"] += 1
        self.LearnedTopics.append(Topic)
        
        print(f"✓ {Results['Facts']} facts, {Results['Causal']} causal")
        
        return LearnedContent(
            Source="simple.wikipedia" if Simple else "wikipedia",
            Title=Topic,
            Text=Text[:500],
            Facts=Results["Facts"],
            Causal=Results["Causal"]
        )
    
    def LearnRandom(self, Count: int = 5, Simple: bool = True) -> List[LearnedContent]:
        """
        Learn from random Wikipedia articles.
        
        Args:
            Count: Number of articles to fetch
            Simple: Use Simple Wikipedia
        
        Returns:
            List of LearnedContent
        """
        Results = []
        
        for i in range(Count):
            print(f"  🎲 Random article {i+1}/{Count}...", end=" ", flush=True)
            
            Title, Text = self.FetchRandomWikipedia(Simple)
            
            if Title and Text:
                Learned = self.LearnFromText(Text)
                self.Stats["ArticlesFetched"] += 1
                
                print(f"'{Title[:30]}' → {Learned['Facts']} facts")
                
                Results.append(LearnedContent(
                    Source="wikipedia",
                    Title=Title,
                    Text=Text[:500],
                    Facts=Learned["Facts"],
                    Causal=Learned["Causal"]
                ))
            else:
                print("❌ Failed")
            
            time.sleep(0.5)  # Be nice to Wikipedia
        
        return Results
    
    def LearnContinuously(self, Callback=None):
        """
        Continuously learn from various sources.
        
        Args:
            Callback: Optional function called after each article
                      Callback(topic, facts, causal) - return False to stop
        """
        print("\n🌐 Starting continuous learning from free sources...")
        print("   Press Ctrl+C to stop\n")
        
        TopicIndex = 0
        RandomCount = 0
        
        try:
            while True:
                # Alternate between curated topics and random
                if TopicIndex < len(self.LEARNING_TOPICS) and RandomCount < 2:
                    # Learn from curated topic
                    Topic = self.LEARNING_TOPICS[TopicIndex]
                    Content = self.LearnTopic(Topic, Simple=True)
                    TopicIndex += 1
                else:
                    # Learn from random article
                    Results = self.LearnRandom(Count=1, Simple=True)
                    Content = Results[0] if Results else None
                    RandomCount += 1
                    if RandomCount >= 3:
                        RandomCount = 0
                
                # Callback
                if Callback and Content:
                    if not Callback(Content.Title, Content.Facts, Content.Causal):
                        break
                
                # Progress update every 10 articles
                if self.Stats["ArticlesFetched"] % 10 == 0:
                    self.PrintStats()
                
                # Be nice to Wikipedia - 1 second delay
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Learning stopped by user")
        
        self.PrintStats()
    
    def PrintStats(self):
        """Print current learning statistics"""
        print(f"\n{'='*50}")
        print(f"📊 Learning Statistics")
        print(f"{'='*50}")
        print(f"   Articles fetched: {self.Stats['ArticlesFetched']}")
        print(f"   Facts learned:    {self.Stats['TotalFacts']}")
        print(f"   Causal relations: {self.Stats['TotalCausal']}")
        print(f"   Errors:           {self.Stats['Errors']}")
        print(f"{'='*50}\n")
    
    def GetStats(self) -> Dict:
        """Get current statistics"""
        return self.Stats.copy()


def QuickLearn(Engine, Topics: List[str] = None, Count: int = 10):
    """
    Quick helper to learn from Wikipedia.
    
    Args:
        Engine: SmartChatEngine instance
        Topics: Specific topics to learn (optional)
        Count: Number of random articles if no topics
    """
    Learner = AutoLearner(Engine)
    
    if Topics:
        for Topic in Topics:
            Learner.LearnTopic(Topic)
            time.sleep(0.5)
    else:
        Learner.LearnRandom(Count)
    
    return Learner.GetStats()