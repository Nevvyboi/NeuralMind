"""
Reasoning Engine with Context Brain Integration
================================================
Intelligent question answering using:
1. Context Brain (smart query understanding + learning)
2. Semantic search (vector embeddings)
3. Symbolic reasoning (knowledge graph)
4. Neural generation (transformer)

The Context Brain provides:
- Fuzzy matching for typos ("Parid" → "Paris")
- Correction understanding ("I mean X")
- Learning from user feedback
- Entity disambiguation
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from storage import KnowledgeBase

# Import Context Brain
try:
    from .context_brain import ContextBrain, SmartSearcher, QueryIntent
    CONTEXT_BRAIN_AVAILABLE = True
except ImportError:
    try:
        from context_brain import ContextBrain, SmartSearcher, QueryIntent
        CONTEXT_BRAIN_AVAILABLE = True
    except ImportError:
        CONTEXT_BRAIN_AVAILABLE = False
        print("⚠️ Context Brain not available")

# Import the advanced reasoner
try:
    from .advanced_reasoner import AdvancedReasoner
    REASONER_AVAILABLE = True
except ImportError:
    try:
        from advanced_reasoner import AdvancedReasoner
        REASONER_AVAILABLE = True
    except ImportError:
        REASONER_AVAILABLE = False
        print("⚠️ AdvancedReasoner not available")


class QuestionType(Enum):
    """Types of questions"""
    DEFINITION = "definition"
    FACTUAL = "factual"
    CAUSAL = "causal"
    PROCEDURAL = "procedural"
    COMPARATIVE = "comparative"
    GREETING = "greeting"
    META = "meta"
    CREATIVE = "creative"
    CORRECTION = "correction"
    CLARIFICATION = "clarification"


@dataclass
class ReasoningResult:
    """Result of reasoning"""
    answer: Optional[str]
    confidence: float
    sources: List[Dict[str, str]]
    question_type: QuestionType
    thought_process: List[Dict[str, Any]]
    needs_search: bool = False
    reasoning_method: str = "unknown"
    understood_query: str = ""
    corrections_applied: List[Tuple[str, str]] = None
    suggestions: List[str] = None


class ReasoningEngine:
    """
    Analyzes questions and generates reasoned answers.
    Now with Context Brain for smart understanding!
    """
    
    PATTERNS = {
        QuestionType.DEFINITION: [
            r'^what is\b', r'^what are\b', r'^define\b', r'^explain what\b',
            r'^who is\b', r'^who are\b', r'^tell me about\b', r'^describe\b'
        ],
        QuestionType.FACTUAL: [
            r'^when\b', r'^where\b', r'^which\b', r'^how many\b', r'^how much\b'
        ],
        QuestionType.CAUSAL: [
            r'^why\b', r'^how come\b', r'^what caused\b', r'^reason for\b'
        ],
        QuestionType.PROCEDURAL: [
            r'^how to\b', r'^how do\b', r'^how can\b', r'^steps to\b'
        ],
        QuestionType.COMPARATIVE: [
            r'compare\b', r'difference\b', r'versus\b', r'\bvs\b', r'better\b'
        ],
        QuestionType.GREETING: [
            r'^hi\b', r'^hello\b', r'^hey\b', r'^good morning\b',
            r'^good afternoon\b', r'^good evening\b', r'^thanks\b'
        ],
        QuestionType.META: [
            r'\byou\b.*\bcan\b', r'\byour\b', r'neuralmind', r'what can you',
            r'who are you', r'what are you', r'about yourself'
        ],
        QuestionType.CREATIVE: [
            r'^write\b', r'^create\b', r'^generate\b', r'^make up\b',
            r'^imagine\b', r'^pretend\b', r'^tell me a story\b'
        ],
        QuestionType.CORRECTION: [
            r'^i mean\b', r'^no[,.]?\s*i mean\b', r'^not that\b', r'^actually\b'
        ],
        QuestionType.CLARIFICATION: [
            r'^the (first|second|third|1st|2nd|3rd|last) one',
            r'^option \d+', r'^number \d+'
        ]
    }
    
    GREETINGS = {
        'hi': "Hi there! How can I help you? Ask me anything!",
        'hello': "Hello! What would you like to know?",
        'hey': "Hey! What can I do for you?",
        'good morning': "Good morning! How can I assist you?",
        'good afternoon': "Good afternoon! What would you like to explore?",
        'good evening': "Good evening! How can I help?",
        'thanks': "You're welcome! Is there anything else you'd like to know?",
        'thank you': "You're welcome! Feel free to ask me anything.",
        'bye': "Goodbye! Come back anytime!",
        'goodbye': "Goodbye! It was nice chatting with you.",
        'how are you': "I'm doing great, always learning! What can I help you with?",
        "what's up": "Not much, just ready to help! What's on your mind?",
        'ok': "Great! Let me know if you have any questions.",
        'okay': "Alright! I'm here if you need anything.",
        'help': "I can help you with:\n\n• **Ask questions** - I'll search my knowledge\n• **Teach me** - Add knowledge directly\n• **Learn from URLs** - Paste any URL and I'll read it\n\nJust ask anything!"
    }
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
    
    def classify_question(self, query: str) -> QuestionType:
        """Classify the type of question"""
        query_lower = query.lower().strip()
        
        for q_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return q_type
        
        return QuestionType.DEFINITION
    
    def reason(self, query: str, understood_query: str = None) -> ReasoningResult:
        """Main reasoning method - now uses understood query from Context Brain"""
        query_clean = query.strip()
        query_lower = query_clean.lower()
        
        # Use understood query if provided (from Context Brain)
        search_query = understood_query if understood_query else query_clean
        
        # Check greetings first
        for greeting, response in self.GREETINGS.items():
            if query_lower == greeting or query_lower.startswith(greeting + ' '):
                return ReasoningResult(
                    answer=response,
                    confidence=1.0,
                    sources=[],
                    question_type=QuestionType.GREETING,
                    thought_process=[],
                    needs_search=False,
                    reasoning_method="greeting",
                    understood_query=query_clean
                )
        
        q_type = self.classify_question(query_lower)
        
        if q_type == QuestionType.META:
            return self._handle_meta_question()
        
        thoughts = []
        
        # Track if we're using a corrected query
        if understood_query and understood_query.lower() != query_lower:
            thoughts.append({
                'step': f"Understood as: '{understood_query}'",
                'type': 'context_brain',
                'confidence': 0.9
            })
        
        thoughts.append({
            'step': f"Searching for: {search_query[:50]}...",
            'type': 'analysis',
            'confidence': 0.9
        })
        
        # Semantic search
        results = self.kb.search(search_query, limit=10, min_score=0.1)
        
        thoughts.append({
            'step': f"Found {len(results)} relevant entries",
            'type': 'retrieval',
            'confidence': 0.8
        })
        
        if not results:
            # If understood query failed, try original
            if understood_query and understood_query != query_clean:
                results = self.kb.search(query_clean, limit=10, min_score=0.1)
                thoughts.append({
                    'step': f"Tried original query, found {len(results)} entries",
                    'type': 'fallback',
                    'confidence': 0.7
                })
        
        if not results:
            return ReasoningResult(
                answer=None,
                confidence=0.1,
                sources=[],
                question_type=q_type,
                thought_process=thoughts,
                needs_search=True,
                reasoning_method="no_results",
                understood_query=search_query
            )
        
        best = results[0]
        confidence = best.get('relevance', 0)
        
        thoughts.append({
            'step': f"Best match: {confidence:.0%} confidence",
            'type': 'evaluation',
            'confidence': confidence
        })
        
        if confidence < 0.25:
            return ReasoningResult(
                answer=None,
                confidence=confidence,
                sources=[],
                question_type=q_type,
                thought_process=thoughts,
                needs_search=True,
                reasoning_method="low_confidence",
                understood_query=search_query
            )
        
        answer = self._synthesize_answer(search_query, results, q_type)
        
        sources = [
            {'url': r.get('source_url', ''), 'title': r.get('source_title', '')}
            for r in results if r.get('source_url')
        ]
        
        return ReasoningResult(
            answer=answer,
            confidence=confidence,
            sources=sources[:5],
            question_type=q_type,
            thought_process=thoughts,
            needs_search=False,
            reasoning_method="vector_search",
            understood_query=search_query
        )
    
    def _handle_meta_question(self) -> ReasoningResult:
        """Handle questions about the AI itself"""
        response = """I'm GroundZero - an AI built completely from scratch!

**My Architecture:**
• 🧠 **Context Brain** - Smart query understanding that learns from you
• 🔍 **Vector Search** - Semantic similarity for finding relevant knowledge
• 🗺️ **Knowledge Graph** - Symbolic reasoning with facts and relationships
• ⚡ **Neural Network** - Transformer model for pattern learning

**What I Can Do:**
• Answer questions from my knowledge base
• Understand typos and corrections ("I mean...")
• Learn from Wikipedia and web pages
• Get smarter every time I learn something new!

I understand context and learn from our conversations!"""
        
        return ReasoningResult(
            answer=response,
            confidence=1.0,
            sources=[],
            question_type=QuestionType.META,
            thought_process=[],
            needs_search=False,
            reasoning_method="meta",
            understood_query=""
        )
    
    def _synthesize_answer(self, query: str, results: List[Dict], q_type: QuestionType) -> str:
        """Synthesize an answer from search results"""
        if not results:
            return None
        
        best = results[0]
        content = best.get('content', '')
        
        if not content:
            return None
        
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        query_words = set(query.lower().split())
        query_words -= {'what', 'is', 'the', 'a', 'an', 'who', 'where', 'when', 'why', 'how'}
        
        scored_sentences = []
        for sent in sentences:
            sent_lower = sent.lower()
            score = sum(1 for word in query_words if word in sent_lower)
            if score > 0:
                scored_sentences.append((sent, score))
        
        scored_sentences.sort(key=lambda x: -x[1])
        
        if scored_sentences:
            answer_sentences = [s[0] for s in scored_sentences[:3]]
            answer = ' '.join(answer_sentences)
        else:
            answer = ' '.join(sentences[:3])
        
        if len(answer) > 500:
            answer = answer[:500] + '...'
        
        return answer


class ResponseGenerator:
    """
    Generates responses using ALL reasoning methods:
    1. Context Brain (smart understanding + learning)
    2. Vector search (semantic similarity)
    3. Knowledge Graph reasoning (symbolic AI)
    4. Neural Network generation (transformer)
    """
    
    def __init__(self, knowledge_base: KnowledgeBase, data_dir=None):
        self.kb = knowledge_base
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        self.reasoning = ReasoningEngine(knowledge_base)
        
        # Context Brain - THE NEW SMART UNDERSTANDING SYSTEM
        self.context_brain = None
        self.smart_searcher = None
        if CONTEXT_BRAIN_AVAILABLE:
            try:
                self.context_brain = ContextBrain(self.data_dir)
                self.smart_searcher = SmartSearcher(knowledge_base, self.context_brain)
                print("✅ Context Brain connected to Response Generator")
            except Exception as e:
                print(f"⚠️ Could not initialize Context Brain: {e}")
        
        # Knowledge Graph (set by server)
        self.graph_reasoner = None
        
        # Neural Brain (set by server)
        self.neural_brain = None
        
        # Context Brain handles all conversation tracking now
        # (context.py is deprecated - context_brain.py replaces it)
        
        print("✅ Response Generator initialized" + 
              (" with Context Brain" if self.context_brain else ""))
    
    def learn_to_graph(self, content: str, source: str = "") -> Dict[str, Any]:
        """Feed learned content to the knowledge graph"""
        # Also feed to Context Brain for phonetic indexing
        if self.context_brain:
            self.context_brain.learn_from_content(content, source)
        
        if not self.graph_reasoner:
            return {'facts_added': 0}
        
        result = self.graph_reasoner.learn(content, source)
        return result
    
    def generate(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Generate a response using hybrid reasoning.
        
        Priority:
        1. Context Brain (understand query first!)
        2. Knowledge Graph (if confident answer)
        3. Vector Search (semantic similarity)
        4. Neural Network (if others fail)
        5. Web Search (as last resort)
        """
        # Context Brain handles all conversation tracking now
        
        # =====================================================
        # STEP 0: USE CONTEXT BRAIN TO UNDERSTAND QUERY
        # =====================================================
        understanding = None
        understood_query = query
        corrections_applied = []
        suggestions = []
        
        if self.context_brain:
            # Get known entities for better matching
            known_entities = self._get_known_entities()
            
            understanding = self.context_brain.understand_query(
                query, session_id, known_entities
            )
            
            understood_query = understanding['understood_query']
            corrections_applied = understanding.get('corrections_applied', [])
            suggestions = understanding.get('suggestions', [])
            
            # Handle special intents
            if understanding['intent'] == QueryIntent.GREETING:
                greeting_response = self._get_greeting_response(query)
                if greeting_response:
                    return {
                        'response': greeting_response,
                        'confidence': 1.0,
                        'sources': [],
                        'needs_search': False,
                        'reasoning_type': 'greeting',
                        'understood_query': query
                    }
            
            # Handle disambiguation needed
            if understanding.get('needs_disambiguation'):
                options = understanding.get('disambiguation_options', [])
                if options:
                    response = self._format_disambiguation_response(query, options)
                    return {
                        'response': response,
                        'confidence': 0.8,
                        'sources': [],
                        'needs_search': False,
                        'disambiguation': True,
                        'options': options,
                        'reasoning_type': 'disambiguation',
                        'understood_query': query
                    }
        
        # NOTE: Pronoun/reference resolution is now handled by Context Brain
        # in understand_query() above - it detects clarification intents like
        # "the first one", "option 2", etc. and resolves them automatically.
        
        # =====================================================
        # METHOD 1: Try Knowledge Graph FIRST
        # =====================================================
        graph_result = None
        if self.graph_reasoner:
            graph_result = self.graph_reasoner.reason(understood_query)
            
            if graph_result and graph_result['confidence'] >= 0.6 and graph_result.get('facts_used', 0) > 0:
                answer = graph_result['answer']
                
                vector_results = self.kb.search(understood_query, limit=3)
                sources = [{'url': r.get('source_url', ''), 'title': r.get('source_title', '')}
                          for r in vector_results if r.get('source_url')]
                
                # Record selection for learning (Context Brain tracks conversation internally)
                if self.context_brain and sources:
                    self.context_brain.record_selection(query, sources[0].get('title', understood_query))
                
                return {
                    'response': answer,
                    'confidence': graph_result['confidence'],
                    'sources': sources[:3],
                    'needs_search': False,
                    'reasoning_type': 'knowledge_graph',
                    'facts_used': graph_result.get('facts_used', 0),
                    'understood_query': understood_query,
                    'corrections_applied': corrections_applied
                }
        
        # =====================================================
        # METHOD 2: Vector Search (with understood query!)
        # =====================================================
        result = self.reasoning.reason(query, understood_query)
        
        # NOTE: Disambiguation is now handled by Context Brain in understand_query()
        # at the start of generate(). It detects ambiguous queries and returns
        # needs_disambiguation=True with options.
        
        # Blend graph result if available
        if graph_result and graph_result.get('confidence', 0) > 0.1 and result.confidence < 0.5:
            if graph_result.get('answer') and "don't have" not in graph_result.get('answer', ''):
                result.answer = f"{graph_result['answer']}\n\n{result.answer}" if result.answer else graph_result['answer']
                result.confidence = max(result.confidence, graph_result['confidence'])
        
        # =====================================================
        # METHOD 3: Neural Network (if others fail)
        # =====================================================
        if self.neural_brain and (not result.answer or result.needs_search):
            try:
                neural_result = self.neural_brain.answer(understood_query)
                
                if neural_result.get('answer') and neural_result.get('confidence', 0) > 0.3:
                    if not result.answer:
                        result.answer = neural_result['answer']
                        result.confidence = neural_result['confidence']
                        result.reasoning_method = 'neural_network'
                        result.needs_search = False
                    else:
                        result.answer = f"{result.answer}\n\n(Neural: {neural_result['answer']})"
                        result.confidence = max(result.confidence, neural_result['confidence'])
            except Exception as e:
                print(f"⚠️ Neural reasoning error: {e}")
        
        # Record selection for learning (Context Brain tracks conversation internally)
        if self.context_brain and result.answer and result.sources:
            self.context_brain.record_selection(query, result.sources[0].get('title', understood_query))
        
        # Build response with Context Brain info
        response = {
            'response': result.answer,
            'confidence': result.confidence,
            'sources': result.sources,
            'needs_search': result.needs_search,
            'reasoning_type': result.reasoning_method,
            'thought_process': result.thought_process,
            'question_type': result.question_type.value,
            'understood_query': understood_query
        }
        
        # Add Context Brain metadata
        if corrections_applied:
            response['corrections_applied'] = corrections_applied
        if suggestions:
            response['suggestions'] = suggestions
        
        return response
    
    def _get_known_entities(self, limit: int = 500) -> List[str]:
        """Get known entity titles from knowledge base"""
        try:
            recent = self.kb.vectors.get_all_knowledge(limit)
            return [r['title'] for r in recent if r.get('title')]
        except:
            return []
    
    def _get_greeting_response(self, query: str) -> Optional[str]:
        """Get greeting response"""
        query_lower = query.lower().strip()
        
        for greeting, response in ReasoningEngine.GREETINGS.items():
            if query_lower == greeting or query_lower.startswith(greeting + ' '):
                return response
        
        return None
    
    def _format_disambiguation_response(self, query: str, options: List[str]) -> str:
        """Format disambiguation response"""
        response = f"**{query}** could refer to:\n\n"
        
        for i, option in enumerate(options[:5], 1):
            response += f"{i}. {option}\n"
        
        response += "\nWhich one would you like to know about?"
        return response
    
    def _extract_focused_answer(self, entity_name: str, content: str) -> str:
        """Extract answer focused on a specific entity"""
        if not entity_name:
            if not content:
                return "I found some information but couldn't extract details."
            return content[:500] + ('...' if len(content) > 500 else '')
        
        if not content:
            return f"I found information about {entity_name} but couldn't extract details."
        
        sentences = re.split(r'(?<=[.!?])\s+', content)
        relevant = []
        
        entity_lower = entity_name.lower()
        entity_parts = set(entity_lower.split())
        
        for sent in sentences:
            sent_lower = sent.lower()
            if entity_lower in sent_lower or any(part in sent_lower for part in entity_parts if len(part) > 3):
                relevant.append(sent)
        
        if relevant:
            answer = ' '.join(relevant[:4])
            if len(answer) > 600:
                answer = answer[:600] + '...'
            return answer
        
        return content[:500] + ('...' if len(content) > 500 else '')
    
    def neural_generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Direct neural generation (for creative tasks)"""
        if not self.neural_brain:
            return "Neural network not available."
        
        try:
            return self.neural_brain.generate(prompt, max_tokens=max_tokens)
        except Exception as e:
            return f"Generation error: {e}"
    
    def generate_after_learning(self, query: str,
                                learned: List[Dict] = None,
                                session_id: str = "default") -> Dict[str, Any]:
        """Generate response after learning"""
        # Feed to context brain for indexing
        if self.context_brain and learned:
            for item in learned:
                if item.get('title'):
                    self.context_brain.learn_from_content("", item['title'])
        
        result = self.reasoning.reason(query)
        
        sources = list(result.sources)
        if learned:
            seen = {s.get('url') for s in sources}
            for lc in learned:
                url = lc.get('url', '')
                if url and url not in seen:
                    sources.append({
                        'url': url,
                        'title': lc.get('title', 'Learned')
                    })
                    seen.add(url)
        
        # Context Brain handles conversation tracking internally via understand_query()
        # No need for separate context tracking anymore
        
        return {
            'response': result.answer,
            'confidence': result.confidence,
            'sources': sources[:5],
            'needs_search': result.needs_search,
            'thought_process': result.thought_process,
            'question_type': result.question_type.value,
            'learned_from': learned
        }
    
    def clear_context(self, session_id: str = "default") -> None:
        """Clear conversation context"""
        # Context Brain handles all conversation tracking now
        if self.context_brain:
            self.context_brain.clear_conversation(session_id)
    
    def get_context_brain_stats(self) -> Dict[str, Any]:
        """Get Context Brain statistics"""
        if self.context_brain:
            return self.context_brain.get_stats()
        return {'available': False}