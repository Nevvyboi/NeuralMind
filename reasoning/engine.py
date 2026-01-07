"""
Reasoning Engine with Neural Network
=====================================
Intelligent question answering using:
1. Semantic search (vector embeddings)
2. Symbolic reasoning (knowledge graph)
3. Neural generation (transformer)

The neural network provides a fallback when other methods fail,
and can generate novel responses based on learned patterns.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from storage import KnowledgeBase

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
    CREATIVE = "creative"  # NEW: For creative/generative questions


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


class ReasoningEngine:
    """
    Analyzes questions and generates reasoned answers.
    Uses semantic search, knowledge graph, AND neural generation.
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
    
    def reason(self, query: str) -> ReasoningResult:
        """Main reasoning method"""
        query_clean = query.strip()
        query_lower = query_clean.lower()
        
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
                    reasoning_method="greeting"
                )
        
        q_type = self.classify_question(query_lower)
        
        if q_type == QuestionType.META:
            return self._handle_meta_question()
        
        thoughts = []
        thoughts.append({
            'step': f"Analyzing: {query_clean[:50]}...",
            'type': 'analysis',
            'confidence': 0.9
        })
        
        # Semantic search
        results = self.kb.search(query_clean, limit=10, min_score=0.1)
        
        thoughts.append({
            'step': f"Found {len(results)} relevant entries",
            'type': 'retrieval',
            'confidence': 0.8
        })
        
        if not results:
            return ReasoningResult(
                answer=None,
                confidence=0.1,
                sources=[],
                question_type=q_type,
                thought_process=thoughts,
                needs_search=True,
                reasoning_method="no_results"
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
                reasoning_method="low_confidence"
            )
        
        answer = self._synthesize_answer(query_clean, results, q_type)
        
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
            reasoning_method="vector_search"
        )
    
    def _handle_meta_question(self) -> ReasoningResult:
        """Handle questions about the AI itself"""
        response = """I'm GroundZero - an AI built completely from scratch!

**My Architecture:**
• 🔍 **Vector Search** - Semantic similarity for finding relevant knowledge
• 🗺️ **Knowledge Graph** - Symbolic reasoning with facts and relationships
• 🧠 **Neural Network** - Transformer model for pattern learning and generation

**What I Can Do:**
• Answer questions from my knowledge base
• Learn from Wikipedia and web pages
• Reason about relationships between facts
• Generate responses using neural patterns

I get smarter every time I learn something new!"""
        
        return ReasoningResult(
            answer=response,
            confidence=1.0,
            sources=[],
            question_type=QuestionType.META,
            thought_process=[],
            needs_search=False,
            reasoning_method="meta"
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
    1. Vector search (semantic similarity)
    2. Knowledge Graph reasoning (symbolic AI)
    3. Neural Network generation (transformer)
    """
    
    def __init__(self, knowledge_base: KnowledgeBase, data_dir=None):
        self.kb = knowledge_base
        self.reasoning = ReasoningEngine(knowledge_base)
        
        # Knowledge Graph (set by server)
        self.graph_reasoner = None
        
        # Neural Brain (set by server)
        self.neural_brain = None
        
        # Context manager
        from .context import get_context, ConversationContext
        self.get_context = get_context
        
        print("✅ Response Generator initialized")
    
    def learn_to_graph(self, content: str, source: str = "") -> Dict[str, Any]:
        """Feed learned content to the knowledge graph"""
        if not self.graph_reasoner:
            return {'facts_added': 0}
        
        result = self.graph_reasoner.learn(content, source)
        return result
    
    def generate(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Generate a response using hybrid reasoning.
        
        Priority:
        1. Knowledge Graph (if confident answer)
        2. Vector Search (semantic similarity)
        3. Neural Network (if available and others fail)
        4. Web Search (as last resort)
        """
        context = self.get_context(session_id)
        
        # Check for pronoun resolution
        resolved_entity = context.resolve_reference(query)
        
        if resolved_entity:
            # Handle both tuple and object formats
            if isinstance(resolved_entity, tuple):
                entity_name = resolved_entity[0] if resolved_entity and resolved_entity[0] else ""
            elif hasattr(resolved_entity, 'name'):
                entity_name = resolved_entity.name if resolved_entity.name else ""
            else:
                entity_name = str(resolved_entity) if resolved_entity else ""
            
            # Skip if we couldn't extract a valid entity name
            if not entity_name:
                pass  # Fall through to normal processing below
            else:
                search_query = f"{entity_name} {query}"
                results = self.kb.search(search_query, limit=5)
                
                if results and results[0].get('relevance', 0) > 0.3:
                    best_match = results[0]
                    answer = self._extract_focused_answer(entity_name, best_match.get('content', ''))
                    
                    context.add_user_message(query)
                    context.add_assistant_response(answer, [{
                        'name': entity_name,
                        'content': answer[:200],
                        'source_url': best_match.get('source_url', ''),
                        'source_title': best_match.get('source_title', ''),
                        'confidence': best_match.get('relevance', 0.7)
                    }])
                    
                    return {
                        'response': answer,
                        'confidence': best_match.get('relevance', 0.7),
                        'sources': [{'url': best_match.get('source_url', ''),
                                    'title': best_match.get('source_title', '')}],
                        'needs_search': False,
                        'resolved_from': entity_name,
                        'thought_process': [{'step': f'Resolved reference to: {entity_name}',
                                            'type': 'context', 'confidence': 0.9}],
                        'question_type': 'definition',
                        'reasoning_method': 'context_resolution'
                    }
        
        # =====================================================
        # METHOD 1: Try Knowledge Graph FIRST
        # =====================================================
        graph_result = None
        if self.graph_reasoner:
            graph_result = self.graph_reasoner.reason(query)
            
            if graph_result and graph_result['confidence'] >= 0.6 and graph_result['facts_used'] > 0:
                answer = graph_result['answer']
                
                vector_results = self.kb.search(query, limit=3)
                sources = [{'url': r.get('source_url', ''), 'title': r.get('source_title', '')}
                          for r in vector_results if r.get('source_url')]
                
                context.add_user_message(query)
                context.add_assistant_response(answer, [{
                    'name': query,
                    'content': answer[:200],
                    'source_url': sources[0]['url'] if sources else '',
                    'source_title': sources[0]['title'] if sources else '',
                    'confidence': graph_result['confidence']
                }])
                
                return {
                    'response': answer,
                    'confidence': graph_result['confidence'],
                    'sources': sources[:3],
                    'needs_search': False,
                    'reasoning_type': 'knowledge_graph',
                    'facts_used': graph_result['facts_used'],
                    'thought_process': [
                        {'step': 'Knowledge Graph Reasoning', 'type': 'graph',
                         'confidence': graph_result['confidence']},
                        {'step': graph_result.get('reasoning_trace', ''), 'type': 'trace'}
                    ],
                    'question_type': graph_result.get('question_type', 'factual'),
                    'reasoning_method': 'knowledge_graph'
                }
        
        # =====================================================
        # METHOD 2: Vector Search
        # =====================================================
        result = self.reasoning.reason(query)
        
        # Check disambiguation
        if result.answer and not result.needs_search:
            raw_results = self.kb.search(query, limit=10)
            
            if context.needs_disambiguation(raw_results):
                disambig_response, entities = context.format_disambiguation(query, raw_results)
                
                if disambig_response and entities:
                    context.add_user_message(query)
                    context.add_assistant_response(disambig_response, entities)
                    
                    sources = [{'url': e['source_url'], 'title': e['source_title']}
                              for e in entities if e['source_url']]
                    
                    return {
                        'response': disambig_response,
                        'confidence': 0.8,
                        'sources': sources[:5],
                        'needs_search': False,
                        'disambiguation': True,
                        'options': [e['name'] for e in entities],
                        'thought_process': [{'step': 'Multiple matches - asking for clarification',
                                            'type': 'disambiguation', 'confidence': 0.8}],
                        'question_type': 'disambiguation',
                        'reasoning_method': 'disambiguation'
                    }
        
        # Blend graph result if available
        if graph_result and graph_result['confidence'] > 0.1 and result.confidence < 0.5:
            if graph_result['answer'] and "don't have" not in graph_result['answer']:
                result.answer = f"{graph_result['answer']}\n\n{result.answer}" if result.answer else graph_result['answer']
                result.confidence = max(result.confidence, graph_result['confidence'])
        
        # =====================================================
        # METHOD 3: Neural Network (if others fail)
        # =====================================================
        if self.neural_brain and (not result.answer or result.needs_search):
            try:
                neural_result = self.neural_brain.answer(query)
                
                if neural_result.get('answer') and neural_result.get('confidence', 0) > 0.3:
                    # Neural network has something!
                    if not result.answer:
                        result.answer = neural_result['answer']
                        result.confidence = neural_result['confidence']
                        result.reasoning_method = 'neural_network'
                        result.needs_search = False
                    else:
                        # Blend with existing answer
                        result.answer = f"{result.answer}\n\n(Neural: {neural_result['answer']})"
                        result.confidence = max(result.confidence, neural_result['confidence'])
            except Exception as e:
                print(f"⚠️ Neural reasoning error: {e}")
        
        # Update context
        context.add_user_message(query)
        
        if result.answer:
            entities = context.extract_entities_from_response(result.answer, result.sources)
            context.add_assistant_response(result.answer, entities if entities else [{
                'name': query,
                'content': result.answer[:200] if result.answer else '',
                'source_url': result.sources[0]['url'] if result.sources else '',
                'source_title': result.sources[0]['title'] if result.sources else '',
                'confidence': result.confidence
            }])
        
        return {
            'response': result.answer,
            'confidence': result.confidence,
            'sources': result.sources,
            'needs_search': result.needs_search,
            'reasoning_type': result.reasoning_method,
            'thought_process': result.thought_process,
            'question_type': result.question_type.value
        }
    
    def _extract_focused_answer(self, entity_name: str, content: str) -> str:
        """Extract answer focused on a specific entity"""
        # Handle None or empty entity_name
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
        
        context = self.get_context(session_id)
        context.add_user_message(query)
        if result.answer:
            entities = [{
                'name': lc.get('title', query),
                'content': result.answer[:200] if result.answer else '',
                'source_url': lc.get('url', ''),
                'source_title': lc.get('title', ''),
                'confidence': result.confidence
            } for lc in (learned or [])]
            context.add_assistant_response(result.answer, entities)
        
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
        from .context import clear_context
        clear_context(session_id)