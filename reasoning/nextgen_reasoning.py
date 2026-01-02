"""
Next-Generation Reasoning Engine
=================================
Advanced reasoning system inspired by state-of-the-art LLMs:
- Claude's constitutional approach and careful reasoning
- GPT-4's chain-of-thought and self-consistency
- DeepSeek's step-by-step verification
- Qwen's multi-path exploration

Implements:
1. Progressive Chain-of-Thought (Progressive CoT)
2. Self-Verification and Error Detection
3. Retrieval-Augmented Reasoning (RAR)
4. Confidence Calibration
5. Multi-Hypothesis Reasoning
6. Metacognitive Monitoring
7. Iterative Refinement
"""

import re
import math
import time
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter
import random


class ReasoningStrategy(Enum):
    """Available reasoning strategies"""
    DIRECT = "direct"                # Simple factual recall
    CHAIN_OF_THOUGHT = "cot"         # Step-by-step reasoning
    DECOMPOSITION = "decomposition"  # Break into subproblems
    ANALOGICAL = "analogical"        # Reason by analogy
    CONTRASTIVE = "contrastive"      # Compare and contrast
    HYPOTHETICAL = "hypothetical"    # What-if reasoning
    VERIFICATION = "verification"   # Verify claims
    SYNTHESIS = "synthesis"          # Combine multiple sources


class ThoughtQuality(Enum):
    """Quality assessment of a thought"""
    STRONG = "strong"        # Well-supported, confident
    MODERATE = "moderate"    # Some support, reasonable
    WEAK = "weak"            # Little support, speculative
    UNCERTAIN = "uncertain"  # Cannot assess


@dataclass
class ReasoningStep:
    """A single step in the reasoning process"""
    step_id: int
    content: str
    step_type: str  # 'observation', 'analysis', 'inference', 'verification', 'conclusion'
    confidence: float
    evidence: List[str] = field(default_factory=list)
    reasoning: str = ""
    quality: ThoughtQuality = ThoughtQuality.MODERATE
    parent_step: Optional[int] = None


@dataclass
class ReasoningHypothesis:
    """A hypothesis being evaluated"""
    hypothesis: str
    support_score: float
    evidence_for: List[str] = field(default_factory=list)
    evidence_against: List[str] = field(default_factory=list)
    confidence: float = 0.5


@dataclass
class ReasoningResult:
    """Complete reasoning result"""
    query: str
    answer: str
    confidence: float
    strategy_used: ReasoningStrategy
    steps: List[ReasoningStep]
    hypotheses_considered: List[ReasoningHypothesis]
    alternative_answers: List[Dict[str, Any]]
    uncertainties: List[str]
    verification_status: str  # 'verified', 'partially_verified', 'unverified'
    reasoning_trace: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeInterface:
    """Interface for accessing knowledge base"""
    
    def __init__(self, memory_store=None):
        self.memory = memory_store
    
    def retrieve(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant knowledge"""
        if not self.memory:
            return []
        
        try:
            results = self.memory.search_knowledge(query, limit=limit)
            return results
        except:
            return []
    
    def has_knowledge(self, topic: str) -> Tuple[bool, float]:
        """Check if we have knowledge about a topic"""
        results = self.retrieve(topic, limit=3)
        
        if not results:
            return False, 0.0
        
        # Calculate coverage score
        avg_confidence = sum(r.get('confidence', 0.5) for r in results) / len(results)
        return True, avg_confidence


class QueryAnalyzer:
    """Analyzes queries to determine best reasoning approach"""
    
    # Question type patterns
    PATTERNS = {
        'definition': [r'what is', r'what are', r'define', r'meaning of', r'explain what'],
        'causal': [r'why', r'how come', r'reason for', r'cause of', r'because'],
        'procedural': [r'how to', r'how do', r'steps to', r'process of', r'way to'],
        'comparative': [r'compare', r'difference', r'versus', r'vs', r'similar to', r'different from'],
        'evaluative': [r'should', r'best', r'recommend', r'better', r'worse', r'opinion'],
        'factual': [r'when', r'where', r'who', r'which', r'how many', r'how much'],
        'hypothetical': [r'what if', r'suppose', r'imagine', r'would.*if', r'could.*if'],
        'analytical': [r'analyze', r'examine', r'investigate', r'assess', r'evaluate'],
    }
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """Analyze a query for reasoning requirements"""
        query_lower = query.lower().strip()
        
        # Detect question type
        question_type = self._detect_question_type(query_lower)
        
        # Extract key concepts
        concepts = self._extract_concepts(query)
        
        # Assess complexity
        complexity = self._assess_complexity(query_lower)
        
        # Recommend strategy
        strategy = self._recommend_strategy(question_type, complexity)
        
        return {
            'query': query,
            'question_type': question_type,
            'concepts': concepts,
            'complexity': complexity,
            'recommended_strategy': strategy,
            'requires_knowledge': self._requires_knowledge(question_type),
            'is_multi_part': self._is_multi_part(query_lower)
        }
    
    def _detect_question_type(self, query: str) -> str:
        """Detect the type of question"""
        for q_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    return q_type
        return 'general'
    
    def _extract_concepts(self, query: str) -> List[str]:
        """Extract key concepts from query"""
        stopwords = {
            'what', 'is', 'are', 'the', 'a', 'an', 'how', 'why', 'when', 'where',
            'who', 'which', 'does', 'do', 'can', 'could', 'would', 'should',
            'will', 'about', 'tell', 'me', 'please', 'explain', 'describe',
            'this', 'that', 'these', 'those', 'it', 'they', 'them', 'i', 'you'
        }
        
        words = re.findall(r'\b\w+\b', query.lower())
        concepts = [w for w in words if w not in stopwords and len(w) > 2]
        
        # Also extract multi-word concepts
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)
                   if words[i] not in stopwords and words[i+1] not in stopwords]
        
        return list(set(concepts + bigrams))[:10]
    
    def _assess_complexity(self, query: str) -> float:
        """Assess query complexity (0-1)"""
        score = 0.3  # Base
        
        # Length factor
        words = len(query.split())
        if words > 20:
            score += 0.2
        elif words > 10:
            score += 0.1
        
        # Complex question indicators
        if any(w in query for w in ['why', 'how', 'analyze', 'compare', 'evaluate']):
            score += 0.2
        
        # Multiple parts
        if ' and ' in query or '?' in query[:-1]:
            score += 0.1
        
        # Technical terms (capitalized words)
        caps = len(re.findall(r'\b[A-Z][a-z]+\b', query))
        score += min(0.1, caps * 0.02)
        
        return min(1.0, score)
    
    def _recommend_strategy(self, q_type: str, complexity: float) -> ReasoningStrategy:
        """Recommend reasoning strategy"""
        strategy_map = {
            'definition': ReasoningStrategy.DIRECT,
            'causal': ReasoningStrategy.CHAIN_OF_THOUGHT,
            'procedural': ReasoningStrategy.DECOMPOSITION,
            'comparative': ReasoningStrategy.CONTRASTIVE,
            'evaluative': ReasoningStrategy.SYNTHESIS,
            'factual': ReasoningStrategy.DIRECT,
            'hypothetical': ReasoningStrategy.HYPOTHETICAL,
            'analytical': ReasoningStrategy.CHAIN_OF_THOUGHT,
        }
        
        base_strategy = strategy_map.get(q_type, ReasoningStrategy.CHAIN_OF_THOUGHT)
        
        # Upgrade for complex queries
        if complexity > 0.7:
            if base_strategy == ReasoningStrategy.DIRECT:
                return ReasoningStrategy.CHAIN_OF_THOUGHT
        
        return base_strategy
    
    def _requires_knowledge(self, q_type: str) -> bool:
        """Check if question requires external knowledge"""
        return q_type in ['definition', 'factual', 'causal', 'analytical']
    
    def _is_multi_part(self, query: str) -> bool:
        """Check if query has multiple parts"""
        return ' and ' in query or query.count('?') > 1 or ' or ' in query


class ChainOfThoughtReasoner:
    """
    Progressive Chain-of-Thought reasoning.
    Builds reasoning step by step with self-checking.
    """
    
    def __init__(self, knowledge: KnowledgeInterface):
        self.knowledge = knowledge
        self.step_counter = 0
    
    def reason(self, query: str, analysis: Dict[str, Any]) -> List[ReasoningStep]:
        """Generate chain of thought reasoning"""
        self.step_counter = 0
        steps = []
        
        # Step 1: Understand the question
        steps.append(self._create_step(
            f"Understanding: This is a {analysis['question_type']} question about {', '.join(analysis['concepts'][:3])}",
            'observation',
            0.9
        ))
        
        # Step 2: Identify what we need
        steps.append(self._create_step(
            f"To answer this, I need to: {self._identify_requirements(analysis)}",
            'analysis',
            0.85
        ))
        
        # Step 3: Retrieve relevant knowledge
        knowledge_steps = self._retrieve_and_reason(analysis['concepts'])
        steps.extend(knowledge_steps)
        
        # Step 4: Build reasoning chain
        reasoning_steps = self._build_reasoning_chain(query, analysis, knowledge_steps)
        steps.extend(reasoning_steps)
        
        # Step 5: Verify reasoning
        verification = self._verify_chain(steps)
        steps.append(verification)
        
        return steps
    
    def _create_step(self, content: str, step_type: str, confidence: float,
                     evidence: List[str] = None, reasoning: str = "") -> ReasoningStep:
        """Create a reasoning step"""
        self.step_counter += 1
        return ReasoningStep(
            step_id=self.step_counter,
            content=content,
            step_type=step_type,
            confidence=confidence,
            evidence=evidence or [],
            reasoning=reasoning,
            quality=self._assess_quality(confidence, evidence)
        )
    
    def _assess_quality(self, confidence: float, evidence: List[str] = None) -> ThoughtQuality:
        """Assess thought quality"""
        if confidence >= 0.8 and evidence:
            return ThoughtQuality.STRONG
        elif confidence >= 0.5:
            return ThoughtQuality.MODERATE
        elif confidence >= 0.3:
            return ThoughtQuality.WEAK
        else:
            return ThoughtQuality.UNCERTAIN
    
    def _identify_requirements(self, analysis: Dict) -> str:
        """Identify what's needed to answer"""
        q_type = analysis['question_type']
        
        requirements = {
            'definition': "find a clear definition and examples",
            'causal': "identify causes, mechanisms, and effects",
            'procedural': "outline the steps in order",
            'comparative': "identify similarities and differences",
            'evaluative': "consider criteria and make a judgment",
            'factual': "find the specific fact or data",
            'hypothetical': "consider conditions and implications",
            'analytical': "break down components and their relationships",
            'general': "gather relevant information and synthesize"
        }
        
        return requirements.get(q_type, requirements['general'])
    
    def _retrieve_and_reason(self, concepts: List[str]) -> List[ReasoningStep]:
        """Retrieve knowledge and create reasoning steps"""
        steps = []
        
        for concept in concepts[:3]:  # Top 3 concepts
            results = self.knowledge.retrieve(concept, limit=2)
            
            if results:
                for r in results[:1]:  # Best result per concept
                    content = r.get('content', '')[:200]
                    if content:
                        steps.append(self._create_step(
                            f"Knowledge about '{concept}': {content}...",
                            'observation',
                            r.get('confidence', 0.5),
                            evidence=[r.get('source_url', '')]
                        ))
        
        if not steps:
            steps.append(self._create_step(
                "No directly relevant knowledge found - will reason from general principles",
                'observation',
                0.4
            ))
        
        return steps
    
    def _build_reasoning_chain(self, query: str, analysis: Dict, 
                               knowledge_steps: List[ReasoningStep]) -> List[ReasoningStep]:
        """Build the main reasoning chain"""
        steps = []
        
        # Analysis based on question type
        q_type = analysis['question_type']
        
        if q_type == 'causal':
            steps.extend(self._causal_reasoning(query, knowledge_steps))
        elif q_type == 'comparative':
            steps.extend(self._comparative_reasoning(query, knowledge_steps))
        elif q_type == 'procedural':
            steps.extend(self._procedural_reasoning(query, knowledge_steps))
        else:
            steps.extend(self._general_reasoning(query, knowledge_steps))
        
        return steps
    
    def _causal_reasoning(self, query: str, knowledge: List[ReasoningStep]) -> List[ReasoningStep]:
        """Reasoning for causal questions"""
        steps = []
        
        steps.append(self._create_step(
            "Identifying potential causes and mechanisms...",
            'analysis',
            0.7
        ))
        
        if knowledge:
            evidence = [s.content for s in knowledge if s.evidence]
            steps.append(self._create_step(
                "Based on available evidence, the causal factors include: " + 
                ", ".join([s.content[:50] for s in knowledge[:2]]),
                'inference',
                0.65,
                evidence=evidence
            ))
        
        return steps
    
    def _comparative_reasoning(self, query: str, knowledge: List[ReasoningStep]) -> List[ReasoningStep]:
        """Reasoning for comparative questions"""
        steps = []
        
        steps.append(self._create_step(
            "Identifying aspects to compare...",
            'analysis',
            0.7
        ))
        
        steps.append(self._create_step(
            "Analyzing similarities and differences...",
            'analysis',
            0.65
        ))
        
        return steps
    
    def _procedural_reasoning(self, query: str, knowledge: List[ReasoningStep]) -> List[ReasoningStep]:
        """Reasoning for procedural questions"""
        steps = []
        
        steps.append(self._create_step(
            "Breaking down into sequential steps...",
            'analysis',
            0.7
        ))
        
        return steps
    
    def _general_reasoning(self, query: str, knowledge: List[ReasoningStep]) -> List[ReasoningStep]:
        """General reasoning approach"""
        steps = []
        
        steps.append(self._create_step(
            "Analyzing the available information...",
            'analysis',
            0.7
        ))
        
        if knowledge:
            steps.append(self._create_step(
                "Synthesizing findings from knowledge base...",
                'inference',
                0.65
            ))
        
        return steps
    
    def _verify_chain(self, steps: List[ReasoningStep]) -> ReasoningStep:
        """Verify the reasoning chain"""
        # Calculate chain integrity
        avg_confidence = sum(s.confidence for s in steps) / max(len(steps), 1)
        evidence_count = sum(1 for s in steps if s.evidence)
        
        if avg_confidence >= 0.7 and evidence_count >= 2:
            status = "Reasoning appears sound and well-supported"
            confidence = 0.8
        elif avg_confidence >= 0.5:
            status = "Reasoning is plausible but could use more support"
            confidence = 0.6
        else:
            status = "Reasoning may need additional verification"
            confidence = 0.4
        
        return self._create_step(
            f"Self-check: {status}",
            'verification',
            confidence
        )


class SelfVerifier:
    """
    Self-verification module.
    Checks reasoning for errors and inconsistencies.
    Inspired by Constitutional AI and DeepSeek's verification approach.
    """
    
    def __init__(self, knowledge: KnowledgeInterface):
        self.knowledge = knowledge
    
    def verify(self, query: str, steps: List[ReasoningStep], 
               answer: str) -> Dict[str, Any]:
        """Verify reasoning and answer"""
        checks = []
        
        # Check 1: Does answer address the question?
        relevance = self._check_relevance(query, answer)
        checks.append({
            'name': 'Relevance',
            'passed': relevance > 0.5,
            'score': relevance,
            'detail': 'Answer addresses the question' if relevance > 0.5 else 'Answer may not fully address the question'
        })
        
        # Check 2: Is reasoning consistent?
        consistency = self._check_consistency(steps)
        checks.append({
            'name': 'Consistency',
            'passed': consistency > 0.6,
            'score': consistency,
            'detail': 'Reasoning is consistent' if consistency > 0.6 else 'Some inconsistencies detected'
        })
        
        # Check 3: Is evidence sufficient?
        evidence = self._check_evidence(steps)
        checks.append({
            'name': 'Evidence',
            'passed': evidence > 0.5,
            'score': evidence,
            'detail': 'Sufficient evidence' if evidence > 0.5 else 'More evidence needed'
        })
        
        # Check 4: Are there logical gaps?
        completeness = self._check_completeness(steps)
        checks.append({
            'name': 'Completeness',
            'passed': completeness > 0.5,
            'score': completeness,
            'detail': 'Reasoning is complete' if completeness > 0.5 else 'Some logical gaps'
        })
        
        # Overall status
        all_passed = all(c['passed'] for c in checks)
        avg_score = sum(c['score'] for c in checks) / len(checks)
        
        if all_passed:
            status = 'verified'
        elif avg_score >= 0.5:
            status = 'partially_verified'
        else:
            status = 'unverified'
        
        return {
            'status': status,
            'overall_score': avg_score,
            'checks': checks,
            'recommendation': self._get_recommendation(checks)
        }
    
    def _check_relevance(self, query: str, answer: str) -> float:
        """Check if answer is relevant to query"""
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        answer_terms = set(re.findall(r'\b\w+\b', answer.lower()))
        
        stopwords = {'the', 'is', 'are', 'a', 'an', 'to', 'of', 'in', 'for', 'on', 'with'}
        query_terms -= stopwords
        answer_terms -= stopwords
        
        if not query_terms:
            return 0.5
        
        overlap = len(query_terms & answer_terms)
        return min(1.0, 0.3 + overlap / len(query_terms) * 0.7)
    
    def _check_consistency(self, steps: List[ReasoningStep]) -> float:
        """Check reasoning consistency"""
        if len(steps) < 2:
            return 0.5
        
        score = 0.7  # Base
        
        # Check for contradictions
        for i, step in enumerate(steps):
            content = step.content.lower()
            
            # Look for negation patterns that might indicate inconsistency
            for j, prev_step in enumerate(steps[:i]):
                prev_content = prev_step.content.lower()
                
                # Simple contradiction check
                if 'not' in content and any(w in prev_content for w in content.split() if len(w) > 4):
                    score -= 0.1
        
        # Bonus for logical connectors
        connectors = ['therefore', 'because', 'thus', 'hence', 'since', 'so']
        for step in steps:
            if any(c in step.content.lower() for c in connectors):
                score += 0.05
        
        return max(0.0, min(1.0, score))
    
    def _check_evidence(self, steps: List[ReasoningStep]) -> float:
        """Check evidence quality"""
        evidence_steps = [s for s in steps if s.evidence]
        strong_steps = [s for s in steps if s.quality == ThoughtQuality.STRONG]
        
        if not steps:
            return 0.0
        
        evidence_ratio = len(evidence_steps) / len(steps)
        quality_ratio = len(strong_steps) / len(steps)
        
        return 0.4 + evidence_ratio * 0.3 + quality_ratio * 0.3
    
    def _check_completeness(self, steps: List[ReasoningStep]) -> float:
        """Check reasoning completeness"""
        # Check for key step types
        has_observation = any(s.step_type == 'observation' for s in steps)
        has_analysis = any(s.step_type == 'analysis' for s in steps)
        has_inference = any(s.step_type in ['inference', 'conclusion'] for s in steps)
        
        score = 0.4
        if has_observation:
            score += 0.2
        if has_analysis:
            score += 0.2
        if has_inference:
            score += 0.2
        
        return score
    
    def _get_recommendation(self, checks: List[Dict]) -> str:
        """Get improvement recommendation"""
        failed = [c for c in checks if not c['passed']]
        
        if not failed:
            return "Reasoning is sound"
        
        recommendations = {
            'Relevance': "Try to more directly address the question",
            'Consistency': "Review for logical contradictions",
            'Evidence': "Seek additional supporting information",
            'Completeness': "Consider if any reasoning steps are missing"
        }
        
        return recommendations.get(failed[0]['name'], "Review reasoning carefully")


class HypothesisGenerator:
    """
    Generates and evaluates multiple hypotheses.
    Implements multi-path reasoning with hypothesis testing.
    """
    
    def __init__(self, knowledge: KnowledgeInterface):
        self.knowledge = knowledge
    
    def generate_hypotheses(self, query: str, analysis: Dict, 
                           max_hypotheses: int = 3) -> List[ReasoningHypothesis]:
        """Generate multiple hypotheses for the query"""
        hypotheses = []
        
        # Get query type
        q_type = analysis['question_type']
        concepts = analysis['concepts']
        
        # Generate hypotheses based on query type
        if q_type == 'causal':
            hypotheses = self._generate_causal_hypotheses(concepts)
        elif q_type == 'comparative':
            hypotheses = self._generate_comparative_hypotheses(concepts)
        elif q_type == 'evaluative':
            hypotheses = self._generate_evaluative_hypotheses(concepts)
        else:
            hypotheses = self._generate_general_hypotheses(concepts)
        
        # Evaluate each hypothesis
        for h in hypotheses:
            self._evaluate_hypothesis(h)
        
        # Sort by support score
        hypotheses.sort(key=lambda x: x.support_score, reverse=True)
        
        return hypotheses[:max_hypotheses]
    
    def _generate_causal_hypotheses(self, concepts: List[str]) -> List[ReasoningHypothesis]:
        """Generate causal hypotheses"""
        hypotheses = []
        
        if len(concepts) >= 1:
            hypotheses.append(ReasoningHypothesis(
                hypothesis=f"The primary cause involves {concepts[0]}",
                support_score=0.5
            ))
            hypotheses.append(ReasoningHypothesis(
                hypothesis=f"Multiple factors contribute including {', '.join(concepts[:2])}",
                support_score=0.5
            ))
        
        return hypotheses
    
    def _generate_comparative_hypotheses(self, concepts: List[str]) -> List[ReasoningHypothesis]:
        """Generate comparative hypotheses"""
        hypotheses = []
        
        if len(concepts) >= 2:
            hypotheses.append(ReasoningHypothesis(
                hypothesis=f"{concepts[0]} and {concepts[1]} are fundamentally similar",
                support_score=0.5
            ))
            hypotheses.append(ReasoningHypothesis(
                hypothesis=f"{concepts[0]} and {concepts[1]} have key differences",
                support_score=0.5
            ))
        
        return hypotheses
    
    def _generate_evaluative_hypotheses(self, concepts: List[str]) -> List[ReasoningHypothesis]:
        """Generate evaluative hypotheses"""
        hypotheses = []
        
        if concepts:
            hypotheses.append(ReasoningHypothesis(
                hypothesis=f"{concepts[0]} is beneficial/recommended",
                support_score=0.5
            ))
            hypotheses.append(ReasoningHypothesis(
                hypothesis=f"{concepts[0]} has both pros and cons to consider",
                support_score=0.5
            ))
        
        return hypotheses
    
    def _generate_general_hypotheses(self, concepts: List[str]) -> List[ReasoningHypothesis]:
        """Generate general hypotheses"""
        hypotheses = []
        
        if concepts:
            hypotheses.append(ReasoningHypothesis(
                hypothesis=f"The answer relates primarily to {concepts[0]}",
                support_score=0.5
            ))
        
        return hypotheses
    
    def _evaluate_hypothesis(self, hypothesis: ReasoningHypothesis) -> None:
        """Evaluate a hypothesis against knowledge"""
        # Search for supporting evidence
        results = self.knowledge.retrieve(hypothesis.hypothesis, limit=3)
        
        for r in results:
            content = r.get('content', '').lower()
            
            # Simple heuristic: check for positive/negative indicators
            positive_words = ['is', 'are', 'can', 'does', 'has', 'supports', 'shows']
            negative_words = ['not', "isn't", "aren't", "can't", "doesn't", "hasn't", 'however']
            
            pos_count = sum(1 for w in positive_words if w in content)
            neg_count = sum(1 for w in negative_words if w in content)
            
            if pos_count > neg_count:
                hypothesis.evidence_for.append(content[:100])
                hypothesis.support_score += 0.1
            elif neg_count > pos_count:
                hypothesis.evidence_against.append(content[:100])
                hypothesis.support_score -= 0.05
        
        # Calculate final confidence
        hypothesis.confidence = min(1.0, max(0.0, hypothesis.support_score))


class NextGenReasoningEngine:
    """
    Main reasoning engine orchestrating all components.
    Implements state-of-the-art reasoning techniques.
    """
    
    def __init__(self, memory_store=None):
        self.knowledge = KnowledgeInterface(memory_store)
        self.analyzer = QueryAnalyzer()
        self.cot_reasoner = ChainOfThoughtReasoner(self.knowledge)
        self.verifier = SelfVerifier(self.knowledge)
        self.hypothesis_gen = HypothesisGenerator(self.knowledge)
    
    def reason(self, query: str, deep_think: bool = False) -> ReasoningResult:
        """
        Main reasoning entry point.
        
        Args:
            query: The question or prompt
            deep_think: Whether to use comprehensive reasoning
        
        Returns:
            Complete reasoning result
        """
        start_time = time.time()
        
        # Step 1: Analyze query
        analysis = self.analyzer.analyze(query)
        
        # Step 2: Check if we have knowledge
        has_knowledge, knowledge_conf = self.knowledge.has_knowledge(
            ' '.join(analysis['concepts'][:3])
        )
        
        # Step 3: Choose reasoning depth
        if deep_think or analysis['complexity'] > 0.7:
            result = self._comprehensive_reasoning(query, analysis)
        elif analysis['complexity'] > 0.4:
            result = self._moderate_reasoning(query, analysis)
        else:
            result = self._quick_reasoning(query, analysis)
        
        result.metadata['processing_time'] = time.time() - start_time
        result.metadata['had_knowledge'] = has_knowledge
        result.metadata['knowledge_confidence'] = knowledge_conf
        
        return result
    
    def _comprehensive_reasoning(self, query: str, analysis: Dict) -> ReasoningResult:
        """Full reasoning with all techniques"""
        # Generate chain of thought
        steps = self.cot_reasoner.reason(query, analysis)
        
        # Generate hypotheses
        hypotheses = self.hypothesis_gen.generate_hypotheses(query, analysis)
        
        # Form answer from reasoning
        answer = self._synthesize_answer(query, steps, hypotheses)
        
        # Verify reasoning
        verification = self.verifier.verify(query, steps, answer)
        
        # Calculate confidence
        confidence = self._calculate_confidence(steps, hypotheses, verification)
        
        # Find alternative answers
        alternatives = self._find_alternatives(hypotheses)
        
        # Identify uncertainties
        uncertainties = self._identify_uncertainties(steps, verification)
        
        return ReasoningResult(
            query=query,
            answer=answer,
            confidence=confidence,
            strategy_used=analysis['recommended_strategy'],
            steps=steps,
            hypotheses_considered=hypotheses,
            alternative_answers=alternatives,
            uncertainties=uncertainties,
            verification_status=verification['status'],
            reasoning_trace=self._format_trace(steps)
        )
    
    def _moderate_reasoning(self, query: str, analysis: Dict) -> ReasoningResult:
        """Moderate depth reasoning"""
        steps = self.cot_reasoner.reason(query, analysis)
        answer = self._synthesize_answer(query, steps, [])
        
        return ReasoningResult(
            query=query,
            answer=answer,
            confidence=self._calculate_simple_confidence(steps),
            strategy_used=analysis['recommended_strategy'],
            steps=steps,
            hypotheses_considered=[],
            alternative_answers=[],
            uncertainties=[],
            verification_status='basic',
            reasoning_trace=self._format_trace(steps)
        )
    
    def _quick_reasoning(self, query: str, analysis: Dict) -> ReasoningResult:
        """Quick reasoning for simple queries"""
        # Just retrieve and respond
        knowledge = self.knowledge.retrieve(' '.join(analysis['concepts'][:2]), limit=2)
        
        if knowledge:
            content = knowledge[0].get('content', '')
            answer = content[:300] if content else "I don't have specific information about this."
            confidence = knowledge[0].get('confidence', 0.5)
        else:
            answer = "I don't have enough information to answer this question."
            confidence = 0.3
        
        return ReasoningResult(
            query=query,
            answer=answer,
            confidence=confidence,
            strategy_used=ReasoningStrategy.DIRECT,
            steps=[],
            hypotheses_considered=[],
            alternative_answers=[],
            uncertainties=['Limited knowledge on this topic'] if not knowledge else [],
            verification_status='quick',
            reasoning_trace=""
        )
    
    def _synthesize_answer(self, query: str, steps: List[ReasoningStep],
                          hypotheses: List[ReasoningHypothesis]) -> str:
        """Synthesize final answer from reasoning"""
        # Gather evidence from steps
        evidence = []
        for step in steps:
            if step.step_type in ['observation', 'inference'] and step.evidence:
                evidence.extend(step.evidence)
            if step.step_type == 'observation' and 'Knowledge about' in step.content:
                # Extract the knowledge content
                match = re.search(r"Knowledge about '[^']+': (.+?)\.{3}$", step.content)
                if match:
                    evidence.append(match.group(1))
        
        # Use best hypothesis if available
        if hypotheses and hypotheses[0].confidence > 0.6:
            best_hypothesis = hypotheses[0].hypothesis
        else:
            best_hypothesis = None
        
        # Build answer
        if evidence:
            answer = '. '.join(evidence[:3])
            if len(answer) > 400:
                answer = answer[:400] + '...'
        elif best_hypothesis:
            answer = best_hypothesis
        else:
            answer = "Based on my analysis, I cannot provide a definitive answer without more information."
        
        return answer
    
    def _calculate_confidence(self, steps: List[ReasoningStep],
                             hypotheses: List[ReasoningHypothesis],
                             verification: Dict) -> float:
        """Calculate overall confidence"""
        # Step confidence
        if steps:
            step_conf = sum(s.confidence for s in steps) / len(steps)
        else:
            step_conf = 0.3
        
        # Hypothesis confidence
        if hypotheses:
            hyp_conf = max(h.confidence for h in hypotheses)
        else:
            hyp_conf = 0.5
        
        # Verification score
        ver_score = verification.get('overall_score', 0.5)
        
        # Weighted combination
        confidence = step_conf * 0.4 + hyp_conf * 0.3 + ver_score * 0.3
        
        return min(1.0, max(0.0, confidence))
    
    def _calculate_simple_confidence(self, steps: List[ReasoningStep]) -> float:
        """Simple confidence calculation"""
        if not steps:
            return 0.3
        
        avg_conf = sum(s.confidence for s in steps) / len(steps)
        evidence_bonus = min(0.2, sum(0.05 for s in steps if s.evidence))
        
        return min(1.0, avg_conf + evidence_bonus)
    
    def _find_alternatives(self, hypotheses: List[ReasoningHypothesis]) -> List[Dict]:
        """Find alternative answers"""
        alternatives = []
        
        for h in hypotheses[1:3]:  # Skip best, take next 2
            alternatives.append({
                'answer': h.hypothesis,
                'confidence': h.confidence
            })
        
        return alternatives
    
    def _identify_uncertainties(self, steps: List[ReasoningStep],
                               verification: Dict) -> List[str]:
        """Identify areas of uncertainty"""
        uncertainties = []
        
        # Weak steps
        weak_steps = [s for s in steps if s.quality in [ThoughtQuality.WEAK, ThoughtQuality.UNCERTAIN]]
        if weak_steps:
            uncertainties.append(f"{len(weak_steps)} reasoning steps have low confidence")
        
        # Failed verification checks
        for check in verification.get('checks', []):
            if not check.get('passed'):
                uncertainties.append(check.get('detail', 'Verification issue'))
        
        return uncertainties
    
    def _format_trace(self, steps: List[ReasoningStep]) -> str:
        """Format reasoning trace for display"""
        lines = []
        
        for step in steps:
            icon = {
                'observation': '👁️',
                'analysis': '🔍',
                'inference': '💡',
                'verification': '✓',
                'conclusion': '✅'
            }.get(step.step_type, '•')
            
            lines.append(f"{icon} {step.content}")
        
        return '\n'.join(lines)
    
    def get_thought_process_for_ui(self, result: ReasoningResult) -> List[Dict]:
        """Get thought process formatted for UI display"""
        thoughts = []
        
        for step in result.steps:
            thoughts.append({
                'step': step.content,
                'type': step.step_type,
                'confidence': step.confidence
            })
        
        return thoughts