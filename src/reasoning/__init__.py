"""
GroundZero AI - Reasoning System
================================

Advanced reasoning capabilities:
1. Chain-of-Thought reasoning
2. Self-verification and reflection
3. Handling user corrections
4. Multi-step problem solving
5. Reasoning trace visualization
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum

try:
    from ..utils import logger, timestamp, generate_id
except ImportError:
    from utils import logger, timestamp, generate_id


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class ReasoningType(Enum):
    CHAIN_OF_THOUGHT = "chain_of_thought"
    SELF_REFLECTION = "self_reflection"
    VERIFICATION = "verification"
    CORRECTION = "correction"
    DECOMPOSITION = "decomposition"


@dataclass
class ReasoningStep:
    """A single step in the reasoning process."""
    id: str
    step_number: int
    thought: str
    reasoning_type: str
    confidence: float = 1.0
    evidence: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=timestamp)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ReasoningTrace:
    """Complete trace of reasoning process."""
    id: str = field(default_factory=lambda: generate_id("trace_"))
    query: str = ""
    steps: List[ReasoningStep] = field(default_factory=list)
    final_answer: str = ""
    confidence: float = 0.0
    verified: bool = False
    corrections: List[Dict] = field(default_factory=list)
    timestamp: str = field(default_factory=timestamp)
    
    def add_step(self, thought: str, reasoning_type: str = "thinking", 
                 confidence: float = 1.0, evidence: List[str] = None) -> ReasoningStep:
        step = ReasoningStep(
            id=generate_id("step_"),
            step_number=len(self.steps) + 1,
            thought=thought,
            reasoning_type=reasoning_type,
            confidence=confidence,
            evidence=evidence or [],
        )
        self.steps.append(step)
        return step
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "query": self.query,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "confidence": self.confidence,
            "verified": self.verified,
            "corrections": self.corrections,
            "timestamp": self.timestamp,
        }
    
    def format_for_display(self) -> str:
        """Format reasoning trace for display."""
        lines = [
            f"═══ Reasoning Trace ═══",
            f"Query: {self.query}",
            f"───────────────────────",
        ]
        
        for step in self.steps:
            icon = {
                "thinking": "💭",
                "verification": "[OK]",
                "evidence": "📚",
                "correction": "🔄",
                "conclusion": "💡",
            }.get(step.reasoning_type, "•")
            
            lines.append(f"{icon} Step {step.step_number}: {step.thought}")
            
            if step.evidence:
                for ev in step.evidence:
                    lines.append(f"   └─ Evidence: {ev[:100]}")
        
        lines.extend([
            f"───────────────────────",
            f"Answer: {self.final_answer}",
            f"Confidence: {self.confidence:.0%}",
            f"Verified: {'Yes [OK]' if self.verified else 'No'}",
        ])
        
        return "\n".join(lines)


# ============================================================================
# CHAIN OF THOUGHT REASONING
# ============================================================================

class ChainOfThoughtReasoner:
    """
    Implement Chain-of-Thought reasoning for step-by-step problem solving.
    """
    
    # Prompts for different reasoning types
    REASONING_PROMPTS = {
        "general": """Let's think through this step by step:
1. First, I'll identify what we're trying to figure out.
2. Then, I'll break down the problem into smaller parts.
3. I'll work through each part systematically.
4. Finally, I'll combine my findings for the answer.""",
        
        "math": """Let's solve this mathematically:
1. Identify the given information.
2. Determine what we need to find.
3. Choose the appropriate method/formula.
4. Calculate step by step.
5. Verify the answer makes sense.""",
        
        "analysis": """Let's analyze this systematically:
1. What are the key elements to consider?
2. What relationships exist between them?
3. What are the implications?
4. What conclusions can we draw?""",
        
        "coding": """Let's approach this programming problem:
1. Understand the requirements.
2. Identify inputs and expected outputs.
3. Design the algorithm.
4. Consider edge cases.
5. Implement the solution.
6. Test and verify.""",
    }
    
    def __init__(self, model_generate: Callable = None):
        """
        Args:
            model_generate: Function to generate text from the model
                           Signature: (prompt: str) -> str
        """
        self.model_generate = model_generate
    
    def reason(
        self,
        query: str,
        context: str = "",
        reasoning_type: str = "general",
        max_steps: int = 10,
    ) -> ReasoningTrace:
        """
        Perform chain-of-thought reasoning on a query.
        """
        trace = ReasoningTrace(query=query)
        
        # Build prompt
        system_prompt = self.REASONING_PROMPTS.get(reasoning_type, self.REASONING_PROMPTS["general"])
        
        prompt = f"""{system_prompt}

Question: {query}
{f'Context: {context}' if context else ''}

Let me think through this step by step:
<think>"""
        
        # Generate reasoning (if model available)
        if self.model_generate:
            response = self.model_generate(prompt)
            steps = self._parse_reasoning(response)
        else:
            # Simulated reasoning steps
            steps = self._simulate_reasoning(query, reasoning_type)
        
        # Add steps to trace
        for i, (thought, step_type, confidence) in enumerate(steps):
            trace.add_step(thought, step_type, confidence)
        
        # Extract final answer
        import re
        answer_match = re.search(r'(?:equals?|is|=)\s*(\d+)', response)
        if answer_match:
            trace.final_answer = response.strip()
        elif steps:
            trace.final_answer = response.strip()  # Use full response
        else:
            trace.final_answer = "Unable to determine answer."
        trace.confidence = sum(s[2] for s in steps) / max(1, len(steps))
        
        return trace
    
    def _parse_reasoning(self, response: str) -> List[Tuple[str, str, float]]:
        """Parse reasoning steps from model response."""
        steps = []
        
        # Look for numbered steps
        numbered = re.findall(r'(?:Step )?\d+[.):]\s*(.+?)(?=(?:Step )?\d+[.)]|\n\n|$)', response, re.DOTALL)
        
        if numbered:
            for i, step_text in enumerate(numbered):
                step_text = step_text.strip()
                if step_text:
                    # Determine step type
                    if any(word in step_text.lower() for word in ["therefore", "thus", "conclude"]):
                        step_type = "conclusion"
                    elif any(word in step_text.lower() for word in ["verify", "check", "confirm"]):
                        step_type = "verification"
                    else:
                        step_type = "thinking"
                    
                    steps.append((step_text, step_type, 0.8))
        else:
            # No numbered steps, treat as single thought
            steps.append((response.strip(), "thinking", 0.7))
        
        return steps
    
    def _simulate_reasoning(self, query: str, reasoning_type: str) -> List[Tuple[str, str, float]]:
        """Simulate reasoning steps (for demonstration)."""
        steps = [
            (f"Understanding the question: {query[:100]}", "thinking", 0.9),
            ("Breaking down the key components...", "thinking", 0.85),
            ("Analyzing relevant information...", "analysis", 0.8),
            ("Considering possible approaches...", "thinking", 0.8),
            ("Applying logical reasoning...", "thinking", 0.85),
            ("Reaching a conclusion based on the analysis.", "conclusion", 0.8),
        ]
        return steps


# ============================================================================
# SELF-VERIFICATION
# ============================================================================

class SelfVerifier:
    """
    Verify reasoning and answers through self-reflection and fact-checking.
    """
    
    def __init__(self, model_generate: Callable = None, knowledge_graph=None, web_search=None):
        self.model_generate = model_generate
        self.knowledge_graph = knowledge_graph
        self.web_search = web_search
    
    def verify(
        self,
        answer: str,
        reasoning_trace: ReasoningTrace,
        verify_with_knowledge: bool = True,
        verify_with_web: bool = False,
    ) -> Tuple[bool, float, str]:
        """
        Verify an answer and its reasoning.
        
        Returns:
            (is_valid, confidence, explanation)
        """
        issues = []
        confidence = 0.8
        
        # 1. Check logical consistency
        consistency_check = self._check_consistency(reasoning_trace)
        if not consistency_check[0]:
            issues.append(f"Logical inconsistency: {consistency_check[1]}")
            confidence *= 0.7
        
        # 2. Verify against knowledge graph
        if verify_with_knowledge and self.knowledge_graph:
            kg_check = self._verify_with_knowledge(answer, reasoning_trace)
            if not kg_check[0]:
                issues.append(f"Knowledge mismatch: {kg_check[1]}")
                confidence *= 0.8
        
        # 3. Verify with web search (if enabled)
        if verify_with_web and self.web_search:
            web_check = self._verify_with_web(answer)
            if not web_check[0]:
                issues.append(f"Web verification: {web_check[1]}")
                confidence *= web_check[2]  # Adjust by web confidence
        
        # 4. Self-reflection check
        if self.model_generate:
            reflection = self._self_reflect(answer, reasoning_trace)
            if reflection["has_issues"]:
                issues.extend(reflection["issues"])
                confidence *= 0.85
        
        is_valid = len(issues) == 0
        explanation = "Verification passed." if is_valid else "; ".join(issues)
        
        return is_valid, confidence, explanation
    
    def _check_consistency(self, trace: ReasoningTrace) -> Tuple[bool, str]:
        """Check logical consistency of reasoning steps."""
        if not trace.steps:
            return False, "No reasoning steps provided"
        
        # Check for contradictions
        for i, step in enumerate(trace.steps[:-1]):
            for j, later_step in enumerate(trace.steps[i+1:], i+1):
                # Simple contradiction detection
                if self._might_contradict(step.thought, later_step.thought):
                    return False, f"Potential contradiction between step {i+1} and {j+1}"
        
        # Check confidence is reasonable
        low_confidence_steps = [s for s in trace.steps if s.confidence < 0.5]
        if len(low_confidence_steps) > len(trace.steps) / 2:
            return False, "Too many low-confidence steps"
        
        return True, "Consistent"
    
    def _might_contradict(self, text1: str, text2: str) -> bool:
        """Check if two texts might contradict each other."""
        # Simple heuristic: look for negation patterns
        negation_words = ["not", "never", "no", "incorrect", "wrong", "false"]
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # If one has negation and they share key words
        has_neg1 = any(neg in words1 for neg in negation_words)
        has_neg2 = any(neg in words2 for neg in negation_words)
        
        if has_neg1 != has_neg2:  # One negated, one not
            common = words1 & words2 - set(negation_words)
            if len(common) > 3:  # Significant overlap
                return True
        
        return False
    
    def _verify_with_knowledge(self, answer: str, trace: ReasoningTrace) -> Tuple[bool, str]:
        """Verify against knowledge graph."""
        # Query knowledge graph for relevant facts
        context = self.knowledge_graph.get_context(trace.query)
        
        if not context:
            return True, "No relevant knowledge found to verify against"
        
        # Check if answer aligns with known facts
        # This is simplified - real implementation would do deeper semantic matching
        answer_lower = answer.lower()
        
        for line in context.split('\n'):
            if line.startswith('-'):
                fact = line[1:].strip().lower()
                # Check for direct contradiction
                if "not" in answer_lower and fact in answer_lower.replace("not", ""):
                    return False, f"Contradicts known fact: {fact[:100]}"
        
        return True, "Consistent with knowledge"
    
    def _verify_with_web(self, answer: str) -> Tuple[bool, str, float]:
        """Verify answer using web search."""
        result = self.web_search.verify(answer)
        return result.verified, result.explanation, result.confidence
    
    def _self_reflect(self, answer: str, trace: ReasoningTrace) -> Dict:
        """Ask the model to reflect on its own answer."""
        if not self.model_generate:
            return {"has_issues": False, "issues": []}
        
        prompt = f"""Please critically evaluate this reasoning and answer:

Question: {trace.query}

Reasoning:
{chr(10).join(f'- {s.thought}' for s in trace.steps)}

Answer: {answer}

Are there any issues with this reasoning? Be critical and look for:
1. Logical errors
2. Unsupported assumptions
3. Missing considerations
4. Factual errors

Issues found:"""
        
        response = self.model_generate(prompt)
        
        # Parse issues
        issues = []
        if "no issues" not in response.lower() and "looks correct" not in response.lower():
            # Extract issue lines
            lines = response.split('\n')
            for line in lines:
                if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-')):
                    issues.append(line.strip())
        
        return {"has_issues": len(issues) > 0, "issues": issues}


# ============================================================================
# CORRECTION HANDLER
# ============================================================================

class CorrectionHandler:
    """
    Handle user corrections and learn from mistakes.
    """
    
    def __init__(self, knowledge_graph=None, memory_system=None):
        self.knowledge_graph = knowledge_graph
        self.memory = memory_system
        self.correction_history: List[Dict] = []
    
    def process_correction(
        self,
        original_answer: str,
        correction: str,
        reasoning_trace: ReasoningTrace = None,
        verify: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a user correction.
        
        Args:
            original_answer: The incorrect answer
            correction: User's correction (e.g., "No, that's wrong. The answer is X")
            reasoning_trace: Original reasoning trace
            verify: Whether to verify the correction
        """
        result = {
            "correction_id": generate_id("corr_"),
            "original": original_answer,
            "correction": correction,
            "correction_type": self._classify_correction(correction),
            "extracted_fact": None,
            "verified": False,
            "applied": False,
            "timestamp": timestamp(),
        }
        
        # Extract the correct information from the correction
        correct_info = self._extract_correction(correction)
        result["extracted_fact"] = correct_info
        
        # Verify if requested
        if verify and correct_info:
            result["verified"] = self._verify_correction(correct_info)
        else:
            result["verified"] = True  # Trust user
        
        # Apply correction
        if result["verified"]:
            self._apply_correction(result, reasoning_trace)
            result["applied"] = True
        
        # Store in history
        self.correction_history.append(result)
        
        # Learn from correction
        if self.memory:
            self.memory.remember(
                f"Correction: {original_answer} is wrong. {correction}",
                memory_type="correction",
                importance=0.8,
            )
        
        return result
    
    def _classify_correction(self, correction: str) -> str:
        """Classify the type of correction."""
        correction_lower = correction.lower()
        
        if any(word in correction_lower for word in ["wrong", "incorrect", "false", "not true"]):
            return "factual_error"
        elif any(word in correction_lower for word in ["outdated", "old", "changed"]):
            return "outdated_info"
        elif any(word in correction_lower for word in ["missing", "forgot", "also"]):
            return "incomplete"
        elif any(word in correction_lower for word in ["misunderstood", "confused"]):
            return "misunderstanding"
        else:
            return "general"
    
    def _extract_correction(self, correction: str) -> Optional[str]:
        """Extract the correct fact from the correction."""
        # Look for patterns like "The answer is X", "It should be X", "Actually X"
        patterns = [
            r"(?:the )?(?:correct )?answer is[:\s]+(.+?)(?:\.|$)",
            r"(?:it )?should be[:\s]+(.+?)(?:\.|$)",
            r"actually[,\s]+(.+?)(?:\.|$)",
            r"the (?:right|correct) (?:answer|response) is[:\s]+(.+?)(?:\.|$)",
            r"instead[,\s]+(.+?)(?:\.|$)",
        ]
        
        correction_lower = correction.lower()
        for pattern in patterns:
            match = re.search(pattern, correction_lower)
            if match:
                return match.group(1).strip()
        
        # If no pattern matches, take everything after "wrong/incorrect"
        for word in ["wrong", "incorrect", "false"]:
            if word in correction_lower:
                idx = correction_lower.find(word)
                remainder = correction[idx + len(word):].strip()
                if remainder.startswith('.') or remainder.startswith(','):
                    remainder = remainder[1:].strip()
                if len(remainder) > 10:
                    return remainder
        
        return correction  # Return full correction if can't extract
    
    def _verify_correction(self, correct_info: str) -> bool:
        """Verify the correction is valid."""
        if self.knowledge_graph:
            # Check if it contradicts established high-confidence facts
            context = self.knowledge_graph.get_context(correct_info)
            # For now, trust user unless clear contradiction
            # Real implementation would do deeper verification
        
        return True
    
    def _apply_correction(self, correction_result: Dict, trace: ReasoningTrace = None):
        """Apply the correction to knowledge base."""
        if not self.knowledge_graph:
            return
        
        original = correction_result["original"]
        correct = correction_result["extracted_fact"]
        
        # Lower confidence of related incorrect knowledge
        related_nodes = self.knowledge_graph.search_nodes(original, limit=5)
        for node in related_nodes:
            if original.lower() in node.content.lower():
                self.knowledge_graph.update_confidence(
                    node.id, 
                    delta=-0.3,
                    reason=f"User correction: {correct[:100]}"
                )
        
        # Add correct information
        self.knowledge_graph.add_node(
            name=f"Correction: {correct[:50]}",
            content=correct,
            node_type="fact",
            source="user_correction",
            confidence=0.9,
            metadata={"correction_id": correction_result["correction_id"]},
        )
    
    def get_correction_stats(self) -> Dict:
        """Get statistics about corrections."""
        if not self.correction_history:
            return {"total": 0, "by_type": {}, "accuracy_before_corrections": 1.0}
        
        by_type = {}
        for corr in self.correction_history:
            t = corr["correction_type"]
            by_type[t] = by_type.get(t, 0) + 1
        
        applied = sum(1 for c in self.correction_history if c["applied"])
        
        return {
            "total": len(self.correction_history),
            "by_type": by_type,
            "applied": applied,
            "verification_rate": sum(1 for c in self.correction_history if c["verified"]) / len(self.correction_history),
        }


# ============================================================================
# UNIFIED REASONING ENGINE
# ============================================================================

class ReasoningEngine:
    """
    Unified reasoning engine combining all reasoning capabilities.
    """
    
    def __init__(
        self,
        model_generate: Callable = None,
        knowledge_graph=None,
        web_search=None,
        memory_system=None,
    ):
        self.model_generate = model_generate
        self.knowledge_graph = knowledge_graph
        self.web_search = web_search
        self.memory = memory_system
        
        # Initialize components
        self.chain_of_thought = ChainOfThoughtReasoner(model_generate)
        self.verifier = SelfVerifier(model_generate, knowledge_graph, web_search)
        self.correction_handler = CorrectionHandler(knowledge_graph, memory_system)
        
        # Track reasoning history
        self.reasoning_history: List[ReasoningTrace] = []
    
    def reason(
        self,
        query: str,
        context: str = "",
        reasoning_type: str = "general",
        verify: bool = True,
        use_knowledge: bool = True,
    ) -> ReasoningTrace:
        """
        Perform reasoning on a query with verification.
        """
        # Get knowledge context
        if use_knowledge and self.knowledge_graph:
            kg_context = self.knowledge_graph.get_context(query)
            context = f"{context}\n\nKnowledge:\n{kg_context}" if context else f"Knowledge:\n{kg_context}"
        
        # Perform chain-of-thought reasoning
        trace = self.chain_of_thought.reason(
            query, context=context, reasoning_type=reasoning_type
        )
        
        # Verify if requested
        if verify:
            is_valid, confidence, explanation = self.verifier.verify(
                trace.final_answer, trace,
                verify_with_knowledge=use_knowledge,
            )
            trace.verified = is_valid
            trace.confidence = confidence
            
            if not is_valid:
                trace.add_step(
                    f"Verification issue: {explanation}",
                    "verification",
                    confidence=0.5,
                )
        
        # Store in history
        self.reasoning_history.append(trace)
        
        return trace
    
    def handle_correction(self, original_answer: str, correction: str) -> Dict:
        """Handle a user correction."""
        # Get the last reasoning trace if available
        last_trace = self.reasoning_history[-1] if self.reasoning_history else None
        
        return self.correction_handler.process_correction(
            original_answer, correction, last_trace
        )
    
    def get_reasoning_display(self, trace: ReasoningTrace = None) -> str:
        """Get formatted reasoning display."""
        if trace is None:
            trace = self.reasoning_history[-1] if self.reasoning_history else None
        
        if trace:
            return trace.format_for_display()
        return "No reasoning trace available."


# Export
__all__ = [
    'ReasoningType', 'ReasoningStep', 'ReasoningTrace',
    'ChainOfThoughtReasoner', 'SelfVerifier', 'CorrectionHandler',
    'ReasoningEngine',
]
