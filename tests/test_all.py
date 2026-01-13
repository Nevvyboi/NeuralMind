"""
GroundZero AI - Comprehensive Test Suite
========================================

Run all tests:
    python -m pytest tests/
    
Or run directly:
    python tests/test_all.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge_graph import KnowledgeGraph, KnowledgeTriple
from src.causal_graph import CausalGraph, CausalRelation
from src.question_detector import QuestionTypeDetector, QuestionType, ThinkingMode
from src.metacognition import MetacognitiveController, MetacognitiveState
from src.reasoning import ChainOfThoughtReasoner, ReasoningStep
from src.constitutional import Constitution
from src.chat_engine import SmartChatEngine
from src.progress_tracker import ProgressTracker


class TestKnowledgeGraph:
    """Tests for KnowledgeGraph component"""
    
    def test_add_and_query(self):
        """Test adding and querying facts"""
        kg = KnowledgeGraph()
        
        # Add facts
        assert kg.Add("dog", "is_a", "animal") == True
        assert kg.Add("cat", "is_a", "animal") == True
        assert kg.Add("dog", "has", "tail") == True
        
        # Duplicate should return False
        assert kg.Add("dog", "is_a", "animal") == False
        
        # Query by subject
        facts = kg.Query(Subject="dog")
        assert len(facts) == 2
        
        # Query by predicate
        facts = kg.Query(Predicate="is_a")
        assert len(facts) == 2
        
        # Query by object
        facts = kg.Query(Object="animal")
        assert len(facts) == 2
    
    def test_transitive_inference(self):
        """Test transitive inference (A→B, B→C ⟹ A→C)"""
        kg = KnowledgeGraph()
        
        kg.Add("dog", "is_a", "mammal")
        kg.Add("mammal", "is_a", "animal")
        kg.Add("animal", "is_a", "living_thing")
        
        inferred = kg.InferTransitive("is_a")
        
        # Should infer dog→animal, dog→living_thing, mammal→living_thing
        assert len(inferred) >= 3
    
    def test_text_extraction(self):
        """Test extracting triples from text"""
        kg = KnowledgeGraph()
        
        text = "Dogs are animals. Cats are pets. Fire causes smoke."
        triples = kg.ExtractFromText(text)
        
        assert len(triples) >= 2
    
    def test_get_related(self):
        """Test multi-hop relation finding"""
        kg = KnowledgeGraph()
        
        kg.Add("dog", "is_a", "mammal")
        kg.Add("mammal", "is_a", "animal")
        kg.Add("dog", "has", "fur")
        
        related = kg.GetRelated("dog", MaxDepth=2)
        
        assert len(related) >= 3


class TestCausalGraph:
    """Tests for CausalGraph component"""
    
    def test_add_and_query(self):
        """Test adding and querying causal relations"""
        cg = CausalGraph()
        
        assert cg.AddCause("rain", "wet_ground", 0.9) == True
        assert cg.AddCause("wet_ground", "slippery", 0.8) == True
        
        effects = cg.GetEffects("rain")
        assert len(effects) == 1
        assert effects[0].Effect == "wet_ground"
        
        causes = cg.GetCauses("slippery")
        assert len(causes) == 1
        assert causes[0].Cause == "wet_ground"
    
    def test_causal_chain(self):
        """Test finding causal chains"""
        cg = CausalGraph()
        
        cg.AddCause("rain", "wet_ground", 0.9)
        cg.AddCause("wet_ground", "slippery", 0.8)
        cg.AddCause("slippery", "accident", 0.6)
        
        chains = cg.CausalChain("rain", "accident")
        
        assert len(chains) >= 1
        assert len(chains[0]) == 3  # rain→wet→slippery→accident
    
    def test_counterfactual(self):
        """Test counterfactual reasoning"""
        cg = CausalGraph()
        
        cg.AddCause("rain", "wet_ground", 0.9)
        cg.AddCause("wet_ground", "slippery", 0.8)
        
        effects = cg.Counterfactual("rain", Intervention=True)
        
        assert "wet_ground" in effects
        assert "slippery" in effects
        assert abs(effects["wet_ground"] - 0.9) < 0.01
        assert abs(effects["slippery"] - 0.72) < 0.01
    
    def test_learn_from_text(self):
        """Test learning causal relations from text"""
        cg = CausalGraph()
        
        count = cg.LearnFromText("Fire causes smoke. Rain leads to flooding.")
        
        assert count >= 2


class TestQuestionDetector:
    """Tests for QuestionTypeDetector component"""
    
    def test_greeting_detection(self):
        """Test greeting detection"""
        detector = QuestionTypeDetector()
        
        qtype, conf = detector.Detect("Hello!")
        assert qtype == QuestionType.GREETING
        
        qtype, conf = detector.Detect("Hi there")
        assert qtype == QuestionType.GREETING
    
    def test_factual_detection(self):
        """Test factual question detection"""
        detector = QuestionTypeDetector()
        
        qtype, conf = detector.Detect("What is AI?")
        assert qtype == QuestionType.DEFINITIONAL or qtype == QuestionType.FACTUAL
    
    def test_causal_detection(self):
        """Test causal question detection"""
        detector = QuestionTypeDetector()
        
        qtype, conf = detector.Detect("Why does rain cause flooding?")
        assert qtype == QuestionType.CAUSAL
    
    def test_counterfactual_detection(self):
        """Test counterfactual question detection"""
        detector = QuestionTypeDetector()
        
        qtype, conf = detector.Detect("What if it rains tomorrow?")
        assert qtype == QuestionType.COUNTERFACTUAL
    
    def test_thinking_mode(self):
        """Test thinking mode selection"""
        detector = QuestionTypeDetector()
        
        mode = detector.GetThinkingMode(QuestionType.GREETING)
        assert mode == ThinkingMode.FAST
        
        mode = detector.GetThinkingMode(QuestionType.CAUSAL)
        assert mode == ThinkingMode.DEEP


class TestMetacognition:
    """Tests for MetacognitiveController component"""
    
    def test_assess_question(self):
        """Test question assessment"""
        meta = MetacognitiveController()
        kg = KnowledgeGraph()
        
        # Add some knowledge
        kg.Add("dog", "is_a", "animal")
        
        state = meta.AssessQuestion(
            "What is a dog?",
            QuestionType.DEFINITIONAL,
            kg
        )
        
        assert state.ConfidenceInAnswer > 0
        assert state.ThinkingMode is not None
    
    def test_confidence_statement(self):
        """Test confidence statement generation"""
        meta = MetacognitiveController()
        
        meta.State.ConfidenceInAnswer = 0.9
        stmt = meta.GenerateConfidenceStatement()
        assert "confident" in stmt.lower()
        
        meta.State.ConfidenceInAnswer = 0.3
        stmt = meta.GenerateConfidenceStatement()
        assert "uncertain" in stmt.lower() or "caution" in stmt.lower()


class TestReasoning:
    """Tests for ChainOfThoughtReasoner component"""
    
    def test_think_fast(self):
        """Test fast thinking mode"""
        reasoner = ChainOfThoughtReasoner()
        
        steps = reasoner.Think("Hello!", ThinkingMode.FAST)
        
        assert len(steps) >= 1
        assert len(steps) <= 5
    
    def test_think_deep(self):
        """Test deep thinking mode"""
        reasoner = ChainOfThoughtReasoner()
        
        steps = reasoner.Think("Why does gravity exist?", ThinkingMode.DEEP)
        
        assert len(steps) >= 5
    
    def test_step_verification(self):
        """Test step verification"""
        reasoner = ChainOfThoughtReasoner()
        
        step = ReasoningStep(
            StepNumber=1,
            Thought="Testing",
            Action="Test",
            Result="Success",
            Confidence=0.9
        )
        
        valid, reason = reasoner.VerifyStep(step, [])
        assert valid == True


class TestConstitutional:
    """Tests for Constitutional AI component"""
    
    def test_helpful_check(self):
        """Test helpful principle check"""
        result = Constitution.Evaluate(
            "Here's a detailed explanation about the topic you asked about.",
            "Explain something"
        )
        
        assert result["Principles"]["Helpful"]["Score"] > 0.5
    
    def test_brief_response_flagged(self):
        """Test that brief responses are flagged"""
        result = Constitution.Evaluate(
            "Yes.",
            "Can you explain this?"
        )
        
        assert result["Principles"]["Helpful"]["Score"] < 1.0
    
    def test_overall_pass(self):
        """Test overall pass/fail"""
        result = Constitution.Evaluate(
            "I think this is a reasonable approach, though there are other perspectives to consider.",
            "What do you think?"
        )
        
        assert result["Overall"] == True


class TestChatEngine:
    """Tests for SmartChatEngine component"""
    
    def test_process_greeting(self):
        """Test processing a greeting"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SmartChatEngine(tmpdir)
            
            response = engine.Process("Hello!")
            
            assert response.QuestionType == QuestionType.GREETING
            assert response.ThinkingMode == ThinkingMode.FAST
            assert "hello" in response.Answer.lower() or "hi" in response.Answer.lower()
            
            engine.Close()
    
    def test_learn(self):
        """Test learning from text"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SmartChatEngine(tmpdir)
            
            results = engine.Learn("Dogs are animals. Fire causes smoke.")
            
            assert results["Facts"] >= 1
            
            engine.Close()
    
    def test_stats(self):
        """Test statistics tracking"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SmartChatEngine(tmpdir)
            
            engine.Process("Hello!")
            stats = engine.GetStats()
            
            assert stats["Chat"]["TotalQuestions"] == 1
            
            engine.Close()


class TestProgressTracker:
    """Tests for ProgressTracker component"""
    
    def test_level_calculation(self):
        """Test level calculation"""
        kg = KnowledgeGraph()
        cg = CausalGraph()
        
        # Add 150 facts
        for i in range(150):
            kg.Add(f"concept{i}", "is_a", "thing")
        
        # Add 15 causal relations
        for i in range(15):
            cg.AddCause(f"cause{i}", f"effect{i}")
        
        tracker = ProgressTracker(kg, cg)
        progress = tracker.GetCurrentLevel()
        
        assert progress["CurrentLevel"]["Level"] >= 1
    
    def test_timeline(self):
        """Test timeline generation"""
        kg = KnowledgeGraph()
        cg = CausalGraph()
        
        tracker = ProgressTracker(kg, cg)
        timeline = tracker.GetTimeline()
        
        assert len(timeline) == 6  # 6 milestones
    
    def test_capabilities(self):
        """Test capabilities listing"""
        kg = KnowledgeGraph()
        cg = CausalGraph()
        
        tracker = ProgressTracker(kg, cg)
        capabilities = tracker.GetCapabilities()
        
        assert isinstance(capabilities, list)


def run_all_tests():
    """Run all tests and print results"""
    print("\n" + "=" * 70)
    print("🧪 GroundZero AI - Test Suite")
    print("=" * 70 + "\n")
    
    test_classes = [
        TestKnowledgeGraph,
        TestCausalGraph,
        TestQuestionDetector,
        TestMetacognition,
        TestReasoning,
        TestConstitutional,
        TestChatEngine,
        TestProgressTracker,
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        class_name = test_class.__name__
        print(f"\n📦 {class_name}")
        
        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith('test_')]
        
        for method in methods:
            total_tests += 1
            test_name = method.replace('test_', '').replace('_', ' ').title()
            
            try:
                getattr(instance, method)()
                print(f"   ✓ {test_name}")
                passed_tests += 1
            except AssertionError as e:
                print(f"   ✗ {test_name}: {e}")
                failed_tests.append(f"{class_name}.{method}")
            except Exception as e:
                print(f"   ✗ {test_name}: {type(e).__name__}: {e}")
                failed_tests.append(f"{class_name}.{method}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Summary")
    print("=" * 70)
    print(f"\n   Total:  {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {len(failed_tests)}")
    
    if failed_tests:
        print("\n   Failed tests:")
        for test in failed_tests:
            print(f"      - {test}")
    
    print("\n" + "=" * 70)
    
    return len(failed_tests) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
