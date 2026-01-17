#!/usr/bin/env python3
"""
GroundZero AI - Test Script
===========================

Tests all major components to ensure they work.
"""

import sys
import os

# Setup path for imports
PROJECT_ROOT = os.path.dirname(__file__)
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, SRC_PATH)

def test_utils():
    print("Testing Utils...")
    from utils import Config, get_config, logger, timestamp, generate_id
    
    config = get_config()
    assert config.model.name == "GroundZero-AI"
    
    ts = timestamp()
    assert ts is not None
    
    uid = generate_id("test_")
    assert uid.startswith("test_")
    
    print("  ✓ Utils OK")

def test_knowledge():
    print("Testing Knowledge Graph...")
    from knowledge import KnowledgeGraph, KnowledgeExtractor
    
    kg = KnowledgeGraph()
    
    # Add knowledge
    node = kg.add_node("Python", "A programming language", node_type="concept")
    assert node.name == "Python"
    
    # Add relationship
    kg.add_knowledge("Python", "is_a", "Programming Language")
    
    # Search
    results = kg.search_nodes("Python")
    assert len(results) > 0
    
    # Query
    triples = kg.query_knowledge(subject="Python")
    assert len(triples) > 0
    
    print(f"  ✓ Knowledge Graph OK ({kg.get_stats()['total_nodes']} nodes)")

def test_memory():
    print("Testing Memory System...")
    from memory import MemorySystem
    
    mem = MemorySystem()
    mem.set_user("test_user")
    mem.start_conversation()
    mem.add_turn("Hello", "Hi there!")
    
    user = mem.get_current_user()
    assert user.id == "test_user"
    
    context = mem.get_full_context("test")
    assert context is not None
    
    print("  ✓ Memory System OK")

def test_reasoning():
    print("Testing Reasoning Engine...")
    from reasoning import ReasoningEngine, ReasoningTrace
    
    engine = ReasoningEngine()
    trace = engine.reason("What is 2+2?")
    
    assert trace.query == "What is 2+2?"
    assert len(trace.steps) > 0
    
    display = trace.format_for_display()
    assert "Reasoning Trace" in display
    
    print(f"  ✓ Reasoning OK ({len(trace.steps)} steps)")

def test_learning():
    print("Testing Continuous Learning...")
    from continuous_learning import ContinuousLearningSystem
    
    learning = ContinuousLearningSystem()
    learning.observe("test input", "test response", rating=4)
    
    stats = learning.get_stats()
    assert stats["orchestrator"]["signals_received"] > 0
    
    print("  ✓ Learning System OK")

def test_groundzero():
    print("Testing GroundZero AI...")
    from groundzero import GroundZeroAI
    
    ai = GroundZeroAI(use_mock=True)
    
    # Test chat
    response, reasoning = ai.chat("Hello, who are you?", return_reasoning=True)
    assert response is not None
    assert reasoning is not None
    
    # Test teach
    result = ai.teach("Test Topic", "This is test content")
    assert result["success"]
    
    # Test knowledge query
    result = ai.ask_knowledge("Test")
    assert "results" in result
    
    # Test stats
    stats = ai.get_stats()
    assert "model" in stats
    assert "knowledge" in stats
    assert "tools" in stats
    
    # Test code execution
    code_result = ai.run_code("print(2 + 2)")
    assert code_result["success"]
    assert "4" in code_result["output"]
    
    print(f"  ✓ GroundZero AI OK")
    print(f"    Model: {stats['model']['name']}")
    print(f"    Knowledge: {stats['knowledge']['total_nodes']} nodes")
    print(f"    Tools: Code execution ✓")
    
    return ai

def main():
    print("=" * 50)
    print("GroundZero AI - System Test")
    print("=" * 50)
    print()
    
    try:
        test_utils()
        test_knowledge()
        test_memory()
        test_reasoning()
        test_learning()
        ai = test_groundzero()
        
        print()
        print("=" * 50)
        print("✓ ALL TESTS PASSED!")
        print("=" * 50)
        print()
        print("Quick start commands:")
        print("  python run.py              # Interactive chat")
        print("  python run.py --dashboard  # Web dashboard")
        print()
        
        return True
        
    except Exception as e:
        print()
        print("=" * 50)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 50)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
