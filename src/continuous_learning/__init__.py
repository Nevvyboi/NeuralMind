"""
GroundZero AI - Continuous Learning System
==========================================

Learn from interactions, feedback, corrections, and research.
"""

import json
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict

try:
    from ..utils import get_data_path, ensure_dir, load_json, save_json, logger, timestamp, generate_id
except ImportError:
    from utils import get_data_path, ensure_dir, load_json, save_json, logger, timestamp, generate_id


@dataclass
class LearningSignal:
    """A signal that triggers learning."""
    id: str = field(default_factory=lambda: generate_id("sig_"))
    signal_type: str = "interaction"
    content: Dict = field(default_factory=dict)
    importance: float = 0.5
    processed: bool = False
    timestamp: str = field(default_factory=timestamp)
    
    def to_dict(self) -> Dict:
        return asdict(self)


class LearningOrchestrator:
    """Orchestrate all learning activities."""
    
    def __init__(self):
        self.data_path = get_data_path("training")
        ensure_dir(self.data_path)
        
        self.signal_queue: List[LearningSignal] = []
        self.training_examples: List[Dict] = []
        self.stats = {"signals_received": 0, "examples_generated": 0}
        self._load()
    
    def _load(self):
        path = self.data_path / "learning_state.json"
        if path.exists():
            data = load_json(path)
            self.training_examples = data.get("examples", [])
            self.stats = data.get("stats", self.stats)
    
    def _save(self):
        save_json(self.data_path / "learning_state.json", {
            "examples": self.training_examples,
            "stats": self.stats,
            "updated_at": timestamp(),
        })
    
    def add_signal(self, signal_type: str, content: Dict, importance: float = 0.5):
        signal = LearningSignal(signal_type=signal_type, content=content, importance=importance)
        self.signal_queue.append(signal)
        self.stats["signals_received"] += 1
        self._process_signal(signal)
        self._save()
        return signal
    
    def _process_signal(self, signal: LearningSignal):
        content = signal.content
        
        if signal.signal_type == "interaction" and content.get("rating", 3) >= 4:
            self.training_examples.append({
                "input": content.get("user_input", ""),
                "output": content.get("response", ""),
                "type": "interaction",
                "importance": signal.importance,
            })
            self.stats["examples_generated"] += 1
        
        elif signal.signal_type == "correction":
            self.training_examples.append({
                "input": content.get("prompt", ""),
                "output": content.get("correction", ""),
                "type": "correction",
                "importance": 0.95,
            })
            self.stats["examples_generated"] += 1
        
        signal.processed = True
    
    def get_training_data(self) -> List[Dict]:
        return self.training_examples
    
    def clear_training_data(self):
        self.training_examples = []
        self._save()
    
    def get_stats(self) -> Dict:
        return {"training_examples": len(self.training_examples), **self.stats}


class ContinuousLearningSystem:
    """Complete continuous learning system."""
    
    def __init__(self, model_trainer: Callable = None, web_search=None):
        self.orchestrator = LearningOrchestrator()
        self.model_trainer = model_trainer
        self.web_search = web_search
        self.recent_interactions: List[Dict] = []
    
    def observe(self, user_input: str, response: str, rating: int = 3):
        self.orchestrator.add_signal("interaction", {
            "user_input": user_input,
            "response": response,
            "rating": rating,
        }, importance=0.3 + (rating - 3) * 0.2)
        
        self.recent_interactions.append({
            "user_input": user_input,
            "response": response,
            "rating": rating,
            "timestamp": timestamp(),
        })
        if len(self.recent_interactions) > 100:
            self.recent_interactions = self.recent_interactions[-100:]
    
    def feedback(self, prompt: str, response: str, rating: int, correction: str = None):
        if correction:
            self.orchestrator.add_signal("correction", {
                "prompt": prompt,
                "original": response,
                "correction": correction,
            }, importance=0.95)
        else:
            self.orchestrator.add_signal("feedback", {
                "prompt": prompt,
                "response": response,
                "rating": rating,
            }, importance=0.5 + abs(rating - 3) * 0.2)
    
    def correct(self, prompt: str, original: str, correction: str):
        self.orchestrator.add_signal("correction", {
            "prompt": prompt,
            "original": original,
            "correction": correction,
        }, importance=0.95)
    
    def learn_topic(self, topic: str) -> Dict:
        if not self.web_search:
            return {"success": False, "reason": "no_web_search"}
        
        research = self.web_search.research(topic, depth="deep")
        
        if research.get("overview"):
            self.orchestrator.add_signal("research", {
                "topic": topic,
                "content": research["overview"],
            }, importance=0.7)
        
        return {"success": True, "topic": topic}
    
    def evolve(self) -> Dict:
        data = self.orchestrator.get_training_data()
        if not data:
            return {"success": False, "reason": "no_data"}
        
        if self.model_trainer:
            result = self.model_trainer(data)
            if result.get("success"):
                self.orchestrator.clear_training_data()
            return result
        
        return {"success": True, "simulated": True, "examples": len(data)}
    
    def get_stats(self) -> Dict:
        return {
            "orchestrator": self.orchestrator.get_stats(),
            "recent_interactions": len(self.recent_interactions),
        }


__all__ = ['LearningSignal', 'LearningOrchestrator', 'ContinuousLearningSystem']
