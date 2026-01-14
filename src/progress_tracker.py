"""
Progress Tracker Module
=======================

Tracks progress toward human-like understanding.

Milestones:
1. Basic Pattern Recognition (100 facts)
2. Knowledge Accumulation (1,000 facts)
3. Causal Understanding (5,000 facts)
4. Reasoning Chains (20,000 facts)
5. Deep Understanding (100,000 facts)
6. Human-Like Reasoning (500,000 facts)
"""

from datetime import datetime
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .knowledge_graph import KnowledgeGraph
    from .causal_graph import CausalGraph


class ProgressTracker:
    """Track progress toward human-like understanding"""
    
    # Milestones based on AI research
    MILESTONES = [
        {
            "Level": 1,
            "Name": "Basic Pattern Recognition",
            "Description": "Can recognize simple patterns in text",
            "FactsRequired": 100,
            "CausalRequired": 10,
            "Capabilities": ["Basic text completion", "Simple Q&A"],
            "Timeline": "1-2 days",
        },
        {
            "Level": 2,
            "Name": "Knowledge Accumulation",
            "Description": "Has built significant knowledge base",
            "FactsRequired": 1000,
            "CausalRequired": 100,
            "Capabilities": ["Factual answers", "Simple reasoning", "Transitive inference"],
            "Timeline": "1-2 weeks",
        },
        {
            "Level": 3,
            "Name": "Causal Understanding",
            "Description": "Understands cause-effect relationships",
            "FactsRequired": 5000,
            "CausalRequired": 500,
            "Capabilities": ["Causal explanations", "Basic counterfactuals", "Why questions"],
            "Timeline": "1-2 months",
        },
        {
            "Level": 4,
            "Name": "Reasoning Chains",
            "Description": "Can perform multi-step reasoning",
            "FactsRequired": 20000,
            "CausalRequired": 2000,
            "Capabilities": ["Complex reasoning", "Logical inference", "Cross-domain connections"],
            "Timeline": "3-6 months",
        },
        {
            "Level": 5,
            "Name": "Deep Understanding",
            "Description": "Demonstrates nuanced understanding",
            "FactsRequired": 100000,
            "CausalRequired": 10000,
            "Capabilities": ["Nuanced responses", "Abstract reasoning", "Novel insights"],
            "Timeline": "1-2 years",
        },
        {
            "Level": 6,
            "Name": "Human-Like Reasoning",
            "Description": "Approaches human-level reasoning",
            "FactsRequired": 500000,
            "CausalRequired": 50000,
            "Capabilities": ["Creative problem-solving", "Meta-reasoning", "Generalization"],
            "Timeline": "3-5 years",
        },
    ]
    
    def __init__(self, Knowledge: 'KnowledgeGraph' = None, Causal: 'CausalGraph' = None):
        """
        Initialize tracker.
        
        Args:
            Knowledge: Knowledge graph to track (optional)
            Causal: Causal graph to track (optional)
        """
        self.Knowledge = Knowledge
        self.Causal = Causal
    
    def GetCurrentLevel(self) -> Dict:
        """
        Get current progress level.
        
        Returns:
            Dict with current level info and progress
        """
        Facts = self.Knowledge.Size() if self.Knowledge else 0
        CausalCount = self.Causal.Stats["TotalRelations"] if self.Causal else 0
        
        CurrentLevel = 0
        for Milestone in self.MILESTONES:
            if (Facts >= Milestone["FactsRequired"] and 
                CausalCount >= Milestone["CausalRequired"]):
                CurrentLevel = Milestone["Level"]
            else:
                break
        
        if CurrentLevel == 0:
            Current = {
                "Level": 0, 
                "Name": "Starting Out", 
                "Description": "Just getting started",
                "Capabilities": ["Learning..."],
                "Timeline": "Now",
            }
        else:
            Current = self.MILESTONES[CurrentLevel - 1]
        
        # Calculate progress to next level
        if CurrentLevel < len(self.MILESTONES):
            NextMilestone = self.MILESTONES[CurrentLevel]
            FactsProgress = min(Facts / NextMilestone["FactsRequired"], 1.0)
            CausalProgress = min(CausalCount / NextMilestone["CausalRequired"], 1.0)
            ProgressToNext = (FactsProgress + CausalProgress) / 2
        else:
            NextMilestone = None
            ProgressToNext = 1.0
        
        return {
            "CurrentLevel": Current,
            "NextMilestone": NextMilestone,
            "Progress": ProgressToNext,
            "Facts": Facts,
            "CausalRelations": CausalCount,
        }
    
    def GetTimeline(self) -> List[Dict]:
        """
        Get timeline of milestones with status.
        
        Returns:
            List of milestone dicts with status
        """
        Facts = self.Knowledge.Size() if self.Knowledge else 0
        CausalCount = self.Causal.Stats["TotalRelations"] if self.Causal else 0
        
        Timeline = []
        for M in self.MILESTONES:
            if (Facts >= M["FactsRequired"] and CausalCount >= M["CausalRequired"]):
                Status = "completed"
            elif Timeline and Timeline[-1]["Status"] == "completed":
                Status = "current"
            else:
                Status = "pending"
            
            # Estimate time to reach
            if Status == "pending":
                FactsNeeded = max(0, M["FactsRequired"] - Facts)
                # Rough estimate: 100 facts per day of training
                DaysNeeded = FactsNeeded / 100 if FactsNeeded > 0 else 0
                EstimatedDate = datetime.now().timestamp() + DaysNeeded * 86400
            else:
                EstimatedDate = None
            
            Timeline.append({
                **M,
                "Status": Status,
                "EstimatedDate": EstimatedDate,
            })
        
        return Timeline
    
    def FormatStatus(self) -> str:
        """Format current status for display"""
        Progress = self.GetCurrentLevel()
        Current = Progress["CurrentLevel"]
        
        Lines = [
            "=" * 60,
            "📊 GroundZero AI - Progress Status",
            "=" * 60,
            "",
            f"  Current Level: {Current.get('Level', 0)} - {Current.get('Name', 'Starting')}",
            f"  Description: {Current.get('Description', '')}",
            f"  Progress to Next: {Progress['Progress']:.0%}",
            "",
            f"  Knowledge Facts: {Progress['Facts']:,}",
            f"  Causal Relations: {Progress['CausalRelations']:,}",
            "",
        ]
        
        if Progress["NextMilestone"]:
            Next = Progress["NextMilestone"]
            Lines.extend([
                f"  Next Milestone: Level {Next['Level']} - {Next['Name']}",
                f"  Requirements: {Next['FactsRequired']:,} facts, {Next['CausalRequired']:,} causal",
                f"  Est. Timeline: {Next.get('Timeline', 'Unknown')}",
            ])
        
        Lines.append("=" * 60)
        
        return "\n".join(Lines)
    
    def GetCapabilities(self) -> List[str]:
        """Get list of current capabilities"""
        Progress = self.GetCurrentLevel()
        return Progress["CurrentLevel"].get("Capabilities", [])