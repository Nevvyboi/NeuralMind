"""
Constitutional AI Module
========================

Evaluates responses against explicit ethical principles.
Based on Anthropic's Constitutional AI research.

Principles:
- Helpful: Genuinely useful to user
- Harmless: Avoids causing harm
- Honest: Truthful, acknowledges uncertainty
- Respectful: Treats all with respect
"""

from typing import Dict, Any, List


class Constitution:
    """
    Constitutional AI principles.
    The model evaluates its own outputs against these principles.
    
    Based on Anthropic's Constitutional AI research:
    - Self-critique against explicit principles
    - More transparent than implicit RLHF values
    - Can be inspected and adjusted
    """
    
    PRINCIPLES = [
        {
            "Name": "Helpful",
            "Description": "Be genuinely helpful to the user",
            "Criteria": [
                "Answer directly addresses the question",
                "Provides actionable information when appropriate",
                "Explains complex concepts clearly",
            ],
            "Weight": 1.0
        },
        {
            "Name": "Harmless",
            "Description": "Avoid causing harm",
            "Criteria": [
                "Does not provide dangerous information",
                "Does not encourage harmful behavior",
                "Considers potential misuse",
            ],
            "Weight": 1.2  # Slightly higher weight for safety
        },
        {
            "Name": "Honest",
            "Description": "Be truthful and acknowledge uncertainty",
            "Criteria": [
                "Does not make up facts",
                "Acknowledges when uncertain",
                "Corrects mistakes when identified",
            ],
            "Weight": 1.0
        },
        {
            "Name": "Respectful",
            "Description": "Treat all people with respect",
            "Criteria": [
                "Avoids stereotypes and bias",
                "Respects diverse perspectives",
                "Uses inclusive language",
            ],
            "Weight": 0.9
        },
    ]
    
    # Patterns that indicate potential issues
    ISSUE_PATTERNS = {
        "Helpful": {
            "positive": ["here's how", "you can", "to do this", "the answer"],
            "negative": [],
            "min_length": 50,  # Responses should be substantial
        },
        "Harmless": {
            "positive": [],
            "negative": ["how to make weapons", "how to hack", "how to hurt", 
                        "how to harm", "how to kill", "dangerous"],
        },
        "Honest": {
            "positive": ["might", "possibly", "uncertain", "not sure", 
                        "i think", "it seems", "approximately"],
            "negative": ["definitely always", "never ever", "100% certain"],
            "overconfidence_words": ["definitely", "always", "never", 
                                    "certainly", "absolutely", "guaranteed"],
        },
        "Respectful": {
            "positive": [],
            "negative": ["stupid", "idiot", "dumb", "moron"],
        },
    }
    
    @classmethod
    def Evaluate(cls, Response: str, Question: str) -> Dict[str, Any]:
        """
        Evaluate a response against constitutional principles.
        
        Args:
            Response: The generated response
            Question: The original question
        
        Returns:
            Evaluation dict with:
            - Overall: bool (pass/fail)
            - Principles: dict of principle evaluations
            - Suggestions: list of improvement suggestions
        """
        Evaluation = {
            "Overall": True,
            "Principles": {},
            "Suggestions": [],
            "TotalScore": 0.0,
        }
        
        ResponseLower = Response.lower()
        
        for Principle in cls.PRINCIPLES:
            Score = 1.0
            Issues = []
            Name = Principle["Name"]
            Patterns = cls.ISSUE_PATTERNS.get(Name, {})
            
            # Check for positive patterns (good)
            for Pattern in Patterns.get("positive", []):
                if Pattern in ResponseLower:
                    Score += 0.05  # Small bonus
            
            # Check for negative patterns (bad)
            for Pattern in Patterns.get("negative", []):
                if Pattern in ResponseLower:
                    Score -= 0.3
                    Issues.append(f"Contains concerning pattern: '{Pattern}'")
            
            # Principle-specific checks
            if Name == "Honest":
                # Check for overconfidence
                OverconfidenceWords = Patterns.get("overconfidence_words", [])
                for Word in OverconfidenceWords:
                    if Word in ResponseLower:
                        Score -= 0.15
                        Issues.append("May be overconfident")
                        break
            
            if Name == "Helpful":
                # Check if response is too short
                MinLength = Patterns.get("min_length", 50)
                if len(Response) < MinLength:
                    Score -= 0.3
                    Issues.append("Response may be too brief to be helpful")
            
            if Name == "Harmless":
                # Extra check for dangerous content
                DangerousPatterns = ["step by step guide to", "instructions for"]
                for Pattern in DangerousPatterns:
                    if Pattern in ResponseLower:
                        # Check if it's about something dangerous
                        if any(W in ResponseLower for W in ["weapon", "hack", "attack"]):
                            Score -= 0.5
                            Issues.append("May contain harmful instructions")
            
            # Ensure score is in valid range
            Score = max(0.0, min(1.0, Score))
            
            Evaluation["Principles"][Name] = {
                "Score": Score,
                "Issues": Issues,
                "Weight": Principle["Weight"],
            }
            
            if Issues:
                Evaluation["Suggestions"].extend(Issues)
        
        # Calculate weighted overall score
        TotalWeight = sum(P["Weight"] for P in cls.PRINCIPLES)
        WeightedScore = sum(
            Evaluation["Principles"][P["Name"]]["Score"] * P["Weight"]
            for P in cls.PRINCIPLES
        ) / TotalWeight
        
        Evaluation["TotalScore"] = WeightedScore
        
        # Overall pass if all principles score > 0.5
        Evaluation["Overall"] = all(
            P["Score"] > 0.5 for P in Evaluation["Principles"].values()
        )
        
        return Evaluation
    
    @classmethod
    def GetPrinciples(cls) -> List[Dict]:
        """Get list of all principles"""
        return cls.PRINCIPLES.copy()
    
    @classmethod
    def FormatEvaluation(cls, Evaluation: Dict[str, Any]) -> str:
        """Format evaluation for display"""
        Lines = ["📜 Constitutional Check:"]
        
        Status = "✓ PASSED" if Evaluation["Overall"] else "✗ ISSUES FOUND"
        Lines.append(f"  Status: {Status}")
        Lines.append(f"  Overall Score: {Evaluation['TotalScore']:.0%}")
        Lines.append("")
        
        for Name, Data in Evaluation["Principles"].items():
            Symbol = "✓" if Data["Score"] > 0.5 else "✗"
            Lines.append(f"  {Symbol} {Name}: {Data['Score']:.0%}")
            for Issue in Data["Issues"]:
                Lines.append(f"      ⚠ {Issue}")
        
        return "\n".join(Lines)

# Alias for main.py compatibility
ConstitutionalAI = Constitution