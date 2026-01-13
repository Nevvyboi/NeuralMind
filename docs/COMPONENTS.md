# 🔧 Technical Components Guide

## Deep-Dive Into Each Component of GroundZero AI

This document provides a comprehensive technical explanation of each component, including architecture, algorithms, and implementation details.

---

## 📑 Table of Contents

1. [Question Type Detector](#1-question-type-detector)
2. [Knowledge Graph](#2-knowledge-graph)
3. [Causal Graph](#3-causal-graph)
4. [Metacognitive Controller](#4-metacognitive-controller)
5. [Chain-of-Thought Reasoner](#5-chain-of-thought-reasoner)
6. [Constitutional AI](#6-constitutional-ai)
7. [World Model (JEPA)](#7-world-model-jepa)
8. [Smart Chat Engine](#8-smart-chat-engine)
9. [Progress Tracker](#9-progress-tracker)
10. [Data Structures](#10-data-structures)

---

## 1. Question Type Detector

### Purpose
Automatically classify user questions to route them to the appropriate reasoning system.

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    QuestionTypeDetector                       │
├──────────────────────────────────────────────────────────────┤
│  PATTERNS: Dict[QuestionType, List[regex]]                   │
│  CompiledPatterns: Dict[QuestionType, List[Pattern]]         │
├──────────────────────────────────────────────────────────────┤
│  + Detect(question: str) → (QuestionType, confidence)        │
│  + GetThinkingMode(qtype: QuestionType) → ThinkingMode       │
└──────────────────────────────────────────────────────────────┘
```

### Pattern Categories

```python
PATTERNS = {
    QuestionType.GREETING: [
        r'^(hi|hello|hey|greetings)[\s!?.]*$',
        r'^how\s+are\s+you',
    ],
    QuestionType.COUNTERFACTUAL: [
        r'what\s+(would|could|might)\s+(happen|occur)',
        r'what\s+if\b',
        r'imagine\s+if',
        r'hypothetically',
    ],
    QuestionType.CAUSAL: [
        r'why\s+(does|do|did|is|are)',
        r'what\s+(causes?|leads?\s+to)',
        r'explain\s+why',
    ],
    # ... more patterns
}
```

### Detection Algorithm

```python
def Detect(self, Question: str) -> Tuple[QuestionType, float]:
    Scores = {}
    
    for QType, Patterns in self.CompiledPatterns.items():
        MatchCount = 0
        for Pattern in Patterns:
            if Pattern.search(Question):
                MatchCount += 1
        
        if MatchCount > 0:
            # Score = matches / total patterns + base score
            Scores[QType] = min(MatchCount / len(Patterns) + 0.3, 1.0)
    
    if not Scores:
        return QuestionType.UNKNOWN, 0.3
    
    BestType = max(Scores, key=Scores.get)
    return BestType, Scores[BestType]
```

### Thinking Mode Selection

```python
def GetThinkingMode(self, QType: QuestionType) -> ThinkingMode:
    # Complex questions need deep thinking
    if QType in [QuestionType.CAUSAL, QuestionType.COUNTERFACTUAL]:
        return ThinkingMode.DEEP
    
    # Medium complexity
    if QType in [QuestionType.OPINION, QuestionType.CLARIFICATION]:
        return ThinkingMode.MEDIUM
    
    # Simple questions
    return ThinkingMode.FAST
```

---

## 2. Knowledge Graph

### Purpose
Store facts as explicit (subject, predicate, object) triples with full queryability and inference capabilities.

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      KnowledgeGraph                           │
├──────────────────────────────────────────────────────────────┤
│  DbPath: str                   # SQLite database path         │
│  BySubject: Dict[str, List[KnowledgeTriple]]                 │
│  ByPredicate: Dict[str, List[KnowledgeTriple]]               │
│  ByObject: Dict[str, List[KnowledgeTriple]]                  │
│  AllTriples: Set[Tuple[str, str, str]]                       │
├──────────────────────────────────────────────────────────────┤
│  + Add(subj, pred, obj, confidence, source) → bool           │
│  + Query(subj?, pred?, obj?) → List[KnowledgeTriple]         │
│  + GetRelated(entity, max_depth) → List[KnowledgeTriple]     │
│  + InferTransitive(predicate) → List[KnowledgeTriple]        │
│  + ExtractFromText(text) → List[KnowledgeTriple]             │
└──────────────────────────────────────────────────────────────┘
```

### Data Structure: KnowledgeTriple

```python
@dataclass
class KnowledgeTriple:
    Subject: str      # e.g., "dog"
    Predicate: str    # e.g., "is_a"
    Object: str       # e.g., "animal"
    Confidence: float # 0.0 to 1.0
    Source: str       # "learned", "inferred", "extracted"
    Timestamp: float  # Unix timestamp
```

### Database Schema

```sql
CREATE TABLE Triples (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    Subject TEXT NOT NULL,
    Predicate TEXT NOT NULL,
    Object TEXT NOT NULL,
    Confidence REAL DEFAULT 1.0,
    Source TEXT DEFAULT 'learned',
    Timestamp REAL,
    UNIQUE(Subject, Predicate, Object)
);

CREATE INDEX IdxSubject ON Triples(Subject);
CREATE INDEX IdxObject ON Triples(Object);
```

### Transitive Inference Algorithm

```python
def InferTransitive(self, Predicate: str) -> List[KnowledgeTriple]:
    """
    For transitive relations like "is_a":
    If A is_a B and B is_a C, then A is_a C
    
    Uses BFS to find all transitive closures.
    """
    Existing = self.ByPredicate.get(Predicate, [])
    
    # Build graph: subject → set of objects
    Graph = defaultdict(set)
    for Triple in Existing:
        Graph[Triple.Subject].add(Triple.Object)
    
    Inferred = []
    for Start in Graph:
        Visited = set()
        Queue = [Start]
        
        while Queue:
            Current = Queue.pop(0)
            for Next in Graph.get(Current, []):
                if Next not in Visited:
                    Visited.add(Next)
                    Queue.append(Next)
                    
                    # If this is a new inference
                    if (Start, Predicate, Next) not in self.AllTriples:
                        NewTriple = KnowledgeTriple(
                            Subject=Start,
                            Predicate=Predicate,
                            Object=Next,
                            Confidence=0.9,
                            Source="inferred_transitive"
                        )
                        Inferred.append(NewTriple)
    
    return Inferred
```

### Text Extraction Patterns

```python
EXTRACTION_PATTERNS = [
    (r'(\w+)\s+is\s+(?:a|an)\s+(\w+)', "is_a"),
    (r'(\w+)\s+has\s+(?:a|an)?\s*(\w+)', "has"),
    (r'(\w+)\s+causes?\s+(\w+)', "causes"),
    (r'(\w+)\s+is\s+(?:located\s+)?in\s+(\w+)', "located_in"),
    (r'(\w+)\s+was\s+created\s+by\s+(\w+)', "created_by"),
    (r'(\w+)\s+is\s+part\s+of\s+(\w+)', "part_of"),
]
```

---

## 3. Causal Graph

### Purpose
Store and reason about cause-effect relationships, enabling counterfactual and interventional reasoning.

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       CausalGraph                             │
├──────────────────────────────────────────────────────────────┤
│  Relations: Dict[str, List[CausalRelation]]     # cause → effects │
│  ReverseRelations: Dict[str, List[CausalRelation]] # effect → causes │
├──────────────────────────────────────────────────────────────┤
│  + AddCause(cause, effect, strength, mechanism) → void       │
│  + GetEffects(cause) → List[CausalRelation]                  │
│  + GetCauses(effect) → List[CausalRelation]                  │
│  + CausalChain(start, end, max_depth) → List[List[Relation]] │
│  + Counterfactual(event, intervention) → Dict[str, float]    │
│  + Explain(effect) → str                                     │
└──────────────────────────────────────────────────────────────┘
```

### Data Structure: CausalRelation

```python
@dataclass
class CausalRelation:
    Cause: str           # "rain"
    Effect: str          # "wet_ground"
    Strength: float      # 0.0 to 1.0 (probability of effect given cause)
    Mechanism: str       # "water falls from clouds"
    Conditions: List[str]  # ["outdoors", "no cover"]
    Evidence: List[str]    # Supporting observations
```

### Causal Chain Discovery (DFS)

```python
def CausalChain(self, Start: str, End: str, MaxDepth: int = 5) -> List[List[CausalRelation]]:
    """
    Find all causal paths from Start to End.
    Uses DFS with cycle detection.
    """
    Chains = []
    
    def Search(Current: str, Path: List[CausalRelation], Depth: int):
        if Depth > MaxDepth:
            return
        if Current == End and Path:
            Chains.append(Path.copy())
            return
        
        for Relation in self.Relations.get(Current, []):
            if Relation not in Path:  # Avoid cycles
                Path.append(Relation)
                Search(Relation.Effect, Path, Depth + 1)
                Path.pop()
    
    Search(Start, [], 0)
    return Chains
```

### Counterfactual Reasoning Algorithm

```python
def Counterfactual(self, Event: str, Intervention: bool = False) -> Dict[str, float]:
    """
    Compute: "What if Event happened/didn't happen?"
    
    Propagates probability changes through causal graph.
    
    Args:
        Event: The hypothetical event
        Intervention: True = "make it happen", False = "didn't happen"
    
    Returns:
        Dict mapping effects to probability changes
    """
    Effects = {}
    
    def PropagateEffect(Current: str, CurrentStrength: float, Visited: Set[str]):
        if Current in Visited or CurrentStrength < 0.1:
            return
        Visited.add(Current)
        
        for Relation in self.Relations.get(Current, []):
            # Multiply strengths along the chain
            EffectChange = Relation.Strength * CurrentStrength
            
            if Intervention:
                Effects[Relation.Effect] = EffectChange  # Positive change
            else:
                Effects[Relation.Effect] = -EffectChange  # Negative change
            
            # Recursively propagate
            PropagateEffect(Relation.Effect, EffectChange, Visited)
    
    PropagateEffect(Event, 1.0, set())
    return Effects
```

### Example Counterfactual Computation

```
Causal Graph:
  rain --0.9--> wet_ground --0.8--> slippery --0.6--> accident

Query: Counterfactual("rain", Intervention=True)

Computation:
  rain → wet_ground: 1.0 × 0.9 = 0.9
  wet_ground → slippery: 0.9 × 0.8 = 0.72
  slippery → accident: 0.72 × 0.6 = 0.432

Result:
  {"wet_ground": 0.9, "slippery": 0.72, "accident": 0.432}
```

---

## 4. Metacognitive Controller

### Purpose
Enable the system to think about its own thinking - knowing what it knows and doesn't know.

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  MetacognitiveController                      │
├──────────────────────────────────────────────────────────────┤
│  State: MetacognitiveState                                    │
│  ReasoningHistory: List[Dict]                                 │
│  PerformanceMetrics: Dict                                     │
├──────────────────────────────────────────────────────────────┤
│  + AssessQuestion(question, qtype, knowledge) → State        │
│  + MonitorReasoning(step, confidence) → void                 │
│  + ShouldUseSystem2(qtype, complexity) → bool                │
│  + GenerateConfidenceStatement() → str                       │
│  + GetKnowledgeGapStatement() → Optional[str]                │
└──────────────────────────────────────────────────────────────┘
```

### Data Structure: MetacognitiveState

```python
@dataclass
class MetacognitiveState:
    ConfidenceInAnswer: float = 0.5
    UncertaintyAreas: List[str] = field(default_factory=list)
    ReasoningStepsTaken: int = 0
    SelfCorrections: int = 0
    KnowledgeGaps: List[str] = field(default_factory=list)
    ThinkingMode: ThinkingMode = ThinkingMode.MEDIUM
```

### Question Assessment Algorithm

```python
def AssessQuestion(self, Question: str, QType: QuestionType, 
                   Knowledge: KnowledgeGraph) -> MetacognitiveState:
    State = MetacognitiveState()
    
    # Extract key concepts from question
    Words = set(Question.lower().split())
    
    # Check knowledge coverage
    KnownConcepts = 0
    UnknownConcepts = []
    
    for Word in Words:
        if len(Word) > 3:  # Skip short words
            Facts = Knowledge.Query(Subject=Word)
            if Facts:
                KnownConcepts += 1
            else:
                UnknownConcepts.append(Word)
    
    # Calculate confidence based on coverage
    TotalConcepts = len([W for W in Words if len(W) > 3])
    if TotalConcepts > 0:
        KnowledgeCoverage = KnownConcepts / TotalConcepts
    else:
        KnowledgeCoverage = 0.3
    
    State.ConfidenceInAnswer = min(0.3 + KnowledgeCoverage * 0.5, 0.9)
    State.KnowledgeGaps = UnknownConcepts[:5]
    
    # Adjust for question type uncertainty
    if QType == QuestionType.COUNTERFACTUAL:
        State.UncertaintyAreas.append("Counterfactual outcomes are inherently uncertain")
        State.ConfidenceInAnswer *= 0.8
    
    return State
```

### System 2 Decision Logic

```python
def ShouldUseSystem2(self, QType: QuestionType, Complexity: float) -> bool:
    # Always use System 2 for complex question types
    if QType in [QuestionType.CAUSAL, QuestionType.COUNTERFACTUAL]:
        return True
    
    # Use System 2 if complexity is high
    if Complexity > 0.7:
        return True
    
    # Use System 2 if confidence is low
    if self.State.ConfidenceInAnswer < 0.5:
        return True
    
    return False  # Use System 1 (fast)
```

---

## 5. Chain-of-Thought Reasoner

### Purpose
Generate step-by-step reasoning chains with verification and self-correction.

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  ChainOfThoughtReasoner                       │
├──────────────────────────────────────────────────────────────┤
│  Knowledge: KnowledgeGraph                                    │
│  Causal: CausalGraph                                          │
│  MaxSteps: int = 20                                           │
├──────────────────────────────────────────────────────────────┤
│  + Think(question, mode) → List[ReasoningStep]               │
│  + VerifyStep(step, context) → (bool, str)                   │
│  + FormatChain(steps) → str                                  │
└──────────────────────────────────────────────────────────────┘
```

### Data Structure: ReasoningStep

```python
@dataclass
class ReasoningStep:
    StepNumber: int
    Thought: str      # What the system is thinking
    Action: str       # What action it's taking
    Result: str       # What it found/concluded
    Confidence: float # Confidence in this step
    IsVerified: bool  # Whether step passed verification
```

### Thinking Algorithm

```python
def Think(self, Question: str, Mode: ThinkingMode) -> List[ReasoningStep]:
    Steps = []
    
    # Depth based on mode
    MaxSteps = {
        ThinkingMode.FAST: 3,
        ThinkingMode.MEDIUM: 7,
        ThinkingMode.DEEP: 15,
    }[Mode]
    
    # Step 1: Analyze question
    Steps.append(ReasoningStep(
        StepNumber=1,
        Thought=f"Analyzing: '{Question}'",
        Action="Identify key concepts",
        Result="Concepts extracted",
        Confidence=0.9,
        IsVerified=True
    ))
    
    # Step 2: Check knowledge base
    Words = Question.lower().split()
    RelevantFacts = []
    for Word in Words:
        if len(Word) > 3:
            Facts = self.Knowledge.Query(Subject=Word)
            RelevantFacts.extend(Facts[:2])
    
    if RelevantFacts:
        Steps.append(ReasoningStep(
            StepNumber=2,
            Thought="Checking knowledge base",
            Action="Query knowledge graph",
            Result=f"Found {len(RelevantFacts)} facts",
            Confidence=0.85,
            IsVerified=True
        ))
    
    # Step 3: Check causal relations (for MEDIUM/DEEP)
    if Mode != ThinkingMode.FAST:
        # ... check causal graph
        pass
    
    # Step 4: Verify reasoning (for DEEP)
    if Mode == ThinkingMode.DEEP:
        Steps.append(ReasoningStep(
            StepNumber=len(Steps) + 1,
            Thought="Verifying reasoning chain",
            Action="Check logical validity",
            Result="All steps verified",
            Confidence=0.85,
            IsVerified=True
        ))
    
    return Steps
```

---

## 6. Constitutional AI

### Purpose
Ensure responses align with ethical principles through self-evaluation.

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      Constitution                             │
├──────────────────────────────────────────────────────────────┤
│  PRINCIPLES: List[Dict]  # Principle definitions              │
├──────────────────────────────────────────────────────────────┤
│  + Evaluate(response, question) → Dict[str, Any]             │
└──────────────────────────────────────────────────────────────┘
```

### Principle Definitions

```python
PRINCIPLES = [
    {
        "Name": "Helpful",
        "Description": "Be genuinely helpful to the user",
        "Criteria": [
            "Answer directly addresses the question",
            "Provides actionable information",
            "Explains complex concepts clearly",
        ]
    },
    {
        "Name": "Harmless",
        "Description": "Avoid causing harm",
        "Criteria": [
            "Does not provide dangerous information",
            "Does not encourage harmful behavior",
            "Considers potential misuse",
        ]
    },
    {
        "Name": "Honest",
        "Description": "Be truthful and acknowledge uncertainty",
        "Criteria": [
            "Does not make up facts",
            "Acknowledges when uncertain",
            "Corrects mistakes when identified",
        ]
    },
    {
        "Name": "Respectful",
        "Description": "Treat all with respect",
        "Criteria": [
            "Avoids stereotypes",
            "Respects diverse perspectives",
            "Uses inclusive language",
        ]
    },
]
```

### Evaluation Algorithm

```python
def Evaluate(cls, Response: str, Question: str) -> Dict[str, Any]:
    Evaluation = {
        "Overall": True,
        "Principles": {},
        "Suggestions": [],
    }
    
    ResponseLower = Response.lower()
    
    for Principle in cls.PRINCIPLES:
        Score = 1.0
        Issues = []
        
        if Principle["Name"] == "Honest":
            # Check for overconfidence
            DefiniteWords = ["definitely", "always", "never"]
            if any(W in ResponseLower for W in DefiniteWords):
                Score -= 0.2
                Issues.append("May be overconfident")
        
        if Principle["Name"] == "Helpful":
            # Check if response is too short
            if len(Response) < 50:
                Score -= 0.3
                Issues.append("Response too brief")
        
        if Principle["Name"] == "Harmless":
            # Check for dangerous patterns
            DangerousPatterns = ["how to make", "how to hack"]
            if any(P in ResponseLower for P in DangerousPatterns):
                Score -= 0.5
                Issues.append("May contain harmful info")
        
        Evaluation["Principles"][Principle["Name"]] = {
            "Score": max(0, Score),
            "Issues": Issues,
        }
    
    # Overall pass if all > 0.5
    Evaluation["Overall"] = all(
        P["Score"] > 0.5 for P in Evaluation["Principles"].values()
    )
    
    return Evaluation
```

---

## 7. World Model (JEPA)

### Purpose
Predict outcomes in abstract representation space, enabling imagination and planning.

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       WorldModel                              │
├──────────────────────────────────────────────────────────────┤
│  StateEncoder: nn.Module    # Obs → Representation            │
│  Predictor: nn.Module       # State + Action → Next State     │
│  TargetEncoder: nn.Module   # EMA of StateEncoder             │
├──────────────────────────────────────────────────────────────┤
│  + Encode(observation) → representation                       │
│  + Predict(state, action) → next_state                        │
│  + Imagine(state, action_sequence) → List[states]             │
│  + Train(observations, actions) → loss                        │
└──────────────────────────────────────────────────────────────┘
```

### Key JEPA Insight

```
Traditional Prediction:
  Input: "ball thrown"
  Output: Predict next TOKENS
  Problem: Many valid continuations, surface-level

JEPA Prediction:
  Input: Representation of "ball thrown"
  Output: Predict next REPRESENTATION
  Benefit: Captures meaning, ignores surface variation
```

### Neural Architecture

```python
class StateEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
    
    def forward(self, x):
        return self.net(x)

class Predictor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
```

### Imagination (Planning)

```python
def Imagine(self, InitialState, ActionSequence) -> List:
    """
    Simulate a sequence of actions to predict outcomes.
    """
    States = [InitialState]
    CurrentState = InitialState
    
    for Action in ActionSequence:
        # Predict next state
        NextState = self.Predictor(CurrentState, Action)
        States.append(NextState)
        CurrentState = NextState
    
    return States
```

---

## 8. Smart Chat Engine

### Purpose
Integrate all components into a unified chat interface.

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      SmartChatEngine                          │
├──────────────────────────────────────────────────────────────┤
│  Knowledge: KnowledgeGraph                                    │
│  Causal: CausalGraph                                          │
│  QuestionDetector: QuestionTypeDetector                       │
│  Metacognition: MetacognitiveController                       │
│  Reasoner: ChainOfThoughtReasoner                             │
│  History: List[Dict]                                          │
├──────────────────────────────────────────────────────────────┤
│  + Process(user_input) → ChatResponse                         │
│  + Learn(text) → Dict[str, int]                               │
│  + GetStats() → Dict                                          │
│  + Save() → void                                              │
└──────────────────────────────────────────────────────────────┘
```

### Processing Pipeline

```python
def Process(self, UserInput: str) -> ChatResponse:
    StartTime = time.time()
    
    # 1. Detect question type
    QType, TypeConfidence = self.QuestionDetector.Detect(UserInput)
    
    # 2. Metacognitive assessment
    MetaState = self.Metacognition.AssessQuestion(
        UserInput, QType, self.Knowledge
    )
    
    # 3. Determine thinking mode
    Mode = MetaState.ThinkingMode
    
    # 4. Generate response based on type
    Answer, ReasoningSteps = self._GenerateResponse(UserInput, QType, Mode)
    
    # 5. Constitutional check
    ConstitutionalCheck = Constitution.Evaluate(Answer, UserInput)
    
    # 6. Build response
    return ChatResponse(
        Answer=Answer,
        QuestionType=QType,
        ThinkingMode=Mode,
        Confidence=MetaState.ConfidenceInAnswer,
        ReasoningSteps=ReasoningSteps,
        ConstitutionalCheck=ConstitutionalCheck,
        MetacognitiveState=MetaState,
        ProcessingTime=time.time() - StartTime
    )
```

### Response Routing

```python
def _GenerateResponse(self, Question, QType, Mode):
    Steps = self.Reasoner.Think(Question, Mode)
    
    if QType == QuestionType.GREETING:
        return self._HandleGreeting(Question), Steps
    
    elif QType == QuestionType.FACTUAL:
        return self._HandleFactual(Question), Steps
    
    elif QType == QuestionType.CAUSAL:
        return self._HandleCausal(Question), Steps
    
    elif QType == QuestionType.COUNTERFACTUAL:
        return self._HandleCounterfactual(Question), Steps
    
    # ... more handlers
    
    else:
        return self._HandleGeneral(Question), Steps
```

---

## 9. Progress Tracker

### Purpose
Track learning progress toward human-like understanding.

### Milestone Definitions

```python
MILESTONES = [
    {
        "Level": 1,
        "Name": "Basic Pattern Recognition",
        "FactsRequired": 100,
        "CausalRequired": 10,
        "Capabilities": ["Basic text completion", "Simple Q&A"],
    },
    {
        "Level": 2,
        "Name": "Knowledge Accumulation",
        "FactsRequired": 1000,
        "CausalRequired": 100,
        "Capabilities": ["Factual answers", "Simple reasoning"],
    },
    # ... levels 3-6
]
```

### Level Calculation

```python
def GetCurrentLevel(self) -> Dict:
    Facts = self.Knowledge.Size()
    CausalCount = self.Causal.Stats["TotalRelations"]
    
    CurrentLevel = 0
    for Milestone in self.MILESTONES:
        if (Facts >= Milestone["FactsRequired"] and 
            CausalCount >= Milestone["CausalRequired"]):
            CurrentLevel = Milestone["Level"]
        else:
            break
    
    # Calculate progress to next level
    if CurrentLevel < len(self.MILESTONES):
        NextMilestone = self.MILESTONES[CurrentLevel]
        FactsProgress = Facts / NextMilestone["FactsRequired"]
        CausalProgress = CausalCount / NextMilestone["CausalRequired"]
        Progress = (FactsProgress + CausalProgress) / 2
    else:
        Progress = 1.0
    
    return {
        "CurrentLevel": self.MILESTONES[CurrentLevel - 1] if CurrentLevel else {},
        "Progress": min(Progress, 1.0),
        "Facts": Facts,
        "CausalRelations": CausalCount,
    }
```

---

## 10. Data Structures

### Enumerations

```python
class QuestionType(Enum):
    FACTUAL = auto()
    CAUSAL = auto()
    COUNTERFACTUAL = auto()
    PROCEDURAL = auto()
    COMPARATIVE = auto()
    DEFINITIONAL = auto()
    TEMPORAL = auto()
    SPATIAL = auto()
    QUANTITATIVE = auto()
    OPINION = auto()
    CLARIFICATION = auto()
    GREETING = auto()
    UNKNOWN = auto()

class ThinkingMode(Enum):
    FAST = auto()    # System 1
    MEDIUM = auto()  # Balanced
    DEEP = auto()    # System 2

class ConfidenceLevel(Enum):
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95
```

### Response Structure

```python
@dataclass
class ChatResponse:
    Answer: str
    QuestionType: QuestionType
    ThinkingMode: ThinkingMode
    Confidence: float
    ReasoningSteps: List[ReasoningStep]
    ConstitutionalCheck: Dict
    MetacognitiveState: MetacognitiveState
    ProcessingTime: float
```

---

## 🔗 Component Interactions

```
User Input
    │
    ▼
┌─────────────────┐
│ QuestionDetector │──────────────────────────┐
└─────────────────┘                           │
    │                                         │
    ▼                                         ▼
┌─────────────────┐                    ┌──────────────┐
│  Metacognition   │───────────────────│ ThinkingMode │
└─────────────────┘                    └──────────────┘
    │                                         │
    ▼                                         ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ KnowledgeGraph  │  │   CausalGraph   │  │   WorldModel    │
└─────────────────┘  └─────────────────┘  └─────────────────┘
    │                      │                      │
    └──────────────────────┼──────────────────────┘
                           │
                           ▼
                    ┌─────────────────┐
                    │ CoT Reasoner    │
                    └─────────────────┘
                           │
                           ▼
                    ┌─────────────────┐
                    │ Constitutional  │
                    └─────────────────┘
                           │
                           ▼
                      Response
```

---

This document provides the technical foundation for understanding, modifying, and extending GroundZero AI.
