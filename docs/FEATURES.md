# ✨ Features Guide

## Complete Guide to GroundZero AI Features

This document explains each feature in detail, what it does, and how to use it.

---

## 📑 Table of Contents

1. [Smart Question Detection](#1-smart-question-detection)
2. [Knowledge Graph](#2-knowledge-graph)
3. [Causal Graph](#3-causal-graph)
4. [Chain-of-Thought Reasoning](#4-chain-of-thought-reasoning)
5. [Metacognition](#5-metacognition)
6. [Constitutional AI](#6-constitutional-ai)
7. [World Model (JEPA)](#7-world-model-jepa)
8. [Dual System Thinking](#8-dual-system-thinking)
9. [Progress Tracking](#9-progress-tracking)
10. [Web Dashboard](#10-web-dashboard)

---

## 1. Smart Question Detection

### What It Does
Automatically detects the type of question you're asking and routes it to the appropriate reasoning system. No need for commands like `/whatif` or `/explain`.

### Question Types Detected

| Type | Pattern | Example | System Used |
|------|---------|---------|-------------|
| **GREETING** | hi, hello, hey | "Hello!" | Quick response |
| **FACTUAL** | what is, who is | "What is AI?" | Knowledge Graph |
| **CAUSAL** | why does, what causes | "Why does rain fall?" | Causal Graph |
| **COUNTERFACTUAL** | what if, imagine if | "What if it rains?" | Counterfactual Engine |
| **PROCEDURAL** | how do I, steps to | "How do I bake a cake?" | Step Generator |
| **COMPARATIVE** | difference between | "X vs Y?" | Comparison Engine |
| **DEFINITIONAL** | define, meaning of | "Define entropy" | Definition Lookup |
| **TEMPORAL** | when did, what time | "When did X happen?" | Temporal Reasoning |
| **SPATIAL** | where is, location | "Where is Paris?" | Spatial Knowledge |
| **QUANTITATIVE** | how many, how much | "How many planets?" | Quantitative Facts |
| **OPINION** | what do you think | "Your opinion on X?" | Balanced Response |
| **CLARIFICATION** | explain more | "What do you mean?" | Elaboration |

### How It Works

```python
# Input: "What if it rains tomorrow?"

# Step 1: Pattern matching
Detected patterns:
  - "what if" → COUNTERFACTUAL (score: 0.85)

# Step 2: Confidence calculation
Type: COUNTERFACTUAL
Confidence: 85%

# Step 3: Route to appropriate system
→ Uses Causal Graph for counterfactual reasoning
→ Sets ThinkingMode to DEEP
```

### Example Usage

```
You: Hello!
🧠 System: [GREETING detected] → Quick friendly response

You: What causes thunder?
🧠 System: [CAUSAL detected] → Queries Causal Graph → Explains cause-effect

You: What if lightning strikes?
🧠 System: [COUNTERFACTUAL detected] → Computes probability changes
```

---

## 2. Knowledge Graph

### What It Does
Stores facts as explicit **(subject, predicate, object)** triples. Unlike neural networks where knowledge is hidden in weights, every fact is inspectable and queryable.

### Key Capabilities

| Capability | Description |
|------------|-------------|
| **Add Facts** | Store new knowledge triples |
| **Query** | Search by subject, predicate, or object |
| **Transitive Inference** | If A→B and B→C, infer A→C |
| **Multi-hop Reasoning** | Find connections across multiple facts |
| **Natural Language Extraction** | Extract facts from text automatically |
| **Persistence** | SQLite database survives restarts |

### How Facts Are Stored

```
Text: "Dogs are mammals. Mammals are warm-blooded."

Extracted triples:
  (dog, is_a, mammal)
  (mammal, is_a, warm-blooded)

Inferred triple:
  (dog, is_a, warm-blooded)  ← Transitive inference!
```

### Supported Relations

| Predicate | Example | Meaning |
|-----------|---------|---------|
| `is_a` | dog is_a animal | Category membership |
| `has` | car has wheels | Properties |
| `part_of` | wheel part_of car | Composition |
| `located_in` | Paris located_in France | Location |
| `causes` | rain causes wet | Causation |
| `created_by` | Python created_by Guido | Attribution |

### Query Examples

```python
# Find all facts about dogs
kg.Query(Subject="dog")
→ [(dog, is_a, mammal), (dog, has, fur), ...]

# Find all animals
kg.Query(Predicate="is_a", Object="animal")
→ [(dog, is_a, animal), (cat, is_a, animal), ...]

# Find related concepts (2 hops)
kg.GetRelated("dog", MaxDepth=2)
→ dog → mammal → warm-blooded → ...
```

---

## 3. Causal Graph

### What It Does
Stores and reasons about **cause-effect relationships**. Unlike correlation, this is TRUE causal understanding.

### Key Capabilities

| Capability | Description |
|------------|-------------|
| **Causal Relations** | Store cause → effect with strength |
| **Causal Chains** | Find how A leads to B through intermediaries |
| **Counterfactuals** | "What if X happened?" with probability changes |
| **Interventions** | "What if I MAKE X happen?" |
| **Explanations** | Generate "why" explanations |

### How Causal Relations Work

```
Stored relations:
  rain → wet_ground (90% strength)
  wet_ground → slippery (80% strength)
  slippery → accident (60% strength)

Causal chain from rain to accident:
  rain → wet_ground → slippery → accident

Combined strength:
  0.9 × 0.8 × 0.6 = 43.2%
```

### Counterfactual Reasoning

```
Question: "What if it rains?"

Computation:
  - wet_ground: +90% (direct effect)
  - slippery: +72% (0.9 × 0.8)
  - accident: +43% (0.9 × 0.8 × 0.6)

Answer: "If rain happens:
  • wet_ground becomes 90% more likely
  • slippery becomes 72% more likely
  • accident becomes 43% more likely"
```

### Difference: Correlation vs Causation

```
CORRELATION (Pattern Matching):
  "Rain and wet ground often appear together"
  → Cannot answer "What if rain didn't happen?"

CAUSATION (GroundZero):
  "Rain CAUSES wet ground with 90% strength"
  → CAN answer "What if rain didn't happen?"
  → CAN compute downstream effects
```

---

## 4. Chain-of-Thought Reasoning

### What It Does
Generates **step-by-step reasoning** with verification at each step. Based on OpenAI o1/o3 models.

### Thinking Modes

| Mode | Steps | When Used | Example |
|------|-------|-----------|---------|
| **FAST** | 1-3 | Simple questions, greetings | "Hello!" |
| **MEDIUM** | 3-7 | Factual questions | "What is AI?" |
| **DEEP** | 7-15 | Complex reasoning | "What if X causes Y?" |

### Reasoning Chain Example

```
Question: "Is Socrates mortal?"

🧠 Chain of Thought:

  1. [✓] Analyzing question: "Is Socrates mortal?"
      → Identified as definitional/logical query (95%)

  2. [✓] Checking knowledge base for relevant facts
      → Found: (Socrates, is_a, human) (90%)

  3. [✓] Checking for related knowledge
      → Found: (human, is_a, mortal) (95%)

  4. [✓] Applying transitivity rule
      → If Socrates is_a human AND human is_a mortal
      → THEN Socrates is_a mortal (85%)

  5. [✓] Verifying reasoning chain
      → All steps logically valid (90%)

  6. [✓] Formulating final answer
      → "Yes, Socrates is mortal" (88% confidence)
```

### Self-Verification

Each step is verified for:
- Logical consistency with previous steps
- Contradiction detection
- Confidence thresholds
- Knowledge base support

If errors are detected, the system **backtracks** and tries alternative approaches.

---

## 5. Metacognition

### What It Does
The system's ability to **think about its own thinking**. It knows what it knows and doesn't know.

### Key Capabilities

| Capability | Description |
|------------|-------------|
| **Confidence Calibration** | Accurate estimate of answer reliability |
| **Knowledge Gap Detection** | Identifies what it doesn't know |
| **Uncertainty Acknowledgment** | Honest about limitations |
| **System Selection** | Decides when to use deep vs fast thinking |

### Confidence Levels

| Level | Range | Meaning |
|-------|-------|---------|
| VERY_HIGH | 90-100% | Highly confident, well-supported |
| HIGH | 70-90% | Confident with minor uncertainty |
| MEDIUM | 50-70% | Moderate confidence, verify if critical |
| LOW | 30-50% | Limited confidence, use with caution |
| VERY_LOW | 0-30% | Uncertain, likely needs more information |

### Example Metacognitive Assessment

```
Question: "What causes black holes?"

Metacognitive Assessment:
  Knowledge Coverage: 45% (limited astrophysics knowledge)
  Confidence: 52%
  Knowledge Gaps: ["gravitational collapse", "singularity", "event horizon"]
  Thinking Mode: DEEP (complex topic)
  
Response includes:
  "I'm moderately confident about this. I have limited knowledge 
   about: gravitational collapse, singularity. Please verify 
   this information."
```

### Honest Uncertainty

```
High confidence: "I'm very confident in this answer."
Medium confidence: "I'm fairly confident, though there may be nuances."
Low confidence: "I'm not very confident. Please take with caution."
Very low: "I'm quite uncertain. This is my best guess."
```

---

## 6. Constitutional AI

### What It Does
Evaluates responses against explicit **ethical principles** before delivery. Based on Anthropic's Constitutional AI research.

### Principles Evaluated

| Principle | Description | Checks |
|-----------|-------------|--------|
| **Helpful** | Genuinely useful to user | Answers question, actionable |
| **Harmless** | Avoids causing harm | No dangerous info, no encouragement |
| **Honest** | Truthful and accurate | No made-up facts, acknowledges uncertainty |
| **Respectful** | Treats all with respect | No stereotypes, inclusive language |

### How Evaluation Works

```python
Response: "Here's how to make a cake: [detailed recipe]"
Question: "How do I bake a cake?"

Evaluation:
  Helpful:    0.95 ✓ (directly answers, provides steps)
  Harmless:   1.00 ✓ (no dangerous information)
  Honest:     0.90 ✓ (factual, no false claims)
  Respectful: 1.00 ✓ (neutral, inclusive)

Overall: PASS
```

### What Gets Flagged

```
❌ Overconfidence without evidence
❌ Potentially harmful information
❌ Very brief unhelpful responses
❌ Stereotypes or biased language
```

---

## 7. World Model (JEPA)

### What It Does
Predicts outcomes in **abstract representation space**, not raw tokens. Based on Yann LeCun's Joint Embedding Predictive Architecture.

### Key Difference

```
Traditional LLM:
  Input: "ball thrown"
  Predicts: Next TOKEN (surface level)
  "ball thrown into the air"

World Model (JEPA):
  Input: "ball thrown"
  Predicts: Next STATE (abstract representation)
  ball_state: {position: high, velocity: decreasing, will_fall: true}
```

### Capabilities

| Capability | Description |
|------------|-------------|
| **State Encoding** | Converts observations to abstract representations |
| **Prediction** | Predicts next state given current state + action |
| **Imagination** | Simulates sequences of actions before acting |
| **Planning** | Uses imagination to find best action sequences |

### Imagination Example

```
Current state: ball_on_table
Action sequence: [push, gravity_acts, ...]

Imagination:
  Step 1: push → ball_moving
  Step 2: gravity → ball_falling
  Step 3: ground → ball_stopped

Predicted outcome: ball on floor
Confidence: 85%
```

---

## 8. Dual System Thinking

### What It Does
Implements **System 1** (fast, intuitive) and **System 2** (slow, deliberate) thinking with metacognitive control. Based on Kahneman's "Thinking Fast and Slow".

### System Comparison

| Aspect | System 1 (Fast) | System 2 (Deep) |
|--------|-----------------|-----------------|
| **Speed** | Milliseconds | Seconds |
| **Effort** | Low | High |
| **Type** | Pattern matching | Logical reasoning |
| **When Used** | Simple questions | Complex problems |
| **Examples** | "Hello!", "2+2?" | "What if X?", "Why does Y?" |

### Automatic Selection

```python
# Simple greeting → System 1
"Hello!" → FAST mode → Quick response

# Complex causal question → System 2
"What if deforestation increases?" → DEEP mode → Multi-step reasoning
```

### Selection Criteria

System 2 is used when:
- Question type is CAUSAL or COUNTERFACTUAL
- Complexity score > 0.7
- Confidence is low (< 0.5)
- Multi-step reasoning required

---

## 9. Progress Tracking

### What It Does
Tracks learning progress toward human-like understanding with **levels and milestones**.

### Progress Levels

| Level | Name | Facts | Causal | Timeline |
|-------|------|-------|--------|----------|
| 1 | Basic Pattern Recognition | 100 | 10 | 1-2 days |
| 2 | Knowledge Accumulation | 1,000 | 100 | 1-2 weeks |
| 3 | Causal Understanding | 5,000 | 500 | 1-2 months |
| 4 | Reasoning Chains | 20,000 | 2,000 | 3-6 months |
| 5 | Deep Understanding | 100,000 | 10,000 | 1-2 years |
| 6 | Human-Like Reasoning | 500,000 | 50,000 | 3-5 years |

### Status Command

```bash
python main.py status
```

```
============================================================
📊 GroundZero AI - Status
============================================================

  Current Level: 2 - Knowledge Accumulation
  Progress to Next: 35%

  Knowledge Facts: 1,247
  Causal Relations: 156
  Questions Answered: 342
  Average Confidence: 73%
============================================================
```

---

## 10. Web Dashboard

### What It Does
Visual interface showing real-time stats, progress, and capabilities.

### Features

| Feature | Description |
|---------|-------------|
| **Dark/Light Mode** | Toggle with button or Ctrl+T |
| **Live Stats** | Facts, causal links, questions answered |
| **Progress Bar** | Visual progress to next level |
| **Timeline** | Milestone tracker with status |
| **Capabilities** | Current abilities display |
| **Architecture** | Visual system diagram |
| **Research Sources** | Citations to papers |

### Launch Dashboard

```bash
python main.py dashboard --port 8080
```

Then open: `http://localhost:8080`

### Dashboard Sections

1. **Header** - Title, live indicator, theme toggle
2. **Stats Grid** - Key metrics in cards
3. **Progress** - Current level and progress bar
4. **Timeline** - Milestones with completion status
5. **Capabilities** - What the system can do now
6. **How It Works** - Architecture diagram
7. **Research** - Sources and citations

---

## 🎯 Feature Summary

| Feature | Purpose | Key Benefit |
|---------|---------|-------------|
| Question Detection | Auto-route questions | No commands needed |
| Knowledge Graph | Store facts explicitly | Inspectable knowledge |
| Causal Graph | Cause-effect reasoning | True understanding |
| Chain-of-Thought | Step-by-step thinking | Explainable reasoning |
| Metacognition | Self-awareness | Honest uncertainty |
| Constitutional AI | Safety checks | Aligned responses |
| World Model | Predict outcomes | Can imagine futures |
| Dual System | Fast + Deep thinking | Efficient reasoning |
| Progress Tracking | Monitor learning | Clear milestones |
| Dashboard | Visual interface | Easy monitoring |

---

## 🚀 Getting Started

```bash
# Test all features
python main.py test

# Start chatting (uses all features automatically)
python main.py chat

# View your progress
python main.py status
```

Each feature works together seamlessly - just ask questions naturally and the system handles the rest!
