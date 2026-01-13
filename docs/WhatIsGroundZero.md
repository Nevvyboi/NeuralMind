# 🧠 What Is GroundZero AI?

> **A Neurosymbolic Knowledge System with State-of-the-Art NLP**

---

## 📖 Overview

GroundZero AI is an **experimental knowledge graph system** that combines:

1. **Structured Knowledge Storage** - Facts as subject-predicate-object triples
2. **Causal Reasoning** - Cause-effect relationships with strength scores
3. **State-of-the-Art NLP** - spaCy-based extraction with dependency parsing
4. **Logical Inference** - Transitive reasoning and counterfactuals

### The Core Philosophy

Unlike neural language models that pattern-match text, GroundZero attempts to build **explicit, structured understanding**. Every fact is stored as a discrete triple, every causal relation has a measurable strength, and all reasoning is traceable.

---

## 🔬 Technical Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GroundZero AI v2.5                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐     ┌─────────────────┐     ┌───────────────┐ │
│  │  NLP Extractor  │────▶│  Knowledge Graph │────▶│   Reasoner    │ │
│  │    (spaCy)      │     │    (SQLite)      │     │  (Inference)  │ │
│  └─────────────────┘     └─────────────────┘     └───────────────┘ │
│          │                        │                      │          │
│          │                        │                      │          │
│          ▼                        ▼                      ▼          │
│  ┌─────────────────┐     ┌─────────────────┐     ┌───────────────┐ │
│  │  Causal Graph   │     │   Metacognition  │     │  Question     │ │
│  │    (JSON)       │     │   (Confidence)   │     │  Detector     │ │
│  └─────────────────┘     └─────────────────┘     └───────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### State-of-the-Art NLP Pipeline

The system uses **spaCy** for advanced natural language processing:

```
Input Text
    │
    ▼
┌───────────────────────────────────────┐
│           spaCy Pipeline              │
├───────────────────────────────────────┤
│  1. Tokenization                      │
│  2. Part-of-Speech Tagging            │
│  3. Dependency Parsing                │
│  4. Named Entity Recognition (NER)    │
│  5. Noun Chunk Detection              │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│        Triple Extraction              │
├───────────────────────────────────────┤
│  • Find ROOT verb                     │
│  • Extract nsubj (subject)            │
│  • Extract dobj/pobj (object)         │
│  • Expand compounds ("Gordian Capital"│
│    not just "Capital")                │
│  • Normalize predicates to lemmas     │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│      Causal Relation Extraction       │
├───────────────────────────────────────┤
│  • Detect causal verbs (causes, leads)│
│  • Match linguistic patterns          │
│  • Validate with dependency tree      │
│  • Extract main concepts from clauses │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│         Quality Filtering             │
├───────────────────────────────────────┤
│  • Minimum entity length (2 chars)    │
│  • Maximum entity words (6)           │
│  • Filter stopwords from entities     │
│  • Blacklist generic predicates       │
│  • Deduplicate facts                  │
└───────────────────────────────────────┘
    │
    ▼
Knowledge Graph & Causal Graph
```

---

## 🎯 What It Does (Capabilities)

### ✅ Fact Extraction (SVO Triples)

Using spaCy's dependency parser:

```python
Input: "Physics is the natural science that studies matter and energy."

Dependency Tree:
  Physics ──nsubj──▶ is
  science ──attr───▶ is
  matter  ──dobj───▶ studies
  energy  ──conj───▶ matter

Extracted:
  (physics, is_a, natural science)
  (physics, studies, matter)
  (physics, studies, energy)
```

### ✅ Named Entity Recognition

```python
Input: "Albert Einstein developed the theory of relativity at Princeton."

Entities:
  • Albert Einstein (PERSON)
  • Princeton (ORG/GPE)
  
Extracted:
  (Albert Einstein, developed, theory of relativity)
  (Albert Einstein, worked_at, Princeton)
```

### ✅ Causal Relation Extraction

Combines pattern matching with linguistic validation:

```python
Input: "Deforestation causes soil erosion which leads to flooding."

Causal Relations:
  • deforestation → soil erosion (85% strength)
  • soil erosion → flooding (80% strength)

Patterns Matched:
  • "X causes Y" → cause_verb
  • "X leads to Y" → lead_to
```

### ✅ Compound Noun Handling

Unlike basic regex, spaCy extracts complete noun phrases:

```
Basic Regex: "Capital" (incomplete)
spaCy:       "Gordian Capital" (complete)

Basic Regex: "theory" (vague)
spaCy:       "theory of relativity" (specific)
```

### ✅ Transitive Inference

```python
Stored Facts:
  (dog, is_a, mammal)
  (mammal, is_a, animal)
  
Query: "What is a dog?"
Inference: dog → mammal → animal
Result: "A dog is an animal" (derived)
```

### ✅ Counterfactual Reasoning

```python
Causal Chain:
  rain → wet_ground (90%)
  wet_ground → slippery (80%)
  slippery → accidents (60%)

Query: "What if there was no rain?"
Result:
  • wet_ground: -90%
  • slippery: -72% (0.9 × 0.8)
  • accidents: -43.2% (0.9 × 0.8 × 0.6)
```

---

## ❌ What It Cannot Do (Limitations)

### Cannot Generate Text

```
User: "Write a poem about the ocean"
GroundZero: ❌ Cannot do this

Why: No language model. Only retrieves stored facts.
```

### Cannot Understand Context

```
User: "It's really hot today"
GroundZero: ❌ Limited understanding

Why: No contextual reasoning about implicit meanings.
```

### Cannot Learn Abstract Concepts

```
Stored: (dog, is_a, mammal)
Not Understood: What makes something a "mammal"?

Why: Only stores explicit facts, not abstract definitions.
```

### Cannot Generalize

```
Known: (dog, has, four_legs), (cat, has, four_legs)
Cannot Infer: "Most mammals have four legs"

Why: No statistical generalization mechanism.
```

### Comparison with Language Models

| Feature | GroundZero AI | ChatGPT/Claude |
|---------|---------------|----------------|
| Generates text | ❌ | ✅ |
| Understands context | ❌ | ✅ |
| Creative writing | ❌ | ✅ |
| Explicit facts | ✅ | ⚠️ (implicit) |
| Traceable reasoning | ✅ | ❌ |
| Causal chains | ✅ | ⚠️ (no explicit) |
| Counterfactuals | ✅ | ⚠️ (text-based) |
| Hallucination | ❌ | ✅ |

---

## 🔧 Technical Details

### NLP Extraction Methods

#### 1. Subject-Verb-Object (SVO) Extraction

```python
def _ExtractTriplesFromVerb(Verb, Sent):
    """
    Uses spaCy dependency tree to extract triples
    
    Key dependency tags:
    - nsubj: nominal subject
    - nsubjpass: passive subject  
    - dobj: direct object
    - pobj: prepositional object
    - attr: attribute (for "is a" relations)
    """
    Subjects = [child for child in Verb.children 
                if child.dep_ in SUBJECT_DEPS]
    
    Objects = [child for child in Verb.children
               if child.dep_ in OBJECT_DEPS]
    
    # Expand to full noun phrases
    for subj in Subjects:
        SubjPhrase = _GetFullPhrase(subj)  # "Gordian Capital" not "Capital"
```

#### 2. Causal Pattern Matching

```python
CAUSAL_PATTERNS = [
    (r'(.+?)\s+causes?\s+(.+)', 'cause_verb'),
    (r'(.+?)\s+leads?\s+to\s+(.+)', 'lead_to'),
    (r'(.+?)\s+results?\s+in\s+(.+)', 'result_in'),
    (r'due\s+to\s+(.+?),\s*(.+)', 'due_to'),
    (r'because\s+of\s+(.+?),\s*(.+)', 'because_of'),
    # ... 15+ patterns
]

# Validate with NLP
def _ValidateCausalRelation(Cause, Effect, Doc):
    # Check entities are valid noun phrases
    # Check not same entity
    # Check minimum length
```

#### 3. Quality Filtering

```python
PREDICATE_BLACKLIST = {
    'is', 'are', 'was', 'were',  # Too generic
    'do', 'does', 'did',
    'have', 'has', 'had',
    # ...
}

GOOD_PREDICATES = {
    'is_a', 'has', 'contains', 'consists_of',
    'located_in', 'created_by', 'causes',
    # ...
}

MIN_ENTITY_LENGTH = 2
MAX_ENTITY_WORDS = 6
```

### Storage Formats

#### Knowledge Graph (SQLite)

```sql
CREATE TABLE facts (
    id INTEGER PRIMARY KEY,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    source TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(subject, predicate, object)
);

CREATE INDEX idx_subject ON facts(subject);
CREATE INDEX idx_object ON facts(object);
CREATE INDEX idx_predicate ON facts(predicate);
```

#### Causal Graph (JSON)

```json
{
  "relations": [
    {
      "cause": "rain",
      "effect": "wet_ground",
      "strength": 0.9,
      "pattern": "cause_verb",
      "source": "Wikipedia:Weather"
    }
  ],
  "stats": {
    "total_relations": 87,
    "chains_found": 12,
    "last_updated": "2026-01-13T15:30:00"
  }
}
```

---

## 📊 Research Background

### Inspiration Sources

1. **Classic Expert Systems** - MYCIN, Cyc
2. **Knowledge Graphs** - Freebase, Wikidata, DBpedia
3. **Neurosymbolic AI** - Combining neural and symbolic approaches
4. **Causal Inference** - Judea Pearl's work on causality

### Academic References

- Yang et al. (2021) - "A Survey on Extraction of Causal Relations from Natural Language Text"
- spaCy Documentation - Dependency Parsing and NER
- textacy - Subject-Verb-Object triple extraction
- Knowledge Graph Construction surveys

### State-of-the-Art Techniques Used

| Technique | Implementation | Source |
|-----------|----------------|--------|
| Dependency Parsing | spaCy | explosion.ai |
| Named Entity Recognition | spaCy NER | explosion.ai |
| SVO Extraction | Custom + textacy patterns | Academic research |
| Causal Patterns | 15+ linguistic patterns | Causal extraction surveys |
| Compound Noun Handling | spaCy subtree expansion | NLP best practices |

---

## 🚀 Usage

### Installation

```bash
# Install dependencies
pip install spacy

# Download spaCy model
python -m spacy download en_core_web_sm

# For better accuracy (larger model):
python -m spacy download en_core_web_md
```

### Commands

```bash
# Test all components
python main.py test

# Interactive chat
python main.py chat

# Train from Wikipedia (with advanced NLP)
python main.py train

# Launch dashboard
python main.py dashboard
```

### API Usage

```python
from nlp_extractor import NLPProcessor

# Initialize
processor = NLPProcessor(UseSpacy=True)

# Extract facts
facts = processor.ExtractFacts("Dogs are mammals. Mammals are warm-blooded.")
# [('dogs', 'are', 'mammals'), ('mammals', 'are', 'warm-blooded')]

# Extract causal relations
causal = processor.ExtractCausal("Rain causes wet ground which leads to flooding.")
# [('rain', 'wet ground', 0.85), ('wet ground', 'flooding', 0.80)]
```

---

## 📈 Progress System

### Levels

| Level | Name | Facts | Causal | What Works |
|-------|------|-------|--------|------------|
| 0 | Starting | 0 | 0 | Nothing yet |
| 1 | Pattern Recognition | 100 | 10 | Basic fact retrieval |
| 2 | Knowledge Building | 1,000 | 100 | Simple questions |
| 3 | Causal Understanding | 5,000 | 500 | "Why" questions |
| 4 | Reasoning Chains | 20,000 | 2,000 | Multi-step inference |
| 5 | Deep Understanding | 100,000 | 10,000 | Complex queries |
| 6 | Human-Like | 500,000 | 50,000 | Comprehensive (aspirational) |

### Component Effectiveness

| Component | Level 0 | Level 1 | Level 2 | Level 3 |
|-----------|---------|---------|---------|---------|
| Knowledge Graph | Empty | 100+ facts | 1000+ | 5000+ |
| Causal Graph | None | 10+ relations | 100+ | 500+ |
| World Model | No data | Building | Predicting | Rich |

---

## 🎯 Honest Assessment

### What This IS

- ✅ A **knowledge graph** with explicit fact storage
- ✅ A **rule-based inference engine**
- ✅ A **causal reasoning system**
- ✅ An **educational project** exploring symbolic AI
- ✅ A **structured database** with query capabilities

### What This IS NOT

- ❌ An AI "model" with learned weights
- ❌ A language model like GPT/Claude
- ❌ Artificial General Intelligence
- ❌ A conversational AI
- ❌ A production-ready system

### When To Use This

| Use Case | Suitable? |
|----------|-----------|
| Learning about knowledge graphs | ✅ Yes |
| Understanding causal reasoning | ✅ Yes |
| Exploring symbolic AI | ✅ Yes |
| Structured fact retrieval | ✅ Yes |
| Natural conversation | ❌ No |
| Creative writing | ❌ No |
| General-purpose AI assistant | ❌ No |

---

## 🔮 Future Directions

### Potential Improvements

1. **Better NLP Models** - Use en_core_web_trf (transformer-based)
2. **Embedding Search** - Add vector similarity for related facts
3. **Neural Knowledge Completion** - Predict missing relations
4. **Coreference Resolution** - Handle "it", "they", etc.
5. **Cross-sentence Relations** - Extract facts spanning sentences

### Research Areas

- Neurosymbolic integration
- Knowledge graph embeddings
- Automated ontology learning
- Causal discovery algorithms

---

## 📜 License

MIT License - Free for educational and experimental use.

---

## 🙏 Acknowledgments

- **spaCy** - Industrial-strength NLP library
- **textacy** - Higher-level NLP utilities
- **Judea Pearl** - Foundational work on causality
- **Knowledge Graph research community**

---

*GroundZero AI - Building understanding from the ground up* 🧠
