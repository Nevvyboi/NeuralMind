# Building Your Own GPT: A Complete Guide
## From GroundZero to GroundZero-LM

**Author:** GroundZero AI Project  
**Version:** 1.0  
**Estimated Time:** 4-6 months  
**Difficulty:** Advanced  
**Prerequisites:** Your existing GroundZero system with 50K+ triples trained

---

## Table of Contents

1. [Introduction & Philosophy](#1-introduction--philosophy)
2. [Prerequisites & Milestones](#2-prerequisites--milestones)
3. [Understanding the Theory](#3-understanding-the-theory)
4. [Architecture Overview](#4-architecture-overview)
5. [Phase 1: Tokenizer](#5-phase-1-tokenizer)
6. [Phase 2: Embeddings](#6-phase-2-embeddings)
7. [Phase 3: Attention Mechanism](#7-phase-3-attention-mechanism)
8. [Phase 4: Transformer Block](#8-phase-4-transformer-block)
9. [Phase 5: Full GPT Model](#9-phase-5-full-gpt-model)
10. [Phase 6: Training Infrastructure](#10-phase-6-training-infrastructure)
11. [Phase 7: Training the Model](#11-phase-7-training-the-model)
12. [Phase 8: Integration with GroundZero](#12-phase-8-integration-with-groundzero)
13. [Phase 9: Fine-tuning & Alignment](#13-phase-9-fine-tuning--alignment)
14. [Phase 10: Optimization & Deployment](#14-phase-10-optimization--deployment)
15. [Appendix: Math Deep Dive](#15-appendix-math-deep-dive)
16. [Appendix: Troubleshooting](#16-appendix-troubleshooting)
17. [Appendix: Resources](#17-appendix-resources)

---

## 1. Introduction & Philosophy

### What You're Building

You're building a **causal language model** - a neural network that predicts the next token given all previous tokens. This is the foundation of GPT, ChatGPT, and most modern AI assistants.

```
Input:  "The dog sat on the"
Output: "mat" (predicted next token)

Input:  "Why do dogs"  
Output: "bark" (predicted next token)
```

By training this on massive amounts of text, the model learns:
- Grammar and syntax
- Facts about the world
- Reasoning patterns
- How to follow instructions

### Why Build From Scratch?

| Reason | Value |
|--------|-------|
| **Deep Understanding** | You'll truly understand how LLMs work |
| **Customization** | Integrate directly with your knowledge graph |
| **No Dependencies** | Not reliant on OpenAI/Anthropic |
| **Learning** | The best way to learn is to build |
| **Ownership** | Your model, your data, your control |

### What Makes This Different

Your GroundZero-LM will be unique:

```
┌─────────────────────────────────────────────────────────────┐
│                    GroundZero-LM                            │
├─────────────────────────────────────────────────────────────┤
│  Standard GPT:        Text → Transformer → Text             │
│                                                             │
│  GroundZero-LM:       Text → Transformer ─┬─► Text          │
│                              ↑            │                 │
│                              │            ▼                 │
│                       Knowledge Graph ◄── Facts             │
│                       (Your TransE)                         │
│                                                             │
│  Your model will be GROUNDED in explicit knowledge,         │
│  not just patterns in text. This reduces hallucination.     │
└─────────────────────────────────────────────────────────────┘
```

### Honest Expectations

| Aspect | Reality |
|--------|---------|
| **Training Time** | Weeks to months on consumer hardware |
| **Model Size** | 10M-100M parameters (tiny by industry standards) |
| **Capability** | Basic conversation, knowledge retrieval, simple reasoning |
| **Limitations** | Won't match GPT-4, may struggle with complex tasks |
| **Value** | Immense learning, custom integration, full control |

---

## 2. Prerequisites & Milestones

### Before You Start: GroundZero Requirements

Your current system should reach these thresholds before beginning GPT implementation:

```
┌─────────────────────────────────────────────────────────────┐
│  MINIMUM REQUIREMENTS TO START                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Knowledge Graph:                                           │
│    □ 50,000+ triples                                        │
│    □ 20,000+ unique entities                                │
│    □ 50+ relation types                                     │
│                                                             │
│  Neural Network (TransE):                                   │
│    □ Loss < 0.01                                            │
│    □ 500+ training epochs                                   │
│    □ Prediction accuracy > 70%                              │
│                                                             │
│  System Stability:                                          │
│    □ Dashboard running smoothly                             │
│    □ Continuous learning working                            │
│    □ All tests passing                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Hardware Requirements

| Component | Minimum | Recommended | Ideal |
|-----------|---------|-------------|-------|
| **RAM** | 16GB | 32GB | 64GB |
| **GPU** | None (CPU only) | RTX 3060 12GB | RTX 4090 24GB |
| **Storage** | 50GB SSD | 200GB SSD | 500GB NVMe |
| **CPU** | 4 cores | 8 cores | 16 cores |

### Software Requirements

```bash
# Core
Python 3.10+
NumPy
 
# Optional but recommended
PyTorch 2.0+        # For GPU acceleration
CUDA 11.8+          # If using NVIDIA GPU

# For training data
datasets            # HuggingFace datasets
sentencepiece       # Tokenization
```

### Project Milestones

```
Month 1: Foundation
├── Week 1-2: Tokenizer (BPE from scratch)
└── Week 3-4: Embeddings + Positional Encoding

Month 2: Core Architecture  
├── Week 1-2: Self-Attention Mechanism
└── Week 3-4: Transformer Block

Month 3: Model Assembly
├── Week 1-2: Full GPT Architecture
└── Week 3-4: Training Loop + Data Pipeline

Month 4: Training
├── Week 1-2: Initial training on small dataset
└── Week 3-4: Scale up training

Month 5: Integration
├── Week 1-2: Connect to GroundZero knowledge graph
└── Week 3-4: Knowledge-grounded generation

Month 6: Polish
├── Week 1-2: Fine-tuning on QA pairs
└── Week 3-4: Optimization + Deployment
```

---

## 3. Understanding the Theory

### The Core Idea

A language model learns the probability distribution over sequences of tokens:

```
P(token_n | token_1, token_2, ..., token_{n-1})
```

Given "The cat sat on the", predict the probability of each possible next word:
- "mat": 0.15
- "floor": 0.12
- "chair": 0.10
- "dog": 0.001
- ...

### Why Transformers?

Before transformers, we had RNNs and LSTMs:

```
RNN/LSTM (Sequential):
token_1 → [process] → token_2 → [process] → token_3 → [process] → output
                ↓                     ↓                     ↓
             state_1 ──────────► state_2 ──────────► state_3

Problem: Information from token_1 gets "diluted" by the time we reach token_100
Problem: Can't parallelize - must process sequentially
```

Transformers use **attention** to look at all tokens simultaneously:

```
Transformer (Parallel):
token_1 ─────┬─────┬─────┬─────► output_1
token_2 ─────┼─────┼─────┼─────► output_2
token_3 ─────┼─────┼─────┼─────► output_3
token_4 ─────┴─────┴─────┴─────► output_4
        [All tokens attend to all other tokens]

Benefit: Token_100 can directly "see" token_1
Benefit: Fully parallelizable on GPU
```

### The Attention Equation

The heart of transformers is scaled dot-product attention:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

Where:
- Q = Queries (what am I looking for?)
- K = Keys (what do I contain?)
- V = Values (what information do I have?)
- d_k = dimension of keys (for numerical stability)
```

Visual intuition:

```
Query: "What comes after 'The dog sat on the'?"

Keys (all previous tokens):
  "The"  → key_1
  "dog"  → key_2  ← High attention (subject)
  "sat"  → key_3  ← High attention (verb)
  "on"   → key_4  ← High attention (preposition)
  "the"  → key_5  ← High attention (article before noun)

Attention weights (after softmax):
  "The"  → 0.05
  "dog"  → 0.20
  "sat"  → 0.25
  "on"   → 0.30
  "the"  → 0.20

Output: Weighted combination of Values
  → Strongly influenced by "on" and "sat"
  → Predicts: "mat", "floor", "chair" (things you sit ON)
```

### Multi-Head Attention

Instead of one attention, use multiple "heads" that can attend to different things:

```
Head 1: Attends to syntactic structure
Head 2: Attends to semantic meaning
Head 3: Attends to positional patterns
Head 4: Attends to entity relationships
...

Final output = Concatenate all heads + Linear projection
```

### The Transformer Block

```
┌─────────────────────────────────────────┐
│           TRANSFORMER BLOCK             │
├─────────────────────────────────────────┤
│                                         │
│   Input                                 │
│     │                                   │
│     ▼                                   │
│   ┌─────────────────┐                   │
│   │ Layer Norm      │                   │
│   └────────┬────────┘                   │
│            │                            │
│     ┌──────┴──────┐                     │
│     │             │                     │
│     ▼             │                     │
│   ┌─────────────┐ │                     │
│   │ Multi-Head  │ │                     │
│   │ Attention   │ │                     │
│   └──────┬──────┘ │                     │
│          │        │                     │
│          ▼        │                     │
│        [ADD] ◄────┘  (Residual)         │
│          │                              │
│          ▼                              │
│   ┌─────────────────┐                   │
│   │ Layer Norm      │                   │
│   └────────┬────────┘                   │
│            │                            │
│     ┌──────┴──────┐                     │
│     │             │                     │
│     ▼             │                     │
│   ┌─────────────┐ │                     │
│   │ Feed Forward│ │                     │
│   │ Network     │ │                     │
│   └──────┬──────┘ │                     │
│          │        │                     │
│          ▼        │                     │
│        [ADD] ◄────┘  (Residual)         │
│          │                              │
│          ▼                              │
│       Output                            │
│                                         │
└─────────────────────────────────────────┘
```

### GPT Architecture (Decoder-Only)

GPT uses only the decoder part of the original transformer, with **causal masking**:

```
Causal Mask (can only attend to past tokens):

Token:    [The]  [dog]  [sat]  [on]  [the]  [mat]
           ↓      ↓      ↓      ↓      ↓      ↓
[The]      ✓      ✗      ✗      ✗      ✗      ✗
[dog]      ✓      ✓      ✗      ✗      ✗      ✗
[sat]      ✓      ✓      ✓      ✗      ✗      ✗
[on]       ✓      ✓      ✓      ✓      ✗      ✗
[the]      ✓      ✓      ✓      ✓      ✓      ✗
[mat]      ✓      ✓      ✓      ✓      ✓      ✓

✓ = Can attend    ✗ = Cannot attend (masked)
```

This ensures the model can only use past context to predict the next token - crucial for generation.

---

## 4. Architecture Overview

### GroundZero-LM Specifications

We'll build a model with these specifications:

```
┌─────────────────────────────────────────────────────────────┐
│              GROUNDZERO-LM ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Model Configurations:                                      │
│  ┌─────────────┬──────────┬──────────┬──────────┐          │
│  │ Config      │ Small    │ Medium   │ Large    │          │
│  ├─────────────┼──────────┼──────────┼──────────┤          │
│  │ Parameters  │ 10M      │ 50M      │ 125M     │          │
│  │ Layers      │ 6        │ 12       │ 12       │          │
│  │ Heads       │ 6        │ 8        │ 12       │          │
│  │ Embed Dim   │ 384      │ 512      │ 768      │          │
│  │ FFN Dim     │ 1536     │ 2048     │ 3072     │          │
│  │ Vocab Size  │ 32000    │ 32000    │ 32000    │          │
│  │ Context     │ 512      │ 1024     │ 2048     │          │
│  │ Train Time  │ ~1 week  │ ~2 weeks │ ~1 month │          │
│  └─────────────┴──────────┴──────────┴──────────┘          │
│                                                             │
│  Start with SMALL, scale up once working!                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Complete Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                      GROUNDZERO-LM                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input Text: "Why do dogs bark"                                     │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     TOKENIZER (BPE)                          │   │
│  │  "Why do dogs bark" → [2438, 517, 9432, 27643]               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  TOKEN EMBEDDING                             │   │
│  │  [2438, 517, 9432, 27643] → [384-dim vectors]                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              POSITIONAL EMBEDDING (RoPE)                     │   │
│  │  Add position information to each token                      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              TRANSFORMER BLOCKS (×6-12)                      │   │
│  │  ┌─────────────────────────────────────────────────────┐    │   │
│  │  │ Block 1                                              │    │   │
│  │  │  ├── Multi-Head Self-Attention (Causal Masked)      │    │   │
│  │  │  ├── Residual Connection + LayerNorm                │    │   │
│  │  │  ├── Feed-Forward Network (MLP)                     │    │   │
│  │  │  └── Residual Connection + LayerNorm                │    │   │
│  │  └─────────────────────────────────────────────────────┘    │   │
│  │  ┌─────────────────────────────────────────────────────┐    │   │
│  │  │ Block 2                                              │    │   │
│  │  │  └── ... (same structure)                           │    │   │
│  │  └─────────────────────────────────────────────────────┘    │   │
│  │                        ...                                   │   │
│  │  ┌─────────────────────────────────────────────────────┐    │   │
│  │  │ Block N                                              │    │   │
│  │  │  └── ... (same structure)                           │    │   │
│  │  └─────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    FINAL LAYER NORM                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                OUTPUT PROJECTION (LM Head)                   │   │
│  │  [384-dim] → [32000 vocab] → softmax → probabilities         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ▼                                                             │
│  Output: Next token probabilities                                   │
│  → "at" (0.12), "?" (0.08), "loudly" (0.07), ...                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Knowledge-Grounded Extension

What makes GroundZero-LM special - integration with your knowledge graph:

```
┌─────────────────────────────────────────────────────────────────────┐
│              KNOWLEDGE-GROUNDED GENERATION                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Query: "Why do dogs bark?"                                         │
│       │                                                             │
│       ├──────────────────────────┐                                  │
│       │                          │                                  │
│       ▼                          ▼                                  │
│  ┌──────────────┐        ┌──────────────────────┐                  │
│  │ GroundZero-LM│        │ Knowledge Retrieval  │                  │
│  │ (Transformer)│        │ (Your NeuralPipeline)│                  │
│  └──────┬───────┘        └──────────┬───────────┘                  │
│         │                           │                               │
│         │                           ▼                               │
│         │                ┌──────────────────────┐                  │
│         │                │ Retrieved Facts:     │                  │
│         │                │ • dog is_a mammal    │                  │
│         │                │ • dog has behavior   │                  │
│         │                │ • bark is sound      │                  │
│         │                │ • bark caused_by     │                  │
│         │                │   territorial        │                  │
│         │                └──────────┬───────────┘                  │
│         │                           │                               │
│         │    ┌──────────────────────┘                               │
│         │    │                                                      │
│         ▼    ▼                                                      │
│  ┌─────────────────────────────────────────────────┐               │
│  │            KNOWLEDGE FUSION LAYER               │               │
│  │                                                 │               │
│  │  Cross-attention between:                       │               │
│  │  - LM hidden states                             │               │
│  │  - Knowledge graph embeddings (TransE)          │               │
│  │                                                 │               │
│  │  Fusion methods:                                │               │
│  │  1. Concatenate KG context to input             │               │
│  │  2. Cross-attention layers                      │               │
│  │  3. Knowledge-aware output projection           │               │
│  │                                                 │               │
│  └──────────────────────┬──────────────────────────┘               │
│                         │                                           │
│                         ▼                                           │
│  ┌─────────────────────────────────────────────────┐               │
│  │              GROUNDED OUTPUT                    │               │
│  │                                                 │               │
│  │  "Dogs bark primarily for communication and    │               │
│  │   territorial reasons. As mammals, they use    │               │
│  │   vocalizations to alert their pack..."        │               │
│  │                                                 │               │
│  │  [Grounded in: dog→mammal, bark→territorial]   │               │
│  │                                                 │               │
│  └─────────────────────────────────────────────────┘               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Phase 1: Tokenizer

### What is Tokenization?

Tokenization converts text into numbers that the neural network can process:

```
Text: "Hello, how are you?"
      ↓
Tokens: ["Hello", ",", " how", " are", " you", "?"]
      ↓
Token IDs: [15496, 11, 703, 389, 345, 30]
```

### Byte-Pair Encoding (BPE)

BPE is the standard tokenization algorithm. It:
1. Starts with individual characters
2. Iteratively merges the most frequent pairs
3. Builds a vocabulary of subwords

```
Initial: ['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']

Step 1: Most frequent pair = ('l', 'l') → merge to 'ĺl'
        ['h', 'e', 'ĺl', 'o', ' ', 'w', 'o', 'r', 'l', 'd']

Step 2: Most frequent pair = ('h', 'e') → merge to 'hè'
        ['hè', 'ĺl', 'o', ' ', 'w', 'o', 'r', 'l', 'd']

... continue until vocabulary size reached ...

Final vocabulary:
- Common words as single tokens: "the", "and", "is"
- Rare words split into subwords: "un" + "believ" + "able"
- Unknown words as characters: "x" + "y" + "z" + "123"
```

### Implementation: BPE Tokenizer

```python
# File: src/tokenizer.py

"""
Byte-Pair Encoding Tokenizer
Built from scratch for GroundZero-LM
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass
class TokenizerConfig:
    """Configuration for the tokenizer"""
    VocabSize: int = 32000
    MinFrequency: int = 2
    SpecialTokens: Dict[str, int] = None
    
    def __post_init__(self):
        if self.SpecialTokens is None:
            self.SpecialTokens = {
                '<PAD>': 0,
                '<UNK>': 1,
                '<BOS>': 2,  # Beginning of sequence
                '<EOS>': 3,  # End of sequence
                '<SEP>': 4,  # Separator
                '<MASK>': 5, # For masked language modeling
                '<KNOW>': 6, # Knowledge injection token
            }


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer built from scratch.
    
    This is the same algorithm used by GPT-2, GPT-3, etc.
    """
    
    def __init__(self, Config: TokenizerConfig = None):
        """Initialize tokenizer with configuration"""
        self.Config = Config or TokenizerConfig()
        
        # Vocabulary mappings
        self.Token2ID: Dict[str, int] = {}
        self.ID2Token: Dict[int, str] = {}
        
        # BPE merges (ordered)
        self.Merges: List[Tuple[str, str]] = []
        self.MergeRanks: Dict[Tuple[str, str], int] = {}
        
        # Initialize with special tokens
        for token, idx in self.Config.SpecialTokens.items():
            self.Token2ID[token] = idx
            self.ID2Token[idx] = token
        
        # Regex pattern for pre-tokenization
        # This splits on whitespace while keeping it attached to the following word
        self.Pattern = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+"""
        )
        
        self.IsTrained = False
    
    def _GetStats(self, Vocab: Dict[str, int]) -> Counter:
        """
        Count frequency of adjacent pairs in vocabulary.
        
        Args:
            Vocab: Dictionary of word -> frequency
            
        Returns:
            Counter of (char1, char2) -> frequency
        """
        Pairs = Counter()
        for word, freq in Vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                Pairs[(symbols[i], symbols[i + 1])] += freq
        return Pairs
    
    def _MergePair(
        self, 
        Pair: Tuple[str, str], 
        Vocab: Dict[str, int]
    ) -> Dict[str, int]:
        """
        Merge all occurrences of a pair in the vocabulary.
        
        Args:
            Pair: The pair to merge (char1, char2)
            Vocab: Current vocabulary
            
        Returns:
            Updated vocabulary with pair merged
        """
        NewVocab = {}
        Bigram = ' '.join(Pair)
        Replacement = ''.join(Pair)
        
        for word, freq in Vocab.items():
            # Replace the pair with merged version
            NewWord = word.replace(Bigram, Replacement)
            NewVocab[NewWord] = freq
        
        return NewVocab
    
    def _PreTokenize(self, Text: str) -> List[str]:
        """
        Split text into initial tokens (words/subwords).
        
        Args:
            Text: Input text
            
        Returns:
            List of pre-tokens
        """
        return self.Pattern.findall(Text)
    
    def Train(self, Texts: List[str], Verbose: bool = True):
        """
        Train the BPE tokenizer on a corpus.
        
        Args:
            Texts: List of training texts
            Verbose: Whether to print progress
        """
        if Verbose:
            print("🔤 Training BPE Tokenizer...")
            print(f"   Target vocabulary size: {self.Config.VocabSize}")
        
        # Step 1: Pre-tokenize and count word frequencies
        WordFreqs = Counter()
        for text in Texts:
            tokens = self._PreTokenize(text)
            for token in tokens:
                # Convert to space-separated characters
                chars = ' '.join(list(token))
                WordFreqs[chars] += 1
        
        if Verbose:
            print(f"   Unique words: {len(WordFreqs)}")
        
        # Step 2: Initialize vocabulary with characters
        Vocab = dict(WordFreqs)
        
        # Get all unique characters
        AllChars = set()
        for word in Vocab.keys():
            AllChars.update(word.split())
        
        # Add characters to vocabulary
        NextID = len(self.Config.SpecialTokens)
        for char in sorted(AllChars):
            if char not in self.Token2ID:
                self.Token2ID[char] = NextID
                self.ID2Token[NextID] = char
                NextID += 1
        
        if Verbose:
            print(f"   Initial vocabulary: {NextID} tokens")
        
        # Step 3: Iteratively merge most frequent pairs
        NumMerges = self.Config.VocabSize - NextID
        
        for i in range(NumMerges):
            # Get pair frequencies
            Pairs = self._GetStats(Vocab)
            
            if not Pairs:
                break
            
            # Find most frequent pair
            BestPair = max(Pairs, key=Pairs.get)
            
            if Pairs[BestPair] < self.Config.MinFrequency:
                break
            
            # Merge the pair
            Vocab = self._MergePair(BestPair, Vocab)
            
            # Add merged token to vocabulary
            MergedToken = ''.join(BestPair)
            if MergedToken not in self.Token2ID:
                self.Token2ID[MergedToken] = NextID
                self.ID2Token[NextID] = MergedToken
                NextID += 1
            
            # Record the merge
            self.Merges.append(BestPair)
            self.MergeRanks[BestPair] = len(self.Merges) - 1
            
            if Verbose and (i + 1) % 1000 == 0:
                print(f"   Merges: {i + 1}/{NumMerges}, Vocab: {NextID}")
        
        self.IsTrained = True
        
        if Verbose:
            print(f"   ✅ Training complete!")
            print(f"   Final vocabulary: {len(self.Token2ID)} tokens")
            print(f"   Total merges: {len(self.Merges)}")
    
    def _BPE(self, Token: str) -> List[str]:
        """
        Apply BPE to a single token.
        
        Args:
            Token: Input token string
            
        Returns:
            List of BPE subwords
        """
        if len(Token) == 1:
            return [Token]
        
        # Start with characters
        Word = list(Token)
        
        while len(Word) > 1:
            # Find the highest-priority merge that can be applied
            MinRank = float('inf')
            BestPair = None
            BestIdx = None
            
            for i in range(len(Word) - 1):
                Pair = (Word[i], Word[i + 1])
                if Pair in self.MergeRanks:
                    Rank = self.MergeRanks[Pair]
                    if Rank < MinRank:
                        MinRank = Rank
                        BestPair = Pair
                        BestIdx = i
            
            if BestPair is None:
                break
            
            # Apply the merge
            NewWord = Word[:BestIdx] + [''.join(BestPair)] + Word[BestIdx + 2:]
            Word = NewWord
        
        return Word
    
    def Encode(self, Text: str) -> List[int]:
        """
        Encode text into token IDs.
        
        Args:
            Text: Input text
            
        Returns:
            List of token IDs
        """
        if not self.IsTrained:
            raise ValueError("Tokenizer must be trained first!")
        
        # Pre-tokenize
        Tokens = self._PreTokenize(Text)
        
        # Apply BPE to each token
        IDs = []
        for token in Tokens:
            subwords = self._BPE(token)
            for subword in subwords:
                if subword in self.Token2ID:
                    IDs.append(self.Token2ID[subword])
                else:
                    IDs.append(self.Token2ID['<UNK>'])
        
        return IDs
    
    def Decode(self, IDs: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            IDs: List of token IDs
            
        Returns:
            Decoded text string
        """
        Tokens = []
        for idx in IDs:
            if idx in self.ID2Token:
                token = self.ID2Token[idx]
                if token not in self.Config.SpecialTokens:
                    Tokens.append(token)
        
        return ''.join(Tokens)
    
    def EncodeWithSpecial(
        self, 
        Text: str, 
        AddBOS: bool = True, 
        AddEOS: bool = True
    ) -> List[int]:
        """
        Encode with special tokens added.
        
        Args:
            Text: Input text
            AddBOS: Add beginning-of-sequence token
            AddEOS: Add end-of-sequence token
            
        Returns:
            Token IDs with special tokens
        """
        IDs = self.Encode(Text)
        
        if AddBOS:
            IDs = [self.Token2ID['<BOS>']] + IDs
        if AddEOS:
            IDs = IDs + [self.Token2ID['<EOS>']]
        
        return IDs
    
    def Save(self, Path: str):
        """Save tokenizer to file"""
        data = {
            'config': {
                'VocabSize': self.Config.VocabSize,
                'MinFrequency': self.Config.MinFrequency,
                'SpecialTokens': self.Config.SpecialTokens,
            },
            'token2id': self.Token2ID,
            'merges': self.Merges,
        }
        
        with open(Path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def Load(self, Path: str):
        """Load tokenizer from file"""
        with open(Path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.Config = TokenizerConfig(**data['config'])
        self.Token2ID = data['token2id']
        self.ID2Token = {int(v): k for k, v in self.Token2ID.items()}
        self.Merges = [tuple(m) for m in data['merges']]
        self.MergeRanks = {m: i for i, m in enumerate(self.Merges)}
        self.IsTrained = True
    
    @property
    def VocabSize(self) -> int:
        """Get vocabulary size"""
        return len(self.Token2ID)
    
    def __len__(self) -> int:
        return self.VocabSize


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Test the tokenizer
    print("Testing BPE Tokenizer...\n")
    
    # Sample training data
    training_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks can learn complex patterns from data.",
        "The transformer architecture revolutionized NLP.",
        "Attention is all you need for sequence modeling.",
    ] * 100  # Repeat for more training data
    
    # Train tokenizer
    tokenizer = BPETokenizer(TokenizerConfig(VocabSize=1000))
    tokenizer.Train(training_texts)
    
    # Test encoding/decoding
    test_text = "The neural network learns patterns."
    encoded = tokenizer.Encode(test_text)
    decoded = tokenizer.Decode(encoded)
    
    print(f"Original:  {test_text}")
    print(f"Encoded:   {encoded}")
    print(f"Decoded:   {decoded}")
    print(f"\nVocabulary size: {tokenizer.VocabSize}")
```

### Training Your Tokenizer

```python
# Train on your Wikipedia corpus
from src.tokenizer import BPETokenizer, TokenizerConfig

# Load all text from your auto_learner's collected articles
texts = load_all_training_texts()  # Your implementation

# Configure tokenizer
config = TokenizerConfig(
    VocabSize=32000,      # Standard size
    MinFrequency=2,       # Minimum pair frequency
)

# Train
tokenizer = BPETokenizer(config)
tokenizer.Train(texts, Verbose=True)

# Save for later use
tokenizer.Save("data/tokenizer.json")
```

---

## 6. Phase 2: Embeddings

### Token Embeddings

Convert token IDs to dense vectors:

```python
# File: src/embeddings.py

"""
Embedding layers for GroundZero-LM
"""

import math
from typing import List, Optional
import random


class TokenEmbedding:
    """
    Learnable token embeddings.
    Maps vocabulary indices to dense vectors.
    """
    
    def __init__(self, VocabSize: int, EmbedDim: int):
        """
        Initialize token embeddings.
        
        Args:
            VocabSize: Size of vocabulary
            EmbedDim: Dimension of embeddings
        """
        self.VocabSize = VocabSize
        self.EmbedDim = EmbedDim
        
        # Initialize embeddings with small random values
        # Using Xavier/Glorot initialization
        scale = math.sqrt(2.0 / (VocabSize + EmbedDim))
        self.Weight: List[List[float]] = [
            [random.gauss(0, scale) for _ in range(EmbedDim)]
            for _ in range(VocabSize)
        ]
    
    def Forward(self, TokenIDs: List[int]) -> List[List[float]]:
        """
        Look up embeddings for token IDs.
        
        Args:
            TokenIDs: List of token indices
            
        Returns:
            List of embedding vectors
        """
        return [self.Weight[idx] for idx in TokenIDs]
    
    def __call__(self, TokenIDs: List[int]) -> List[List[float]]:
        return self.Forward(TokenIDs)


class PositionalEmbedding:
    """
    Sinusoidal positional embeddings (original Transformer).
    Encodes position information into the sequence.
    """
    
    def __init__(self, MaxLen: int, EmbedDim: int):
        """
        Initialize positional embeddings.
        
        Args:
            MaxLen: Maximum sequence length
            EmbedDim: Dimension of embeddings
        """
        self.MaxLen = MaxLen
        self.EmbedDim = EmbedDim
        
        # Pre-compute positional encodings
        self.Encodings: List[List[float]] = []
        
        for pos in range(MaxLen):
            encoding = []
            for i in range(EmbedDim):
                if i % 2 == 0:
                    # Even indices: sin
                    encoding.append(
                        math.sin(pos / (10000 ** (i / EmbedDim)))
                    )
                else:
                    # Odd indices: cos
                    encoding.append(
                        math.cos(pos / (10000 ** ((i - 1) / EmbedDim)))
                    )
            self.Encodings.append(encoding)
    
    def Forward(self, SeqLen: int) -> List[List[float]]:
        """
        Get positional encodings for a sequence.
        
        Args:
            SeqLen: Length of sequence
            
        Returns:
            Positional encodings [SeqLen, EmbedDim]
        """
        return self.Encodings[:SeqLen]
    
    def __call__(self, SeqLen: int) -> List[List[float]]:
        return self.Forward(SeqLen)


class RoPEEmbedding:
    """
    Rotary Position Embedding (RoPE) - used in modern LLMs like LLaMA.
    
    Instead of adding position info, RoPE rotates the query and key
    vectors based on their position. This has better length generalization.
    """
    
    def __init__(self, HeadDim: int, MaxLen: int = 4096, Base: float = 10000.0):
        """
        Initialize RoPE.
        
        Args:
            HeadDim: Dimension per attention head
            MaxLen: Maximum sequence length
            Base: Base for frequency computation
        """
        self.HeadDim = HeadDim
        self.MaxLen = MaxLen
        self.Base = Base
        
        # Compute frequency bands
        # For dimension i, freq = 1 / (base ^ (2i / dim))
        self.Frequencies: List[float] = []
        for i in range(0, HeadDim, 2):
            freq = 1.0 / (Base ** (i / HeadDim))
            self.Frequencies.append(freq)
        
        # Pre-compute sin/cos for all positions
        self.SinCache: List[List[float]] = []
        self.CosCache: List[List[float]] = []
        
        for pos in range(MaxLen):
            sins = []
            coss = []
            for freq in self.Frequencies:
                angle = pos * freq
                sins.append(math.sin(angle))
                coss.append(math.cos(angle))
            self.SinCache.append(sins)
            self.CosCache.append(coss)
    
    def Apply(
        self, 
        X: List[List[float]], 
        StartPos: int = 0
    ) -> List[List[float]]:
        """
        Apply rotary embeddings to input.
        
        Args:
            X: Input tensor [SeqLen, HeadDim]
            StartPos: Starting position (for caching during generation)
            
        Returns:
            Rotated tensor [SeqLen, HeadDim]
        """
        SeqLen = len(X)
        Result = []
        
        for i, vec in enumerate(X):
            pos = StartPos + i
            sin = self.SinCache[pos]
            cos = self.CosCache[pos]
            
            rotated = []
            for j in range(0, self.HeadDim, 2):
                # Apply rotation to pairs of dimensions
                x0 = vec[j]
                x1 = vec[j + 1] if j + 1 < self.HeadDim else 0
                
                freq_idx = j // 2
                c = cos[freq_idx]
                s = sin[freq_idx]
                
                # Rotation: [cos -sin] [x0]
                #           [sin  cos] [x1]
                rotated.append(x0 * c - x1 * s)
                if j + 1 < self.HeadDim:
                    rotated.append(x0 * s + x1 * c)
            
            Result.append(rotated)
        
        return Result
    
    def __call__(
        self, 
        X: List[List[float]], 
        StartPos: int = 0
    ) -> List[List[float]]:
        return self.Apply(X, StartPos)


class CombinedEmbedding:
    """
    Combined token + positional embeddings.
    This is the input layer of the transformer.
    """
    
    def __init__(
        self, 
        VocabSize: int, 
        EmbedDim: int, 
        MaxLen: int,
        DropoutRate: float = 0.1,
        UseRoPE: bool = True
    ):
        """
        Initialize combined embeddings.
        
        Args:
            VocabSize: Vocabulary size
            EmbedDim: Embedding dimension
            MaxLen: Maximum sequence length
            DropoutRate: Dropout rate
            UseRoPE: Whether to use RoPE (vs sinusoidal)
        """
        self.TokenEmbed = TokenEmbedding(VocabSize, EmbedDim)
        self.UseRoPE = UseRoPE
        
        if not UseRoPE:
            self.PosEmbed = PositionalEmbedding(MaxLen, EmbedDim)
        
        self.DropoutRate = DropoutRate
        self.EmbedDim = EmbedDim
        self.Scale = math.sqrt(EmbedDim)
    
    def Forward(
        self, 
        TokenIDs: List[int], 
        Training: bool = True
    ) -> List[List[float]]:
        """
        Get combined embeddings for tokens.
        
        Args:
            TokenIDs: Input token IDs
            Training: Whether in training mode (for dropout)
            
        Returns:
            Combined embeddings [SeqLen, EmbedDim]
        """
        # Get token embeddings and scale
        X = self.TokenEmbed(TokenIDs)
        X = [[val * self.Scale for val in vec] for vec in X]
        
        # Add positional embeddings (if not using RoPE)
        if not self.UseRoPE:
            PosEnc = self.PosEmbed(len(TokenIDs))
            X = [
                [x + p for x, p in zip(tok, pos)]
                for tok, pos in zip(X, PosEnc)
            ]
        
        # Apply dropout during training
        if Training and self.DropoutRate > 0:
            X = self._Dropout(X)
        
        return X
    
    def _Dropout(self, X: List[List[float]]) -> List[List[float]]:
        """Apply dropout"""
        Result = []
        scale = 1.0 / (1.0 - self.DropoutRate)
        
        for vec in X:
            new_vec = []
            for val in vec:
                if random.random() > self.DropoutRate:
                    new_vec.append(val * scale)
                else:
                    new_vec.append(0.0)
            Result.append(new_vec)
        
        return Result
    
    def __call__(
        self, 
        TokenIDs: List[int], 
        Training: bool = True
    ) -> List[List[float]]:
        return self.Forward(TokenIDs, Training)
```

---

## 7. Phase 3: Attention Mechanism

### Self-Attention Implementation

```python
# File: src/attention.py

"""
Multi-Head Self-Attention for GroundZero-LM
"""

import math
from typing import List, Tuple, Optional


class MultiHeadAttention:
    """
    Multi-Head Self-Attention mechanism.
    
    This is the core of the transformer - allows each position
    to attend to all other positions in the sequence.
    """
    
    def __init__(
        self, 
        EmbedDim: int, 
        NumHeads: int, 
        DropoutRate: float = 0.1,
        UseCausalMask: bool = True
    ):
        """
        Initialize multi-head attention.
        
        Args:
            EmbedDim: Model dimension
            NumHeads: Number of attention heads
            DropoutRate: Attention dropout rate
            UseCausalMask: Whether to use causal (autoregressive) masking
        """
        assert EmbedDim % NumHeads == 0, "EmbedDim must be divisible by NumHeads"
        
        self.EmbedDim = EmbedDim
        self.NumHeads = NumHeads
        self.HeadDim = EmbedDim // NumHeads
        self.Scale = math.sqrt(self.HeadDim)
        self.DropoutRate = DropoutRate
        self.UseCausalMask = UseCausalMask
        
        # Initialize projection weights
        # Q, K, V projections (combined for efficiency)
        self.Wq = self._InitWeight(EmbedDim, EmbedDim)
        self.Wk = self._InitWeight(EmbedDim, EmbedDim)
        self.Wv = self._InitWeight(EmbedDim, EmbedDim)
        
        # Output projection
        self.Wo = self._InitWeight(EmbedDim, EmbedDim)
        
        # Biases (optional, can be None)
        self.Bq = [0.0] * EmbedDim
        self.Bk = [0.0] * EmbedDim
        self.Bv = [0.0] * EmbedDim
        self.Bo = [0.0] * EmbedDim
    
    def _InitWeight(self, OutDim: int, InDim: int) -> List[List[float]]:
        """Initialize weight matrix with Xavier initialization"""
        import random
        scale = math.sqrt(2.0 / (InDim + OutDim))
        return [
            [random.gauss(0, scale) for _ in range(InDim)]
            for _ in range(OutDim)
        ]
    
    def _MatMul(
        self, 
        A: List[List[float]], 
        B: List[List[float]]
    ) -> List[List[float]]:
        """Matrix multiplication A @ B"""
        RowsA = len(A)
        ColsA = len(A[0])
        ColsB = len(B[0])
        
        Result = []
        for i in range(RowsA):
            Row = []
            for j in range(ColsB):
                Val = sum(A[i][k] * B[k][j] for k in range(ColsA))
                Row.append(Val)
            Result.append(Row)
        
        return Result
    
    def _MatVecMul(
        self, 
        M: List[List[float]], 
        V: List[float]
    ) -> List[float]:
        """Matrix-vector multiplication"""
        return [sum(m * v for m, v in zip(row, V)) for row in M]
    
    def _Transpose(self, M: List[List[float]]) -> List[List[float]]:
        """Transpose a matrix"""
        Rows = len(M)
        Cols = len(M[0])
        return [[M[i][j] for i in range(Rows)] for j in range(Cols)]
    
    def _Softmax(self, X: List[float], Mask: List[float] = None) -> List[float]:
        """Softmax with optional masking"""
        if Mask is not None:
            X = [x + m for x, m in zip(X, Mask)]
        
        MaxVal = max(X)
        Exp = [math.exp(x - MaxVal) for x in X]
        SumExp = sum(Exp)
        return [e / SumExp for e in Exp]
    
    def _CreateCausalMask(self, SeqLen: int) -> List[List[float]]:
        """
        Create causal attention mask.
        
        Returns a matrix where position i can only attend to positions <= i.
        Masked positions are set to -inf (becomes 0 after softmax).
        """
        Mask = []
        NegInf = -1e9  # Large negative number (effectively -inf)
        
        for i in range(SeqLen):
            Row = []
            for j in range(SeqLen):
                if j <= i:
                    Row.append(0.0)  # Can attend
                else:
                    Row.append(NegInf)  # Cannot attend
            Mask.append(Row)
        
        return Mask
    
    def _SplitHeads(
        self, 
        X: List[List[float]]
    ) -> List[List[List[float]]]:
        """
        Split embedding into multiple heads.
        
        Args:
            X: [SeqLen, EmbedDim]
            
        Returns:
            [NumHeads, SeqLen, HeadDim]
        """
        SeqLen = len(X)
        Heads = []
        
        for h in range(self.NumHeads):
            Head = []
            for seq in range(SeqLen):
                Start = h * self.HeadDim
                End = Start + self.HeadDim
                Head.append(X[seq][Start:End])
            Heads.append(Head)
        
        return Heads
    
    def _MergeHeads(
        self, 
        Heads: List[List[List[float]]]
    ) -> List[List[float]]:
        """
        Merge attention heads back together.
        
        Args:
            Heads: [NumHeads, SeqLen, HeadDim]
            
        Returns:
            [SeqLen, EmbedDim]
        """
        SeqLen = len(Heads[0])
        Result = []
        
        for seq in range(SeqLen):
            Merged = []
            for h in range(self.NumHeads):
                Merged.extend(Heads[h][seq])
            Result.append(Merged)
        
        return Result
    
    def _LinearProject(
        self, 
        X: List[List[float]], 
        W: List[List[float]], 
        B: List[float]
    ) -> List[List[float]]:
        """Apply linear projection: X @ W^T + B"""
        Result = []
        for vec in X:
            projected = self._MatVecMul(W, vec)
            projected = [p + b for p, b in zip(projected, B)]
            Result.append(projected)
        return Result
    
    def Forward(
        self, 
        X: List[List[float]], 
        Training: bool = True
    ) -> Tuple[List[List[float]], List[List[List[float]]]]:
        """
        Forward pass of multi-head attention.
        
        Args:
            X: Input tensor [SeqLen, EmbedDim]
            Training: Whether in training mode
            
        Returns:
            Tuple of (output, attention_weights)
        """
        SeqLen = len(X)
        
        # Project to Q, K, V
        Q = self._LinearProject(X, self.Wq, self.Bq)
        K = self._LinearProject(X, self.Wk, self.Bk)
        V = self._LinearProject(X, self.Wv, self.Bv)
        
        # Split into heads
        Q_heads = self._SplitHeads(Q)  # [NumHeads, SeqLen, HeadDim]
        K_heads = self._SplitHeads(K)
        V_heads = self._SplitHeads(V)
        
        # Create causal mask if needed
        CausalMask = None
        if self.UseCausalMask:
            CausalMask = self._CreateCausalMask(SeqLen)
        
        # Compute attention for each head
        AttnOutputs = []
        AttnWeights = []
        
        for h in range(self.NumHeads):
            Q_h = Q_heads[h]  # [SeqLen, HeadDim]
            K_h = K_heads[h]
            V_h = V_heads[h]
            
            # Compute attention scores: Q @ K^T / sqrt(d)
            K_t = self._Transpose(K_h)  # [HeadDim, SeqLen]
            Scores = self._MatMul(Q_h, K_t)  # [SeqLen, SeqLen]
            Scores = [[s / self.Scale for s in row] for row in Scores]
            
            # Apply softmax (with mask)
            AttnProbs = []
            for i, row in enumerate(Scores):
                Mask = CausalMask[i] if CausalMask else None
                AttnProbs.append(self._Softmax(row, Mask))
            
            # Apply attention to values: Attn @ V
            Output_h = self._MatMul(AttnProbs, V_h)  # [SeqLen, HeadDim]
            
            AttnOutputs.append(Output_h)
            AttnWeights.append(AttnProbs)
        
        # Merge heads
        Merged = self._MergeHeads(AttnOutputs)  # [SeqLen, EmbedDim]
        
        # Output projection
        Output = self._LinearProject(Merged, self.Wo, self.Bo)
        
        return Output, AttnWeights
    
    def __call__(
        self, 
        X: List[List[float]], 
        Training: bool = True
    ) -> Tuple[List[List[float]], List[List[List[float]]]]:
        return self.Forward(X, Training)


class CrossAttention(MultiHeadAttention):
    """
    Cross-attention for attending to external context.
    Used for knowledge-grounded generation.
    """
    
    def ForwardCross(
        self,
        X: List[List[float]],      # Query source
        Context: List[List[float]], # Key/Value source (knowledge)
        Training: bool = True
    ) -> Tuple[List[List[float]], List[List[List[float]]]]:
        """
        Cross-attention: X attends to Context.
        
        Args:
            X: Query tensor [SeqLen, EmbedDim]
            Context: Context tensor [ContextLen, EmbedDim]
            Training: Whether in training mode
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Q from X, K and V from Context
        Q = self._LinearProject(X, self.Wq, self.Bq)
        K = self._LinearProject(Context, self.Wk, self.Bk)
        V = self._LinearProject(Context, self.Wv, self.Bv)
        
        # Split into heads
        Q_heads = self._SplitHeads(Q)
        K_heads = self._SplitHeads(K)
        V_heads = self._SplitHeads(V)
        
        # No causal mask for cross-attention (can attend to all context)
        
        # Compute attention for each head
        AttnOutputs = []
        AttnWeights = []
        
        for h in range(self.NumHeads):
            Q_h = Q_heads[h]
            K_h = K_heads[h]
            V_h = V_heads[h]
            
            # Attention scores
            K_t = self._Transpose(K_h)
            Scores = self._MatMul(Q_h, K_t)
            Scores = [[s / self.Scale for s in row] for row in Scores]
            
            # Softmax (no mask)
            AttnProbs = [self._Softmax(row) for row in Scores]
            
            # Apply to values
            Output_h = self._MatMul(AttnProbs, V_h)
            
            AttnOutputs.append(Output_h)
            AttnWeights.append(AttnProbs)
        
        # Merge and project
        Merged = self._MergeHeads(AttnOutputs)
        Output = self._LinearProject(Merged, self.Wo, self.Bo)
        
        return Output, AttnWeights
```

---

## 8. Phase 4: Transformer Block

### Feed-Forward Network

```python
# File: src/feedforward.py

"""
Feed-Forward Network (MLP) for Transformer
"""

import math
from typing import List
import random


class FeedForward:
    """
    Position-wise Feed-Forward Network.
    
    FFN(x) = GELU(xW1 + b1)W2 + b2
    
    This is applied identically to each position.
    """
    
    def __init__(
        self, 
        EmbedDim: int, 
        FFNDim: int = None, 
        DropoutRate: float = 0.1
    ):
        """
        Initialize feed-forward network.
        
        Args:
            EmbedDim: Model dimension
            FFNDim: Hidden dimension (default: 4 * EmbedDim)
            DropoutRate: Dropout rate
        """
        self.EmbedDim = EmbedDim
        self.FFNDim = FFNDim or 4 * EmbedDim
        self.DropoutRate = DropoutRate
        
        # Initialize weights
        self.W1 = self._InitWeight(self.FFNDim, EmbedDim)
        self.B1 = [0.0] * self.FFNDim
        
        self.W2 = self._InitWeight(EmbedDim, self.FFNDim)
        self.B2 = [0.0] * EmbedDim
    
    def _InitWeight(self, OutDim: int, InDim: int) -> List[List[float]]:
        """Xavier initialization"""
        scale = math.sqrt(2.0 / (InDim + OutDim))
        return [
            [random.gauss(0, scale) for _ in range(InDim)]
            for _ in range(OutDim)
        ]
    
    def _MatVecMul(self, M: List[List[float]], V: List[float]) -> List[float]:
        """Matrix-vector multiplication"""
        return [sum(m * v for m, v in zip(row, V)) for row in M]
    
    def _GELU(self, X: float) -> float:
        """
        Gaussian Error Linear Unit activation.
        Smoother than ReLU, used in modern transformers.
        
        GELU(x) = x * Φ(x) where Φ is the CDF of standard normal
        """
        # Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        return 0.5 * X * (1 + math.tanh(
            math.sqrt(2 / math.pi) * (X + 0.044715 * X ** 3)
        ))
    
    def _SiLU(self, X: float) -> float:
        """
        Sigmoid Linear Unit (SiLU/Swish) - alternative activation.
        SiLU(x) = x * sigmoid(x)
        """
        return X / (1 + math.exp(-X))
    
    def Forward(
        self, 
        X: List[List[float]], 
        Training: bool = True
    ) -> List[List[float]]:
        """
        Forward pass.
        
        Args:
            X: Input [SeqLen, EmbedDim]
            Training: Whether in training mode
            
        Returns:
            Output [SeqLen, EmbedDim]
        """
        Result = []
        
        for vec in X:
            # First linear: [EmbedDim] -> [FFNDim]
            hidden = self._MatVecMul(self.W1, vec)
            hidden = [h + b for h, b in zip(hidden, self.B1)]
            
            # Activation (GELU)
            hidden = [self._GELU(h) for h in hidden]
            
            # Second linear: [FFNDim] -> [EmbedDim]
            output = self._MatVecMul(self.W2, hidden)
            output = [o + b for o, b in zip(output, self.B2)]
            
            Result.append(output)
        
        return Result
    
    def __call__(
        self, 
        X: List[List[float]], 
        Training: bool = True
    ) -> List[List[float]]:
        return self.Forward(X, Training)


class SwiGLU(FeedForward):
    """
    SwiGLU activation - used in LLaMA and other modern LLMs.
    
    SwiGLU(x) = SiLU(xW1) ⊙ (xV)
    
    Has a "gate" that learns what information to pass through.
    """
    
    def __init__(
        self, 
        EmbedDim: int, 
        FFNDim: int = None, 
        DropoutRate: float = 0.1
    ):
        super().__init__(EmbedDim, FFNDim, DropoutRate)
        
        # Additional gate projection
        self.Wgate = self._InitWeight(self.FFNDim, EmbedDim)
        self.Bgate = [0.0] * self.FFNDim
    
    def Forward(
        self, 
        X: List[List[float]], 
        Training: bool = True
    ) -> List[List[float]]:
        """SwiGLU forward pass"""
        Result = []
        
        for vec in X:
            # First projection (with SiLU activation)
            hidden = self._MatVecMul(self.W1, vec)
            hidden = [h + b for h, b in zip(hidden, self.B1)]
            hidden = [self._SiLU(h) for h in hidden]
            
            # Gate projection (no activation)
            gate = self._MatVecMul(self.Wgate, vec)
            gate = [g + b for g, b in zip(gate, self.Bgate)]
            
            # Element-wise multiplication (gating)
            gated = [h * g for h, g in zip(hidden, gate)]
            
            # Output projection
            output = self._MatVecMul(self.W2, gated)
            output = [o + b for o, b in zip(output, self.B2)]
            
            Result.append(output)
        
        return Result
```

### Layer Normalization

```python
# File: src/layernorm.py

"""
Layer Normalization for Transformer
"""

import math
from typing import List


class LayerNorm:
    """
    Layer Normalization.
    
    Normalizes across the feature dimension, not the batch dimension.
    LN(x) = γ * (x - μ) / (σ + ε) + β
    """
    
    def __init__(self, Dim: int, Eps: float = 1e-6):
        """
        Initialize layer norm.
        
        Args:
            Dim: Feature dimension
            Eps: Small constant for numerical stability
        """
        self.Dim = Dim
        self.Eps = Eps
        
        # Learnable parameters
        self.Gamma = [1.0] * Dim  # Scale
        self.Beta = [0.0] * Dim   # Shift
    
    def Forward(self, X: List[List[float]]) -> List[List[float]]:
        """
        Apply layer normalization.
        
        Args:
            X: Input [SeqLen, Dim]
            
        Returns:
            Normalized output [SeqLen, Dim]
        """
        Result = []
        
        for vec in X:
            # Compute mean
            mean = sum(vec) / len(vec)
            
            # Compute variance
            var = sum((v - mean) ** 2 for v in vec) / len(vec)
            
            # Normalize
            std = math.sqrt(var + self.Eps)
            normalized = [(v - mean) / std for v in vec]
            
            # Scale and shift
            output = [
                g * n + b 
                for g, n, b in zip(self.Gamma, normalized, self.Beta)
            ]
            
            Result.append(output)
        
        return Result
    
    def __call__(self, X: List[List[float]]) -> List[List[float]]:
        return self.Forward(X)


class RMSNorm:
    """
    Root Mean Square Layer Normalization.
    
    Simpler and slightly faster than LayerNorm.
    Used in LLaMA and other modern LLMs.
    
    RMSNorm(x) = x / RMS(x) * γ
    where RMS(x) = sqrt(mean(x^2) + ε)
    """
    
    def __init__(self, Dim: int, Eps: float = 1e-6):
        """
        Initialize RMS norm.
        
        Args:
            Dim: Feature dimension
            Eps: Numerical stability constant
        """
        self.Dim = Dim
        self.Eps = Eps
        self.Gamma = [1.0] * Dim  # Scale only, no shift
    
    def Forward(self, X: List[List[float]]) -> List[List[float]]:
        """Apply RMS normalization"""
        Result = []
        
        for vec in X:
            # Compute RMS
            mean_sq = sum(v ** 2 for v in vec) / len(vec)
            rms = math.sqrt(mean_sq + self.Eps)
            
            # Normalize and scale
            output = [g * v / rms for g, v in zip(self.Gamma, vec)]
            
            Result.append(output)
        
        return Result
    
    def __call__(self, X: List[List[float]]) -> List[List[float]]:
        return self.Forward(X)
```

### Complete Transformer Block

```python
# File: src/transformer_block.py

"""
Complete Transformer Block for GroundZero-LM
"""

from typing import List, Tuple, Optional
from .attention import MultiHeadAttention, CrossAttention
from .feedforward import FeedForward, SwiGLU
from .layernorm import LayerNorm, RMSNorm


class TransformerBlock:
    """
    Single Transformer block (decoder-only architecture).
    
    Pre-norm architecture (more stable):
    x = x + Attention(LayerNorm(x))
    x = x + FFN(LayerNorm(x))
    """
    
    def __init__(
        self,
        EmbedDim: int,
        NumHeads: int,
        FFNDim: int = None,
        DropoutRate: float = 0.1,
        UseRMSNorm: bool = True,
        UseSwiGLU: bool = True
    ):
        """
        Initialize transformer block.
        
        Args:
            EmbedDim: Model dimension
            NumHeads: Number of attention heads
            FFNDim: FFN hidden dimension
            DropoutRate: Dropout rate
            UseRMSNorm: Use RMSNorm vs LayerNorm
            UseSwiGLU: Use SwiGLU vs standard FFN
        """
        self.EmbedDim = EmbedDim
        
        # Normalization layers
        NormClass = RMSNorm if UseRMSNorm else LayerNorm
        self.AttnNorm = NormClass(EmbedDim)
        self.FFNNorm = NormClass(EmbedDim)
        
        # Self-attention
        self.Attention = MultiHeadAttention(
            EmbedDim, NumHeads, DropoutRate, UseCausalMask=True
        )
        
        # Feed-forward network
        FFNClass = SwiGLU if UseSwiGLU else FeedForward
        self.FFN = FFNClass(EmbedDim, FFNDim, DropoutRate)
    
    def _AddVectors(
        self, 
        A: List[List[float]], 
        B: List[List[float]]
    ) -> List[List[float]]:
        """Element-wise addition of two matrices"""
        return [
            [a + b for a, b in zip(row_a, row_b)]
            for row_a, row_b in zip(A, B)
        ]
    
    def Forward(
        self, 
        X: List[List[float]], 
        Training: bool = True
    ) -> Tuple[List[List[float]], List[List[List[float]]]]:
        """
        Forward pass through transformer block.
        
        Args:
            X: Input [SeqLen, EmbedDim]
            Training: Whether in training mode
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Self-attention with residual
        Normalized = self.AttnNorm(X)
        AttnOut, AttnWeights = self.Attention(Normalized, Training)
        X = self._AddVectors(X, AttnOut)
        
        # FFN with residual
        Normalized = self.FFNNorm(X)
        FFNOut = self.FFN(Normalized, Training)
        X = self._AddVectors(X, FFNOut)
        
        return X, AttnWeights
    
    def __call__(
        self, 
        X: List[List[float]], 
        Training: bool = True
    ) -> Tuple[List[List[float]], List[List[List[float]]]]:
        return self.Forward(X, Training)


class KnowledgeAwareBlock(TransformerBlock):
    """
    Transformer block with cross-attention to knowledge.
    
    This allows the model to attend to retrieved knowledge
    from your GroundZero knowledge graph.
    """
    
    def __init__(
        self,
        EmbedDim: int,
        NumHeads: int,
        FFNDim: int = None,
        DropoutRate: float = 0.1,
        UseRMSNorm: bool = True,
        UseSwiGLU: bool = True
    ):
        super().__init__(
            EmbedDim, NumHeads, FFNDim, 
            DropoutRate, UseRMSNorm, UseSwiGLU
        )
        
        # Additional cross-attention for knowledge
        NormClass = RMSNorm if UseRMSNorm else LayerNorm
        self.KnowledgeNorm = NormClass(EmbedDim)
        self.KnowledgeAttn = CrossAttention(
            EmbedDim, NumHeads, DropoutRate, UseCausalMask=False
        )
    
    def Forward(
        self, 
        X: List[List[float]],
        Knowledge: List[List[float]] = None,
        Training: bool = True
    ) -> Tuple[List[List[float]], List]:
        """
        Forward pass with optional knowledge context.
        
        Args:
            X: Input [SeqLen, EmbedDim]
            Knowledge: Knowledge embeddings [KnowledgeLen, EmbedDim]
            Training: Whether in training mode
            
        Returns:
            Tuple of (output, attention_info)
        """
        AllAttnWeights = []
        
        # Self-attention
        Normalized = self.AttnNorm(X)
        AttnOut, AttnWeights = self.Attention(Normalized, Training)
        X = self._AddVectors(X, AttnOut)
        AllAttnWeights.append(('self', AttnWeights))
        
        # Cross-attention to knowledge (if provided)
        if Knowledge is not None:
            Normalized = self.KnowledgeNorm(X)
            KnowOut, KnowWeights = self.KnowledgeAttn.ForwardCross(
                Normalized, Knowledge, Training
            )
            X = self._AddVectors(X, KnowOut)
            AllAttnWeights.append(('knowledge', KnowWeights))
        
        # FFN
        Normalized = self.FFNNorm(X)
        FFNOut = self.FFN(Normalized, Training)
        X = self._AddVectors(X, FFNOut)
        
        return X, AllAttnWeights
```

---

## 9. Phase 5: Full GPT Model

### Complete Model Architecture

```python
# File: src/gpt_model.py

"""
GroundZero-LM: Complete GPT Model
"""

import math
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import random

from .tokenizer import BPETokenizer
from .embeddings import TokenEmbedding, RoPEEmbedding
from .transformer_block import TransformerBlock, KnowledgeAwareBlock
from .layernorm import RMSNorm


@dataclass
class GPTConfig:
    """Model configuration"""
    VocabSize: int = 32000
    EmbedDim: int = 384
    NumLayers: int = 6
    NumHeads: int = 6
    FFNDim: int = 1536
    MaxSeqLen: int = 512
    DropoutRate: float = 0.1
    UseRoPE: bool = True
    UseRMSNorm: bool = True
    UseSwiGLU: bool = True
    UseKnowledgeAttention: bool = True
    KnowledgeLayerIndices: List[int] = None  # Which layers use knowledge
    
    def __post_init__(self):
        if self.KnowledgeLayerIndices is None:
            # Default: knowledge attention in later layers
            self.KnowledgeLayerIndices = [
                self.NumLayers - 3,
                self.NumLayers - 2,
                self.NumLayers - 1
            ]
    
    @classmethod
    def Small(cls) -> 'GPTConfig':
        """~10M parameters"""
        return cls(
            VocabSize=32000,
            EmbedDim=384,
            NumLayers=6,
            NumHeads=6,
            FFNDim=1536,
            MaxSeqLen=512
        )
    
    @classmethod
    def Medium(cls) -> 'GPTConfig':
        """~50M parameters"""
        return cls(
            VocabSize=32000,
            EmbedDim=512,
            NumLayers=12,
            NumHeads=8,
            FFNDim=2048,
            MaxSeqLen=1024
        )
    
    @classmethod
    def Large(cls) -> 'GPTConfig':
        """~125M parameters"""
        return cls(
            VocabSize=32000,
            EmbedDim=768,
            NumLayers=12,
            NumHeads=12,
            FFNDim=3072,
            MaxSeqLen=2048
        )


class GroundZeroLM:
    """
    GroundZero Language Model.
    
    A GPT-style decoder-only transformer with:
    - BPE tokenization
    - RoPE positional embeddings
    - RMSNorm
    - SwiGLU activation
    - Knowledge-grounded generation (cross-attention to KG)
    """
    
    def __init__(self, Config: GPTConfig):
        """
        Initialize the language model.
        
        Args:
            Config: Model configuration
        """
        self.Config = Config
        
        # Token embeddings
        self.TokenEmbed = TokenEmbedding(Config.VocabSize, Config.EmbedDim)
        
        # Rotary embeddings (applied in attention)
        if Config.UseRoPE:
            self.RoPE = RoPEEmbedding(
                Config.EmbedDim // Config.NumHeads,
                Config.MaxSeqLen
            )
        
        # Transformer blocks
        self.Blocks: List = []
        for i in range(Config.NumLayers):
            if (Config.UseKnowledgeAttention and 
                i in Config.KnowledgeLayerIndices):
                # Knowledge-aware block
                block = KnowledgeAwareBlock(
                    Config.EmbedDim,
                    Config.NumHeads,
                    Config.FFNDim,
                    Config.DropoutRate,
                    Config.UseRMSNorm,
                    Config.UseSwiGLU
                )
            else:
                # Standard block
                block = TransformerBlock(
                    Config.EmbedDim,
                    Config.NumHeads,
                    Config.FFNDim,
                    Config.DropoutRate,
                    Config.UseRMSNorm,
                    Config.UseSwiGLU
                )
            self.Blocks.append(block)
        
        # Final normalization
        NormClass = RMSNorm if Config.UseRMSNorm else LayerNorm
        self.FinalNorm = NormClass(Config.EmbedDim)
        
        # Output projection (LM head)
        # Tied with token embeddings for efficiency
        self.LMHead = None  # Will use TokenEmbed.Weight transposed
        
        # Track training state
        self.TrainingStep = 0
        self.LossHistory: List[float] = []
    
    def _MatVecMul(self, M: List[List[float]], V: List[float]) -> List[float]:
        """Matrix-vector multiplication"""
        return [sum(m * v for m, v in zip(row, V)) for row in M]
    
    def _Softmax(self, X: List[float], Temperature: float = 1.0) -> List[float]:
        """Softmax with temperature"""
        X = [x / Temperature for x in X]
        MaxVal = max(X)
        Exp = [math.exp(x - MaxVal) for x in X]
        SumExp = sum(Exp)
        return [e / SumExp for e in Exp]
    
    def Forward(
        self,
        TokenIDs: List[int],
        Knowledge: List[List[float]] = None,
        Training: bool = True
    ) -> Tuple[List[List[float]], List]:
        """
        Forward pass through the model.
        
        Args:
            TokenIDs: Input token IDs [SeqLen]
            Knowledge: Optional knowledge embeddings [KnowledgeLen, EmbedDim]
            Training: Whether in training mode
            
        Returns:
            Tuple of (logits [SeqLen, VocabSize], attention_info)
        """
        # Get token embeddings
        X = self.TokenEmbed(TokenIDs)  # [SeqLen, EmbedDim]
        
        # Transformer blocks
        AllAttnInfo = []
        for i, block in enumerate(self.Blocks):
            if isinstance(block, KnowledgeAwareBlock):
                X, AttnInfo = block.Forward(X, Knowledge, Training)
            else:
                X, AttnInfo = block.Forward(X, Training)
            AllAttnInfo.append((i, AttnInfo))
        
        # Final normalization
        X = self.FinalNorm(X)
        
        # Project to vocabulary (using token embedding weights)
        # This is "weight tying" - saves parameters and improves performance
        Logits = []
        for vec in X:
            # Compute logits for all vocab tokens
            logit = []
            for token_embed in self.TokenEmbed.Weight:
                # Dot product with each token embedding
                score = sum(v * e for v, e in zip(vec, token_embed))
                logit.append(score)
            Logits.append(logit)
        
        return Logits, AllAttnInfo
    
    def ComputeLoss(
        self,
        Logits: List[List[float]],
        Targets: List[int]
    ) -> float:
        """
        Compute cross-entropy loss.
        
        Args:
            Logits: Model outputs [SeqLen, VocabSize]
            Targets: Target token IDs [SeqLen]
            
        Returns:
            Average cross-entropy loss
        """
        TotalLoss = 0.0
        Count = 0
        
        for logit, target in zip(Logits, Targets):
            # Softmax
            probs = self._Softmax(logit)
            
            # Cross-entropy: -log(prob of correct token)
            prob = max(probs[target], 1e-10)  # Avoid log(0)
            loss = -math.log(prob)
            
            TotalLoss += loss
            Count += 1
        
        return TotalLoss / Count if Count > 0 else 0.0
    
    def Generate(
        self,
        Prompt: List[int],
        MaxTokens: int = 100,
        Temperature: float = 1.0,
        TopK: int = 50,
        TopP: float = 0.9,
        Knowledge: List[List[float]] = None
    ) -> List[int]:
        """
        Generate text autoregressively.
        
        Args:
            Prompt: Initial token IDs
            MaxTokens: Maximum tokens to generate
            Temperature: Sampling temperature (higher = more random)
            TopK: Top-k sampling (0 = disabled)
            TopP: Nucleus sampling threshold
            Knowledge: Optional knowledge context
            
        Returns:
            Generated token IDs (including prompt)
        """
        Generated = list(Prompt)
        
        for _ in range(MaxTokens):
            # Truncate to max sequence length
            Input = Generated[-self.Config.MaxSeqLen:]
            
            # Forward pass
            Logits, _ = self.Forward(Input, Knowledge, Training=False)
            
            # Get logits for last position
            LastLogits = Logits[-1]
            
            # Apply temperature
            LastLogits = [l / Temperature for l in LastLogits]
            
            # Top-k filtering
            if TopK > 0:
                LastLogits = self._TopKFilter(LastLogits, TopK)
            
            # Top-p (nucleus) filtering
            if TopP < 1.0:
                LastLogits = self._TopPFilter(LastLogits, TopP)
            
            # Sample from distribution
            Probs = self._Softmax(LastLogits)
            NextToken = self._Sample(Probs)
            
            Generated.append(NextToken)
            
            # Stop if EOS token
            if NextToken == 3:  # <EOS>
                break
        
        return Generated
    
    def _TopKFilter(
        self, 
        Logits: List[float], 
        K: int
    ) -> List[float]:
        """Keep only top-k logits"""
        # Get top-k indices
        IndexedLogits = list(enumerate(Logits))
        IndexedLogits.sort(key=lambda x: -x[1])
        TopKIndices = set(idx for idx, _ in IndexedLogits[:K])
        
        # Set non-top-k to -inf
        return [
            l if i in TopKIndices else -float('inf')
            for i, l in enumerate(Logits)
        ]
    
    def _TopPFilter(
        self, 
        Logits: List[float], 
        P: float
    ) -> List[float]:
        """Nucleus sampling - keep smallest set with cumulative prob >= P"""
        Probs = self._Softmax(Logits)
        
        # Sort by probability
        IndexedProbs = list(enumerate(Probs))
        IndexedProbs.sort(key=lambda x: -x[1])
        
        # Find cutoff
        CumProb = 0.0
        KeepIndices = set()
        for idx, prob in IndexedProbs:
            CumProb += prob
            KeepIndices.add(idx)
            if CumProb >= P:
                break
        
        # Filter logits
        return [
            l if i in KeepIndices else -float('inf')
            for i, l in enumerate(Logits)
        ]
    
    def _Sample(self, Probs: List[float]) -> int:
        """Sample from probability distribution"""
        r = random.random()
        CumProb = 0.0
        for i, p in enumerate(Probs):
            CumProb += p
            if r < CumProb:
                return i
        return len(Probs) - 1
    
    def CountParameters(self) -> int:
        """Count total parameters in the model"""
        Total = 0
        
        # Token embeddings
        Total += self.Config.VocabSize * self.Config.EmbedDim
        
        # Per block
        for _ in self.Blocks:
            # Attention: Q, K, V, O projections
            Total += 4 * self.Config.EmbedDim * self.Config.EmbedDim
            # FFN
            Total += 2 * self.Config.EmbedDim * self.Config.FFNDim
            # LayerNorms
            Total += 2 * self.Config.EmbedDim
        
        # Final norm
        Total += self.Config.EmbedDim
        
        return Total
    
    def Save(self, Path: str):
        """Save model to file"""
        # Save configuration and weights
        # (Implementation depends on serialization format)
        pass
    
    def Load(self, Path: str):
        """Load model from file"""
        pass
    
    def GetStats(self) -> Dict:
        """Get model statistics"""
        return {
            "Parameters": self.CountParameters(),
            "Config": {
                "VocabSize": self.Config.VocabSize,
                "EmbedDim": self.Config.EmbedDim,
                "NumLayers": self.Config.NumLayers,
                "NumHeads": self.Config.NumHeads,
                "MaxSeqLen": self.Config.MaxSeqLen,
            },
            "TrainingStep": self.TrainingStep,
            "LossHistory": self.LossHistory[-100:],  # Last 100
        }


# =============================================================================
# MODEL FACTORY
# =============================================================================

def CreateModel(Size: str = "small") -> GroundZeroLM:
    """
    Create a GroundZero-LM model.
    
    Args:
        Size: "small", "medium", or "large"
        
    Returns:
        Initialized model
    """
    if Size.lower() == "small":
        Config = GPTConfig.Small()
    elif Size.lower() == "medium":
        Config = GPTConfig.Medium()
    elif Size.lower() == "large":
        Config = GPTConfig.Large()
    else:
        raise ValueError(f"Unknown model size: {Size}")
    
    Model = GroundZeroLM(Config)
    
    print(f"Created GroundZero-LM ({Size}):")
    print(f"  Parameters: {Model.CountParameters():,}")
    print(f"  Layers: {Config.NumLayers}")
    print(f"  Heads: {Config.NumHeads}")
    print(f"  Embed Dim: {Config.EmbedDim}")
    
    return Model
```

---

## 10. Phase 6: Training Infrastructure

### Training Loop

```python
# File: src/trainer.py

"""
Training infrastructure for GroundZero-LM
"""

import math
import time
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterator
from dataclasses import dataclass

from .gpt_model import GroundZeroLM, GPTConfig
from .tokenizer import BPETokenizer


@dataclass
class TrainingConfig:
    """Training configuration"""
    BatchSize: int = 4
    SeqLen: int = 512
    LearningRate: float = 3e-4
    MinLR: float = 1e-5
    WarmupSteps: int = 1000
    TotalSteps: int = 100000
    GradClipNorm: float = 1.0
    WeightDecay: float = 0.1
    Beta1: float = 0.9
    Beta2: float = 0.95
    LogInterval: int = 100
    SaveInterval: int = 1000
    EvalInterval: int = 500


class DataLoader:
    """
    Data loader for language model training.
    Provides batches of token sequences.
    """
    
    def __init__(
        self,
        Tokenizer: BPETokenizer,
        Texts: List[str],
        SeqLen: int,
        BatchSize: int
    ):
        """
        Initialize data loader.
        
        Args:
            Tokenizer: Trained tokenizer
            Texts: List of training texts
            SeqLen: Sequence length for training
            BatchSize: Batch size
        """
        self.Tokenizer = Tokenizer
        self.SeqLen = SeqLen
        self.BatchSize = BatchSize
        
        # Tokenize all texts
        print("Tokenizing training data...")
        AllTokens = []
        for text in Texts:
            tokens = Tokenizer.Encode(text)
            AllTokens.extend(tokens)
        
        self.Tokens = AllTokens
        self.NumTokens = len(AllTokens)
        print(f"Total tokens: {self.NumTokens:,}")
        
        # Calculate number of batches
        TokensPerBatch = BatchSize * SeqLen
        self.NumBatches = self.NumTokens // TokensPerBatch
        print(f"Batches per epoch: {self.NumBatches:,}")
    
    def GetBatch(self) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Get a random batch for training.
        
        Returns:
            Tuple of (inputs, targets) where:
            - inputs: [BatchSize, SeqLen]
            - targets: [BatchSize, SeqLen] (inputs shifted by 1)
        """
        Inputs = []
        Targets = []
        
        for _ in range(self.BatchSize):
            # Random starting position
            MaxStart = self.NumTokens - self.SeqLen - 1
            Start = random.randint(0, MaxStart)
            
            # Get sequence
            Seq = self.Tokens[Start : Start + self.SeqLen + 1]
            
            Inputs.append(Seq[:-1])   # All but last
            Targets.append(Seq[1:])   # All but first
        
        return Inputs, Targets
    
    def __iter__(self) -> Iterator[Tuple[List[List[int]], List[List[int]]]]:
        """Iterate over batches"""
        for _ in range(self.NumBatches):
            yield self.GetBatch()


class AdamW:
    """
    AdamW optimizer with weight decay.
    
    This is the standard optimizer for training transformers.
    """
    
    def __init__(
        self,
        Params: Dict[str, List[List[float]]],
        LR: float = 3e-4,
        Beta1: float = 0.9,
        Beta2: float = 0.95,
        WeightDecay: float = 0.1,
        Eps: float = 1e-8
    ):
        """
        Initialize optimizer.
        
        Args:
            Params: Dictionary of parameter name -> weights
            LR: Learning rate
            Beta1: First moment decay
            Beta2: Second moment decay
            WeightDecay: Weight decay coefficient
            Eps: Numerical stability
        """
        self.Params = Params
        self.LR = LR
        self.Beta1 = Beta1
        self.Beta2 = Beta2
        self.WeightDecay = WeightDecay
        self.Eps = Eps
        
        # Moment estimates
        self.M: Dict[str, List[List[float]]] = {}  # First moment
        self.V: Dict[str, List[List[float]]] = {}  # Second moment
        
        # Initialize moments to zero
        for name, param in Params.items():
            self.M[name] = [[0.0] * len(row) for row in param]
            self.V[name] = [[0.0] * len(row) for row in param]
        
        self.Step = 0
    
    def Update(
        self,
        Grads: Dict[str, List[List[float]]],
        LR: float = None
    ):
        """
        Update parameters with gradients.
        
        Args:
            Grads: Dictionary of parameter name -> gradients
            LR: Optional learning rate override
        """
        self.Step += 1
        lr = LR or self.LR
        
        # Bias correction
        BiasCorr1 = 1 - self.Beta1 ** self.Step
        BiasCorr2 = 1 - self.Beta2 ** self.Step
        
        for name, grad in Grads.items():
            param = self.Params[name]
            m = self.M[name]
            v = self.V[name]
            
            for i in range(len(param)):
                for j in range(len(param[i])):
                    g = grad[i][j]
                    
                    # Update moments
                    m[i][j] = self.Beta1 * m[i][j] + (1 - self.Beta1) * g
                    v[i][j] = self.Beta2 * v[i][j] + (1 - self.Beta2) * g * g
                    
                    # Bias-corrected moments
                    m_hat = m[i][j] / BiasCorr1
                    v_hat = v[i][j] / BiasCorr2
                    
                    # Update parameter
                    update = lr * m_hat / (math.sqrt(v_hat) + self.Eps)
                    
                    # Weight decay (decoupled)
                    param[i][j] -= update + lr * self.WeightDecay * param[i][j]


class LRScheduler:
    """
    Learning rate scheduler with warmup and cosine decay.
    """
    
    def __init__(
        self,
        MaxLR: float,
        MinLR: float,
        WarmupSteps: int,
        TotalSteps: int
    ):
        """
        Initialize scheduler.
        
        Args:
            MaxLR: Maximum learning rate
            MinLR: Minimum learning rate
            WarmupSteps: Linear warmup steps
            TotalSteps: Total training steps
        """
        self.MaxLR = MaxLR
        self.MinLR = MinLR
        self.WarmupSteps = WarmupSteps
        self.TotalSteps = TotalSteps
    
    def GetLR(self, Step: int) -> float:
        """
        Get learning rate for current step.
        
        Args:
            Step: Current training step
            
        Returns:
            Learning rate
        """
        if Step < self.WarmupSteps:
            # Linear warmup
            return self.MaxLR * Step / self.WarmupSteps
        elif Step >= self.TotalSteps:
            return self.MinLR
        else:
            # Cosine decay
            Progress = (Step - self.WarmupSteps) / (self.TotalSteps - self.WarmupSteps)
            Cosine = 0.5 * (1 + math.cos(math.pi * Progress))
            return self.MinLR + (self.MaxLR - self.MinLR) * Cosine


class Trainer:
    """
    Complete training loop for GroundZero-LM.
    """
    
    def __init__(
        self,
        Model: GroundZeroLM,
        Tokenizer: BPETokenizer,
        Config: TrainingConfig
    ):
        """
        Initialize trainer.
        
        Args:
            Model: The language model
            Tokenizer: Trained tokenizer
            Config: Training configuration
        """
        self.Model = Model
        self.Tokenizer = Tokenizer
        self.Config = Config
        
        # Learning rate scheduler
        self.Scheduler = LRScheduler(
            Config.LearningRate,
            Config.MinLR,
            Config.WarmupSteps,
            Config.TotalSteps
        )
        
        # Training state
        self.Step = 0
        self.BestLoss = float('inf')
        self.LossHistory: List[float] = []
    
    def ComputeGradients(
        self,
        Inputs: List[List[int]],
        Targets: List[List[int]]
    ) -> Tuple[float, Dict]:
        """
        Compute gradients via numerical differentiation.
        
        Note: In practice, you'd use automatic differentiation (PyTorch/JAX).
        This is a simplified implementation for educational purposes.
        
        Args:
            Inputs: Input token IDs [BatchSize, SeqLen]
            Targets: Target token IDs [BatchSize, SeqLen]
            
        Returns:
            Tuple of (loss, gradients)
        """
        # For educational purposes, we'll use numerical gradients
        # In practice, use autograd!
        
        TotalLoss = 0.0
        
        for inp, tgt in zip(Inputs, Targets):
            Logits, _ = self.Model.Forward(inp, Training=True)
            Loss = self.Model.ComputeLoss(Logits, tgt)
            TotalLoss += Loss
        
        AvgLoss = TotalLoss / len(Inputs)
        
        # Placeholder for gradients
        # Real implementation would compute these via backpropagation
        Gradients = {}
        
        return AvgLoss, Gradients
    
    def TrainStep(
        self,
        Inputs: List[List[int]],
        Targets: List[List[int]]
    ) -> float:
        """
        Single training step.
        
        Args:
            Inputs: Input token IDs
            Targets: Target token IDs
            
        Returns:
            Loss value
        """
        # Get learning rate
        LR = self.Scheduler.GetLR(self.Step)
        
        # Compute loss and gradients
        Loss, Grads = self.ComputeGradients(Inputs, Targets)
        
        # Update step counter
        self.Step += 1
        
        # Track loss
        self.LossHistory.append(Loss)
        if Loss < self.BestLoss:
            self.BestLoss = Loss
        
        return Loss
    
    def Train(
        self,
        DataLoader: DataLoader,
        CheckpointDir: str = None
    ):
        """
        Full training loop.
        
        Args:
            DataLoader: Training data loader
            CheckpointDir: Directory for saving checkpoints
        """
        print("\n" + "=" * 70)
        print("Starting GroundZero-LM Training")
        print("=" * 70)
        print(f"Model parameters: {self.Model.CountParameters():,}")
        print(f"Training steps: {self.Config.TotalSteps:,}")
        print(f"Batch size: {self.Config.BatchSize}")
        print(f"Sequence length: {self.Config.SeqLen}")
        print("=" * 70 + "\n")
        
        StartTime = time.time()
        
        while self.Step < self.Config.TotalSteps:
            for Inputs, Targets in DataLoader:
                # Training step
                Loss = self.TrainStep(Inputs, Targets)
                
                # Logging
                if self.Step % self.Config.LogInterval == 0:
                    LR = self.Scheduler.GetLR(self.Step)
                    Elapsed = time.time() - StartTime
                    TokensPerSec = (self.Step * self.Config.BatchSize * 
                                   self.Config.SeqLen) / Elapsed
                    
                    print(f"Step {self.Step:6d} | "
                          f"Loss: {Loss:.4f} | "
                          f"LR: {LR:.2e} | "
                          f"Tokens/s: {TokensPerSec:.0f}")
                
                # Checkpoint
                if CheckpointDir and self.Step % self.Config.SaveInterval == 0:
                    self.SaveCheckpoint(CheckpointDir)
                
                if self.Step >= self.Config.TotalSteps:
                    break
        
        print("\n" + "=" * 70)
        print("Training Complete!")
        print(f"Final loss: {self.LossHistory[-1]:.4f}")
        print(f"Best loss: {self.BestLoss:.4f}")
        print("=" * 70 + "\n")
    
    def SaveCheckpoint(self, Dir: str):
        """Save training checkpoint"""
        Path(Dir).mkdir(parents=True, exist_ok=True)
        # Implementation depends on serialization format
        print(f"   💾 Checkpoint saved at step {self.Step}")
    
    def LoadCheckpoint(self, Path: str):
        """Load training checkpoint"""
        pass
```

---

## 11. Phase 7: Training the Model

### Training Data Preparation

You'll need significant training data. Sources:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRAINING DATA SOURCES                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Wikipedia (Your existing auto_learner already fetches this!)    │
│     - ~20GB of text                                                 │
│     - High quality, factual                                         │
│     - Already in your system                                        │
│                                                                     │
│  2. Books (Project Gutenberg)                                       │
│     - Free, public domain                                           │
│     - Good for language patterns                                    │
│     - ~50K books available                                          │
│                                                                     │
│  3. OpenWebText                                                     │
│     - Reddit outbound links (upvoted content)                       │
│     - ~40GB                                                         │
│     - Diverse topics                                                │
│                                                                     │
│  4. Your Knowledge Graph Facts                                      │
│     - Convert triples to sentences                                  │
│     - "X is_a Y" → "X is a type of Y."                             │
│     - Unique to your system!                                        │
│                                                                     │
│  MINIMUM FOR TRAINING:                                              │
│  - Small model: 100MB-1GB of text                                   │
│  - Medium model: 5-10GB of text                                     │
│  - Large model: 20-50GB of text                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Training Script

```python
# File: train_groundzero_lm.py

"""
Training script for GroundZero-LM
"""

from pathlib import Path
from src.gpt_model import CreateModel, GPTConfig
from src.tokenizer import BPETokenizer, TokenizerConfig
from src.trainer import Trainer, TrainingConfig, DataLoader


def LoadTrainingTexts(DataDir: str) -> List[str]:
    """Load all training texts from data directory"""
    texts = []
    
    # Load from saved Wikipedia articles
    wiki_dir = Path(DataDir) / "wikipedia"
    if wiki_dir.exists():
        for file in wiki_dir.glob("*.txt"):
            texts.append(file.read_text(encoding='utf-8'))
    
    # Load from knowledge graph (convert triples to sentences)
    kg_file = Path(DataDir) / "knowledge.db"
    if kg_file.exists():
        # Convert triples to natural language
        # ... implementation ...
        pass
    
    return texts


def main():
    # Configuration
    DATA_DIR = "./data"
    CHECKPOINT_DIR = "./checkpoints"
    
    # Load training data
    print("Loading training data...")
    texts = LoadTrainingTexts(DATA_DIR)
    print(f"Loaded {len(texts)} documents")
    
    # Train or load tokenizer
    tokenizer_path = Path(DATA_DIR) / "tokenizer.json"
    tokenizer = BPETokenizer(TokenizerConfig(VocabSize=32000))
    
    if tokenizer_path.exists():
        print("Loading existing tokenizer...")
        tokenizer.Load(str(tokenizer_path))
    else:
        print("Training tokenizer...")
        tokenizer.Train(texts)
        tokenizer.Save(str(tokenizer_path))
    
    # Create model
    print("\nCreating model...")
    model = CreateModel("small")  # Start small!
    
    # Create data loader
    data_loader = DataLoader(
        Tokenizer=tokenizer,
        Texts=texts,
        SeqLen=512,
        BatchSize=4
    )
    
    # Create trainer
    train_config = TrainingConfig(
        BatchSize=4,
        SeqLen=512,
        LearningRate=3e-4,
        TotalSteps=100000,
        LogInterval=100,
        SaveInterval=1000
    )
    
    trainer = Trainer(model, tokenizer, train_config)
    
    # Train!
    trainer.Train(data_loader, CHECKPOINT_DIR)
    
    # Save final model
    model.Save(f"{CHECKPOINT_DIR}/final_model.pkl")
    print("Training complete!")


if __name__ == "__main__":
    main()
```

---

## 12. Phase 8: Integration with GroundZero

### Knowledge-Grounded Generation

This is where your system becomes unique:

```python
# File: src/knowledge_grounded_lm.py

"""
Knowledge-Grounded Language Model
Combines GroundZero-LM with your existing neural pipeline
"""

from typing import List, Dict, Tuple, Optional
from .gpt_model import GroundZeroLM, GPTConfig
from .neural_pipeline import NeuralPipeline
from .tokenizer import BPETokenizer


class KnowledgeGroundedLM:
    """
    Language model grounded in explicit knowledge.
    
    This combines:
    1. Your trained TransE knowledge embeddings
    2. Your neural pipeline for retrieval
    3. GroundZero-LM for generation
    
    The result is an LLM that:
    - Retrieves relevant facts before generating
    - Can cite its sources
    - Hallucinates less than pure LLMs
    """
    
    def __init__(
        self,
        LM: GroundZeroLM,
        Pipeline: NeuralPipeline,
        Tokenizer: BPETokenizer
    ):
        """
        Initialize knowledge-grounded LM.
        
        Args:
            LM: Trained language model
            Pipeline: Your existing neural pipeline
            Tokenizer: Trained tokenizer
        """
        self.LM = LM
        self.Pipeline = Pipeline
        self.Tokenizer = Tokenizer
    
    def _EncodeKnowledge(
        self,
        Facts: List[Tuple[str, str, str]]
    ) -> List[List[float]]:
        """
        Encode knowledge facts as embeddings.
        
        Args:
            Facts: List of (subject, predicate, object) triples
            
        Returns:
            Knowledge embeddings [NumFacts, EmbedDim]
        """
        Embeddings = []
        
        for subj, pred, obj in Facts:
            # Get entity embeddings from TransE
            SubjEmbed = self.Pipeline.EntityEmbeddings.get(subj, [0] * self.LM.Config.EmbedDim)
            ObjEmbed = self.Pipeline.EntityEmbeddings.get(obj, [0] * self.LM.Config.EmbedDim)
            RelEmbed = self.Pipeline.RelationEmbeddings.get(pred, [0] * self.LM.Config.EmbedDim)
            
            # Combine: subject + relation + object
            Combined = [
                (s + r + o) / 3 
                for s, r, o in zip(SubjEmbed, RelEmbed, ObjEmbed)
            ]
            
            Embeddings.append(Combined)
        
        return Embeddings
    
    def _RetrieveKnowledge(
        self,
        Query: str,
        TopK: int = 10
    ) -> Tuple[List[Tuple[str, str, str]], List[List[float]]]:
        """
        Retrieve relevant knowledge for a query.
        
        Args:
            Query: User query
            TopK: Number of facts to retrieve
            
        Returns:
            Tuple of (facts, embeddings)
        """
        # Use your neural pipeline to find relevant facts
        Result = self.Pipeline.Process(Query)
        
        # Get facts from reasoning path
        Facts = []
        for hop in Result.ReasoningPath:
            Facts.append((hop.FromEntity, hop.Relation, hop.ToEntity))
        
        # Add facts from candidate answers
        for entity, score in Result.CandidateAnswers[:5]:
            # Look up facts about this entity
            if entity in self.Pipeline.KnowledgeTriples:
                for fact in self.Pipeline.KnowledgeTriples[entity][:3]:
                    Facts.append(fact)
        
        # Deduplicate
        Facts = list(set(Facts))[:TopK]
        
        # Encode
        Embeddings = self._EncodeKnowledge(Facts)
        
        return Facts, Embeddings
    
    def _FormatKnowledgePrompt(
        self,
        Query: str,
        Facts: List[Tuple[str, str, str]]
    ) -> str:
        """
        Format query with knowledge context.
        
        Args:
            Query: User query
            Facts: Retrieved facts
            
        Returns:
            Formatted prompt
        """
        # Convert facts to natural language
        FactStrings = []
        for subj, pred, obj in Facts:
            FactStrings.append(f"- {subj} {pred.replace('_', ' ')} {obj}")
        
        if FactStrings:
            KnowledgeStr = "\n".join(FactStrings)
            Prompt = f"""Knowledge:
{KnowledgeStr}

Question: {Query}

Based on the knowledge above, """
        else:
            Prompt = f"Question: {Query}\n\nAnswer: "
        
        return Prompt
    
    def Generate(
        self,
        Query: str,
        MaxTokens: int = 200,
        Temperature: float = 0.7,
        UseKnowledge: bool = True
    ) -> Dict:
        """
        Generate a response grounded in knowledge.
        
        Args:
            Query: User query
            MaxTokens: Maximum tokens to generate
            Temperature: Sampling temperature
            UseKnowledge: Whether to use knowledge retrieval
            
        Returns:
            Dictionary with response and metadata
        """
        # Retrieve knowledge
        Facts = []
        KnowledgeEmbeddings = None
        
        if UseKnowledge:
            Facts, KnowledgeEmbeddings = self._RetrieveKnowledge(Query)
        
        # Format prompt
        Prompt = self._FormatKnowledgePrompt(Query, Facts)
        
        # Tokenize
        PromptIDs = self.Tokenizer.Encode(Prompt)
        
        # Generate
        OutputIDs = self.LM.Generate(
            Prompt=PromptIDs,
            MaxTokens=MaxTokens,
            Temperature=Temperature,
            Knowledge=KnowledgeEmbeddings
        )
        
        # Decode
        Response = self.Tokenizer.Decode(OutputIDs[len(PromptIDs):])
        
        return {
            "Query": Query,
            "Response": Response,
            "RetrievedFacts": Facts,
            "FactCount": len(Facts),
            "TokensGenerated": len(OutputIDs) - len(PromptIDs)
        }
    
    def Chat(self, Query: str) -> str:
        """
        Simple chat interface.
        
        Args:
            Query: User input
            
        Returns:
            Response text
        """
        Result = self.Generate(Query)
        return Result["Response"]
```

### Updated Main Entry Point

```python
# Addition to main.py for GroundZero AI v5.0

def StartGroundedChat():
    """Start chat with knowledge-grounded LM"""
    from src.knowledge_grounded_lm import KnowledgeGroundedLM
    from src.gpt_model import CreateModel
    from src.tokenizer import BPETokenizer
    
    print("\n" + "=" * 70)
    print("🧠 GroundZero AI v5.0 - Knowledge-Grounded LM")
    print("=" * 70 + "\n")
    
    # Load components
    GlobalState.Initialize()
    
    # Load trained LM
    lm = CreateModel("small")
    lm.Load("checkpoints/final_model.pkl")
    
    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.Load("data/tokenizer.json")
    
    # Create knowledge-grounded LM
    grounded_lm = KnowledgeGroundedLM(
        LM=lm,
        Pipeline=GlobalState.Engine.NeuralPipeline,
        Tokenizer=tokenizer
    )
    
    print("Ready! Your questions will be answered using both:")
    print("  - Neural language model (generation)")
    print("  - Knowledge graph (facts)")
    print("\nType 'quit' to exit.\n")
    
    while True:
        try:
            query = input("You: ").strip()
        except EOFError:
            break
        
        if query.lower() == 'quit':
            break
        
        result = grounded_lm.Generate(query)
        
        print(f"\n🧠 GroundZero: {result['Response']}")
        if result['RetrievedFacts']:
            print(f"   [Grounded in {result['FactCount']} facts]")
        print()
```

---

## 13. Phase 9: Fine-tuning & Alignment

### Instruction Fine-tuning

After pretraining, fine-tune on instruction-following:

```python
# File: src/instruction_tuning.py

"""
Instruction fine-tuning for GroundZero-LM
"""

INSTRUCTION_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}"""


def CreateInstructionDataset(
    QAPairs: List[Tuple[str, str]]
) -> List[str]:
    """
    Create instruction-tuning dataset.
    
    Args:
        QAPairs: List of (question, answer) pairs
        
    Returns:
        List of formatted training examples
    """
    Examples = []
    
    for question, answer in QAPairs:
        example = INSTRUCTION_TEMPLATE.format(
            instruction=question,
            response=answer
        )
        Examples.append(example)
    
    return Examples


def GenerateQAFromKnowledge(
    KnowledgeTriples: Dict[str, List[Tuple]]
) -> List[Tuple[str, str]]:
    """
    Generate Q&A pairs from your knowledge graph.
    
    This creates training data for instruction tuning
    based on your existing knowledge!
    """
    QAPairs = []
    
    for entity, triples in KnowledgeTriples.items():
        for subj, pred, obj in triples:
            # Generate questions based on relation type
            if pred == "is_a":
                Q = f"What is {subj}?"
                A = f"{subj.title()} is a type of {obj}."
            elif pred == "has_property":
                Q = f"What are the properties of {subj}?"
                A = f"{subj.title()} has the property of {obj}."
            elif pred == "located_in":
                Q = f"Where is {subj} located?"
                A = f"{subj.title()} is located in {obj}."
            elif pred == "part_of":
                Q = f"What is {subj} a part of?"
                A = f"{subj.title()} is part of {obj}."
            elif "cause" in pred:
                Q = f"What causes {obj}?"
                A = f"{subj.title()} causes {obj}."
            else:
                Q = f"What do you know about {subj}?"
                A = f"{subj.title()} {pred.replace('_', ' ')} {obj}."
            
            QAPairs.append((Q, A))
    
    return QAPairs
```

---

## 14. Phase 10: Optimization & Deployment

### Quantization

Reduce model size for deployment:

```python
# File: src/quantization.py

"""
Model quantization for efficient inference
"""

def QuantizeWeights(
    Weights: List[List[float]],
    Bits: int = 8
) -> Tuple[List[List[int]], float, float]:
    """
    Quantize weights to lower precision.
    
    Args:
        Weights: Full precision weights
        Bits: Target bit width (8, 4, etc.)
        
    Returns:
        Tuple of (quantized_weights, scale, zero_point)
    """
    # Find range
    MinVal = min(min(row) for row in Weights)
    MaxVal = max(max(row) for row in Weights)
    
    # Compute scale and zero point
    Range = 2 ** Bits - 1
    Scale = (MaxVal - MinVal) / Range
    ZeroPoint = round(-MinVal / Scale)
    
    # Quantize
    Quantized = []
    for row in Weights:
        QRow = []
        for val in row:
            q = round(val / Scale + ZeroPoint)
            q = max(0, min(Range, q))  # Clamp
            QRow.append(int(q))
        Quantized.append(QRow)
    
    return Quantized, Scale, ZeroPoint


def DequantizeWeights(
    Quantized: List[List[int]],
    Scale: float,
    ZeroPoint: float
) -> List[List[float]]:
    """Dequantize weights back to float"""
    return [
        [(q - ZeroPoint) * Scale for q in row]
        for row in Quantized
    ]
```

### Key-Value Cache

Speed up generation:

```python
class KVCache:
    """
    Key-Value cache for efficient autoregressive generation.
    
    Instead of recomputing K,V for all previous tokens,
    cache them and only compute for new token.
    """
    
    def __init__(self, NumLayers: int, NumHeads: int, HeadDim: int):
        self.NumLayers = NumLayers
        self.NumHeads = NumHeads
        self.HeadDim = HeadDim
        
        # Cache: [Layer][Head] -> (Keys, Values)
        self.Cache: List[List[Tuple]] = [
            [None for _ in range(NumHeads)]
            for _ in range(NumLayers)
        ]
    
    def Update(
        self,
        Layer: int,
        Head: int,
        NewK: List[float],
        NewV: List[float]
    ):
        """Add new key-value pair to cache"""
        if self.Cache[Layer][Head] is None:
            self.Cache[Layer][Head] = ([NewK], [NewV])
        else:
            Keys, Values = self.Cache[Layer][Head]
            Keys.append(NewK)
            Values.append(NewV)
    
    def Get(self, Layer: int, Head: int) -> Tuple:
        """Get cached keys and values"""
        return self.Cache[Layer][Head]
    
    def Clear(self):
        """Clear all caches"""
        self.Cache = [
            [None for _ in range(self.NumHeads)]
            for _ in range(self.NumLayers)
        ]
```

---

## 15. Appendix: Math Deep Dive

### Attention Mathematics

```
Given input X ∈ ℝ^(n×d):

1. Linear projections:
   Q = XW_Q,  K = XW_K,  V = XW_V
   where W_Q, W_K, W_V ∈ ℝ^(d×d_k)

2. Attention scores:
   A = softmax(QK^T / √d_k)
   
   The softmax is:
   softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
   
   Division by √d_k prevents dot products from growing too large
   as d_k increases (variance stabilization).

3. Attention output:
   Output = AV

4. Multi-head:
   head_i = Attention(XW_Q^i, XW_K^i, XW_V^i)
   MultiHead = Concat(head_1, ..., head_h)W_O
```

### Transformer Block Mathematics

```
Given input X:

1. Self-attention with residual:
   X = X + MultiHead(LayerNorm(X))

2. FFN with residual:
   X = X + FFN(LayerNorm(X))
   
   where FFN(x) = GELU(xW_1 + b_1)W_2 + b_2

3. GELU activation:
   GELU(x) = x · Φ(x)
   where Φ is the CDF of standard normal distribution
   
   Approximation:
   GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
```

### Cross-Entropy Loss

```
For vocabulary size V and target token t:

Loss = -log(P(t|x))
     = -log(softmax(logits)[t])
     = -logits[t] + log(Σ_i exp(logits[i]))

Gradient with respect to logits:
∂L/∂logits[i] = softmax(logits)[i] - 1(i=t)

Where 1(i=t) is 1 if i=t, else 0.
```

---

## 16. Appendix: Troubleshooting

### Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| Loss not decreasing | LR too high/low | Try different learning rates |
| Loss is NaN | Numerical instability | Add gradient clipping, reduce LR |
| OOM error | Model too large | Reduce batch size, use gradient checkpointing |
| Generation repeats | Temperature too low | Increase temperature, add repetition penalty |
| Nonsense output | Undertrained | Train longer, check data quality |
| Slow training | CPU-bound | Use GPU, optimize data loading |

### Debugging Checklist

```
□ Check data is correctly tokenized (encode/decode should be identity)
□ Verify shapes at each layer
□ Monitor gradient norms (should not explode or vanish)
□ Check attention weights sum to 1
□ Verify causal mask is correct (no future leakage)
□ Test generation with temperature=0 (should be deterministic)
```

---

## 17. Appendix: Resources

### Papers to Read

1. **"Attention Is All You Need"** (Vaswani et al., 2017)
   - Original transformer paper

2. **"Language Models are Unsupervised Multitask Learners"** (GPT-2, Radford et al., 2019)
   - GPT-2 paper, decoder-only architecture

3. **"Language Models are Few-Shot Learners"** (GPT-3, Brown et al., 2020)
   - Scaling and in-context learning

4. **"LLaMA: Open and Efficient Foundation Language Models"** (Touvron et al., 2023)
   - Modern efficient architecture (RoPE, SwiGLU, RMSNorm)

5. **"QA-GNN"** (Yasunaga et al., 2021)
   - Knowledge graph + LM integration

### Code References

- **nanoGPT** (Karpathy): Minimal GPT implementation
- **LLaMA** (Meta): Production-quality implementation
- **GPT-2** (OpenAI): Original GPT-2 code

### When to Start

```
┌─────────────────────────────────────────────────────────────────────┐
│                 MILESTONES TO BEGIN GPT IMPLEMENTATION              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Your current system:                                               │
│    Triples: 8,058      Target: 50,000                              │
│    Entities: 6,800     Target: 20,000                              │
│    TransE Loss: 0.004  Target: < 0.01 ✓                            │
│                                                                     │
│  When to start:                                                     │
│    □ 50K+ triples in knowledge graph                               │
│    □ 20K+ unique entities                                          │
│    □ TransE loss < 0.01                                            │
│    □ Continuous learning running stable                             │
│    □ At least 500 training epochs completed                        │
│                                                                     │
│  Estimated time to reach milestones:                                │
│    At ~100 triples/article × ~500 articles = 50K triples           │
│    Running continuously: ~2-3 days                                  │
│                                                                     │
│  START GPT IMPLEMENTATION WHEN GREEN:                               │
│    Triples:    ████████░░░░░░░░  50K    [16% - KEEP LEARNING]      │
│    Entities:   ████████░░░░░░░░  20K    [34% - KEEP LEARNING]      │
│    TransE:     ████████████████  0.01   [READY ✓]                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Summary

You're building a **knowledge-grounded GPT** from scratch. This is ambitious but achievable.

**Key milestones:**
1. ✅ TransE embeddings (you have this!)
2. □ 50K+ knowledge triples
3. □ BPE Tokenizer
4. □ Transformer architecture
5. □ Training infrastructure
6. □ Pretrain on text
7. □ Integrate with knowledge graph
8. □ Instruction fine-tuning

**What makes yours special:**
- Grounded in explicit knowledge (less hallucination)
- Interpretable (can cite sources)
- Customizable (your knowledge, your model)
- Educational (deep understanding)

**Be patient:** This is months of work. Let your current system keep learning while you implement each phase.

Good luck! 🚀
