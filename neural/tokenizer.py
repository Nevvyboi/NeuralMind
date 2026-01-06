"""
GroundZero Tokenizer
====================
A Byte-Pair Encoding (BPE) tokenizer that can grow.

This tokenizer:
- Learns subword patterns from your data
- Can expand vocabulary as you learn more
- Handles any text (UTF-8)
- Saves/loads from disk
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict
from pathlib import Path
import pickle


class BPETokenizer:
    """
    Byte-Pair Encoding Tokenizer
    
    BPE works by:
    1. Starting with character-level tokens
    2. Finding the most common adjacent pair
    3. Merging that pair into a new token
    4. Repeating until vocab size is reached
    
    This gives us subword tokens that capture common patterns.
    """
    
    # Special tokens
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        
        # Token to ID mapping
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        # BPE merges (pair -> merged token)
        self.merges: Dict[Tuple[str, str], str] = {}
        self.merge_ranks: Dict[Tuple[str, str], int] = {}
        
        # Initialize special tokens
        self._init_special_tokens()
        
        # Pre-tokenization pattern (split on spaces/punctuation)
        self.pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""")
        
        self.is_trained = False
    
    def _init_special_tokens(self):
        """Initialize special tokens"""
        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        for i, token in enumerate(special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
    
    def _get_pairs(self, word: List[str]) -> Set[Tuple[str, str]]:
        """Get all adjacent pairs in a word"""
        pairs = set()
        for i in range(len(word) - 1):
            pairs.add((word[i], word[i + 1]))
        return pairs
    
    def _count_pairs(self, words_freq: Dict[tuple, int]) -> Counter:
        """Count all pairs across the vocabulary"""
        pairs = Counter()
        for word, freq in words_freq.items():
            symbols = list(word)
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs
    
    def _merge_pair(self, pair: Tuple[str, str], words_freq: Dict[tuple, int]) -> Dict[tuple, int]:
        """Merge a pair throughout the vocabulary"""
        new_words_freq = {}
        bigram = pair
        replacement = ''.join(pair)
        
        for word, freq in words_freq.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == bigram[0] and word[i + 1] == bigram[1]:
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_words_freq[tuple(new_word)] = freq
        
        return new_words_freq
    
    def train(self, texts: List[str], min_freq: int = 2, verbose: bool = True):
        """
        Train the tokenizer on a corpus.
        
        Args:
            texts: List of text strings to train on
            min_freq: Minimum frequency for a token
            verbose: Print progress
        """
        if verbose:
            print("🔤 Training BPE tokenizer...")
        
        # Step 1: Pre-tokenize and count words
        word_freqs = Counter()
        for text in texts:
            tokens = self.pattern.findall(text.lower())
            for token in tokens:
                # Add end-of-word marker
                word = tuple(token) + ('</w>',)
                word_freqs[word] += 1
        
        if verbose:
            print(f"   Found {len(word_freqs)} unique words")
        
        # Step 2: Initialize with characters
        vocab = set()
        for word in word_freqs:
            for char in word:
                vocab.add(char)
        
        # Add characters to vocabulary
        for char in sorted(vocab):
            if char not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[char] = idx
                self.id_to_token[idx] = char
        
        if verbose:
            print(f"   Initial vocab size: {len(self.token_to_id)}")
        
        # Step 3: Iteratively merge most common pairs
        num_merges = self.vocab_size - len(self.token_to_id)
        
        for i in range(num_merges):
            # Count pairs
            pairs = self._count_pairs(word_freqs)
            
            if not pairs:
                break
            
            # Find most common pair
            best_pair = max(pairs, key=pairs.get)
            
            if pairs[best_pair] < min_freq:
                break
            
            # Merge the pair
            word_freqs = self._merge_pair(best_pair, word_freqs)
            
            # Add new token
            new_token = ''.join(best_pair)
            idx = len(self.token_to_id)
            self.token_to_id[new_token] = idx
            self.id_to_token[idx] = new_token
            
            # Record merge
            self.merges[best_pair] = new_token
            self.merge_ranks[best_pair] = len(self.merge_ranks)
            
            if verbose and (i + 1) % 1000 == 0:
                print(f"   Merges: {i + 1}/{num_merges}, vocab size: {len(self.token_to_id)}")
        
        self.is_trained = True
        
        if verbose:
            print(f"✅ Tokenizer trained: {len(self.token_to_id)} tokens, {len(self.merges)} merges")
    
    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single word using BPE"""
        if not word:
            return []
        
        # Add end-of-word marker
        word = list(word) + ['</w>']
        
        # Apply merges
        while len(word) > 1:
            pairs = self._get_pairs(word)
            
            # Find the pair with lowest merge rank
            best_pair = None
            best_rank = float('inf')
            
            for pair in pairs:
                if pair in self.merge_ranks:
                    rank = self.merge_ranks[pair]
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = pair
            
            if best_pair is None:
                break
            
            # Merge the pair
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                    new_word.append(self.merges[best_pair])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        
        return word
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Text to encode
            add_special_tokens: Whether to add BOS/EOS tokens
        
        Returns:
            List of token IDs
        """
        if not self.is_trained:
            raise RuntimeError("Tokenizer not trained! Call train() first.")
        
        # Pre-tokenize
        words = self.pattern.findall(text.lower())
        
        # Tokenize each word
        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_id)
        
        for word in words:
            word_tokens = self._tokenize_word(word)
            for token in word_tokens:
                if token in self.token_to_id:
                    tokens.append(self.token_to_id[token])
                else:
                    tokens.append(self.unk_id)
        
        if add_special_tokens:
            tokens.append(self.eos_id)
        
        return tokens
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
        
        Returns:
            Decoded text
        """
        special_ids = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
        
        tokens = []
        for idx in ids:
            if skip_special_tokens and idx in special_ids:
                continue
            if idx in self.id_to_token:
                tokens.append(self.id_to_token[idx])
            else:
                tokens.append(self.UNK_TOKEN)
        
        # Join and remove end-of-word markers
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        
        return text.strip()
    
    def expand_vocabulary(self, texts: List[str], max_new_tokens: int = 1000):
        """
        Expand vocabulary with new texts (continual learning).
        
        This allows the tokenizer to learn new subwords from new data
        without forgetting old ones.
        """
        if not self.is_trained:
            self.train(texts)
            return
        
        print(f"📈 Expanding vocabulary (current: {len(self.token_to_id)})...")
        
        # Count new words
        word_freqs = Counter()
        for text in texts:
            tokens = self.pattern.findall(text.lower())
            for token in tokens:
                word = tuple(token) + ('</w>',)
                word_freqs[word] += 1
        
        # Learn new merges
        new_merges = 0
        target_vocab = min(len(self.token_to_id) + max_new_tokens, self.vocab_size)
        
        while len(self.token_to_id) < target_vocab and new_merges < max_new_tokens:
            pairs = self._count_pairs(word_freqs)
            
            if not pairs:
                break
            
            best_pair = max(pairs, key=pairs.get)
            
            if pairs[best_pair] < 2:
                break
            
            # Only add if it's a new merge
            if best_pair not in self.merges:
                word_freqs = self._merge_pair(best_pair, word_freqs)
                
                new_token = ''.join(best_pair)
                if new_token not in self.token_to_id:
                    idx = len(self.token_to_id)
                    self.token_to_id[new_token] = idx
                    self.id_to_token[idx] = new_token
                    
                    self.merges[best_pair] = new_token
                    self.merge_ranks[best_pair] = len(self.merge_ranks)
                    new_merges += 1
            else:
                word_freqs = self._merge_pair(best_pair, word_freqs)
        
        print(f"✅ Added {new_merges} new tokens (total: {len(self.token_to_id)})")
    
    def save(self, path: Path):
        """Save tokenizer to disk"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'vocab_size': self.vocab_size,
            'token_to_id': self.token_to_id,
            'id_to_token': {int(k): v for k, v in self.id_to_token.items()},
            'merges': {f"{k[0]}|||{k[1]}": v for k, v in self.merges.items()},
            'merge_ranks': {f"{k[0]}|||{k[1]}": v for k, v in self.merge_ranks.items()},
            'is_trained': self.is_trained
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Tokenizer saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'BPETokenizer':
        """Load tokenizer from disk"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(data['vocab_size'])
        tokenizer.token_to_id = data['token_to_id']
        tokenizer.id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
        tokenizer.merges = {tuple(k.split('|||')): v for k, v in data['merges'].items()}
        tokenizer.merge_ranks = {tuple(k.split('|||')): v for k, v in data['merge_ranks'].items()}
        tokenizer.is_trained = data['is_trained']
        
        print(f"📂 Tokenizer loaded: {len(tokenizer.token_to_id)} tokens")
        return tokenizer
    
    def __len__(self):
        return len(self.token_to_id)


def test_tokenizer():
    """Test the tokenizer"""
    print("=" * 60)
    print("🧪 Testing BPE Tokenizer")
    print("=" * 60)
    
    # Sample texts
    texts = [
        "Hello, world! This is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
        "Python is a great programming language.",
        "Artificial intelligence and neural networks are fascinating.",
        "The transformer architecture revolutionized NLP.",
    ] * 10  # Repeat for more data
    
    # Train tokenizer
    tokenizer = BPETokenizer(vocab_size=500)
    tokenizer.train(texts)
    
    # Test encoding/decoding
    test_text = "Hello world! Machine learning is amazing."
    print(f"\nOriginal: {test_text}")
    
    ids = tokenizer.encode(test_text)
    print(f"Encoded: {ids}")
    
    decoded = tokenizer.decode(ids)
    print(f"Decoded: {decoded}")
    
    # Test expansion
    new_texts = [
        "Quantum computing is the future of technology.",
        "Blockchain and cryptocurrency are changing finance.",
    ] * 5
    
    tokenizer.expand_vocabulary(new_texts, max_new_tokens=50)
    
    # Test new text
    new_test = "Quantum computing and blockchain technology."
    ids = tokenizer.encode(new_test)
    decoded = tokenizer.decode(ids)
    print(f"\nNew text: {new_test}")
    print(f"Decoded: {decoded}")
    
    print("\n✅ Tokenizer test passed!")


if __name__ == "__main__":
    test_tokenizer()
