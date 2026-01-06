"""
GroundZero Neural Integration
=============================
Connects the neural transformer to the rest of the system.

This makes the neural network:
- Automatically learn from Wikipedia articles
- Generate responses
- Get smarter over time
- Work alongside the knowledge graph
"""

import threading
from typing import Dict, Any, List, Optional
from pathlib import Path
from collections import deque


class NeuralBrain:
    """
    The neural brain of GroundZero.
    
    This integrates the transformer model with the rest of the system,
    providing a unified interface for:
    - Learning from text
    - Generating responses
    - Answering questions
    """
    
    def __init__(self, data_dir: Path, model_size: str = "small"):
        self.data_dir = Path(data_dir)
        self.neural_dir = self.data_dir / "neural"
        self.neural_dir.mkdir(parents=True, exist_ok=True)
        
        # Lazy load to avoid import errors
        self.trainer = None
        self.model_size = model_size
        
        # Training buffer
        self.text_buffer: deque = deque(maxlen=100)
        self.buffer_lock = threading.Lock()
        
        # Stats
        self.texts_learned = 0
        self.tokens_generated = 0
        self.is_training = False
        
        # Try to initialize
        self._initialize()
    
    def _initialize(self):
        """Initialize the neural components"""
        try:
            from .trainer import NeuralTrainer, TrainerConfig
            
            config = TrainerConfig(
                model_size=self.model_size,
                batch_size=4,
                learning_rate=3e-4,
                replay_buffer_size=10000,
                save_every_steps=500
            )
            
            self.trainer = NeuralTrainer(config, self.neural_dir)
            print("✅ Neural Brain initialized")
            
        except Exception as e:
            print(f"⚠️ Neural Brain initialization failed: {e}")
            print("   Install PyTorch: pip install torch")
            self.trainer = None
    
    @property
    def is_available(self) -> bool:
        """Check if neural components are available"""
        return self.trainer is not None
    
    def learn(self, text: str, source: str = "") -> Dict[str, Any]:
        """
        Learn from a piece of text.
        
        This adds the text to the training buffer.
        Actual training happens in batches.
        """
        if not self.is_available:
            return {'status': 'unavailable', 'error': 'Neural brain not initialized'}
        
        with self.buffer_lock:
            self.text_buffer.append(text)
            self.texts_learned += 1
        
        # Train if buffer is full
        if len(self.text_buffer) >= 20:
            return self.train_batch()
        
        return {
            'status': 'buffered',
            'buffer_size': len(self.text_buffer),
            'texts_learned': self.texts_learned
        }
    
    def train_batch(self) -> Dict[str, Any]:
        """Train on buffered texts"""
        if not self.is_available:
            return {'status': 'unavailable'}
        
        with self.buffer_lock:
            if not self.text_buffer:
                return {'status': 'empty', 'message': 'No texts in buffer'}
            
            texts = list(self.text_buffer)
            self.text_buffer.clear()
        
        # Train
        try:
            self.is_training = True
            stats = self.trainer.train_on_texts(texts, epochs=1, verbose=True)
            self.is_training = False
            return {
                'status': 'trained',
                'texts': len(texts),
                **stats
            }
        except Exception as e:
            self.is_training = False
            return {'status': 'error', 'error': str(e)}
    
    def generate(self, prompt: str, max_tokens: int = 100,
                 temperature: float = 0.8) -> str:
        """Generate text from a prompt"""
        if not self.is_available:
            return "Neural brain not available. Install PyTorch: pip install torch"
        
        if not self.trainer.tokenizer.is_trained:
            return "Model not trained yet. Please learn some content first."
        
        try:
            generated = self.trainer.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            self.tokens_generated += max_tokens
            return generated
        except Exception as e:
            return f"Generation error: {e}"
    
    def answer(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using the neural model.
        
        This is more sophisticated than simple generation -
        it formats the question as a prompt and extracts the answer.
        """
        if not self.is_available:
            return {
                'answer': None,
                'confidence': 0,
                'error': 'Neural brain not available'
            }
        
        # Format as Q&A prompt
        prompt = f"Question: {question}\nAnswer:"
        
        # Generate response
        response = self.generate(prompt, max_tokens=150, temperature=0.7)
        
        # Extract answer (text after "Answer:")
        if "Answer:" in response:
            answer = response.split("Answer:")[-1].strip()
            # Take first sentence or line
            answer = answer.split('\n')[0].split('.')[0] + '.'
        else:
            answer = response
        
        return {
            'answer': answer,
            'full_response': response,
            'confidence': 0.5,  # Neural confidence is harder to estimate
            'method': 'neural_generation'
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get neural brain statistics for the dashboard"""
        stats = {
            'available': self.is_available,
            'texts_learned': self.texts_learned,
            'tokens_generated': self.tokens_generated,
            'buffer_size': len(self.text_buffer),
            'is_training': self.is_training
        }
        
        if self.is_available:
            trainer_stats = self.trainer.get_stats()
            
            # Model architecture info
            model_config = self.trainer.config
            stats.update({
                # Core stats
                'model_params': trainer_stats['model_params'],
                'vocab_size': trainer_stats['vocab_size'],
                'total_tokens_trained': trainer_stats['total_tokens_trained'],
                'global_step': trainer_stats['global_step'],
                'replay_buffer_size': trainer_stats['replay_buffer_size'],
                'device': trainer_stats['device'],
                'recent_losses': trainer_stats['recent_losses'],
                
                # Architecture details for display
                'model_size': self.model_size,
                'n_layers': getattr(model_config, 'n_layers', self._get_model_layers()),
                'n_heads': getattr(model_config, 'n_heads', self._get_model_heads()),
                'd_model': getattr(model_config, 'd_model', self._get_model_dim()),
                'max_seq_len': getattr(model_config, 'max_seq_len', 512),
            })
        else:
            # Default values when not available
            stats.update({
                'model_params': 0,
                'vocab_size': 0,
                'total_tokens_trained': 0,
                'global_step': 0,
                'replay_buffer_size': 0,
                'device': 'N/A',
                'recent_losses': [],
                'model_size': self.model_size,
                'n_layers': 0,
                'n_heads': 0,
                'd_model': 0,
                'max_seq_len': 0,
            })
        
        return stats
    
    def _get_model_layers(self) -> int:
        """Get number of layers based on model size"""
        sizes = {'tiny': 2, 'small': 4, 'medium': 6, 'large': 12, 'xl': 24}
        return sizes.get(self.model_size, 4)
    
    def _get_model_heads(self) -> int:
        """Get number of attention heads based on model size"""
        sizes = {'tiny': 2, 'small': 4, 'medium': 8, 'large': 12, 'xl': 16}
        return sizes.get(self.model_size, 4)
    
    def _get_model_dim(self) -> int:
        """Get embedding dimension based on model size"""
        sizes = {'tiny': 128, 'small': 256, 'medium': 512, 'large': 768, 'xl': 1024}
        return sizes.get(self.model_size, 256)
    
    def start_background_training(self):
        """Start background training thread"""
        if self.is_available:
            self.trainer.start_background_training()
    
    def stop_background_training(self):
        """Stop background training"""
        if self.is_available:
            self.trainer.stop_background_training()
    
    def save(self):
        """Save neural state"""
        if self.is_available:
            self.trainer.save_checkpoint()


# Global instance
_neural_brain: Optional[NeuralBrain] = None


def get_neural_brain(data_dir: Path = None, model_size: str = "small") -> NeuralBrain:
    """Get or create the neural brain instance"""
    global _neural_brain
    
    if _neural_brain is None:
        if data_dir is None:
            data_dir = Path("data")
        _neural_brain = NeuralBrain(data_dir, model_size)
    
    return _neural_brain


def test_neural_brain():
    """Test the neural brain"""
    import tempfile
    
    print("=" * 60)
    print("🧪 Testing Neural Brain")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        brain = NeuralBrain(Path(tmpdir), model_size="tiny")
        
        if not brain.is_available:
            print("⚠️ PyTorch not available, skipping test")
            return
        
        # Learn some texts
        texts = [
            "Python is a programming language created by Guido van Rossum.",
            "Machine learning uses algorithms to learn patterns from data.",
            "The transformer architecture was introduced in the paper Attention is All You Need.",
            "Neural networks are inspired by the human brain.",
            "Deep learning has revolutionized artificial intelligence.",
        ] * 5
        
        print("\n📚 Learning texts...")
        for text in texts:
            result = brain.learn(text)
            if result['status'] == 'trained':
                print(f"   Trained batch: {result.get('loss', 'N/A'):.4f} loss")
        
        # Generate
        print("\n🔮 Generating text...")
        generated = brain.generate("Python is", max_tokens=30)
        print(f"   Generated: {generated}")
        
        # Answer
        print("\n❓ Answering question...")
        answer = brain.answer("What is Python?")
        print(f"   Answer: {answer['answer']}")
        
        # Stats
        print(f"\n📊 Stats: {brain.get_stats()}")
    
    print("\n✅ Neural Brain test passed!")


if __name__ == "__main__":
    test_neural_brain()
