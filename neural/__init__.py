"""
GroundZero Neural Module
========================
The neural network components for GroundZero.

This provides:
- Transformer model (GPT-style)
- BPE tokenizer
- Continual learning trainer
- Neural brain integration
"""

try:
    from .transformer import GroundZeroTransformer, TransformerConfig
    from .tokenizer import BPETokenizer
    from .trainer import NeuralTrainer, TrainerConfig
    from .brain import NeuralBrain, get_neural_brain
    
    NEURAL_AVAILABLE = True
    
    __all__ = [
        'GroundZeroTransformer',
        'TransformerConfig',
        'BPETokenizer',
        'NeuralTrainer',
        'TrainerConfig',
        'NeuralBrain',
        'get_neural_brain',
        'NEURAL_AVAILABLE'
    ]
except ImportError as e:
    print(f"⚠️ Neural module not available: {e}")
    NEURAL_AVAILABLE = False
    __all__ = ['NEURAL_AVAILABLE']
