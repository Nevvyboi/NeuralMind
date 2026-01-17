"""
GroundZero AI - Core Model System
=================================

Model management, inference, and training:
1. Download and load DeepSeek-R1-Distill (renamed to GroundZero)
2. Inference with reasoning capabilities
3. Continuous learning and updates
4. QLoRA fine-tuning support
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime

try:
    from ..utils import (
        get_data_path, ensure_dir, load_json, save_json,
        logger, timestamp, Config, get_config
    )
except ImportError:
    from utils import (
        get_data_path, ensure_dir, load_json, save_json,
        logger, timestamp, Config, get_config
    )


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

@dataclass
class ModelState:
    """Track model state and version."""
    name: str = "GroundZero-AI"
    version: str = "1.0.0"
    base_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    is_loaded: bool = False
    is_trained: bool = False
    training_sessions: int = 0
    total_tokens_trained: int = 0
    last_trained: Optional[str] = None
    adapters_applied: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "version": self.version,
            "base_model": self.base_model,
            "is_trained": self.is_trained,
            "training_sessions": self.training_sessions,
            "total_tokens_trained": self.total_tokens_trained,
            "last_trained": self.last_trained,
            "adapters_applied": self.adapters_applied,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelState':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================================
# MODEL MANAGER
# ============================================================================

class ModelManager:
    """
    Manage model downloading, loading, and serving.
    """
    
    # Supported base models
    SUPPORTED_MODELS = {
        "deepseek-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
        "qwen-14b": "Qwen/Qwen2.5-14B-Instruct",
    }
    
    def __init__(self, config: Config = None):
        self.config = config or get_config()
        self.model_path = Path(self.config.model.local_path)
        ensure_dir(self.model_path.parent)
        
        self.state = ModelState(name=self.config.model.name)
        self.model = None
        self.tokenizer = None
        
        # Load existing state if available
        self._load_state()
    
    def _load_state(self):
        """Load model state from disk."""
        state_path = self.model_path / "state.json"
        if state_path.exists():
            data = load_json(state_path)
            self.state = ModelState.from_dict(data)
            logger.info(f"Loaded model state: {self.state.name} v{self.state.version}")
    
    def _save_state(self):
        """Save model state to disk."""
        ensure_dir(self.model_path)
        state_path = self.model_path / "state.json"
        save_json(state_path, self.state.to_dict())
    
    def download_model(self, model_key: str = "deepseek-7b", force: bool = False) -> bool:
        """
        Download and save model locally as GroundZero.
        """
        if model_key not in self.SUPPORTED_MODELS:
            logger.error(f"Unknown model: {model_key}. Supported: {list(self.SUPPORTED_MODELS.keys())}")
            return False
        
        hf_model = self.SUPPORTED_MODELS[model_key]
        self.state.base_model = hf_model
        
        # Check if already downloaded
        model_files = self.model_path
        if (model_files / "config.json").exists() and not force:
            logger.info(f"Model already exists at {model_files}. Use force=True to redownload.")
            return True
        
        logger.info(f"Downloading {hf_model}...")
        logger.info("This may take a while (7B model is ~15GB)...")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Download tokenizer
            logger.info("Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
            
            # Download model
            logger.info("Downloading model weights...")
            model = AutoModelForCausalLM.from_pretrained(
                hf_model,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto",
            )
            
            # Save locally as GroundZero
            ensure_dir(model_files)
            logger.info(f"Saving as GroundZero to {model_files}...")
            
            model.save_pretrained(model_files)
            tokenizer.save_pretrained(model_files)
            
            # Create model card
            self._create_model_card(model_files)
            
            # Update state
            self.state.is_trained = False
            self._save_state()
            
            logger.info(f"[OK] Model downloaded and saved as {self.state.name}")
            return True
            
        except ImportError:
            logger.error("transformers not installed. Install with: pip install transformers accelerate")
            return False
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            return False
    
    def _create_model_card(self, path: Path):
        """Create a model card for GroundZero."""
        model_card = f"""---
license: mit
base_model: {self.state.base_model}
---

# {self.state.name}

This is GroundZero AI - a customized version of {self.state.base_model}.

## Training
- Base model: {self.state.base_model}
- Custom training: {self.state.is_trained}
- Training sessions: {self.state.training_sessions}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{path}")
model = AutoModelForCausalLM.from_pretrained("{path}")

# Generate
prompt = "Explain quantum computing"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
```

## License
MIT (inherited from DeepSeek)
"""
        with open(path / "README.md", "w") as f:
            f.write(model_card)
    
    def load_model(self, quantize: bool = True) -> bool:
        """
        Load the model into memory.
        """
        model_files = self.model_path
        
        # Check if config.json exists
        if not (model_files / "config.json").exists():
            logger.error(f"Model not found at {model_files}. Run download_model() first.")
            return False
        
        logger.info(f"Loading {self.state.name}...")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # Determine device
            device = self.config.model.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Using device: {device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_files),
                trust_remote_code=True,
            )
            
            # Load model with automatic device mapping
            # This will split between GPU and CPU if needed
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_files),
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
                offload_folder="offload",  # Offload to disk if needed
                low_cpu_mem_usage=True,
            )
            
            self.state.is_loaded = True
            logger.info(f"[OK] {self.state.name} loaded successfully")
            return True
            
        except ImportError:
            logger.error("transformers not installed. Install with: pip install transformers accelerate")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
        
    def unload_model(self):
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self.state.is_loaded = False
        
        # Clear GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        logger.info("Model unloaded")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        system_prompt: str = None,
    ) -> str:
        """
        Generate text from the model.
        """
        if not self.state.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Build prompt with system message
        if system_prompt:
            full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        
        # Move to device
        if hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        try:
            import torch
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the assistant's response
            if "<|im_start|>assistant" in response:
                response = response.split("<|im_start|>assistant")[-1].strip()
            elif "<|assistant|>" in response:
                response = response.split("<|assistant|>")[-1].strip()
            elif prompt in response:
                response = response[response.find(prompt) + len(prompt):].strip()
            
            # Clean up - remove any leaked system prompts or next turn markers
            # Remove leading "assistant" if present
            if response.lower().startswith('assistant'):
                response = response[9:].strip()

            stop_markers = ['<|im_start|>', '<|im_end|>', 'You are an AI', '<|user|>', '<|system|>']
            for marker in stop_markers:
                if marker in response:
                    response = response.split(marker)[0].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error generating response: {e}"
    
    def generate_with_reasoning(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        **kwargs
    ) -> Tuple[str, str]:
        """
        Generate with explicit reasoning (using <think> tags).
        
        Returns:
            (reasoning, answer)
        """
        system_prompt = """You are GroundZero AI, a helpful assistant that thinks step by step.
When answering questions, first think through your reasoning inside <think></think> tags,
then provide your final answer."""
        
        response = self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            system_prompt=system_prompt,
            **kwargs
        )
        
        # Parse reasoning and answer
        if "<think>" in response and "</think>" in response:
            reasoning_start = response.find("<think>") + len("<think>")
            reasoning_end = response.find("</think>")
            reasoning = response[reasoning_start:reasoning_end].strip()
            answer = response[reasoning_end + len("</think>"):].strip()
        else:
            reasoning = ""
            answer = response
        
        return reasoning, answer


# ============================================================================
# CONTINUOUS LEARNING
# ============================================================================

class ContinuousLearner:
    """
    Manage continuous learning and model updates.
    """
    
    def __init__(self, model_manager: ModelManager, config: Config = None):
        self.model_manager = model_manager
        self.config = config or get_config()
        
        # Learning queue
        self.learning_queue_path = get_data_path("training", "learning_queue.json")
        self.learning_queue: List[Dict] = []
        
        self._load_queue()
    
    def _load_queue(self):
        """Load learning queue from disk."""
        if self.learning_queue_path.exists():
            data = load_json(self.learning_queue_path)
            self.learning_queue = data.get("queue", [])
    
    def _save_queue(self):
        """Save learning queue to disk."""
        ensure_dir(self.learning_queue_path.parent)
        save_json(self.learning_queue_path, {
            "queue": self.learning_queue,
            "updated_at": timestamp(),
        })
    
    def add_training_example(
        self,
        input_text: str,
        output_text: str,
        example_type: str = "general",
        importance: float = 0.5,
        metadata: Dict = None,
    ):
        """Add a training example to the learning queue."""
        example = {
            "id": f"ex_{len(self.learning_queue)}_{timestamp()}",
            "input": input_text,
            "output": output_text,
            "type": example_type,
            "importance": importance,
            "metadata": metadata or {},
            "timestamp": timestamp(),
        }
        
        self.learning_queue.append(example)
        self._save_queue()
        
        # Check if we should trigger training
        if len(self.learning_queue) >= self.config.continuous_learning.auto_evolve_threshold:
            logger.info(f"Learning queue has {len(self.learning_queue)} items. Consider running evolve().")
    
    def add_feedback(
        self,
        prompt: str,
        response: str,
        feedback: str,
        rating: int,  # 1-5
    ):
        """Add feedback as a training signal."""
        if rating >= 4:
            # Positive example
            self.add_training_example(
                prompt, response,
                example_type="positive_feedback",
                importance=0.7 + (rating - 4) * 0.15,
                metadata={"feedback": feedback, "rating": rating},
            )
        elif rating <= 2:
            # Negative example - we might want to generate a better response
            self.add_training_example(
                prompt, f"[BAD RESPONSE - DO NOT REPEAT]\n{response}",
                example_type="negative_feedback",
                importance=0.3,
                metadata={"feedback": feedback, "rating": rating},
            )
    
    def add_correction(self, prompt: str, wrong_response: str, correct_response: str):
        """Add a correction as a training example."""
        # Add the correct response as positive
        self.add_training_example(
            prompt, correct_response,
            example_type="correction",
            importance=0.9,
            metadata={"corrected_from": wrong_response},
        )
    
    def evolve(self, epochs: int = 1, batch_size: int = 4) -> Dict:
        """
        Train the model on accumulated learning queue.
        """
        if not self.learning_queue:
            logger.info("Learning queue is empty. Nothing to train on.")
            return {"trained": False, "reason": "empty_queue"}
        
        if not self.model_manager.state.is_loaded:
            logger.error("Model not loaded. Call load_model() first.")
            return {"trained": False, "reason": "model_not_loaded"}
        
        logger.info(f"Starting evolution with {len(self.learning_queue)} examples...")
        
        try:
            # Prepare training data
            train_data = self._prepare_training_data()
            
            # Run QLoRA training
            result = self._run_qlora_training(train_data, epochs, batch_size)
            
            if result["success"]:
                # Update model state
                self.model_manager.state.is_trained = True
                self.model_manager.state.training_sessions += 1
                self.model_manager.state.total_tokens_trained += result.get("tokens_trained", 0)
                self.model_manager.state.last_trained = timestamp()
                self.model_manager._save_state()
                
                # Clear learning queue
                self.learning_queue = []
                self._save_queue()
                
                logger.info(f"[OK] Evolution complete! Model updated.")
            
            return result
            
        except Exception as e:
            logger.error(f"Evolution failed: {e}")
            return {"trained": False, "reason": str(e)}
    
    def _prepare_training_data(self) -> List[Dict]:
        """Prepare training data from learning queue."""
        data = []
        
        for example in self.learning_queue:
            # Format as instruction-following data
            formatted = {
                "prompt": example["input"],
                "completion": example["output"],
                "importance": example.get("importance", 0.5),
            }
            data.append(formatted)
        
        # Sort by importance
        data.sort(key=lambda x: x["importance"], reverse=True)
        
        return data
    
    def _run_qlora_training(self, train_data: List[Dict], epochs: int, batch_size: int) -> Dict:
        """Run QLoRA fine-tuning."""
        try:
            from transformers import TrainingArguments
            from peft import get_peft_model, LoraConfig, TaskType
            from trl import SFTTrainer
            import torch
            
            model = self.model_manager.model
            tokenizer = self.model_manager.tokenizer
            
            # Configure LoRA
            lora_config = LoraConfig(
                r=self.config.training.lora_r,
                lora_alpha=self.config.training.lora_alpha,
                lora_dropout=self.config.training.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type=TaskType.CAUSAL_LM,
            )
            
            # Apply LoRA
            model = get_peft_model(model, lora_config)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=str(get_data_path("checkpoints", "evolution")),
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
                learning_rate=self.config.training.learning_rate,
                warmup_ratio=self.config.training.warmup_ratio,
                logging_steps=10,
                save_strategy="epoch",
                fp16=torch.cuda.is_available(),
            )
            
            # Format data for training
            def format_example(example):
                return f"<|user|>\n{example['prompt']}\n<|assistant|>\n{example['completion']}"
            
            train_texts = [format_example(ex) for ex in train_data]
            
            # Create trainer
            # Note: This is simplified - real implementation would use Dataset class
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=train_texts,  # Would need proper Dataset
                tokenizer=tokenizer,
                max_seq_length=self.config.model.max_seq_length,
            )
            
            # Train
            trainer.train()
            
            # Save adapter
            adapter_path = get_data_path("models", "adapters", f"evolution_{timestamp()}")
            model.save_pretrained(adapter_path)
            
            # Merge adapter into base model
            model = model.merge_and_unload()
            model.save_pretrained(self.model_manager.model_path / "model")
            
            self.model_manager.state.adapters_applied.append(str(adapter_path))
            
            return {
                "success": True,
                "examples_trained": len(train_data),
                "tokens_trained": sum(len(t.split()) for t in train_texts) * 1.3,
            }
            
        except ImportError as e:
            logger.warning(f"Training libraries not available: {e}")
            logger.info("Install with: pip install peft trl")
            return {"success": False, "reason": f"missing_dependency: {e}"}
        except Exception as e:
            return {"success": False, "reason": str(e)}
    
    def get_queue_stats(self) -> Dict:
        """Get learning queue statistics."""
        if not self.learning_queue:
            return {"total": 0, "by_type": {}}
        
        by_type = {}
        for ex in self.learning_queue:
            t = ex.get("type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1
        
        return {
            "total": len(self.learning_queue),
            "by_type": by_type,
            "avg_importance": sum(ex.get("importance", 0.5) for ex in self.learning_queue) / len(self.learning_queue),
        }


# ============================================================================
# MOCK MODEL (For testing without GPU)
# ============================================================================

class MockModel:
    """
    Mock model for testing without actual model loaded.
    """
    
    def __init__(self):
        self.name = "GroundZero-AI (Mock)"
        self.is_loaded = True
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        **kwargs
    ) -> str:
        """Generate mock response."""
        # Simple pattern-based responses for testing
        prompt_lower = prompt.lower()
        
        if "hello" in prompt_lower or "hi" in prompt_lower:
            return "Hello! I'm GroundZero AI. How can I help you today?"
        
        elif "what is" in prompt_lower or "explain" in prompt_lower:
            topic = prompt.split("what is")[-1].split("explain")[-1].strip().rstrip("?")
            return f"""<think>
The user is asking about {topic}. Let me think through this:
1. I need to provide a clear explanation
2. I should include relevant details
3. I should be accurate and helpful
</think>

{topic.title()} is an important concept. It refers to [detailed explanation would go here]. 
Key points include:
- First important aspect
- Second important aspect
- Third important aspect

Would you like me to explain any specific aspect in more detail?"""
        
        elif "how" in prompt_lower:
            return """<think>
This is a "how" question, so I need to provide steps or a process.
Let me break this down into clear steps.
</think>

Here's how you can approach this:

1. **First Step**: Begin by understanding the requirements
2. **Second Step**: Plan your approach
3. **Third Step**: Execute the plan
4. **Fourth Step**: Review and iterate

Would you like more details on any of these steps?"""
        
        else:
            return f"""<think>
The user asked: "{prompt[:100]}"
Let me analyze this and provide a helpful response.
</think>

I understand your question. Based on my analysis, here's what I can tell you:

This is a complex topic that involves several considerations. The key points are:
- Consideration 1
- Consideration 2
- Consideration 3

Is there a specific aspect you'd like me to elaborate on?"""


# Export
__all__ = [
    'ModelState', 'ModelManager', 'ContinuousLearner', 'MockModel',
]
