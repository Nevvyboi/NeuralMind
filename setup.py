#!/usr/bin/env python3
"""
GroundZero AI - Setup Script
============================

This script sets up GroundZero AI:
1. Installs dependencies
2. Downloads the model
3. Initializes data directories
4. Runs initial tests
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd: list, description: str = None) -> bool:
    """Run a command and return success status."""
    if description:
        print(f"\n{'='*60}")
        print(f"  {description}")
        print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False
    except FileNotFoundError:
        print(f"Command not found: {cmd[0]}")
        return False


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version < (3, 9):
        print("Error: Python 3.9+ required")
        return False
    return True


def check_gpu():
    """Check if GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
            return True
        else:
            print("No GPU detected. Will use CPU (slower).")
            return False
    except ImportError:
        print("PyTorch not installed yet.")
        return None


def install_dependencies():
    """Install Python dependencies."""
    print("\nInstalling dependencies...")
    
    # Core dependencies
    core_deps = [
        "torch",
        "transformers",
        "accelerate",
        "peft",
        "trl",
        "bitsandbytes",
        "datasets",
        "sentencepiece",
        "protobuf",
    ]
    
    # Optional dependencies
    optional_deps = [
        "flask",
        "flask-cors",
        "pyyaml",
        "numpy",
        "wikipedia",
        "duckduckgo-search",
        "arxiv",
        "requests",
    ]
    
    # Install core
    print("\nInstalling core dependencies (this may take a while)...")
    for dep in core_deps:
        print(f"  Installing {dep}...")
        run_command([sys.executable, "-m", "pip", "install", dep, "-q"])
    
    # Install optional
    print("\nInstalling optional dependencies...")
    for dep in optional_deps:
        print(f"  Installing {dep}...")
        run_command([sys.executable, "-m", "pip", "install", dep, "-q"])
    
    return True


def setup_directories():
    """Create required directories."""
    print("\nSetting up directories...")
    
    dirs = [
        "data/knowledge",
        "data/memory",
        "data/models/groundzero",
        "data/training",
        "data/cache",
        "data/embeddings",
        "data/checkpoints",
        "data/users",
        "data/conversations",
        "logs",
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"  Created: {d}")
    
    return True


def download_model(model_key: str = "deepseek-7b"):
    """Download the model."""
    print(f"\nDownloading model: {model_key}")
    print("This will download approximately 15GB of data...")
    print("Press Ctrl+C to cancel.\n")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_map = {
            "deepseek-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "deepseek-14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        }
        
        hf_model = model_map.get(model_key, model_map["deepseek-7b"])
        local_path = Path("data/models/groundzero/model")
        
        print(f"Downloading from: {hf_model}")
        print(f"Saving to: {local_path}")
        
        # Download tokenizer
        print("\nDownloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
        
        # Download model
        print("Downloading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            hf_model,
            trust_remote_code=True,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
        
        # Save locally
        print("Saving model locally...")
        local_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(local_path)
        tokenizer.save_pretrained(local_path)
        
        print(f"\n✓ Model downloaded and saved as GroundZero!")
        return True
        
    except KeyboardInterrupt:
        print("\nDownload cancelled.")
        return False
    except Exception as e:
        print(f"\nError downloading model: {e}")
        return False


def test_system():
    """Run basic tests."""
    print("\nRunning system tests...")
    
    try:
        # Test imports
        print("  Testing imports...")
        from groundzero import GroundZeroAI
        print("    ✓ Main module")
        
        from src.knowledge import KnowledgeGraph
        print("    ✓ Knowledge graph")
        
        from src.memory import MemorySystem
        print("    ✓ Memory system")
        
        from src.search import WebSearch
        print("    ✓ Web search")
        
        from src.reasoning import ReasoningEngine
        print("    ✓ Reasoning engine")
        
        # Test initialization
        print("\n  Testing initialization (mock mode)...")
        ai = GroundZeroAI(use_mock=True)
        ai.setup(download_model=False)
        
        # Test chat
        print("  Testing chat...")
        response = ai.chat("Hello, who are you?")
        print(f"    Response: {response['response'][:100]}...")
        
        # Test knowledge
        print("  Testing knowledge graph...")
        ai.teach("Python", "Python is a programming language created by Guido van Rossum.")
        
        # Test status
        status = ai.get_status()
        print(f"    Knowledge nodes: {status['knowledge_nodes']}")
        
        print("\n✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main setup function."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                    GroundZero AI Setup                        ║
║                                                               ║
║  An AI that learns and grows from your conversations          ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Check GPU
    gpu_available = check_gpu()
    
    # Ask what to do
    print("\n" + "="*60)
    print("Setup Options:")
    print("="*60)
    print("1. Full setup (install deps + download model)")
    print("2. Quick setup (install deps only, use mock model)")
    print("3. Just download model")
    print("4. Run tests only")
    print("5. Exit")
    
    choice = input("\nEnter choice [1-5]: ").strip()
    
    if choice == "1":
        # Full setup
        setup_directories()
        install_dependencies()
        
        if gpu_available is not False:
            print("\n" + "="*60)
            model_choice = input("Download model? This requires ~15GB disk and ~8GB GPU RAM [y/N]: ").strip().lower()
            if model_choice == 'y':
                download_model()
        
        test_system()
        
    elif choice == "2":
        # Quick setup
        setup_directories()
        install_dependencies()
        test_system()
        
    elif choice == "3":
        # Just download
        download_model()
        
    elif choice == "4":
        # Just test
        test_system()
        
    else:
        print("Exiting.")
        return 0
    
    print("\n" + "="*60)
    print("Setup complete!")
    print("="*60)
    print("\nTo start using GroundZero AI:")
    print("  python groundzero.py --chat     # Start chatting")
    print("  python groundzero.py --dashboard  # Start web interface")
    print("  python groundzero.py --status   # Check status")
    print("\nFor more options: python groundzero.py --help")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
