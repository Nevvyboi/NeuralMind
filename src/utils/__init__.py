"""
GroundZero AI - Utilities Module
================================

Configuration management, logging, and common utilities.
"""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
from dataclasses import dataclass, field, asdict
from logging.handlers import RotatingFileHandler
import hashlib

import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ============================================================================
# PATH MANAGEMENT
# ============================================================================

def get_project_root() -> Path:
    """Get the project root directory."""
    # Look for config directory to identify root
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "config").exists() or (current / "config.yaml").exists():
            return current
        # Also check if we're in src/
        if current.name == "src" and (current.parent / "config").exists():
            return current.parent
        current = current.parent
    # Default to two levels up from this file
    return Path(__file__).resolve().parent.parent.parent


PROJECT_ROOT = get_project_root()


def get_data_path(*parts) -> Path:
    """Get path within data directory."""
    return PROJECT_ROOT / "data" / Path(*parts)


def get_config_path(*parts) -> Path:
    """Get path within config directory."""
    return PROJECT_ROOT / "config" / Path(*parts)


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    name: str = "GroundZero-AI"
    version: str = "1.0.0"
    base_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    local_path: str = "data/models/groundzero"
    max_seq_length: int = 4096
    device: str = "auto"
    quantization: str = "4bit"


@dataclass
class TrainingConfig:
    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05


@dataclass
class KnowledgeConfig:
    graph_path: str = "data/knowledge/knowledge_graph.json"
    embeddings_path: str = "data/embeddings/knowledge_embeddings.npy"
    min_confidence: float = 0.7
    max_nodes: int = 1000000
    auto_verify: bool = True
    connect_threshold: float = 0.8


@dataclass
class MemoryConfig:
    conversation_limit: int = 10000
    user_profile_path: str = "data/users"
    long_term_path: str = "data/memory/long_term.json"
    working_memory_size: int = 20
    episodic_memory_size: int = 1000
    semantic_memory_size: int = 100000


@dataclass
class SearchConfig:
    engines: list = field(default_factory=lambda: ["duckduckgo", "wikipedia", "arxiv"])
    max_results: int = 10
    cache_duration_hours: int = 24
    verify_sources: bool = True
    min_source_reliability: float = 0.6


@dataclass
class ContinuousLearningConfig:
    enabled: bool = True
    learning_queue_path: str = "data/training/learning_queue.json"
    auto_evolve_threshold: int = 100
    feedback_learning: bool = True
    correction_learning: bool = True
    web_learning: bool = True


@dataclass
class DashboardConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    secret_key: str = "change-this-in-production"


@dataclass 
class Config:
    """Main configuration container."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    continuous_learning: ContinuousLearningConfig = field(default_factory=ContinuousLearningConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    
    @classmethod
    def load(cls, path: str = None) -> 'Config':
        """Load configuration from YAML file."""
        if path is None:
            path = get_config_path("config.yaml")
        
        config = cls()
        
        if Path(path).exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            
            # Update each section
            if "model" in data:
                config.model = ModelConfig(**{k: v for k, v in data["model"].items() 
                                              if k in ModelConfig.__dataclass_fields__})
            if "training" in data:
                config.training = TrainingConfig(**{k: v for k, v in data["training"].items()
                                                    if k in TrainingConfig.__dataclass_fields__})
            if "knowledge" in data:
                config.knowledge = KnowledgeConfig(**{k: v for k, v in data["knowledge"].items()
                                                     if k in KnowledgeConfig.__dataclass_fields__})
            if "memory" in data:
                config.memory = MemoryConfig(**{k: v for k, v in data["memory"].items()
                                               if k in MemoryConfig.__dataclass_fields__})
            if "search" in data:
                config.search = SearchConfig(**{k: v for k, v in data["search"].items()
                                               if k in SearchConfig.__dataclass_fields__})
            if "continuous_learning" in data:
                config.continuous_learning = ContinuousLearningConfig(
                    **{k: v for k, v in data["continuous_learning"].items()
                       if k in ContinuousLearningConfig.__dataclass_fields__})
            if "dashboard" in data:
                config.dashboard = DashboardConfig(**{k: v for k, v in data["dashboard"].items()
                                                     if k in DashboardConfig.__dataclass_fields__})
        
        return config
    
    def save(self, path: str = None):
        """Save configuration to YAML file."""
        if path is None:
            path = get_config_path("config.yaml")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "model": asdict(self.model),
            "training": asdict(self.training),
            "knowledge": asdict(self.knowledge),
            "memory": asdict(self.memory),
            "search": asdict(self.search),
            "continuous_learning": asdict(self.continuous_learning),
            "dashboard": asdict(self.dashboard),
        }
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = Config.load()
    return _config


def reload_config():
    """Reload configuration from file."""
    global _config
    _config = Config.load()
    return _config


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(
    name: str = "groundzero",
    level: str = "INFO",
    log_file: str = None
) -> logging.Logger:
    """Set up logging with file and console handlers."""
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console.setFormatter(console_format)
    logger.addHandler(console)
    
    # File handler
    if log_file:
        log_path = PROJECT_ROOT / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=100*1024*1024,  # 100MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


# Default logger
logger = setup_logging("groundzero", log_file="logs/groundzero.log")


# ============================================================================
# COMMON UTILITIES
# ============================================================================

def generate_id(prefix: str = "") -> str:
    """Generate a unique ID."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    hash_part = hashlib.md5(f"{timestamp}{os.urandom(8)}".encode()).hexdigest()[:8]
    return f"{prefix}{timestamp}_{hash_part}" if prefix else f"{timestamp}_{hash_part}"


def timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().isoformat()


def load_json(path: Path) -> Dict:
    """Load JSON file safely."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_json(path: Path, data: Dict, indent: int = 2):
    """Save data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text with suffix."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


class Timer:
    """Simple timer context manager."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, *args):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        logger.info(f"{self.name} completed in {duration:.2f}s")
    
    @property
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()


# ============================================================================
# DATA CLASSES FOR COMMON STRUCTURES
# ============================================================================

@dataclass
class Message:
    """A chat message."""
    role: str  # user, assistant, system
    content: str
    timestamp: str = field(default_factory=timestamp)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Message':
        return cls(**data)


@dataclass
class Conversation:
    """A conversation with messages."""
    id: str = field(default_factory=lambda: generate_id("conv_"))
    user_id: str = "default"
    messages: list = field(default_factory=list)
    created_at: str = field(default_factory=timestamp)
    updated_at: str = field(default_factory=timestamp)
    metadata: Dict = field(default_factory=dict)
    
    def add_message(self, role: str, content: str, **kwargs):
        msg = Message(role=role, content=content, **kwargs)
        self.messages.append(msg)
        self.updated_at = timestamp()
        return msg
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "messages": [m.to_dict() if isinstance(m, Message) else m for m in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Conversation':
        conv = cls(
            id=data.get("id", generate_id("conv_")),
            user_id=data.get("user_id", "default"),
            created_at=data.get("created_at", timestamp()),
            updated_at=data.get("updated_at", timestamp()),
            metadata=data.get("metadata", {}),
        )
        for msg_data in data.get("messages", []):
            if isinstance(msg_data, dict):
                conv.messages.append(Message.from_dict(msg_data))
            else:
                conv.messages.append(msg_data)
        return conv


# Export all
__all__ = [
    'PROJECT_ROOT', 'get_project_root', 'get_data_path', 'get_config_path', 'ensure_dir',
    'Config', 'ModelConfig', 'TrainingConfig', 'KnowledgeConfig', 'MemoryConfig',
    'SearchConfig', 'ContinuousLearningConfig', 'DashboardConfig',
    'get_config', 'reload_config',
    'setup_logging', 'logger',
    'generate_id', 'timestamp', 'load_json', 'save_json', 'truncate_text', 'Timer',
    'Message', 'Conversation',
]
