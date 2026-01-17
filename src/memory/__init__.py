"""
GroundZero AI - Memory System
============================

Comprehensive memory for:
1. Conversation history (previous chats)
2. User profiles (name, preferences, behaviors)
3. Long-term memory (facts learned about users)
4. Working memory (current context)
5. Episodic memory (specific events/interactions)
"""

import json
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict

try:
    from ..utils import (
        get_data_path, ensure_dir, load_json, save_json,
        logger, timestamp, generate_id, Message, Conversation
    )
except ImportError:
    from utils import (
        get_data_path, ensure_dir, load_json, save_json,
        logger, timestamp, generate_id, Message, Conversation
    )


# ============================================================================
# USER PROFILE
# ============================================================================

@dataclass
class UserPreferences:
    """User preferences and settings."""
    response_style: str = "balanced"  # concise, balanced, detailed
    tone: str = "friendly"  # formal, friendly, casual
    expertise_level: str = "intermediate"  # beginner, intermediate, expert
    interests: List[str] = field(default_factory=list)
    dislikes: List[str] = field(default_factory=list)
    language: str = "en"
    timezone: str = "UTC"
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserPreferences':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class UserProfile:
    """Complete user profile."""
    id: str
    name: str = "User"
    nickname: str = None
    email: str = None
    preferences: UserPreferences = field(default_factory=UserPreferences)
    created_at: str = field(default_factory=timestamp)
    last_seen: str = field(default_factory=timestamp)
    total_conversations: int = 0
    total_messages: int = 0
    facts: Dict[str, str] = field(default_factory=dict)  # key -> value
    behaviors: Dict[str, Any] = field(default_factory=dict)  # behavior patterns
    favorites: Dict[str, List] = field(default_factory=dict)  # category -> items
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "nickname": self.nickname,
            "email": self.email,
            "preferences": self.preferences.to_dict() if isinstance(self.preferences, UserPreferences) else self.preferences,
            "created_at": self.created_at,
            "last_seen": self.last_seen,
            "total_conversations": self.total_conversations,
            "total_messages": self.total_messages,
            "facts": self.facts,
            "behaviors": self.behaviors,
            "favorites": self.favorites,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserProfile':
        prefs = data.get("preferences", {})
        if isinstance(prefs, dict):
            prefs = UserPreferences.from_dict(prefs)
        
        return cls(
            id=data.get("id", generate_id("user_")),
            name=data.get("name", "User"),
            nickname=data.get("nickname"),
            email=data.get("email"),
            preferences=prefs,
            created_at=data.get("created_at", timestamp()),
            last_seen=data.get("last_seen", timestamp()),
            total_conversations=data.get("total_conversations", 0),
            total_messages=data.get("total_messages", 0),
            facts=data.get("facts", {}),
            behaviors=data.get("behaviors", {}),
            favorites=data.get("favorites", {}),
            metadata=data.get("metadata", {}),
        )


# ============================================================================
# USER MANAGER
# ============================================================================

class UserManager:
    """Manage user profiles and data."""
    
    def __init__(self, users_path: str = None):
        self.users_path = Path(users_path) if users_path else get_data_path("users")
        ensure_dir(self.users_path)
        
        self.users: Dict[str, UserProfile] = {}
        self._load_users()
    
    def _load_users(self):
        """Load all user profiles."""
        for user_file in self.users_path.glob("*.json"):
            try:
                data = load_json(user_file)
                user = UserProfile.from_dict(data)
                self.users[user.id] = user
            except Exception as e:
                logger.error(f"Error loading user {user_file}: {e}")
        
        logger.info(f"Loaded {len(self.users)} user profiles")
    
    def get_user(self, user_id: str) -> Optional[UserProfile]:
        """Get user by ID, creating if needed."""
        if user_id not in self.users:
            self.users[user_id] = UserProfile(id=user_id)
        
        user = self.users[user_id]
        user.last_seen = timestamp()
        return user
    
    def save_user(self, user: UserProfile):
        """Save user profile to disk."""
        path = self.users_path / f"{user.id}.json"
        save_json(path, user.to_dict())
    
    def save_all(self):
        """Save all user profiles."""
        for user in self.users.values():
            self.save_user(user)
        logger.info(f"Saved {len(self.users)} user profiles")
    
    def update_user_fact(self, user_id: str, key: str, value: str):
        """Update a fact about the user."""
        user = self.get_user(user_id)
        user.facts[key] = value
        self.save_user(user)
    
    def get_user_context(self, user_id: str) -> str:
        """Get context string about a user for prompts."""
        user = self.get_user(user_id)
        
        context_parts = []
        
        # Basic info
        if user.name != "User":
            context_parts.append(f"User's name: {user.name}")
        if user.nickname:
            context_parts.append(f"Preferred name: {user.nickname}")
        
        # Preferences
        prefs = user.preferences
        context_parts.append(f"Communication style: {prefs.tone}, {prefs.response_style}")
        context_parts.append(f"Expertise level: {prefs.expertise_level}")
        
        if prefs.interests:
            context_parts.append(f"Interests: {', '.join(prefs.interests[:5])}")
        
        # Facts about user
        if user.facts:
            context_parts.append("Known facts about user:")
            for key, value in list(user.facts.items())[:10]:
                context_parts.append(f"  - {key}: {value}")
        
        # Favorites
        if user.favorites:
            for category, items in user.favorites.items():
                if items:
                    context_parts.append(f"Favorite {category}: {', '.join(items[:3])}")
        
        return "\n".join(context_parts)


# ============================================================================
# CONVERSATION MANAGER
# ============================================================================

class ConversationManager:
    """Manage conversation history."""
    
    def __init__(self, conversations_path: str = None, max_conversations: int = 10000):
        self.conversations_path = Path(conversations_path) if conversations_path else get_data_path("conversations")
        ensure_dir(self.conversations_path)
        
        self.max_conversations = max_conversations
        self.conversations: Dict[str, Conversation] = {}
        self.user_conversations: Dict[str, List[str]] = defaultdict(list)  # user_id -> conv_ids
        
        self._load_conversations()
    
    def _load_conversations(self):
        """Load conversation index."""
        index_path = self.conversations_path / "index.json"
        if index_path.exists():
            data = load_json(index_path)
            self.user_conversations = defaultdict(list, data.get("user_conversations", {}))
            
            # Load recent conversations into memory
            recent_ids = set()
            for conv_ids in self.user_conversations.values():
                recent_ids.update(conv_ids[-100:])  # Last 100 per user
            
            for conv_id in recent_ids:
                self._load_conversation(conv_id)
        
        logger.info(f"Loaded {len(self.conversations)} conversations")
    
    def _load_conversation(self, conv_id: str) -> Optional[Conversation]:
        """Load a single conversation from disk."""
        path = self.conversations_path / f"{conv_id}.json"
        if path.exists():
            data = load_json(path)
            conv = Conversation.from_dict(data)
            self.conversations[conv_id] = conv
            return conv
        return None
    
    def _save_index(self):
        """Save conversation index."""
        index_path = self.conversations_path / "index.json"
        save_json(index_path, {
            "user_conversations": dict(self.user_conversations),
            "total_conversations": sum(len(v) for v in self.user_conversations.values()),
            "updated_at": timestamp(),
        })
    
    def new_conversation(self, user_id: str = "default") -> Conversation:
        """Start a new conversation."""
        conv = Conversation(user_id=user_id)
        self.conversations[conv.id] = conv
        self.user_conversations[user_id].append(conv.id)
        return conv
    
    def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        if conv_id not in self.conversations:
            self._load_conversation(conv_id)
        return self.conversations.get(conv_id)
    
    def get_user_conversations(self, user_id: str, limit: int = 50) -> List[Conversation]:
        """Get conversations for a user."""
        conv_ids = self.user_conversations.get(user_id, [])[-limit:]
        
        conversations = []
        for conv_id in reversed(conv_ids):  # Most recent first
            conv = self.get_conversation(conv_id)
            if conv:
                conversations.append(conv)
        
        return conversations
    
    def add_message(self, conv_id: str, role: str, content: str, **kwargs) -> Message:
        """Add a message to a conversation."""
        conv = self.get_conversation(conv_id)
        if not conv:
            raise ValueError(f"Conversation not found: {conv_id}")
        
        msg = conv.add_message(role, content, **kwargs)
        self.save_conversation(conv)
        return msg
    
    def save_conversation(self, conv: Conversation):
        """Save a conversation to disk."""
        path = self.conversations_path / f"{conv.id}.json"
        save_json(path, conv.to_dict())
    
    def save_all(self):
        """Save all conversations and index."""
        for conv in self.conversations.values():
            self.save_conversation(conv)
        self._save_index()
        logger.info(f"Saved {len(self.conversations)} conversations")
    
    def search_conversations(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
    ) -> List[Tuple[Conversation, List[Message]]]:
        """Search conversations for matching messages."""
        results = []
        query_lower = query.lower()
        
        for conv in self.get_user_conversations(user_id, limit=100):
            matching_messages = []
            for msg in conv.messages:
                if isinstance(msg, Message) and query_lower in msg.content.lower():
                    matching_messages.append(msg)
            
            if matching_messages:
                results.append((conv, matching_messages))
        
        return results[:limit]
    
    def get_conversation_summary(self, conv_id: str) -> str:
        """Get a summary of a conversation."""
        conv = self.get_conversation(conv_id)
        if not conv:
            return ""
        
        # Get first and last messages
        messages = conv.messages
        if not messages:
            return "Empty conversation"
        
        first_msg = messages[0]
        first_content = first_msg.content if isinstance(first_msg, Message) else first_msg.get("content", "")
        
        return f"Conversation about: {first_content[:100]}... ({len(messages)} messages)"


# ============================================================================
# LONG-TERM MEMORY
# ============================================================================

@dataclass
class MemoryEntry:
    """A single memory entry."""
    id: str
    content: str
    memory_type: str  # fact, event, preference, behavior, skill
    importance: float = 0.5
    confidence: float = 1.0
    user_id: str = "default"
    source: str = "conversation"
    created_at: str = field(default_factory=timestamp)
    last_accessed: str = field(default_factory=timestamp)
    access_count: int = 0
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryEntry':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class LongTermMemory:
    """
    Long-term memory storage for facts, events, and learned information.
    """
    
    def __init__(self, path: str = None, max_memories: int = 100000):
        self.path = Path(path) if path else get_data_path("memory", "long_term.json")
        ensure_dir(self.path.parent)
        
        self.max_memories = max_memories
        self.memories: Dict[str, MemoryEntry] = {}
        self.user_memories: Dict[str, Set[str]] = defaultdict(set)  # user_id -> memory_ids
        self.type_index: Dict[str, Set[str]] = defaultdict(set)  # type -> memory_ids
        
        self._load()
    
    def _load(self):
        """Load memories from disk."""
        if self.path.exists():
            data = load_json(self.path)
            for mem_data in data.get("memories", []):
                mem = MemoryEntry.from_dict(mem_data)
                self.memories[mem.id] = mem
                self.user_memories[mem.user_id].add(mem.id)
                self.type_index[mem.memory_type].add(mem.id)
            
            logger.info(f"Loaded {len(self.memories)} long-term memories")
    
    def save(self):
        """Save memories to disk."""
        data = {
            "memories": [m.to_dict() for m in self.memories.values()],
            "total_memories": len(self.memories),
            "saved_at": timestamp(),
        }
        save_json(self.path, data)
    
    def add_memory(
        self,
        content: str,
        memory_type: str = "fact",
        importance: float = 0.5,
        user_id: str = "default",
        source: str = "conversation",
        metadata: Dict = None,
    ) -> MemoryEntry:
        """Add a new memory."""
        mem_id = generate_id("mem_")
        
        memory = MemoryEntry(
            id=mem_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            user_id=user_id,
            source=source,
            metadata=metadata or {},
        )
        
        self.memories[mem_id] = memory
        self.user_memories[user_id].add(mem_id)
        self.type_index[memory_type].add(mem_id)
        
        # Trim if over limit
        if len(self.memories) > self.max_memories:
            self._trim_memories()
        
        return memory
    
    def _trim_memories(self):
        """Remove least important/accessed memories."""
        # Score memories by importance * recency * access
        scored = []
        for mem in self.memories.values():
            # Days since last access
            try:
                last_access = datetime.fromisoformat(mem.last_accessed)
                days_ago = (datetime.now() - last_access).days
            except:
                days_ago = 30
            
            recency = 1.0 / (1 + days_ago * 0.1)
            score = mem.importance * recency * (1 + mem.access_count * 0.1)
            scored.append((score, mem.id))
        
        # Sort and remove bottom 10%
        scored.sort()
        to_remove = scored[:len(scored) // 10]
        
        for _, mem_id in to_remove:
            self.delete_memory(mem_id)
    
    def get_memory(self, mem_id: str) -> Optional[MemoryEntry]:
        """Get a memory by ID."""
        mem = self.memories.get(mem_id)
        if mem:
            mem.access_count += 1
            mem.last_accessed = timestamp()
        return mem
    
    def delete_memory(self, mem_id: str) -> bool:
        """Delete a memory."""
        if mem_id not in self.memories:
            return False
        
        mem = self.memories[mem_id]
        self.user_memories[mem.user_id].discard(mem_id)
        self.type_index[mem.memory_type].discard(mem_id)
        del self.memories[mem_id]
        return True
    
    def search_memories(
        self,
        query: str,
        user_id: str = None,
        memory_type: str = None,
        limit: int = 20,
    ) -> List[MemoryEntry]:
        """Search memories by content."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        results = []
        
        # Filter by user and type
        if user_id:
            candidate_ids = self.user_memories.get(user_id, set())
        else:
            candidate_ids = set(self.memories.keys())
        
        if memory_type:
            candidate_ids &= self.type_index.get(memory_type, set())
        
        for mem_id in candidate_ids:
            mem = self.memories.get(mem_id)
            if not mem:
                continue
            
            # Score by word overlap
            content_words = set(mem.content.lower().split())
            overlap = len(query_words & content_words)
            
            if overlap > 0 or query_lower in mem.content.lower():
                score = overlap + (1 if query_lower in mem.content.lower() else 0)
                results.append((score, mem))
        
        # Sort by score * importance
        results.sort(key=lambda x: x[0] * x[1].importance, reverse=True)
        
        return [mem for _, mem in results[:limit]]
    
    def get_user_memories(
        self,
        user_id: str,
        memory_type: str = None,
        limit: int = 50,
    ) -> List[MemoryEntry]:
        """Get all memories for a user."""
        mem_ids = self.user_memories.get(user_id, set())
        
        if memory_type:
            mem_ids &= self.type_index.get(memory_type, set())
        
        memories = [self.memories[mid] for mid in mem_ids if mid in self.memories]
        
        # Sort by importance and recency
        memories.sort(key=lambda m: (m.importance, m.last_accessed), reverse=True)
        
        return memories[:limit]
    
    def get_context(self, query: str, user_id: str = "default", max_tokens: int = 1000) -> str:
        """Get memory context for a query."""
        memories = self.search_memories(query, user_id=user_id, limit=20)
        
        context_parts = []
        token_count = 0
        
        for mem in memories:
            line = f"- [{mem.memory_type}] {mem.content}"
            tokens = len(line.split()) * 1.3
            
            if token_count + tokens > max_tokens:
                break
            
            context_parts.append(line)
            token_count += tokens
        
        return "\n".join(context_parts)


# ============================================================================
# WORKING MEMORY
# ============================================================================

class WorkingMemory:
    """
    Short-term working memory for current conversation context.
    """
    
    def __init__(self, max_items: int = 20):
        self.max_items = max_items
        self.items: List[Dict] = []
        self.current_topic: Optional[str] = None
        self.current_intent: Optional[str] = None
        self.pending_actions: List[Dict] = []
        self.context_vars: Dict[str, Any] = {}
    
    def add(self, item_type: str, content: Any, importance: float = 0.5):
        """Add item to working memory."""
        self.items.append({
            "type": item_type,
            "content": content,
            "importance": importance,
            "timestamp": timestamp(),
        })
        
        # Trim to max
        if len(self.items) > self.max_items:
            # Remove least important
            self.items.sort(key=lambda x: x["importance"])
            self.items = self.items[1:]
    
    def get_recent(self, n: int = 5, item_type: str = None) -> List[Dict]:
        """Get recent items."""
        items = self.items
        if item_type:
            items = [i for i in items if i["type"] == item_type]
        return items[-n:]
    
    def clear(self):
        """Clear working memory."""
        self.items = []
        self.current_topic = None
        self.current_intent = None
        self.pending_actions = []
        self.context_vars = {}
    
    def set_context(self, key: str, value: Any):
        """Set a context variable."""
        self.context_vars[key] = value
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a context variable."""
        return self.context_vars.get(key, default)
    
    def to_string(self) -> str:
        """Convert working memory to context string."""
        parts = []
        
        if self.current_topic:
            parts.append(f"Current topic: {self.current_topic}")
        
        if self.current_intent:
            parts.append(f"User intent: {self.current_intent}")
        
        if self.context_vars:
            parts.append("Context variables:")
            for k, v in list(self.context_vars.items())[:10]:
                parts.append(f"  {k}: {v}")
        
        recent = self.get_recent(5)
        if recent:
            parts.append("Recent context:")
            for item in recent:
                parts.append(f"  [{item['type']}] {str(item['content'])[:100]}")
        
        return "\n".join(parts)


# ============================================================================
# MAIN MEMORY SYSTEM
# ============================================================================

class MemorySystem:
    """
    Unified memory system integrating all memory types.
    """
    
    def __init__(self):
        self.users = UserManager()
        self.conversations = ConversationManager()
        self.long_term = LongTermMemory()
        self.working = WorkingMemory()
        
        self.current_user_id: str = "default"
        self.current_conversation: Optional[Conversation] = None
    
    def set_user(self, user_id: str) -> UserProfile:
        """Set current user."""
        self.current_user_id = user_id
        return self.users.get_user(user_id)
    
    def get_current_user(self) -> UserProfile:
        """Get current user profile."""
        return self.users.get_user(self.current_user_id)
    
    def start_conversation(self) -> Conversation:
        """Start a new conversation."""
        self.current_conversation = self.conversations.new_conversation(self.current_user_id)
        self.working.clear()
        
        # Update user stats
        user = self.get_current_user()
        user.total_conversations += 1
        self.users.save_user(user)
        
        return self.current_conversation
    
    def add_turn(self, user_message: str, assistant_response: str):
        """Add a conversation turn."""
        if not self.current_conversation:
            self.start_conversation()
        
        conv = self.current_conversation
        conv.add_message("user", user_message)
        conv.add_message("assistant", assistant_response)
        
        # Update working memory
        self.working.add("user_input", user_message, 0.8)
        self.working.add("assistant_response", assistant_response, 0.6)
        
        # Update user stats
        user = self.get_current_user()
        user.total_messages += 2
        
        # Extract and store facts from conversation
        self._extract_memories(user_message, assistant_response)
        
        # Save
        self.conversations.save_conversation(conv)
        self.users.save_user(user)
    
    def _extract_memories(self, user_message: str, assistant_response: str):
        """Extract memories from conversation."""
        # Look for patterns like "my name is", "I am", "I like", etc.
        patterns = [
            (r"my name is (\w+)", "name", "fact"),
            (r"i am (\w+)", "identity", "fact"),
            (r"i like (\w+(?:\s+\w+)*)", "likes", "preference"),
            (r"i prefer (\w+(?:\s+\w+)*)", "preferences", "preference"),
            (r"i work at (\w+(?:\s+\w+)*)", "workplace", "fact"),
            (r"i live in (\w+(?:\s+\w+)*)", "location", "fact"),
            (r"my favorite (\w+) is (\w+(?:\s+\w+)*)", "favorite", "preference"),
        ]
        
        text = user_message.lower()
        user = self.get_current_user()
        
        for pattern, key, mem_type in patterns:
            match = re.search(pattern, text)
            if match:
                if key == "name":
                    user.name = match.group(1).title()
                    self.users.save_user(user)
                elif key == "favorite":
                    category = match.group(1)
                    value = match.group(2)
                    if category not in user.favorites:
                        user.favorites[category] = []
                    if value not in user.favorites[category]:
                        user.favorites[category].append(value)
                    self.users.save_user(user)
                else:
                    value = match.group(1) if len(match.groups()) == 1 else match.group(2)
                    user.facts[key] = value
                    self.users.save_user(user)
                    
                    # Also add to long-term memory
                    self.long_term.add_memory(
                        f"User's {key}: {value}",
                        memory_type=mem_type,
                        user_id=self.current_user_id,
                        importance=0.7,
                    )
    
    def remember(self, content: str, memory_type: str = "fact", importance: float = 0.5):
        """Explicitly remember something."""
        return self.long_term.add_memory(
            content,
            memory_type=memory_type,
            user_id=self.current_user_id,
            importance=importance,
        )
    
    def recall(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Recall memories related to a query."""
        return self.long_term.search_memories(
            query,
            user_id=self.current_user_id,
            limit=limit,
        )
    
    def get_full_context(self, query: str) -> str:
        """Get complete context for a query."""
        parts = []
        
        # User context
        user_ctx = self.users.get_user_context(self.current_user_id)
        if user_ctx:
            parts.append("=== User Profile ===")
            parts.append(user_ctx)
        
        # Working memory
        working_ctx = self.working.to_string()
        if working_ctx:
            parts.append("\n=== Current Context ===")
            parts.append(working_ctx)
        
        # Long-term memory
        memory_ctx = self.long_term.get_context(query, self.current_user_id)
        if memory_ctx:
            parts.append("\n=== Relevant Memories ===")
            parts.append(memory_ctx)
        
        # Recent conversation
        if self.current_conversation:
            recent_msgs = self.current_conversation.messages[-2:]
            if recent_msgs:
                parts.append("\n=== Recent Conversation ===")
                for msg in recent_msgs:
                    if isinstance(msg, Message):
                        parts.append(f"{msg.role}: {msg.content[:200]}")
                    else:
                        parts.append(f"{msg.get('role', '?')}: {msg.get('content', '')[:200]}")
        
        return "\n".join(parts)
    
    def save_all(self):
        """Save all memory systems."""
        self.users.save_all()
        self.conversations.save_all()
        self.long_term.save()
        logger.info("All memory systems saved")


# Export
__all__ = [
    'UserPreferences', 'UserProfile', 'UserManager',
    'ConversationManager',
    'MemoryEntry', 'LongTermMemory',
    'WorkingMemory',
    'MemorySystem',
]
