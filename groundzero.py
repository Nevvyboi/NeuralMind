"""
GroundZero AI - Main System
===========================

The unified AI system that integrates:
- Model management & inference
- Knowledge graph
- Memory system
- Web search & verification
- Reasoning engine
- Continuous learning
- Dashboard
- FILE UPLOAD IN CHAT (NEW!)

This is THE main interface for GroundZero AI.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    Config, get_config, logger, timestamp, generate_id,
    Message, Conversation, ensure_dir, get_data_path
)
from src.core import ModelManager, ContinuousLearner, MockModel, ModelState
from src.knowledge import KnowledgeGraph, KnowledgeExtractor
from src.memory import MemorySystem, UserProfile
from src.search import WebSearch, SearchResult
from src.reasoning import ReasoningEngine, ReasoningTrace
from src.continuous_learning import ContinuousLearningSystem


class GroundZeroAI:
    """
    GroundZero AI - Your Personal Evolving AI Assistant
    
    Features:
    - Natural conversation with memory
    - Knowledge graph for storing and connecting facts
    - Web search when knowledge is missing
    - Verification of facts and claims
    - Learning from corrections
    - Continuous improvement from interactions
    - Reasoning visualization
    - FILE UPLOAD IN CHAT (like Claude!)
    
    Example:
        ai = GroundZeroAI()
        ai.setup()  # Download model, initialize systems
        
        response = ai.chat("What is machine learning?")
        print(response['response'])
        
        # Chat with a file
        response = ai.chat("Analyze this data", file_content="col1,col2\n1,2\n3,4")
        
        # Learn a new topic
        ai.learn_topic("quantum computing")
        
        # Handle correction
        ai.handle_correction("Wrong answer", "The correct answer is...")
        
        # Run evolution
        ai.evolve()
    """
    
    def __init__(self, config: Config = None, use_mock: bool = False):
        """
        Initialize GroundZero AI.
        
        Args:
            config: Configuration object (loads from config.yaml if None)
            use_mock: Use mock model for testing (no GPU required)
        """
        self.config = config or get_config()
        self.use_mock = use_mock
        
        # Core components
        self.model_manager: Optional[ModelManager] = None
        self.knowledge_graph: Optional[KnowledgeGraph] = None
        self.memory: Optional[MemorySystem] = None
        self.web_search: Optional[WebSearch] = None
        self.reasoning: Optional[ReasoningEngine] = None
        self.continuous_learning: Optional[ContinuousLearningSystem] = None
        
        # Tools for file handling
        self.tools = None
        
        # State
        self.is_initialized = False
        self.current_user_id = "default"
        self.current_conversation_id: Optional[str] = None
        
        # Statistics
        self.stats = {
            "total_chats": 0,
            "topics_learned": 0,
            "corrections_applied": 0,
            "evolutions": 0,
        }
        
        logger.info(f"GroundZero AI initialized (mock={use_mock})")
    
    def setup(self, download_model: bool = True) -> bool:
        """
        Setup all components.
        
        Args:
            download_model: Whether to download the model if not present
        
        Returns:
            True if setup successful
        """
        logger.info("Setting up GroundZero AI...")
        
        try:
            # 1. Initialize Knowledge Graph
            logger.info("Initializing Knowledge Graph...")
            self.knowledge_graph = KnowledgeGraph()
            
            # 2. Initialize Memory System
            logger.info("Initializing Memory System...")
            self.memory = MemorySystem()
            
            # 3. Initialize Web Search
            logger.info("Initializing Web Search...")
            self.web_search = WebSearch()
            
            # 4. Initialize Model Manager
            logger.info("Initializing Model Manager...")
            self.model_manager = ModelManager(self.config)
            
            # Download model if needed
            model_path = self.model_manager.model_path / "model"
            if download_model and not model_path.exists() and not self.use_mock:
                logger.info("Model not found. Downloading...")
                logger.info("This will download ~15GB. You can skip with download_model=False")
                success = self.model_manager.download_model()
                if not success:
                    logger.warning("Model download failed. Using mock mode.")
                    self.use_mock = True
            
            # Load model (or use mock)
            if not self.use_mock and model_path.exists():
                logger.info("Loading model...")
                self.model_manager.load_model()
            else:
                logger.info("Using mock model for testing")
                self.model_manager.model = MockModel()
                self.model_manager.state.is_loaded = True
            
            # 5. Initialize Reasoning Engine
            logger.info("Initializing Reasoning Engine...")
            self.reasoning = ReasoningEngine(
                model_generate=self._generate,
                knowledge_graph=self.knowledge_graph,
                web_search=self.web_search,
                memory_system=self.memory,
            )
            
            # 6. Initialize Continuous Learning
            logger.info("Initializing Continuous Learning...")
            self.continuous_learning = ContinuousLearningSystem(
                model_generate=self._generate,
                knowledge_graph=self.knowledge_graph,
                memory_system=self.memory,
                web_search=self.web_search,
            )
            
            # 7. Initialize Tools (for file handling)
            logger.info("Initializing Tools...")
            try:
                from src.tools import ToolsManager
                self.tools = ToolsManager(model_generate=self._generate)
                logger.info("Tools initialized")
            except Exception as e:
                logger.warning(f"Tools initialization failed: {e}")
                self.tools = None
            
            self.is_initialized = True
            logger.info("GroundZero AI setup complete!")
            
            return True
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _generate(self, prompt: str, **kwargs) -> str:
        """Internal method to generate from model."""
        if self.use_mock or isinstance(self.model_manager.model, MockModel):
            return self.model_manager.model.generate(prompt, **kwargs)
        else:
            return self.model_manager.generate(prompt, **kwargs)
    
    # ========================================================================
    # CHAT INTERFACE - UPGRADED WITH FILE SUPPORT
    # ========================================================================
    
    def chat(
        self,
        message: str,
        user_id: str = None,
        conversation_id: str = None,
        return_reasoning: bool = True,
        search_web: bool = False,  # Disabled by default to avoid mixing
        file_content: str = None,  # NEW: File content for inline analysis
        file_name: str = None,     # NEW: File name for context
    ) -> Dict[str, Any]:
        """
        Chat with GroundZero AI.
        
        UPGRADED: Now supports file uploads directly in chat!
        
        Args:
            message: User's message
            user_id: User ID for memory
            conversation_id: Existing conversation ID (or creates new)
            return_reasoning: Include reasoning steps in response
            search_web: Allow web search if knowledge is missing
            file_content: Content of an uploaded file to analyze (NEW)
            file_name: Name of the uploaded file (NEW)
        
        Returns:
            {
                'response': str,
                'reasoning': List[dict] (if return_reasoning),
                'conversation_id': str,
                'confidence': float,
                'sources': List[str] (if used web search),
                'file_analyzed': str (if file was provided),
            }
        """
        if not self.is_initialized:
            return {"response": "Please run setup() first.", "error": True}
        
        # Set user
        if user_id:
            self.current_user_id = user_id
        self.memory.set_user(self.current_user_id)
        
        # Handle conversation
        if conversation_id:
            self.current_conversation_id = conversation_id
            conv = self.memory.conversations.get_conversation(conversation_id)
            if not conv:
                conv = self.memory.start_conversation()
                self.current_conversation_id = conv.id
        else:
            conv = self.memory.start_conversation()
            self.current_conversation_id = conv.id
        
        # Build the prompt - UPGRADED to handle file content
        if file_content:
            # User uploaded a file - include its content
            file_preview = file_content[:8000]  # Limit to ~8000 chars
            prompt = f"""The user has uploaded a file called "{file_name or 'document'}".

FILE CONTENT:
{file_preview}

USER QUESTION: {message}

Please analyze the file content and answer the user's question based on what you see in the file:"""
        else:
            # Regular chat - just use the message directly (no context mixing!)
            prompt = message
        
        # Generate response
        response = self._generate(prompt)
        
        # Generate reasoning trace for display (optional)
        reasoning_steps = []
        confidence = 0.8
        
        if return_reasoning:
            try:
                trace = self.reasoning.reason(
                    message,
                    context="",  # Empty context to avoid mixing
                    reasoning_type=self._classify_query(message),
                )
                reasoning_steps = [
                    {
                        "step": s.step_number,
                        "thought": s.thought,
                        "type": s.reasoning_type,
                    }
                    for s in trace.steps
                ]
                confidence = trace.confidence
                # Override with our direct response
                trace.final_answer = response
            except Exception as e:
                logger.warning(f"Reasoning failed: {e}")
        
        # Store in memory (minimal - just for history)
        try:
            self.memory.add_turn(message, response)
        except:
            pass
        
        # Update stats
        self.stats["total_chats"] += 1
        
        result = {
            "response": response,
            "conversation_id": self.current_conversation_id,
            "confidence": confidence,
        }
        
        if return_reasoning:
            result["reasoning"] = reasoning_steps
        
        if file_content:
            result["file_analyzed"] = file_name or "uploaded_file"
        
        return result
    
    def chat_with_file(self, message: str, filepath: str, **kwargs) -> Dict[str, Any]:
        """
        Chat with a specific file loaded.
        
        Args:
            message: Your question about the file
            filepath: Path to the file to analyze
            **kwargs: Additional args passed to chat()
        
        Returns:
            Chat response dict
        
        Example:
            response = ai.chat_with_file("What are the total sales?", "sales.csv")
        """
        # Read the file
        file_content = ""
        file_name = Path(filepath).name
        
        try:
            if self.tools:
                result = self.tools.read_file(filepath)
                if result.success:
                    doc_id = result.result.get("id")
                    doc = self.tools.loaded_docs.get(doc_id)
                    if doc and hasattr(doc, 'raw_content'):
                        file_content = doc.raw_content
            
            if not file_content:
                # Fallback: read directly
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    file_content = f.read()
        except Exception as e:
            return {"response": f"Error reading file: {e}", "error": True}
        
        return self.chat(
            message=message,
            file_content=file_content,
            file_name=file_name,
            **kwargs
        )
    
    def _should_search(self, message: str, knowledge_context: str) -> bool:
        """Determine if web search is needed."""
        # Search indicators
        search_words = ["current", "latest", "recent", "today", "now", "news", "2024", "2025"]
        
        # Check if message contains search indicators
        message_lower = message.lower()
        if any(word in message_lower for word in search_words):
            return True
        
        # Check if we have relevant knowledge
        if len(knowledge_context.strip()) < 50:
            return True
        
        return False
    
    def _classify_query(self, message: str) -> str:
        """Classify query type for reasoning."""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["code", "program", "function", "bug", "error"]):
            return "coding"
        elif any(word in message_lower for word in ["calculate", "math", "equation", "solve"]):
            return "math"
        elif any(word in message_lower for word in ["analyze", "compare", "evaluate"]):
            return "analysis"
        else:
            return "general"
    
    # ========================================================================
    # DOCUMENT METHODS
    # ========================================================================
    
    def read_document(self, filepath: str) -> Dict:
        """Read and process a document."""
        if self.tools:
            result = self.tools.read_file(filepath)
            if result.success:
                return result.result
            else:
                return {"error": result.error}
        return {"error": "Tools not initialized"}
    
    def get_loaded_documents(self) -> List[Dict]:
        """Get list of loaded documents."""
        if self.tools:
            docs = []
            for doc_id, doc in self.tools.loaded_docs.items():
                if hasattr(doc, 'filename'):
                    docs.append({
                        'id': doc_id,
                        'filename': doc.filename,
                        'file_type': getattr(doc, 'file_type', 'unknown'),
                        'word_count': getattr(doc, 'word_count', 0),
                    })
            return docs
        return []
    
    def ask_documents(self, question: str) -> str:
        """Ask a question about loaded documents."""
        if not self.tools:
            return "Tools not initialized"
        
        # Load any recent uploads if not already loaded
        try:
            uploads_path = get_data_path("uploads")
            if uploads_path.exists():
                import glob
                files = sorted(uploads_path.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
                for f in files[:5]:  # Load 5 most recent
                    if f.suffix.lower() in ['.csv', '.xlsx', '.pdf', '.txt', '.json', '.docx']:
                        doc_id = str(f.stem)
                        if doc_id not in self.tools.loaded_docs:
                            try:
                                self.tools.read_file(str(f))
                            except:
                                pass
        except:
            pass
        
        # Now ask the question
        if self.tools.loaded_docs:
            # Build context from loaded docs
            context_parts = []
            for doc_id, doc in list(self.tools.loaded_docs.items())[:3]:
                if hasattr(doc, 'raw_content'):
                    context_parts.append(f"[{doc.filename}]\n{doc.raw_content[:3000]}")
            
            if context_parts:
                combined = "\n\n---\n\n".join(context_parts)
                prompt = f"""Based on the following documents, answer the question.

DOCUMENTS:
{combined}

QUESTION: {question}

Answer based on the document content:"""
                
                return self._generate(prompt)
        
        return "No documents loaded. Please upload a document first."
    
    # ========================================================================
    # CODE EXECUTION
    # ========================================================================
    
    def run_code(self, code: str, language: str = "python") -> Dict:
        """Execute code."""
        if self.tools:
            if language == "python":
                result = self.tools.run_python(code)
            else:
                result = self.tools.run_bash(code)
            
            return {
                "success": result.success,
                "output": result.result,
                "error": result.error,
                "files_created": result.files_created,
            }
        return {"success": False, "error": "Tools not initialized"}
    
    # ========================================================================
    # LEARNING INTERFACE
    # ========================================================================
    
    def learn_topic(self, topic: str, depth: str = "standard") -> Dict:
        """
        Learn about a topic from the web.
        
        Args:
            topic: Topic to learn about
            depth: Research depth (quick, standard, deep, comprehensive)
        
        Returns:
            Learning result with events created
        """
        if not self.is_initialized:
            return {"error": "Not initialized"}
        
        logger.info(f"Learning about: {topic} (depth={depth})")
        
        # Conduct research
        events = self.continuous_learning.learn_topic(topic, depth)
        
        # Update stats
        self.stats["topics_learned"] += 1
        
        return {
            "topic": topic,
            "depth": depth,
            "events_created": len(events),
            "knowledge_nodes": len(self.knowledge_graph.nodes),
        }
    
    def teach(self, subject: str, content: str, node_type: str = "fact") -> Dict:
        """
        Directly teach GroundZero new knowledge.
        
        Args:
            subject: Subject/name for the knowledge
            content: The knowledge content
            node_type: Type of knowledge (fact, concept, skill)
        
        Returns:
            Result of teaching
        """
        if not self.is_initialized:
            return {"error": "Not initialized"}
        
        # Add to knowledge graph
        node = self.knowledge_graph.add_node(
            name=subject,
            content=content,
            node_type=node_type,
            source="direct_teaching",
            confidence=0.9,
        )
        
        # Add to memory
        self.memory.remember(
            f"{subject}: {content}",
            memory_type="fact",
            importance=0.8,
        )
        
        logger.info(f"Learned: {subject}")
        
        return {
            "node_id": node.id,
            "subject": subject,
            "stored": True,
        }
    
    # ========================================================================
    # CORRECTION INTERFACE
    # ========================================================================
    
    def handle_correction(
        self,
        original_response: str,
        correction: str,
        verify: bool = True,
    ) -> Dict:
        """
        Handle a user correction.
        
        Args:
            original_response: The response that was wrong
            correction: User's correction
            verify: Whether to verify the correction
        
        Returns:
            Correction result
        """
        if not self.is_initialized:
            return {"error": "Not initialized"}
        
        result = self.reasoning.handle_correction(original_response, correction)
        
        # Also process for continuous learning
        self.continuous_learning.process_correction(
            original_response, correction, verify=verify
        )
        
        # Update stats
        if result.get("applied"):
            self.stats["corrections_applied"] += 1
        
        return result
    
    # ========================================================================
    # EVOLUTION INTERFACE
    # ========================================================================
    
    def evolve(self, min_events: int = 10) -> Dict:
        """
        Evolve the model by learning from accumulated interactions.
        
        This:
        1. Consolidates learning events
        2. Prepares training data
        3. Fine-tunes the model (if enough data)
        
        Args:
            min_events: Minimum events needed to trigger training
        
        Returns:
            Evolution result
        """
        if not self.is_initialized:
            return {"error": "Not initialized"}
        
        logger.info("Starting evolution...")
        
        # Consolidate learning
        consolidation = self.continuous_learning.consolidate(min_events=min_events)
        
        if not consolidation.get("consolidated"):
            return {
                "evolved": False,
                "reason": consolidation.get("reason"),
            }
        
        # If we have a real model and training data, run training
        if not self.use_mock and consolidation.get("training_data"):
            # Create continuous learner for model
            learner = ContinuousLearner(self.model_manager, self.config)
            
            # Add training examples
            for example in consolidation["training_data"]:
                learner.add_training_example(
                    example["input"],
                    example["output"],
                    example_type=example.get("type", "general"),
                )
            
            # Run evolution
            training_result = learner.evolve()
            
            self.stats["evolutions"] += 1
            
            return {
                "evolved": True,
                "events_processed": consolidation["events_processed"],
                "training_examples": consolidation["training_examples"],
                "training_result": training_result,
            }
        
        self.stats["evolutions"] += 1
        
        return {
            "evolved": True,
            "events_processed": consolidation["events_processed"],
            "training_examples": consolidation["training_examples"],
            "note": "Mock mode - no actual model training",
        }
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_conversations(self, user_id: str = None, limit: int = 20) -> List[Conversation]:
        """Get conversations for a user."""
        uid = user_id or self.current_user_id
        return self.memory.conversations.get_user_conversations(uid, limit=limit)
    
    def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        """Get a specific conversation."""
        return self.memory.conversations.get_conversation(conv_id)
    
    def get_user_profile(self, user_id: str = None) -> Optional[UserProfile]:
        """Get user profile."""
        uid = user_id or self.current_user_id
        return self.memory.users.get_user(uid)
    
    def process_feedback(
        self,
        conversation_id: str,
        message_id: str,
        rating: int,
        feedback_text: str = "",
    ):
        """Process user feedback on a response."""
        # Get the conversation
        conv = self.get_conversation(conversation_id)
        if not conv:
            return
        
        # Find the relevant messages
        for i, msg in enumerate(conv.messages):
            if hasattr(msg, 'id') and msg.id == message_id:
                # Found the message
                if i > 0:
                    user_msg = conv.messages[i-1]
                    self.continuous_learning.process_interaction(
                        user_msg.content if hasattr(user_msg, 'content') else user_msg.get('content', ''),
                        msg.content if hasattr(msg, 'content') else msg.get('content', ''),
                        self.current_user_id,
                        feedback=rating,
                    )
                break
    
    def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search the web."""
        return self.web_search.search(query, max_results=max_results)
    
    def verify(self, claim: str) -> Dict:
        """Verify a claim."""
        result = self.web_search.verify(claim)
        return {
            "verified": result.verified,
            "confidence": result.confidence,
            "explanation": result.explanation,
        }
    
    def get_status(self) -> Dict:
        """Get system status."""
        return {
            "initialized": self.is_initialized,
            "model_loaded": self.model_manager.state.is_loaded if self.model_manager else False,
            "model_name": self.config.model.name,
            "version": self.config.model.version,
            "use_mock": self.use_mock,
            "knowledge_nodes": len(self.knowledge_graph.nodes) if self.knowledge_graph else 0,
            "knowledge_edges": len(self.knowledge_graph.edges) if self.knowledge_graph else 0,
            "memory_entries": len(self.memory.long_term.memories) if self.memory else 0,
            "users": len(self.memory.users.users) if self.memory else 0,
            "pending_learning": len(self.continuous_learning.pending_events) if self.continuous_learning else 0,
            "stats": self.stats,
        }
    
    def get_stats(self) -> Dict:
        """Get comprehensive stats for dashboard."""
        return {
            "model": {
                "name": self.config.model.name if self.config else "Unknown",
                "version": self.config.model.version if self.config else "N/A",
                "loaded": self.model_manager.state.is_loaded if self.model_manager else False,
            },
            "knowledge": {
                "total_nodes": len(self.knowledge_graph.nodes) if self.knowledge_graph else 0,
                "total_edges": len(self.knowledge_graph.edges) if self.knowledge_graph else 0,
            },
            "memory": {
                "conversations": len(self.memory.conversations.conversations) if self.memory else 0,
                "current_user": self.current_user_id,
            },
            "learning": {
                "queue_size": len(self.continuous_learning.pending_events) if self.continuous_learning else 0,
            },
            "tools": {
                "total": self.stats.get("total_chats", 0),
            },
        }
    
    def save(self):
        """Save all state to disk."""
        logger.info("Saving GroundZero AI state...")
        
        if self.knowledge_graph:
            self.knowledge_graph.save()
        
        if self.memory:
            self.memory.save_all()
        
        if self.continuous_learning:
            self.continuous_learning._save_events()
        
        if self.model_manager:
            self.model_manager._save_state()
        
        logger.info("State saved")
    
    def run_dashboard(self, host: str = "0.0.0.0", port: int = 8080):
        """Run the web dashboard."""
        from src.dashboard import run_dashboard
        run_dashboard(self, host=host, port=port)


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Command-line interface for GroundZero AI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GroundZero AI")
    parser.add_argument("--setup", action="store_true", help="Run initial setup")
    parser.add_argument("--download", action="store_true", help="Download model")
    parser.add_argument("--chat", action="store_true", help="Start chat mode")
    parser.add_argument("--dashboard", action="store_true", help="Start dashboard")
    parser.add_argument("--mock", action="store_true", help="Use mock model")
    parser.add_argument("--port", type=int, default=8080, help="Dashboard port")
    parser.add_argument("--learn", type=str, help="Learn about a topic")
    parser.add_argument("--status", action="store_true", help="Show status")
    
    args = parser.parse_args()
    
    # Initialize
    ai = GroundZeroAI(use_mock=args.mock)
    
    if args.setup or args.download:
        ai.setup(download_model=args.download)
    elif args.status:
        ai.setup(download_model=False)
        status = ai.get_status()
        print("\n=== GroundZero AI Status ===")
        for key, value in status.items():
            print(f"  {key}: {value}")
    elif args.learn:
        ai.setup(download_model=False)
        result = ai.learn_topic(args.learn, depth="deep")
        print(f"\nLearned about: {args.learn}")
        print(f"Events created: {result.get('events_created', 0)}")
    elif args.dashboard:
        ai.setup(download_model=False)
        ai.run_dashboard(port=args.port)
    elif args.chat:
        ai.setup(download_model=False)
        print("\n=== GroundZero AI Chat ===")
        print("Type 'quit' to exit, 'learn X' to learn about X, 'status' for status\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                elif user_input.lower() == 'quit':
                    ai.save()
                    break
                elif user_input.lower() == 'status':
                    print(ai.get_status())
                    continue
                elif user_input.lower().startswith('learn '):
                    topic = user_input[6:].strip()
                    result = ai.learn_topic(topic)
                    print(f"Learned about {topic}: {result.get('events_created', 0)} events")
                    continue
                
                response = ai.chat(user_input)
                
                print(f"\nGroundZero: {response['response']}")
                
                if response.get('reasoning'):
                    print("\n  [Reasoning]")
                    for step in response['reasoning'][:3]:
                        print(f"    {step['step']}. {step['thought'][:80]}...")
                
                print()
                
            except KeyboardInterrupt:
                print("\n\nSaving and exiting...")
                ai.save()
                break
            except Exception as e:
                print(f"Error: {e}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()