"""
GroundZero AI - Main System
===========================

The main GroundZero AI class that integrates all components:
- Model management and inference
- Knowledge graph
- Memory system
- Web search and verification
- Reasoning engine
- Continuous learning
- Dashboard
- FILE UPLOAD IN CHAT (NEW!)

This is your AI - it learns, remembers, and grows with you.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Add src to path for imports
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from utils import (
    Config, get_config, logger, timestamp, 
    get_data_path, ensure_dir
)
from core import ModelManager, ContinuousLearner, MockModel
from knowledge import KnowledgeGraph, KnowledgeExtractor
from memory import MemorySystem
from search import WebSearch
from reasoning import ReasoningEngine, ReasoningTrace
from continuous_learning import ContinuousLearningSystem
from tools import ToolsManager, ToolResult


class GroundZeroAI:
    """
    GroundZero AI - Your Personal AI That Learns
    
    Features:
    - Chat with reasoning display
    - Knowledge graph that grows
    - Memory of conversations and user preferences
    - Web search when knowledge is missing
    - Learns from feedback and corrections
    - Continuous improvement
    - FILE UPLOAD IN CHAT (like Claude!)
    """
    
    def __init__(
        self,
        config: Config = None,
        load_model: bool = False,
        use_mock: bool = False,
    ):
        """
        Initialize GroundZero AI.
        
        Args:
            config: Configuration object (uses default if None)
            load_model: Whether to load the actual model (requires GPU)
            use_mock: Use mock model for testing (no GPU needed)
        """
        self.config = config or get_config()
        logger.info("Initializing GroundZero AI...")
        
        # Initialize components
        self._init_model(load_model, use_mock)
        self._init_knowledge()
        self._init_memory()
        self._init_search()
        self._init_reasoning()
        self._init_learning()
        self._init_tools()
        
        logger.info("[OK] GroundZero AI initialized!")
    
    def _init_model(self, load_model: bool, use_mock: bool):
        """Initialize model manager."""
        self.model_manager = ModelManager(self.config)
        
        if use_mock:
            self.mock_model = MockModel()
            self._generate = self.mock_model.generate
            logger.info("Using mock model (no GPU required)")
        else:
            # Try to load real model
            model_path = self.model_manager.model_path / "config.json"
            if model_path.exists() and self.model_manager.load_model():
                self._generate = lambda p, **k: self.model_manager.generate(p, **k)
                self.mock_model = None
            else:
                self.mock_model = MockModel()
                self._generate = self.mock_model.generate
                logger.info("Falling back to mock model")
    
    def _init_knowledge(self):
        """Initialize knowledge graph."""
        self.knowledge = KnowledgeGraph()
        self.knowledge_extractor = KnowledgeExtractor(self.knowledge)
        logger.info(f"Knowledge graph: {self.knowledge.get_stats()['total_nodes']} nodes")
    
    def _init_memory(self):
        """Initialize memory system."""
        self.memory = MemorySystem()
        self.memory.set_user("default")
        logger.info("Memory system initialized")
    
    def _init_search(self):
        """Initialize web search."""
        self.web_search = WebSearch()
        logger.info("Web search initialized")
    
    def _init_reasoning(self):
        """Initialize reasoning engine."""
        self.reasoning = ReasoningEngine(
            model_generate=self._generate,
            knowledge_graph=self.knowledge,
            web_search=self.web_search,
            memory_system=self.memory,
        )
        logger.info("Reasoning engine initialized")
    
    def _init_learning(self):
        """Initialize continuous learning."""
        def model_trainer(data):
            if self.model_manager.state.is_loaded:
                learner = ContinuousLearner(self.model_manager, self.config)
                for ex in data:
                    learner.add_training_example(ex["input"], ex["output"])
                return learner.evolve()
            return {"success": True, "simulated": True}
        
        self.learning = ContinuousLearningSystem(
            model_trainer=model_trainer,
            web_search=self.web_search,
        )
        logger.info("Continuous learning initialized")
    
    def _init_tools(self):
        """Initialize tools (code execution, document understanding, file creation)."""
        self.tools = ToolsManager(
            model_generate=self._generate
        )
        logger.info("Tools initialized (code execution, documents, files)")
    
    # ========================================================================
    # CHAT INTERFACE - UPGRADED WITH FILE SUPPORT
    # ========================================================================
    
    def chat(
        self,
        message: str,
        user_id: str = None,
        conversation_id: str = None,  # Added for dashboard compatibility
        use_reasoning: bool = True,
        use_knowledge: bool = True,
        use_memory: bool = True,
        return_reasoning: bool = False,
        file_content: str = None,  # NEW: File content for inline analysis
        file_name: str = None,     # NEW: File name for context
    ) -> Tuple[str, Optional[ReasoningTrace]]:
        """
        Chat with GroundZero AI.
        
        UPGRADED: Now supports file uploads directly in chat!
        
        Args:
            message: User's message
            user_id: User identifier (uses current user if None)
            conversation_id: Conversation ID (for dashboard compatibility)
            use_reasoning: Enable chain-of-thought reasoning
            use_knowledge: Include knowledge graph context
            use_memory: Include memory context
            return_reasoning: Return reasoning trace
            file_content: Content of an uploaded file to analyze (NEW)
            file_name: Name of the uploaded file (NEW)
        
        Returns:
            (response, reasoning_trace) if return_reasoning else response
        """
        if user_id:
            self.memory.set_user(user_id)
        
        # Build the prompt - UPGRADED to handle file content
        if file_content:
            # User uploaded a file - include its content in the prompt
            file_preview = file_content[:8000]  # Limit to ~8000 chars
            prompt = f"""The user has uploaded a file called "{file_name or 'document'}".

FILE CONTENT:
{file_preview}

USER QUESTION: {message}

Please analyze the file content and answer the user's question based on what you see:"""
        else:
            # Regular chat - just use the message directly (no context mixing!)
            prompt = message
        
        # Generate response
        response = self._generate(prompt)
        
        # Generate reasoning trace for display (if requested)
        trace = None
        if use_reasoning:
            try:
                trace = self.reasoning.reason(
                    message,
                    context="",  # Empty context to avoid mixing
                    use_knowledge=False,
                )
                # Use our direct response, not reasoning's interpretation
                trace.final_answer = response
            except Exception as e:
                logger.warning(f"Reasoning failed: {e}")
                trace = None
        
        # Store in memory (minimal)
        try:
            self.memory.add_turn(message, response)
        except:
            pass
        
        # Learn from interaction
        try:
            self.learning.observe(message, response)
        except:
            pass
        
        if return_reasoning:
            return response, trace
        return response, None
    
    def chat_with_file(self, message: str, filepath: str, **kwargs) -> Tuple[str, Optional[ReasoningTrace]]:
        """
        Chat with a specific file loaded.
        
        Args:
            message: Your question about the file
            filepath: Path to the file to analyze
            **kwargs: Additional args passed to chat()
        
        Returns:
            (response, reasoning_trace)
        
        Example:
            response, _ = ai.chat_with_file("What are the total sales?", "sales.csv")
        """
        # Read the file
        file_content = ""
        file_name = Path(filepath).name
        
        try:
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
            return f"Error reading file: {e}", None
        
        return self.chat(
            message=message,
            file_content=file_content,
            file_name=file_name,
            **kwargs
        )
    
    def _build_context(self, message: str, use_knowledge: bool, use_memory: bool) -> str:
        """Build context from knowledge and memory."""
        # Disabled for now - context was confusing the model on new topics
        # The model works better when each question is treated independently
        return ""
    
    # ========================================================================
    # LEARNING INTERFACE
    # ========================================================================
    
    def learn(self, topic: str, depth: str = "deep") -> Dict:
        """
        Learn about a topic from the web.
        
        This searches the web, verifies information, and stores
        it in the knowledge graph.
        """
        logger.info(f"Learning about: {topic}")
        
        # Research the topic
        research = self.web_search.research(topic, depth=depth)
        
        # Store in knowledge graph
        if research.get("overview"):
            self.knowledge.add_node(
                name=topic,
                content=research["overview"][:2000],
                node_type="concept",
                source="web_research",
                confidence=0.8,
            )
        
        # Add related concepts
        for concept in research.get("key_concepts", [])[:10]:
            self.knowledge.add_knowledge(
                topic, "related_to", concept,
                source="web_research",
                confidence=0.7,
            )
        
        # Extract more knowledge from content
        extracted = 0
        if research.get("overview"):
            extracted = self.knowledge_extractor.extract_and_store(
                research["overview"],
                source="web_research"
            )
            logger.info(f"Extracted {extracted} knowledge triples")
        
        # Add to learning system
        self.learning.learn_topic(topic)
        
        # Save
        self.knowledge.save()
        
        return {
            "success": True,
            "topic": topic,
            "facts_learned": len(research.get("key_concepts", [])),
            "sources": len(research.get("sources", [])),
            "knowledge_extracted": extracted if research.get("overview") else 0,
        }
    
    def teach(self, subject: str, content: str, confidence: float = 0.9):
        """
        Teach GroundZero directly by adding knowledge.
        
        Args:
            subject: The topic/concept
            content: What to teach
            confidence: How confident this knowledge is
        """
        self.knowledge.add_node(
            name=subject,
            content=content,
            node_type="fact",
            source="user_taught",
            confidence=confidence,
        )
        
        # Extract relationships
        extracted = self.knowledge_extractor.extract_and_store(content, "user_taught")
        
        self.knowledge.save()
        
        logger.info(f"Learned about: {subject}")
        return {"success": True, "subject": subject, "extracted": extracted}
    
    # ========================================================================
    # FEEDBACK INTERFACE
    # ========================================================================
    
    def feedback(self, prompt: str, response: str, rating: int):
        """
        Provide feedback on a response.
        
        Args:
            prompt: The original prompt
            response: The response to rate
            rating: 1-5 rating (5 is best)
        """
        self.learning.feedback(prompt, response, rating)
        logger.info(f"Feedback recorded: {rating}/5")
    
    def correct(self, prompt: str, wrong_response: str, correct_response: str):
        """
        Correct a wrong response.
        
        This is a strong learning signal - GroundZero will
        learn from the correction.
        """
        # Handle correction in reasoning
        self.reasoning.handle_correction(wrong_response, correct_response)
        
        # Add to learning
        self.learning.correct(prompt, wrong_response, correct_response)
        
        # Remember the correction
        self.memory.remember(
            f"Correction: For '{prompt[:100]}', the correct response is '{correct_response[:200]}'",
            memory_type="correction",
            importance=0.9,
        )
        
        logger.info("Correction recorded and learned")
    
    # ========================================================================
    # KNOWLEDGE INTERFACE
    # ========================================================================
    
    def ask_knowledge(self, query: str) -> Dict:
        """
        Query the knowledge graph.
        """
        results = self.knowledge.search_nodes(query=query, limit=10)
        
        return {
            "query": query,
            "results": [
                {
                    "name": n.name,
                    "content": n.content[:500],
                    "type": n.node_type,
                    "confidence": n.confidence,
                }
                for n in results
            ],
            "context": self.knowledge.get_context(query),
        }
    
    def verify_fact(self, claim: str) -> Dict:
        """
        Verify a fact using knowledge graph and web search.
        """
        # Check knowledge graph first
        kg_results = self.knowledge.search_nodes(query=claim, limit=5)
        
        kg_verified = False
        kg_confidence = 0.0
        
        for node in kg_results:
            if claim.lower() in node.content.lower():
                kg_verified = True
                kg_confidence = node.confidence
                break
        
        # Verify with web if not found or low confidence
        if not kg_verified or kg_confidence < 0.7:
            web_result = self.web_search.verify(claim)
            return {
                "claim": claim,
                "verified": web_result.verified,
                "confidence": web_result.confidence,
                "source": "web_search",
                "explanation": web_result.explanation,
                "supporting_sources": len(web_result.supporting_sources),
            }
        
        return {
            "claim": claim,
            "verified": kg_verified,
            "confidence": kg_confidence,
            "source": "knowledge_graph",
            "explanation": f"Found in knowledge base: {kg_results[0].name if kg_results else 'N/A'}",
        }
    
    # ========================================================================
    # TOOLS INTERFACE - Code, Documents, Files
    # ========================================================================
    
    def run_code(self, code: str, language: str = "python") -> Dict:
        """
        Execute code (Python or Bash).
        
        Args:
            code: Code to execute
            language: 'python' or 'bash'
        
        Returns:
            Dict with success, output, error
        
        Example:
            result = ai.run_code("print(2 + 2)")
            print(result["output"])  # "4"
        """
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
    
    def read_document(self, filepath: str) -> Dict:
        """
        Read and understand a document (PDF, Excel, Word, CSV, etc.)
        
        Args:
            filepath: Path to the document
        
        Returns:
            Dict with document info, summary, and content
        
        Example:
            doc = ai.read_document("report.pdf")
            print(doc["summary"])
        """
        result = self.tools.read_file(filepath)
        
        if result.success:
            return {
                "success": True,
                **result.result,
            }
        
        return {
            "success": False,
            "error": result.error,
        }
    
    def ask_documents(self, question: str) -> str:
        """
        Ask a question about loaded documents.
        
        UPGRADED: Now auto-loads documents from uploads folder.
        
        Args:
            question: Your question
        
        Returns:
            Answer based on document content
        """
        # Auto-load recent uploads if not already loaded
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
        
        # Fallback to tool's ask
        result = self.tools.ask_document(question)
        if result.success and result.result:
            return result.result
        
        return "No documents loaded. Please upload a document first."
    
    def create_document(
        self,
        filename: str,
        content: any,
        doc_type: str = None,
        **kwargs
    ) -> str:
        """
        Create a document (Word, PDF, Excel, PowerPoint, etc.)
        
        Args:
            filename: Output filename with extension
            content: Content (string, list of dicts, etc. depending on type)
            doc_type: Force type (auto-detected from extension if None)
            **kwargs: Additional options (title, author, etc.)
        
        Returns:
            Path to created file
        
        Examples:
            # Create Word doc
            ai.create_document("report.docx", "This is my report", title="Q4 Report")
            
            # Create Excel
            data = [{"Name": "Alice", "Age": 30}, {"Name": "Bob", "Age": 25}]
            ai.create_document("people.xlsx", data)
            
            # Create PDF
            ai.create_document("summary.pdf", "Summary content", title="Summary")
        """
        result = self.tools.create_file(filename, content, file_type=doc_type, **kwargs)
        
        if result.success:
            return result.result
        raise Exception(result.error)
    
    def create_word(self, filename: str, content: any, title: str = None) -> str:
        """Create a Word document."""
        result = self.tools.create_word(filename, content, title=title)
        if result.success:
            return result.result
        raise Exception(result.error)
    
    def create_excel(self, filename: str, data: list, sheet_name: str = "Sheet1") -> str:
        """Create an Excel file."""
        result = self.tools.create_excel(filename, data, sheet_name=sheet_name)
        if result.success:
            return result.result
        raise Exception(result.error)
    
    def create_pdf(self, filename: str, content: any, title: str = None) -> str:
        """Create a PDF document."""
        result = self.tools.create_pdf(filename, content, title=title)
        if result.success:
            return result.result
        raise Exception(result.error)
    
    def create_powerpoint(self, filename: str, slides: list, title: str = None) -> str:
        """
        Create a PowerPoint presentation.
        
        Args:
            filename: Output filename
            slides: List of slide definitions
            title: Presentation title
        
        Example:
            slides = [
                {"type": "title", "title": "My Presentation", "subtitle": "By Me"},
                {"type": "content", "title": "Key Points", "content": ["Point 1", "Point 2"]},
            ]
            ai.create_powerpoint("deck.pptx", slides)
        """
        result = self.tools.create_powerpoint(filename, slides, title=title)
        if result.success:
            return result.result
        raise Exception(result.error)
    
    def analyze_data(self, filepath: str, analysis_code: str = None) -> Dict:
        """
        Analyze data from a file (CSV, Excel).
        
        Args:
            filepath: Path to data file
            analysis_code: Optional Python code to run on the data
        
        Returns:
            Analysis results
        """
        result = self.tools.analyze_data(filepath, analysis_code)
        return {
            "success": result.success,
            "output": result.result,
            "error": result.error,
        }
    
    def get_loaded_documents(self) -> list:
        """Get list of loaded documents."""
        return self.tools.get_loaded_documents()
    
    # ========================================================================
    # EVOLUTION INTERFACE
    # ========================================================================
    
    def evolve(self) -> Dict:
        """
        Trigger an evolution cycle.
        
        This trains the model on accumulated learning signals
        (feedback, corrections, interactions, research).
        """
        logger.info("Starting evolution cycle...")
        
        result = self.learning.evolve()
        
        if result.get("success"):
            logger.info("[OK] Evolution complete!")
        else:
            logger.warning(f"Evolution issue: {result.get('reason')}")
        
        return result
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_stats(self) -> Dict:
        """Get system statistics."""
        return {
            "model": {
                "name": self.model_manager.state.name,
                "version": self.model_manager.state.version,
                "loaded": self.model_manager.state.is_loaded,
                "trained": self.model_manager.state.is_trained,
                "training_sessions": self.model_manager.state.training_sessions,
            },
            "knowledge": self.knowledge.get_stats(),
            "memory": {
                "conversations": len(self.memory.conversations.conversations),
                "long_term_memories": len(self.memory.long_term.memories),
                "current_user": self.memory.current_user_id,
            },
            "learning": self.learning.get_stats(),
            "tools": self.tools.get_stats(),
        }
    
    def save(self):
        """Save all data."""
        self.knowledge.save()
        self.memory.save_all()
        self.model_manager._save_state()
        logger.info("All data saved")
    
    def download_model(self, model_key: str = "deepseek-7b"):
        """Download and setup the base model."""
        return self.model_manager.download_model(model_key)
    
    def load_model(self):
        """Load the model into memory."""
        return self.model_manager.load_model()


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Command-line interface for GroundZero AI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GroundZero AI")
    parser.add_argument("--download", action="store_true", help="Download the model")
    parser.add_argument("--dashboard", action="store_true", help="Start the dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Dashboard port")
    parser.add_argument("--chat", action="store_true", help="Start interactive chat")
    parser.add_argument("--learn", type=str, help="Learn about a topic")
    args = parser.parse_args()
    
    # Initialize
    ai = GroundZeroAI(use_mock=False)
    
    if args.download:
        print("Downloading model...")
        ai.download_model()
    
    elif args.dashboard:
        from dashboard import run_dashboard
        run_dashboard(ai, port=args.port)
    
    elif args.chat:
        print("=" * 50)
        print("GroundZero AI - Interactive Chat")
        print("Type 'quit' to exit, 'learn <topic>' to learn")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == "quit":
                    print("Goodbye!")
                    break
                
                if user_input.lower().startswith("learn "):
                    topic = user_input[6:].strip()
                    print(f"Learning about: {topic}...")
                    result = ai.learn(topic)
                    print(f"Learned {result.get('facts_learned', 0)} facts from {result.get('sources', 0)} sources")
                    continue
                
                response, reasoning = ai.chat(user_input, return_reasoning=True)
                
                if reasoning and reasoning.steps:
                    print(f"\n🧠 Reasoning ({len(reasoning.steps)} steps, {reasoning.confidence:.0%} confidence)")
                
                print(f"\nGroundZero: {response}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
        
        ai.save()
    
    elif args.learn:
        print(f"Learning about: {args.learn}")
        result = ai.learn(args.learn)
        print(f"Result: {result}")
        ai.save()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()