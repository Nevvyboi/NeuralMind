# 🧠 GroundZero AI

**Your Personal AI That Learns and Grows With You**

GroundZero AI is a complete AI system that continuously learns from your interactions, builds a knowledge graph, remembers conversations, and improves over time. It's built to be "your own" AI that you can train and customize.

---

## ✨ Features

### 🧠 Intelligent Reasoning
- **Chain-of-thought reasoning** with visible thinking steps
- **Self-verification** against knowledge graph and web
- **Confidence scoring** for all responses

### 📚 Knowledge Graph
- Stores and connects concepts, facts, and relationships
- Grows automatically as you interact
- Verifies information from multiple sources
- Semantic search and path finding

### 💾 Memory System
- **Conversation history** - remembers all chats
- **User profiles** - learns your name, preferences, behaviors
- **Long-term memory** - stores facts and events
- **Working memory** - context for current conversation

### 🔍 Web Search & Verification
- Multi-engine search (DuckDuckGo, Wikipedia, arXiv)
- Source reliability scoring
- Fact verification with evidence
- **Deep research mode** for comprehensive learning

### 📈 Continuous Learning
- Learns from your feedback (👍/👎)
- Learns from corrections ("no, that's wrong...")
- Learns topics from the web
- Background learning cycles

### 💻 Code Execution (NEW!)
- Run Python code with full output capture
- Execute bash/shell commands
- Install packages on the fly
- Persistent execution environment

### 📄 Document Understanding (NEW!)
- Read ANY file type (PDF, Excel, Word, CSV, images, etc.)
- Extract tables, text, and structure
- Ask questions about document content
- Multi-document analysis for analytics

### 📝 File Creation (NEW!)
- Create Word documents (.docx)
- Create PDF files
- Create Excel spreadsheets (.xlsx)
- Create PowerPoint presentations (.pptx)
- Create CSV, Markdown, and more

### 🎨 Modern Dashboard (Claude-like)
- Beautiful chat interface
- Real-time reasoning display
- Conversation history
- Knowledge graph visualization
- User settings

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd groundzero
pip install -r requirements.txt
```

### 2. Run Tests (Verify Everything Works)

```bash
python test.py
```

### 3. Start Interactive Chat

```bash
python run.py
# or
python run.py --chat
```

### 4. Start Web Dashboard

```bash
python run.py --dashboard
```

Then open http://localhost:8080

---

## 📦 Project Structure

```
groundzero/
├── config/
│   └── config.yaml              # Main configuration
├── data/                        # All data storage
│   ├── knowledge/               # Knowledge graph
│   ├── memory/                  # Long-term memory
│   ├── models/                  # Downloaded models
│   ├── training/                # Training data
│   ├── conversations/           # Chat history
│   └── users/                   # User profiles
├── src/
│   ├── core/                    # Model management
│   ├── knowledge/               # Knowledge graph system
│   ├── memory/                  # Memory system
│   ├── search/                  # Web search & verification
│   ├── reasoning/               # Chain-of-thought reasoning
│   ├── continuous_learning/     # Learning from interactions
│   ├── dashboard/               # Web interface
│   │   ├── templates/           # HTML
│   │   └── static/              # CSS & JS
│   ├── utils/                   # Utilities
│   └── groundzero.py            # Main AI class
├── run.py                       # Entry point
├── test.py                      # Test suite
└── requirements.txt
```

---

## 💻 Usage Examples

### Basic Chat

```python
from src.groundzero import GroundZeroAI

ai = GroundZeroAI()

# Chat
response, reasoning = ai.chat("Hello! What can you do?", return_reasoning=True)
print(response)

# Show reasoning
if reasoning:
    print(f"Confidence: {reasoning.confidence:.0%}")
    for step in reasoning.steps:
        print(f"  {step.step_number}. {step.thought}")
```

### Teaching Knowledge

```python
# Teach directly
ai.teach("Python", "Python is a programming language created by Guido van Rossum")

# Have it learn from the web
ai.learn("machine learning transformers")
```

### Feedback & Corrections

```python
# Rate a response
ai.feedback("What is X?", "Response...", rating=5)  # 1-5

# Correct a mistake
ai.correct(
    prompt="What is the capital of Australia?",
    wrong_response="Sydney",
    correct_response="Canberra"
)
```

### Query Knowledge

```python
# Search knowledge
result = ai.ask_knowledge("machine learning")
print(result["results"])

# Verify a fact
verification = ai.verify_fact("The Earth orbits the Sun")
print(f"Verified: {verification['verified']}, Confidence: {verification['confidence']}")
```

---

## ⚙️ Configuration

Edit `config/config.yaml`:

```yaml
model:
  name: "GroundZero-AI"
  version: "1.0.0"
  base_model: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
  quantization: "4bit"  # For GPU memory efficiency

knowledge:
  auto_verify: true
  min_confidence: 0.7

search:
  engines:
    - duckduckgo
    - wikipedia
    - arxiv
  verify_sources: true

continuous_learning:
  enabled: true
  auto_evolve_threshold: 100

dashboard:
  host: "0.0.0.0"
  port: 8080
```

---

## 🖥️ System Requirements

### Demo Mode (No GPU)
- Python 3.9+
- 4GB RAM
- Works on any CPU

### Full Mode (With Training)
- Python 3.9+
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM (RTX 3060+)
- CUDA 11.8+

---

## 🔄 How Learning Works

```
User Input → Generate Response → Observe Interaction
                                        ↓
                              [Learning Signal Queue]
                                        ↓
Feedback (👍/👎) ─────────────────→ Process Signals
Corrections ──────────────────────→      ↓
Web Research ─────────────────────→ Training Data
                                        ↓
                              [Training Cycle]
                                        ↓
                              Improved Model
```

1. **Observe**: Every interaction is recorded
2. **Feedback**: Ratings create learning signals
3. **Corrections**: Direct corrections are high-priority
4. **Research**: Web learning adds knowledge
5. **Evolve**: Training on accumulated signals

---

## 📊 Dashboard Features

### Chat Interface
- Clean, modern design
- Typing indicators
- Reasoning panel
- Feedback buttons

### History
- Browse past conversations
- Search chats

### Knowledge
- Visualize the knowledge graph
- Search concepts

### Stats
- Model status
- Knowledge count
- Learning progress

---

## 🛡️ Privacy

- **All data stored locally** in `/data`
- No external transmission except explicit web searches
- User profiles stay on your machine
- You control what it learns

---

## 📝 CLI Reference

```bash
# Interactive chat
python run.py --chat

# Web dashboard
python run.py --dashboard --port 8080

# Learn about a topic
python run.py --learn "quantum computing"

# Download model (requires GPU)
python run.py --download
```

---

## 🔧 API Reference

### GroundZeroAI

| Method | Description |
|--------|-------------|
| `chat(message)` | Chat and get response with optional reasoning |
| `learn(topic)` | Research and learn from web |
| `teach(subject, content)` | Directly add knowledge |
| `feedback(prompt, response, rating)` | Rate a response (1-5) |
| `correct(prompt, wrong, correct)` | Correct a mistake |
| `ask_knowledge(query)` | Query knowledge graph |
| `verify_fact(claim)` | Verify a fact |
| `evolve()` | Run training on learning queue |
| `get_stats()` | Get system statistics |
| `save()` | Save all data |

---

## 🤝 Contributing

Areas to help:
- Better reasoning algorithms
- More search engines
- UI improvements
- Test coverage
- Documentation

---

## 📄 License

MIT License - Use freely!

---

## 🙏 Acknowledgments

- Built on DeepSeek-R1-Distill
- Inspired by Claude AI's interface
- Uses QLoRA for efficient fine-tuning

---

**Made with ❤️ for AI enthusiasts who want their own personal AI**
