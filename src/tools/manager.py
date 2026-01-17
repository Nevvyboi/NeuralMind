"""
GroundZero AI - Tools Manager
============================

Unified interface for all tools:
1. Code Execution (Python, Bash)
2. Document Understanding (read any file, Q&A)
3. File Creation (Word, PDF, Excel, PowerPoint)
4. Data Analysis (pandas, visualization)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field

try:
    from ..utils import get_data_path, ensure_dir, logger, timestamp
except ImportError:
    from utils import get_data_path, ensure_dir, logger, timestamp

from .code_executor import CodeExecutor, ExecutionResult
from .documents import DocumentProcessor, DocumentQA, Document
from .files import FileCreator


# ============================================================================
# TOOL RESULT
# ============================================================================

@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    tool: str
    action: str
    result: Any = None
    error: str = ""
    files_created: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    
    def __str__(self) -> str:
        if self.success:
            return str(self.result) if self.result else "Success"
        return f"Error: {self.error}"


# ============================================================================
# TOOLS MANAGER
# ============================================================================

class ToolsManager:
    """
    Unified interface for all GroundZero AI tools.
    
    This is the main entry point for:
    - Running code (Python/Bash)
    - Reading and understanding documents
    - Creating files (Word, PDF, Excel, etc.)
    - Data analysis and visualization
    """
    
    def __init__(self, workspace: str = None, model_generate=None):
        """
        Initialize tools manager.
        
        Args:
            workspace: Working directory for files
            model_generate: Function to generate AI responses (for document Q&A)
        """
        self.workspace = Path(workspace) if workspace else get_data_path("workspace")
        ensure_dir(self.workspace)
        
        self.model_generate = model_generate
        
        # Initialize tools
        self.code = CodeExecutor(str(self.workspace))
        self.documents = DocumentProcessor()
        self.doc_qa = DocumentQA(self.documents, model_generate)
        self.files = FileCreator(str(self.workspace / "outputs"))
        
        # Track loaded documents for easy reference
        self.loaded_docs: Dict[str, Document] = {}
        
        logger.info(f"Tools Manager initialized (workspace: {self.workspace})")
    
    # ========================================================================
    # CODE EXECUTION
    # ========================================================================
    
    def run_python(self, code: str, timeout: int = 60) -> ToolResult:
        """
        Execute Python code.
        
        Args:
            code: Python code to run
            timeout: Execution timeout in seconds
        
        Returns:
            ToolResult with output
        
        Example:
            result = tools.run_python('''
                import pandas as pd
                df = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
                print(df.describe())
            ''')
        """
        exec_result = self.code.run_python(code, timeout=timeout)
        
        return ToolResult(
            success=exec_result.success,
            tool="code_executor",
            action="run_python",
            result=exec_result.output or exec_result.return_value,
            error=exec_result.error,
            files_created=exec_result.files_created,
            execution_time=exec_result.execution_time,
        )
    
    def run_bash(self, command: str, timeout: int = 60) -> ToolResult:
        """
        Execute bash/shell command.
        
        Args:
            command: Command to run
            timeout: Timeout in seconds
        
        Returns:
            ToolResult with output
        
        Example:
            result = tools.run_bash("ls -la")
        """
        exec_result = self.code.run_bash(command, timeout=timeout)
        
        return ToolResult(
            success=exec_result.success,
            tool="code_executor",
            action="run_bash",
            result=exec_result.output,
            error=exec_result.error,
            files_created=exec_result.files_created,
            execution_time=exec_result.execution_time,
        )
    
    def install_package(self, package: str) -> ToolResult:
        """Install a Python package."""
        exec_result = self.code.install_package(package)
        
        return ToolResult(
            success=exec_result.success,
            tool="code_executor",
            action="install_package",
            result=f"Installed {package}" if exec_result.success else None,
            error=exec_result.error,
        )
    
    # ========================================================================
    # DOCUMENT OPERATIONS
    # ========================================================================
    
    def read_file(self, filepath: str) -> ToolResult:
        """
        Read and understand a file (any type: PDF, Excel, Word, CSV, etc.)
        
        Args:
            filepath: Path to the file
        
        Returns:
            ToolResult with document content and metadata
        
        Example:
            result = tools.read_file("report.pdf")
            print(result.result.summary)
        """
        try:
            doc = self.documents.process(filepath)
            self.loaded_docs[doc.id] = doc
            self.doc_qa.active_documents.append(doc.id)
            
            return ToolResult(
                success=True,
                tool="document_processor",
                action="read_file",
                result={
                    "id": doc.id,
                    "filename": doc.filename,
                    "file_type": doc.file_type,
                    "word_count": doc.word_count,
                    "page_count": doc.page_count,
                    "chunks": len(doc.chunks),
                    "tables": len(doc.tables),
                    "summary": doc.summary,
                    "key_topics": doc.key_topics,
                },
                execution_time=doc.processing_time,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                tool="document_processor",
                action="read_file",
                error=str(e),
            )
    
    def read_files(self, filepaths: List[str]) -> ToolResult:
        """
        Read multiple files at once.
        
        Args:
            filepaths: List of file paths
        
        Returns:
            ToolResult with all document summaries
        """
        results = []
        errors = []
        
        for fp in filepaths:
            result = self.read_file(fp)
            if result.success:
                results.append(result.result)
            else:
                errors.append(f"{fp}: {result.error}")
        
        return ToolResult(
            success=len(results) > 0,
            tool="document_processor",
            action="read_files",
            result=results,
            error="; ".join(errors) if errors else "",
        )
    
    def ask_document(self, question: str, doc_ids: List[str] = None) -> ToolResult:
        """
        Ask a question about loaded documents.
        
        Args:
            question: Your question
            doc_ids: Specific document IDs to search (uses all loaded if None)
        
        Returns:
            ToolResult with answer
        
        Example:
            tools.read_file("financial_report.pdf")
            result = tools.ask_document("What was the total revenue?")
        """
        answer = self.doc_qa.ask(question, doc_ids)
        
        return ToolResult(
            success=answer.get("confidence", 0) > 0,
            tool="document_qa",
            action="ask",
            result=answer.get("answer"),
            error="" if answer.get("confidence", 0) > 0 else "No relevant information found",
        )
    
    def summarize_documents(self) -> ToolResult:
        """Summarize all loaded documents."""
        summary = self.doc_qa.summarize()
        
        return ToolResult(
            success=True,
            tool="document_qa",
            action="summarize",
            result=summary,
        )
    
    def get_tables(self, doc_id: str = None) -> ToolResult:
        """Get tables from documents."""
        tables = self.doc_qa.get_tables(doc_id)
        
        return ToolResult(
            success=len(tables) > 0,
            tool="document_processor",
            action="get_tables",
            result=[{
                "id": t.id,
                "headers": t.headers,
                "rows": t.rows[:10],  # First 10 rows
                "total_rows": len(t.rows),
                "source": t.source,
            } for t in tables],
        )
    
    def search_documents(self, query: str) -> ToolResult:
        """Search across all loaded documents."""
        results = self.documents.search_documents(query, list(self.loaded_docs.keys()))
        
        formatted = []
        for doc, chunks in results:
            formatted.append({
                "document": doc.filename,
                "matches": [c.content[:200] + "..." for c in chunks[:3]],
            })
        
        return ToolResult(
            success=len(formatted) > 0,
            tool="document_processor",
            action="search",
            result=formatted,
        )
    
    # ========================================================================
    # FILE CREATION
    # ========================================================================
    
    def create_word(self, filename: str, content: Any, title: str = None) -> ToolResult:
        """
        Create a Word document.
        
        Args:
            filename: Output filename (e.g., "report.docx")
            content: String or structured content
            title: Document title
        
        Returns:
            ToolResult with file path
        
        Example:
            tools.create_word("report.docx", "This is my report content", title="Q4 Report")
        """
        try:
            filepath = self.files.create_word(filename, content, title=title)
            return ToolResult(
                success=True,
                tool="file_creator",
                action="create_word",
                result=filepath,
                files_created=[filepath],
            )
        except Exception as e:
            return ToolResult(
                success=False,
                tool="file_creator",
                action="create_word",
                error=str(e),
            )
    
    def create_pdf(self, filename: str, content: Any, title: str = None) -> ToolResult:
        """Create a PDF document."""
        try:
            filepath = self.files.create_pdf(filename, content, title=title)
            return ToolResult(
                success=True,
                tool="file_creator",
                action="create_pdf",
                result=filepath,
                files_created=[filepath],
            )
        except Exception as e:
            return ToolResult(
                success=False,
                tool="file_creator",
                action="create_pdf",
                error=str(e),
            )
    
    def create_excel(self, filename: str, data: Any, sheet_name: str = "Sheet1") -> ToolResult:
        """
        Create an Excel file.
        
        Args:
            filename: Output filename (e.g., "data.xlsx")
            data: List of dicts or list of lists
            sheet_name: Sheet name
        
        Returns:
            ToolResult with file path
        
        Example:
            data = [
                {"Name": "Alice", "Age": 30, "City": "NYC"},
                {"Name": "Bob", "Age": 25, "City": "LA"},
            ]
            tools.create_excel("people.xlsx", data)
        """
        try:
            filepath = self.files.create_excel(filename, data, sheet_name=sheet_name)
            return ToolResult(
                success=True,
                tool="file_creator",
                action="create_excel",
                result=filepath,
                files_created=[filepath],
            )
        except Exception as e:
            return ToolResult(
                success=False,
                tool="file_creator",
                action="create_excel",
                error=str(e),
            )
    
    def create_powerpoint(self, filename: str, slides: List[Dict], title: str = None) -> ToolResult:
        """
        Create a PowerPoint presentation.
        
        Args:
            filename: Output filename (e.g., "presentation.pptx")
            slides: List of slide definitions
            title: Presentation title
        
        Returns:
            ToolResult with file path
        
        Example:
            slides = [
                {"type": "title", "title": "My Presentation", "subtitle": "By Me"},
                {"type": "content", "title": "Key Points", "content": ["Point 1", "Point 2"]},
            ]
            tools.create_powerpoint("deck.pptx", slides)
        """
        try:
            filepath = self.files.create_powerpoint(filename, slides, title=title)
            return ToolResult(
                success=True,
                tool="file_creator",
                action="create_powerpoint",
                result=filepath,
                files_created=[filepath],
            )
        except Exception as e:
            return ToolResult(
                success=False,
                tool="file_creator",
                action="create_powerpoint",
                error=str(e),
            )
    
    def create_csv(self, filename: str, data: List, headers: List[str] = None) -> ToolResult:
        """Create a CSV file."""
        try:
            filepath = self.files.create_csv(filename, data, headers=headers)
            return ToolResult(
                success=True,
                tool="file_creator",
                action="create_csv",
                result=filepath,
                files_created=[filepath],
            )
        except Exception as e:
            return ToolResult(
                success=False,
                tool="file_creator",
                action="create_csv",
                error=str(e),
            )
    
    def create_file(self, filename: str, content: Any, **kwargs) -> ToolResult:
        """Create any file type (auto-detected from extension)."""
        try:
            filepath = self.files.create(filename, content, **kwargs)
            return ToolResult(
                success=True,
                tool="file_creator",
                action="create_file",
                result=filepath,
                files_created=[filepath],
            )
        except Exception as e:
            return ToolResult(
                success=False,
                tool="file_creator",
                action="create_file",
                error=str(e),
            )
    
    # ========================================================================
    # DATA ANALYSIS
    # ========================================================================
    
    def analyze_data(self, data_source: str, analysis_code: str = None) -> ToolResult:
        """
        Analyze data from a file.
        
        Args:
            data_source: Path to data file (CSV, Excel, etc.)
            analysis_code: Optional Python code to run on the data
        
        Returns:
            ToolResult with analysis results
        """
        # First load the data
        ext = Path(data_source).suffix.lower()
        
        if ext in ['.csv', '.tsv']:
            load_code = f"import pandas as pd\ndf = pd.read_csv('{data_source}')"
        elif ext in ['.xlsx', '.xls']:
            load_code = f"import pandas as pd\ndf = pd.read_excel('{data_source}')"
        else:
            return ToolResult(
                success=False,
                tool="data_analysis",
                action="analyze",
                error=f"Unsupported data format: {ext}",
            )
        
        # Default analysis if no code provided
        if not analysis_code:
            analysis_code = """
print("=== Data Shape ===")
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
print()
print("=== Column Types ===")
print(df.dtypes)
print()
print("=== Summary Statistics ===")
print(df.describe())
print()
print("=== First 5 Rows ===")
print(df.head())
"""
        
        full_code = f"{load_code}\n{analysis_code}"
        return self.run_python(full_code)
    
    def visualize(self, data_source: str, chart_type: str, x: str, y: str = None, 
                  title: str = None, output_file: str = None) -> ToolResult:
        """
        Create a visualization from data.
        
        Args:
            data_source: Path to data file
            chart_type: Type of chart (line, bar, scatter, pie, histogram)
            x: X-axis column
            y: Y-axis column (optional for some charts)
            title: Chart title
            output_file: Output image file
        
        Returns:
            ToolResult with image path
        """
        output_file = output_file or f"chart_{timestamp().replace(':', '-')}.png"
        output_path = self.workspace / "outputs" / output_file
        ensure_dir(output_path.parent)
        
        code = f"""
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('{data_source}') if '{data_source}'.endswith('.csv') else pd.read_excel('{data_source}')

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Create chart
chart_type = '{chart_type}'
if chart_type == 'line':
    df.plot(x='{x}', y='{y or ""}' or None, ax=ax, kind='line')
elif chart_type == 'bar':
    df.plot(x='{x}', y='{y or ""}' or None, ax=ax, kind='bar')
elif chart_type == 'scatter':
    df.plot(x='{x}', y='{y}', ax=ax, kind='scatter')
elif chart_type == 'pie':
    df.set_index('{x}').plot(y='{y or ""}' or df.columns[1], ax=ax, kind='pie')
elif chart_type == 'histogram':
    df['{x}'].hist(ax=ax)
else:
    df.plot(ax=ax)

plt.title('{title or "Data Visualization"}')
plt.tight_layout()
plt.savefig('{output_path}', dpi=150)
plt.close()

print(f"Chart saved to: {output_path}")
"""
        
        result = self.run_python(code)
        
        if result.success:
            result.files_created = [str(output_path)]
        
        return result
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def list_files(self, directory: str = ".") -> ToolResult:
        """List files in a directory."""
        return ToolResult(
            success=True,
            tool="file_system",
            action="list",
            result=self.code.list_files(directory).return_value,
        )
    
    def get_loaded_documents(self) -> List[Dict]:
        """Get info about loaded documents."""
        return [
            {
                "id": doc.id,
                "filename": doc.filename,
                "file_type": doc.file_type,
                "word_count": doc.word_count,
            }
            for doc in self.loaded_docs.values()
        ]
    
    def clear_documents(self):
        """Clear all loaded documents."""
        self.loaded_docs.clear()
        self.doc_qa.clear()
    
    def get_workspace(self) -> str:
        """Get workspace path."""
        return str(self.workspace)
    
    def get_stats(self) -> Dict:
        """Get tools usage statistics."""
        return {
            "code_executions": self.code.get_stats(),
            "documents_loaded": len(self.loaded_docs),
            "workspace": str(self.workspace),
        }


# Export
__all__ = ['ToolsManager', 'ToolResult']
