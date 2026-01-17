"""
GroundZero AI - Code Execution Engine
=====================================

Execute Python and Bash code safely with output capture.
"""

import os
import sys
import subprocess
import threading
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, field, asdict
import io

try:
    from ..utils import get_data_path, ensure_dir, logger
except ImportError:
    from utils import get_data_path, ensure_dir, logger


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    output: str = ""
    error: str = ""
    return_value: Any = None
    execution_time: float = 0.0
    language: str = "python"
    code: str = ""
    files_created: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def __str__(self) -> str:
        if self.success:
            return self.output if self.output else "(No output)"
        return f"Error: {self.error}"


class PythonExecutor:
    """Execute Python code safely."""
    
    def __init__(self, workspace: str = None, timeout: int = 60):
        self.workspace = Path(workspace) if workspace else get_data_path("workspace")
        ensure_dir(self.workspace)
        self.timeout = timeout
        self.namespace = {'__builtins__': __builtins__}
        self._setup_namespace()
    
    def _setup_namespace(self):
        """Pre-import common libraries."""
        imports = [
            "import os", "import sys", "import json", "import re", "import math",
            "from datetime import datetime, timedelta", "from pathlib import Path",
            "from collections import defaultdict, Counter"
        ]
        for code in imports:
            try:
                exec(code, self.namespace)
            except:
                pass
        
        # Optional data science imports
        for code in ["import pandas as pd", "import numpy as np"]:
            try:
                exec(code, self.namespace)
            except ImportError:
                pass
    
    def execute(self, code: str, timeout: int = None) -> ExecutionResult:
        """Execute Python code."""
        timeout = timeout or self.timeout
        start_time = datetime.now()
        
        original_cwd = os.getcwd()
        os.chdir(self.workspace)
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        result = ExecutionResult(success=False, language="python", code=code)
        
        try:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = stdout_capture, stderr_capture
            
            files_before = set(self.workspace.glob("**/*"))
            
            # Execute with timeout
            exec_result = {"success": False}
            
            def run():
                try:
                    exec(code, self.namespace)
                    exec_result["success"] = True
                except Exception as e:
                    exec_result["error"] = f"{type(e).__name__}: {e}"
            
            thread = threading.Thread(target=run)
            thread.start()
            thread.join(timeout)
            
            if thread.is_alive():
                result.error = f"Timeout after {timeout}s"
            else:
                result.success = exec_result.get("success", False)
                result.error = exec_result.get("error", "")
            
            files_after = set(self.workspace.glob("**/*"))
            result.files_created = [str(f) for f in (files_after - files_before)]
            
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            result.output = stdout_capture.getvalue()
            if stderr_capture.getvalue():
                result.error += stderr_capture.getvalue()
            os.chdir(original_cwd)
            result.execution_time = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def install_package(self, package: str) -> ExecutionResult:
        """Install a Python package."""
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pip", "install", package, "-q"],
                capture_output=True, text=True, timeout=120
            )
            return ExecutionResult(
                success=proc.returncode == 0,
                output=proc.stdout,
                error=proc.stderr
            )
        except Exception as e:
            return ExecutionResult(success=False, error=str(e))


class BashExecutor:
    """Execute bash/shell commands."""
    
    def __init__(self, workspace: str = None, timeout: int = 60):
        self.workspace = Path(workspace) if workspace else get_data_path("workspace")
        ensure_dir(self.workspace)
        self.timeout = timeout
    
    def execute(self, command: str, timeout: int = None) -> ExecutionResult:
        """Execute a bash command."""
        timeout = timeout or self.timeout
        start_time = datetime.now()
        
        result = ExecutionResult(success=False, language="bash", code=command)
        
        try:
            files_before = set(self.workspace.glob("**/*"))
            
            proc = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=timeout, cwd=str(self.workspace)
            )
            
            result.success = proc.returncode == 0
            result.output = proc.stdout
            result.error = proc.stderr
            result.return_value = proc.returncode
            
            files_after = set(self.workspace.glob("**/*"))
            result.files_created = [str(f) for f in (files_after - files_before)]
            
        except subprocess.TimeoutExpired:
            result.error = f"Timeout after {timeout}s"
        except Exception as e:
            result.error = str(e)
        finally:
            result.execution_time = (datetime.now() - start_time).total_seconds()
        
        return result


class CodeExecutor:
    """Unified code execution engine."""
    
    def __init__(self, workspace: str = None):
        self.workspace = Path(workspace) if workspace else get_data_path("workspace")
        ensure_dir(self.workspace)
        
        self.python = PythonExecutor(str(self.workspace))
        self.bash = BashExecutor(str(self.workspace))
        
        self.stats = {"total": 0, "successful": 0, "failed": 0}
    
    def execute(self, code: str, language: str = "python", timeout: int = 60) -> ExecutionResult:
        """Execute code in specified language."""
        if language.lower() in ["python", "py"]:
            result = self.python.execute(code, timeout)
        elif language.lower() in ["bash", "sh", "shell"]:
            result = self.bash.execute(code, timeout)
        else:
            result = ExecutionResult(success=False, error=f"Unsupported: {language}")
        
        self.stats["total"] += 1
        self.stats["successful" if result.success else "failed"] += 1
        return result
    
    def run_python(self, code: str, **kwargs) -> ExecutionResult:
        return self.execute(code, "python", **kwargs)
    
    def run_bash(self, command: str, **kwargs) -> ExecutionResult:
        return self.execute(command, "bash", **kwargs)
    
    def install_package(self, package: str) -> ExecutionResult:
        return self.python.install_package(package)
    
    def list_files(self, directory: str = ".") -> ExecutionResult:
        """List files in directory."""
        try:
            path = self.workspace / directory
            files = [{"name": f.name, "type": "dir" if f.is_dir() else "file", 
                     "size": f.stat().st_size} for f in path.glob("*")]
            return ExecutionResult(success=True, return_value=files,
                                  output="\n".join(f["name"] for f in files))
        except Exception as e:
            return ExecutionResult(success=False, error=str(e))
    
    def get_workspace(self) -> Path:
        return self.workspace
    
    def get_stats(self) -> Dict:
        return self.stats


__all__ = ['ExecutionResult', 'PythonExecutor', 'BashExecutor', 'CodeExecutor']
