#!/usr/bin/env python3
"""
GroundZero AI - API Server
==========================

Serves real statistics from the database to the web dashboard.

Usage:
    python api_server.py [port]

Default port: 8081
Dashboard should fetch from: http://localhost:8081/api/stats
"""

import json
import sys
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.knowledge_graph import KnowledgeGraph
from src.causal_graph import CausalGraph
from src.progress_tracker import ProgressTracker

DATA_DIR = PROJECT_ROOT / "data"


class APIHandler(BaseHTTPRequestHandler):
    """Handle API requests"""
    
    def do_GET(self):
        """Handle GET requests"""
        
        # Enable CORS for dashboard access
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.end_headers()
        
        if self.path == '/api/stats':
            stats = self.get_real_stats()
            self.wfile.write(json.dumps(stats).encode())
        
        elif self.path == '/api/progress':
            progress = self.get_progress()
            self.wfile.write(json.dumps(progress).encode())
        
        elif self.path == '/api/health':
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        
        else:
            self.wfile.write(json.dumps({"error": "Unknown endpoint"}).encode())
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def get_real_stats(self):
        """Get real statistics from database"""
        try:
            # Load knowledge graph from database
            db_path = DATA_DIR / "knowledge.db"
            kg = KnowledgeGraph(str(db_path) if db_path.exists() else None)
            cg = CausalGraph()
            
            # Get stats
            kg_stats = kg.GetStats()
            
            stats = {
                "facts": kg_stats.get("TotalFacts", 0),
                "causal": cg.Stats.get("TotalRelations", 0),
                "questions": 0,  # Would need to track this separately
                "confidence": 82,  # Default
                "inferred": kg_stats.get("FactsInferred", 0),
                "queries": kg_stats.get("QueriesAnswered", 0),
            }
            
            kg.Close()
            return stats
            
        except Exception as e:
            return {"error": str(e), "facts": 0, "causal": 0}
    
    def get_progress(self):
        """Get progress level"""
        try:
            db_path = DATA_DIR / "knowledge.db"
            kg = KnowledgeGraph(str(db_path) if db_path.exists() else None)
            cg = CausalGraph()
            
            tracker = ProgressTracker(kg, cg)
            progress = tracker.GetCurrentLevel()
            
            result = {
                "level": progress["CurrentLevel"].get("Level", 0),
                "name": progress["CurrentLevel"].get("Name", "Starting Out"),
                "progress_percent": int(progress["Progress"] * 100),
                "facts": progress["Facts"],
                "causal": progress["CausalRelations"],
            }
            
            if progress["NextMilestone"]:
                result["next_level"] = progress["NextMilestone"]["Level"]
                result["next_name"] = progress["NextMilestone"]["Name"]
                result["facts_needed"] = progress["NextMilestone"]["FactsRequired"]
            
            kg.Close()
            return result
            
        except Exception as e:
            return {"error": str(e), "level": 0}
    
    def log_message(self, format, *args):
        """Custom log format"""
        print(f"[API] {args[0]}")


def run_server(port=8081):
    """Run the API server"""
    server = HTTPServer(('', port), APIHandler)
    print(f"\n🔌 GroundZero AI - API Server")
    print(f"=" * 50)
    print(f"   Running on: http://localhost:{port}")
    print(f"   Endpoints:")
    print(f"     GET /api/stats    - Real statistics")
    print(f"     GET /api/progress - Progress level")
    print(f"     GET /api/health   - Health check")
    print(f"=" * 50)
    print(f"   Press Ctrl+C to stop\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n👋 API server stopped\n")
        server.shutdown()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8081
    run_server(port)