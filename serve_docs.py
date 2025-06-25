#!/usr/bin/env python3
"""
Simple script to serve QUSIM documentation
"""
import os
import http.server
import socketserver
import webbrowser
from pathlib import Path

def serve_docs():
    """Serve the documentation on localhost:8000"""
    
    # Get the docs directory
    docs_dir = Path(__file__).parent / "docs" / "_build" / "html"
    
    if not docs_dir.exists():
        print("❌ Documentation not found!")
        print("Please run: cd nvcore && make docs")
        return
    
    # Change to docs directory
    os.chdir(docs_dir)
    
    PORT = 8000
    Handler = http.server.SimpleHTTPRequestHandler
    
    print(f"🌐 Starting QUSIM Documentation Server...")
    print(f"📖 Documentation: http://localhost:{PORT}")
    print(f"📁 Serving from: {docs_dir}")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            # Try to open browser
            try:
                webbrowser.open(f'http://localhost:{PORT}')
                print("🚀 Opened browser automatically")
            except:
                print("💡 Open your browser and go to http://localhost:8000")
            
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Documentation server stopped")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Port {PORT} is already in use")
            print("Try a different port or stop the existing server")
        else:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    serve_docs()