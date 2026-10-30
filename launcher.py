import os
import http.server
import socketserver
import threading
import webbrowser
import time
import urllib.parse

PORT = 8000
DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public")

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
        
    # Ensure proper MIME types, especially if some viewers are strict
    extensions_map = http.server.SimpleHTTPRequestHandler.extensions_map.copy()
    extensions_map.update({
        '.gltf': 'model/gltf+json',
        '.glb': 'model/gltf-binary',
        '.js': 'application/javascript',
    })

def start_server():
    # Try ports in case 8000 is taken
    global PORT
    while True:
        try:
            httpd = socketserver.TCPServer(("", PORT), Handler)
            break
        except OSError:
            PORT += 1
            
    print(f"Serving directory {DIRECTORY} at http://localhost:{PORT}")
    httpd.serve_forever()

def main():
    # Start the HTTP server in a background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Wait a brief moment to ensure the server starts
    time.sleep(1)
    
    # The file we want to open
    filename = "index.html"
    
    # Properly encode the filename for the URL
    url = f"http://localhost:{PORT}/{urllib.parse.quote(filename)}"
    
    print(f"Opening {url} in your default browser...")
    webbrowser.open(url)
    
    print("\nThe server is running to allow the 3D .gltf file to load!")
    print("Keep this script open while you are viewing the dashboard.")
    print("Press Ctrl+C to stop the server and exit.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping server...")

if __name__ == "__main__":
    main()
