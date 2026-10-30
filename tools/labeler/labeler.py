import http.server
import socketserver
import json
import webbrowser
import os
import sys
import threading
import time

PORT = 8080

class LabelerHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS and disable caching
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        if self.path == '/save':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode('utf-8'))
                
                labels_file = 'data/engine_labels.json'
                existing_data = []
                if os.path.exists(labels_file):
                    try:
                        with open(labels_file, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)
                    except Exception:
                        existing_data = []
                
                # Check if label key already exists and update it, else append
                updated = False
                for idx, item in enumerate(existing_data):
                    if item.get('key') == data.get('key'):
                        existing_data[idx] = data
                        updated = True
                        break
                
                if not updated:
                    existing_data.append(data)
                
                with open(labels_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, indent=2)
                
                print(f"[SUCCESS] Saved box label: '{data.get('key')}'")
                print(f"          Center: {data.get('center')}")
                print(f"          Size:   {data.get('size')}")
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "message": "Box label saved successfully"}).encode('utf-8'))
            except Exception as e:
                print(f"[ERROR] Failed to save label: {e}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode('utf-8'))
        elif self.path == '/rename':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode('utf-8'))
                old_key = data.get('oldKey')
                new_key = data.get('newKey')
                
                if not old_key or not new_key:
                    raise ValueError("oldKey and newKey are required")
                
                labels_file = 'data/engine_labels.json'
                existing_data = []
                if os.path.exists(labels_file):
                    try:
                        with open(labels_file, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)
                    except Exception:
                        existing_data = []
                
                # Check if new_key already exists to prevent duplicate keys
                if any(item.get('key') == new_key for item in existing_data if item.get('key') != old_key):
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "error", "message": f"A box with key '{new_key}' already exists."}).encode('utf-8'))
                    return
                
                # Rename the matching key
                found = False
                for idx, item in enumerate(existing_data):
                    if item.get('key') == old_key:
                        existing_data[idx]['key'] = new_key
                        found = True
                        break
                
                if not found:
                    self.send_response(404)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "error", "message": f"Box with key '{old_key}' not found."}).encode('utf-8'))
                    return
                
                with open(labels_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, indent=2)
                
                print(f"[SUCCESS] Renamed box: '{old_key}' -> '{new_key}'")
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "message": "Box renamed successfully"}).encode('utf-8'))
            except Exception as e:
                print(f"[ERROR] Failed to rename label: {e}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode('utf-8'))
        elif self.path == '/delete':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode('utf-8'))
                key = data.get('key')
                
                if not key:
                    raise ValueError("key is required")
                
                labels_file = 'data/engine_labels.json'
                existing_data = []
                if os.path.exists(labels_file):
                    try:
                        with open(labels_file, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)
                    except Exception:
                        existing_data = []
                
                # Filter out the matching key
                new_data = [item for item in existing_data if item.get('key') != key]
                
                if len(new_data) == len(existing_data):
                    self.send_response(404)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "error", "message": f"Box with key '{key}' not found."}).encode('utf-8'))
                    return
                
                with open(labels_file, 'w', encoding='utf-8') as f:
                    json.dump(new_data, f, indent=2)
                
                print(f"[SUCCESS] Deleted box: '{key}'")
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "message": "Box deleted successfully"}).encode('utf-8'))
            except Exception as e:
                print(f"[ERROR] Failed to delete label: {e}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def open_browser():
    time.sleep(1.0)
    print("\n[INFO] Opening visual labeler in browser...")
    webbrowser.open(f"http://localhost:{PORT}/labeler_ui.html")

if __name__ == '__main__':
    # Ensure server starts in the directory of this script so gltf is served
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'public'))
    
    handler = LabelerHTTPRequestHandler
    # Allow socket reuse to prevent port-in-use errors on restarts
    socketserver.TCPServer.allow_reuse_address = True
    
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"[INFO] 3D CAD Bounding Box Labeler Server running at http://localhost:{PORT}/")
        print("[INFO] Press Ctrl+C to stop the server.")
        
        # Start browser in a background thread
        threading.Thread(target=open_browser, daemon=True).start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[INFO] Stopping server. Bye!")
            sys.exit(0)
