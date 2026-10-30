import threading
import webbrowser

import uvicorn

PORT = 8000

if __name__ == "__main__":
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{PORT}/index.html")).start()
    uvicorn.run("backend.app:app", host="127.0.0.1", port=PORT)
