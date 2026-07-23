import http.server
import socketserver
import json
import os
import sys

# Ensure UTF-8 output encoding for Windows terminal
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

PORT = 9900
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(DASHBOARD_DIR, "reddit_insights_data.json")

print("================================================================================")
print(f"🚀 STARTING ECHOREGENT 100K REDDIT INSIGHTS DASHBOARD SERVER ON PORT {PORT}")
print("================================================================================")

class InsightsDashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)

    def do_GET(self):
        if self.path == "/api/insights":
            try:
                if os.path.exists(DATA_PATH):
                    with open(DATA_PATH, "r", encoding="utf-8") as f:
                        data = json.load(f)
                else:
                    data = {"error": "Data file not found. Run reddit_word_frequency_analyzer.py first."}

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(data).encode('utf-8'))
            except Exception as err:
                self.send_error(500, f"Error loading insights: {err}")
        else:
            super().do_GET()

class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

def run_server():
    with ReusableTCPServer(("", PORT), InsightsDashboardHandler) as httpd:
        print(f"🌐 ECHOREGENT INSIGHTS DASHBOARD LIVE AT http://localhost:{PORT}/insights_dashboard.html", flush=True)
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()
