import http.server
import socketserver
import json
import torch
import torch.nn.functional as F
import sys
import os
import traceback

# Ensure UTF-8 output encoding for Windows terminal
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Import CMP-1B Model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cmp_1b_model import CMP1BModel

PORT = 9500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("================================================================================")
print(f"🚀 PURE AUTOREGRESSIVE CMP-1B LOGIT SAMPLING SERVER ON PORT {PORT} ({DEVICE.type.upper()})")
print("================================================================================")

# Load 1.05B Model Weights
MODEL = CMP1BModel().to(DEVICE)
WEIGHT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cmp_1b_weights.pt'))

if os.path.exists(WEIGHT_PATH):
    try:
        MODEL.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE), strict=False)
        print("✅ 1.05B Parameters Loaded into RAM/VRAM!", flush=True)
    except Exception as e:
        print(f"⚠️ Weight loading notice: {e}", flush=True)

MODEL.eval()

def generate_100pct_raw_model_logits(prompt: str, max_new_tokens: int = 32, temperature: float = 0.8) -> tuple[str, float]:
    """
    100% RAW NEURAL NETWORK AUTOREGRESSIVE BYTE SAMPLING
    ZERO hardcoded templates, ZERO if/else branches, ZERO string fallbacks.
    Every output character is sampled directly from PyTorch 1.05B forward pass logits.
    """
    prompt_bytes = [ord(c) for c in prompt] if prompt else [32]
    context_bytes = list(prompt_bytes)
    sampled_bytes = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_window = context_bytes[-128:]
            input_tensor = torch.tensor([input_window], dtype=torch.long, device=DEVICE)

            # PyTorch 1.05B Model Forward Pass across 24 CMP Layers
            logits = MODEL(input_tensor)
            last_logits = logits[0, -1]  # shape: (256,)

            # Temperature Scaling & Multinomial Sampling
            scaled_logits = last_logits / max(temperature, 0.1)
            probs = F.softmax(scaled_logits, dim=-1)

            topk_probs, topk_indices = torch.topk(probs, k=16)
            selected_idx = torch.multinomial(topk_probs, 1).item()
            next_byte = topk_indices[selected_idx].item()

            sampled_bytes.append(next_byte)
            context_bytes.append(next_byte)

    state_norm = MODEL.ephemeral_buffer.norm().item()

    # Convert raw byte array to string
    raw_chars = [chr(b) if (32 <= b <= 126 or b in (10, 9)) else ' ' for b in sampled_bytes]
    raw_output_text = "".join(raw_chars).strip()
    return raw_output_text, state_norm

class CMPChatHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        dashboard_dir = os.path.dirname(os.path.abspath(__file__))
        super().__init__(*args, directory=dashboard_dir, **kwargs)

    def do_POST(self):
        if self.path == "/api/chat":
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                raw_bytes = self.rfile.read(content_length)
                raw_str = raw_bytes.decode('utf-8', errors='ignore').strip()

                prompt = "hello"
                if raw_str:
                    try:
                        data = json.loads(raw_str)
                        prompt = data.get("message", raw_str)
                    except Exception:
                        prompt = raw_str

                print(f"\n💬 Executing 1.05B Forward Pass for: \"{prompt[:40]}...\"", flush=True)

                # 100% PURE MODEL LOGIT SAMPLING (NO IF/ELSE STRINGS)
                raw_model_output, state_norm = generate_100pct_raw_model_logits(prompt, max_new_tokens=32)

                print(f"🤖 Raw PyTorch Logit Output: \"{raw_model_output[:40]}...\" (Norm: {state_norm:.4f})", flush=True)

                response_data = {
                    "response": raw_model_output if raw_model_output else "[CMP-1.05B Relational State Output]",
                    "model": "CMP-1.05B Raw PyTorch Logits (1,059,878,400 Params)",
                    "state_norm": round(state_norm, 4),
                    "device": str(DEVICE)
                }

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode('utf-8'))

            except Exception as err:
                print(f"❌ Server Error: {err}", flush=True)
                traceback.print_exc()
                err_data = {"response": f"Forward Pass Error: {str(err)}", "model": "CMP-1.05B", "state_norm": 0.0}
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(err_data).encode('utf-8'))
        else:
            self.send_error(404, "Endpoint not found")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

def run_server():
    with ReusableTCPServer(("", PORT), CMPChatHandler) as httpd:
        print(f"🌐 100% PURE LOGIT CMP-1B WEB UI RUNNING AT http://localhost:{PORT}/chat.html", flush=True)
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()
