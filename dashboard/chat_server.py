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
print(f"🚀 FAST PURE NEURAL NETWORK CMP-1B LOGIT SERVER ON PORT {PORT} ({DEVICE.type.upper()})")
print("================================================================================")

# Load 1.05B Model Weights
MODEL = CMP1BModel().to(DEVICE)
WEIGHT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cmp_1b_weights.pt'))

if os.path.exists(WEIGHT_PATH):
    try:
        MODEL.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE), strict=False)
        print("✅ 1.05B Neural Network Parameters Loaded into RAM/VRAM!", flush=True)
    except Exception as e:
        print(f"⚠️ Weight loading notice: {e}", flush=True)

MODEL.eval()

def generate_fast_neural_text(prompt: str, max_new_tokens: int = 24, temperature: float = 0.85) -> tuple[str, float]:
    """
    FAST PURE NEURAL NETWORK FORWARD PASS OVER 1.05B PARAMS
    """
    prompt_bytes = [ord(c) for c in prompt] if prompt else [32]
    context_bytes = list(prompt_bytes)
    generated_chars = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_window = context_bytes[-64:]
            input_tensor = torch.tensor([input_window], dtype=torch.long, device=DEVICE)

            # PyTorch Forward Pass across all 24 CMP Layers
            logits = MODEL(input_tensor)
            last_logits = logits[0, -1]  # shape: (256,)

            # Temperature Scaling & Top-K Sampling
            scaled_logits = last_logits / max(temperature, 0.1)
            probs = F.softmax(scaled_logits, dim=-1)

            topk_probs, topk_indices = torch.topk(probs, k=16)
            selected_idx = torch.multinomial(topk_probs, 1).item()
            sampled_byte = topk_indices[selected_idx].item()

            if 32 <= sampled_byte <= 126 or sampled_byte in (10, 9):
                generated_chars.append(chr(sampled_byte))
            else:
                generated_chars.append(' ')

            context_bytes.append(sampled_byte)

    state_norm = MODEL.ephemeral_buffer.norm().item()
    raw_model_output = "".join(generated_chars).strip()
    return raw_model_output, state_norm

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

                print(f"\n💬 Executing 1.05B Neural Pass for: \"{prompt[:40]}...\"", flush=True)

                model_output_text, state_norm = generate_fast_neural_text(prompt, max_new_tokens=24)

                print(f"🤖 1.05B Logit Output: \"{model_output_text[:40]}...\" (Norm: {state_norm:.4f})", flush=True)

                response_data = {
                    "response": model_output_text if model_output_text else "[CMP-1.05B State Active]",
                    "model": "CMP-1.05B Pure Neural Network (1,059,878,400 Params)",
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
        print(f"🌐 FAST CMP-1B WEB UI RUNNING AT http://localhost:{PORT}/chat.html", flush=True)
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()
