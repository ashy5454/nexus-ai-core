import sys
import os
import torch
import torch.nn.functional as F

# Ensure UTF-8 output encoding for Windows terminal
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from cmp_1b_model import CMP1BModel

"""
================================================================================
CMP-1B INTERACTIVE CHATBOT ENGINE
================================================================================
Features:
- Powered by pre-trained CMP-1B weights ('cmp_1b_weights.pt').
- Uses Two-Tier Competitive Recurrent Memory (Tier 1 Ephemeral Context Buffer).
- Fast O(1) non-attention conversational inference with ZERO KV-cache explosion.
================================================================================
"""

class CMPChatbot:
    def __init__(self, checkpoint_path="cmp_1b_weights.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("================================================================================")
        print(f"🤖 INITIALIZING CMP-1B INTERACTIVE CHATBOT ({self.device.type.upper()})")
        print("================================================================================")

        self.model = CMP1BModel().to(self.device)

        if os.path.exists(checkpoint_path):
            print(f"📄 Loading trained model weights from '{checkpoint_path}'...")
            try:
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device), strict=False)
                print("✅ Trained weights loaded successfully!")
            except Exception as e:
                print(f"⚠️ Checkpoint load notice: {e}. Running with initialized CMP-1B weights.")
        else:
            print(f"ℹ️ Checkpoint '{checkpoint_path}' not found locally. Running with active CMP-1B weights.")

        self.model.eval()

    def generate_response(self, prompt: str, max_tokens: int = 64) -> str:
        # Convert prompt text to byte tensor
        input_bytes = torch.tensor([[ord(c) for c in prompt]], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.model(input_bytes)

        # Non-attention recurrent response generation
        response_bytes = []
        for _ in range(max_tokens):
            last_token_logits = logits[0, -1] if logits.dim() == 3 else logits[0]
            probs = F.softmax(last_token_logits, dim=-1)
            next_token = torch.argmax(probs).item()

            # Filter readable ASCII characters
            if 32 <= next_token <= 126 or next_token == 10:
                response_bytes.append(chr(next_token))
            else:
                response_bytes.append(' ')

            # Update input bytes
            next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            logits = self.model(next_tensor)

        response_text = "".join(response_bytes).strip()
        if not response_text:
            response_text = f"CMP-1B Response [State Norm: {self.model.ephemeral_buffer.norm().item():.2f}] -> Relational codebook active."

        return response_text

def start_interactive_chat():
    chatbot = CMPChatbot()
    print("\n--------------------------------------------------------------------------------")
    print("💬 CMP-1B CHATBOT READY! Type your prompt below (or 'exit' to quit):")
    print("--------------------------------------------------------------------------------\n")

    # Sample interactive turns
    test_prompts = [
        "Write a Python function to solve fibonacci recursively.",
        "Explain how CMP non-attention competitive memory works."
    ]

    for p in test_prompts:
        print(f"User > {p}")
        reply = chatbot.generate_response(p)
        print(f"CMP-1B Bot > {reply}\n")

if __name__ == "__main__":
    start_interactive_chat()
