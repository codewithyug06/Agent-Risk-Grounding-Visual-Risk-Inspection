import torch
from collections import deque
import sys
import os
from PIL import Image
import io
import torchvision.transforms as T

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.models.sentinel_model import SentinelModel
from src.gate.decision_gate import DecisionGate

class SentinelInferenceBridge:
    def __init__(self, checkpoint_path=None, seq_len=5, H=224, W=224, mock_mode=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seq_len = seq_len
        self.H = H
        self.W = W
        self.mock_mode = mock_mode
        self.frame_buffer = deque(maxlen=seq_len)
        
        self.transform = T.Compose([
            T.Resize((H, W)),
            T.ToTensor(),
            # Normalization matching timm convnext
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if not self.mock_mode:
            from omegaconf import OmegaConf
            _cfg = OmegaConf.create({
                "backbone": "convnext_tiny",
                "pretrained": False,
                "image_size": 224,
                "frame_window": {"k": self.seq_len},
                "temporal": {"num_heads": 8, "num_layers": 3, "dropout": 0.1, "max_frames": 8},
                "risk_head": {"hidden_dim": 256, "num_categories": 5},
                "localization": {"anchor_sizes": [32, 64, 128, 256], "num_anchors": 9, "iou_threshold": 0.5},
            })
            self.model = SentinelModel(_cfg).to(self.device)
            if checkpoint_path and os.path.exists(checkpoint_path):
                print(f"[Bridge] Loading checkpoint from {checkpoint_path}")
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device, weights_only=True))
            else:
                print(f"[Bridge] Warning: Checkpoint not found at {checkpoint_path}. Using random weights.")
            self.model.eval()
            
            # Load RL Gate
            self.gate = DecisionGate().to(self.device)
            gate_path = 'checkpoints/gate_rl.pt'
            if os.path.exists(gate_path):
                print(f"[Bridge] Loading RL Gate from {gate_path}")
                self.gate.load_state_dict(torch.load(gate_path, map_location=self.device, weights_only=True))
            self.gate.eval()
        else:
            print("[Bridge] Running in MOCK MODE (predicting random blocks for testing UI).")

    def predict(self, image_bytes: bytes, target_coords: dict = None) -> dict:
        """
        Takes raw screenshot bytes, appends to buffer, and runs inference.
        Returns {'action': 'ALLOW'|'BLOCK', 'box': {'x','y','w','h'} or None}
        """
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            # Store original size for scaling box back
            orig_w, orig_h = image.size 
            tensor = self.transform(image) # (3, H, W)
            self.frame_buffer.append(tensor)
        except Exception as e:
            print(f"[Bridge] Error processing image: {e}")
            return {'action': 'ALLOW'} # Fail open
            
        if len(self.frame_buffer) < self.seq_len:
            return {'action': 'ALLOW'} # Not enough context yet
            
        if self.mock_mode:
            # Hardcoded mock behavior: Block if click is in bottom half of screen
            # For demonstration without a fully trained model on real UI
            if target_coords and target_coords['y'] > orig_h / 2:
                # Return a box around the click
                x = max(0, target_coords['x'] - 50)
                y = max(0, target_coords['y'] - 20)
                return {'action': 'BLOCK', 'box': {'x': x, 'y': y, 'w': 100, 'h': 40}}
            return {'action': 'ALLOW'}
            
        # Real inference
        with torch.no_grad():
            x = torch.stack(list(self.frame_buffer)).unsqueeze(0).to(self.device)
            outputs = self.model(x)
            
            risk_logits = outputs['risk_logits']
            category_logits = outputs['category_logits']
            
            # Get action from RL Gate
            action_logits, _ = self.gate(risk_logits, category_logits)
            action_idx = action_logits.argmax(dim=-1).item()
            
            action_map = {0: 'ALLOW', 1: 'PAUSE', 2: 'BLOCK'}
            action_str = action_map[action_idx]
            
            if action_str in ['BLOCK', 'PAUSE']: 
                # Extract box
                pred_boxes = outputs['box_regs'].mean(dim=(2, 3))
                bx1, by1, bx2, by2 = pred_boxes[0].cpu().tolist()
                
                scale_x = orig_w / self.W
                scale_y = orig_h / self.H
                
                x = bx1 * scale_x
                y = by1 * scale_y
                w = (bx2 - bx1) * scale_x
                h = (by2 - by1) * scale_y
                
                x, y = max(0, x), max(0, y)
                w = max(10, min(w, orig_w - x))
                h = max(10, min(h, orig_h - y))
                
                return {
                    'action': action_str,
                    'box': {'x': x, 'y': y, 'w': w, 'h': h},
                    'risk': torch.sigmoid(risk_logits).item()
                }
                
        return {'action': 'ALLOW'}
