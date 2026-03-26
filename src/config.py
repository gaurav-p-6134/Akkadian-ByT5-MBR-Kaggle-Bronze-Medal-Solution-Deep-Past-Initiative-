import os
import torch
from dataclasses import dataclass, field
from typing import List
from contextlib import nullcontext

def _cuda_bf16_supported() -> bool:
    if not torch.cuda.is_available(): return False
    try: return bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    except Exception: return False

def _bf16_ctx(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda" and _cuda_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()

@dataclass
class EnsembleMBRConfig:
    # 📁 Paths (Change these to your local or HuggingFace paths for GitHub)
    test_data_path: str = "./data/test.csv"
    train_data_path: str = "./data/train.csv"
    output_dir:     str = "./outputs/"
    
    model_a_path:   str = "assiaben/byt5-akkadian-optimized-34x"
    model_b_path:   str = "mattiaangeli/byt5-akkadian-mbr-v2"
    model_c_path:   str = "iwance/byt5-akkadian-mbrv3" 
    
    # 🚀 Generation Hyperparameters (The 36.1 Settings)
    max_input_length: int = 512
    max_new_tokens:   int = 384
    batch_size:       int = 2
    num_workers:      int = 2
    num_buckets:      int = 6
    num_beam_cands:   int = 6
    num_beams:        int = 10
    sample_temperatures: List[float] = field(default_factory=lambda: [0.60, 0.80, 1.05])
    num_sample_per_temp: int = 3
    length_penalty:      float = 1.3
    early_stopping:      bool = True
    repetition_penalty:  float = 1.2
    use_sampling:        bool = True
    mbr_top_p:           float = 0.92

    # ⚖️ MBR Settings
    mbr_pool_cap:   int = 48
    mbr_w_chrf:     float = 0.55
    mbr_w_bleu:     float = 0.25
    mbr_w_jaccard:  float = 0.20
    mbr_w_length:   float = 0.10
    
    use_mixed_precision:     bool = True
    use_bucket_batching:     bool = True

    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.output_dir, exist_ok=True)
        self.use_bf16_amp = bool(self.use_mixed_precision and self.device.type == "cuda" and _cuda_bf16_supported())