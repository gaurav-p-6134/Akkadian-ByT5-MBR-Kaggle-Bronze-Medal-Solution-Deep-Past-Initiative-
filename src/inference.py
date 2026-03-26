import gc
import re
import math
import random
import logging
import torch
import pandas as pd
from typing import List
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rapidfuzz import process, fuzz

# Import your custom modules
from config import EnsembleMBRConfig, _bf16_ctx
from data_processing import OptimizedPreprocessor, VectorizedPostprocessor
from mbr_selection import MBRSelector

# --- DATASET & SAMPLER ---
class AkkadianDataset(Dataset):
    def __init__(self, df: pd.DataFrame, preprocessor: OptimizedPreprocessor):
        self.sample_ids = df["id"].tolist()
        proc = preprocessor.preprocess_batch(df["transliteration"].tolist())
        self.input_texts = ["translate Akkadian to English: " + t for t in proc]

    def __len__(self): return len(self.sample_ids)
    def __getitem__(self, idx): return self.sample_ids[idx], self.input_texts[idx]

class BucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_buckets, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        lengths = [len(t.split()) for _, t in dataset]
        sorted_idx = sorted(range(len(lengths)), key=lambda i: lengths[i])
        bsize = max(1, len(sorted_idx) // max(1, num_buckets))
        self.buckets = [sorted_idx[i*bsize : None if i == num_buckets-1 else (i+1)*bsize] for i in range(num_buckets)]

    def __iter__(self):
        for bucket in self.buckets:
            b = list(bucket)
            if self.shuffle: random.shuffle(b)
            for i in range(0, len(b), self.batch_size):
                yield b[i:i+self.batch_size]

    def __len__(self):
        return sum(math.ceil(len(b) / self.batch_size) for b in self.buckets)

# --- MODEL WRAPPER ---
class ModelWrapper:
    def __init__(self, model_path: str, cfg: EnsembleMBRConfig, label: str):
        self.cfg = cfg
        self.label = label
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(cfg.device).eval()
        self.model.config.use_cache = True # Force cache ON to prevent timeout bug

    def collate(self, batch_samples):
        ids = [s[0] for s in batch_samples]
        texts = [s[1] for s in batch_samples]
        enc = self.tokenizer(texts, max_length=self.cfg.max_input_length, padding=True, truncation=True, return_tensors="pt")
        return ids, enc

    def generate_candidates(self, input_ids, attention_mask, beam_size: int) -> List[List[str]]:
        cfg = self.cfg
        B = input_ids.shape[0]
        ctx = _bf16_ctx(cfg.device, cfg.use_bf16_amp)
        all_samp_texts = []
        
        with ctx:
            # 1. Base Beam Search
            beam_out = self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask, do_sample=False, 
                num_beams=cfg.num_beams, num_return_sequences=cfg.num_beam_cands, 
                max_new_tokens=cfg.max_new_tokens, length_penalty=cfg.length_penalty, early_stopping=cfg.early_stopping
            )
            all_samp_texts.extend(self.tokenizer.batch_decode(beam_out, skip_special_tokens=True))

            # 2. Temperature Sampling (High-Risk/High-Reward 36.1 Setup)
            if cfg.use_sampling:
                for temp in cfg.sample_temperatures:
                    samp_out = self.model.generate(
                        input_ids=input_ids, attention_mask=attention_mask, do_sample=True, 
                        num_beams=1, top_p=cfg.mbr_top_p, temperature=temp, 
                        num_return_sequences=cfg.num_sample_per_temp, max_new_tokens=cfg.max_new_tokens
                    )
                    all_samp_texts.extend(self.tokenizer.batch_decode(samp_out, skip_special_tokens=True))
                    
        # Reshape candidates per batch item
        total_cands = cfg.num_beam_cands + (len(cfg.sample_temperatures) * cfg.num_sample_per_temp if cfg.use_sampling else 0)
        candidates = []
        for i in range(B):
            start = i * total_cands
            candidates.append(all_samp_texts[start:start+total_cands])
            
        return candidates

    def unload(self):
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()

# --- MAIN ENGINE ---
class EnsembleMBREngine:
    def __init__(self, cfg: EnsembleMBRConfig):
        self.cfg = cfg
        self.preprocessor = OptimizedPreprocessor()
        self.postprocessor = VectorizedPostprocessor()
        self.mbr = MBRSelector(pool_cap=cfg.mbr_pool_cap, w_chrf=cfg.mbr_w_chrf, w_bleu=cfg.mbr_w_bleu, w_jaccard=cfg.mbr_w_jaccard)

    def _run_one_model(self, wrapper: ModelWrapper, dataset: AkkadianDataset) -> dict:
        sampler = BucketBatchSampler(dataset, self.cfg.batch_size, self.cfg.num_buckets)
        dl = DataLoader(dataset, batch_sampler=sampler, collate_fn=wrapper.collate, pin_memory=True)
        pools_by_id = {}

        with torch.inference_mode():
            for batch_ids, enc in tqdm(dl, desc=f"[{wrapper.label}] Inference"):
                input_ids = enc.input_ids.to(self.cfg.device, non_blocking=True)
                attn = enc.attention_mask.to(self.cfg.device, non_blocking=True)
                batch_pools = wrapper.generate_candidates(input_ids, attn, self.cfg.num_beams)
                for sid, pool in zip(batch_ids, batch_pools):
                    pools_by_id[str(sid)] = pool
        return pools_by_id

    def run(self, test_df: pd.DataFrame, train_dict: dict) -> pd.DataFrame:
        dataset = AkkadianDataset(test_df, self.preprocessor)
        sample_ids = [str(s) for s in dataset.sample_ids]
        
        # Sequentially run models to save VRAM
        pools = {sid: [] for sid in sample_ids}
        model_paths = [("Model-A", self.cfg.model_a_path), ("Model-B", self.cfg.model_b_path), ("Model-C", self.cfg.model_c_path)]
        
        for label, path in model_paths:
            wrapper = ModelWrapper(path, self.cfg, label)
            m_pool = self._run_one_model(wrapper, dataset)
            for sid in sample_ids: pools[sid].extend(m_pool.get(sid, []))
            wrapper.unload()

        # RAG + MBR Resolution
        results = []
        train_keys = list(train_dict.keys())
        
        for idx, sid in enumerate(tqdm(sample_ids, desc="MBR Selection")):
            raw_text = str(test_df.iloc[idx]['transliteration'])
            cln_input = self.preprocessor.preprocess_batch([raw_text])[0]
            cln_input_no_hints = re.sub(r'\s*\[Commodities:[^\]]*\]', '', cln_input).strip()

            match = process.extractOne(cln_input_no_hints, train_keys, scorer=fuzz.token_sort_ratio)
            length_ratio = len(cln_input_no_hints) / max(1, len(match[0])) if match else 0
            
            if match and match[1] >= 88.0 and 0.8 <= length_ratio <= 1.2:
                chosen = train_dict[match[0]]
            else:
                pp = self.postprocessor.postprocess_batch(pools[sid])
                chosen = self.mbr.pick(pp)
                
            results.append((sid, chosen or "The tablet is too damaged to translate."))
            
        return pd.DataFrame(results, columns=["id", "translation"])

if __name__ == "__main__":
    cfg = EnsembleMBRConfig()
    
    # 1. Load Datasets
    test_df = pd.read_csv(cfg.test_data_path)
    train_df = pd.read_csv(cfg.train_data_path)
    
    # 2. Build RAG Dictionary
    preprocessor = OptimizedPreprocessor()
    train_dict = {}
    for _, row in train_df.dropna(subset=['transliteration', 'translation']).iterrows():
        cln_akk = preprocessor.preprocess_batch([str(row['transliteration'])])[0]
        cln_akk = re.sub(r'\s*\[Commodities:[^\]]*\]', '', cln_akk).strip()
        train_dict[cln_akk] = str(row['translation']).strip()
        
    # 3. Execute
    engine = EnsembleMBREngine(cfg)
    results_df = engine.run(test_df, train_dict)
    
    # 4. Save
    results_df.to_csv(cfg.output_dir + "submission.csv", index=False)
    print("Inference Complete. Saved to outputs/submission.csv")