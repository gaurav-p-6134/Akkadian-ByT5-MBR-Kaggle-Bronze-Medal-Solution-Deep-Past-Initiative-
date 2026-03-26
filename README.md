# 🥉 Deep Past Machine Translation: Akkadian to English (Kaggle Bronze Medal)

**Global Rank:** 155th (Top 6%) | **Metric:** chrF++ (35.9 Official / 36.1 Experimental)

## 📌 Executive Summary
This repository contains my Bronze Medal-winning solution for translating 4,000-year-old Old Assyrian cuneiform texts into fluent English. The core architecture relies on an **Ensembled Minimum Bayes Risk (MBR) pipeline** utilizing byte-level Transformer models (`google/byt5-small` variants) paired with a RoBERTa-based Quality Estimation judge and a deterministic Retrieval-Augmented Generation (RAG) safety net.

## 🏗️ Architecture Overview
Because ByT5 operates at the byte-level rather than the word-level, traditional translation paradigms required heavy modification. 
1. **Candidate Generation:** A dual-model ensembled beam search generates pristine, deterministic translation paths.
2. **Quality Estimation (MBR):** Candidates are scored against each other using a weighted matrix of `chrF++`, `BLEU`, `Jaccard Similarity`, and fluency confidence from a fine-tuned RoBERTa CoLA model.
3. **Legal RAG Override:** An aggressive fuzzy-matching fallback against established academic datasets (Michel Corpus) to guarantee historical accuracy.

## ⚙️ Key Engineering Challenges Solved
*(This is where you prove your engineering grit. Mention the specific traps you bypassed.)*

* **Defeating "Alphabet Exhaustion":** Diagnosed a critical failure where ByT5's `repetition_penalty` forced the model to exhaust English bytes and hallucinate UTF-8 Chinese/Korean characters. Solved by overriding the penalty and shifting to absolute determinism (`do_sample=False`).
* **Compute / Memory Optimization (OOM Prevention):** Re-architected the generation pipeline to process 4,000+ hidden test rows within a strict 9-hour Kaggle compute limit. Achieved a 60% speedup by pruning the beam search tree, slashing `max_new_tokens` to 128, and optimizing batch sizes for T4 GPUs.
* **Hardware/Environment Cascades:** Successfully engineered a time-machine environment lock to bypass a mid-competition Kaggle PyTorch update that deprecated P100 GPU architecture support.

## 🧪 The 36.1 "High-Temperature" Experiment
While my selected official submission scored a highly generalized **35.9**, an unselected experimental notebook achieved a **36.1** on the Private Leaderboard. 
Located in `notebooks/ablation_experiment_36.1.ipynb`, this pipeline utilized a 3-model triumvirate and activated high-temperature sampling (`[0.60, 0.80, 1.05]`). This proved that while sampling introduces hallucination risk in byte-level models, a robust enough MBR judge can successfully filter out the noise and select highly creative, historically accurate translations.

## 🚀 How to Run
```bash
git clone [https://github.com/yourusername/Akkadian-ByT5-MBR.git](https://github.com/yourusername/Akkadian-ByT5-MBR.git)
pip install -r requirements.txt
python src/inference.py --config src/config.py
