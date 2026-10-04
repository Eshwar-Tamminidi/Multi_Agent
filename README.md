# Intelligent Multi-Modal Enterprise AI System

An end-to-end multi-modal artificial intelligence system integrating **10 deep learning and large language models** for enterprise knowledge retrieval, conversational reasoning, speech recognition, object detection, image understanding, sentiment analysis, anomaly detection, semantic search, and symbolic music generation.

Enhanced with a **Liquid Glass** user interface featuring frosted glassmorphism, dynamic glowing ambient mesh orbs, specular highlights, and seamless **Light / Dark theme** switching.

---

## 🏛️ Model Portfolio & Architecture Overview

| # | Model | Architecture Family | Task & Domain | Primary Metric | Compute / Footprint |
|---|---|---|---|---|---|
| **1** | **Character-Level GPT** | Decoder-Only Transformer | Autoregressive text generation | Val Perplexity | CPU/MPS · ~1.8M params |
| **2** | **RAG + Sentence-BERT** | Bi-Encoder + Cross-Entropy | Enterprise knowledge retrieval | QA Accuracy & MRR | CPU · 22M-param MiniLM |
| **3** | **Qwen 2.5 / Llama 3.2 + LoRA** | Instruction LLM + PEFT | Enterprise chat & policy Q&A | Tokens/sec & QA Acc | MPS/CUDA · 1.5B params |
| **4** | **Whisper Small/Base** | Seq2Seq Audio Transformer | Speech recognition & timestamps | Word Error Rate (WER) | CPU/MPS · ~74M params |
| **5** | **YOLOv8** | Single-Stage CNN Detector | Real-time object detection | Count F1 & FPS | CPU/MPS · ~3.2M params |
| **6** | **BERT** | Bidirectional Encoder | Customer sentiment & explainability | Macro F1 & Star Acc | CPU/MPS · ~167M params |
| **7** | **CLIP + Transformer Decoder** | ViT + Prefix Transformer Decoder | Image captioning & zero-shot tags | CLIP Score & Top-5 Acc | CPU/MPS · ~154M params |
| **8** | **LSTM Autoencoder** | Recurrent Seq2Seq Autoencoder | Time-series telemetry anomaly detection | Precision, Recall, F1, ROC-AUC | CPU · ~119K params |
| **9** | **Two-Tower Semantic Search** | Contrastive Dual-Encoder (InfoNCE) | Enterprise service catalog retrieval | Recall@3 & MRR vs TF-IDF | CPU · ~23M params |
| **10** | **Tiny Music Transformer** | Symbolic Event Transformer | MIDI melody composition | In-scale ratio & Entropy | CPU/MPS · ~0.8M params |

---

## 💎 Liquid Glass User Interface Features

- **Liquid Glass Aesthetics**: Ultra-clean frosted glass cards (`backdrop-filter: blur(20px) saturate(190%)`), dynamic specular border sheen, soft inset glows, fluid pill badges.
- **Light & Dark Modes**: Pearlescent frosted glass with subtle pastel glowing meshes in Light Mode; Obsidian glass with glowing neon cyan/purple/emerald orbs in Dark Mode.
- **Comparative Evaluation Matrix**: Side-by-side dashboard benchmarking all 10 models on accuracy, response latency, inference time, parameter scales, and memory footprint.
- **In-Browser Web Audio Synthesizer**: Instant playback of generated MIDI notes directly in the browser via Web Audio API with an animated piano roll visualizer.
- **Interactive YOLOv8 Canvas**: Real-time object detection overlay with normalized bounding boxes, confidence tags, and class counts.
- **BERT Occlusion Heat Map**: Token-level sensitivity explanation revealing exactly which words drove sentiment polarity.
- **Live Memory Management**: LRU residency eviction to maintain peak performance on machines with limited RAM (e.g., 8 GB MacBooks).

---

## 🚀 Quick Start Guide

### 1. Backend Setup
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 2. Frontend Setup (Development)
```bash
cd frontend
npm install
npm run dev
```
Open **`http://localhost:3000`** in your browser. (Alternatively, the production build is also served directly by FastAPI at **`http://localhost:8000`**).

---

## 🧪 Training & Benchmarking Scripts

### Train LoRA Adapter on Enterprise Data
```bash
cd backend
source .venv/bin/activate
python scripts/train_lora.py --steps 50 --lr 2e-4
```

### Sequential Benchmark of All 10 Models
```bash
cd backend
source .venv/bin/activate
python scripts/benchmark_all.py
```
