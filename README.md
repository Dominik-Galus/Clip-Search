# CLIP Search Project

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.124.4-green)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)
![DVC](https://img.shields.io/badge/DVC-Data_Version_Control-purple)

This is an end-to-end Machine Learning system that allows searching through video collections using natural language queries (e.g., *"a person playing tennis"*).

It leverages **Contrastive Learning (CLIP)** to understand video content and **FAISS** for vector search. The system is fully containerized, accessible via a REST API, and includes a Streamlit frontend.

---

## Features

- Deep Learning Core: Custom PyTorch Lightning module wrapping OpenAI's CLIP (ViT-B/32) fine-tuned on the UCF101 dataset.
- Vector Search: FAISS (Facebook AI Similarity Search) index for efficient embedding retrieval.
- API First: High-performance FastAPI backend using asynccontextmanager for resource lifecycle management (loading heavy models once at startup).
- MLOps & Deployment:
  - Docker: Multi-stage builds optimized for CPU inference (separated API and UI containers).
  - CI/CD: GitHub Actions pipeline with ruff linting, pyrefly type checking, and Docker build verification.
  - DVC: Data Version Control used for tracking heavy artifacts (weights, datasets) and replicating the environment.
- Robustness: Traffic control using Locust.

## Tech Stack
ML & Data:
- PyTorch & PyTorch Lightning - Deep Learning framework and training loop abstraction.
- Transformers (Hugging Face) - CLIP model architecture and tokenization.
- FAISS (Facebook AI Similarity Search) - Vector database for dense retrieval.
- Albumentations & PyAV - Video processing and data augmentation pipeline.
- Hydra - Hierarchical configuration for training loop.

Backend & Frontend:
- FastAPI - Async REST API with Lifespan events.
- Streamlit - Rapid UI prototyping.
- Uvicorn - ASGI Server.

MLOps & DevOps:
- Docker & Docker Compose - Containerization and multi-service orchestration.
- GitHub Actions - CI pipeline (Linting, Type Checking, Build Verification).
- DVC (Data Version Control) - Artifact and dataset versioning.
- Locust - Load testing and performance benchmarking.
- Ruff & PyRefly - Code quality and static type analysis.
- uv - Blazing fast Python package and project manager.

## Installation
### Prerequisites
- Docker & Docker Compose
- Python 3.10+ (for local development)
- `uv` or the package manager of your choice (uv recommended)

### Quick Start with Docker
1. Clone the repository  (here is example with ssh):
```bash
git clone git@github.com:Dominik-Galus/Clip-Search.git
```
3. Pull data and Models (DVC):
```bash
# Currently it is only on my local machine but ideally it would be on some kind of cloud
dvc pull
```
3. Launch system:
```bash
docker-compose up --build
```
4. Access:
   - Frontend: http://localhost:8501
   - API Docs: http://localhost:8000/docs
  
### Local Development
If you want to run the code without Docker:
1. Install dependencies:
```bash
uv sync
```

2. Run API:
```bash
uv run uvicorn app:app --reload
```

3. Run UI:
```bash
uv run streamlit run ui.py
```

# UI

<img width="1618" height="991" alt="Screenshot 2025-12-14 at 21 10 38" src="https://github.com/user-attachments/assets/a9cb4430-3a52-457a-bf02-033559d123d8" />

## Performance & Load Testing
The system was verified using Locust (Grafana + Prometheus would be also good for monitoring)
- The system successfully handled up to 51.7 RPS on a standard CPU environment. This demonstrates that the decoupling of the heavy CLIP model and the lightweight FAISS index is effective choice.
- With an average response time of 638ms, the search experience remains under the 1-second threshold, ensuring a smooth user experience despite heavy matrix operations involved in the text embedding process.
- The primary bottleneck is the Text Encoder inference (CLIP) performed for every query. Future optimization could involve ONNX Runtime export or quantization (int8) to further reduce latency and increase throughput.
 
<img width="1488" height="900" alt="total_requests_per_second_1765742612 352" src="https://github.com/user-attachments/assets/b0ee8673-a6ff-46ba-8f3f-2627cf979d9c" />

## CI Pipeline
Project have CI running when pushing on main
### Stages
- *Lint:* Running ruff check and pyrefly check
- *Build-check:* Checking if the package and docker built correctly
