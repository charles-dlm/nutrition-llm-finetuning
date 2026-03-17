# NutriLLM — Fine-tuned LLM for Human Nutrition Guidance

> A domain-specific language model trained on ANSES research publications to provide evidence-based recommendations on food ingredients and dietary practices.

## Important Notice

> **This fine-tuning is not yet functional.**
>
> The model currently suffers from **hallucinations** — it tends to generate plausible-sounding but factually incorrect responses. This is largely due to hardware constraints encountered during training (see [Limitations](#limitations-hardware--training-constraints) below).

## Project Overview

This project fine-tunes **TinyLlama-1.1B-Chat** on a curated corpus of 900+ scientific publications sourced from the French food safety agency [ANSES](https://www.anses.fr), with the goal of building a reliable assistant capable of answering questions about human nutrition based on peer-reviewed evidence.

The full pipeline covers:
- Automated scraping of the ANSES publication portal (paginated)
- PDF extraction and content parsing from each article
- Intelligent token-aware text chunking that preserves semantic boundaries
- Synthetic QA dataset construction from document content and metadata
- Parameter-Efficient Fine-Tuning (PEFT) using LoRA + 4-bit quantization

## Pipeline Description

### 1. Web Scraping
The ANSES publication portal is scraped across all pages (0–90+), filtering specifically for documents from the Human Nutrition expert committee. Each page is parsed with BeautifulSoup to extract article links and metadata.

### 2. PDF Extraction & Chunking
Each article URL is fetched. If it resolves to a PDF, `PyPDF2` extracts the raw text. The text is then split into chunks of approximately **200 tokens**, always respecting natural sentence and clause boundaries (`. ` and `;` delimiters) to preserve semantic coherence.

### 3. Dataset Construction
Each text chunk is wrapped into an instruction-following format:

```
<|system|>
Tu es un assistant expert en nutrition humaine.
<|user|>
Réponds uniquement à partir du contexte ci-dessous.

Contexte :
{chunk}

Question :
Que dit ce passage ?
<|assistant|>
{chunk}
```

### 4. Fine-Tuning (QLoRA)
- **Base model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Quantization**: BitsAndBytes 4-bit (NF4, double quantization)
- **LoRA targets**: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **LoRA rank**: r=16, alpha=32
- **Optimizer**: `paged_adamw_8bit`
- **Epochs**: 4 | **Batch size**: 8 | **Learning rate**: 2e-4

## Limitations: Hardware & Training Constraints

This project was developed under **significant hardware constraints**, which directly impacted the training quality and final model performance.

### GPU limitations
Training was performed on a **Tesla T4 GPU** via Google Colab Free tier. This GPU offers limited VRAM and compute throughput, making it very slow to train on the full corpus.

As a result:
- **Only ~5% of the available data was used for training**, as running the full dataset would have required prohibitively long training sessions on this hardware.
- **Certain hyperparameter choices may appear unconventional** (e.g., high batch size with gradient accumulation, relatively high learning rate), but these were dictated by the need to keep training sessions within Colab's session time limits — not by theoretical optimality.
- Due to these constraints, **model convergence cannot be guaranteed**. The model may not have trained long enough or on enough data to reliably produce accurate, grounded answers — which is likely a contributing factor to the hallucinations observed at inference time.

### What would be needed to fix this
- Access to a more powerful GPU (e.g., A100 or H100) or a multi-GPU setup
- Training on 100% of the scraped corpus (~917 documents)
- Longer training runs with proper convergence monitoring (loss curves, validation perplexity)
- Possibly a larger base model or a retrieval-augmented generation (RAG) approach as an alternative to pure fine-tuning

## Getting Started

### Prerequisites

```bash
pip install transformers datasets accelerate peft bitsandbytes
pip install requests beautifulsoup4 PyPDF2 tiktoken langchain-text-splitters
```

### Run the scraper

```bash
python scraper.py
# Output: articles_anses.json
```

### Run fine-tuning

```bash
python train.py
# Output: ./tinyllama-anses-lora/
```

### Run inference

```python
from inference import generate

response = generate("Quels sont les effets du fenugrec sur la glycémie ?")
print(response)
```

## Dependencies

| Library | Role |
|---|---|
| `transformers` | Model loading, tokenization, training |
| `peft` | LoRA adapter injection |
| `bitsandbytes` | 4-bit quantization |
| `datasets` | Dataset formatting and splitting |
| `PyPDF2` | PDF text extraction |
| `BeautifulSoup4` | HTML parsing for web scraping |
| `tiktoken` | Token counting for chunking |
| `langchain-text-splitters` | Utility text splitting |

## Model Card

| Parameter | Value |
|---|---|
| Base model | TinyLlama-1.1B-Chat-v1.0 |
| Fine-tuning method | QLoRA (LoRA + 4-bit quantization) |
| Training data | ~5% of 917 ANSES documents |
| Domain | Human nutrition (French) |
| Language | French |
| Status | Experimental — hallucinations present |

## Data Source

All training data is sourced from publicly available scientific opinions and reports published by [ANSES](https://www.anses.fr) (Agence nationale de sécurité sanitaire de l'alimentation, de l'environnement et du travail), specifically from the **Human Nutrition Expert Committee** section.

## License

This project is for educational and research purposes only. The scraped content belongs to ANSES and is subject to their terms of use.
