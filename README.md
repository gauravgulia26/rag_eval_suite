<div align="center">

# RAG Eval Suite

**A practical evaluation workspace for retrieval-augmented generation systems.**

Built with LangChain, Milvus, OpenAI embeddings, and [DeepEval](https://deepeval.com/).

</div>

## What it does

RAG Eval Suite provides a repeatable path from source documents to evaluation:

```text
Documents → PDF loading → Recursive chunking → OpenAI embeddings → Milvus → DeepEval
```

The project is designed to compare retrieval and generation quality with measurable, reproducible evaluation metrics.

## Stack

| Layer | Tooling |
| --- | --- |
| Ingestion | PyMuPDF + LangChain |
| Chunking | Recursive character splitter |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | Zilliz / Milvus |
| Evaluation | DeepEval |
| Project tooling | Python 3.14 + `uv` |

## Quick start

Install dependencies:

```bash
uv sync
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-openai-key
ZILLIZ_ENDPOINT=your-zilliz-endpoint
ZILLIZ_APIKEY=your-zilliz-api-key
```

Run ingestion for the first three PDFs in `data/raw`:

```bash
uv run ingest-data
```

The ingestion limit, batch size, chunk size, overlap, and Milvus settings are configured in [`config/params.yaml`](config/params.yaml).

## Evaluation

Use the evaluation notebook to inspect the RAG workflow and extend it with DeepEval metrics, test cases, and experiment tracking:

```bash
uv run jupyter notebook notebooks/1-Testing.ipynb
```

Typical evaluation dimensions include answer relevancy, faithfulness, contextual relevancy, and retrieval quality.

## Useful commands

```bash
# Download a Kaggle dataset
uv run download-data <owner/dataset>

# Run ingestion with the configured defaults
uv run ingest-data
```

## Project layout

```text
config/                 Runtime parameters
data/raw/               Source documents
notebooks/              Exploration and evaluation workflows
src/rag_eval_suite/
├── application/        Pipelines and orchestration
├── cli/                Command-line entrypoints
├── infra/              Loaders, chunkers, embeddings, vector stores
├── contracts/          Typed pipeline results
└── utils/              Shared helpers
```

## Status

The ingestion pipeline is operational. Evaluation workflows are being expanded into a complete DeepEval suite for systematic RAG benchmarking.
