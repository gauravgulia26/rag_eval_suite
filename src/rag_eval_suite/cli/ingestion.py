import asyncio
from pathlib import Path
from typing import Annotated

import typer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich import print
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from rag_eval_suite.application import IngestionPipeline
from rag_eval_suite.config.paths import PARAM_FILE_PATH, RAW_DATA_PATH
from rag_eval_suite.infra.chunkers import Chunker
from rag_eval_suite.infra.loaders import Loader
from rag_eval_suite.infra.vectorstores import MilvusStore
from rag_eval_suite.utils.loaders import load_yaml

app = typer.Typer(help="Ingest a document into the configured Milvus collection.")


def get_pipeline(
    document_path: Path,
    collection_name: str,
) -> IngestionPipeline:
    from rag_eval_suite.infra.embeddings import get_openai_embeddings

    params = load_yaml(PARAM_FILE_PATH)
    splitter_params = params["splitter"]

    loader_obj = Loader(document_path=document_path)
    chunker_obj = Chunker(
        splitter=RecursiveCharacterTextSplitter(
            chunk_size=splitter_params["chunk_size"],
            chunk_overlap=splitter_params["chunk_overlap"],
        )
    )
    db_store = MilvusStore(
        embedding_function=get_openai_embeddings(),
        collection_name=collection_name,
        param_file_path=PARAM_FILE_PATH,
        param_file_key="milvus_db",
    )
    return IngestionPipeline(
        loader=loader_obj,
        chunker=chunker_obj,
        vector_store=db_store.run(),
    )


@app.command()
def run_ingestion(
    collection_name: Annotated[
        str,
        typer.Option(help="Milvus collection name."),
    ] = "EvaluationRagCollection",
):
    """Load the first configured PDFs and ingest them into Milvus."""
    params = load_yaml(PARAM_FILE_PATH)
    ingestion_params = params["ingestion"]
    document_limit = ingestion_params["document_limit"]
    batch_size = ingestion_params["batch_size"]

    document_paths = sorted(RAW_DATA_PATH.glob("*.pdf"))[:document_limit]
    if len(document_paths) < document_limit:
        raise typer.BadParameter(
            f"Expected at least {document_limit} PDF documents in {RAW_DATA_PATH}, "
            f"found {len(document_paths)}"
        )

    pipeline = get_pipeline(document_paths[0], collection_name)

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
    ) as progress:
        tasks = {
            stage: progress.add_task(description, total=0)
            for stage, description in (
                ("loading", "Loading documents"),
                ("chunking", "Chunking documents"),
                ("ingesting", "Ingesting batches"),
            )
        }

        def update_progress(stage: str, completed: int, total: int) -> None:
            progress.update(tasks[stage], total=total, completed=completed)

        result = asyncio.run(
            pipeline.run_async(
                document_paths=document_paths,
                batch_size=batch_size,
                progress_callback=update_progress,
            )
        )

    print(
        "[green]Ingestion completed[/green] — "
        f"documents={result['total_documents']}, "
        f"chunks={result['total_chunks']}, "
        f"inserted={len(result['inserted_ids'])}"
    )
