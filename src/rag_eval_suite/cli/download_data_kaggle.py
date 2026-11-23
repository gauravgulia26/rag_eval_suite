from pathlib import Path
from typing import Annotated

import kagglehub
import typer
from rich import print

from rag_eval_suite.config.paths import DATA_PATH

app = typer.Typer()


@app.command()
def download_data(
    dataset_name: Annotated[str, typer.Argument()],
    save_dir: Annotated[Path, typer.Argument()] = DATA_PATH,
):
    saved_path = kagglehub.dataset_download(handle=dataset_name, output_dir=save_dir)
    print(f"Data:{dataset_name} has been downloaded to:{saved_path}")


if __name__ == "__main__":
    app()
