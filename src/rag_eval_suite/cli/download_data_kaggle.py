from typing import Annotated

import kagglehub
import typer
from rich import print

app = typer.Typer()


@app.command()
def download_data(
    dataset_name: Annotated[str, typer.Argument()],
    save_dir: Annotated[str, typer.Argument()] = "data",
):
    saved_path = kagglehub.dataset_download(handle=dataset_name, output_dir=save_dir)
    print(f"Data:{dataset_name} has been downloaded to:{saved_path}")


if __name__ == "__main__":
    app()
