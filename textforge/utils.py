import typer
import os
import pkg_resources
import psutil
import re
import subprocess
import time
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn


def extract_label_value(text, key="label"):
    pattern = rf"'{key}'\s*:\s*'([^']+)'"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None


def print_success(message: str):
    typer.echo(typer.style(message, fg=typer.colors.GREEN))


def print_error(message: str):
    typer.echo(typer.style(message, fg=typer.colors.RED))


def print_success_bold(message: str):
    typer.echo(typer.style(message, fg=typer.colors.GREEN, bold=True))


def print_neutral(message: str):
    typer.echo(typer.style(message, fg=typer.colors.BLUE))


def sanitize_model_name(model: str):
    return model.replace("/", "_")


def unsanitize_model_name(model: str):
    return model.replace("_", "/")


def get_package_dir() -> str:
    return pkg_resources.resource_filename("textforge", "")


def get_models_dir():
    os.makedirs(os.path.join(get_package_dir(), "data", "models"), exist_ok=True)
    return os.path.join(get_package_dir(), "data", "models")


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def install_ollama(model="llama3.1:8b-instruct-q4_0"):
    """Install Ollama and pull specified model.

    Args:
        model (str, optional): Name of Model. Defaults to "llama3.1:8b-instruct-q4_0".
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
    ) as progress:

        task1 = progress.add_task("[yellow]Updating system packages...", total=None)
        subprocess.run(
            "apt-get update && apt-get upgrade -y",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        progress.update(task1, completed=True)

        task2 = progress.add_task("[yellow]Installing dependencies...", total=None)
        subprocess.run(
            ["apt-get install lshw"],
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        progress.update(task2, completed=True)

        task3 = progress.add_task("[yellow]Installing Ollama...", total=None)
        subprocess.run(
            ["curl https://ollama.ai/install.sh | sh"],
            shell=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        progress.update(task3, completed=True)

        task4 = progress.add_task("[yellow]Starting Ollama server...", total=None)
        serve_process = subprocess.Popen(
            ["ollama serve"], shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE
        )
        time.sleep(3)
        progress.update(task4, completed=True)

        task5 = progress.add_task("[yellow]Pulling model...", total=None)
        subprocess.run(
            [f"ollama pull {model}"],
            shell=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        progress.update(task5, completed=True)
    print_success_bold("OLLAMA installed successfully!")
