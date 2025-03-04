import typer
import os
import pkg_resources
import psutil
import re
import subprocess
import time
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    BarColumn,
)

console = Console()


def extract_label_value(text, key="label"):
    pattern = rf"'{key}'\s*:\s*'([^']+)'"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None


def print_success(message: str):
    console.print(f"[green]{message}[/green]")


def print_error(message: str):
    console.print(f"[red]{message}[/red]")


def print_success_bold(message: str):
    console.print(f"[green bold]{message}[/green bold]")


def print_neutral(message: str):
    console.print(f"[blue]{message}[/blue]")


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
    """Install Ollama and pull specified model with progress tracking.

    Args:
        model (str, optional): Name of Model. Defaults to "llama3.1:8b-instruct-q4_0".
    """
    steps = [
        ("Updating system packages", "apt-get update && apt-get upgrade -y"),
        ("Installing dependencies", "apt-get install lshw"),
        ("Installing Ollama", "curl https://ollama.ai/install.sh | sh"),
        ("Starting Ollama server", "ollama serve"),
        (f"Pulling model {model}", f"ollama pull {model}"),
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
    ) as progress:
        overall_task = progress.add_task(
            "[bright_cyan]Installation progress...", total=len(steps)
        )

        for i, (description, command) in enumerate(steps):
            task = progress.add_task(f"[yellow]{description}...", total=1)

            try:
                if description == "Starting Ollama server":
                    serve_process = subprocess.Popen(
                        command,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    time.sleep(3)
                else:
                    result = subprocess.run(
                        command,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=True,
                    )

                progress.update(task, completed=1)
                progress.update(overall_task, advance=1)

            except subprocess.CalledProcessError as e:
                progress.stop()
                console.print(f"[red]Error during {description}: {e}[/red]")

            progress.remove_task(task)

    console.print("[green bold]OLLAMA installed successfully![/green bold]")


def start_ollama():
    command = "ollama serve"
    try:
        serve_process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(3)
        console.print("[green bold]OLLAMA server started successfully![/green bold]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error starting OLLAMA server: {e}[/red]")
