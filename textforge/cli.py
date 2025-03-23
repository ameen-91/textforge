import typer
from typing import List
from rich.console import Console
from rich.table import Table
from textforge.manager import ModelManager

app = typer.Typer(help="TextForge CLI - Text Classification Pipeline")
model_app = typer.Typer(help="Model management commands")
app.add_typer(model_app, name="models")

console = Console()

def parse_metadata(metadata: List[str]) -> dict:
    """Parse key=value metadata pairs into a dictionary."""
    result = {}
    for item in metadata:
        try:
            key, value = item.split('=')
            result[key.strip()] = value.strip()
        except ValueError:
            typer.echo(f"Invalid metadata format: {item}. Use key=value format.")
    return result

@app.command("run")
def serve_model(model_id: str) -> None:
    """Serve a model using the TextForge API."""
    manager = ModelManager()
    try:
        manager.serve_model(model_id)
    except ValueError as e:
        console.print(f"[red]Error:[/] {e}")



@model_app.command("list")
def list_models(
    format: str = typer.Option("table", "--format", "-f", help="Output format (table/json)")
) -> None:
    """List all registered models."""
    manager = ModelManager()
    models = manager.list_models()
    
    if format == "json":
        import json
        console.print_json(json.dumps(models))
        return
        
    table = Table(show_header=True)
    table.add_column("Model ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Version", style="yellow")
    table.add_column("Created At", style="blue")
    table.add_column("Path", style="magenta")
    
    for model_id, info in models.items():
        table.add_row(
            model_id,
            info['model_name'],
            info['version'],
            info['created_at'],
            info['path']
        )
    
    console.print(table)

@model_app.command()
def info(model_id: str) -> None:
    """Show detailed information about a model."""
    manager = ModelManager()
    try:
        model_info = manager.load_model(model_id)
        console.print_json(data=model_info)
    except ValueError as e:
        console.print(f"[red]Error:[/] {e}")

@model_app.command()
def delete(
    model_id: str,
    force: bool = typer.Option(False, '--force', '-f', help="Force delete without confirmation")
) -> None:
    """Delete a model from the registry."""
    if not force:
        confirm = typer.confirm(f"Are you sure you want to delete model {model_id}?")
        if not confirm:
            return
    
    manager = ModelManager()
    try:
        manager.delete_model(model_id)
        console.print(f"[green]✓[/] Model {model_id} deleted successfully")
    except ValueError as e:
        console.print(f"[red]Error:[/] {e}")

def main():
    app()

if __name__ == "__main__":
    main()