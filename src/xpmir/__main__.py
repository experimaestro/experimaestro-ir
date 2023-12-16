import click
from experimaestro.experiments.cli import experiments_cli


@click.group()
def cli():
    pass


# Add some commands
cli.add_command(experiments_cli, "run-experiment")


@cli.group()
def huggingface():
    pass


@click.argument("hf_id", type=str)
@huggingface.command()
def preload(hf_id: str):
    """Pre-load the HuggingFace model using AutoModel and AutoTokenizer"""
    from transformers import AutoModel, AutoTokenizer

    AutoModel.from_pretrained(hf_id)
    AutoTokenizer.from_pretrained(hf_id)


def main():
    cli(obj=None)


if __name__ == "__main__":
    main()
