import click
from xpmir.papers.cli import papers_cli
from xpmir.experiments.cli import experiments_cli


@click.group()
def cli():
    pass


# Add some commands
cli.add_command(papers_cli, "papers")
cli.add_command(experiments_cli, "run-experiment")


def main():
    cli(obj=None)


if __name__ == "__main__":
    main()
