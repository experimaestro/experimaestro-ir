import click
from xpmir.papers.cli import papers_cli


@click.group()
def cli():
    pass


# Add paper comand
cli.add_command(papers_cli, "papers")


def main():
    cli(obj=None)


if __name__ == "__main__":
    main()
