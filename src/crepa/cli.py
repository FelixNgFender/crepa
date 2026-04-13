import logging

import pydantic_settings as ps
import rich.logging
import rich.prompt

from crepa import settings

logger = logging.getLogger(__name__)


class Eval(settings.Eval):
    """Evaluates model on stuff."""

    def cli_cmd(self) -> None: ...


class Command(
    settings.Base,
    cli_parse_args=True,
    cli_use_class_docs_for_groups=True,
    cli_kebab_case=True,
):
    """CLI for evaluating JEPA-style models on input corruption benchmarks."""

    eval: ps.CliSubCommand[Eval]

    def cli_cmd(self) -> None:
        logging.basicConfig(
            level=logging.DEBUG if self.verbose else logging.INFO,
            format="%(message)s",
            handlers=[rich.logging.RichHandler(rich_tracebacks=True)],
        )
        logger.debug("running with settings %s", self)
        ps.CliApp.run_subcommand(self)


def main() -> None:
    ps.CliApp.run(Command)


if __name__ == "__main__":
    main()
