import logging
import os

import pydantic_settings as ps
import rich.logging
import rich.prompt

from crepa import evaluate, settings

logger = logging.getLogger(__name__)


class Eval(settings.Eval):
    """Evaluates model on stuff."""

    def cli_cmd(self) -> None:
        evaluate.evaluate(self)


class Command(
    settings.Base,
    cli_parse_args=True,
    cli_use_class_docs_for_groups=True,
    cli_kebab_case=True,
):
    """CLI for evaluating JEPA-style models on input corruption benchmarks."""

    eval: ps.CliSubCommand[Eval]

    def cli_cmd(self) -> None:
        class DistRankFilter(logging.Filter):
            """Only allows logs from rank 0 (master process) in DDP training."""

            def filter(self, record: logging.LogRecord) -> bool:  # noqa: ARG002
                rank = os.getenv("RANK") or os.getenv("LOCAL_RANK")
                if rank is None:
                    return True
                return int(rank) == 0

        logging.basicConfig(
            level=logging.DEBUG if self.verbose else logging.INFO,
            format="%(message)s",
            handlers=[rich.logging.RichHandler(rich_tracebacks=True)],
        )
        for handler in logging.getLogger().handlers:
            handler.addFilter(DistRankFilter())

        logger.debug("running with settings %s", self)
        ps.CliApp.run_subcommand(self)


def main() -> None:
    ps.CliApp.run(Command)


if __name__ == "__main__":
    main()
