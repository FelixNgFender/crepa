import trackio


class Tracker:
    def __init__(
        self,
        project: str,
        name: str | None = None,
        config: dict | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        self.project = project
        self.name = name
        self.config = config
        self.enabled = enabled

    def init(self) -> None:
        """Light wrapper around `trackio.init` to adjust enable behavior for our use case."""
        if not self.enabled:
            return

        trackio.init(
            project=self.project,
            name=self.name,
            config=self.config,
        )

    def log(self, metrics: dict, step: int | None = None) -> None:
        if not self.enabled:
            return
        trackio.log(metrics, step=step)
