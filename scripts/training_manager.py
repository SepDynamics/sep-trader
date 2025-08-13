#!/usr/bin/env python3
"""High-level training orchestration for SEP Trader-Bot."""

import json
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingManager:
    """Coordinates data preparation, model training and evaluation."""

    def __init__(self, config_path: str = "config/training_config.json") -> None:
        self.config_path = Path(config_path)
        self.config: dict[str, str] = {}
        if self.config_path.exists():
            self.config = json.loads(self.config_path.read_text())
            logger.info("Loaded training config from %s", self.config_path)
        else:
            logger.warning("Training config %s not found", self.config_path)

    def _run(self, cmd: str) -> None:
        logger.info("Executing: %s", cmd)
        subprocess.run(cmd, shell=True, check=True)

    def prepare_data(self) -> None:
        script = self.config.get("data_script")
        if script:
            self._run(script)

    def train_model(self) -> None:
        cmd = self.config.get("train_cmd")
        if cmd:
            self._run(cmd)

    def evaluate(self) -> None:
        cmd = self.config.get("eval_cmd")
        if cmd:
            self._run(cmd)

    def run(self) -> None:
        """Execute full training pipeline."""
        self.prepare_data()
        self.train_model()
        self.evaluate()


if __name__ == "__main__":
    TrainingManager().run()
