import json
import os
import subprocess
from typing import List, Dict, Any


class CLIBridge:
    """Wrapper around the trader CLI."""

    def __init__(self, executable: str | None = None) -> None:
        if executable is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            executable = os.path.join(repo_root, 'build', 'src', 'cli', 'trader-cli')
        self.executable = executable

    def run(self, args: List[str]) -> str:
        """Execute trader-cli with provided arguments.

        Parameters
        ----------
        args: List[str]
            Arguments to pass to trader-cli.

        Returns
        -------
        str
            JSON string containing stdout, stderr, and exit status.
        """
        result = subprocess.run([self.executable, *args], capture_output=True, text=True)
        output: Dict[str, Any] = {
            'stdout': result.stdout,
            'stderr': result.stderr,
            'status': result.returncode,
        }
        return json.dumps(output)
