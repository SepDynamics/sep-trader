#!/usr/bin/env python3
"""
SEP Professional Trading System - CLI Bridge
Provides secure bridge between web API and CLI commands
"""

import os
import sys
import json
import logging
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import queue

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CommandStatus(Enum):
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class CommandResult:
    """Result of CLI command execution"""
    command: str
    status: CommandStatus
    stdout: str = ""
    stderr: str = ""
    return_code: int = -1
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    error_message: str = ""

class CLIBridge:
    """Bridge for executing CLI commands safely from web interface"""
    
    def __init__(self):
        self.sep_root = Path(__file__).parent.parent
        self.cli_executable = self.sep_root / "bin" / "quantum_tracker"
        
        # Command whitelist for security
        self.allowed_commands = {
            'status', 'version', 'pairs', 'list-pairs', 'enable-pair', 'disable-pair',
            'train', 'analyze', 'monitor', 'cache-clear', 'cache-status', 
            'config', 'help', 'validate', 'test-connection'
        }
        
        # Command queue for async execution
        self.command_queue = queue.Queue()
        self.results = {}
        self.running_commands = {}
        
        # Start background worker
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        
        logger.info("CLI Bridge initialized")

    def execute_command(self, command: str, args: List[str] = None, timeout: int = 60) -> CommandResult:
        """Execute a CLI command synchronously"""
        cmd_parts = command.split() if isinstance(command, str) else [command]
        base_command = cmd_parts[0] if cmd_parts else ""
        
        # Validate command
        if not self._is_command_allowed(base_command):
            return CommandResult(
                command=command,
                status=CommandStatus.FAILED,
                error_message=f"Command '{base_command}' not allowed. Allowed: {', '.join(self.allowed_commands)}"
            )
        
        # Prepare command
        full_cmd = [str(self.cli_executable)] + cmd_parts
        if args:
            full_cmd.extend(args)
        
        result = CommandResult(
            command=" ".join(full_cmd),
            status=CommandStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            logger.info(f"Executing command: {result.command}")
            
            # Execute command
            process = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.sep_root
            )
            
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
            result.return_code = process.returncode
            result.stdout = process.stdout
            result.stderr = process.stderr
            
            if process.returncode == 0:
                result.status = CommandStatus.COMPLETED
            else:
                result.status = CommandStatus.FAILED
                result.error_message = f"Command failed with return code {process.returncode}"
            
        except subprocess.TimeoutExpired:
            result.end_time = datetime.now()
            result.duration = timeout
            result.status = CommandStatus.TIMEOUT
            result.error_message = f"Command timed out after {timeout} seconds"
            
        except Exception as e:
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds() if result.end_time else 0
            result.status = CommandStatus.FAILED
            result.error_message = str(e)
        
        logger.info(f"Command completed: {result.status.value} in {result.duration:.2f}s")
        return result

    def execute_command_async(self, command: str, args: List[str] = None, timeout: int = 60) -> str:
        """Execute a CLI command asynchronously, returns job ID"""
        job_id = f"job_{int(time.time() * 1000)}"
        
        job = {
            'id': job_id,
            'command': command,
            'args': args or [],
            'timeout': timeout,
            'created': datetime.now()
        }
        
        self.command_queue.put(job)
        logger.info(f"Queued async command: {job_id}")
        return job_id

    def get_command_result(self, job_id: str) -> Optional[CommandResult]:
        """Get result of async command"""
        return self.results.get(job_id)

    def get_running_commands(self) -> Dict[str, Any]:
        """Get list of currently running commands"""
        return {
            job_id: {
                'command': info['command'],
                'started': info['started'].isoformat(),
                'duration': (datetime.now() - info['started']).total_seconds()
            }
            for job_id, info in self.running_commands.items()
        }

    def cancel_command(self, job_id: str) -> bool:
        """Cancel a running command (if possible)"""
        if job_id in self.running_commands:
            # Note: This is a simplified implementation
            # In practice, you'd need to track process IDs and send signals
            logger.warning(f"Cancel requested for {job_id} - not fully implemented")
            return False
        return False

    def get_available_commands(self) -> Dict[str, str]:
        """Get list of available commands with descriptions"""
        command_descriptions = {
            'status': 'Get system status and health',
            'version': 'Display version information',
            'pairs': 'List all available trading pairs',
            'list-pairs': 'List enabled trading pairs',
            'enable-pair': 'Enable trading for a specific pair',
            'disable-pair': 'Disable trading for a specific pair',
            'train': 'Train the quantum model on historical data',
            'analyze': 'Analyze current market conditions',
            'monitor': 'Monitor trading activity',
            'cache-clear': 'Clear system cache',
            'cache-status': 'Show cache status',
            'config': 'Show configuration',
            'help': 'Display help information',
            'validate': 'Validate system configuration',
            'test-connection': 'Test API connections'
        }
        
        return {cmd: command_descriptions.get(cmd, 'No description') 
                for cmd in self.allowed_commands}

    def _is_command_allowed(self, command: str) -> bool:
        """Check if command is in whitelist"""
        return command in self.allowed_commands

    def _worker(self):
        """Background worker for async command execution"""
        while True:
            try:
                job = self.command_queue.get(timeout=1)
                
                job_id = job['id']
                self.running_commands[job_id] = {
                    'command': job['command'],
                    'started': datetime.now()
                }
                
                # Execute command
                result = self.execute_command(
                    job['command'], 
                    job['args'], 
                    job['timeout']
                )
                
                # Store result
                self.results[job_id] = result
                
                # Clean up
                if job_id in self.running_commands:
                    del self.running_commands[job_id]
                
                # Mark task done
                self.command_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")

    def to_dict(self, result: CommandResult) -> Dict[str, Any]:
        """Convert CommandResult to dictionary for JSON serialization"""
        return {
            'command': result.command,
            'status': result.status.value,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'return_code': result.return_code,
            'start_time': result.start_time.isoformat() if result.start_time else None,
            'end_time': result.end_time.isoformat() if result.end_time else None,
            'duration': result.duration,
            'error_message': result.error_message
        }

def main():
    """CLI Bridge test interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SEP CLI Bridge')
    parser.add_argument('command', help='Command to execute')
    parser.add_argument('args', nargs='*', help='Command arguments')
    parser.add_argument('--timeout', type=int, default=60, help='Timeout in seconds')
    parser.add_argument('--async', action='store_true', dest='async_exec', help='Execute asynchronously')
    
    args = parser.parse_args()
    
    bridge = CLIBridge()
    
    if args.async_exec:
        job_id = bridge.execute_command_async(args.command, args.args, args.timeout)
        print(f"Job ID: {job_id}")
        
        # Wait for result
        while True:
            result = bridge.get_command_result(job_id)
            if result:
                print(json.dumps(bridge.to_dict(result), indent=2))
                break
            time.sleep(0.1)
    else:
        result = bridge.execute_command(args.command, args.args, args.timeout)
        print(json.dumps(bridge.to_dict(result), indent=2))

if __name__ == '__main__':
    main()