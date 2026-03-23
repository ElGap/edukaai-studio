"""
Hardware Resource Monitor for Apple Silicon Macs

Monitors CPU, RAM, and MLX GPU memory usage during training.
"""

import os
import time
import subprocess
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class ResourceStats:
    """Container for resource statistics."""
    cpu_percent: float = 0.0
    cpu_cores_used: int = 0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    ram_percent: float = 0.0
    mlx_gpu_used_gb: float = 0.0  # Apple Silicon unified memory used by MLX
    mlx_gpu_total_gb: float = 0.0
    mlx_gpu_percent: float = 0.0
    timestamp: float = 0.0


class ResourceMonitor:
    """Monitors system resources on Apple Silicon Macs."""
    
    def __init__(self):
        self.stats = ResourceStats()
        self._get_system_info()
    
    def _get_system_info(self):
        """Get static system information."""
        try:
            # Get total RAM
            result = subprocess.run(['sysctl', '-n', 'hw.memsize'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                total_bytes = int(result.stdout.strip())
                self.stats.ram_total_gb = total_bytes / (1024**3)
            
            # Get CPU core count
            result = subprocess.run(['sysctl', '-n', 'hw.ncpu'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                self.cpu_cores = int(result.stdout.strip())
            else:
                self.cpu_cores = os.cpu_count() or 8
                
        except Exception as e:
            print(f"[RESOURCE] Error getting system info: {e}")
            self.stats.ram_total_gb = 16.0  # Default assumption
            self.cpu_cores = 8
    
    def get_stats(self) -> ResourceStats:
        """Get current resource statistics."""
        try:
            self.stats.timestamp = time.time()
            
            # Get CPU usage using top command (more accurate on macOS)
            result = subprocess.run(
                ['top', '-l', '1', '-n', '0', '-F'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                # Parse CPU usage from top output
                for line in result.stdout.split('\n'):
                    if 'CPU usage:' in line:
                        # Format: "CPU usage: 12.34% user, 5.67% sys, 81.99% idle"
                        parts = line.split(',')
                        for part in parts:
                            if 'user' in part or 'sys' in part:
                                percent_str = part.split('%')[0].strip()
                                try:
                                    self.stats.cpu_percent += float(percent_str)
                                except:
                                    pass
                        break
            
            # Get RAM usage using vm_stat
            result = subprocess.run(['vm_stat'], capture_output=True, text=True)
            if result.returncode == 0:
                # Parse vm_stat output
                pages_free = 0
                pages_active = 0
                pages_inactive = 0
                pages_wired = 0
                pages_compressed = 0
                
                for line in result.stdout.split('\n'):
                    if 'Pages free:' in line:
                        pages_free = int(line.split(':')[1].strip().rstrip('.'))
                    elif 'Pages active:' in line:
                        pages_active = int(line.split(':')[1].strip().rstrip('.'))
                    elif 'Pages inactive:' in line:
                        pages_inactive = int(line.split(':')[1].strip().rstrip('.'))
                    elif 'Pages wired down:' in line:
                        pages_wired = int(line.split(':')[1].strip().rstrip('.'))
                    elif 'Pages occupied by compressor:' in line:
                        pages_compressed = int(line.split(':')[1].strip().rstrip('.'))
                
                # Calculate used memory (page size is typically 4096 bytes on macOS)
                page_size = 4096
                used_pages = pages_active + pages_inactive + pages_wired + pages_compressed
                self.stats.ram_used_gb = (used_pages * page_size) / (1024**3)
                
                if self.stats.ram_total_gb > 0:
                    self.stats.ram_percent = (self.stats.ram_used_gb / self.stats.ram_total_gb) * 100
            
            # Get MLX GPU memory usage (via system_profiler or Activity Monitor)
            # This is approximate - full Metal memory tracking requires more complex approaches
            try:
                # Check if MLX is using GPU memory
                result = subprocess.run(
                    ['ps', '-o', 'rss=', '-p', str(os.getpid())],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    # This is the process memory, not strictly GPU
                    # On Apple Silicon, unified memory means CPU/GPU share RAM
                    process_mb = int(result.stdout.strip()) / 1024
                    self.stats.mlx_gpu_used_gb = process_mb / 1024
            except:
                pass
            
        except Exception as e:
            print(f"[RESOURCE] Error getting stats: {e}")
        
        return self.stats
    
    def format_stats(self) -> str:
        """Format stats for display in log."""
        self.get_stats()
        return (
            f"CPU: {self.stats.cpu_percent:5.1f}% | "
            f"RAM: {self.stats.ram_used_gb:.2f}/{self.stats.ram_total_gb:.0f}GB ({self.stats.ram_percent:.1f}%) | "
            f"MLX: {self.stats.mlx_gpu_used_gb:.2f}GB"
        )
    
    def get_summary(self) -> str:
        """Get a detailed summary for final report."""
        self.get_stats()
        return f"""Resource Usage Summary:
CPU Usage: {self.stats.cpu_percent:.1f}%
RAM Usage: {self.stats.ram_used_gb:.2f} GB / {self.stats.ram_total_gb:.0f} GB ({self.stats.ram_percent:.1f}%)
MLX GPU Memory: {self.stats.mlx_gpu_used_gb:.2f} GB
"""


if __name__ == "__main__":
    # Test the monitor
    monitor = ResourceMonitor()
    print("Testing ResourceMonitor...")
    print(monitor.format_stats())
    print()
    print(monitor.get_summary())
