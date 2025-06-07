#!/usr/bin/env python3
"""
Managers module - Configuration and project management
"""

from .config_manager import ConfigManager, ConfigError
from .project_manager import ProjectManager

__all__ = [
    'ConfigManager',
    'ConfigError', 
    'ProjectManager'
]