"""
environment.py - Trading Environment Manager

Manages trading environment configuration including mode selection (paper/testnet/live)
and API credentials loading. Provides safe environment switching with validation.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum
import logging

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from core.state import Environment


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EnvironmentConfig:
    """
    Configuration for a specific trading environment.
    
    Attributes:
        mode: Environment mode (paper/testnet/live)
        api_key: API key for the environment
        api_secret: API secret for the environment
        base_url: Base URL for API endpoint
        websocket_url: WebSocket URL for real-time data
        additional_config: Additional configuration parameters
    """
    mode: Environment
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    base_url: Optional[str] = None
    websocket_url: Optional[str] = None
    additional_config: Optional[Dict[str, str]] = None
    
    def is_valid(self) -> bool:
        """
        Check if configuration has required credentials.
        
        Returns:
            True if api_key and api_secret are present
        """
        return bool(self.api_key and self.api_secret)
    
    def __repr__(self) -> str:
        """Safe string representation (hides secrets)."""
        return (f"EnvironmentConfig(mode={self.mode.value}, "
                f"api_key={'***' if self.api_key else None}, "
                f"api_secret={'***' if self.api_secret else None}, "
                f"base_url={self.base_url})")


class EnvironmentManager:
    """
    Manages trading environment configuration and safe mode switching.
    
    Handles loading API credentials from .env files and provides validated
    environment configuration for different trading modes.
    """
    
    # Expected environment variable prefixes for each mode
    ENV_VAR_PREFIXES = {
        Environment.PAPER: "PAPER_",
        Environment.TESTNET: "TESTNET_",
        Environment.LIVE: "LIVE_"
    }
    
    def __init__(self, env_file: Optional[str] = None, auto_load: bool = True):
        """
        Initialize environment manager.
        
        Args:
            env_file: Path to .env file (default: .env in current directory)
            auto_load: Automatically load environment variables on init
        """
        self._env_file = env_file or ".env"
        self._current_environment: Optional[Environment] = None
        self._configs: Dict[Environment, EnvironmentConfig] = {}
        self._env_loaded = False
        
        if auto_load:
            self.load_environment_variables()
    
    def load_environment_variables(self, env_file: Optional[str] = None) -> bool:
        """
        Load environment variables from .env file.
        
        Args:
            env_file: Path to .env file (overrides init value)
            
        Returns:
            True if file was loaded successfully, False otherwise
        """
        target_file = env_file or self._env_file
        
        if not Path(target_file).exists():
            logger.warning(f"Environment file not found: {target_file}")
            return False
        
        if load_dotenv is None:
            logger.error("python-dotenv not installed. Install with: pip install python-dotenv")
            return False
        
        try:
            load_dotenv(target_file, override=True)
            logger.info(f"Loaded environment variables from: {target_file}")
            self._env_loaded = True
            
            # Load configurations for all environments
            self._load_all_configs()
            return True
            
        except Exception as e:
            logger.error(f"Failed to load environment file: {e}")
            return False
    
    def _load_all_configs(self) -> None:
        """Load configurations for all environment modes."""
        for env_mode in Environment:
            config = self._load_config_for_mode(env_mode)
            self._configs[env_mode] = config
            
            if config.is_valid():
                logger.info(f"Loaded valid configuration for {env_mode.value}")
            else:
                logger.warning(f"Incomplete configuration for {env_mode.value}")
    
    def _load_config_for_mode(self, mode: Environment) -> EnvironmentConfig:
        """
        Load configuration for specific environment mode.
        
        Args:
            mode: Environment mode to load config for
            
        Returns:
            EnvironmentConfig with loaded values
        """
        prefix = self.ENV_VAR_PREFIXES[mode]
        
        # Load standard configuration
        api_key = os.getenv(f"{prefix}API_KEY")
        api_secret = os.getenv(f"{prefix}API_SECRET")
        base_url = os.getenv(f"{prefix}BASE_URL")
        websocket_url = os.getenv(f"{prefix}WEBSOCKET_URL")
        
        # Load additional configuration (any other vars with the prefix)
        additional_config = {}
        for key, value in os.environ.items():
            if key.startswith(prefix) and key not in [
                f"{prefix}API_KEY",
                f"{prefix}API_SECRET",
                f"{prefix}BASE_URL",
                f"{prefix}WEBSOCKET_URL"
            ]:
                config_key = key[len(prefix):].lower()
                additional_config[config_key] = value
        
        return EnvironmentConfig(
            mode=mode,
            api_key=api_key,
            api_secret=api_secret,
            base_url=base_url,
            websocket_url=websocket_url,
            additional_config=additional_config if additional_config else None
        )
    
    def get_config(self, mode: Optional[Environment] = None) -> Optional[EnvironmentConfig]:
        """
        Get configuration for specified mode.
        
        Args:
            mode: Environment mode (uses current if None)
            
        Returns:
            EnvironmentConfig or None if not found
        """
        target_mode = mode or self._current_environment
        if target_mode is None:
            logger.warning("No environment mode specified or set")
            return None
        
        return self._configs.get(target_mode)
    
    def get_current_config(self) -> Optional[EnvironmentConfig]:
        """
        Get current environment configuration.
        
        Returns:
            Current EnvironmentConfig or None
        """
        return self.get_config(self._current_environment)
    
    def switch_environment(
        self,
        target_mode: Environment,
        force: bool = False,
        require_valid_config: bool = True
    ) -> bool:
        """
        Safely switch to a different environment mode.
        
        Args:
            target_mode: Target environment to switch to
            force: Force switch even if credentials missing
            require_valid_config: Require valid API credentials before switching
            
        Returns:
            True if switch was successful, False otherwise
        """
        if not isinstance(target_mode, Environment):
            logger.error(f"Invalid environment mode: {target_mode}")
            return False
        
        # Check if target config exists
        if target_mode not in self._configs:
            logger.error(f"No configuration found for {target_mode.value}")
            return False
        
        target_config = self._configs[target_mode]
        
        # Validate configuration if required
        if require_valid_config and not target_config.is_valid() and not force:
            logger.error(
                f"Cannot switch to {target_mode.value}: missing API credentials. "
                f"Use force=True to override."
            )
            return False
        
        # Warn if switching to live without valid credentials
        if target_mode == Environment.LIVE and not target_config.is_valid():
            logger.warning(
                "⚠️  WARNING: Switching to LIVE mode without valid credentials!"
            )
        
        old_mode = self._current_environment
        self._current_environment = target_mode
        
        logger.info(
            f"Environment switched: {old_mode.value if old_mode else 'None'} "
            f"-> {target_mode.value}"
        )
        return True
    
    def get_current_mode(self) -> Optional[Environment]:
        """
        Get current environment mode.
        
        Returns:
            Current Environment or None if not set
        """
        return self._current_environment
    
    def is_live_mode(self) -> bool:
        """
        Check if currently in live trading mode.
        
        Returns:
            True if in LIVE mode
        """
        return self._current_environment == Environment.LIVE
    
    def is_paper_mode(self) -> bool:
        """
        Check if currently in paper trading mode.
        
        Returns:
            True if in PAPER mode
        """
        return self._current_environment == Environment.PAPER
    
    def is_testnet_mode(self) -> bool:
        """
        Check if currently in testnet mode.
        
        Returns:
            True if in TESTNET mode
        """
        return self._current_environment == Environment.TESTNET
    
    def validate_all_configs(self) -> Dict[Environment, bool]:
        """
        Validate all environment configurations.
        
        Returns:
            Dictionary mapping each environment to validation status
        """
        return {
            mode: config.is_valid()
            for mode, config in self._configs.items()
        }
    
    def get_missing_credentials(self) -> Dict[Environment, Set[str]]:
        """
        Get missing credentials for each environment.
        
        Returns:
            Dictionary mapping environments to sets of missing credential names
        """
        missing = {}
        
        for mode, config in self._configs.items():
            mode_missing = set()
            if not config.api_key:
                mode_missing.add("api_key")
            if not config.api_secret:
                mode_missing.add("api_secret")
            
            if mode_missing:
                missing[mode] = mode_missing
        
        return missing
    
    def get_status_summary(self) -> Dict:
        """
        Get comprehensive status summary.
        
        Returns:
            Dictionary with current status and all configs
        """
        return {
            "env_file": self._env_file,
            "env_loaded": self._env_loaded,
            "current_mode": self._current_environment.value if self._current_environment else None,
            "configurations": {
                mode.value: {
                    "valid": config.is_valid(),
                    "has_api_key": bool(config.api_key),
                    "has_api_secret": bool(config.api_secret),
                    "has_base_url": bool(config.base_url),
                    "has_websocket_url": bool(config.websocket_url),
                    "additional_config_keys": list(config.additional_config.keys()) if config.additional_config else []
                }
                for mode, config in self._configs.items()
            }
        }
    
    def __repr__(self) -> str:
        """String representation."""
        current = self._current_environment.value if self._current_environment else "None"
        return (f"EnvironmentManager(current_mode={current}, "
                f"env_loaded={self._env_loaded})")


# Convenience functions

def create_environment_manager(env_file: str = ".env") -> EnvironmentManager:
    """
    Create and initialize environment manager.
    
    Args:
        env_file: Path to .env file
        
    Returns:
        Initialized EnvironmentManager instance
    """
    return EnvironmentManager(env_file=env_file, auto_load=True)


def load_config(mode: Environment, env_file: str = ".env") -> Optional[EnvironmentConfig]:
    """
    Load configuration for specific environment mode.
    
    Args:
        mode: Environment mode to load
        env_file: Path to .env file
        
    Returns:
        EnvironmentConfig or None if loading failed
    """
    manager = EnvironmentManager(env_file=env_file, auto_load=True)
    return manager.get_config(mode)


# Example usage
if __name__ == "__main__":
    print("=== Trading Environment Manager Example ===\n")
    
    # Create example .env file for demonstration
    example_env_content = """
# Paper Trading Configuration
PAPER_API_KEY=paper_key_12345
PAPER_API_SECRET=paper_secret_67890
PAPER_BASE_URL=https://paper-api.example.com
PAPER_WEBSOCKET_URL=wss://paper-ws.example.com

# Testnet Configuration
TESTNET_API_KEY=testnet_key_abcde
TESTNET_API_SECRET=testnet_secret_fghij
TESTNET_BASE_URL=https://testnet-api.example.com

# Live Trading Configuration (example - keep these secure!)
LIVE_API_KEY=live_key_real
LIVE_API_SECRET=live_secret_real
LIVE_BASE_URL=https://api.example.com
LIVE_WEBSOCKET_URL=wss://ws.example.com
LIVE_RATE_LIMIT=100
"""
    
    # Write example .env if it doesn't exist
    if not Path(".env.example").exists():
        with open(".env.example", "w") as f:
            f.write(example_env_content.strip())
        print("Created .env.example file for reference\n")
    
    # Initialize manager
    manager = EnvironmentManager(env_file=".env", auto_load=False)
    print(f"Manager created: {manager}\n")
    
    # Try loading environment variables
    loaded = manager.load_environment_variables()
    print(f"Environment loaded: {loaded}\n")
    
    # Get status summary
    print("Configuration Status:")
    status = manager.get_status_summary()
    print(f"  Current mode: {status['current_mode']}")
    print(f"  Env loaded: {status['env_loaded']}")
    print("\nConfigurations:")
    for mode, config_status in status['configurations'].items():
        print(f"  {mode}:")
        print(f"    Valid: {config_status['valid']}")
        print(f"    Has API Key: {config_status['has_api_key']}")
        print(f"    Has API Secret: {config_status['has_api_secret']}")
    print()
    
    # Validate all configs
    validation = manager.validate_all_configs()
    print("Validation Results:")
    for mode, is_valid in validation.items():
        status_icon = "✓" if is_valid else "✗"
        print(f"  {status_icon} {mode.value}: {is_valid}")
    print()
    
    # Check for missing credentials
    missing = manager.get_missing_credentials()
    if missing:
        print("Missing Credentials:")
        for mode, creds in missing.items():
            print(f"  {mode.value}: {', '.join(creds)}")
    else:
        print("All credentials present!")
    print()
    
    # Switch to paper mode
    print("Switching to PAPER mode...")
    success = manager.switch_environment(Environment.PAPER, require_valid_config=False)
    print(f"Switch successful: {success}")
    print(f"Current mode: {manager.get_current_mode()}\n")
    
    # Get current config
    config = manager.get_current_config()
    if config:
        print(f"Current config: {config}\n")
    
    # Try switching to testnet
    print("Switching to TESTNET mode...")
    success = manager.switch_environment(Environment.TESTNET, require_valid_config=False)
    print(f"Switch successful: {success}")
    print(f"Is testnet mode: {manager.is_testnet_mode()}\n")
    
    # Demonstrate safe switching (would fail without valid credentials in real scenario)
    print("Attempting to switch to LIVE mode with validation...")
    success = manager.switch_environment(
        Environment.LIVE,
        require_valid_config=True,
        force=False
    )
    print(f"Switch successful: {success}")
    if not success:
        print("(Expected to fail if credentials not in .env)\n")
    
    # Final status
    print(f"Final manager state: {manager}")