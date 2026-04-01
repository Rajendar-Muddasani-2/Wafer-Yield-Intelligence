"""
Configuration Loader Module

Handles loading configuration from environment variables and config files.
Provides centralized access to all application settings.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing"""
    pass


class Config:
    """Application configuration management"""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            env_file: Path to .env file. If None, searches in standard locations.
        """
        self._load_environment(env_file)
        self._validate_required_config()
        
    def _load_environment(self, env_file: Optional[str] = None):
        """Load environment variables from .env file"""
        if env_file:
            env_path = Path(env_file)
        else:
            # Search for .env in current directory and parent directories
            current_dir = Path.cwd()
            env_path = current_dir / '.env'
            
            if not env_path.exists():
                # Try parent directories
                for parent in current_dir.parents:
                    potential_path = parent / '.env'
                    if potential_path.exists():
                        env_path = potential_path
                        break
        
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded environment from: {env_path}")
        else:
            logger.warning("No .env file found. Using system environment variables only.")
    
    def _validate_required_config(self):
        """Validate that required configuration is present"""
        required_vars = [
            'DB_USERNAME',
            'DB_PASSWORD',
            'DB_TNS',
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ConfigurationError(
                f"Missing required environment variables: {', '.join(missing_vars)}\n"
                f"Please ensure .env file is configured properly."
            )
    
    # Database Configuration
    @property
    def db_username(self) -> str:
        return os.getenv('DB_USERNAME', '')
    
    @property
    def db_password(self) -> str:
        return os.getenv('DB_PASSWORD', '')
    
    @property
    def db_tns(self) -> str:
        return os.getenv('DB_TNS', '')
    
    @property
    def db_port(self) -> int:
        return int(os.getenv('DB_PORT', '1521'))
    
    @property
    def db_service_name(self) -> str:
        return os.getenv('DB_SERVICE_NAME', '')
    
    @property
    def db_connection_string(self) -> str:
        """Get Oracle connection string"""
        return f'oracle+cx_oracle://{self.db_username}:{self.db_password}@{self.db_tns}'
    
    # Data Paths
    @property
    def input_data_path(self) -> str:
        return os.getenv('INPUT_DATA_PATH', '')
    
    @property
    def oracle_client_path(self) -> str:
        return os.getenv('ORACLE_CLIENT_PATH', './oracle_client_lib')
    
    @property
    def data_dir(self) -> Path:
        return Path(os.getenv('DATA_DIR', './data'))
    
    @property
    def models_dir(self) -> Path:
        return Path(os.getenv('MODELS_DIR', './models'))
    
    @property
    def output_dir(self) -> Path:
        return Path(os.getenv('OUTPUT_DIR', './data/output'))
    
    # Model Configuration
    @property
    def default_retest_model(self) -> str:
        return os.getenv('DEFAULT_RETEST_MODEL', 'VC')
    
    @property
    def default_pattern_model(self) -> str:
        return os.getenv('DEFAULT_PATTERN_MODEL', 'VE')
    
    @property
    def product_type(self) -> str:
        return os.getenv('PRODUCT_TYPE', 'M4289B00012')
    
    @property
    def tp_version(self) -> str:
        return os.getenv('TP_VERSION', 'P20')
    
    # Processing Configuration
    @property
    def product_family(self) -> str:
        return os.getenv('PRODUCT_FAMILY', 'A1G')
    
    @property
    def batch_size(self) -> int:
        return int(os.getenv('BATCH_SIZE', '1000'))
    
    @property
    def max_workers(self) -> int:
        return int(os.getenv('MAX_WORKERS', '4'))
    
    # Logging
    @property
    def log_level(self) -> str:
        return os.getenv('LOG_LEVEL', 'INFO')
    
    @property
    def log_file(self) -> str:
        return os.getenv('LOG_FILE', './logs/wafer_yield_intelligence.log')
    
    @property
    def log_format(self) -> str:
        return os.getenv('LOG_FORMAT', 
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Feature Flags
    @property
    def enable_pattern_recognition(self) -> bool:
        return os.getenv('ENABLE_PATTERN_RECOGNITION', 'true').lower() == 'true'
    
    @property
    def enable_retest_prediction(self) -> bool:
        return os.getenv('ENABLE_RETEST_PREDICTION', 'true').lower() == 'true'
    
    @property
    def enable_root_cause_analysis(self) -> bool:
        return os.getenv('ENABLE_ROOT_CAUSE_ANALYSIS', 'true').lower() == 'true'
    
    # Performance Tuning
    @property
    def db_pool_size(self) -> int:
        return int(os.getenv('DB_POOL_SIZE', '5'))
    
    @property
    def db_max_overflow(self) -> int:
        return int(os.getenv('DB_MAX_OVERFLOW', '10'))
    
    @property
    def cache_enabled(self) -> bool:
        return os.getenv('CACHE_ENABLED', 'true').lower() == 'true'
    
    @property
    def cache_ttl(self) -> int:
        return int(os.getenv('CACHE_TTL', '3600'))
    
    # Debug
    @property
    def debug_mode(self) -> bool:
        return os.getenv('DEBUG_MODE', 'false').lower() == 'true'
    
    @property
    def profile_performance(self) -> bool:
        return os.getenv('PROFILE_PERFORMANCE', 'false').lower() == 'true'
    
    @property
    def save_intermediate_files(self) -> bool:
        return os.getenv('SAVE_INTERMEDIATE_FILES', 'false').lower() == 'true'
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary (excluding sensitive data)"""
        return {
            'db_tns': self.db_tns,
            'db_port': self.db_port,
            'input_data_path': self.input_data_path,
            'product_family': self.product_family,
            'product_type': self.product_type,
            'tp_version': self.tp_version,
            'default_retest_model': self.default_retest_model,
            'default_pattern_model': self.default_pattern_model,
            'batch_size': self.batch_size,
            'max_workers': self.max_workers,
            'log_level': self.log_level,
            'debug_mode': self.debug_mode,
        }


# Global configuration instance
_config: Optional[Config] = None


def get_config(env_file: Optional[str] = None) -> Config:
    """
    Get or create global configuration instance
    
    Args:
        env_file: Optional path to .env file
        
    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config(env_file)
    return _config


def reload_config(env_file: Optional[str] = None):
    """
    Force reload configuration
    
    Args:
        env_file: Optional path to .env file
    """
    global _config
    _config = Config(env_file)
    logger.info("Configuration reloaded")


if __name__ == "__main__":
    # Test configuration loading
    import json
    
    try:
        config = get_config()
        print("✅ Configuration loaded successfully!")
        print("\nConfiguration Summary:")
        print(json.dumps(config.to_dict(), indent=2))
    except ConfigurationError as e:
        print(f"❌ Configuration Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
