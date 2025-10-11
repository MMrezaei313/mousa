"""
Setup utilities for Mousa Trading Bot
Environment setup, dependency checking, and configuration validation
"""

import os
import sys
import subprocess
import importlib
import platform
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import pkg_resources
import shutil

class EnvironmentSetup:
    """
    Handles environment setup and dependency management for Mousa Trading Bot
    """
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or Path.cwd())
        self.logger = self._setup_logging()
        self.required_packages = self._get_required_packages()
        self.optional_packages = self._get_optional_packages()
        
    def _setup_logging(self):
        """Setup logging for environment setup"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _get_required_packages(self) -> Dict[str, str]:
        """Get required packages and their versions"""
        return {
            'numpy': '>=1.21.0',
            'pandas': '>=1.3.0',
            'requests': '>=2.25.0',
            'python-dotenv': '>=0.19.0',
            'sqlalchemy': '>=1.4.0',
            'scipy': '>=1.7.0',
            'scikit-learn': '>=1.0.0',
            'matplotlib': '>=3.5.0',
            'seaborn': '>=0.11.0',
            'websocket-client': '>=1.2.0'
        }
    
    def _get_optional_packages(self) -> Dict[str, str]:
        """Get optional packages and their versions"""
        return {
            'ta-lib': '>=0.4.0',  # Technical analysis library
            'xgboost': '>=1.5.0',  # Machine learning
            'tensorflow': '>=2.6.0',  # Deep learning
            'keras': '>=2.6.0',  # Deep learning
            'plotly': '>=5.0.0',  # Interactive plotting
            'dash': '>=2.0.0',  # Web dashboard
            'flask': '>=2.0.0',  # Web framework
            'fastapi': '>=0.68.0',  # API framework
            'uvicorn': '>=0.15.0',  # ASGI server
            'python-telegram-bot': '>=13.0',  # Telegram notifications
            'ccxt': '>=3.0.0',  # Crypto exchange support
            'alpaca-trade-api': '>=2.0.0',  # Alpaca trading API
        }
    
    def check_system_requirements(self) -> Dict[str, Any]:
        """
        Check system requirements and compatibility
        """
        system_info = {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'python_version': platform.python_version(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor()
        }
        
        # Check Python version
        python_version = tuple(map(int, platform.python_version_tuple()[:2]))
        python_ok = python_version >= (3, 8)
        
        # Check available memory (approximate)
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024 ** 3)
            memory_ok = memory_gb >= 4.0  # 4GB minimum
        except ImportError:
            memory_gb = None
            memory_ok = True  # Assume sufficient if we can't check
        
        # Check disk space
        try:
            disk_usage = shutil.disk_usage(self.project_root)
            disk_space_gb = disk_usage.free / (1024 ** 3)
            disk_ok = disk_space_gb >= 1.0  # 1GB minimum
        except Exception:
            disk_space_gb = None
            disk_ok = True
        
        requirements_met = all([python_ok, memory_ok, disk_ok])
        
        return {
            'system_info': system_info,
            'requirements_met': requirements_met,
            'checks': {
                'python_version': {'met': python_ok, 'required': '>=3.8', 'actual': platform.python_version()},
                'memory': {'met': memory_ok, 'required': '>=4GB', 'actual': f'{memory_gb:.1f}GB' if memory_gb else 'Unknown'},
                'disk_space': {'met': disk_ok, 'required': '>=1GB', 'actual': f'{disk_space_gb:.1f}GB' if disk_space_gb else 'Unknown'}
            }
        }
    
    def check_package_installed(self, package_name: str, version_spec: str = None) -> Tuple[bool, Optional[str]]:
        """
        Check if a package is installed and meets version requirements
        
        Returns:
            Tuple of (is_installed, installed_version)
        """
        try:
            # Try to get package distribution
            dist = pkg_resources.get_distribution(package_name)
            installed_version = dist.version
            
            if version_spec:
                # Check if installed version meets requirement
                requirement = pkg_resources.Requirement.parse(f"{package_name}{version_spec}")
                is_compatible = dist in requirement
                return is_compatible, installed_version
            else:
                return True, installed_version
                
        except pkg_resources.DistributionNotFound:
            return False, None
        except Exception as e:
            self.logger.warning(f"Error checking package {package_name}: {e}")
            return False, None
    
    def check_dependencies(self, include_optional: bool = False) -> Dict[str, Any]:
        """
        Check all dependencies
        
        Args:
            include_optional: Whether to check optional packages too
            
        Returns:
            Dictionary with dependency check results
        """
        results = {
            'required': {},
            'optional': {},
            'all_required_met': True,
            'missing_required': [],
            'incompatible_versions': []
        }
        
        # Check required packages
        for package, version_spec in self.required_packages.items():
            is_installed, installed_version = self.check_package_installed(package, version_spec)
            results['required'][package] = {
                'installed': is_installed,
                'required_version': version_spec,
                'installed_version': installed_version
            }
            
            if not is_installed:
                results['all_required_met'] = False
                results['missing_required'].append(package)
            elif installed_version and not self._version_meets_requirement(installed_version, version_spec):
                results['all_required_met'] = False
                results['incompatible_versions'].append(f"{package} {installed_version} (requires {version_spec})")
        
        # Check optional packages if requested
        if include_optional:
            for package, version_spec in self.optional_packages.items():
                is_installed, installed_version = self.check_package_installed(package, version_spec)
                results['optional'][package] = {
                    'installed': is_installed,
                    'required_version': version_spec,
                    'installed_version': installed_version
                }
        
        return results
    
    def _version_meets_requirement(self, installed_version: str, requirement: str) -> bool:
        """Check if installed version meets requirement"""
        try:
            spec = pkg_resources.Requirement.parse(f"dummy{requirement}")
            installed = pkg_resources.parse_version(installed_version)
            return installed in spec
        except Exception:
            # If version parsing fails, assume it's compatible
            return True
    
    def install_package(self, package_spec: str, upgrade: bool = False) -> bool:
        """
        Install a package using pip
        
        Args:
            package_spec: Package specification (e.g., "numpy>=1.21.0")
            upgrade: Whether to upgrade if already installed
            
        Returns:
            True if installation successful
        """
        try:
            cmd = [sys.executable, "-m", "pip", "install"]
            if upgrade:
                cmd.append("--upgrade")
            cmd.append(package_spec)
            
            self.logger.info(f"Installing package: {package_spec}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Successfully installed {package_spec}")
                return True
            else:
                self.logger.error(f"Failed to install {package_spec}: {result.stderr}")
                return False
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Package installation failed for {package_spec}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error installing {package_spec}: {e}")
            return False
    
    def install_missing_dependencies(self, include_optional: bool = False) -> Dict[str, Any]:
        """
        Install missing required dependencies
        
        Returns:
            Dictionary with installation results
        """
        dependency_check = self.check_dependencies(include_optional=include_optional)
        installation_results = {
            'installed': [],
            'failed': [],
            'skipped': []
        }
        
        # Install missing required packages
        for package in dependency_check['missing_required']:
            version_spec = self.required_packages[package]
            package_spec = f"{package}{version_spec}"
            
            if self.install_package(package_spec):
                installation_results['installed'].append(package)
            else:
                installation_results['failed'].append(package)
        
        # Install optional packages if requested
        if include_optional:
            for package, info in dependency_check['optional'].items():
                if not info['installed']:
                    version_spec = self.optional_packages[package]
                    package_spec = f"{package}{version_spec}"
                    
                    if self.install_package(package_spec):
                        installation_results['installed'].append(package)
                    else:
                        installation_results['failed'].append(package)
                else:
                    installation_results['skipped'].append(package)
        
        return installation_results
    
    def create_directory_structure(self) -> bool:
        """
        Create necessary directory structure for the project
        """
        directories = [
            'data/raw',
            'data/processed',
            'data/models',
            'logs',
            'config',
            'backups',
            'reports',
            'cache'
        ]
        
        try:
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created directory: {dir_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create directory structure: {e}")
            return False
    
    def setup_environment(self, install_deps: bool = True, create_dirs: bool = True) -> Dict[str, Any]:
        """
        Complete environment setup
        
        Returns:
            Dictionary with setup results
        """
        setup_results = {
            'system_check': self.check_system_requirements(),
            'dependency_check': None,
            'installation_results': None,
            'directory_creation': None,
            'success': False
        }
        
        # Check system requirements
        if not setup_results['system_check']['requirements_met']:
            self.logger.error("System requirements not met")
            return setup_results
        
        # Check dependencies
        dependency_check = self.check_dependencies()
        setup_results['dependency_check'] = dependency_check
        
        # Install dependencies if requested
        if install_deps and not dependency_check['all_required_met']:
            installation_results = self.install_missing_dependencies()
            setup_results['installation_results'] = installation_results
            
            # Re-check dependencies after installation
            dependency_check = self.check_dependencies()
            setup_results['dependency_check'] = dependency_check
        
        # Create directory structure
        if create_dirs:
            dir_success = self.create_directory_structure()
            setup_results['directory_creation'] = {'success': dir_success}
        
        # Final success check
        setup_results['success'] = (
            setup_results['system_check']['requirements_met'] and
            dependency_check['all_required_met'] and
            (not create_dirs or setup_results['directory_creation']['success'])
        )
        
        return setup_results
    
    def generate_requirements_file(self, include_optional: bool = False, 
                                 output_path: str = "requirements.txt") -> bool:
        """
        Generate requirements.txt file
        
        Returns:
            True if file generated successfully
        """
        try:
            requirements = []
            
            # Add required packages
            for package, version_spec in self.required_packages.items():
                requirements.append(f"{package}{version_spec}")
            
            # Add optional packages if requested
            if include_optional:
                for package, version_spec in self.optional_packages.items():
                    requirements.append(f"{package}{version_spec}")
            
            # Write to file
            with open(self.project_root / output_path, 'w') as f:
                for req in sorted(requirements):
                    f.write(f"{req}\n")
            
            self.logger.info(f"Requirements file generated: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate requirements file: {e}")
            return False

# Utility functions
def get_package_version(package_name: str) -> Optional[str]:
    """Get version of installed package"""
    try:
        return pkg_resources.get_distribution(package_name).version
    except:
        return None

def is_environment_ready() -> bool:
    """Quick check if environment is ready"""
    setup = EnvironmentSetup()
    system_check = setup.check_system_requirements()
    dependency_check = setup.check_dependencies()
    
    return system_check['requirements_met'] and dependency_check['all_required_met']

def setup_development_environment() -> Dict[str, Any]:
    """Convenience function for development environment setup"""
    setup = EnvironmentSetup()
    return setup.setup_environment(install_deps=True, create_dirs=True)

if __name__ == "__main__":
    # Test the setup
    setup = EnvironmentSetup()
    results = setup.setup_environment()
    print("Environment Setup Results:", results)
