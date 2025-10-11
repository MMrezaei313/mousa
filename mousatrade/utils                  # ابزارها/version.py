"""
Version utilities for Mousa Trading Bot
Version management, compatibility checking, and update notifications
"""

import importlib
import pkg_resources
from typing import Dict, List, Optional, Tuple, Any
import logging
import requests
from datetime import datetime, timedelta
import json
from pathlib import Path

class VersionManager:
    """
    Manages version information and compatibility for Mousa Trading Bot
    """
    
    def __init__(self, package_name: str = "mousatrade"):
        self.package_name = package_name
        self.logger = self._setup_logging()
        self._version_cache = {}
        self._last_update_check = None
        
    def _setup_logging(self):
        """Setup logging for version manager"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def get_current_version(self) -> Dict[str, Any]:
        """
        Get current version information for the package
        
        Returns:
            Dictionary with version details
        """
        try:
            # Try to get package distribution
            dist = pkg_resources.get_distribution(self.package_name)
            version = dist.version
            location = dist.location
            
            # Parse version components
            version_parts = version.split('.')
            major = int(version_parts[0]) if version_parts[0].isdigit() else 0
            minor = int(version_parts[1]) if len(version_parts) > 1 and version_parts[1].isdigit() else 0
            patch = int(version_parts[2]) if len(version_parts) > 2 and version_parts[2].isdigit() else 0
            
            return {
                'version': version,
                'major': major,
                'minor': minor,
                'patch': patch,
                'location': str(location),
                'is_editable': '.egg-info' in str(location) or location.endswith('/src'),
                'full_version': f"v{version}",
                'semantic_version': f"{major}.{minor}.{patch}"
            }
            
        except pkg_resources.DistributionNotFound:
            # Package not installed via pip, try to get from __version__
            try:
                module = importlib.import_module(self.package_name)
                version = getattr(module, '__version__', '0.0.0-dev')
                
                return {
                    'version': version,
                    'major': 0,
                    'minor': 0,
                    'patch': 0,
                    'location': 'development',
                    'is_editable': True,
                    'full_version': f"v{version}",
                    'semantic_version': version
                }
                
            except ImportError:
                return {
                    'version': '0.0.0-unknown',
                    'major': 0,
                    'minor': 0,
                    'patch': 0,
                    'location': 'unknown',
                    'is_editable': False,
                    'full_version': 'v0.0.0-unknown',
                    'semantic_version': '0.0.0'
                }
    
    def get_dependency_versions(self) -> Dict[str, Dict[str, str]]:
        """
        Get versions of all dependencies
        
        Returns:
            Dictionary mapping package names to version info
        """
        dependencies = {}
        required_packages = [
            'numpy', 'pandas', 'scikit-learn', 'scipy', 'matplotlib',
            'seaborn', 'requests', 'sqlalchemy', 'websocket-client'
        ]
        
        for package in required_packages:
            try:
                dist = pkg_resources.get_distribution(package)
                dependencies[package] = {
                    'version': dist.version,
                    'location': str(dist.location)
                }
            except pkg_resources.DistributionNotFound:
                dependencies[package] = {
                    'version': 'not installed',
                    'location': 'unknown'
                }
        
        return dependencies
    
    def check_pypi_version(self, package_name: str = None) -> Optional[str]:
        """
        Check the latest version available on PyPI
        
        Args:
            package_name: Package name (defaults to main package)
            
        Returns:
            Latest version string or None if unable to check
        """
        package = package_name or self.package_name
        
        # Check cache first
        if package in self._version_cache:
            cached_time, version = self._version_cache[package]
            if datetime.now() - cached_time < timedelta(hours=1):
                return version
        
        try:
            url = f"https://pypi.org/pypi/{package}/json"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                latest_version = data['info']['version']
                
                # Cache the result
                self._version_cache[package] = (datetime.now(), latest_version)
                self._last_update_check = datetime.now()
                
                return latest_version
            else:
                self.logger.warning(f"Failed to fetch PyPI info for {package}: HTTP {response.status_code}")
                return None
                
        except requests.RequestException as e:
            self.logger.warning(f"Network error checking PyPI for {package}: {e}")
            return None
        except Exception as e:
            self.logger.warning(f"Unexpected error checking PyPI for {package}: {e}")
            return None
    
    def check_for_updates(self, package_name: str = None) -> Dict[str, Any]:
        """
        Check if updates are available
        
        Returns:
            Dictionary with update information
        """
        package = package_name or self.package_name
        current_info = self.get_current_version()
        latest_version = self.check_pypi_version(package)
        
        if not latest_version:
            return {
                'update_available': False,
                'error': 'Unable to check for updates',
                'current_version': current_info['version'],
                'latest_version': None
            }
        
        # Compare versions
        current = current_info['version']
        update_available = self._compare_versions(current, latest_version) < 0
        
        result = {
            'update_available': update_available,
            'current_version': current,
            'latest_version': latest_version,
            'current_version_info': current_info,
            'last_checked': datetime.now().isoformat()
        }
        
        if update_available:
            result['update_type'] = self._get_update_type(current, latest_version)
            result['message'] = f"Update available: {current} -> {latest_version}"
        else:
            result['message'] = "You have the latest version"
        
        return result
    
    def _compare_versions(self, v1: str, v2: str) -> int:
        """
        Compare two version strings
        
        Returns:
            -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2
        """
        try:
            # Normalize versions
            v1_parts = [int(x) for x in v1.split('.')]
            v2_parts = [int(x) for x in v2.split('.')]
            
            # Pad with zeros if necessary
            max_len = max(len(v1_parts), len(v2_parts))
            v1_parts.extend([0] * (max_len - len(v1_parts)))
            v2_parts.extend([0] * (max_len - len(v2_parts)))
            
            for i in range(max_len):
                if v1_parts[i] < v2_parts[i]:
                    return -1
                elif v1_parts[i] > v2_parts[i]:
                    return 1
            
            return 0
            
        except ValueError:
            # Fallback for non-numeric versions
            if v1 < v2:
                return -1
            elif v1 > v2:
                return 1
            else:
                return 0
    
    def _get_update_type(self, current: str, latest: str) -> str:
        """Determine the type of update (major, minor, patch)"""
        try:
            current_parts = [int(x) for x in current.split('.')]
            latest_parts = [int(x) for x in latest.split('.')]
            
            if current_parts[0] < latest_parts[0]:
                return 'major'
            elif current_parts[1] < latest_parts[1]:
                return 'minor'
            else:
                return 'patch'
                
        except (ValueError, IndexError):
            return 'unknown'
    
    def get_version_compatibility(self) -> Dict[str, Any]:
        """
        Check compatibility between package versions
        
        Returns:
            Dictionary with compatibility information
        """
        current_version = self.get_current_version()
        dependencies = self.get_dependency_versions()
        
        # Known compatibility issues (this would normally come from a config file)
        known_issues = {
            'pandas': {
                'min_version': '1.3.0',
                'max_version': '2.0.0',
                'issues': {
                    '2.0.0': 'Some breaking changes in pandas 2.0'
                }
            },
            'numpy': {
                'min_version': '1.21.0',
                'max_version': '2.0.0',
                'issues': {}
            }
        }
        
        compatibility = {
            'package_version': current_version['version'],
            'compatible': True,
            'issues': [],
            'warnings': []
        }
        
        # Check each dependency
        for dep, info in dependencies.items():
            if dep in known_issues:
                requirements = known_issues[dep]
                dep_version = info['version']
                
                if dep_version == 'not installed':
                    compatibility['issues'].append(f"{dep} is not installed")
                    compatibility['compatible'] = False
                    continue
                
                # Check minimum version
                min_ok = self._compare_versions(dep_version, requirements['min_version']) >= 0
                if not min_ok:
                    compatibility['issues'].append(
                        f"{dep} {dep_version} is below minimum required {requirements['min_version']}"
                    )
                    compatibility['compatible'] = False
                
                # Check maximum version
                if requirements['max_version']:
                    max_ok = self._compare_versions(dep_version, requirements['max_version']) <= 0
                    if not max_ok:
                        compatibility['warnings'].append(
                            f"{dep} {dep_version} is above tested version {requirements['max_version']}"
                        )
                
                # Check for specific known issues
                for problem_version, issue in requirements['issues'].items():
                    if dep_version.startswith(problem_version):
                        compatibility['warnings'].append(
                            f"{dep} {dep_version}: {issue}"
                        )
        
        return compatibility
    
    def generate_version_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive version report
        
        Returns:
            Dictionary with complete version information
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'package': self.get_current_version(),
            'dependencies': self.get_dependency_versions(),
            'update_check': self.check_for_updates(),
            'compatibility': self.get_version_compatibility(),
            'environment': {
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'platform': sys.platform,
                'machine': platform.machine()
            }
        }
    
    def save_version_report(self, filepath: str = "version_report.json") -> bool:
        """
        Save version report to file
        
        Returns:
            True if successful
        """
        try:
            report = self.generate_version_report()
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Version report saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save version report: {e}")
            return False
    
    def print_version_info(self):
        """Print version information to console"""
        current = self.get_current_version()
        dependencies = self.get_dependency_versions()
        update_info = self.check_for_updates()
        
        print(f"\n{'='*50}")
        print(f"MOUSATRADE VERSION INFORMATION")
        print(f"{'='*50}")
        print(f"Package: {self.package_name}")
        print(f"Version: {current['version']}")
        print(f"Location: {current['location']}")
        print(f"Editable install: {current['is_editable']}")
        
        if update_info['update_available']:
            print(f"\n🚀 UPDATE AVAILABLE: {update_info['current_version']} → {update_info['latest_version']}")
            print(f"Update type: {update_info['update_type']}")
        else:
            print(f"\n✅ You have the latest version")
        
        print(f"\nDependencies:")
        for dep, info in dependencies.items():
            status = "✅" if info['version'] != 'not installed' else "❌"
            print(f"  {status} {dep}: {info['version']}")
        
        compatibility = self.get_version_compatibility()
        if not compatibility['compatible']:
            print(f"\n⚠️  COMPATIBILITY ISSUES:")
            for issue in compatibility['issues']:
                print(f"  • {issue}")
        
        if compatibility['warnings']:
            print(f"\n📝 WARNINGS:")
            for warning in compatibility['warnings']:
                print(f"  • {warning}")
        
        print(f"{'='*50}\n")

# Global version manager instance
_version_manager = None

def get_version_manager() -> VersionManager:
    """Get global version manager instance"""
    global _version_manager
    if _version_manager is None:
        _version_manager = VersionManager()
    return _version_manager

def get_version() -> str:
    """Get current package version"""
    return get_version_manager().get_current_version()['version']

def check_updates() -> Dict[str, Any]:
    """Check for package updates"""
    return get_version_manager().check_for_updates()

def is_compatible() -> bool:
    """Check if current environment is compatible"""
    return get_version_manager().get_version_compatibility()['compatible']

# Version constants (should match pyproject.toml)
__version__ = "1.0.0"
__version_info__ = (1, 0, 0)

if __name__ == "__main__":
    # Test version utilities
    vm = VersionManager()
    vm.print_version_info()
    
    # Save detailed report
    vm.save_version_report("version_report.json")
