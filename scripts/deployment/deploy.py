#!/usr/bin/env python3
"""
Mousa Trading Bot - Deployment Script
Automates the deployment process for different environments
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
import logging
from datetime import datetime
import yaml
import json

class DeploymentManager:
    def __init__(self, environment='development'):
        self.environment = environment
        self.project_root = Path(__file__).parent.parent.parent
        self.setup_logging()
        self.config = self.load_deployment_config()
        
    def setup_logging(self):
        """Setup deployment logging"""
        log_dir = self.project_root / 'logs' / 'deployment'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'deployment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_deployment_config(self):
        """Load deployment configuration"""
        config_path = self.project_root / 'config' / 'deployment_config.yaml'
        
        default_config = {
            'development': {
                'database_path': 'data/trading_bot_dev.db',
                'log_level': 'DEBUG',
                'enable_debug': True,
                'api_timeout': 30,
                'backup_before_deploy': False
            },
            'staging': {
                'database_path': 'data/trading_bot_staging.db',
                'log_level': 'INFO',
                'enable_debug': False,
                'api_timeout': 15,
                'backup_before_deploy': True
            },
            'production': {
                'database_path': 'data/trading_bot_prod.db',
                'log_level': 'WARNING',
                'enable_debug': False,
                'api_timeout': 10,
                'backup_before_deploy': True,
                'require_confirmation': True
            }
        }
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            self.logger.warning(f"Deployment config not found at {config_path}, using defaults")
            return default_config
    
    def run_command(self, command, check=True, capture_output=False):
        """Run shell command with logging"""
        self.logger.info(f"Running command: {command}")
        
        try:
            if capture_output:
                result = subprocess.run(
                    command, shell=True, check=check,
                    capture_output=True, text=True, cwd=self.project_root
                )
                return result.stdout.strip()
            else:
                subprocess.run(command, shell=True, check=check, cwd=self.project_root)
                return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {e}")
            if check:
                raise
            return False
    
    def backup_database(self):
        """Backup database before deployment"""
        db_path = self.project_root / self.config[self.environment].get('database_path', 'data/trading_bot.db')
        
        if db_path.exists():
            backup_dir = self.project_root / 'backups' / 'database'
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"{db_path.stem}_backup_{timestamp}.db"
            
            shutil.copy2(db_path, backup_path)
            self.logger.info(f"Database backed up to: {backup_path}")
            return backup_path
        else:
            self.logger.warning("Database file not found, skipping backup")
            return None
    
    def setup_virtual_environment(self):
        """Setup Python virtual environment"""
        venv_path = self.project_root / 'venv'
        
        if not venv_path.exists():
            self.logger.info("Creating virtual environment...")
            self.run_command('python -m venv venv')
        else:
            self.logger.info("Virtual environment already exists")
        
        # Use venv binaries
        if sys.platform == "win32":
            python_bin = venv_path / 'Scripts' / 'python.exe'
            pip_bin = venv_path / 'Scripts' / 'pip.exe'
        else:
            python_bin = venv_path / 'bin' / 'python'
            pip_bin = venv_path / 'bin' / 'pip'
        
        return python_bin, pip_bin
    
    def install_dependencies(self, pip_bin):
        """Install project dependencies"""
        self.logger.info("Installing dependencies...")
        
        requirements_files = [
            'requirements.txt',
            'requirements-dev.txt' if self.environment == 'development' else None
        ]
        
        for req_file in requirements_files:
            if req_file and (self.project_root / req_file).exists():
                self.run_command(f'"{pip_bin}" install -r {req_file}')
        
        # Install core packages if no requirements file
        if not (self.project_root / 'requirements.txt').exists():
            core_packages = [
                'pandas', 'numpy', 'matplotlib', 'seaborn',
                'requests', 'websocket-client', 'python-dotenv',
                'pandas-ta', 'sqlalchemy', 'flask'
            ]
            for package in core_packages:
                self.run_command(f'"{pip_bin}" install {package}')
    
    def run_tests(self, python_bin):
        """Run test suite"""
        self.logger.info("Running tests...")
        
        test_commands = [
            f'"{python_bin}" -m pytest tests/ -v',
            f'"{python_bin}" -c "import scripts.data_collection.fetch_market_data; print(\"Data collection import OK\")"',
            f'"{python_bin}" -c "import scripts.analysis.technical_analysis; print(\"Technical analysis import OK\")"',
            f'"{python_bin}" scripts/utilities/database_setup.py --check'
        ]
        
        for cmd in test_commands:
            success = self.run_command(cmd, check=False)
            if not success:
                self.logger.warning(f"Test command failed: {cmd}")
    
    def update_configuration(self):
        """Update configuration for target environment"""
        self.logger.info(f"Updating configuration for {self.environment} environment...")
        
        # Update app configuration
        app_config_path = self.project_root / 'config' / 'app_config.json'
        if app_config_path.exists():
            with open(app_config_path, 'r') as f:
                app_config = json.load(f)
            
            app_config['environment'] = self.environment
            app_config['debug'] = self.config[self.environment].get('enable_debug', False)
            
            with open(app_config_path, 'w') as f:
                json.dump(app_config, f, indent=2)
        
        # Update trading configuration
        trading_config_path = self.project_root / 'config' / 'trading_config.json'
        if trading_config_path.exists():
            with open(trading_config_path, 'r') as f:
                trading_config = json.load(f)
            
            # Disable trading in production by default for safety
            if self.environment == 'production':
                trading_config['enabled'] = False
                trading_config['demo_mode'] = True
            
            with open(trading_config_path, 'w') as f:
                json.dump(trading_config, f, indent=2)
        
        # Create environment-specific .env file
        env_template_path = self.project_root / '.env.template'
        env_path = self.project_root / '.env'
        
        if env_template_path.exists() and not env_path.exists():
            shutil.copy2(env_template_path, env_path)
            self.logger.info("Created .env file from template")
    
    def run_database_migrations(self, python_bin):
        """Run database migrations and setup"""
        self.logger.info("Running database setup...")
        
        db_script = self.project_root / 'scripts' / 'utilities' / 'database_setup.py'
        if db_script.exists():
            self.run_command(f'"{python_bin}" {db_script}')
        else:
            self.logger.warning("Database setup script not found")
    
    def deploy_to_production_checks(self):
        """Perform safety checks for production deployment"""
        if self.environment != 'production':
            return True
        
        self.logger.warning("PRODUCTION DEPLOYMENT SAFETY CHECKS:")
        
        # Check if trading is disabled
        trading_config_path = self.project_root / 'config' / 'trading_config.json'
        if trading_config_path.exists():
            with open(trading_config_path, 'r') as f:
                trading_config = json.load(f)
            
            if trading_config.get('enabled', False):
                self.logger.error("TRADING IS ENABLED IN PRODUCTION CONFIG!")
                response = input("Continue anyway? (yes/NO): ")
                if response.lower() != 'yes':
                    return False
        
        # Check for API keys
        env_path = self.project_root / '.env'
        if env_path.exists():
            with open(env_path, 'r') as f:
                env_content = f.read()
            
            if 'your_binance_api_key_here' in env_content:
                self.logger.error("DEFAULT API KEYS DETECTED!")
                return False
        
        # Confirm deployment
        if self.config['production'].get('require_confirmation', True):
            response = input(f"Confirm production deployment to {self.project_root}? (yes/NO): ")
            if response.lower() != 'yes':
                return False
        
        return True
    
    def create_deployment_report(self):
        """Create deployment report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'environment': self.environment,
            'project_path': str(self.project_root),
            'python_version': sys.version,
            'system_platform': sys.platform,
            'deployment_status': 'success'
        }
        
        report_dir = self.project_root / 'reports' / 'deployment'
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = report_dir / f"deployment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Deployment report saved to: {report_path}")
        return report_path
    
    def deploy(self, skip_tests=False, force=False):
        """Main deployment method"""
        self.logger.info(f"Starting deployment to {self.environment} environment...")
        
        try:
            # Production safety checks
            if not force and not self.deploy_to_production_checks():
                self.logger.error("Deployment aborted due to safety checks")
                return False
            
            # Backup database if configured
            if self.config[self.environment].get('backup_before_deploy', False):
                self.backup_database()
            
            # Setup virtual environment
            python_bin, pip_bin = self.setup_virtual_environment()
            
            # Install dependencies
            self.install_dependencies(pip_bin)
            
            # Run tests (unless skipped)
            if not skip_tests:
                self.run_tests(python_bin)
            
            # Update configuration
            self.update_configuration()
            
            # Run database setup
            self.run_database_migrations(python_bin)
            
            # Create deployment report
            self.create_deployment_report()
            
            self.logger.info(f"Deployment to {self.environment} completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Mousa Trading Bot Deployment Script')
    parser.add_argument('--environment', '-e', choices=['development', 'staging', 'production'],
                       default='development', help='Target environment')
    parser.add_argument('--skip-tests', action='store_true', help='Skip test execution')
    parser.add_argument('--force', '-f', action='store_true', 
                       help='Force deployment without safety checks')
    parser.add_argument('--backup-only', action='store_true', help='Only backup database')
    
    args = parser.parse_args()
    
    deployer = DeploymentManager(environment=args.environment)
    
    if args.backup_only:
        deployer.backup_database()
        return
    
    success = deployer.deploy(skip_tests=args.skip_tests, force=args.force)
    
    if success:
        print(f"\n✅ Deployment to {args.environment} completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Deployment to {args.environment} failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()
