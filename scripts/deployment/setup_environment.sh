#!/bin/bash

# Mousa Trading Bot - Environment Setup Script
# This script sets up the complete trading bot environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_warning "Running as root is not recommended. Consider running as a regular user."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Detect operating system
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        OS="windows"
    else
        OS="unknown"
    fi
    log_info "Detected OS: $OS"
}

# Check Python version
check_python() {
    if command -v python3 &>/dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
        log_info "Python version: $PYTHON_VERSION"
        
        # Check if Python version is >= 3.8
        python3 -c 'import sys; exit(1) if sys.version_info < (3, 8) else exit(0)'
        if [[ $? -ne 0 ]]; then
            log_error "Python 3.8 or higher is required. Current version: $PYTHON_VERSION"
            exit 1
        fi
    else
        log_error "Python 3 is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    log_info "Creating Python virtual environment..."
    
    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
        log_success "Virtual environment created"
    else
        log_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    log_info "Activating virtual environment..."
    source venv/bin/activate
}

# Install Python dependencies
install_dependencies() {
    log_info "Installing Python dependencies..."
    
    # Upgrade pip first
    pip install --upgrade pip
    
    # Install from requirements.txt if exists
    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt
    else
        # Install core dependencies
        pip install pandas numpy matplotlib seaborn
        pip install requests websocket-client python-dotenv
        pip install ta-lib pandas-ta
        pip install sqlalchemy datasets
        pip install flask fastapi uvicorn
        pip install python-telegram-bot
        pip install schedule APScheduler
        
        log_warning "No requirements.txt found. Installed core dependencies."
    fi
    
    log_success "Python dependencies installed"
}

# Install TA-Lib
install_talib() {
    log_info "Installing TA-Lib..."
    
    case $OS in
        "linux")
            log_info "Installing TA-Lib dependencies for Linux..."
            sudo apt-get update
            sudo apt-get install -y build-essential cmake
            wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
            tar -xzf ta-lib-0.4.0-src.tar.gz
            cd ta-lib/
            ./configure --prefix=/usr
            make
            sudo make install
            cd ..
            rm -rf ta-lib-0.4.0-src.tar.gz ta-lib/
            pip install TA-Lib
            ;;
        "macos")
            log_info "Installing TA-Lib for macOS..."
            brew install ta-lib
            pip install TA-Lib
            ;;
        "windows")
            log_warning "TA-Lib installation on Windows may require manual setup"
            pip install TA-Lib
            ;;
        *)
            log_warning "Skipping TA-Lib installation for unknown OS"
            ;;
    esac
    
    log_success "TA-Lib installed"
}

# Create directory structure
create_directories() {
    log_info "Creating directory structure..."
    
    directories=(
        "data"
        "logs"
        "config"
        "backups"
        "reports"
        "scripts/data_collection"
        "scripts/analysis"
        "scripts/utilities"
        "scripts/deployment"
    )
    
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        fi
    done
    
    log_success "Directory structure created"
}

# Create configuration files
create_config_files() {
    log_info "Creating configuration files..."
    
    # Create app config
    cat > config/app_config.json << EOF
{
    "name": "Mousa Trading Bot",
    "version": "1.0.0",
    "environment": "development",
    "debug": true,
    "host": "0.0.0.0",
    "port": 5000
}
EOF

    # Create trading config
    cat > config/trading_config.json << EOF
{
    "enabled": false,
    "demo_mode": true,
    "risk_management": {
        "max_position_size": 0.1,
        "stop_loss_percent": 2.0,
        "take_profit_percent": 4.0,
        "max_daily_loss": 0.05,
        "max_portfolio_risk": 0.1
    },
    "strategies": {
        "technical": {
            "enabled": true,
            "timeframes": ["15m", "1h", "4h"],
            "indicators": ["rsi", "macd", "bollinger_bands"]
        }
    },
    "symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT"]
}
EOF

    # Create environment template
    cat > .env.template << EOF
# Binance API Configuration
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here

# News API Configuration
NEWS_API_KEY=your_news_api_key_here

# Database Configuration
DATABASE_URL=sqlite:///data/trading_bot.db

# Trading Configuration
TRADING_ENABLED=false
DEMO_MODE=true
EOF

    log_success "Configuration files created"
}

# Setup database
setup_database() {
    log_info "Setting up database..."
    
    # Run database setup script
    if [[ -f "scripts/utilities/database_setup.py" ]]; then
        python scripts/utilities/database_setup.py
        log_success "Database setup completed"
    else
        log_warning "Database setup script not found"
    fi
}

# Setup logging
setup_logging() {
    log_info "Setting up logging system..."
    
    # Create log files
    touch logs/trading_bot.log
    touch logs/errors.log
    touch logs/trading_signals.log
    
    log_success "Logging system setup completed"
}

# Setup systemd service (Linux only)
setup_systemd_service() {
    if [[ "$OS" != "linux" ]]; then
        return
    fi
    
    log_info "Setting up systemd service..."
    
    SERVICE_FILE="/etc/systemd/system/mousa-trading-bot.service"
    
    if [[ ! -f "$SERVICE_FILE" ]]; then
        sudo tee "$SERVICE_FILE" > /dev/null << EOF
[Unit]
Description=Mousa Trading Bot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PWD
Environment=PATH=$PWD/venv/bin
ExecStart=$PWD/venv/bin/python app.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

        sudo systemctl daemon-reload
        log_success "Systemd service created"
        
        read -p "Enable and start the service now? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            sudo systemctl enable mousa-trading-bot.service
            sudo systemctl start mousa-trading-bot.service
            log_success "Service enabled and started"
        fi
    else
        log_warning "Systemd service already exists"
    fi
}

# Setup cron jobs
setup_cron_jobs() {
    log_info "Setting up cron jobs..."
    
    CRON_JOBS=(
        "0 3 * * * cd $PWD && $PWD/venv/bin/python scripts/utilities/database_setup.py cleanup > $PWD/logs/cron_cleanup.log 2>&1"
        "*/5 * * * * cd $PWD && $PWD/venv/bin/python scripts/data_collection/fetch_market_data.py > $PWD/logs/cron_data.log 2>&1"
        "0 2 * * * cd $PWD && $PWD/venv/bin/python scripts/analysis/backtesting.py > $PWD/logs/cron_backtest.log 2>&1"
    )
    
    # Add cron jobs
    for job in "${CRON_JOBS[@]}"; do
        (crontab -l 2>/dev/null | grep -F "$job") || (crontab -l 2>/dev/null; echo "$job") | crontab -
    done
    
    log_success "Cron jobs setup completed"
}

# Run tests
run_tests() {
    log_info "Running basic tests..."
    
    # Test Python environment
    python -c "import pandas, numpy, requests; print('Core imports successful')"
    
    # Test database connection
    if [[ -f "scripts/utilities/database_setup.py" ]]; then
        python scripts/utilities/database_setup.py
    fi
    
    # Test market data fetching
    if [[ -f "scripts/data_collection/fetch_market_data.py" ]]; then
        python scripts/data_collection/fetch_market_data.py --test
    fi
    
    log_success "Basic tests completed"
}

# Display setup summary
display_summary() {
    log_success "Mousa Trading Bot setup completed!"
    echo
    echo "=== Setup Summary ==="
    echo "✓ Python virtual environment: venv/"
    echo "✓ Dependencies installed"
    echo "✓ Directory structure created"
    echo "✓ Configuration files created"
    echo "✓ Database initialized"
    echo "✓ Logging system setup"
    echo
    echo "=== Next Steps ==="
    echo "1. Copy .env.template to .env and configure your API keys"
    echo "2. Activate virtual environment: source venv/bin/activate"
    echo "3. Run the bot: python app.py"
    echo "4. Check logs in logs/ directory"
    echo
    echo "=== Important Files ==="
    echo "• Configuration: config/ directory"
    echo "• Logs: logs/ directory"
    echo "• Data: data/ directory"
    echo "• Scripts: scripts/ directory"
    echo
}

# Main setup function
main() {
    log_info "Starting Mousa Trading Bot setup..."
    
    check_root
    detect_os
    check_python
    create_venv
    install_dependencies
    install_talib
    create_directories
    create_config_files
    setup_database
    setup_logging
    
    if [[ "$OS" == "linux" ]]; then
        setup_systemd_service
    fi
    
    setup_cron_jobs
    run_tests
    display_summary
    
    log_success "Setup completed successfully!"
}

# Run main function
main "$@"
