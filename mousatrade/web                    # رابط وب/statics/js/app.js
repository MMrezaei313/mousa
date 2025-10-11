// Mousa Trading Bot - Main JavaScript Application

class MousaApp {
    constructor() {
        this.apiBase = '/api';
        this.currentUser = null;
        this.strategies = [];
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.checkAuthentication();
        this.loadInitialData();
    }

    setupEventListeners() {
        // Global error handler
        window.addEventListener('error', this.handleGlobalError.bind(this));
        
        // Network status monitoring
        window.addEventListener('online', this.handleOnline.bind(this));
        window.addEventListener('offline', this.handleOffline.bind(this));
    }

    async checkAuthentication() {
        try {
            const token = localStorage.getItem('auth_token');
            if (token) {
                const response = await this.apiRequest('/auth/verify');
                if (response.success) {
                    this.currentUser = response.user;
                    this.updateUIForUser();
                } else {
                    this.handleLogout();
                }
            }
        } catch (error) {
            console.warn('Authentication check failed:', error);
        }
    }

    async apiRequest(endpoint, options = {}) {
        const url = `${this.apiBase}${endpoint}`;
        const config = {
            headers: {
                'Content-Type': 'application/json',
            },
            ...options
        };

        // Add authentication token if available
        const token = localStorage.getItem('auth_token');
        if (token) {
            config.headers['Authorization'] = `Bearer ${token}`;
        }

        try {
            const response = await fetch(url, config);
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'API request failed');
            }

            return data;
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }

    async loadInitialData() {
        if (!this.currentUser) return;

        try {
            // Load strategies
            const strategiesResponse = await this.apiRequest('/strategies/list');
            if (strategiesResponse.success) {
                this.strategies = strategiesResponse.strategies;
                this.updateStrategiesUI();
            }

            // Load system status
            const statusResponse = await this.apiRequest('/system/status');
            if (statusResponse.success) {
                this.updateSystemStatus(statusResponse.status);
            }

        } catch (error) {
            console.error('Failed to load initial data:', error);
        }
    }

    updateUIForUser() {
        // Update UI elements based on user authentication
        const authElements = document.querySelectorAll('[data-auth]');
        authElements.forEach(element => {
            const authType = element.getAttribute('data-auth');
            if (authType === 'required' && this.currentUser) {
                element.style.display = element.dataset.display || 'block';
            } else if (authType === 'anonymous' && !this.currentUser) {
                element.style.display = element.dataset.display || 'block';
            } else {
                element.style.display = 'none';
            }
        });

        // Update user-specific content
        const userElements = document.querySelectorAll('[data-user]');
        userElements.forEach(element => {
            const userProperty = element.getAttribute('data-user');
            if (this.currentUser && this.currentUser[userProperty]) {
                element.textContent = this.currentUser[userProperty];
            }
        });
    }

    updateStrategiesUI() {
        // Update strategies list in the UI
        const strategiesContainer = document.getElementById('strategies-list');
        if (strategiesContainer) {
            strategiesContainer.innerHTML = this.strategies
                .map(strategy => this.createStrategyHTML(strategy))
                .join('');
        }
    }

    createStrategyHTML(strategy) {
        return `
            <div class="col-md-6 col-lg-4 mb-4">
                <div class="card strategy-card h-100">
                    <div class="card-body">
                        <h5 class="card-title">${strategy.name}</h5>
                        <p class="card-text text-muted">${strategy.description}</p>
                        <div class="strategy-metrics">
                            <small class="text-success">Return: ${strategy.return}%</small>
                            <small class="text-info">Sharpe: ${strategy.sharpe}</small>
                        </div>
                    </div>
                    <div class="card-footer">
                        <button class="btn btn-sm btn-primary" onclick="app.runBacktest('${strategy.id}')">
                            Backtest
                        </button>
                        <button class="btn btn-sm btn-outline-secondary" onclick="app.optimizeStrategy('${strategy.id}')">
                            Optimize
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    updateSystemStatus(status) {
        // Update system status in the UI
        const statusElements = {
            'system-version': status.version,
            'system-uptime': status.uptime,
            'system-memory': status.memory_usage,
            'system-cpu': status.cpu_usage
        };

        Object.entries(statusElements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
            }
        });
    }

    async runBacktest(strategyId) {
        try {
            this.showLoading('Running backtest...');
            
            const response = await this.apiRequest('/strategies/backtest', {
                method: 'POST',
                body: JSON.stringify({
                    strategy_id: strategyId,
                    parameters: this.getStrategyParameters(strategyId)
                })
            });

            if (response.success) {
                this.showBacktestResults(response.results);
            } else {
                this.showError('Backtest failed: ' + response.error);
            }
        } catch (error) {
            this.showError('Backtest error: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    async optimizeStrategy(strategyId) {
        try {
            this.showLoading('Optimizing strategy...');

            const response = await this.apiRequest('/optimization/run', {
                method: 'POST',
                body: JSON.stringify({
                    strategy_id: strategyId,
                    parameter_space: this.getParameterSpace(strategyId),
                    method: 'genetic'
                })
            });

            if (response.success) {
                this.showOptimizationResults(response.results);
            } else {
                this.showError('Optimization failed: ' + response.error);
            }
        } catch (error) {
            this.showError('Optimization error: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    getStrategyParameters(strategyId) {
        // Get parameters for specific strategy
        // This would be implemented based on the strategy type
        return {
            fast_window: 10,
            slow_window: 30,
            rsi_period: 14
        };
    }

    getParameterSpace(strategyId) {
        // Define parameter space for optimization
        return {
            fast_window: { min: 5, max: 50, type: 'int' },
            slow_window: { min: 20, max: 100, type: 'int' },
            rsi_period: { min: 7, max: 21, type: 'int' }
        };
    }

    showBacktestResults(results) {
        // Display backtest results in a modal or dedicated section
        const resultsHTML = `
            <div class="backtest-results">
                <h4>Backtest Results</h4>
                <div class="row">
                    <div class="col-md-6">
                        <p><strong>Total Return:</strong> ${(results.performance.total_return * 100).toFixed(2)}%</p>
                        <p><strong>Sharpe Ratio:</strong> ${results.performance.sharpe_ratio.toFixed(2)}</p>
                        <p><strong>Max Drawdown:</strong> ${(results.performance.max_drawdown * 100).toFixed(2)}%</p>
                    </div>
                    <div class="col-md-6">
                        <p><strong>Win Rate:</strong> ${(results.performance.win_rate * 100).toFixed(1)}%</p>
                        <p><strong>Total Trades:</strong> ${results.performance.total_trades}</p>
                        <p><strong>Profit Factor:</strong> ${results.performance.profit_factor.toFixed(2)}</p>
                    </div>
                </div>
            </div>
        `;

        this.showModal('Backtest Results', resultsHTML);
    }

    showOptimizationResults(results) {
        const resultsHTML = `
            <div class="optimization-results">
                <h4>Optimization Results</h4>
                <p><strong>Best Score:</strong> ${results.best_score.toFixed(4)}</p>
                <p><strong>Best Parameters:</strong></p>
                <pre>${JSON.stringify(results.best_parameters, null, 2)}</pre>
                <div class="optimization-chart">
                    <canvas id="optimizationChart" width="400" height="200"></canvas>
                </div>
            </div>
        `;

        this.showModal('Optimization Results', resultsHTML);
        this.renderOptimizationChart(results.optimization_history);
    }

    renderOptimizationChart(history) {
        const ctx = document.getElementById('optimizationChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: history.map(h => h.iteration),
                datasets: [{
                    label: 'Best Score',
                    data: history.map(h => h.score),
                    borderColor: '#667eea',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Optimization Progress'
                    }
                }
            }
        });
    }

    showModal(title, content) {
        // Create and show a modal with the given content
        const modalId = 'dynamic-modal';
        let modal = document.getElementById(modalId);
        
        if (!modal) {
            modal = document.createElement('div');
            modal.id = modalId;
            modal.className = 'modal fade';
            modal.innerHTML = `
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">${title}</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            ${content}
                        </div>
                    </div>
                </div>
            `;
            document.body.appendChild(modal);
        }

        const bsModal = new bootstrap.Modal(modal);
        bsModal.show();
    }

    showLoading(message = 'Loading...') {
        // Show loading indicator
        let loading = document.getElementById('global-loading');
        if (!loading) {
            loading = document.createElement('div');
            loading.id = 'global-loading';
            loading.className = 'loading-overlay';
            loading.innerHTML = `
                <div class="loading-content">
                    <div class="spinner-border text-primary"></div>
                    <p>${message}</p>
                </div>
            `;
            document.body.appendChild(loading);
        }
        loading.style.display = 'flex';
    }

    hideLoading() {
        const loading = document.getElementById('global-loading');
        if (loading) {
            loading.style.display = 'none';
        }
    }

    showError(message) {
        // Show error message
        const alert = document.createElement('div');
        alert.className = 'alert alert-danger alert-dismissible fade show';
        alert.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        const container = document.querySelector('.flash-messages') || document.body;
        container.appendChild(alert);
        
        setTimeout(() => {
            alert.remove();
        }, 5000);
    }

    handleGlobalError(event) {
        console.error('Global error:', event.error);
        this.showError('An unexpected error occurred');
    }

    handleOnline() {
        this.showError('Connection restored', 'success');
    }

    handleOffline() {
        this.showError('Connection lost - working offline', 'warning');
    }

    handleLogout() {
        localStorage.removeItem('auth_token');
        this.currentUser = null;
        this.updateUIForUser();
        window.location.href = '/login';
    }
}

// Initialize the application
const app = new MousaApp();

// Global utility functions
function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(amount);
}

function formatPercent(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value);
}

function formatNumber(value) {
    return new Intl.NumberFormat('en-US').format(value);
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { MousaApp, app };
}
