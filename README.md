# SEP Professional Trading System - Web GUI

A comprehensive web-based graphical user interface for the SEP (Scalable Enterprise Platform) Professional Trading System, featuring real-time market data visualization, trading execution capabilities, and system monitoring.

## 🏗️ Architecture Overview

The system follows a modern microservices architecture with the following components:

### Backend Services
- **Trading API Service** (`scripts/trading_service.py`) - RESTful API for trading operations
- **WebSocket Service** (`scripts/websocket_service.py`) - Real-time data streaming
- **CLI Bridge** (`scripts/cli_bridge.py`) - Interface to existing SEP CLI tools
- **Redis Cache** - Session management and real-time data storage

### Frontend Application
- **React SPA** - Modern single-page application built with React 18
- **Real-time WebSocket Integration** - Live market data and system updates
- **Responsive UI Components** - Dashboard, trading panels, system monitoring
- **Nginx Reverse Proxy** - Production-ready web server with API routing

### Infrastructure
- **Docker Containerization** - All services containerized for easy deployment
- **Docker Compose Orchestration** - Multi-service deployment and management
- **Health Checks** - Comprehensive service health monitoring
- **Automated Deployment** - One-command deployment script

## 🚀 Quick Start

### Prerequisites
- Docker 20.10+ and Docker Compose
- 2GB RAM minimum, 4GB recommended
- Ports 80, 5000, 6379, 8765 available

### Installation & Deployment

1. **Clone and prepare the system:**
```bash
git clone <repository>
cd sep-trading-system
chmod +x deploy.sh
```

2. **Deploy the entire system:**
```bash
./deploy.sh start
```

3. **Access the application:**
- **Web Interface**: http://localhost
- **Backend API**: http://localhost:5000
- **WebSocket**: ws://localhost:8765

### Deployment Commands

```bash
./deploy.sh start     # Build and start all services
./deploy.sh stop      # Stop all services
./deploy.sh restart   # Restart all services
./deploy.sh status    # Show service status
./deploy.sh logs      # View all logs
./deploy.sh health    # Perform health checks
./deploy.sh clean     # Clean up containers and volumes
```

## 🎯 Key Features

### Real-Time Trading Interface
- **Live Market Data**: Real-time price feeds and market indicators
- **Order Management**: Place, modify, and cancel trading orders
- **Position Monitoring**: Track open positions and P&L
- **Trading Signals**: Algorithmic trading signal display with confidence metrics

### System Monitoring
- **Health Dashboard**: System component status monitoring
- **Performance Metrics**: CPU, memory, and trading performance indicators
- **Real-Time Logs**: Live system event and error tracking
- **Configuration Management**: Dynamic system configuration updates

### Advanced Features
- **WebSocket Integration**: Sub-second latency for critical updates
- **Responsive Design**: Works seamlessly across devices
- **Security**: API authentication and CORS protection
- **Scalability**: Microservices architecture for horizontal scaling

## 📊 API Endpoints

### Trading Operations
```http
GET  /api/market-data          # Current market data
POST /api/place-order          # Place trading order
GET  /api/positions            # Current positions
GET  /api/trading-signals      # Active trading signals
```

### System Management
```http
GET  /api/health               # System health check
GET  /api/system-status        # Detailed system status
GET  /api/performance/current  # Current performance metrics
GET  /api/config/get          # Get configuration
POST /api/config/set          # Update configuration
```

### CLI Integration
```http
POST /api/cli/execute         # Execute CLI commands
GET  /api/cli/status          # CLI bridge status
```

## 🔧 Configuration

### Environment Variables

**Backend Services:**
```bash
FLASK_ENV=production
REDIS_URL=redis://redis:6379/0
SEP_CONFIG_PATH=/app/config
```

**Frontend:**
```bash
REACT_APP_API_URL=http://localhost:5000
REACT_APP_WS_URL=ws://localhost:8765
REACT_APP_ENVIRONMENT=production
```

### Docker Configuration

The system uses Docker Compose for orchestration with:
- **Automatic Health Checks** for all services
- **Data Persistence** via Docker volumes
- **Network Isolation** with custom bridge network
- **Resource Management** with memory and CPU limits

## 🛡️ Security Features

- **API Authentication**: JWT-based authentication middleware
- **CORS Protection**: Configured for secure cross-origin requests
- **Command Whitelisting**: CLI bridge restricts unauthorized commands
- **Input Validation**: Comprehensive request validation and sanitization
- **Security Headers**: Nginx configured with security-focused headers

## 🔍 Monitoring & Debugging

### Health Checks
Each service includes comprehensive health checks:
- **Redis**: Connection and response validation
- **Backend API**: HTTP endpoint availability
- **WebSocket**: Socket connection testing
- **Frontend**: Nginx server responsiveness

### Logging
- **Structured Logs**: JSON-formatted logs for all services
- **Log Aggregation**: Centralized logging via Docker
- **Real-time Monitoring**: WebSocket-based log streaming
- **Error Tracking**: Automatic error capture and reporting

### Performance Monitoring
- **Metrics Collection**: System resource utilization tracking
- **Trading Performance**: P&L, win rates, drawdown analysis
- **Latency Monitoring**: API response times and WebSocket delays
- **Throughput Tracking**: Order execution and data processing rates

## 🏗️ Development

### Project Structure
```
├── frontend/                 # React application
│   ├── src/components/      # UI components
│   ├── src/hooks/          # Custom React hooks
│   ├── public/             # Static assets
│   └── Dockerfile          # Frontend container
├── scripts/                # Backend services
│   ├── trading_service.py  # Main API service
│   ├── websocket_service.py # WebSocket server
│   ├── cli_bridge.py       # CLI integration
│   └── requirements.txt    # Python dependencies
├── docker-compose.yml      # Service orchestration
└── deploy.sh              # Deployment script
```

### Technology Stack
- **Frontend**: React 18, WebSocket API, Modern CSS
- **Backend**: Flask, WebSockets, Redis, Gunicorn
- **Infrastructure**: Docker, Nginx, Docker Compose
- **Integration**: SEP CLI tools, Real-time data streams

## 📈 Performance Specifications

### System Requirements
- **CPU**: 2+ cores recommended for production
- **Memory**: 4GB RAM for optimal performance
- **Storage**: 10GB for logs, data, and container images
- **Network**: Low-latency connection for real-time trading

### Performance Metrics
- **API Response Time**: <100ms for standard operations
- **WebSocket Latency**: <50ms for real-time updates
- **Throughput**: 1000+ requests/second sustained
- **Uptime**: 99.9% availability with health monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support & Troubleshooting

### Common Issues

**Container Build Failures:**
```bash
./deploy.sh clean  # Clean up containers
./deploy.sh start  # Rebuild from scratch
```

**Service Not Starting:**
```bash
./deploy.sh logs SERVICE_NAME  # Check specific service logs
./deploy.sh health            # Run health diagnostics
```

**Port Conflicts:**
```bash
docker ps  # Check running containers
netstat -tulpn | grep :PORT  # Check port usage
```

### Getting Help
- Check service logs: `./deploy.sh logs`
- Verify service status: `./deploy.sh status`
- Run health checks: `./deploy.sh health`
- Review configuration files for environment-specific settings

---

**Built with ❤️ for professional trading operations**