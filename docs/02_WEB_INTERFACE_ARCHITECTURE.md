# SEP Professional Trading System: Web Interface Architecture

**Document Version:** 2.0  
**Last Updated:** August 21, 2025  
**Target Audience:** Frontend Developers, API Integrators, System Architects  

## Overview

The SEP Professional Trading System features a modern, production-ready web interface built with **React 18** and **TypeScript**. The interface provides real-time trading operations, performance monitoring, and system configuration through a sophisticated multi-service architecture leveraging RESTful APIs and WebSocket connections.

## Frontend Architecture

### Technology Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Core Framework** | React | 18.2+ | Component-based UI framework |
| **Language** | TypeScript | 4.9+ | Type-safe JavaScript development |
| **State Management** | React Context + Hooks | Native | Centralized application state |
| **HTTP Client** | Native fetch API | ES6+ | RESTful API communication |
| **WebSocket Client** | Native WebSocket API | ES6+ | Real-time data streaming |
| **Styling** | Tailwind CSS + styled-components | Latest | Utility-first and dynamic styling |
| **Build Tool** | Create React App (react-scripts) | 5.0+ | Development and production builds |
| **Deployment** | Nginx | 1.20+ | Production web server |

### Application Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   React Application                    │
├─────────────────┬───────────────────┬───────────────────┤
│   Components    │   State Management │   Services        │
│                 │                   │                   │
│ • HomeDashboard     │ • ConfigContext   │ • ApiClient       │
│ • ManifoldVisualizer│ • SymbolContext   │                   │
│ • OandaCandleChart  │ • WebSocketContext│                   │
│                     │                   │                   │
└─────────────────┴───────────────────┴───────────────────┘
```

### Project Structure

```
frontend/
├── src/
│   ├── components/        # Reusable UI components
│   ├── context/           # React context providers
│   ├── hooks/             # Custom React hooks
│   ├── services/          # API client and service layer
│   ├── styles/            # CSS and styling resources
│   ├── utils/             # Utility helpers
│   └── index.js           # Application entry point
├── public/               # Static assets
├── nginx.conf            # Production nginx configuration
└── package.json          # Project configuration
```

## API Architecture

### RESTful API Specification

#### Base Configuration
```typescript
API Base URL: http://[host]:5000/api
Content-Type: application/json
Authentication: X-SEP-API-KEY header (for protected endpoints)
CORS Policy: Configured for frontend origins
```

#### Core Endpoints

##### System Management
```typescript
// System health and status
GET  /api/health
Response: { status: "healthy" | "unhealthy", timestamp: string }

GET  /api/status  
Response: {
  system_status: "running" | "stopped" | "error",
  uptime: number,
  memory_usage: number,
  cpu_usage: number,
  services: ServiceStatus[]
}

GET  /api/system/info
Response: {
  version: string,
  build_date: string,
  environment: string,
  features: string[]
}
```

##### Performance Analytics
```typescript
// Performance metrics and analytics
GET  /api/performance/metrics
Response: {
  total_return: number,
  daily_return: number,
  sharpe_ratio: number,
  max_drawdown: number,
  win_rate: number,
  trades_count: number
}

GET  /api/performance/current
Response: {
  current_pnl: number,
  daily_pnl: number,
  open_positions_value: number,
  available_balance: number,
  last_updated: string
}

GET  /api/performance/history
Query: { from?: string, to?: string, interval?: "1h"|"1d"|"1w" }
Response: {
  data_points: PerformanceDataPoint[],
  summary: PerformanceSummary
}
```

##### Configuration Management
```typescript
// System configuration
GET  /api/config/get
Response: {
  trading: TradingConfig,
  risk_management: RiskConfig,
  system: SystemConfig
}

POST /api/config/set
Body: { section: string, config: object }
Response: { success: boolean, message: string }

GET  /api/config/schema
Response: {
  trading_schema: ConfigSchema,
  risk_schema: ConfigSchema,
  system_schema: ConfigSchema
}
```

### WebSocket Architecture

#### Connection Management
```typescript
// WebSocket endpoints
ws://[host]:8765/market-data     // Real-time market data
ws://[host]:8765/trade-signals   // Trading signal updates  
ws://[host]:8765/system-status   // System status updates
ws://[host]:8765/performance     // Performance metric updates

// Connection lifecycle
interface WebSocketService {
  connect(endpoint: string): Promise<WebSocket>
  disconnect(): void
  subscribe(channel: string, callback: (data: any) => void): void
  unsubscribe(channel: string): void
  send(message: object): void
}
```

#### Real-Time Data Streams

##### Market Data Stream
```typescript
interface MarketDataUpdate {
  timestamp: string
  symbol: string
  bid: number
  ask: number
  spread: number
  volume: number
  trend: "up" | "down" | "stable"
}
```

##### Trading Signal Stream  
```typescript
interface TradingSignal {
  timestamp: string
  type: "buy" | "sell" | "hold"
  symbol: string
  confidence: number
  price: number
  reasoning: string
  metadata: object
}
```

##### System Status Stream
```typescript
interface SystemStatusUpdate {
  timestamp: string
  component: string
  status: "healthy" | "warning" | "error"
  message: string
  metrics: {
    cpu_usage: number
    memory_usage: number
    disk_usage: number
  }
}
```

##### Performance Stream
```typescript
interface PerformanceUpdate {
  timestamp: string
  current_pnl: number
  total_return: number
  daily_return: number
  position_count: number
  last_trade: Trade | null
}
```

## Component Architecture

### Core Components

#### HomeDashboard Component
```typescript
interface HomeDashboardProps {
  tradingStatus: TradingStatus
  performanceMetrics: PerformanceMetrics
  systemStatus: SystemStatus
}

const HomeDashboard: React.FC<HomeDashboardProps> = () => {
  // Real-time data integration
  const { tradingData } = useTradingData()
  const { performanceData } = usePerformance()
  const { systemData } = useSystemStatus()

  return (
    <DashboardLayout>
      <StatusOverview />
      <PerformanceCharts />
      <RecentTrades />
      <SystemMonitoring />
    </DashboardLayout>
  )
}
```

#### Trading Interface Component
```typescript
interface TradingInterfaceProps {
  onStartTrading: (config: TradingConfig) => Promise<void>
  onStopTrading: () => Promise<void>
  isTrading: boolean
}

const TradingInterface: React.FC<TradingInterfaceProps> = ({
  onStartTrading,
  onStopTrading,
  isTrading
}) => {
  const [config, setConfig] = useState<TradingConfig>()
  
  return (
    <TradingLayout>
      <TradingControls />
      <PositionManagement />
      <RealTimeSignals />
      <RiskManagement />
    </TradingLayout>
  )
}
```

#### Performance Analytics Component
```typescript
interface PerformanceAnalyticsProps {
  timeRange: TimeRange
  onTimeRangeChange: (range: TimeRange) => void
}

const PerformanceAnalytics: React.FC<PerformanceAnalyticsProps> = ({
  timeRange,
  onTimeRangeChange
}) => {
  const { performanceHistory } = usePerformanceHistory(timeRange)
  
  return (
    <AnalyticsLayout>
      <PerformanceCharts data={performanceHistory} />
      <MetricsSummary />
      <TradeAnalysis />
      <RiskAnalytics />
    </AnalyticsLayout>
  )
}
```

### State Management

Trading control hooks have been removed pending integration with a real trading backend. Current state management focuses on configuration data.

#### Configuration Context
```typescript
interface ConfigContextType {
  config: SystemConfig
  updateConfig: (section: string, newConfig: object) => Promise<void>
  resetConfig: () => Promise<void>
  validateConfig: (config: object) => ValidationResult
}

export const ConfigProvider: React.FC<PropsWithChildren> = ({ children }) => {
  const [config, setConfig] = useState<SystemConfig>()
  
  const updateConfig = useCallback(async (section: string, newConfig: object) => {
    try {
      const response = await api.post('/config/set', { section, config: newConfig })
      if (response.data.success) {
        setConfig(prev => ({ ...prev, [section]: newConfig }))
      }
    } catch (error) {
      throw new Error('Failed to update configuration')
    }
  }, [])
  
  return (
    <ConfigContext.Provider value={{ config, updateConfig, resetConfig, validateConfig }}>
      {children}
    </ConfigContext.Provider>
  )
}
```

## Real-Time Integration

### WebSocket Service Implementation

```typescript
class WebSocketService {
  private connections: Map<string, WebSocket> = new Map()
  private subscriptions: Map<string, Set<(data: any) => void>> = new Map()
  
  async connect(endpoint: string): Promise<WebSocket> {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(endpoint)
      
      ws.onopen = () => {
        this.connections.set(endpoint, ws)
        resolve(ws)
      }
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data)
        this.handleMessage(data)
      }
      
      ws.onerror = (error) => reject(error)
      ws.onclose = () => this.connections.delete(endpoint)
    })
  }
  
  subscribe(channel: string, callback: (data: any) => void): void {
    if (!this.subscriptions.has(channel)) {
      this.subscriptions.set(channel, new Set())
    }
    this.subscriptions.get(channel)?.add(callback)
  }
  
  private handleMessage(data: any): void {
    const { channel, payload } = data
    const callbacks = this.subscriptions.get(channel)
    callbacks?.forEach(callback => callback(payload))
  }
}
```

### Custom Hooks for Real-Time Data

```typescript
// Hook for trading data with real-time updates
export const useTradingData = () => {
  const [tradingData, setTradingData] = useState<TradingData>()
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)
  
  useEffect(() => {
    // Initial data fetch
    api.get('/trading/status')
      .then(response => setTradingData(response.data))
      .catch(setError)
      .finally(() => setLoading(false))
    
    // Real-time updates
    const ws = new WebSocketService()
    ws.connect(`${process.env.REACT_APP_WS_URL}/trade-signals`)
      .then(() => {
        ws.subscribe('trading-status', (data) => {
          setTradingData(prev => ({ ...prev, ...data }))
        })
      })
    
    return () => ws.disconnect()
  }, [])
  
  return { tradingData, loading, error }
}

// Hook for performance metrics with real-time updates
export const usePerformance = () => {
  const [metrics, setMetrics] = useState<PerformanceMetrics>()
  const [history, setHistory] = useState<PerformanceDataPoint[]>([])
  
  useEffect(() => {
    // Initial metrics fetch
    api.get('/performance/metrics').then(response => setMetrics(response.data))
    
    // Real-time performance updates
    const ws = new WebSocketService()
    ws.connect(`${process.env.REACT_APP_WS_URL}/performance`)
      .then(() => {
        ws.subscribe('metrics-update', setMetrics)
        ws.subscribe('history-update', (data) => {
          setHistory(prev => [...prev, data].slice(-1000)) // Keep last 1000 points
        })
      })
    
    return () => ws.disconnect()
  }, [])
  
  return { metrics, history }
}
```

## Security and Authentication

Authentication is not yet implemented; the frontend communicates with trusted backend services.

### CORS Configuration
```typescript
// Backend CORS configuration
const CORS_CONFIG = {
  origins: [
    'http://localhost:3000',  // Development frontend
    'http://localhost',       // Production frontend  
    'http://129.212.145.195', // Remote production
  ],
  methods: ['GET', 'POST', 'PUT', 'DELETE'],
  allowedHeaders: ['Content-Type'],
  credentials: true
}
```

## Error Handling and Resilience

Error handling is performed at call sites using standard fetch error checks.

### WebSocket Reconnection
```typescript
class ReconnectingWebSocket {
  private url: string
  private ws: WebSocket | null = null
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectDelay = 1000
  
  constructor(url: string) {
    this.url = url
    this.connect()
  }
  
  private connect(): void {
    try {
      this.ws = new WebSocket(this.url)
      
      this.ws.onopen = () => {
        this.reconnectAttempts = 0
        console.log('WebSocket connected')
      }
      
      this.ws.onclose = () => {
        this.scheduleReconnect()
      }
      
      this.ws.onerror = () => {
        this.scheduleReconnect()
      }
      
    } catch (error) {
      this.scheduleReconnect()
    }
  }
  
  private scheduleReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      setTimeout(() => {
        this.reconnectAttempts++
        this.connect()
      }, this.reconnectDelay * Math.pow(2, this.reconnectAttempts))
    }
  }
}
```

## Performance Optimization

### Code Splitting and Lazy Loading
```typescript
// Lazy-loaded route components
const HomeDashboard = lazy(() => import('./components/HomeDashboard'))
const Trading = lazy(() => import('./components/Trading'))
const Performance = lazy(() => import('./components/Performance'))
const Configuration = lazy(() => import('./components/Configuration'))

// Route configuration with lazy loading
const AppRoutes = () => (
  <Suspense fallback={<LoadingSpinner />}>
    <Routes>
      <Route path="/" element={<HomeDashboard />} />
      <Route path="/trading" element={<Trading />} />
      <Route path="/performance" element={<Performance />} />
      <Route path="/config" element={<Configuration />} />
    </Routes>
  </Suspense>
)
```

### Caching Strategies
```typescript
// API response caching
class CacheService {
  private cache: Map<string, { data: any, timestamp: number }> = new Map()
  private ttl = 60000 // 1 minute TTL
  
  set(key: string, data: any): void {
    this.cache.set(key, { data, timestamp: Date.now() })
  }
  
  get(key: string): any | null {
    const cached = this.cache.get(key)
    if (cached && Date.now() - cached.timestamp < this.ttl) {
      return cached.data
    }
    this.cache.delete(key)
    return null
  }
  
  clear(): void {
    this.cache.clear()
  }
}
```

## Production Deployment

### Build Configuration
```dockerfile
# Multi-stage Docker build for frontend
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80 443
CMD ["nginx", "-g", "daemon off;"]
```

### Nginx Configuration
```nginx
server {
    listen 80;
    listen [::]:80;
    server_name _;
    
    root /usr/share/nginx/html;
    index index.html;
    
    # Frontend routing (SPA support)
    location / {
        try_files $uri $uri/ /index.html;
    }
    
    # API proxy
    location /api/ {
        proxy_pass http://trading-backend:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
    
    # WebSocket proxy
    location /ws/ {
        proxy_pass http://websocket-service:8765;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
    
    # Health check endpoint
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
    
    # Static asset caching
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

## Testing Strategy

### Component Testing
Frontend components are exercised with React Testing Library using the real
`apiClient` implementation. This ensures tests reflect actual data flow rather
than mocked responses.

```typescript
import { render, screen } from '@testing-library/react';
import OandaCandleChart from '../components/OandaCandleChart';

test('renders OANDA candle chart header', () => {
  render(<OandaCandleChart />);
  expect(screen.getByText(/OANDA Candles/i)).toBeInTheDocument();
});
```

### API Integration Testing
The fetch-based API client can be exercised directly against a test backend or
a controlled fixture server.

```typescript
import apiClient from '../services/api';

test('fetches system status', async () => {
  const status = await apiClient.getSystemStatus();
  expect(status).toHaveProperty('status');
});
```

---

**Document Control**  
*This document defines the complete web interface architecture and should be referenced for all frontend development, API integration, and real-time feature implementations.*