# SEP Trader Documentation

This directory consolidates project READMEs from across the repository into a single reference.

## README.md

# SEP Professional Trading System

**Version:** 2.0 Production  
**Build Status:** ✅ 177/177 Targets Successful  
**Deployment Status:** ✅ Production Ready with Web Interface  
**Last Updated:** August 21, 2025

## 🚀 Quick Start

```bash
# Clone and setup
git clone [repository-url]
cd sep-trader

# Local development deployment
./install.sh --minimal --no-docker
./build.sh --no-docker
./deploy.sh start    # launches Redis, trader service, WebSocket, and frontend

# Optional: run GPU metrics engine (can run remotely with REDIS_HOST/PORT)
./bin/quantum_pair_trainer &

# Access the system
# Web Interface: http://localhost
# Backend API:   http://localhost:5000
# WebSocket:     ws://localhost:8765
# Redis:         redis://localhost:6380
```

## 📋 System Overview

The **SEP Professional Trading System** is a modular trading platform composed of separate GPU and CPU services:

- **🧠 Remote GPU Metrics Engine**: C++/CUDA service that generates trading metrics
- **🖥️ CPU Trader Frontend**: React/TypeScript dashboard and CLI that operate on standard CPU hardware
- **📊 Redis Metrics Pipeline**: Streams GPU-generated metrics to the server and UI
- **⚡ Real-Time Operations**: WebSocket-based live trading and monitoring
- **🐳 Containerized Deployment**: Docker-based local and production deployment
- **☁️ Cloud-Native Architecture**: Deployable on commodity cloud instances

### Architecture Highlights

| Component | Technology | Purpose | Status |
|-----------|------------|---------|--------|
| **GPU Metrics Engine** | C++/CUDA | Remote strategy and metric computation | ✅ Operational |
| **Trader Frontend** | React/TypeScript | CPU-runnable interface and CLI | ✅ Operational |
| **API Gateway** | Python/Flask | REST API and metric ingestion | ✅ Operational |
| **Real-Time Streamer** | WebSocket/Python | Live data distribution | ✅ Operational |
| **Redis Layer** | Redis 7 | Metrics bus and session cache | ✅ Operational |

### Current Components

- **GPU Metrics Engine** – runs on dedicated GPU hardware and pushes performance metrics through Redis.
- **Trader Frontend** – React/TypeScript dashboard and CLI that interact with the API over HTTP/WebSocket.
- **API Gateway** – Python/Flask service that consumes Redis streams and exposes REST endpoints.
- **Real-Time Streamer** – Python WebSocket service broadcasting live updates to clients.
- **Redis Layer** – central cache and message bus linking GPU outputs to CPU services.

## 📚 Documentation Architecture

### Core Documentation

| Document | Purpose | Target Audience |
|----------|---------|-----------------|
| **[System Overview](docs/00_SEP_PROFESSIONAL_SYSTEM_OVERVIEW.md)** | Complete system architecture and operational status | System Architects, Technical Leadership |
| **[Deployment Guide](docs/01_DEPLOYMENT_INTEGRATION_GUIDE.md)** | Comprehensive deployment and integration procedures | DevOps Engineers, System Integrators |
| **[Web Interface Architecture](docs/02_WEB_INTERFACE_ARCHITECTURE.md)** | Frontend architecture, API specs, real-time integration | Frontend Developers, API Integrators |

## 🏗️ System Architecture

### Decoupled GPU/CPU Architecture

```
┌───────────────────────────┐
│   GPU Metrics Engine      │
│ (quantum_pair_trainer)    │
└──────────────┬────────────┘
               │ publishes metrics
               ▼
           ┌────────┐
           │ Redis  │
           └────┬───┘
                │ consumes metrics
┌───────────────▼───────────────┐
│        Trader Service         │
│  REST + WebSocket API (CPU)   │
└───────────────┬───────────────┘
                │
                ▼
┌────────────────────────────────┐
│      React Web Dashboard       │
└────────────────────────────────┘
```

### GPU/CPU Separation

The GPU metrics engine operates as an independent service and communicates with the CPU-side API and WebSocket services through Redis streams. This separation allows metrics to be generated on remote GPU hardware and pushed to the server for distribution to connected clients.

## 🚀 Deployment Options

### 1. Local Development Environment

**Recommended for:** Development, testing, backtesting, model training

```bash
# Complete local setup
./install.sh --minimal --no-docker
./build.sh --no-docker
./deploy.sh start

# Services available at:
# http://localhost      - Web Dashboard
# http://localhost:5000 - API Backend
# ws://localhost:8765   - WebSocket Service
# redis://localhost:6380 - Redis Cache
```

**Features:**
- Connects to a remote GPU metrics engine or local GPU if available
- Hot-reload development workflow
- Local data persistence
- Comprehensive debugging capabilities

### 2. Remote Production Environment

**Recommended for:** Live trading, 24/7 operations, production monitoring

```bash
# Automated droplet deployment
./scripts/deploy_to_droplet_complete.sh

# Manual production deployment
docker-compose -f docker-compose.production.yml up -d
```

**Infrastructure:**
- DigitalOcean Droplet (8GB RAM, 2 vCPUs)
- Persistent volume storage (50GB)
- Professional container orchestration
- Production-grade monitoring and health checks

## 🔧 Key Features

### Professional Trading Interface
- **Real-Time Dashboard**: Live trading metrics and performance analytics
- **Position Management**: Advanced position tracking and risk management
- **Configuration Management**: Dynamic
## alpha/README.md

# Alpha Strategy Development

This directory contains our systematic approach to improving SEP Engine's trading signal accuracy through iterative testing and refinement.

## Current Status (Updated Jan 8, 2025)

**✅ MAJOR PROGRESS - Build Fixed & Performance Enhanced:**
- **Current Performance**: 41.35% overall accuracy, 36.00% high-confidence accuracy
- **Build Status**: ✅ Fixed critical linker errors (engine.cu → engine.cpp)
- **QFH Optimization**: ✅ Parameters tuned and validated 
- **Volatility Integration**: ✅ Phase 1 volatility adaptation restored (+0.88% improvement)
- **Signal Quality**: High-confidence signal rate improved 5.1% → 6.9%

**Historical Baseline:**
- Phase 1: 50.94% accuracy (76 signals)
- Phase 2 (Previous): 40.91% accuracy (176 signals) - *regression identified*
- **Phase 3 (Current)**: 41.35% accuracy + volatility enhancement active

## Directory Structure

- `docs/` - Strategy analysis and findings documentation
- `experiments/` - Individual experiment configurations and scripts  
- `configs/` - Parameter configurations for testing
- `results/` - Test results and performance metrics

## Methodology

1. **Systematic Testing**: Each parameter change is tested and documented
2. **Performance Tracking**: Accuracy, signal count, and quality metrics recorded
3. **Regression Analysis**: When performance drops, identify root causes
4. **Iterative Improvement**: Build on successful modifications

## Key Findings So Far

### Phase 2 Regression Analysis
- **Quality Filter Impact**: Overly restrictive thresholds reduce signal count but don't improve accuracy
- **Regime Complexity**: Market regime adjustments add noise rather than value
- **Missing Components**: Phase 1's volatility adaptation was key to performance

### ✅ Completed Steps (Jan 8, 2025)
- ✅ **Build System Fixed**: Resolved engine.cu linker errors
- ✅ **QFH Parameter Optimization**: Systematic testing of k1/k2/trajectory_weight
- ✅ **Volatility Adaptation**: Successfully integrated Phase 1 volatility enhancement
- ✅ **Performance Validation**: Confirmed 41.35% baseline with improvements

### 🎯 Next Immediate Steps  
- **Pattern Vocabulary Enhancement**: Improve coherence avg 0.406 → >0.5
- **Multi-Timeframe Analysis**: Integrate M1/M5/M15 signal alignment
- **Advanced Parameter Tuning**: Test more aggressive optimization strategies
- **Target**: 45%+ overall accuracy, 45%+ high-confidence accuracy

### 📋 Latest Documentation
- `PROGRESS_SUMMARY_AUG1_2025.md` - Complete status and achievements
- `PHASE3_UNIFICATION_REPORT.md` - QFH integration details
- `QFH_TUNING_PROTOCOL.md` - Parameter optimization methodology

## frontend/readme/README.md

# Frontend Overview

This directory documents the structure of the React-based client for the SEP trading system.

## Dockerfile
Defines a containerized build that installs dependencies and serves the production build through Nginx, ensuring a consistent runtime for deployment.

## Environment Scripts
Scripts such as `env.sh` configure environment variables used during local development and image builds.

## `public/` Directory
Holds static assets like `index.html`, icons, and manifest files that are copied directly into the final build.

## `src/` Directory
Contains the React source code, including components and hooks that drive the user interface.

## Backend Dependencies
The frontend relies on backend REST APIs and WebSocket endpoints for data and live updates.

## frontend/src/components/readme/README.md

# Components Overview

## Dashboard
- **Context**: `useWebSocket` provides `connected`, `systemStatus`, `marketData`, `performanceData`, and `tradingSignals`.
- **State**: `systemInfo`, `performance`, `loading`, and `error` manage initial data load and error handling.
- **Hardcoded/Placeholders**:
  - Displays only the first four market symbols and first five signals.
  - Quick action buttons ("Start Trading", "Pause System", etc.) trigger backend operations.

## TradingPanel
- **Context**: `useWebSocket` supplies `connected`, `marketData`, and `tradingSignals`.
- **State**: `selectedSymbol` from shared context, `orderType` (`'market'`), `quantity` (`10000`), `price`, `side` (`'buy'`), `loading`, and `message`.
- **Notes**:
  - Allowed currency pairs are imported from `src/config/symbols.ts`.
  - After submitting an order, `quantity` resets to `100`.

## ConfigurationPanel
- **State**: `config`, `loading`, `saving`, and `message` drive configuration forms.
- **Hardcoded Defaults**: risk level `'medium'`, max position size `10000`, stop loss `5%`, refresh interval `30s`, debug mode `false`, log level `'INFO'`, API timeout `30s`, rate limit `60`.

## HomeDashboard
- **Context**: `useWebSocket` exposes connection state, system metrics, and live data.
- **State**: `apiHealth`, `activeTab`, `selectedPairs`, `commandHistory`, `commandInput`, and `isExecutingCommand` control dashboard interaction.
- **Hardcoded/Placeholders**:
  - `API_URL` defaults to `http://localhost:5000` if environment variable is missing.
  - Quick actions ("Upload Training Data", "Start Model Training", etc.) call backend services.

## PerformanceMetrics
- **Context**: Receives `connected` and `performanceData` from `useWebSocket`.
- **State**: `metrics`, `loading`, and `error` manage API fetch and display.

## MarketData
- **Context**: `useWebSocket` supplies live `marketData`.
- **State**: `selectedSymbol` obtained from shared context and used to highlight the active pair.

## SystemStatus
- **Context**: `useWebSocket` provides `connected` and `systemStatus`.
- **State**: `systemInfo`, `loading`, `error`, and `lastRefresh` handle polling logic.
- **Dynamic Behavior**:
  - Poll interval and component list are provided by `/api/system-status/config`.
  - Refresh cadence is configurable on the backend rather than hardcoded.

## TradingSignals
- **Context**: `useWebSocket` provides `connected` and `tradingSignals`.
- **State**: `filter` toggles between `all`, `buy`, and `sell` views.
- **Hardcoded/Placeholders**: Filter options and empty-state message strings.


## frontend/src/context/README.md

# WebSocket Context

Provides a React context for real-time communication with the trading backend via WebSocket.

## Responsibilities
- Establishes and maintains the WebSocket connection.
- Shares market data, system status, signals, and performance metrics across components.
- Manages reconnect logic, channel subscriptions, and heartbeat pings.

## Configuration
- `REACT_APP_WS_URL`: WebSocket endpoint (default `ws://localhost:8765`).
- `maxReconnectAttempts`: how many times to retry connecting (default `10`).
- `reconnectDelay`: delay in ms between reconnection attempts (default `3000`).

## frontend/src/services/README.md

# Services

Contains the API client used for all HTTP requests to the backend.

## Responsibilities
- Wraps fetch calls for authentication, market data, trading operations, and configuration.
- Manages an auth token and attaches it to request headers.

## Configuration
- `REACT_APP_API_URL`: base URL for API requests (default `http://localhost:5000`).
- `auth_token`: persisted in `localStorage` and used for `Authorization` headers.

## frontend/src/styles/README.md

# Styles

CSS modules defining the look and feel of the frontend.

## Responsibilities
- Declares global variables for colors, typography, and layout.
- Supplies component-specific styles for the dashboard, system status, and trading panels.
- Includes utility classes for grids, flex layouts, and badges.

## Configuration
- Customize theme via CSS variables such as `--primary-color` and `--background` in `index.css`.
- Media queries provide responsive design and optional dark mode.

## frontend/src/utils/README.md

# Utils

Hosts a complete API client for interacting with various backend endpoints.

## Responsibilities
- Sends REST requests for system status, trading actions, performance metrics, and configuration changes.
- Provides helper methods for pairs management, command execution, and data reload.

## Configuration
- `REACT_APP_API_URL`: base URL for all requests (default `http://localhost:5000`).
- Default headers include `Content-Type: application/json`; additional headers can be supplied per request.

## pitch/README.md

# SEP Dynamics - Investor Pitch Package
**Complete Investment Materials for Series A Funding**

---

## **📁 Package Overview**

This folder contains a comprehensive investment package for SEP Dynamics' **$15M Series A funding round**. Our patent-pending Quantum Field Harmonics (QFH) technology achieves **60.73% prediction accuracy** in live forex trading, representing a breakthrough in algorithmic trading.

**Investment Opportunity Summary:**
- **Raising:** $15M Series A
- **Valuation:** $85M pre-money, $100M post-money
- **Market Opportunity:** $7.4T daily forex market, $200B algorithmic trading software
- **Technology:** Patent-pending QFH system with proven live performance
- **Performance:** 60.73% accuracy vs ~55% industry best

---

## **📋 Document Index**

### **Core Investment Materials**

**[EXECUTIVE_SUMMARY.md](./EXECUTIVE_SUMMARY.md)**
- Complete business overview in executive format
- Key metrics, technology highlights, and investment thesis
- **Use:** First document to share with potential investors

**[PITCH_DECK.md](./PITCH_DECK.md)**
- 16-slide presentation covering full investment story
- Problem, solution, market, team, financials, and ask
- **Use:** Primary presentation for investor meetings

**[FINANCIAL_MODEL.md](./FINANCIAL_MODEL.md)**
- Detailed 5-year financial projections and business model
- Revenue streams, unit economics, and profitability analysis
- **Use:** Financial due diligence and modeling validation

### **Technical & Market Materials**

**[TECHNOLOGY_OVERVIEW.md](./TECHNOLOGY_OVERVIEW.md)**
- Deep dive into patent-pending QFH technology
- Technical architecture, competitive advantages, and roadmap
- **Use:** Technical due diligence and differentiation

**[MARKET_ANALYSIS.md](./MARKET_ANALYSIS.md)**
- Comprehensive market opportunity and competitive landscape
- Customer segmentation and go-to-market strategy
- **Use:** Market validation and opportunity sizing

### **Legal & Process Materials**

**[INVESTMENT_TERMS.md](./INVESTMENT_TERMS.md)**
- Series A term sheet with detailed investment structure
- Governance, rights, and exit strategy provisions
- **Use:** Term sheet negotiation and legal framework

**[DUE_DILIGENCE_MATERIALS.md](./DUE_DILIGENCE_MATERIALS.md)**
- Complete checklist of available due diligence materials
- Data room organization and access procedures
- **Use:** Due diligence process management

---

## **🎯 Investment Highlights**

### **Proven Technology**
✅ **60.73% Live Trading Accuracy** - Beats industry gold standard  
✅ **$50K+ Daily Profits** - Consistent profitability across 16+ currency pairs  
✅ **Patent Protection** - Filed August 3, 2025 (584961162ABX)  
✅ **Sub-millisecond Processing** - Real-time CUDA-accelerated analysis  

### **Massive Market Opportunity**
✅ **$7.4T Daily Market** - Global forex trading volume  
✅ **$200B Software Market** - Algorithmic trading technology  
✅ **60%+ Failure Rate** - Traditional systems create opportunity  
✅ **25% Annual Growth** - Rapidly expanding market  

### **Strong Financial Model**
✅ **$275M Revenue by 2029** - 158% 5-year CAGR  
✅ **85% Gross Margins** - Software scalability  
✅ **15:1 LTV/CAC Ratio** - Efficient customer economics  
✅ **20x+ Investor Returns** - Attractive exit scenarios  

### **Experienced Leadership**
✅ **Technical Founder** - Inventor of breakthrough technology  
✅ **Proven Execution** - Live system generating profits  
✅ **Industry Recognition** - Featured in Mark Rober's GlitterBomb 2.0  
✅ **Engineering Background** - Deep technical expertise  

---

## **📞 Next Steps for Investors**

### **Initial Interest**
1. **Review Executive Summary** - [EXECUTIVE_SUMMARY.md](./EXECUTIVE_SUMMARY.md)
2. **Schedule Initial Call** - alex@sepdynamics.com
3. **NDA Execution** - Required for detailed materials
4. **Pitch Presentation** - Full presentation and demo

### **Due Diligence Process**
1. **Data Room Access** - Complete documentation review
2. **Technology Demonstration** - Live trading system demo
3. **Reference Calls** - Customer and technical references
4. **Financial Validation** - Model review and verification

### **Investment Process**
1. **Term Sheet Negotiation** - Based on [INVESTMENT_TERMS.md](./INVESTMENT_TERMS.md)
2. **Legal Documentation** - Purchase agreement preparation
3. **Final Due Diligence** - Comprehensive third-party review
4. **Closing** - Funds transfer and board establishment

---

## **💡 Key Questions Answered**

### **Technology Questions**
- **"How does QFH work?"** → See [TECHNOLOGY_OVERVIEW.md](./TECHNOLOGY_OVERVIEW.md)
- **"What's your competitive advantage?"** → Patent protection + 60.73% accuracy
- **"Is this proven?"** → Yes, live trading with $50K+ daily profits
- **"Can this scale?"** → Enterprise architecture with CUDA acceleration

### **Market Questions**
- **"How big is the opportunity?"** → See [MARKET_ANALYSIS.md](./MARKET_ANALYSIS.md)
- **"Who are the competitors?"** → Renaissance (~55%), Two Sigma (~48%)
- **"What's your go-to-market?"** → Direct sales to hedge funds and institutions
- **"Why now?"** → First-mover advantage in quantum-inspired finance

### **Financial Questions**
- **"What's the business model?"** → See [FINANCIAL_MODEL.md](./FINANCIAL_MODEL.md)
- **"How do you make money?"** → SaaS + API + Enterprise + IP licensing
- **"What are the unit economics?"** → 15:1 LTV/CAC, 85% gross margins
- **"What's the exit strategy?"** → Strategic acquisition ($2-5B) or IPO

### **Investment Questions**
- **"What are the terms?"** → See [INVESTMENT_TERMS.md](./INVESTMENT_TERMS.md)
- **"What's the use of funds?"** → Team (40%), Product (27%), Sales (20%)
- **"What's the timeline?"** → 18 months to $10M ARR
- **"What's my return?"** → 20-40x potential return over 5-7 years

---

## **🔐 Confidentiality Notice**

**Important:** This pitch package contains confidential and proprietary information about SEP Dynamics' business, technology, and financial projections. 

**Access Requirements:**
- Executed Mutual Non-Disclosure Agreement (NDA)
- Verified accredited investor status
- Legitimate investment interest

**Restrictions:**
- No distribution without written consent
- No use for competitive purposes
- Return or destroy upon request
- Watermarked documents for tracking

---

## **📞 Contact Information**

### **Company**
**SEP Dynamics, Inc.**  
**Alexander J Nagy, Founder & CEO**  
📧 alex@sepdynamics.com  
🌐 [sepdynamics.com](https://sepdynamics.com)  
💼 [LinkedIn](https://linkedin.com/in/alexanderjnagy)  

### **Investment Inquiries**
For qualified investors interested in participating in our Series A round:
1. **Email:** alex@sepdynamics.com
2. **Subject Line:** "Series A Investment Interest - [Your Fund Name]"
3. **Include:** Brief fund overview and investment thesis fit
4. **Response Time:** 24-48 hours for qualified inquiries

### **Data Room Access**
**Secure Virtual Data Room**  
Access provided upon NDA execution and investor qualification  
**Administrator:** [Contact information provided separately]  

---

## **📈 Performance Dashboard**

### **Live System Metrics (Updated August 2025)**
- **Accuracy:** 60.73% (high-confidence signals)
- **Daily Profit:** $50,000+ average
- **Uptime:** 99.9%+ (24/7 operation)
- **Pairs Trading:** 16+ simultaneous currencies
- **Processing Speed:** <1ms signal generation

### **Business Metrics**
- **Patent Status:** Filed and under examination
- **Team Size:** 1 (founder) → scaling to 25+
- **Revenue Target:** $10M ARR by month 18
- **Customer Pipeline:** 50+ qualified prospects

---

*Last Updated: August 2025*  
*This package represents current information and forward-looking projections. All materials subject to confidentiality agreements and investor qualification requirements.*

---

## **📋 Document Checklist for Investors**

Before your first meeting, please review:
- [ ] [Executive Summary](./EXECUTIVE_SUMMARY.md) - Business overview
- [ ] [Pitch Deck](./PITCH_DECK.md) - Complete presentation
- [ ] [Market Analysis](./MARKET_ANALYSIS.md) - Market opportunity

For detailed due diligence:
- [ ] [Financial Model](./FINANCIAL_MODEL.md) - Complete financial projections
- [ ] [Technology Overview](./TECHNOLOGY_OVERVIEW.md) - Technical deep dive
- [ ] [Investment Terms](./INVESTMENT_TERMS.md) - Deal structure
- [ ] [Due Diligence Materials](./DUE_DILIGENCE_MATERIALS.md) - Process overview

**Estimated Review Time:** 2-3 hours for initial materials, 8-12 hours for complete package

## public/README.md

# SEP Dynamics Public Website

This is the commercial website for SEP Dynamics, showcasing our revolutionary Quantum Field Harmonics (QFH) technology for financial markets.

## Overview

The website presents:
- Quantum Field Harmonics technology overview
- Live trading performance metrics (60.73% accuracy)
- Patent portfolio information (Application #584961162ABX)
- Company information and leadership
- Investment and partnership opportunities

## Structure

```
public/
├── index.html          # Main website page
├── css/
│   └── main.css       # Styles with quantum-inspired design
├── js/
│   └── main.js        # Interactive features and animations
├── assets/
│   ├── favicon.ico    # Website favicon
│   ├── favicon.svg    # SVG favicon source
│   └── images/
│       └── sep-logo.svg  # Animated company logo
├── docs/
│   ├── technology/    # Technical documentation
│   ├── results/       # Performance metrics
│   └── patent/        # Patent portfolio details
└── commercial_package/
    └── INVESTOR_PRESENTATION.md  # Investment deck
```

## Key Features

- **Responsive Design**: Optimized for all devices
- **Dark Theme**: Professional quantum-inspired aesthetic
- **Animated Elements**: Smooth transitions and visual effects
- **Performance Optimized**: Fast loading with lazy loading
- **SEO Ready**: Proper meta tags and structure

## Technology Stack

- HTML5 with semantic markup
- CSS3 with custom properties and animations
- Vanilla JavaScript (no dependencies)
- SVG graphics with animations
- Google Fonts (Inter, JetBrains Mono)

## Deployment

The website is designed to be deployed on any static hosting service:

1. **GitHub Pages**: Push to `gh-pages` branch
2. **Netlify**: Connect repository and deploy
3. **Vercel**: Import project and deploy
4. **Traditional Hosting**: Upload all files via FTP

## Domain

The website is configured for `sepdynamics.com` (see CNAME file).

## Performance

- Lighthouse Score: 95+ (estimated)
- Page Load Time: <2s on 3G
- First Contentful Paint: <1s
- No external dependencies (except fonts)

## Browser Support

- Chrome/Edge: Full support
- Firefox: Full support
- Safari: Full support
- Mobile browsers: Fully responsive

## Contact

For inquiries about the website or technology:
- Email: alex@sepdynamics.com
- LinkedIn: [Alexander J Nagy](https://www.linkedin.com/in/alexanderjnagy)

## License

© 2025 SEP Dynamics, Inc. All rights reserved.
SEP Engine™ and Quantum Field Harmonics™ are trademarks of SEP Dynamics, Inc.
## _sep/testbed/README.md

# SEP Testbed

Experimental utilities and backtesting helpers live here. Define the
`SEP_BACKTESTING` macro at compile time to enable associated stubs in the main
codebase. Production builds should ignore this directory.

## src/app/readme/README.md

# Application Services Overview

## DataAccessService
**Files:** `DataAccessService.h`, `DataAccessService.cpp`

**Key functions**
- `bool isReady() const`
- `Result<std::string> storeObject(const std::string& collection, const std::map<std::string, std::any>& data, const std::string& id = "")`
- `Result<std::map<std::string, std::any>> retrieveObject(const std::string& collection, const std::string& id)`
- `Result<void> updateObject(const std::string& collection, const std::string& id, const std::map<std::string, std::any>& data)`
- `Result<void> deleteObject(const std::string& collection, const std::string& id)`
- `Result<std::vector<std::map<std::string, std::any>>> queryObjects(const std::string& collection, const std::vector<QueryFilter>& filters = {}, const std::vector<SortSpec>& sortSpecs = {}, int limit = 0, int skip = 0)`
- `Result<int> countObjects(const std::string& collection, const std::vector<QueryFilter>& filters = {})`
- `Result<std::shared_ptr<TransactionContext>> beginTransaction()`
- `Result<void> executeTransaction(std::function<Result<void>(std::shared_ptr<TransactionContext>)> operations)`
- `int registerChangeListener(const std::string& collection, std::function<void(const std::string&, const std::string&)> callback)`
- `Result<void> unregisterChangeListener(int subscriptionId)`
- `Result<void> createCollection(const std::string& collection, const std::string& schema = "")`
- `Result<void> deleteCollection(const std::string& collection)`
- `Result<std::vector<std::string>> getCollections()`

**TODO / Mock notes**
- Filtering currently supports only basic equality; full query support remains a TODO.

## MemoryTierService
**Files:** `MemoryTierService.h`, `MemoryTierService.cpp`

**Key functions**
- `Result<sep::memory::MemoryBlock*> allocate(std::size_t size, sep::memory::MemoryTierEnum tier)`
- `Result<void> deallocate(sep::memory::MemoryBlock* block)`
- `Result<sep::memory::MemoryBlock*> findBlockByPtr(void* ptr)`
- `Result<sep::memory::MemoryTier*> getTier(sep::memory::MemoryTierEnum tier)`
- `Result<float> getTierUtilization(sep::memory::MemoryTierEnum tier)`
- `Result<float> getTierFragmentation(sep::memory::MemoryTierEnum tier)`
- `Result<float> getTotalUtilization()`
- `Result<float> getTotalFragmentation()`
- `Result<void> defragmentTier(sep::memory::MemoryTierEnum tier)`
- `Result<void> optimizeBlocks()`
- `Result<void> optimizeTiers()`
- `Result<sep::memory::MemoryBlock*> promoteBlock(sep::memory::MemoryBlock* block)`
- `Result<sep::memory::MemoryBlock*> demoteBlock(sep::memory::MemoryBlock* block)`
- `Result<sep::memory::MemoryBlock*> updateBlockMetrics(sep::memory::MemoryBlock* block, float coherence, float stability, uint32_t generation, float contextScore)`
- `Result<std::string> getMemoryAnalytics()`
- `Result<std::string> getMemoryVisualization()`
- `Result<void> configureTierPolicies(const sep::memory::MemoryThresholdConfig& config)`
- `Result<void> optimizeRedisIntegration(int optimizationLevel)`
- `Result<std::string> allocateBlock(uint64_t size, MemoryTierLevel tier, const std::string& contentType, const std::vector<uint8_t>& initialData = {}, const std::map<std::string, std::string>& tags = {})`
- `Result<void> deallocateBlock(const std::string& blockId)`
- `Result<void> storeData(const std::string& blockId, const std::vector<uint8_t>& data, uint64_t offset = 0)`
- `Result<std::vector<uint8_t>> retrieveData(const std::string& blockId, uint64_t size, uint64_t offset = 0)`
- `Result<MemoryBlockMetadata> getBlockMetadata(const std::string& blockId)`
- `Result<void> moveBlockToTier(const std::string& blockId, MemoryTierLevel destinationTier, const std::string& reason = "Manual transition")`
- `Result<TierStatistics> getTierStatistics(MemoryTierLevel tier)`
- `Result<std::map<MemoryTierLevel, TierStatistics>> getAllTierStatistics()`
- `Result<void> configureTier(MemoryTierLevel tier, uint64_t totalCapacity, const std::map<std::string, std::string>& policies = {})`
- `Result<void> optimizeTiers(bool aggressive = false)`
- `int registerTransitionCallback(std::function<void(const TierTransitionRecord&)> callback)`
- `Result<void> unregisterTransitionCallback(int subscriptionId)`
- `Result<std::vector<TierTransitionRecord>> getTransitionHistory(int maxRecords = 100)`
- `Result<std::vector<MemoryAccessPattern>> getAccessPatterns(uint32_t minFrequency = 5)`

**TODO / Mock notes**
- No outstanding TODOs noted.

## PatternRecognitionService
**Files:** `PatternRecognitionService.h`, `PatternRecognitionService.cpp`

**Key functions**
- `bool isReady() const`
- `Result<std::string> registerPattern(const Pattern& pattern)`
- `Result<Pattern> getPattern(const std::string& patternId)`
- `Result<void> updatePattern(const std::string& patternId, const Pattern& pattern)`
- `Result<void> deletePattern(const std::string& patternId)`
- `Result<PatternClassification> classifyPattern(const Pattern& pattern)`
- `Result<std::vector<PatternMatch>> findSimilarPatterns(const Pattern& pattern, int maxResults = 10, float minScore = 0.7f)`
- `Result<PatternEvolution> getPatternEvolution(const std::string& patternId)`
- `Result<void> addEvolutionStage(const std::string& patternId, const Pattern& newStage)`
- `Result<std::vector<PatternCluster>> clusterPatterns(const std::vector<std::string>& patternIds = {}, int numClusters = 0)`
- `Result<float> calculateCoherence(const Pattern& pattern)`
- `Result<float> calculateStability(const Pattern& pattern)`
- `int registerChangeListener(std::function<void(const std::string&, const Pattern&)> callback)`
- `Result<void> unregisterChangeListener(int subscriptionId)`

**TODO / Mock notes**
- No outstanding TODOs noted.

## QuantumProcessingService
**Files:** `QuantumProcessingService.h`, `QuantumProcessingService.cpp`

**Key functions**
- `bool isReady() const`
- `Result<BinaryStateVector> processBinaryStateAnalysis(const QuantumState& state)`
- `Result<std::vector<QuantumFourierComponent>> applyQuantumFourierHierarchy(const QuantumState& state, int hierarchyLevels)`
- `Result<CoherenceMatrix> calculateCoherence(const QuantumState& state)`
- `Result<StabilityMetrics> determineStability(const QuantumState& state, const std::vector<QuantumState>& historicalStates)`
- `Result<QuantumState> evolveQuantumState(const QuantumState& state, const std::map<std::string, double>& evolutionParameters)`
- `Result<QuantumState> runQuantumPipeline(const QuantumState& state)`
- `std::map<std::string, std::string> getAvailableAlgorithms() const`

**TODO / Mock notes**
- Initialization check for `runQuantumPipeline` is temporarily skipped to avoid diamond inheritance issues.

## TradingLogicService
**Files:** `TradingLogicService.h`, `TradingLogicService.cpp`

**Key functions**
- `Result<void> processMarketData(const MarketDataPoint& dataPoint)`
- `Result<void> processMarketDataBatch(const std::vector<MarketDataPoint>& dataPoints)`
- `Result<OHLCVCandle> updateOHLCVCandle(const std::string& symbol, TradingTimeframe timeframe, const MarketDataPoint& dataPoint)`
- `Result<std::vector<OHLCVCandle>> getHistoricalCandles(const std::string& symbol, TradingTimeframe timeframe, int count, std::chrono::system_clock::time_point endTime)`
- `Result<std::vector<TradingSignal>> generateSignals(const MarketContext& context, const std::vector<std::string>& patternIds)`
- `Result<std::vector<TradingSignal>> generateSignalsFromPatterns(const std::vector<std::shared_ptr<Pattern>>& patterns, const MarketContext& context)`
- `Result<std::vector<TradingDecision>> makeDecisions(const std::vector<TradingSignal>& signals, const MarketContext& context)`
- `Result<PerformanceMetrics> evaluatePerformance(const std::vector<TradingDecision>& decisions, const MarketContext& currentContext)`
- `Result<PerformanceMetrics> backtestStrategy(const std::map<std::string, std::vector<OHLCVCandle>>& historicalData, const std::map<std::string, double>& parameters)`
- `int registerSignalCallback(std::function<void(const TradingSignal&)> callback)`
- `Result<void> unregisterSignalCallback(int subscriptionId)`
- `std::map<std::string, std::string> getAvailableStrategies() const`
- `Result<MarketContext> getCurrentMarketContext() const`

**TODO / Mock notes**
- No outstanding TODOs noted.

## MarketModelCache
**Files:** `market_model_cache.hpp`, `market_model_cache.cpp`

**Key functions**
- `bool ensureCacheForLastWeek(const std::string& instrument = "EUR_USD")`
- `const std::map<std::string, sep::trading::QuantumTradingSignal>& getSignalMap() const`
- `bool loadCache(const std::string& filepath)`
- `bool saveCache(const std::string& filepath) const`
- `void processAndCacheData(const std::vector<Candle>& raw_candles, const std::string& filepath)`
- `std::string getCacheFilepathForLastWeek(const std::string& instrument) const`

**TODO / Mock notes**
- Signal generation uses a placeholder based on simple price movement; integrate the full quantum pipeline.

## EnhancedMarketModelCache
**Files:** `enhanced_market_model_cache.hpp`, `enhanced_market_model_cache.cpp`

**Key functions**
- `bool ensureEnhancedCacheForInstrument(const std::string& instrument, TimeFrame timeframe = TimeFrame::M1)`
- `ProcessedSignal generateCorrelationEnhancedSignal(const std::string& target_asset, const std::string& timestamp)`
- `void updateCorrelatedAssets(const std::string& primary_asset)`
- `void optimizeCacheHierarchy()`
- `bool loadEnhancedCache(const std::string& filepath)`
- `bool saveEnhancedCache(const std::string& filepath) const`
- `CrossAssetCorrelation calculateCrossAssetCorrelation(const std::string& primary_asset, const std::vector<std::string>& correlated_assets)`
- `double calculatePairwiseCorrelation(const std::vector<double>& asset1_prices, const std::vector<double>& asset2_prices, std::chrono::milliseconds& optimal_lag)`
- `const std::unordered_map<std::string, CacheEntry>& getCacheEntries() const`
- `std::vector<ProcessedSignal> getCorrelationEnhancedSignals(const std::string& instrument) const`
- `CachePerformanceMetrics getPerformanceMetrics() const`

**TODO / Mock notes**
- No outstanding TODOs noted.

## MultiAssetSignalFusion
**Files:** `multi_asset_signal_fusion.hpp`, `multi_asset_signal_fusion.cpp`

**Key functions**
- `FusedSignal generateFusedSignal(const std::string& target_asset)`
- `std::vector<std::string> getCorrelatedAssets(const std::string& target_asset)`
- `CrossAssetCorrelation calculateDynamicCorrelation(const std::string& asset1, const std::string& asset2)`
- `double calculateCrossAssetBoost(const sep::trading::QuantumIdentifiers& signal, const CrossAssetCorrelation& correlation)`
- `FusedSignal fuseSignals(const std::vector<AssetSignal>& asset_signals)`
- `std::vector<double> calculateCorrelationMatrix(const std::vector<std::string>& assets)`
- `double calculateCrossAssetCoherence(const std::vector<AssetSignal>& signals)`
- `void updateCorrelationCache()`
- `void invalidateCorrelationCache()`
- `void logFusionDetails(const FusedSignal& signal)`
- `std::string serializeFusionResult(const FusedSignal& signal)`

**TODO / Mock notes**
- Correlation calculations currently use default values; historical data fetching via connector is a TODO.

## MarketRegimeAdaptiveProcessor
**Files:** `market_regime_adaptive.hpp`, `market_regime_adaptive.cpp`

**Key functions**
- `AdaptiveThresholds calculateRegimeOptimalThresholds(const std::string& asset)`
- `MarketRegime detectCurrentRegime(const std::string& asset)`
- `VolatilityLevel calculateVolatilityLevel(const std::vector<Candle>& data)`
- `TrendStrength calculateTrendStrength(const std::vector<Candle>& data)`
- `LiquidityLevel calculateLiquidityLevel(const std::string& asset)`
- `NewsImpactLevel calculateNewsImpact()`
- `QuantumCoherenceLevel calculateQuantumCoherence(const std::vector<Candle>& data)`
- `AdaptiveThresholds adaptThresholdsForRegime(const MarketRegime& regime)`
- `double calculateVolatilityAdjustment(VolatilityLevel volatility)`
- `double calculateTrendAdjustment(TrendStrength trend)`
- `double calculateLiquidityAdjustment(LiquidityLevel liquidity)`
- `double calculateNewsAdjustment(NewsImpactLevel news)`
- `double calculateCoherenceAdjustment(QuantumCoherenceLevel coherence)`
- `double calculateATR(const std::vector<Candle>& data, int periods = 14)`
- `double calculateRSI(const std::vector<Candle>& data, int periods = 14)`
- `double calculateSMA(const std::vector<Candle>& data, int periods)`
- `bool isLondonSession()`
- `bool isNewYorkSession()`
- `bool isTokyoSession()`
- `void logRegimeDetails(const MarketRegime& regime, const AdaptiveThresholds& thresholds)`
- `std::string serializeRegimeData(const MarketRegime& regime, const AdaptiveThresholds& thresholds)`
- `void updateRegimeCache(const std::string& asset)`
- `void invalidateRegimeCache()`

**TODO / Mock notes**
- No outstanding TODOs noted.

## Health Monitor (C interface)
**Files:** `health_monitor_c_wrapper.h`, `health_monitor_c_wrapper.cpp`

**Key functions**
- `int c_health_monitor_init()`
- `int c_health_monitor_get_status(CHealthStatus* status)`
- `void c_health_monitor_cleanup()`

**TODO / Mock notes**
- Lightweight C-style wrapper intended for integration; no TODOs currently documented.


## src/common/readme/README.md

# Common Headers

Shared headers live here to coordinate symbol handling across modules.

## Namespace Protection
- `namespace_protection.hpp` guards standard names like `std`, `string` and
  `cout` from macro pollution.
- Include it before and after any third‑party headers that redefine core
  symbols to push/restore the original macros.

## src/core/readme/README.md

# Core Component Overview

This document summarizes the intent of major components in `src/core` and highlights open
areas for future development. Notes on expected interactions with the `app` layer and
`cuda` utilities are included for context.

## cli_commands
- **Intent:** expose training operations through a command‑line interface.
- **Status:** current file is a stub that only logs actions.
- **TODO:** replace stub with real command dispatch and error handling.
- **Interactions:** invoked by `app` CLI utilities and expected to trigger CUDA‑backed
  training through other core managers.

## dynamic_config_manager
- **Intent:** provide a runtime configuration store sourced from files,
  environment variables, or command‑line arguments.
- **Status:** most getters/setters return defaults; load/save routines are not implemented.
- **TODO:** implement persistence, source tracking, and change callbacks.
- **Interactions:** supplies configuration to both `app` services and CUDA modules
  (e.g., GPU settings).

## kernel_implementations
- **Intent:** host‑side wrappers that launch CUDA kernels such as QBSA and QSH.
- **Status:** validates parameters then zeroes buffers; real kernel launches are missing.
- **TODO:** wire in actual CUDA kernels and comprehensive error handling.
- **Interactions:** called by higher‑level training code in `app` and core, bridging to
  low‑level `cuda` routines.

## Training Managers
### training_session_manager
- **Intent:** manage the lifecycle of a single training session for a currency pair.
- **Status:** placeholder start/end methods.
- **TODO:** initialize session state, manage coherence targets, and finalize metrics.
- **Interactions:** orchestrated by `app` services and coordinates CUDA training kernels.

### training_coordinator (Orchestrator)
- **Intent:** coordinate data fetching, feature encoding, model training/evaluation,
  registry persistence, and optional remote pushes.
- **Status:** functional sequential pipeline; limited logging and parallelism.
- **TODO:** enhance error reporting, parallel execution, and recovery logic.
- **Interactions:** invoked by CLI or service layers in `app` and delegates to trainers and
  evaluators that may leverage CUDA.

## dynamic_pair_manager
- **Intent:** manage runtime‑enabled trading pairs and enforce resource limits.
- **Status:** basic validation with placeholder resource checks.
- **TODO:** implement detailed resource requirement validation based on configuration.
- **Interactions:** used by `app` to control active pairs and to ensure resources are
  available before launching CUDA workloads.

## Interactions with `app` and `cuda`
Core modules expose APIs consumed by the `app` layer. CUDA utilities provide GPU
acceleration for compute‑intensive routines. Components such as CLI commands, dynamic
configuration, and training managers act as the glue connecting user‑facing `app`
services with CUDA implementations.


## src/cuda/readme/README.md

# CUDA Kernels Overview

This directory summarizes GPU entry points used across the trading engine.
Each entry point is a host function that configures and launches a device
kernel located in the surrounding `src/cuda` sources.

## Kernel entry points

- `launchAnalyzeBitPatternsKernel` – analyzes bit windows and returns
  coherence, stability and entropy metrics. Current implementation includes
  placeholder heuristics and stubbed helpers for trend detection, coherence
  and stability calculations.
- `launchQBSAKernel` – wraps `qbsa_kernel` for quantum bit state alignment
  by comparing expected vs. observed bitfields.
- `launchQSHKernel` – dispatches `qsh_kernel` to measure symmetry heuristics
  across 64‑bit chunks.
- `launchQFHBitTransitionsKernel` – evaluates forward‑window bit transitions
  through `qfhKernel`, producing damped coherence and stability scores.
- `launchProcessPatternKernel` – runs `processPatternKernel` to evolve and
  interact pattern attributes.
- `launchMultiPairProcessingKernel` – CUDA façade that forwards work to the
  core multi‑pair processing kernel.
- `launchTickerOptimizationKernel` – front‑end for ticker optimization. The
  device `optimization_kernel` only performs a simple gradient step and is
  intended as a placeholder for a more sophisticated optimizer.

## Additional kernels

`similarity_kernel` and `blend_kernel` exist in `quantum_kernels.cu` but are
not wired to public launchers. They are kept as references for future
embedding comparison and context blending work.


## src/io/readme/README.md

# IO Connectors Overview

Modules under `src/io` provide external interfaces for data ingress and
language bindings.

## OANDA connector

Files: `oanda_connector.*`, `oanda_constants.h`

- Handles REST and streaming communication with the OANDA trading platform
  using libcurl.
- Features historical candle retrieval, real‑time price streaming, account
  and order queries, plus basic order placement.
- Implements caching, ATR‑based volatility assessment and simple candle
  validation.
- Helper `MarketDataConverter` transforms OANDA structures to byte streams
  and bitstreams for CUDA processing.
- **Mock/placeholder aspects:** timestamp hashing and simplified
  normalization in `MarketDataConverter`, minimal optimization in ATR and
  volatility calculations, and order tracking utilities that expose limited
  error handling.

## C API

Files: `sep_c_api.*`, `sep.pc.in`

- Exposes the DSL interpreter through a C interface for embedding or
  scripting from non‑C++ languages.
- Supports executing source strings or files, fetching variable values and
  querying interpreter state.
- `sep.pc.in` supplies pkg‑config metadata for downstream build systems.
- **Mock/placeholder aspects:** the API targets alpha functionality and does
  not currently manage asynchronous execution or advanced type conversions.


## src/readme/README.md

# Source Directory Overview

The `src` folder hosts the project's primary code modules:

- **app** – Application services, command-line interfaces, and trading logic orchestrators.
- **core** – Engine components for pattern analysis, training, and system coordination.
- **cuda** – GPU kernels and CUDA utilities for accelerated computation.
- **io** – Connectors for market data, external APIs, and C interface bindings.
- **util** – Shared utilities, helpers, and infrastructure support.
- **common** – Minimal shared headers and scaffolding.

Several submodules contain stubbed or placeholder implementations pending full development.

## src/util/readme/README.md

# Utility Helpers

This directory documents core helpers powering the SEP DSL environment.

## Compiler
- `compiler.*` converts parsed DSL into bytecode via `BytecodeProgram` and
  `CompiledProgram` wrappers.
- Simplified AST stubs underpin compilation; stream mocks previously used for
  testing have been removed to keep behaviour deterministic.

## Interpreter
- `interpreter.*` executes AST nodes against a scoped `Environment` and exposes
  built‑in functions via a registration map.
- Module loading and export tracking remain TODO items and some legacy function
  hooks persist until the built‑ins are fully migrated.

## Memory Tiers
- `memory_tier*` and `memory_tier_manager*` provide a tiered allocation system
  (STM, MTM, LTM) with promotion, fragmentation metrics and optional
  serialization.

## Mocked or Deprecated Components
- `array_protection.h` is deprecated in favor of the consolidated
  `sep_precompiled` header.
- `stdlib.cpp` carries placeholder modules and interpreter notes flag incomplete
  pattern input handling.

## testing/readme/README.md

# Testing and Real Data Protocols

This directory describes how to run SEP's real data validation scripts and what
outputs to expect. **All procedures enforce a strict zero‑tolerance policy for
synthetic or fabricated market data.** Only authentic records from the OANDA
Practice API are acceptable.

## Shell Scripts

### `real_data_validation.sh`
Collects and verifies two weeks of historical forex data and performs a short
live‑stream test.

**Key steps**
1. Loads OANDA credentials from `OANDA.env`.
2. Fetches 14 days of candles for major pairs (`EUR_USD`, `GBP_USD`, `USD_JPY`,
   `AUD_USD`, `USD_CHF`) across multiple granularities.
3. Writes JSON files and fetch logs under `testing/real_data/historical/` and
   `testing/real_data/validation_reports/`.
4. Generates a markdown integrity report summarising file sizes, record counts
   and timestamp ranges.
5. Executes a 5‑minute live‑stream test to verify real‑time connectivity.

**Expected output**
- `testing/real_data/historical/*.json` – authentic candle data.
- `testing/real_data/validation_reports/` – logs and the integrity report.
- `testing/real_data/live_streams/` – live stream captures when enabled.

### `retail_data_validation.sh`
Validates the broader retail development kit using only real market data.

**Key steps**
1. Checks system status and confirms OANDA API connectivity.
2. Validates the existing real‑data cache and fetches fresh weekly data for a
   wider set of major pairs.
3. Runs quantum processing tests on the retrieved data.
4. Produces authenticity evidence in `validation/retail_kit_proof/`.

**Expected output**
- `validation/real_data_logs/` – system and cache validation logs.
- `validation/weekly_validation/<PAIR>/` – per‑pair weekly fetch logs.
- `validation/quantum_tests/` and `validation/pair_analysis/` – quantum
  processing results.
- `validation/retail_kit_proof/data_authenticity_report_<DATE>.md` – final report
  confirming that only genuine OANDA data was used.

## Directory Overview

### `examples/`
Contains simple `.sep` patterns (`test_basic.sep`, `test_fixed.sep`) to verify the
DSL runtime. These examples are purely demonstrative and do **not** use market
prices.

### `real_data/`
Created automatically by `real_data_validation.sh` to store authentic market
records and validation reports. This directory is not version‑controlled and
should contain **only** genuine OANDA outputs.

---
Running either script without legitimate OANDA credentials or by introducing
synthetic data violates project policy and invalidates all results.

## tests/README.md

# SEP Test Suite

This directory defines the skeleton for a new GoogleTest based suite.

## Test Categories
### Unit Tests
* **core/** – engine algorithms, pattern processing, memory management, CUDA kernel dispatch
* **util/** – parser, serialization, mathematical helpers, memory tier management
* **io/** – OANDA connector, market data conversion, C API surface
* **app/** – `trader_cli`, `data_downloader`, `sep_dsl_interpreter`, `quantum_tracker`

### Integration Tests
Validate interactions across modules such as data download → processing → CLI
commands and live OANDA connectivity (sandbox).

### Performance Tests
Micro benchmarks for CUDA kernels, memory pool throughput, and real-time
streaming performance.

## Building
Testing is not yet implemented. Add test executables with `add_sep_test` as
fixtures are created. No CUDA runtime is required for placeholder configuration.

## tests/readme/README.md

# Test Suite Overview

This directory summarizes the layout of the test suite.

## Unit
Focused checks for individual components. Current coverage targets core engine logic, utility helpers, I/O surfaces, and application wrappers.

## Integration
End-to-end flows combining multiple modules. These tests validate that subsystems interoperate correctly.

## Performance
Benchmarks measuring computational throughput and latency for critical paths.


## tests/unit/app/README.md

# App Unit Tests

No unit tests are present yet. Future work should cover command-line utilities and application-level behavior.


## tests/unit/core/README.md

# Core Unit Tests

Existing tests:
- `coherence_manager_test.cpp`
- `result_types_test.cpp`

Additional coverage is needed for other engine components and edge cases.


## tests/unit/io/README.md

# IO Unit Tests

Existing tests:
- `sep_c_api_test.cpp`

Future tests should exercise additional I/O interfaces and failure modes.


## tests/unit/util/README.md

# Util Unit Tests

Existing tests:
- `error_handling_test.cpp`

Further tests are required for remaining utility functions and helpers.

