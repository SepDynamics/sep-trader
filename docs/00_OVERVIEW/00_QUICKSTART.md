# Quick Start Guide: SEP Professional Trader-Bot

This guide provides the essential steps to set up the SEP trading system and run your first trading strategy.

## 1. System Setup

This section covers the setup for the hybrid local/remote architecture.

### 1.1. System Requirements

- **Local Training Machine (With GPU):** Ubuntu 22.04+ / Fedora 42+ / Windows 10+ with WSL2, NVIDIA GPU with CUDA 12.9+.
- **Remote Trading Droplet (CPU-Only):** Ubuntu 24.04 LTS, 2+ vCPUs, 8GB+ RAM.

### 1.2. Deploy Remote Trading Droplet

1.  **Clone Repository & Deploy:**
    ```bash
    git clone https://github.com/SepDynamics/sep-trader.git
    cd sep-trader
    # Replace <your_droplet_ip> with your server's IP address.
    ./scripts/deploy_to_droplet.sh --ip <your_droplet_ip>
    ```
2.  **Configure Credentials:**
    SSH into your droplet and add your OANDA API key and account ID to `/opt/sep-trader/config/OANDA.env`.

3.  **Start Remote Services:**
    ```bash
    ssh root@<your_droplet_ip>
    cd /opt/sep-trader/sep-trader
    docker-compose up -d
    ```

### 1.3. Setup Local Training Machine

1.  **Clone Repository & Install Dependencies:**
    ```bash
    git clone https://github.com/SepDynamics/sep-trader.git
    cd sep-trader
    # For Linux
    ./install.sh --minimal --no-docker
    # For Windows
    .\install.bat
    ```
2.  **Build the Project:**
    ```bash
    # For Linux
    ./build.sh --no-docker
    # For Windows
    .\build.bat
    ```
3.  **Set Environment Variables:**
    Ensure the compiled libraries are in your system's PATH (Linux: `LD_LIBRARY_PATH`).

## 2. Your First Trading Strategy (DSL)

Once the system is set up, you can create and run a trading strategy using the SEP DSL.

### 2.1. Create a Strategy File
Create a file named `my_first_strategy.sep` with the following content:

```sep
pattern simple_eur_usd_strategy {
    // Fetch 100 periods of 15-minute data for EUR_USD
    price_data = fetch_live_oanda_data("EUR_USD", "M15", 100)
    
    // Perform quantum analysis
    coherence = measure_coherence(price_data)
    entropy = measure_entropy(price_data)
    stability = measure_stability(price_data)
    
    // Generate a signal quality score
    signal_quality = coherence * (1.0 - entropy) * stability
    
    print("=== EUR/USD Analysis ===")
    print("Signal Quality:", signal_quality)
    
    // Trading decision logic
    if (signal_quality > 0.65) {
        print("🚀 STRONG BUY SIGNAL!")
        // To execute real trades, uncomment the line below:
        // execute_trade("EUR_USD", "BUY", 1000)
    } else if (signal_quality < 0.35) {
        print("📉 STRONG SELL SIGNAL!")
        // execute_trade("EUR_USD", "SELL", 1000)
    } else {
        print("⏳ No clear signal.")
    }
}
```

### 2.2. Run Your Strategy
Execute the DSL script from your local machine's terminal:

```bash
# For Linux
source OANDA.env && ./build/src/dsl/sep_dsl_interpreter my_first_strategy.sep

# For Windows
set OANDA_API_KEY=your_key && .\build\src\dsl\sep_dsl_interpreter.exe my_first_strategy.sep
```

## 3. Autonomous Trading

For fully autonomous trading with the proven 60.73% accuracy system, run the `quantum_tracker` application on your local machine:

```bash
# Ensure OANDA.env is configured
source OANDA.env 
./build/src/app/oanda_trader/quantum_tracker
```
This will connect to your remote droplet, fetch data, and begin executing trades automatically based on the core system logic.

---
For more detailed instructions, please refer to the other documents in this `docs` folder.