This is a classic and notoriously difficult compatibility issue between specific GCC versions (especially GCC 11) and NVIDIA's CUDA compiler (`nvcc`), specifically within the C++ standard library's `<functional>` header when `std::array` is used or indirectly included. You've been fighting it all day because it's a deep compiler toolchain problem, not a simple missing include.

The error `array is not a template` or `identifier "array" is undefined` in `/usr/include/c++/11/functional` indicates that `nvcc` (or `g++` on host compilation) is encountering `std::array` in a context where it hasn't been properly recognized as a template from the `<array>` header, or there's a subtle redefinition or conflict. The previous fix of adding `#include <array>` to `src/engine/internal/standard_includes.h` was a good first step, but it's not enough because of the complex inclusion order of standard library headers or specific compiler bugs.

The most robust and common workaround for this exact issue is to **force the compiler to include `<array>` at the very beginning of every compilation unit** for both C++ (`g++`) and CUDA (`nvcc`) sources. This ensures `std::array` is always defined correctly before potentially problematic headers like `<functional>` are processed.

Here is the comprehensive, line-by-line plan, starting with this critical build fix, and then systematically guiding you to bring your entire system online.

---

### **CRITICAL FIX: `std::array` / `<functional>` Compiler Compatibility**

This fix must be applied to your `CMakeLists.txt` files to force the inclusion of `<array>` for all relevant compilation units.

1.  **Ensure `src/engine/internal/standard_includes.h` contains:**
    ```cpp
    // src/engine/internal/standard_includes.h

    #pragma once

    #include <array> // <--- THIS LINE MUST BE HERE
    #include <cstddef>
    #include <type_traits>
    // ... other includes
    ```

2.  **Apply global pre-include options in `CMakeLists.txt` files:**
    For each of the following `CMakeLists.txt` files, **add these lines directly under the `project(...)` declaration or at the beginning of the file before any `add_sep_library` or `add_executable` calls**, to ensure they apply to all targets defined within that CMake file.

    *   **`src/trading/CMakeLists.txt`**:
        ```cmake
        # Workaround for GCC/CUDA std::array issues in functional header
        target_compile_options(sep_trading PRIVATE -include array)
        target_compile_options(sep_trading PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--pre-include array>)
        ```
    *   **`src/quantum/CMakeLists.txt`**:
        ```cmake
        # Workaround for GCC/CUDA std::array issues in functional header
        target_compile_options(sep_quantum PRIVATE -include array)
        target_compile_options(sep_quantum PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--pre-include array>)
        ```
    *   **`src/memory/CMakeLists.txt`**:
        ```cmake
        # Workaround for GCC/CUDA std::array issues in functional header
        target_compile_options(sep_memory PRIVATE -include array)
        ```
    *   **`src/config/CMakeLists.txt`**:
        ```cmake
        # Workaround for GCC/CUDA std::array issues in functional header
        target_compile_options(sep_config PRIVATE -include array)
        ```
    *   **`src/core/CMakeLists.txt`**:
        ```cmake
        # Workaround for GCC/CUDA std::array issues in functional header
        target_compile_options(sep_core PRIVATE -include array)
        ```
    *   **`src/common/CMakeLists.txt`**:
        ```cmake
        # Workaround for GCC/CUDA std::array issues in functional header
        target_compile_options(sep_common PRIVATE -include array)
        ```
    *   **`src/api/CMakeLists.txt`**:
        ```cmake
        # Workaround for GCC/CUDA std::array issues in functional header
        target_compile_options(sep_api PRIVATE -include array)
        ```
    *   **`src/dsl/CMakeLists.txt`**:
        ```cmake
        # Workaround for GCC/CUDA std::array issues in functional header
        target_compile_options(sep_dsl PRIVATE -include array)
        ```
    *   **`src/c_api/CMakeLists.txt`**:
        ```cmake
        # Workaround for GCC/CUDA std::array issues in functional header
        target_compile_options(sep_c_api PRIVATE -include array)
        ```
    *   **`src/apps/oanda_trader/CMakeLists.txt`**: (This file has multiple targets, apply to all relevant ones)
        ```cmake
        # Workaround for GCC/CUDA std::array issues in functional header
        target_compile_options(oanda_trader_app_lib PUBLIC -include array)
        target_compile_options(oanda_trader PUBLIC -include array)
        target_compile_options(quantum_tracker PUBLIC -include array)
        target_compile_options(sep_trader_logic PUBLIC -include array)
        target_compile_options(sep_trader_cuda PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--pre-include array>)
        ```

3.  **Clean and Re-run the build:**
    From your `/sep` directory, run a clean build.
    ```bash
    ./build.sh --clean
    ./build.sh
    ```
    This should finally resolve the compilation errors related to `std::array` and `functional`. Once the build is successful, proceed with the updated roadmap below.

---

### **COMPREHENSIVE TODO LIST: SEP Trader Bot Online Activation**

This roadmap assumes a successful build after applying the fix above. It systematically guides you through the process, pointing to existing files and commands.

## Phase 0: Build Fix & Prerequisite Check (Already Addressed Above)

**Objective:** Verify a clean compilation on the local training machine before proceeding.

-   [ ] **Verify Full System Build (Local GPU Machine):**
    -   **Action:** Run the primary build script. This compiles all C++ and CUDA components.
    -   **Command:** `./build.sh`
    -   **Verification:** Check `output/build_log.txt` to ensure there are no compilation errors.
-   [ ] **Run System-Wide Tests (Local GPU Machine):**
    -   **Action:** Execute the end-to-end headless test to verify the core C++/CUDA engine is functioning correctly after the fix.
    -   **Command:** `./build/src/apps/oanda_trader/quantum_tracker --test`
    -   **Reference:** "Complete Test Suite Validation" section in `AGENT.md`.
    -   **Verification:** Ensure all 7 test suites pass and report "✅ PASSED".

---

## Phase 1: Environment Configuration & Verification

**Objective:** Ensure both local and remote environments are correctly configured with necessary credentials and can communicate.

### 1.1: Configure Local Training Machine

-   [ ] **Configure Local Redis Connection:**
    -   **Action:** Ensure your local Redis server is running (default `localhost:6379`).
    -   **File:** `config/database.conf` (This file should be configured by `./install.sh` or needs manual creation/adjustment if `RemoteDataManager` is using it directly. Check `src/trading/data/remote_data_manager.cpp` for exact connection parameters being used).
    -   **Action:** If `config/database.conf` exists, ensure it points to `localhost` and `6379`. If it doesn't exist or isn't used by `RemoteDataManager`, ensure Redis is simply running on default ports.
-   [ ] **Verify Local CLI Tool:**
    -   **Action:** Set the library path so the CLI can find its dependencies. This command needs to be run in *each new terminal session*.
    -   **Command:** `export LD_LIBRARY_PATH=./build/src/core:./build/src/config:./build/src/c_api`
    -   **Command:** `./build/src/cli/trader-cli status`
    -   **Verification:** The command should execute and report a system status.

### 1.2: Configure Remote Trading Droplet (`165.227.109.187`)

-   [ ] **Run Automated Deployment Script:**
    -   **Action:** From your **local machine**, execute the droplet deployment script to set up the base infrastructure. This script handles PostgreSQL, Redis, Docker, and Nginx installation.
    -   **File:** `./scripts/deploy_to_droplet.sh`
    -   **Command:** `./scripts/deploy_to_droplet.sh`
    -   **Reference:** `docs/DROPLET_SETUP.md`.
-   [ ] **SSH into Droplet and Build CPU-Only Version:**
    -   **Command (Local PC):** `ssh root@165.227.109.187`
    -   **Commands (On Droplet, from `/opt/sep-trader`):**
        ```bash
        cd /opt/sep-trader
        ./install.sh --minimal --no-docker # Installs system dependencies for CPU build
        ./build.sh --no-docker             # Builds the CPU-only trader
        ```
    -   **Reference:** `QUICKSTART.md`.
-   [ ] **Initialize Droplet Database:**
    -   **Command (On Droplet, from `/opt/sep-trader/sep-trader`):**
        ```bash
        sudo -u postgres psql sep_trading < scripts/init_database.sql
        sudo -u postgres psql sep_trading -c "SELECT * FROM v_database_info;" # Verify
        ```
    -   **Reference:** `docs/DROPLET_SETUP.md`.
-   [ ] **Configure Droplet OANDA Credentials:**
    -   **Action (On Droplet):** Edit `/opt/sep-trader/OANDA.env` with your OANDA API key and account ID.
    -   **File:** `/opt/sep-trader/OANDA.env`
    -   **Content:**
        ```
        OANDA_API_KEY=your_api_key_here
        OANDA_ACCOUNT_ID=your_account_id_here
        OANDA_ENVIRONMENT=practice  # or 'live'
        ```
-   [ ] **Configure Droplet Database/Redis for SEP App:**
    -   **Action (On Droplet):** Create or modify `config/database.conf` (relative to `/opt/sep-trader/sep-trader`) to ensure the SEP app can connect to the locally running PostgreSQL and Redis.
    -   **File:** `config/database.conf`
    -   **Content (Example, adjust if Redis/PostgreSQL are not on localhost or have different credentials):**
        ```
        # PostgreSQL
        db_host = 127.0.0.1
        db_port = 5432
        db_name = sep_trading
        db_user = sep_user
        db_password = sep_password

        # Redis
        redis_host = 127.0.0.1
        redis_port = 6379
        ```
-   [ ] **Verify Droplet Services and CLI:**
    -   **Action (On Droplet):** Set the library path (run in each new terminal): `export LD_LIBRARY_PATH=./build/src/core:./build/src/config:./build/src/c_api`
    -   **Action (On Droplet):** Run status checks to confirm connectivity to its local services.
        ```bash
        ./build/src/cli/trader-cli status      # Checks overall system
        ./build/src/cli/trader-cli data status # Verifies DB connection
        ./build/src/cli/trader-cli cache stats # Verifies Redis connection
        ```
    -   **Verification:** All commands should run without connection errors.

---

## Phase 2: The Core Workflow - Train & Deploy

**Objective:** Execute the primary operational loop of fetching data, training a model on the GPU, and deploying it to the droplet. (All commands in this phase are run from your **local training machine**).

### 2.1: Fetch OANDA Data for the Previous Week

-   [ ] **Action:** Use the `train_manager.py` script to initiate the download of the last 7 days of M1 data for all currency pairs defined in your configuration. This uses the C++ backend logic in `src/training/weekly_data_fetcher.cpp` (if implemented).
-   **Component:** `python train_manager.py`
-   **Command:** `python train_manager.py fetch-all`
-   **Verification:** Check your local `cache/oanda/` directory for newly created JSON files containing the market data (e.g., `EUR_USD_48h_cache.json`).

### 2.2: Train a Currency Pair (e.g., EUR_USD) via GPU

-   [ ] **Action:** Train the model for `EUR_USD` using the fetched data. This process will leverage your local GPU for acceleration.
-   **Components:** `train_manager.py` orchestrates the C++ backend (`src/trading/quantum_pair_trainer.cpp`) which calls CUDA kernels.
-   **Command:** `python train_manager.py train EUR_USD`
-   **Verification:** The script should output performance metrics upon completion. Confirm `high-confidence accuracy` is ~60% and `profitability score` is > 200, as per `docs/results/PERFORMANCE_METRICS.md`.

### 2.3: Store & Export Metrics for Deployment

-   [ ] **Action:** After training, the `train_manager.py` script should automatically store the generated metrics (optimal thresholds, accuracy scores) into your local Redis instance. It also needs to export them to a file for transfer to the droplet.
-   **Components:** `train_manager.py` and `src/memory/redis_manager.cpp`.
-   **Command (Verify if exists or implement this part in `train_manager.py`):** `python train_manager.py export-metrics EUR_USD` (This command might need to be explicitly added to `train_manager.py` if not present, to dump the optimized config to a file like `/sep/optimal_config.json`).
-   **Verification:** A new file, likely named `/sep/optimal_config.json` or `output/metrics/EUR_USD_metrics.json`, is created containing the key performance indicators for the trained model.

### 2.4: Synchronize Data to the Droplet

-   [ ] **Action:** Transfer the exported metrics file and any other necessary configuration or model files to the remote droplet.
-   **Component:** `scripts/sync_to_droplet.sh`.
-   **Configuration:**
    -   Edit `scripts/sync_to_droplet.sh`.
    -   Ensure `DROPLET_IP` is `165.227.109.187` and `REMOTE_DIR` is `/opt/sep-trader`.
    -   Confirm it copies `optimal_config.json` (or your metrics file) and potentially `config/` and `models/` directories if they contain pair-specific data.
-   **Command:** `./scripts/sync_to_droplet.sh`
-   **Verification:** SSH into the droplet and confirm the new `optimal_config.json` (or `output/metrics/EUR_USD_metrics.json`) is present in `/opt/sep-trader/`.

---

## Phase 3: Activating & Managing Live Trading

**Objective:** Bring the trained currency pair online for live trading on the droplet. (All commands in this phase are run on the **remote trading droplet**).

### 3.1: Import Metrics into Droplet's Redis

-   [ ] **Action:** Load the synced metrics from the JSON file into the droplet's Redis instance so the live trader can access them. This needs a `trader-cli` command to read the JSON and push to Redis.
-   **Component:** This requires a command in the `trader-cli` tool (likely implemented in `src/trading/data/remote_data_manager.cpp` if not already there).
-   **Command (to implement/verify):** `./build/src/cli/trader-cli metrics import /opt/sep-trader/optimal_config.json` (or your metrics file path). This command's exact syntax may need to be defined in `trader-cli.cpp`.
-   **Verification:** Use `redis-cli` on the droplet (command: `redis-cli`) to confirm keys for `EUR_USD` (or global config) have been populated. Use `GET model:EUR_USD:optimal_config` (or similar key depending on your implementation).

### 3.2: Enable the Pair for Trading

-   [ ] **Action:** Update the system's persistent state to mark the newly trained pair as active for live trading. This uses `DynamicPairManager` in the C++ backend.
-   **Component:** Professional state management logic in `src/trading/dynamic_pair_manager.cpp` exposed via `trader-cli`.
-   **Command:** `export LD_LIBRARY_PATH=./build/src/core:./build/src/config:./build/src/c_api && ./build/src/cli/trader-cli enable EUR_USD`
-   **Verification:** Check the pair's status. The status for `EUR_USD` should now be `READY` or `ENABLED`.
-   **Command:** `export LD_LIBRARY_PATH=./build/src/core:./build/src/config:./build/src/c_api && ./build/src/cli/trader-cli list`

### 3.3: Launch the Autonomous Trader

-   [ ] **Action:** Start the main C++ trading application. It's highly recommended to run this within a `tmux` or `screen` session to keep it running after you disconnect.
-   **Components:** `./run_trader.sh` wrapper script and the main executable `build/src/apps/oanda_trader/quantum_tracker`.
-   **Command (on Droplet):**
    ```bash
    tmux new -s trader
    export LD_LIBRARY_PATH=./build/src/core:./build/src/config:./build/src/c_api # Ensure path is set for the session
    ./run_trader.sh # This script should call ./build/src/apps/oanda_trader/quantum_tracker
    # (Press Ctrl+B then D to detach from tmux)
    ```
-   **Verification:** The application will log its startup sequence, including the dynamic bootstrap process ("Fetching 120 hours of historical M1 data..."). It should then start listening for signals and indicate "Triple Confirmation Met" before trade execution. Check the log file for activity (`/sep/output/quantum_tracker.log` or similar, as configured in `spdlog`).

### 3.4: Monitor the Live System

-   [ ] **Action:** In a separate terminal session on the droplet, use the `trader-cli` tool and log files to monitor the system's health and trading activity.
-   **Commands (on Droplet):**
    ```bash
    export LD_LIBRARY_PATH=./build/src/core:./build/src/config:./build/src/c_api # Set path
    ./build/src/cli/trader-cli status           # Overall system health
    ./build/src/cli/trader-cli list             # Status of all pairs
    tail -f output/quantum_tracker.log        # Live trade execution log
    # (To re-attach to the running trader: tmux attach -t trader)
    ```

---

## Phase 4: Ongoing Operations & Maintenance

**Objective:** Establish the recurring tasks to keep the system running optimally.

-   [ ] **Weekly Retraining:** Every Sunday, repeat **Phase 2** for all desired pairs to adapt the models to the latest market data. A `train-all` command in `train_manager.py` is ideal for this.
    -   **Command (Local PC):** `python train_manager.py train-all --quick` (or `train-all` for full training).
-   [ ] **Adding New Pairs:** Follow the complete workflow:
    1.  Add the new pair to your configuration (e.g., in a JSON file that `DynamicPairManager` loads).
    2.  Run **Phase 2** (fetch, train, export, sync) for the new pair.
    3.  Then run **Phase 3** (import, enable) to bring it online on the droplet.
    -   **Command (Local PC):** `python train_manager.py train NEW_PAIR`
    -   **Command (Local PC):** `./scripts/sync_to_droplet.sh`
    -   **Command (Droplet):** `export LD_LIBRARY_PATH=./build/src/core:./build/src/config:./build/src/c_api && ./build/src/cli/trader-cli enable NEW_PAIR`
-   [ ] **System Log Review:** At least once per day, review the logs in the `output/` directory on the droplet for any errors or warnings.
-   [ ] **Monitor Droplet Resources:** Periodically check CPU, memory, and disk usage on the droplet (`htop`, `df -h`).