# OANDA Trader build configuration

# Attempt to locate optional libraries
find_package(oanda_trader_app_lib QUIET)
find_package(sep_ui QUIET)

# Build local app library if not provided
if(NOT TARGET oanda_trader_app_lib)
    set(OANDA_TRADER_SOURCES
        oanda_trader_app.cpp
        quantum_tracker_app.cpp
        quantum_tracker_window.cpp
        rolling_window_chart.cpp
        quantum_signal_bridge.cpp
        multi_asset_signal_fusion.cpp
        market_model_cache.cpp
        enhanced_market_model_cache.cpp
        market_regime_adaptive.cpp
        realtime_aggregator.cpp
        tick_data_manager.cpp
        weekend_optimizer.cpp
        forward_window_kernels.cpp
    )

    set(OANDA_TRADER_CUDA_SOURCES)
    if(SEP_USE_CUDA)
        list(APPEND OANDA_TRADER_CUDA_SOURCES
            forward_window_kernels.cu
            tick_cuda_kernels.cu
        )
    endif()

    add_sep_library(oanda_trader_app_lib
        SOURCES ${OANDA_TRADER_SOURCES}
        CUDA_SOURCES ${OANDA_TRADER_CUDA_SOURCES}
        PCH_HEADER ../common/sep_precompiled.h
        DEPENDENCIES
            sep_connectors
            sep_common
            sep_core_integrated
            sep_quantum
            sep_engine
            sep_core_deps
    )
endif()

# Stub sep_ui if not available
if(NOT TARGET sep_ui)
    add_library(sep_ui INTERFACE)
endif()

# Main executable
add_sep_executable(oanda_trader
    SOURCES main.cpp
    DEPENDENCIES oanda_trader_app_lib sep_ui
)

# Ensure library paths are visible to the linker and build order is maintained
if(TARGET oanda_trader_app_lib)
    target_link_directories(oanda_trader PRIVATE $<TARGET_FILE_DIR:oanda_trader_app_lib>)
    add_dependencies(oanda_trader oanda_trader_app_lib)
endif()

if(TARGET sep_ui)
    get_target_property(_sep_ui_type sep_ui TYPE)
    if(NOT _sep_ui_type STREQUAL "INTERFACE_LIBRARY")
        target_link_directories(oanda_trader PRIVATE $<TARGET_FILE_DIR:sep_ui>)
    endif()
    add_dependencies(oanda_trader sep_ui)
endif()
