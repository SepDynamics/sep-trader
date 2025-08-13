if(SEP_CPU_ONLY)
    add_library(sep_testbed_cuda_stubs STATIC ../_sep/testbed/cuda_stubs.cpp)
    set(TESTBED_DEPS sep_testbed_cuda_stubs)
else()
    set(TESTBED_DEPS "")
endif()

add_sep_test(data_pipeline_tests SOURCES data_pipeline/test_data_integrity.cpp DEPENDENCIES ${TESTBED_DEPS})
add_sep_test(pipeline_full_cycle SOURCES pipeline/test_full_cycle.cpp DEPENDENCIES ${TESTBED_DEPS})
add_sep_test(oanda_integration_test SOURCES integration/oanda_integration_test.cpp DEPENDENCIES sep_connectors ${TESTBED_DEPS})

if(NOT SEP_CPU_ONLY)
    add_sep_test(cuda_validation SOURCES performance/cuda_validation.cpp ../_sep/testbed/cuda_marketdata_harness.cu)
endif()

add_sep_test(trading_signal_trace SOURCES integration/test_signal_trace.cpp DEPENDENCIES ${TESTBED_DEPS})
add_sep_test(unit_data_pipeline SOURCES unit/data_pipeline_test.cpp DEPENDENCIES ${TESTBED_DEPS})
add_sep_test(placeholder_detection_test SOURCES unit/placeholder_detection_test.cpp DEPENDENCIES ${TESTBED_DEPS})
add_sep_test(accuracy_benchmark SOURCES performance/accuracy_benchmark.cpp DEPENDENCIES ${TESTBED_DEPS})

if(NOT SEP_CPU_ONLY)
    add_sep_test(cuda_real_data_test SOURCES performance/cuda_real_data_test.cpp ../_sep/testbed/perf/cuda_real_data_harness.cu)
endif()

add_sep_test(oanda_connector_live_test SOURCES integration/oanda_connector_live_test.cpp DEPENDENCIES sep_connectors ${TESTBED_DEPS})
