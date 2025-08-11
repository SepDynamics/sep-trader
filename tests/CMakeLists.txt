add_sep_test(data_pipeline_tests SOURCES data_pipeline/test_data_integrity.cpp)
add_sep_test(pipeline_full_cycle SOURCES pipeline/test_full_cycle.cpp)
add_sep_test(cuda_validation SOURCES performance/cuda_validation.cpp ../_sep/testbed/cuda_marketdata_harness.cu)
