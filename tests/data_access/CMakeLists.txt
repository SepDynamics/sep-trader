add_sep_test(filter_test
    SOURCES filter_test.cpp
    DEPENDENCIES sep_lib
)

add_sep_test(filter_fuzz_test
    SOURCES filter_fuzz_test.cpp
    DEPENDENCIES sep_lib
)
