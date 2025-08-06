# SEP Apps - Standalone Applications


add_executable(data_downloader data_downloader.cpp)
target_link_libraries(data_downloader PRIVATE
    sep_global_includes
    sep_trading
    sep_connectors
    sep_common
    nlohmann_json::nlohmann_json
    pqxx
    pq
    curl
    tbb
)
