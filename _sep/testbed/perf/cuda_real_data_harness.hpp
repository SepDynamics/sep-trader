#pragma once
#include <vector>
#include <string>
#include "src/connectors/oanda_connector.h"

std::vector<sep::connectors::MarketData> loadRealCandleData(const std::string& csv_path);
std::vector<double> cpuDoubleMid(const std::vector<sep::connectors::MarketData>& data);
std::vector<double> gpuDoubleMid(const std::vector<sep::connectors::MarketData>& data);
