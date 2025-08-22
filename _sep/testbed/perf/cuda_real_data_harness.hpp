#pragma once
#include <string>
#include <vector>
#include "io/oanda_connector.h"

std::vector<sep::connectors::MarketData> loadRealCandleData(const std::string& csv_path);
std::vector<double> cpuDoubleMid(const std::vector<sep::connectors::MarketData>& data);
std::vector<double> gpuDoubleMid(const std::vector<sep::connectors::MarketData>& data);
