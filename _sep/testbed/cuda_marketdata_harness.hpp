#pragma once
#include <vector>
#include "src/connectors/oanda_connector.h"

std::vector<double> cpuDoubleMid(const std::vector<sep::connectors::MarketData>& data);
std::vector<double> gpuDoubleMid(const std::vector<sep::connectors::MarketData>& data);
