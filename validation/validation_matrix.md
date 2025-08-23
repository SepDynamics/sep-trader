# Validation Matrix

| Claim | Evidence Path | SHA256 |
| --- | --- | --- |
| Authentic OANDA data only | testing/retail_data_validation.sh | 6a532fac8559d6dc9ce68051066ff3a44997e035cbb91a960826e2434201974a |
| Fused signals carry input hash and config version | src/app/multi_asset_signal_fusion.cpp | e7017c47ca4261036cc513219510aad3ebcd32bce1fe249f666c15a4681bd4de |
| Replayable fused-signal windows | _sep/testbed/replay_window.py | 0cc31885fa8c5b0ab4ef6fda08644974e254b2f1aef29d1e60ac994be64b818b |
