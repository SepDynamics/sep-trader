import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..', 'src'))
from trading.risk import RiskManager, RiskLimits  # noqa: E402


def test_risk_limits():
    rm = RiskManager(
        RiskLimits(max_daily_loss=10, max_position_size=5, max_drawdown=0.5)
    )
    allowed, reason = rm.can_open(10)
    assert not allowed and reason == 'position_size'
    allowed, _ = rm.can_open(3)
    assert allowed
    rm.record(-11)
    allowed, reason = rm.can_open(1)
    assert not allowed and reason == 'daily_loss_limit'
