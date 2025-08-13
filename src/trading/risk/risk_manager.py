import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    max_daily_loss: float = 1000.0
    max_position_size: int = 100000
    max_drawdown: float = 0.2  # fraction of peak equity


class RiskManager:
    """Implements basic multi-level risk controls."""

    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()
        self.daily_pl = 0.0
        self.current_equity = 0.0
        self.peak_equity = 0.0

    def can_open(self, units: int) -> tuple[bool, str]:
        """Check whether a position of given size is allowed."""
        if abs(units) > self.limits.max_position_size:
            return False, "position_size"
        if self.daily_pl <= -self.limits.max_daily_loss:
            return False, "daily_loss_limit"
        if self.peak_equity and self.drawdown() >= self.limits.max_drawdown:
            return False, "max_drawdown"
        return True, ""

    def record(self, pl: float) -> None:
        """Record profit or loss for a completed trade."""
        self.daily_pl += pl
        self.current_equity += pl
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
        logger.debug("Recorded P/L %s, equity=%s", pl, self.current_equity)

    def drawdown(self) -> float:
        if not self.peak_equity:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity
