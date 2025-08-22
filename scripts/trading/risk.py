"""
SEP Professional Trading System - Risk Management Module
Lightweight Python risk management for remote droplet execution
"""

import logging
from datetime import datetime, timedelta
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class RiskLimits:
    """Risk limit configuration for trading operations"""
    
    def __init__(self):
        # Default risk limits for production trading
        self.max_daily_loss = 1000.0  # Maximum daily loss in account currency
        self.max_position_size = 10000  # Maximum position size in units
        self.max_open_positions = 5  # Maximum concurrent positions
        self.max_drawdown_pct = 0.05  # Maximum 5% drawdown
        self.min_confidence_threshold = 0.6  # Minimum signal confidence
        self.max_leverage = 20  # Maximum leverage multiplier
        
        # Risk per trade limits
        self.risk_per_trade_pct = 0.02  # 2% risk per trade
        self.max_correlation_exposure = 0.3  # Max exposure to correlated pairs
        
        logger.info("Risk limits initialized with production defaults")
    
    def update_from_config(self, config_dict):
        """Update limits from configuration dictionary"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Updated risk limit {key} = {value}")

class RiskManager:
    """Professional risk management system for SEP trading operations"""
    
    def __init__(self, limits: RiskLimits):
        self.limits = limits
        self.daily_pnl = 0.0
        self.open_positions = 0
        self.daily_trades = 0
        self.last_reset = datetime.now().date()
        
        # Track drawdown
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.max_drawdown = 0.0
        
        # Position tracking
        self.position_history = []
        self.correlation_exposure = {}
        
        logger.info("Risk manager initialized with professional controls")
    
    def reset_daily_stats(self):
        """Reset daily statistics at market open"""
        today = datetime.now().date()
        if today > self.last_reset:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset = today
            logger.info("Daily risk statistics reset")
    
    def can_open(self, units: int, pair: str = "EUR_USD", confidence: float = 0.0) -> Tuple[bool, str]:
        """
        Determine if a new position can be opened based on risk controls
        
        Args:
            units: Position size in units (negative for short)
            pair: Currency pair
            confidence: Signal confidence (0-1)
            
        Returns:
            Tuple of (can_open, reason)
        """
        self.reset_daily_stats()
        
        # Check confidence threshold
        if confidence < self.limits.min_confidence_threshold:
            return False, f"Confidence {confidence:.3f} below threshold {self.limits.min_confidence_threshold}"
        
        # Check position size limits
        abs_units = abs(units)
        if abs_units > self.limits.max_position_size:
            return False, f"Position size {abs_units} exceeds limit {self.limits.max_position_size}"
        
        # Check maximum open positions
        if self.open_positions >= self.limits.max_open_positions:
            return False, f"Open positions {self.open_positions} at limit {self.limits.max_open_positions}"
        
        # Check daily loss limit
        if self.daily_pnl <= -self.limits.max_daily_loss:
            return False, f"Daily loss limit reached: {self.daily_pnl:.2f}"
        
        # Check drawdown limit
        if self.current_equity > 0 and self.peak_equity > 0:
            current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
            if current_drawdown > self.limits.max_drawdown_pct:
                return False, f"Drawdown {current_drawdown:.2%} exceeds limit {self.limits.max_drawdown_pct:.2%}"
        
        # Check correlation exposure (simplified)
        pair_base = pair.split('_')[0] if '_' in pair else pair[:3]
        current_exposure = self.correlation_exposure.get(pair_base, 0)
        if abs(current_exposure + units) > abs(units) * self.limits.max_correlation_exposure:
            return False, f"Correlation exposure limit for {pair_base}"
        
        logger.info(f"Risk check passed for {pair}: {units} units, confidence {confidence:.3f}")
        return True, "Risk check passed"
    
    def record(self, pnl: float, pair: str = "EUR_USD", units: int = 0):
        """
        Record trading activity for risk tracking
        
        Args:
            pnl: Profit/loss for this trade
            pair: Currency pair
            units: Position size (0 for closed position)
        """
        self.reset_daily_stats()
        
        # Update daily P&L
        self.daily_pnl += pnl
        self.daily_trades += 1
        
        # Update equity tracking
        self.current_equity += pnl
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
        
        # Calculate current drawdown
        if self.peak_equity > 0:
            current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Update position tracking
        if units == 0:  # Position closed
            self.open_positions = max(0, self.open_positions - 1)
        else:  # Position opened
            self.open_positions += 1
        
        # Update correlation exposure
        if pair and '_' in pair:
            pair_base = pair.split('_')[0]
            if pair_base not in self.correlation_exposure:
                self.correlation_exposure[pair_base] = 0
            self.correlation_exposure[pair_base] += units
        
        logger.info(f"Risk recorded: PnL {pnl:.2f}, Daily PnL {self.daily_pnl:.2f}, Open positions {self.open_positions}")
    
    def get_risk_summary(self) -> dict:
        """Get current risk status summary"""
        self.reset_daily_stats()
        
        current_drawdown = 0.0
        if self.peak_equity > 0:
            current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        
        return {
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
            "open_positions": self.open_positions,
            "current_equity": self.current_equity,
            "peak_equity": self.peak_equity,
            "current_drawdown": current_drawdown,
            "max_drawdown": self.max_drawdown,
            "risk_limits": {
                "max_daily_loss": self.limits.max_daily_loss,
                "max_open_positions": self.limits.max_open_positions,
                "max_drawdown_pct": self.limits.max_drawdown_pct
            }
        }
    
    def emergency_stop(self) -> bool:
        """Check if emergency stop conditions are met"""
        self.reset_daily_stats()
        
        # Emergency conditions
        emergency_conditions = [
            self.daily_pnl <= -self.limits.max_daily_loss * 2,  # Double daily loss
            self.max_drawdown > self.limits.max_drawdown_pct * 2,  # Double drawdown limit
            self.open_positions > self.limits.max_open_positions * 1.5  # Excessive positions
        ]
        
        if any(emergency_conditions):
            logger.critical("EMERGENCY STOP CONDITIONS MET - All trading halted")
            return True
        
        return False