"""
Risk Management Module for SEP Trading Engine
Implements position sizing, risk limits, and portfolio management
"""

import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RiskLimits:
    """Risk limit configuration"""
    max_position_size: float = 10000.0  # Maximum position size in units
    max_daily_loss: float = 1000.0      # Maximum daily loss in currency
    max_drawdown: float = 0.10          # Maximum drawdown (10%)
    max_positions: int = 5              # Maximum concurrent positions
    risk_per_trade: float = 0.02        # Risk per trade (2% of account)
    stop_loss_pct: float = 0.015        # Stop loss percentage (1.5%)
    take_profit_pct: float = 0.03       # Take profit percentage (3%)
    leverage_limit: float = 10.0        # Maximum leverage
    correlation_limit: float = 0.7      # Maximum correlation between positions

class RiskManager:
    """
    Risk management system for controlling trading exposure
    """
    
    def __init__(self, limits: RiskLimits):
        self.limits = limits
        self.positions: Dict[str, dict] = {}
        self.daily_pnl: float = 0.0
        self.last_reset: datetime = datetime.now(timezone.utc)
        self.trade_history: List[dict] = []
        
    def can_open(self, units: int, instrument: str = None) -> Tuple[bool, str]:
        """
        Check if a trade can be opened based on risk limits
        
        Args:
            units: Position size in units (negative for short)
            instrument: Trading instrument
            
        Returns:
            Tuple of (allowed, reason)
        """
        try:
            # Check position count limit
            if len(self.positions) >= self.limits.max_positions:
                return False, f"Maximum positions limit reached ({self.limits.max_positions})"
            
            # Check position size limit
            abs_units = abs(units)
            if abs_units > self.limits.max_position_size:
                return False, f"Position size ({abs_units}) exceeds limit ({self.limits.max_position_size})"
            
            # Check daily loss limit
            if self.daily_pnl < -abs(self.limits.max_daily_loss):
                return False, f"Daily loss limit exceeded ({self.daily_pnl:.2f})"
            
            # Check if we already have a position in this instrument
            if instrument and instrument in self.positions:
                return False, f"Position already exists for {instrument}"
            
            # Additional checks would go here (leverage, correlation, etc.)
            
            return True, "Risk check passed"
            
        except Exception as e:
            logger.error(f"Risk check error: {e}")
            return False, f"Risk check error: {e}"
    
    def record(self, pnl: float, instrument: str = None) -> None:
        """
        Record a trade result for risk tracking
        
        Args:
            pnl: Profit/loss amount
            instrument: Trading instrument
        """
        try:
            # Reset daily PnL if it's a new day
            now = datetime.now(timezone.utc)
            if now.date() != self.last_reset.date():
                self.daily_pnl = 0.0
                self.last_reset = now
            
            # Update daily PnL
            self.daily_pnl += pnl
            
            # Record trade
            trade_record = {
                'timestamp': now.isoformat(),
                'instrument': instrument,
                'pnl': pnl,
                'daily_pnl': self.daily_pnl
            }
            self.trade_history.append(trade_record)
            
            # Keep only last 1000 trades
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-1000:]
                
            logger.info(f"Trade recorded: {instrument} PnL: {pnl:.2f}, Daily PnL: {self.daily_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
    
    def open_position(self, instrument: str, units: int, entry_price: float) -> bool:
        """
        Record opening of a position
        
        Args:
            instrument: Trading instrument
            units: Position size (negative for short)
            entry_price: Entry price
            
        Returns:
            True if position recorded successfully
        """
        try:
            self.positions[instrument] = {
                'units': units,
                'entry_price': entry_price,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'unrealized_pnl': 0.0
            }
            logger.info(f"Position opened: {instrument} {units} units @ {entry_price}")
            return True
            
        except Exception as e:
            logger.error(f"Error opening position: {e}")
            return False
    
    def close_position(self, instrument: str, exit_price: float) -> Optional[float]:
        """
        Record closing of a position and calculate PnL
        
        Args:
            instrument: Trading instrument
            exit_price: Exit price
            
        Returns:
            Realized PnL or None if error
        """
        try:
            if instrument not in self.positions:
                logger.warning(f"No position found for {instrument}")
                return None
            
            position = self.positions[instrument]
            units = position['units']
            entry_price = position['entry_price']
            
            # Calculate PnL (simplified - actual calculation would depend on instrument type)
            if units > 0:  # Long position
                pnl = (exit_price - entry_price) * abs(units)
            else:  # Short position
                pnl = (entry_price - exit_price) * abs(units)
            
            # Remove position
            del self.positions[instrument]
            
            # Record the trade
            self.record(pnl, instrument)
            
            logger.info(f"Position closed: {instrument} PnL: {pnl:.2f}")
            return pnl
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return None
    
    def update_unrealized_pnl(self, instrument: str, current_price: float) -> None:
        """
        Update unrealized PnL for an open position
        
        Args:
            instrument: Trading instrument
            current_price: Current market price
        """
        try:
            if instrument not in self.positions:
                return
            
            position = self.positions[instrument]
            units = position['units']
            entry_price = position['entry_price']
            
            # Calculate unrealized PnL
            if units > 0:  # Long position
                unrealized_pnl = (current_price - entry_price) * abs(units)
            else:  # Short position
                unrealized_pnl = (entry_price - current_price) * abs(units)
            
            position['unrealized_pnl'] = unrealized_pnl
            
        except Exception as e:
            logger.error(f"Error updating unrealized PnL: {e}")
    
    def get_risk_summary(self) -> Dict:
        """
        Get current risk summary
        
        Returns:
            Dictionary with risk metrics
        """
        try:
            total_unrealized = sum(pos['unrealized_pnl'] for pos in self.positions.values())
            
            return {
                'open_positions': len(self.positions),
                'max_positions': self.limits.max_positions,
                'daily_pnl': self.daily_pnl,
                'max_daily_loss': self.limits.max_daily_loss,
                'unrealized_pnl': total_unrealized,
                'total_pnl': self.daily_pnl + total_unrealized,
                'risk_level': self._calculate_risk_level(),
                'positions': list(self.positions.keys()),
                'limits': {
                    'max_position_size': self.limits.max_position_size,
                    'max_positions': self.limits.max_positions,
                    'max_daily_loss': self.limits.max_daily_loss,
                    'risk_per_trade': self.limits.risk_per_trade
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting risk summary: {e}")
            return {'error': str(e)}
    
    def _calculate_risk_level(self) -> str:
        """Calculate current risk level"""
        try:
            # Simple risk level calculation based on multiple factors
            risk_score = 0
            
            # Position count factor
            position_ratio = len(self.positions) / self.limits.max_positions
            risk_score += position_ratio * 30
            
            # Daily loss factor
            if self.limits.max_daily_loss > 0:
                loss_ratio = abs(self.daily_pnl) / self.limits.max_daily_loss
                risk_score += loss_ratio * 40
            
            # Unrealized PnL factor
            total_unrealized = sum(pos['unrealized_pnl'] for pos in self.positions.values())
            if total_unrealized < 0:
                risk_score += min(abs(total_unrealized) / 1000, 30)
            
            # Determine risk level
            if risk_score < 25:
                return 'low'
            elif risk_score < 60:
                return 'medium'
            else:
                return 'high'
                
        except Exception as e:
            logger.error(f"Error calculating risk level: {e}")
            return 'unknown'
    
    def should_stop_trading(self) -> bool:
        """
        Check if trading should be stopped due to risk limits
        
        Returns:
            True if trading should be stopped
        """
        try:
            # Stop if daily loss limit exceeded
            if self.daily_pnl < -abs(self.limits.max_daily_loss):
                return True
            
            # Stop if too many losing trades in a row
            if len(self.trade_history) >= 5:
                recent_trades = self.trade_history[-5:]
                if all(trade['pnl'] < 0 for trade in recent_trades):
                    logger.warning("5 consecutive losing trades - stopping trading")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking stop trading condition: {e}")
            return False
    
    def get_position_size(self, account_balance: float, risk_amount: float = None) -> int:
        """
        Calculate appropriate position size based on account balance and risk
        
        Args:
            account_balance: Current account balance
            risk_amount: Risk amount (defaults to risk_per_trade * balance)
            
        Returns:
            Recommended position size in units
        """
        try:
            if risk_amount is None:
                risk_amount = account_balance * self.limits.risk_per_trade
            
            # Simple position sizing - would be more sophisticated in practice
            # Assuming 1 unit = 1 currency unit
            stop_loss_amount = account_balance * self.limits.stop_loss_pct
            
            if stop_loss_amount > 0:
                position_size = int(risk_amount / stop_loss_amount)
            else:
                position_size = int(account_balance * 0.1)  # Default to 10% of balance
            
            # Apply maximum position size limit
            position_size = min(position_size, int(self.limits.max_position_size))
            
            return max(1, position_size)  # Minimum position size of 1
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 100  # Default position size