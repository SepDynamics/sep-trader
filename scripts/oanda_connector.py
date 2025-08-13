import os
import requests
import logging

logger = logging.getLogger(__name__)


class OandaConnector:
    """Simple REST connector for OANDA v20 API."""

    def __init__(self, api_key=None, account_id=None, practice=True):
        self.api_key = api_key or os.environ.get("OANDA_API_KEY")
        self.account_id = account_id or os.environ.get("OANDA_ACCOUNT_ID")
        self.base_url = (
            "https://api-fxpractice.oanda.com/v3"
            if practice
            else "https://api-fxtrade.oanda.com/v3"
        )
        if not self.api_key or not self.account_id:
            raise ValueError("Missing OANDA credentials")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    def place_market_order(self, instrument: str, units: int) -> dict:
        """Place a market order."""
        url = f"{self.base_url}/accounts/{self.account_id}/orders"
        order = {
            "order": {
                "units": str(units),
                "instrument": instrument,
                "timeInForce": "FOK",
                "type": "MARKET",
                "positionFill": "DEFAULT",
            }
        }
        logger.info("Placing market order: %s", order)
        response = self.session.post(url, json=order, timeout=10)
        response.raise_for_status()
        return response.json()

    def get_account_summary(self) -> dict:
        """Fetch account summary for risk checks."""
        url = f"{self.base_url}/accounts/{self.account_id}/summary"
        response = self.session.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
