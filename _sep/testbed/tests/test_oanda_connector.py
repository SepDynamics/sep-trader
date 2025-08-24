import sys
import os
from unittest import mock

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..', 'scripts'))
from oanda_connector import OandaConnector  # noqa: E402


def test_place_market_order():
    connector = OandaConnector(config_path="/dev/null")
    connector.api_key = 'k'
    connector.account_id = 'a'
    connector.api_base = 'https://example.com'
    connector.connected = True

    class Resp:
        status_code = 201

        def raise_for_status(self):
            pass

        def json(self):
            return {'orderFillTransaction': {'id': '1'}}

    with mock.patch.object(connector.session, 'post', return_value=Resp()) as post:
        res = connector.place_market_order('EUR_USD', 100)
        assert res['orderFillTransaction']['id'] == '1'
        assert post.call_count == 1
