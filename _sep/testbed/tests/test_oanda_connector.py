import sys
import os
from unittest import mock

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..', 'scripts'))
from oanda_connector import OandaConnector  # noqa: E402


def test_place_market_order():
    connector = OandaConnector(api_key='k', account_id='a')

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {'orderCreateTransaction': {'id': '1'}}

    with mock.patch.object(connector.session, 'post', return_value=Resp()) as post:
        res = connector.place_market_order('EUR_USD', 100)
        assert res['orderCreateTransaction']['id'] == '1'
        assert post.call_count == 1
