# Integration Tests

The OANDA integration tests require sandbox credentials. Set the following environment variables before running:

```bash
export OANDA_API_KEY="your_api_key"
export OANDA_ACCOUNT_ID="your_account_id"
```

These tests connect to the OANDA sandbox API and will be skipped if the variables are not set.
