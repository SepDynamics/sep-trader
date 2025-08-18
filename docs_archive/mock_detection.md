# Mock Data Detection

The trading system now embeds a runtime flag in core data structures to mark whether a data object represents **real** or **mock** information.

## Enabling Strict Checks

Set the `SEP_STRICT_MOCK_CHECK=1` environment variable in staging or production builds to enable runtime verification. When enabled, any `CandleData` marked as mock will trigger a runtime error when passed through production code paths.

Example:

```bash
export SEP_STRICT_MOCK_CHECK=1
```

Unset the variable or set it to `0` to disable the stricter checks.

## Rationale

These checks help ensure that simulated or placeholder data never reaches live trading logic. The default is disabled to avoid overhead during development, while staging or production environments can opt into strict validation.

