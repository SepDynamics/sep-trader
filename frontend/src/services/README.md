# Services

Contains the API client used for all HTTP requests to the backend.

## Responsibilities
- Wraps fetch calls for authentication, market data, trading operations, and configuration.
- Manages an auth token and attaches it to request headers.

## Configuration
- `REACT_APP_API_URL`: base URL for API requests (default `http://localhost:5000`).
- `auth_token`: persisted in `localStorage` and used for `Authorization` headers.
