# Utils

Hosts a complete API client for interacting with various backend endpoints.

## Responsibilities
- Sends REST requests for system status, trading actions, performance metrics, and configuration changes.
- Provides helper methods for pairs management, command execution, and data reload.

## Configuration
- `REACT_APP_API_URL`: base URL for all requests (default `http://localhost:5000`).
- Default headers include `Content-Type: application/json`; additional headers can be supplied per request.
