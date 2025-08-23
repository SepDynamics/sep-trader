# Frontend Overview

This directory documents the structure of the React-based client for the SEP trading system.

## Dockerfile
Defines a containerized build that installs dependencies and serves the production build through Nginx, ensuring a consistent runtime for deployment.

## Environment Scripts
Scripts such as `env.sh` configure environment variables used during local development and image builds.

## `public/` Directory
Holds static assets like `index.html`, icons, and manifest files that are copied directly into the final build.

## `src/` Directory
Contains the React source code, including components and hooks that drive the user interface.

## Backend Dependencies
The frontend relies on backend REST APIs and WebSocket endpoints for data and live updates.
