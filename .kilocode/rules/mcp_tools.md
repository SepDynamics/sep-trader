# MCP Tool Documentation

This document provides a reference for the available tools provided by the connected MCP servers.


## Memory Server (`memory`)

This server provides tools to interact with a knowledge graph.

### Tools

#### `create_entities`
- **Description**: Create multiple new entities in the knowledge graph.
- **Parameters**:
    - `entities` (*required*): An array of entities to create.

#### `create_relations`
- **Description**: Create multiple new relations between entities in the knowledge graph. Relations should be in active voice.
- **Parameters**:
    - `relations` (*required*): An array of relations to create.

#### `add_observations`
- **Description**: Add new observations to existing entities in the knowledge graph.
- **Parameters**:
    - `observations` (*required*): An array of observations to add.

#### `delete_entities`
- **Description**: Delete multiple entities and their associated relations from the knowledge graph.
- **Parameters**:
    - `entityNames` (*required*): An array of entity names to delete.

#### `delete_observations`
- **Description**: Delete specific observations from entities in the knowledge graph.
- **Parameters**:
    - `deletions` (*required*): An array of observations to delete.

#### `delete_relations`
- **Description**: Delete multiple relations from the knowledge graph.
- **Parameters**:
    - `relations` (*required*): An array of relations to delete.

#### `read_graph`
- **Description**: Read the entire knowledge graph.
- **Parameters**: None.

#### `search_nodes`
- **Description**: Search for nodes in the knowledge graph based on a query.
- **Parameters**:
    - `query` (*required*): The search query to match against entity names, types, and observation content.

#### `open_nodes`
- **Description**: Open specific nodes in the knowledge graph by their names.
- **Parameters**:
    - `names` (*required*): An array of entity names to retrieve.

---

## Context7 Server (`context7`)

This server provides tools for retrieving documentation and code examples for any library.

### Tools

#### `resolve-library-id`
- **Description**: Resolves a package/product name to a Context7-compatible library ID and returns a list of matching libraries. You MUST call this function before 'get-library-docs' to obtain a valid Context7-compatible library ID.
- **Parameters**:
    - `libraryName` (*required*): Library name to search for and retrieve a Context7-compatible library ID.

#### `get-library-docs`
- **Description**: Fetches up-to-date documentation for a library. You must call 'resolve-library-id' first to obtain the exact Context7-compatible library ID required to use this tool.
- **Parameters**:
    - `context7CompatibleLibraryID` (*required*): Exact Context7-compatible library ID (e.g., '/mongodb/docs').
    - `topic` (optional): Topic to focus documentation on (e.g., 'hooks', 'routing').
    - `tokens` (optional): Maximum number of tokens of documentation to retrieve (default: 10000).