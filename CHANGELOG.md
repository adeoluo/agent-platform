# Changelog

All notable changes to this project are documented here. The file is generated
from Conventional Commits by [git-cliff](https://git-cliff.org) when a release is cut
with `make cut-release`.

## [0.2.0] 2026-11-02

### Features

- **providers:** OpenAI and Anthropic providers with classified errors and failover
- **api:** Stream agent steps over server-sent events
- **mcp:** Serve the platform to MCP clients over stdio, add chat command

### Documentation

- Standard README structure and language framing (#2)
- Vertical layered architecture diagram
- Horizontal layered architecture diagram
- Compact horizontal architecture diagram
- Providers, streaming, MCP and chat in README, diagram and agent notes
- List the provider and MCP modules in AGENTS.md and llms.txt

### Build and CI

- Run CodeQL on main and on a weekly schedule only

### Maintenance

- Add release automation, AGENTS.md, llms.txt and contributor docs

## [0.1.0] 2026-11-02
