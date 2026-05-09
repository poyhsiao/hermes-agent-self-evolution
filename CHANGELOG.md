# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Fixed
- **SyntheticDatasetBuilder JSON parsing** — `ast.literal_eval` fallback for MiniMax JSON parsing failures (d5a7d01)
- **Python dict wrapper in evolve_skill_body output** — Prevent double-wrapping of skill body in Python dict objects (50dd609)
- **GEPA fallback** — Graceful degradation to MIPROv2 when GEPA is unavailable; robust adapter initialization (7daac8b)
- **Holdout improvements** — Cleaner holdout evaluation logic and robust adapter (7daac8b)
- **Hermes session importer** — Fix for short skill name matching in importer (4693c8f)

### Added
- **External session importers** — Claude Code, Copilot, and Hermes session history support (#4, #2)
- **Hermes session importer** — Import sessions from your own Hermes Agent installation

## [0.1.0] — 2026-01-15

### Added
- Initial fork from hermes-forge → hermes-agent-self-evolution rebrand
- Phase 1 skill evolution via DSPy + GEPA
- CLI model support for dataset generation
- Full pipeline validation

### Documentation
- Phase 1 validation report (PDF)
- Complete PLAN.md architecture document