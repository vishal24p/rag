# Frontend cleanup Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans task-by-task.

**Goal:** Remove unused frontend scaffolding, validate current chat, and commit frontend only locally.
**Architecture:** Retain entrypoint import closure and two used AI Elements message exports. Remove abandoned components/packages, obsolete attachment test helper, and superseded frontend planning files. Keep documentation aligned with shipped UI.
**Tech Stack:** React, TypeScript, Vite, Tailwind, prompt-kit, shadcn.
**Spec:** User requests checking, cleaning and a local frontend-only commit.

## Constraints
Only frontend edits and staging. Preserve all existing changes outside frontend. No push. Existing Ponytail/ECC guidance: delete unused code instead of adding abstractions.

- [x] Remove unused components, dependencies and stale scaffolding; retain demo prompt tests.
- [x] Validate production build, tests, chat and theme/citation interactions.
- [x] Inspect staged paths and commit only frontend locally.
