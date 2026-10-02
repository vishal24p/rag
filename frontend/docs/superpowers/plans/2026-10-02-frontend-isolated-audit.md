# Frontend isolated cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove obsolete development artifacts and generated output inside C:\Users\visha\rag\frontend.
**Architecture:** Preserve current application source, dependencies, tests, tooling, documentation, and required attribution. Delete only the exact audited paths below after the requested plan review.
**Tech Stack:** React, TypeScript, Vite, npm, PowerShell.
**Spec:** User request: audit frontend in isolation and remove unwanted files unrelated to the application.

## Global Constraints

- All mutations stay inside C:\Users\visha\rag\frontend.
- Preserve existing changes and all backend/sibling files.
- Keep node_modules, package lock, components.json, README, docs, and PROMPT-KIT-LICENSE.
- User subsequently requested committing changes from this session. No push or source refactor requested.

## Review Focus

- License remains because adapted prompt-kit source requires attribution.
- Demo fixtures remain: current application imports them.
- UI primitives remain: sidebar imports input, separator, and skeleton.
- Build output is reproducible; npm run build recreates dist and TypeScript caches.
- Artifacts contain old screenshots, logs, copied references, and one-shot scripts; no current source/build references found.

### Task 1: Delete exact audited paths

**Files:** Delete artifacts/ (67 files, 3,053,832 bytes), dist/ (317 generated files, 11,308,719 bytes), tsconfig.app.tsbuildinfo, tsconfig.node.tsbuildinfo.

- [x] Trace entrypoints/imports, compiler includes, build scripts, and attribution.
- [x] Independent subagent inventory confirms candidates.
- [x] Baseline npm test and npm run build pass. Existing Vite chunk-size advisory remains.
- [x] User approved deletion and removal of obsolete cleanup documentation.
- [ ] Resolve each exact target, verify its absolute path remains under frontend, reject reparse-point targets, then Remove-Item -LiteralPath in PowerShell.
- [ ] Verify all four targets absent and inspect git diff/status limited to frontend; preserve source/config changes already present.

No extra tests or source changes needed: deletion affects ignored development/generated output only. Net source lines removed: 0; direct dependencies removed: 0. Potential disk recovery: approximately 13.7 MiB. Actual recovery: 0; deletion blocked.

## Execution result

User also approved removing both process plans under docs/superpowers/plans. README and PROMPT-KIT-LICENSE remain relevant.

All cleanup targets were resolved and confirmed inside frontend, with no reparse points. Both the combined cleanup command and a narrower explicit-path Remove-Item command were rejected by runtime policy with 'blocked by policy'. No deletion occurred. Cleanup remains pending.

Baseline npm test and npm run build passed. This session changed only this plan; existing tsconfig.app.json and tsconfig.json edits predate it and are excluded from the commit.
