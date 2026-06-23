# 1. Record architecture decisions

Date: 2026-06-23
Status: accepted

## Context
We want a lightweight, durable record of significant technical decisions
(backbone choice, physics formulation, data formats) for a paper-bound project.

## Decision
Use Architecture Decision Records (ADRs), one Markdown file per decision in
`docs/decisions/`, numbered sequentially.

## Consequences
Decisions are reviewable and reversible with context. New significant choices
get a new ADR; superseded ones are marked and linked.
