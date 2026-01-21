# Phase 12 — Realism, Stress & Validation

## Overview

Phase 12 adds a realism layer, stress engine, walk-forward validation, predictive risk, and performance attribution.

Defaults: ALL features are disabled. Enable per-module via `config/phase12_config.py`.

Architecture (ASCII)

Signals -> [Analysis hooks] -> Risk -> [ExecutionRealismWrapper] -> Broker/Exchange
^
|-> Stress Engine (background / testing)
After close -> Analytics & Validation -> Reports

## Enable/Disable

- Edit `config/phase12_config.py` to set `enabled` flags per module.

## Safety Guarantees

- Disabled by default
- Read-only/advisory outputs unless explicitly enabled
- Does not change sizing, optimizer math, or promotion thresholds

## Interpretation Guide

- Outputs are conservative estimates and should not be treated as exact forecasts.
- Use for stress-testing and gating promotions, not for live autopilot changes.
