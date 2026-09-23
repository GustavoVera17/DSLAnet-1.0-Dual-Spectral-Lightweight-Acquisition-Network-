# DSLA-net 1.0: Dual Light Spectral Acquisition Network

This repository contains the architecture, orchestrator, and ablation study framework for DSLA-net (Dual Light Spectral Acquisition Network).

## Ablation Study: Model Complexity & Parameter Scaling

The framework parametrizes the DSLA-net architecture into four variants:
- `baseline`: Original architecture without CAB modules.
- `spatcab`: Enhances the spatial RGB branch with a Spatial Context Attention Block.
- `speccab`: Enhances the chemical/HSI branch with a Spectral Context Attention Block (3D Convolutions).
- `fullcab`: Dual Context implementation combining both.

Below is the parameter count and exact memory footprint (FP32) for all 20 scale models tested during the ablation study:

| Architectural Variant | 4 Blocks | 8 Blocks | 12 Blocks | 16 Blocks | 20 Blocks |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`baseline`** | 73.0k / 0.28 MB | 135.0k / 0.52 MB | 197.1k / 0.75 MB | 259.1k / 0.99 MB | 321.2k / 1.23 MB |
| **`spatcab`** | 75.9k / 0.29 MB | 140.8k / 0.54 MB | 205.7k / 0.78 MB | 270.6k / 1.03 MB | 335.6k / 1.28 MB |
| **`speccab`** | 73.1k / 0.28 MB | 135.2k / 0.52 MB | 197.3k / 0.75 MB | 259.4k / 0.99 MB | 321.5k / 1.23 MB |
| **`fullcab`** | 76.0k / 0.29 MB | 141.0k / 0.54 MB | 205.9k / 0.79 MB | 270.9k / 1.03 MB | 335.9k / 1.28 MB |

*(Calculations based on standard FP32 representation - 4 bytes per parameter)*

### Key Insights:
- **Ultra-Lightweight Profile:** Even at maximum depth (20 blocks with dual context), the model consumes only 1.28 Megabytes.
- **SpecCAB Efficiency:** The integration of 3D Convolutions via `SpecCAB` adds less than ~200 parameters per block due to its single-dimensional channel extraction design.
