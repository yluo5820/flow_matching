#!/usr/bin/env bash
set -euo pipefail

fm-lab-train --config configs/toy/two_moons_baseline.yaml "$@"
