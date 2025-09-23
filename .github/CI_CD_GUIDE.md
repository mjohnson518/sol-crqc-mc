# CI/CD Guide

## Overview

This repository uses GitHub Actions for continuous integration and testing. **Important: No simulation results are ever committed to the repository.**

## Workflows

### 1. Tests (`tests.yml`)
- **Triggers:** Push to main/develop, Pull Requests
- **Purpose:** Run test suite, check code quality
- **Matrix:** Python 3.9, 3.10, 3.11
- **Artifacts:** Coverage reports only

### 2. Simulation (`simulation.yml`)
- **Triggers:** Manual dispatch only
- **Purpose:** Run full simulations in cloud
- **Important:** Results are NOT committed to repo
- **Storage:** 
  - Artifacts are temporary (30 days)
  - Only summaries uploaded (<10MB)
  - Full results stay on runner

### 3. Dependencies (`dependencies.yml`)
- **Triggers:** Weekly (Mondays 3am UTC), Manual
- **Purpose:** Security scanning, update checks
- **Actions:** Creates issues for outdated packages

## Storage Policy

### What Gets Committed ❌
- NEVER commit `simulation_results/` directory
- Already in `.gitignore` for safety
- Full simulation runs can be 100MB-10GB+

### What Gets Uploaded as Artifacts ✅
- Summary reports (markdown)
- Key visualizations (dashboard PNGs)
- Configuration files (JSON)
- Total: <10MB per run

### Why This Matters
- **Repository size:** Keeps repo fast to clone
- **GitHub limits:** Free tier = 5GB artifact storage
- **Performance:** Large files slow everything down

## Running Simulations

### Locally (Recommended for Large Runs)
```bash
python run_simulation.py --iterations 10000
# Results stay on your machine
```

### Via GitHub Actions (For CI/Testing)
1. Go to Actions tab
2. Select "Run Simulation"
3. Click "Run workflow"
4. Set iterations (keep <5000 for GitHub)
5. Download summary from Artifacts

## Artifact Retention

| Artifact Type | Retention | Size Limit |
|--------------|-----------|------------|
| Test Coverage | 90 days | 50MB |
| Simulation Summary | 30 days | 10MB |
| PR Comments | Permanent | 1MB |

## Best Practices

1. **Large simulations:** Run locally, not in CI
2. **Testing:** Use small iterations (100-1000)
3. **Storage:** Download artifacts you need promptly
4. **Cleanup:** Artifacts auto-delete after retention period

## Questions?

- Artifacts are temporary cloud storage
- They do NOT affect repository size
- They are NOT version controlled
- They ARE useful for sharing results without committing
