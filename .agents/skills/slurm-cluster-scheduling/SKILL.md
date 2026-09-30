---
name: slurm-cluster-scheduling
description: >-
  Runbook for inspecting Slurm cluster queue status, calculating backfill scheduling windows,
  and resolving jobs stuck in PD (Priority) or PD (Resources) on node h100.
---

# Slurm Cluster Scheduling & Backfill Runbook

Use this runbook to inspect cluster availability and guarantee instant execution of GPU jobs on node `h100`.

## 1. Inspect Node and Queue State

```bash
# Check running and pending jobs
squeue -w h100

# Check node slice allocation
scontrol show node h100 | grep -E "Gres|AllocTRES"
```

## 2. Inspect Pending Jobs and Reservations

If any higher-priority job (e.g., PhD or Faculty QOS) is pending with `(Resources)`:
```bash
scontrol show job <pending_job_id> | grep -E "StartTime|TresPerNode"
```
Note the `StartTime` (e.g., `2026-09-30T15:38:52`).

## 3. Calculate Available Backfill Window

$$\text{Window} = \text{ReservationStartTime} - \text{CurrentTime}$$

- If $\text{Window} > \text{ExpectedJobDuration}$, set:
  $$\text{TimeLimit} = \min(\text{ExpectedJobDuration} + 15\text{m}, \text{Window} - 10\text{m})$$
- Example: If current time is 11:50 and reservation is at 15:38, window is 3h 48m. Set `--time=03:30:00`.

## 4. Remediation Commands

### For New Submissions:
Specify the adjusted `--time` in the `sbatch` command line or script:
```bash
sbatch --time=03:30:00 scripts/run2_local_only/submit_h1.sbatch 17
```

### For Already Submitted Pending Jobs:
Update the job in-flight without losing your queue position:
```bash
scontrol update job=<job_id> TimeLimit=03:30:00
```
Slurm's backfill scheduler will evaluate the job on the next tick (within 30 seconds) and launch it immediately if the target slice is free.
