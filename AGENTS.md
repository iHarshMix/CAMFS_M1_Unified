# Cluster & Slurm Scheduling Guidelines

## Mandatory Hardware Constraints
- **Target Node**: Node `h100`.
- **Mandatory GRES Binding**: Strictly use `--gres=gpu:ugpg_3g47gb:1`.
  - NEVER use generic `--gres=gpu:1`.
  - NEVER use `--gres=gpu:phd_4g47gb:1` or `all_7g94gb:1`.
- **Max Account QOS Limit**: QOS `mtech` has an absolute max walltime of `03:59:00`.

## Slurm Backfill Scheduling Protocol
Before submitting or whenever monitoring a Slurm job on `h100`:
1. **Check Queue Reservations**: Run `squeue` and `scontrol show job <pending_job_id>`. Look for the nearest `StartTime` of any higher-priority pending job.
2. **Fit the Time Window**:
   - Do NOT blindly submit jobs with `--time=03:59:00` if a higher-priority reservation starts in less than 4 hours.
   - Calculate: `AvailableWindow = ReservationStartTime - CurrentTime`.
   - Set `--time` slightly below `AvailableWindow` (e.g., if window is 3h 45m, use `--time=03:30:00`).
3. **Unstick Pending Jobs**:
   - If a job is pending with `Reason=Priority` while `ugpg_3g47gb` is idle, calculate the window and immediately run:
     `scontrol update job=<job_id> TimeLimit=<calculated_window>`
   - Slurm backfill will dispatch the job within seconds.
