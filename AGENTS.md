# Agent notes

## JarvisLabs (`jl`)

- **Waiting for a `jl run` to finish.** `jl run status <run_id> --json` reports `state` (`running`, `succeeded`, `failed`) and `exit_code` (`null` while running). It has no `run_exit_code` field; that one is in `jl run logs --json`. A wait loop that checks a field the command never prints never ends, and the run's finish goes unnoticed. On 2026-11-12 that hid a finished smoke run for about 45 minutes and wasted the user's precious $0.30. Stop the loop on `exit_code` not null, or on `state` other than `running`:

  ```bash
  until s=$(jl run status "$RUN" --json 2>/dev/null) && python3 -c '
  import json, sys; d = json.loads(sys.argv[1])
  sys.exit(0 if d.get("exit_code") is not None or d.get("state") != "running" else 1)' "$s"; do sleep 60; done
  ```

- **Test a wait loop before relying on it.** Run its stop condition once on a run that has finished (it must stop) and once on a live run (it must keep waiting). A wait that never sends its notification is a fault, not a slow run. Look at the logs directly instead of waiting longer.
