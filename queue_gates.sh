#!/bin/bash
# Wait for sweep PID 37641 to finish, then run gate checks
while kill -0 37641 2>/dev/null; do
    sleep 30
done
bash /Users/test/error-propagation-agents/run_gate_checks.sh
