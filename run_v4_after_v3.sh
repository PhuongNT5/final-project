#!/bin/bash
# Wait for V3 to finish, then run V4
echo "Waiting for V3 (PID 16219) to finish..."
while kill -0 16219 2>/dev/null; do
    sleep 30
done
echo "V3 finished at $(date). Starting V4..."
cd /Users/nguyenthiphuong/Learning/Proposal/project/my-project
source /Users/nguyenthiphuong/Learning/Projects/final-project/venv/bin/activate
python3 -u run_v4_final.py 2>&1 | tee sweep_results/v4_final_log.txt
