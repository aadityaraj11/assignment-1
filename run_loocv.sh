#!/bin/bash
export LLM_PROVIDER=gemini
export GEMINI_MODEL=gemini-3.5-flash-lite
cd /home/aaditya/debu/userready/project
python3 -u scripts/loocv.py > /home/aaditya/debu/userready/project/loocv3.log 2>&1
echo "Finished with exit code $?" >> /home/aaditya/debu/userready/project/loocv3.log
