#!/bin/bash
# Move to script directory
cd "$(dirname "$0")"

echo "============================================================"
echo " Starting PPG Drag-and-Drop Analysis Pipeline Server... "
echo "============================================================"
echo "Active environment: /Users/su-younlee/miniforge3/envs/ppg-stress"
echo "Web URL: http://localhost:8050"
echo "Press Ctrl+C to stop the server."
echo "============================================================"
echo ""

/Users/su-younlee/miniforge3/envs/ppg-stress/bin/python3 server.py
