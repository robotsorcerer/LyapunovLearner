#!/bin/sh
echo "Move to home position"
./scripts/armctl.py --servo ON --joint=all
./scripts/armctl.py --mode 20 --joint=all
./scripts/armctl.py --tc --joint=all
./scripts/armctl.py --tpts 0 5 --joint=1
./scripts/armctl.py --tpts 0 5 --joint=2
./scripts/armctl.py --tpts 0 5 --joint=3
./scripts/armctl.py --tpts 0 5 --joint=4
./scripts/armctl.py --tpts 0 5 --joint=5
./scripts/armctl.py --tpts 0 5 --joint=6
./scripts/armctl.py --tpts 0 5 --joint=7
./scripts/armctl.py --ts --joint=all
