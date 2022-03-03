#!/bin/bash

opt="--n-grids 20 --len-episode 50 --dx 0.1 --dt 0.02 --noise-std 1e-3 --outdir ./"
range="--range-init-mag 0.5 1.5 --range-dcoeff 1e-2 1e-1 --range-ccoeff 1e-2 1e-1"

python generate.py ${opt} --name train --n-samples 1000 --seed 1234 ${range}
python generate.py ${opt} --name valid --n-samples 500 --seed 1235 ${range}
python generate.py ${opt} --name test --n-samples 1000 --seed 1236 ${range}
