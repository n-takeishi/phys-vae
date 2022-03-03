#!/bin/bash

# architecture
phy="--range-omega 0.392 3.53"
dec="--hidlayers-aux1-dec 64 64 --hidlayers-aux2-dec 128 128 --x-lnvar -9.0"
feat="--arch-feat mlp --hidlayers-feat 128 128 --num-units-feat 256"
unmix="--hidlayers-unmixer 128 128"
enc="--hidlayers-aux1-enc 64 32 --hidlayers-aux2-enc 64 32 --hidlayers-omega 64 32"

# optimization
optim="--learning-rate 1e-3 --train-size 1000 --batch-size 200 --epochs 5000 --grad-clip 5.0 --intg-lev 1 --weight-decay 1e-6 --adam-eps 1e-3 --balance-kld 1.0"

# other options
others="--save-interval 999999 --num-workers 0 --activation elu" # --cuda

# ------------------------------------------------

outdir="./out_pendulum/"
options="--datadir ./data/pendulum/ --outdir ${outdir} ${phy} ${dec} ${feat} ${unmix} ${enc} ${optim} ${others}"

if [ "$1" = "physonly" ]; then
    # Physcs-only
    beta=1e-1; gamma=1e-3
    commands="--dim-z-aux1 -1 --dim-z-aux2 -1 --balance-unmix ${gamma} --balance-dataug ${beta}"
elif [ "$1" = "nnonly0" ]; then
    # NN-only, w/o ODE
    commands=" --dim-z-aux1 -1 --dim-z-aux2 4 --no-phy"
elif [ "$1" = "nnonly1" ]; then
    # NN-only, w/ ODE
    commands="--dim-z-aux1 2 --dim-z-aux2 2 --no-phy"
elif [ "$1" = "physnn" ]; then
    # Phys+NN
    alpha=1e-2; beta=1e-1; gamma=1e-3  
    commands="--dim-z-aux1 1 --dim-z-aux2 2 --balance-unmix ${gamma} --balance-dataug ${beta} --balance-lact-dec ${alpha} --balance-lact-enc ${alpha}"
else
    echo "unknown option"
    commands=""
fi

mkdir ${outdir}
python -m physvae.pendulum.train ${options} ${commands}
