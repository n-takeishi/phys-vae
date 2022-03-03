#!/bin/bash

# architecture
phy="--range-dcoeff 5e-3 2e-1"
dec="--hidlayers-aux1-dec 64 64 --x-lnvar -13.8"
feat="--arch-feat mlp --hidlayers-feat 256 256 --num-units-feat 256"
unmix="--hidlayers-unmixer 256 256"
enc="--hidlayers-aux1-enc 64 32 --hidlayers-aux2-enc 64 32 --hidlayers-dcoeff 64 32"

# optimization
optim="--learning-rate 1e-3 --train-size 1000 --batch-size 200 --epochs 20000 --grad-clip 5.0 --intg-lev 2 --weight-decay 1e-6 --adam-eps 1e-3 --balance-kld 1.0"

# other options
others="--save-interval 9999999 --num-workers 0 --activation elu" # --cuda

# ------------------------------------------------

outdir="./out_advdif/"
options="--datadir ./data/advdif/ --outdir ${outdir} ${dec} ${feat} ${unmix} ${enc} ${optim} ${others}"

if [ "$1" = "physonly" ]; then
    # Physcs-only
    beta=1e6; gamma=1e-2
    commands="--dim-z-aux1 -1 --dim-z-aux2 -1 --balance-unmix ${gamma} --balance-dataug ${beta}"
elif [ "$1" = "nnonly0" ]; then
    # NN-only, w/o PDE
    commands="--dim-z-aux1 -1 --dim-z-aux2 5 --no-phy"
elif [ "$1" = "nnonly1" ]; then
    # NN-only, w/ PDE
    commands="--dim-z-aux1 5 --dim-z-aux2 -1 --no-phy"
elif [ "$1" = "physnn" ]; then
    # Phys+NN
    alpha=1e-1; beta=1e6; gamma=1e-2
    commands="--dim-z-aux1 4 --dim-z-aux2 -1 --balance-unmix ${gamma} --balance-dataug ${beta} --balance-lact-dec ${alpha} --balance-lact-enc ${alpha}"
else
    echo "unknown option"
    commands=""
fi

mkdir ${outdir}
python -m physvae.advdif.train ${options} ${commands}
