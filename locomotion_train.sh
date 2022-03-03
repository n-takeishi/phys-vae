#!/bin/bash

# architecture
phy="--dt 0.02 --hidlayers-H 128 128"
dec="--hidlayers-aux2-dec 512 512 --x-lnvar -13.8"
feat="--arch-feat mlp --hidlayers-feat 512 512 --num-units-feat 512"
enc="--hidlayers-init-yy 64 32 --hidlayers-aux2-enc 64 32"

# optimization
optim="--learning-rate 1e-3 --train-size 400 --batch-size 100 --epochs 3000 --grad-clip 5.0 --intg-lev 1 --weight-decay 1e-6 --adam-eps 1e-3 --balance-kld 1.0"

# other options
others="--save-interval 9999999 --num-workers 0 --activation elu" # --cuda

# ------------------------------------------------

outdir="./out_locomotion/"
options="--datadir ./data/locomotion/ --outdir ${outdir} ${phy} ${dec} ${feat} ${enc} ${optim} ${others}"

dimy=3; dimz=15

if [ "$1" = "nnonly" ]; then
    # NN-only
    commands="${options} --dim-y ${dimy} --dim-z-aux2 ${dimz} --no-phy"
elif [ "$1" = "physnn" ]; then
    # Phys+NN
    alpha=1e-1
    commands="${options} --dim-y ${dimy} --dim-z-aux2 ${dimz} --balance-lact-dec ${alpha} --balance-lact-enc ${alpha}"
else
    echo "unknown option"
    commands=""
fi

mkdir ${outdir}
python -m physvae.locomotion.train ${options} ${commands}
