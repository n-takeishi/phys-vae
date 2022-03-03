#!/bin/bash

# architecture
phy="--range-I0 0.5 1.0 --range-A 0.1 1.0 --range-e 0.2 0.8 --range-theta 0.0 3.142"
dec="--x-lnvar -11 --unet-size 2"
feat="--num-units-feat 512"
unmix=""
enc="--hidlayers-aux2-enc 256 128 --hidlayers-z-phy 64 32"

# optimization
optim="--learning-rate 1e-3 --train-size 400 --valid-size 100 --data-augmentation 20 --batch-size 200 --epochs 10000 --grad-clip 5.0 --weight-decay 1e-6 --adam-eps 1e-3 --balance-kld 1.0 --epochs-pretrain 0"

# other options
others="--save-interval 9999999 --num-workers 0 --activation elu --cuda"

# ------------------------------------------------

outdir="./out_galaxy/"
options="--datadir ./data/galaxy/ --outdir ${outdir} ${dec} ${feat} ${unmix} ${enc} ${optim} ${others}"

dimz=2

if [ "$1" = "nnonly" ]; then
    # NN-only
    commands="--dim-z-aux2 $((${dimz}+4)) --no-phy"
elif [ "$1" = "physnn" ]; then
    # Phys+NN
    alpha=1e-1; beta=1e3; gamma=1e0
    commands="--dim-z-aux2 ${dimz} --balance-unmix ${gamma} --balance-dataug ${beta} --balance-lact-dec ${alpha} --balance-lact-enc ${alpha}"
fi

mkdir ${outdir}
python -m physvae.galaxy.train ${options} ${commands}
