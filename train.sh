#!/bin/bash
export PYTHONPATH=`pwd`/SAN
alias python_='python'

yecho(){ # yellow echo
    echo "\e[1;33m$1\e[m"
}

arg=$1 
if [ "$arg" = "pdb" ]; then
    yecho 'Launch pdb ...'
    alias python_='python -m pdb -c c'
else 
    yecho 'Launch without pdb ...'
fi

python_ polos/cli.py train -f configs/polos-trainer.yaml && \
sh validate_cvpr.sh
