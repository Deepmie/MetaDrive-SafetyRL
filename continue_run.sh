#!/bin/zsh
cd /workspace/metadrive-github/
git checkout baseline/rl_mpc_cbf
run train
git checkout baseline/rl_mpc_cbf_traj
run train
git checkout exp/rl_mpc_cbf_ppc_traj
run train
echo "train successful!"