#! /bin/zsh

case "$1" in 
    train)
        echo "start to train..."
        CUDA_VISIBLE_DEVICES=1 python dp_single_version2/main.py
    ;;
    check_env)
        echo "check env..."
        python metadrive/custom2_version2/create_env.py
    ;;
    plots)
        echo "plot rewards..."
        python dp_single_version2/plots.py
    ;;
    plotsims)
        echo "plot rewards for simulator..."
        python dp_single_version2/plot_sims.py
    ;;
    eval)
        echo "evaluate policy from newest checkpoint..."
        CUDA_VISIBLE_DEVICES=1 python dp_single_version2/eval.py --type newest
    ;;
    best_eval)
        echo "evaluate best policy from checkpoint..."
        CUDA_VISIBLE_DEVICES=1 python dp_single_version2/eval.py --type best
    ;;
    generate)
        echo "generate code from project..."
        python generate_code.py
    ;;
    github)
        git add .
        git commit -m "update file"
        git push
    ;;
    *)
        echo "instruction error, must choose in [train, check_env, eval, best_eval], you give \`$1\`."
    ;;
esac
