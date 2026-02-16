#! /bin/zsh

case "$1" in 
    train)
        echo "start to train..."
        PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python deep/main.py
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
