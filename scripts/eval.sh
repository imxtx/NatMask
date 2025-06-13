export CUDA_VISIBLE_DEVICES=0

python eval.py \
    --device cuda:0 \
    --results_dir results/lfw \
    --save_dir results/lfw/eval

python eval.py \
    --device cuda:0 \
    --results_dir results/celeba_hq \
    --save_dir results/celeba_hq/eval