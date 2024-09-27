set -e
set -x 

export PYTHONPATH=:/home/fshi/xft-moe/src:/home/fshi/xft-moe/src/magicoder
MODEL_KEY=deepseek-ai/deepseek-coder-1.3b-base
MODEL=/home/fshi/xft-moe/src/magicoder/ds-8x1.3b-top-6-universal-evol-instruct-5e-5_bs_64_epoch_4
DATASET=humaneval
SAVE_PATH=./evalplus-$(basename $MODEL)-$DATASET.jsonl

CUDA_VISIBLE_DEVICES=0 python experiments/text2code.py \
  --model_key $MODEL_KEY \
  --model_name_or_path $MODEL \
  --use_moe True \
  --save_path $SAVE_PATH \
  --dataset $DATASET \
  --temperature 0.0 \
  --top_p 1.0 \
  --max_new_tokens 512 \
  --n_problems_per_batch 16 \
  --n_samples_per_problem 1 \
  --n_batches 1

evalplus.evaluate --dataset $DATASET --samples $SAVE_PATH --min-time-limit 5
