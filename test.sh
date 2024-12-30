MODEL_KEY=deepseek-ai/deepseek-coder-1.3b-base
DATASET=humaneval
SAVE_PATH="/scratch-shared/fshi/xft-data/evalplus-$(basename $MODEL)-$DATASET.jsonl"

# remove previous saved result
rm -rf "/scratch-shared/fshi/xft-data/*eval_results.json"

echo "eval xft $model ..."
CUDA_VISIBLE_DEVICES=0 python experiments/text2code.py \
  --model_key $MODEL_KEY \
  --model_name_or_path $MODEL \
  --save_path $SAVE_PATH \
  --dataset $DATASET \
  --temperature 0.0 \
  --top_p 1.0 \
  --max_new_tokens 512 \
  --n_problems_per_batch 16 \
  --n_samples_per_problem 1 \
  --n_batches 1

evalplus.evaluate --dataset "$DATASET" --samples "$SAVE_PATH" --min-time-limit 5
