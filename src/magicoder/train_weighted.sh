set -e
set -x 

MODEL_KEY=deepseek-ai/deepseek-coder-1.3b-base
MODEL_NAME_OR_PATH="$SCARTCH_DIR/ds-8x1.3b-top-6-universal-evol-instruct-5e-5_bs_64_epoch_4_weighted_dense"
OUTPUT_DIR="$SCARTCH_DIR/ds-8x1.3b-top-6-universal-evol-instruct-5e-5_bs_64_epoch_4_weighted_dense-lambda-75-1e-5_bs_64_epoch_1"

if [ -d "$OUTPUT_DIR" ]; then
  echo "Directory $OUTPUT_DIR exists. Skip training ..."
else
  echo "Directory $OUTPUT_DIR does not exist."
  CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29500 train.py \
      --model_key $MODEL_KEY \
      --model_name_or_path $MODEL_NAME_OR_PATH \
      --use_flash_attention True \
      --use_weighted True \
      --datafile_paths \
          $DATA_DIR \
      --output_dir $OUTPUT_DIR \
      --bf16 True \
      --num_train_epochs 1 \
      --per_device_train_batch_size 2 \
      --gradient_accumulation_steps 8 \
      --max_training_seq_length 1024 \
      --group_by_length False \
      --ddp_find_unused_parameters False \
      --logging_steps 1 \
      --log_level info \
      --save_strategy no \
      --save_steps 20 \
      --optim adafactor \
      --max_grad_norm -1 \
      --warmup_steps 125 \
      --learning_rate 1e-5 \
      --lr_scheduler_type linear
fi