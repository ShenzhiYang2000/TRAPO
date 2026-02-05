ROOT=/ossfs/workspace/aitech_aidata/code/TraPO
TASK=math # math arc_c gpqa mmlu_pro 
CKPT=labeled_1k_unlabeled_1k_ood

DATA=$ROOT/data/valid.$TASK.parquet
OUTPUT_DIR=$ROOT/results/$CKPT-$TASK/
mkdir -p $OUTPUT_DIR

# If you want to evaluate other models, you can change the model path and name.
MODEL_PATH=$ROOT/trapo/verl/checkpoints/semi-grpo-indomain/$CKPT/best/actor

if [ $MODEL_NAME == "eurus-2-7b-prime-zero" ]; then 
  TEMPLATE=prime
elif [ $MODEL_NAME == "simple-rl-zero" ]; then
  TEMPLATE=qwen
else
  TEMPLATE=own
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 python $ROOT/eval_scripts/generate_vllm.py \
  --model_path $MODEL_PATH \
  --input_file $DATA \
  --remove_system True \
  --add_oat_evaluate True \
  --output_file $OUTPUT_DIR/$MODEL_NAME.jsonl \
  --template $TEMPLATE > $OUTPUT_DIR/$MODEL_NAME.log

