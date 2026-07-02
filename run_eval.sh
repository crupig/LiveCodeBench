GENERATIONS_PATH=$1
NUM_WORKERS=$2
OUTPUT_PATH="${GENERATIONS_PATH/generations/generations-tested}"
OUTPUT_PATH="${OUTPUT_PATH/\.jsonl/\.eval_results\.json}"
echo "OUTPUT PATH: $OUTPUT_PATH"

if [ -z "$NUM_WORKERS" ]; then
    NUM_WORKERS=8
fi

python -m lcb_runner.runner.main_eval \
    --generations_path $GENERATIONS_PATH \
    --scenario codegeneration \
    --num_process_evaluate $NUM_WORKERS \
    --timeout 120