# ./run_eval_tests.sh
GENERATIONS_PATH=$1
NUM_WORKERS=$2

if [ -z "$NUM_WORKERS" ]; then
    NUM_WORKERS=8
fi

echo "Processing: $GENERATIONS_PATH"
python -m lcb_runner.runner.main_eval_tests \
    --generations_path $GENERATIONS_PATH \
    --scenario codegeneration \
    --num_process_evaluate $NUM_WORKERS \
    --timeout 20