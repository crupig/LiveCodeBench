# ./run_gen_tests.sh
DEVICE=$1
MODEL_PATH=$2
NUM_SAMPLES=2
TEMPERATURE=1.0
MAX_TOKENS=768
RELEASE_VERSION=release_v6

IDS_FILE="../../constants/ids_train_val_test.json"

CUDA_VISIBLE_DEVICES=$DEVICE python -m lcb_runner.runner.main_gen_tests \
    --model $MODEL_PATH \
    --scenario codegeneration \
    --n $NUM_SAMPLES \
    --temperature $TEMPERATURE \
    --max_tokens $MAX_TOKENS \
    --stop null \
    --release_version $RELEASE_VERSION \
    --all_ids_dict $IDS_FILE \
    --split test # train, val, test, or all
