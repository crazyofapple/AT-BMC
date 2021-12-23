#!/bin/sh
mkdir -p logs;
source $YOUR_ANACONDA_PATH/bin/activate;
conda activate $YOUR_ANACONDA_ENVIRONMENT;
OUTPUT_BASE_DIR=./models-multirc
GPU_ID=1
for seed in `seq 1 3`; do 
    CUDA_VISIBLE_DEVICES=$GPU_ID unbuffer python main.py \
        --data_dir datasets/multirc_adv \
        --batch_size 4 \
        --learning_rate 2e-5 \
        --max_seq_len 512 \
        --epochs 30 \
        --dataset multi_rc \
        --evaluate_every 10000 \
        --save_extraction_model ${OUTPUT_BASE_DIR}/ \
        --save_prediction_model ${OUTPUT_BASE_DIR}/ \
        --include_label_embedding_features \
        --seed $seed \
        --upper_case \
        --gradient_accumulation_steps 16  | tee -a \
        logs/logs_crf_match_adv_label_embedding_batch_size=4_max_seq_len=512_epochs=30_gradient_accumulation_steps=16_dataset=multirc_include_double_bert_features_upper_case_seed=$seed.txt;
done;
