# AT-BMC
Unifying Model Explainability and Robustness for Joint Text Classification and Rationale Extraction (AAAI 2022) [Paper](https://arxiv.org/abs/2112.10424)

## Prerequisites
Install packages by referring to [pip_reqs.txt](https://github.com/crazyofapple/AT-BMC/blob/main/pip_reqs.txt)

## Datasets
- [Movie Reviews](https://github.com/crazyofapple/AT-BMC/blob/main/datasets/movie_reviews_with_some_rats_adv) (Paper: Pruthi et al. 2020, Weakly- and Semi-supervised Evidence Extraction, Findings of EMNLP)
- [MultiRC](https://github.com/crazyofapple/AT-BMC/blob/main/datasets/multirc_adv) (Paper: Khashabi et al. 2018, Looking Beyond the Surface: A Challenge
Set for Reading Comprehension over Multiple Sentences, NAACL-HLT)

## Run

### Adv Data Preparation
Please refer to `augment_with_mask.py` for data preparation. And the data folder contains the final used data.
### Training
```
for seed in `seq 1 1`; do 
    CUDA_VISIBLE_DEVICES=$GPU_ID unbuffer python main.py \
        --data_dir datasets/movie_reviews_with_some_rats_adv \
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
        --gradient_accumulation_steps 8  | tee -a logs/logs.txt
done;
```
