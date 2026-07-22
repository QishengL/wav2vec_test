#!/bin/bash
# Stage 2 Batch Submission — XLSR-53 (unique experiments, no duplicates)
DIR="/nfs/qisheng/github/wav2vec_test/slurm_config/lora/26_6_5"

echo "=== S2 53 Batch 1/5 ==="
sbatch $DIR/same-family_mt_ar53.sh
sbatch $DIR/same-family_af_en53.sh
sbatch $DIR/same-family_da_en53.sh
sbatch $DIR/same-family_ky_tr53.sh
sbatch $DIR/same-family_tk_tr53.sh
sbatch $DIR/same-family_kk_tr53.sh
sbatch $DIR/same-family_sk_ru53.sh

echo "=== S2 53 Batch 2/5 ==="
sbatch $DIR/same-family_id_ta53.sh
sbatch $DIR/phoible_mt_hu53.sh
sbatch $DIR/phoible_af_cs53.sh
sbatch $DIR/phoible_da_fr53.sh
sbatch $DIR/phoible_tk_ug53.sh
sbatch $DIR/phoible_kk_ug53.sh
sbatch $DIR/phoible_sk_lv53.sh

echo "=== S2 53 Batch 3/5 ==="
sbatch $DIR/wiki-js_mt_it53.sh
sbatch $DIR/wiki-js_af_nl53.sh
sbatch $DIR/wiki-js_da_nl53.sh
sbatch $DIR/wiki-js_ky_ba53.sh
sbatch $DIR/wiki-js_tk_ba53.sh
sbatch $DIR/wiki-js_kk_tt53.sh
sbatch $DIR/wiki-js_sk_ro53.sh

echo "=== S2 53 Batch 4/5 ==="
sbatch $DIR/wiki-js_id_ca53.sh
sbatch $DIR/classifier_mt_en53.sh
sbatch $DIR/classifier_ky_ug53.sh
sbatch $DIR/classifier_kk_ba53.sh
sbatch $DIR/classifier_sk_lt53.sh
sbatch $DIR/classifier_id_cs53.sh
# af_100-nl53 already submitted in batch 3 (wiki-js), skip duplicate

echo "=== S2 53 Batch 5/5 ==="
sbatch $DIR/phoible_id_sw53.sh
sbatch $DIR/contrastive_mt_eo53.sh
sbatch $DIR/contrastive_sk_cs53.sh

echo "Done! 30 S2 XLSR-53 jobs submitted."
