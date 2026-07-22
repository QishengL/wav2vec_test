#!/bin/bash
# Stage 2 Batch Submission — Base (unique experiments, no duplicates)
DIR="/nfs/qisheng/github/wav2vec_test/slurm_config/lora/26_6_5_base"

echo "=== S2 Base Batch 1/5 ==="
sbatch $DIR/same-family_mt_arbase.sh
sbatch $DIR/same-family_af_enbase.sh
sbatch $DIR/same-family_da_enbase.sh
sbatch $DIR/same-family_ky_trbase.sh
sbatch $DIR/same-family_tk_trbase.sh
sbatch $DIR/same-family_kk_trbase.sh
sbatch $DIR/same-family_sk_rubase.sh

echo "=== S2 Base Batch 2/5 ==="
sbatch $DIR/same-family_id_tabase.sh
sbatch $DIR/phoible_mt_hubase.sh
sbatch $DIR/phoible_af_csbase.sh
sbatch $DIR/phoible_da_frbase.sh
sbatch $DIR/phoible_tk_ugbase.sh
sbatch $DIR/phoible_kk_ugbase.sh
sbatch $DIR/phoible_sk_lvbase.sh
sbatch $DIR/phoible_id_swbase.sh

echo "=== S2 Base Batch 3/5 ==="
sbatch $DIR/wiki-js_mt_itbase.sh
sbatch $DIR/wiki-js_af_nlbase.sh
sbatch $DIR/wiki-js_da_nlbase.sh
sbatch $DIR/wiki-js_ky_babase.sh
sbatch $DIR/wiki-js_tk_babase.sh
sbatch $DIR/wiki-js_kk_ttbase.sh
sbatch $DIR/wiki-js_sk_robase.sh

echo "=== S2 Base Batch 4/5 ==="
sbatch $DIR/wiki-js_id_cabase.sh
sbatch $DIR/classifier_mt_enbase.sh
sbatch $DIR/classifier_ky_ugbase.sh
sbatch $DIR/classifier_kk_babase.sh
sbatch $DIR/classifier_sk_ltbase.sh
sbatch $DIR/classifier_id_csbase.sh
# af_100-nlbas already submitted in batch 3 (wiki-js), skip duplicate

echo "=== S2 Base Batch 5/5 ==="
sbatch $DIR/contrastive_mt_eobase.sh
sbatch $DIR/contrastive_sk_csbase.sh

echo "Done! 30 S2 Base jobs submitted."
