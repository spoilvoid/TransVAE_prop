python scripts/predict.py \
--model transvae \
--model_ckpt checkpoints/transvae_prop.ckpt \
--mols data/test.txt \
--props data/test_qed.txt \
--save_path data/test_pred.txt