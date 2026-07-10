$ErrorActionPreference = "Stop"
python src/preprocessing/prepare_data.py
python src/models/train_baselines.py
python src/graph/build_pyg_data.py
python src/models/train_gat.py
python src/evaluation/compare_models.py
Write-Host "Pipeline completed!"
