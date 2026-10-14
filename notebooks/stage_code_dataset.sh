rm -rf kaggle_build/pa2-code
mkdir -p kaggle_build/pa2-code
cp alignment_core.py submission_adapter.py pyproject.toml requirements.txt README.md kaggle_build/pa2-code/
cp -r scripts tests data eval_sets kaggle_build/pa2-code/
find kaggle_build/pa2-code -name "__pycache__" -type d -exec rm -rf {} +
echo '{"title": "pa2 code", "id": "mibrahim1717/pa2-code", "licenses": [{"name": "CC0-1.0"}]}' > kaggle_build/pa2-code/dataset-metadata.json
