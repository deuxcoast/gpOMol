module load python/3.11-24.1.0
python -m venv gpomol
gpomol source/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name gpomol --display-name gpomol
