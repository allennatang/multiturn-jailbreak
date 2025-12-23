# multiturn-jailbreak

### Optional: Conda setup
```bash
conda create -n prompt-runner python=3.10
conda activate prompt-runner
pip install -e ".[dev]"

### To run the deepseek model 
python -m src.runner --model-name deepseek