# Quantifying Error Propagation Dynamics in Multi-Step Agent Workflows

## How to Run
```bash
# Step 1:
pip install -r requirements.txt
cp .env.example .env # Then enter API keys

# Step 2:
python run.py --mode run --models llama-3.1-8b qwen-2.5-7b --trials 10 # Open source models
python run.py --mode run --use-api --trials 10 # API models
python run.py --mode run --models llama-3.1-8b gpt-4o-mini claude-haiku --trials 10 # Mixed usage

# Step 3: 
python run.py --mode analyze --results-file results/experiment_YYYYMMDD_HHMMSS.json
```

## Output Files
- `results/experiment_*.json`: Raw experiment data
- `results/error_propagation.png`: Error propagation curves
- `results/pattern_comparison.png`: Pattern fitting visualization
- `results/pattern_summary.csv`: Summary of propagation patterns

## TODO: possible figures for our reports include:
Figure 1: Error propagation curves (line plot)
Figure 2: Heatmap (model × step)
Figure 3: Pattern fitting comparison
Table 1: Critical steps by model
Table 2: Pattern classification results