CAREER ODDS MODEL: ELITE STUDY VS HIGH-LEVEL SOCCER
===================================================

Files
-----
career_odds_report.tex
    Full LaTeX report. It automatically imports outputs/results_snippet.tex after you run the Python simulation.

career_pathway_simulation.py
    Full Python code for Monte Carlo simulation, sensitivity tables, plots, and ML feature importance.

requirements.txt
    Python packages to install.

outputs/
    Example output generated with:
        python career_pathway_simulation.py --n 300000 --seed 42 --out outputs --ml-sample 60000

How to run
----------
1. Create a virtual environment:

   Windows:
        python -m venv venv
        venv\Scripts\activate

   macOS/Linux:
        python -m venv venv
        source venv/bin/activate

2. Install packages:

        pip install -r requirements.txt

3. Run a test:

        python career_pathway_simulation.py --n 50000 --seed 42 --out outputs --skip-ml

4. Run the recommended version:

        python career_pathway_simulation.py --n 300000 --seed 42 --out outputs

5. For more stable rare soccer estimates:

        python career_pathway_simulation.py --n 1000000 --seed 42 --out outputs



What to send back
-----------------
After you run the model on your PC, send back:

    outputs/model_summary.txt
    outputs/profile_rates.csv
    outputs/ml_feature_importances.csv

Then the conclusion can be rewritten using your actual run.
