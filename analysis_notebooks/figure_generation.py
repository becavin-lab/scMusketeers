import os
import argparse
import logging

# python analysis_notebooks/figure_generation.py --process_csv 
# python analysis_notebooks/figure_generation.py --fig2
# python analysis_notebooks/figure_generation.py --fig3


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from figure_scripts.csv_processing import csv_process
from figure_scripts.generate_figure_2 import produce_fig_2
from figure_scripts.generate_figure_3 import produce_fig_3

def main():
    working_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_dir = os.path.join(working_dir, 'experiment_script/results')
    
    checkpoint_dir = os.path.join(working_dir, 'analysis_notebooks', 'csv_batches')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'metrics_checkpoint.csv')

    parser = argparse.ArgumentParser(description="Generate Figures from SC Musketeers Benchmark metrics")
    parser.add_argument("--process_csv", action="store_true", help="Process raw result metrics into unified checkpoint CSVs per task")
    parser.add_argument("--fig2", action="store_true", help="Load the Task 1 checkpoint CSV and produce Figure 2")
    parser.add_argument("--fig3", action="store_true", help="Load the Task 2 checkpoint CSV and produce Figure 3")
    parser.add_argument("--all", action="store_true", help="Run everything from start to finish")

    args = parser.parse_args()
    
    run_csv = args.process_csv or args.all
    run_fig2 = args.fig2 or args.all
    run_fig3 = args.fig3 or args.all
    
    if not (run_csv or run_fig2 or run_fig3):
        logger.info("No explicit flags passed, defaulting to '--all'.")
        run_csv = True
        run_fig2 = True
        run_fig3 = True

    if run_csv:
        logger.info("--- Starting CSV Processing ---")
        csv_process(results_dir, checkpoint_path)
        
    if run_fig2:
        logger.info("--- Starting Figure 2 (Task 1) Generation ---")
        produce_fig_2(checkpoint_path, working_dir)
        
    if run_fig3:
        logger.info("--- Starting Figure 3 (Task 2) Generation ---")
        produce_fig_3(checkpoint_path, working_dir)

if __name__ == "__main__":
    main()
