import logging
import neptune
from neptune.utils import stringify_unsupported
import numpy as np

NEPTUNE_INFO = {}
NEPTUNE_INFO["benchmark"] = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiMmRkMWRjNS03ZGUwLTQ1MzQtYTViOS0yNTQ3MThlY2Q5NzUifQ=="
NEPTUNE_INFO["sc-musketeers"] = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1Zjg5NGJkNC00ZmRkLTQ2NjctODhmYy0zZDAzYzM5ZTgxOTAifQ=="
NEPTUNE_INFO["scmusk-review"] = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1Zjg5NGJkNC00ZmRkLTQ2NjctODhmYy0zZDAzYzM5ZTgxOTAifQ=="
NEPTUNE_INFO["scmusk-scheme"] = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1Zjg5NGJkNC00ZmRkLTQ2NjctODhmYy0zZDAzYzM5ZTgxOTAifQ=="
NEPTUNE_INFO["scmusk-tasks"] = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1Zjg5NGJkNC00ZmRkLTQ2NjctODhmYy0zZDAzYzM5ZTgxOTAifQ=="
NEPTUNE_INFO["scmusk-hp"] = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1Zjg5NGJkNC00ZmRkLTQ2NjctODhmYy0zZDAzYzM5ZTgxOTAifQ=="

logger = logging.getLogger("Sc-Musketeers")

def start_neptune_log(workflow):
    logger.info(f"Use Neptune.ai log : {workflow.run_file.log_neptune}")
    if workflow.run_file.log_neptune:
        logger.info(f"Use Neptune project name = {workflow.run_file.neptune_name}")
        if workflow.run_file.neptune_name == "benchmark":
            workflow.run_neptune = neptune.init_run(
                project="becavin-lab/benchmark",
                api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiMmRkMWRjNS03ZGUwLTQ1MzQtYTViOS0yNTQ3MThlY2Q5NzUifQ==",
            )
        elif workflow.run_file.neptune_name == "scmusk-hp":
            workflow.run_neptune = neptune.init_run(
                project="becavin-lab/scmusk-hp",
                api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1Zjg5NGJkNC00ZmRkLTQ2NjctODhmYy0zZDAzYzM5ZTgxOTAifQ==",
            )
        elif workflow.run_file.neptune_name == "scmusk-scheme":
            workflow.run_neptune = neptune.init_run(
                project="becavin-lab/scmusk-scheme",
                api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1Zjg5NGJkNC00ZmRkLTQ2NjctODhmYy0zZDAzYzM5ZTgxOTAifQ==",
            )
        elif workflow.run_file.neptune_name == "scmusk-tasks":
            workflow.run_neptune = neptune.init_run(
                project="becavin-lab/scmusk-tasks",
                api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1Zjg5NGJkNC00ZmRkLTQ2NjctODhmYy0zZDAzYzM5ZTgxOTAifQ==",
            )
        elif workflow.run_file.neptune_name == "sc-musketeers":
            workflow.run_neptune = neptune.init_run(
                project="becavin-lab/sc-musketeers",
                api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1Zjg5NGJkNC00ZmRkLTQ2NjctODhmYy0zZDAzYzM5ZTgxOTAifQ==",
            )
        elif workflow.run_file.neptune_name == "scmusk-review":
            workflow.run_neptune = neptune.init_run(
                project="becavin-lab/scmusk-review",
                api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1Zjg5NGJkNC00ZmRkLTQ2NjctODhmYy0zZDAzYzM5ZTgxOTAifQ==",
            )
        else:
            logger.info("No neptune_name was provided !!!")

        for par, val in workflow.run_file.__dict__.items():
            if par in dir(workflow):
                workflow.run_neptune[f"parameters/{par}"] = (
                    stringify_unsupported(getattr(workflow, par))
                )
            # elif par in dir(workflow.ae_param):
            #    workflow.run_neptune[f"parameters/{par}"] = stringify_unsupported(getattr(workflow.ae_param, par))

   
        if hasattr(workflow.run_file, 'hp_params'):
            for par, val in workflow.hp_params.items():
                workflow.run_neptune[f"parameters/{par}"] = (
                    stringify_unsupported(val)
                )


def add_custom_log(workflow, name, value):
    if workflow.run_file.log_neptune:
        workflow.run_neptune[f"parameters/{name}"] = stringify_unsupported(value)


def stop_neptune_log(workflow):
    if workflow.run_file.log_neptune:
        workflow.run_neptune.stop()

def load_run_df(task):
    print("Run neptune benchmark")
    project = neptune.init_project(
            project="becavin-lab/benchmark",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiMmRkMWRjNS03ZGUwLTQ1MzQtYTViOS0yNTQ3MThlY2Q5NzUifQ==",
        mode="read-only",
            )# For checkpoint

    print(f"Load df with task = {task}")
    if task != "":
        print(f'`sys/failed`:string = "{task}"')
        runs_table_df = project.fetch_runs_table(query = f'sys/failed:bool = "{task}"').to_pandas()
    else:
        runs_table_df = project.fetch_runs_table().to_pandas()
    project.stop()

    f =  lambda x : x.replace('evaluation/', '').replace('parameters/', '').replace('/', '_')
    runs_table_df.columns = np.array(list(map(f, runs_table_df.columns)))
    return runs_table_df


def get_table_benchmark():
    import neptune
    import pandas as pd

    # 1. Initialize your project
    # Replace "YOUR_WORKSPACE/YOUR_PROJECT" with your actual workspace and project name
    # Make sure your NEPTUNE_API_TOKEN environment variable is set, or pass it here.
    project = neptune.init_project(
                project="becavin-lab/benchmark",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiMmRkMWRjNS03ZGUwLTQ1MzQtYTViOS0yNTQ3MThlY2Q5NzUifQ==",
            mode="read-only",
                )# For checkpoint
    # Directory to save the Excel files
    output_directory = './neptune_runs_by_year_optimized'
    import os
    os.makedirs(output_directory, exist_ok=True)

    # 1. Define the base query to fetch all successful runs
    base_query = '`sys/failed`:bool = False'

    # 2. Pagination settings
    batch_size = 5000  # Number of runs to fetch in each API call
    all_successful_runs_data = [] # List to store DataFrames from each batch
    offset = 0
    total_runs_fetched = 0

    print(f"Starting to fetch all successful runs in batches of {batch_size}...")

    while True:
        try:
            print(f"Fetching batch: offset={offset}")
            current_batch_df = project.fetch_runs_table(
                query=base_query,
                offset=offset,
                limit=batch_size
            ).to_pandas()

            if current_batch_df.empty:
                print("No more runs to fetch. Breaking loop.")
                break # No more runs to fetch

            all_successful_runs_data.append(current_batch_df)
            total_runs_fetched += len(current_batch_df)
            print(f"Fetched {len(current_batch_df)} runs in this batch. Total fetched: {total_runs_fetched}")

            if len(current_batch_df) < batch_size:
                # If the last batch was smaller than batch_size, it means we've fetched all available runs
                print("Last batch was smaller than batch size, likely reached end of runs.")
                break

            offset += batch_size
            # Add a small delay to avoid hitting API rate limits, especially for very large fetches
            time.sleep(0.5)

        except Exception as e:
            print(f"An error occurred during batch fetching at offset {offset}: {e}")
            print("Attempting to continue or break if error persists.")
            # Depending on the error, you might want to break or retry.
            break # Breaking for now to prevent infinite loops on persistent errors

    print(f"\n--- Finished fetching all successful runs. Total runs fetched: {total_runs_fetched} ---")

    if not all_successful_runs_data:
        print("No runs were fetched in total. Exiting.")
        project.stop()
        exit()

    # 3. Concatenate all batches into a single large DataFrame
    all_runs_df = pd.concat(all_successful_runs_data, ignore_index=True)
    print(f"Combined all fetched runs into a single DataFrame of {len(all_runs_df)} entries.")

    # 4. Filter and save by year using the combined DataFrame
    years_to_extract = [2025, 2024, 2023, 2022, 2021, 2020, 2019] # Add or remove years as needed

    print("\n--- Processing and saving runs by year ---")
    for year in years_to_extract:
        # Get the start and end of the year in UTC
        start_of_year = pd.Timestamp(f'{year}-01-01', tz='UTC')
        end_of_year = pd.Timestamp(f'{year + 1}-01-01', tz='UTC')

        # Filter the combined DataFrame locally
        # Ensure 'sys/creation_time' is a datetime object, convert if necessary
        if not pd.api.types.is_datetime64_any_dtype(all_runs_df['sys/creation_time']):
            all_runs_df['sys/creation_time'] = pd.to_datetime(all_runs_df['sys/creation_time'], utc=True)

        yearly_df = all_runs_df[
            (all_runs_df['sys/creation_time'] >= start_of_year) &
            (all_runs_df['sys/creation_time'] < end_of_year)
        ].copy() # Use .copy() to avoid SettingWithCopyWarning

        if not yearly_df.empty:
            print(f"Found {len(yearly_df)} runs for year {year}.")
            excel_filename = os.path.join(output_directory, f'neptune_runs_{year}.xlsx')
            yearly_df.to_excel(excel_filename, index=False, sheet_name=f'Runs_{year}')
            print(f"Saved {len(yearly_df)} runs to '{excel_filename}'")
        else:
            print(f"No runs found for year {year}. No Excel file will be created for this year.")

    # Don't forget to stop the project connection
    project.stop()

    print("\nAll requested years processed and saved. Check the 'neptune_runs_by_year_optimized' directory.")
