import json
from constants import PRIVACY_RESULTS_DIR, BASELINE_SUMMARY_TASK, IN_CONTEXT_SUMMARY_TASK


def run_reidentification_eval(target_model, tasks):
    for task in tasks:        
        with(open(f"{PRIVACY_RESULTS_DIR}/{task}_raw_privacy/{target_model}.json")) as f:
            all_data = json.load(f)
            priv_task = task
            # baseline_task = f"{task}{BASELINE_SUMMARY_TASK}"
            # icl_task = f"{task}{IN_CONTEXT_SUMMARY_TASK}"
            priv_data = all_data[priv_task]
            print(priv_data.keys())
        

if __name__ == '__main__':
    run_reidentification_eval('gpt-4o-mini', ['cnn'])