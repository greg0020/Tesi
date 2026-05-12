"""
Walk-Forward Analysis for DRL Crack Spread Strategy

Expanding windows:
WF1 -> Train: 2000-2012 | Test: 2013-2015
WF2 -> Train: 2000-2015 | Test: 2016-2018
WF3 -> Train: 2000-2018 | Test: 2019-2021
"""

import os
import subprocess
import pandas as pd

FULL_DATA_PATH = "Data/naphtha_crack_full.csv"

TRAIN_SCRIPT = "train.py"
EVAL_SCRIPT = "evaluate_and_compare.py"

OUTPUT_DIR = "walk_forward_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


walk_forward_windows = [

    {
        "label": "WF1",
        "train_end": "2012-12-31",
        "test_start": "2013-01-01",
        "test_end": "2015-12-31"
    },

    {
        "label": "WF2",
        "train_end": "2015-12-31",
        "test_start": "2016-01-01",
        "test_end": "2018-12-31"
    },

    {
        "label": "WF3",
        "train_end": "2018-12-31",
        "test_start": "2019-01-01",
        "test_end": "2021-12-31"
    }
]


df = pd.read_csv(FULL_DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])

summary_rows = []

for wf in walk_forward_windows:

    label = wf["label"]

    print("\n" + "=" * 70)
    print("RUNNING WALK-FORWARD:", label)
    print("=" * 70)


    train_df = df[
        df["Date"] <= wf["train_end"]
    ].copy()

    test_df = df[
        (df["Date"] >= wf["test_start"])
        &
        (df["Date"] <= wf["test_end"])
    ].copy()


    wf_dir = os.path.join(OUTPUT_DIR, label)

    os.makedirs(wf_dir, exist_ok=True)

    train_path = os.path.join(wf_dir, "train.csv")
    test_path = os.path.join(wf_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)


    model_dir = os.path.join(wf_dir, "model")

    train_cmd = [

        "python",
        TRAIN_SCRIPT,

        "--train_data", train_path,

        "--save_dir", model_dir,

        "--transaction_cost", "0.00001"
    ]

    subprocess.run(train_cmd, check=True)

    
    run_dirs = [

        os.path.join(model_dir, d)

        for d in os.listdir(model_dir)

        if os.path.isdir(os.path.join(model_dir, d))
    ]

    latest_run_dir = max(
        run_dirs,
        key=os.path.getmtime
    )

    model_path = os.path.join(
        latest_run_dir,
        "final_model.pt"
    )


    eval_dir = os.path.join(wf_dir, "evaluation")

    eval_cmd = [

        "python",
        EVAL_SCRIPT,

        "--model_path", model_path,

        "--test_data", test_path,

        "--output_dir", eval_dir,

        "--transaction_cost", "0.00001"
    ]

    subprocess.run(eval_cmd, check=True)

    comparison_path = os.path.join(
        eval_dir,
        "comparison_table.csv"
    )

    comp_df = pd.read_csv(comparison_path)

    metrics = {
        row["Metric"]: row["DRL_Agent"]
        for _, row in comp_df.iterrows()
    }

    summary_rows.append({

        "Window":
            label,

        "Train_End":
            wf["train_end"],

        "Test_Period":
            "{} -> {}".format(
                wf["test_start"],
                wf["test_end"]
            ),

        "Total_Return":
            metrics.get("Total Return"),

        "Sharpe":
            metrics.get("Sharpe Ratio"),

        "Sortino":
            metrics.get("Sortino Ratio"),

        "Max_Drawdown":
            metrics.get("Max Drawdown"),

        "Trades":
            metrics.get("N. Trades"),

        "Win_Rate":
            metrics.get("Win Rate")
    })


summary_df = pd.DataFrame(summary_rows)

summary_path = os.path.join(
    OUTPUT_DIR,
    "walk_forward_summary.csv"
)

summary_df.to_csv(summary_path, index=False)

print("\n")
print("=" * 80)
print("WALK-FORWARD SUMMARY")
print("=" * 80)

print(summary_df)

print("=" * 80)

print("\nResults saved in:")
print(OUTPUT_DIR)