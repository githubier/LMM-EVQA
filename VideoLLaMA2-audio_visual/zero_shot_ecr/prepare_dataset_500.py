"""Build the inference json for the 500-video test subset.

`test_data_500.csv` has NO header row; its columns are:
    Id, Title, Description, Download_link
(matching the header of the full `test_data.csv`).

Ground-truth ECR is joined from `test_truth.csv` (Id, ECR in [0, 1]).
We store ECR * 100 to stay consistent with `prepare_dataset.py`.
"""

import json
import os

import pandas as pd

VIDEO_DIR = "/workspace/LMM-EVQA/SnapUGC/test_videos"
CSV_PATH = "/workspace/LMM-EVQA/SnapUGC/test_data_500.csv"
TRUTH_CSV_PATH = "/workspace/LMM-EVQA/SnapUGC/test_truth.csv"
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_500.json")

COLUMNS = ["Id", "Title", "Description", "Download_link"]


def create_conversation(title, description):
    title = "None" if pd.isna(title) else title
    description = "None" if pd.isna(description) else description
    return [
        {
            "from": "human",
            "value": (
                f"<video>\nHow would you judge the engagement continuation rate of the "
                f"given content, where engagement continuation rate represents the "
                f"probability of watch time exceeding 5 seconds. The title of the video "
                f"is {title}, and the description of the video is {description}"
            ),
        },
        {"from": "gpt", "value": "The engagement continuation rate of the video."},
    ]


def main():
    df = pd.read_csv(CSV_PATH, header=None, names=COLUMNS, dtype=str)
    truth = pd.read_csv(TRUTH_CSV_PATH)

    df = df.merge(truth, on="Id", how="inner")
    print(f"rows after joining ground truth: {len(df)}")

    entries, missing = [], 0
    for _, row in df.iterrows():
        video_path = os.path.join(VIDEO_DIR, f"{row['Id']}.mp4")
        if not os.path.exists(video_path):
            missing += 1
            continue
        entries.append(
            {
                "id": row["Id"],
                "ECR": float(row["ECR"]) * 100,
                "video": video_path,
                "conversations": create_conversation(row["Title"], row["Description"]),
            }
        )

    print(f"valid samples: {len(entries)} (missing videos: {missing})")
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
