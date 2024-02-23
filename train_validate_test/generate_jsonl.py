import os
import json


def generate_jsonl(dataset_path, output_path):
    with open(output_path, "w") as f:
        for bear in os.listdir(dataset_path):
            bear_path = os.path.join(dataset_path, bear)
            for activity in os.listdir(bear_path):
                activity_path = os.path.join(bear_path, activity)
                for img in os.listdir(activity_path):
                    img_path = os.path.join(activity_path, img)
                    img_path = img_path.replace(dataset_path + "/", "")
                    f.write(
                        json.dumps({"file_name": img_path, 
                                    "bear": bear,
                                    "activity": activity,})
                        + "\n"
                    )


if __name__ == "__main__":
    dataset_path = "training_data"
    output_path = "metadata.jsonl"
    generate_jsonl(dataset_path, output_path)
