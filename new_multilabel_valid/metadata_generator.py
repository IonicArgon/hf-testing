import csv
import os

def generate_metadata(dataset_path, output_path):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file_name",
            "labels",
            "giant_panda",
            "grizzly_bear",
            "polar_bear",
            "eating",
            "sitting",
            "standing"
        ])

        for bear in os.listdir(dataset_path):
            bear_path = os.path.join(dataset_path, bear)
            for activity in os.listdir(bear_path):
                activity_path = os.path.join(bear_path, activity)
                for img in os.listdir(activity_path):
                    img_path = os.path.join(activity_path, img)
                    img_path = img_path.replace(dataset_path + "/", "")

                    # get the bear and activity
                    bear_activity = img_path.split("/")
                    bear = bear_activity[0]
                    activity = bear_activity[1]

                    # encode into one-hot vector
                    one_hot = [0] * 6
                    bear_index = ["giant_panda", "grizzly_bear", "polar_bear"].index(bear)
                    activity_index = ["eating", "sitting", "standing"].index(activity) + 3
                    one_hot[bear_index] = 1
                    one_hot[activity_index] = 1

                    labels = [bear, activity]
                    labels_string = " ".join(labels)

                    writer.writerow([img_path, labels_string] + one_hot)

if __name__ == "__main__":
    dataset_path = "training_data"
    output_path = "metadata.csv"
    generate_metadata(dataset_path, output_path)
