from Dataset import Dataset
import unittest

class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = Dataset(data_dir="training_data").get_dataset()

    def test_split_verification(self):
        self.assertTrue("train" in self.dataset, "train not in hf_dataset")
        self.assertTrue("test" in self.dataset, "test not in hf_dataset")
        self.assertTrue("valid" in self.dataset, "valid not in hf_dataset")

    def test_split_size_verification(self):
        size_train = len(self.dataset["train"])
        size_test = len(self.dataset["test"])
        size_valid = len(self.dataset["valid"])

        percent_train = round(size_train / (size_train + size_test + size_valid), 1)
        percent_test = round(size_test / (size_train + size_test + size_valid), 1)
        percent_valid = round(size_valid / (size_train + size_test + size_valid), 1)

        self.assertEqual(percent_train, 0.8, f"percent_train is {percent_train}, expected 0.8")
        self.assertEqual(percent_test, 0.1, f"percent_test is {percent_test}, expected 0.1")
        self.assertEqual(percent_valid, 0.1, f"percent_valid is {percent_valid}, expected 0.1")

    def test_feature_verification(self):
        print()
        for split in self.dataset:
            for i in range(len(self.dataset[split])):
                bears = ["giant_panda", "grizzly_bear", "polar_bear"]
                activities = ["eating", "sitting", "standing"]

                self.assertTrue("image" in self.dataset[split][i], f"file_name not in {split} at index {i}")
                self.assertTrue("labels" in self.dataset[split][i], f"labels not in {split} at index {i}")

                for bear in bears:
                    self.assertTrue(bear in self.dataset[split][i], f"{bear} not in {split} at index {i}")

                for activity in activities:
                    self.assertTrue(activity in self.dataset[split][i], f"{activity} not in {split} at index {i}")

                print(f"\r{i}/{len(self.dataset[split])} {split} verified {'':>15}", end="")
            print()

if __name__ == "__main__":
    # print the dataset out just to see what it looks like
    test = Dataset()
    print(test.get_dataset())

    # print out a random image from the dataset and its features
    import random
    import numpy as np
    import matplotlib.pyplot as plt

    random_image = random.choice(test.get_dataset()["train"])
    image = np.array(random_image["image"])
    labels = random_image["labels"]
    one_hot = [0] * 6

    # figure out bears
    for key in list(random_image)[2:5]:
        bear_index = ["giant_panda", "grizzly_bear", "polar_bear"].index(key)
        if random_image[key] == 1:
            one_hot[bear_index] = 1

    for key in list(random_image)[5:]:
        activity_index = ["eating", "sitting", "standing"].index(key)
        if random_image[key] == 1:
            one_hot[activity_index + 3] = 1

    print(f"Labels: {labels}")
    print(f"One-hot: {one_hot}")

    plt.imshow(image)
    plt.show()

    unittest.main(verbosity=2)