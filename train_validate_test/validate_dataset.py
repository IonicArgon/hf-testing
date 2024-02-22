from datasets import load_dataset, DatasetDict
import unittest

class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset("imagefolder", data_dir="training_data")
        cls.split_train_testvalid = cls.dataset["train"].train_test_split(test_size=0.2)
        cls.split_test_valid = cls.split_train_testvalid["test"].train_test_split(test_size=0.5)
        cls.hf_dataset = DatasetDict({
            "train": cls.split_train_testvalid["train"],
            "test": cls.split_test_valid["test"],
            "valid": cls.split_test_valid["train"]
        })

    def test_split_verification(self):
        self.assertTrue("train" in self.hf_dataset, "train not in hf_dataset")
        self.assertTrue("test" in self.hf_dataset, "test not in hf_dataset")
        self.assertTrue("valid" in self.hf_dataset, "valid not in hf_dataset")

    def test_split_size_verification(self):
        size_train = len(self.hf_dataset["train"])
        size_test = len(self.hf_dataset["test"])
        size_valid = len(self.hf_dataset["valid"])

        percent_train = round(size_train / (size_train + size_test + size_valid), 1)
        percent_test = round(size_test / (size_train + size_test + size_valid), 1)
        percent_valid = round(size_valid / (size_train + size_test + size_valid), 1)

        self.assertEqual(percent_train, 0.8, f"percent_train is {percent_train}, expected 0.8")
        self.assertEqual(percent_test, 0.1, f"percent_test is {percent_test}, expected 0.1")
        self.assertEqual(percent_valid, 0.1, f"percent_valid is {percent_valid}, expected 0.1")

    def test_feature_verification(self):
        for split in self.hf_dataset:
            for i in range(len(self.hf_dataset[split])):
                self.assertTrue("bear" in self.hf_dataset[split][i], f"bear not in {split} at index {i}")
                self.assertTrue("activity" in self.hf_dataset[split][i], f"activity not in {split} at index {i}")

    def test_bear_verification(self):
        bears = set([item["bear"] for item in self.hf_dataset["train"]])
        self.assertEqual(bears, {"giant_panda", "polar_bear", "grizzly_bear"}, f"bears are {bears}, expected {'giant_panda', 'polar_bear', 'grizzly_bear'}")

    def test_activity_verification(self):
        activities = set([item["activity"] for item in self.hf_dataset["train"]])
        self.assertEqual(activities, {"eating", "sitting", "standing"}, f"activities are {activities}, expected {'eating', 'sitting', 'standing'}")

if __name__ == "__main__":
    unittest.main(verbosity=2)