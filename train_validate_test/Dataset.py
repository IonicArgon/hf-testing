from datasets import load_dataset, DatasetDict

class Dataset():
    def __init__(self, data_dir: str = "training_data"):
        self.data_dir = data_dir
        self.dataset = load_dataset("imagefolder", data_dir=self.data_dir)
        self.split_train_testvalid = self.dataset["train"].train_test_split(test_size=0.2)
        self.split_test_valid = self.split_train_testvalid["test"].train_test_split(test_size=0.5)
        self.hf_dataset = DatasetDict({
            "train": self.split_train_testvalid["train"],
            "test": self.split_test_valid["test"],
            "valid": self.split_test_valid["train"]
        })

    def get_dataset(self):
        print("Train split length: ", len(self.hf_dataset["train"]))
        print("Test split length: ", len(self.hf_dataset["test"]))
        print("Valid split length: ", len(self.hf_dataset["valid"]))
        return self.hf_dataset
