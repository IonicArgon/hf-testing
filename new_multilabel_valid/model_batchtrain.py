from Dataset import Dataset
from ModelTrainer import ModelTrainer

if __name__ == "__main__":
    GLOBAL_DATASET = Dataset()
    GLOBAL_MODEL_PARAMS = {
        "problem_type": "multi_label_classification",
        "num_labels": 6
    }
    GLOBAL_TRAINING_ARGS = {
        "num_train_epochs": 5,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "logging_steps": 10,
        "evaluation_strategy": "epoch",
        "use_cpu": False
    }

    list_of_models = [
        "google/vit-large-patch16-384"
    ]

    for model in list_of_models:
        print(f"Training model: {model}")

        model_trainer = ModelTrainer(
            hf_dataset=GLOBAL_DATASET.get_dataset(),
            model_checkpoint=model,
            model_params=GLOBAL_MODEL_PARAMS
        )

        model_trainer.train(GLOBAL_TRAINING_ARGS)
        input("Press Enter to continue...")
