# script to rename files in the training_data folder
import os

# test->humans are named h_te_xxx.jpg
# train->humans are named h_tr_xxx.jpg
# test->monkeys are named m_te_xxx.jpg
# train->monkeys are named m_tr_xxx.jpg

# get cwd
cwd = os.getcwd()

# list of files in respective folders
test_cat = os.listdir(cwd + '/training_data/test/cat')
train_cat = os.listdir(cwd + '/training_data/train/cat')
test_dog = os.listdir(cwd + '/training_data/test/dog')
train_dog = os.listdir(cwd + '/training_data/train/dog')

if __name__ == "__main__":
    print("Renaming test data for: cat")
    for i in range(len(test_cat)):
        current_file = test_cat[i]
        new_file = "cat_test_" + f"{i:03}" + ".jpg"

        current_file_dir = cwd + '/training_data/test/cat/' + current_file
        new_file_dir = cwd + '/training_data/test/cat/' + new_file

        if current_file != new_file:
            os.rename(current_file_dir, new_file_dir)
            print(f"Renamed {current_file} to {new_file}")
        else:
            print(f"Skipping file {current_file}: already renamed")
            continue

    print("Renaming test data for: dog")
    for i in range(len(test_dog)):
        current_file = test_dog[i]
        new_file = "dog_test_" + f"{i:03}" + ".jpg"

        current_file_dir = cwd + '/training_data/test/dog/' + current_file
        new_file_dir = cwd + '/training_data/test/dog/' + new_file

        if current_file != new_file:
            os.rename(current_file_dir, new_file_dir)
            print(f"Renamed {current_file} to {new_file}")
        else:
            print(f"Skipping file {current_file}: already renamed")
            continue

    print("Renaming train data for: cat")
    for i in range(len(train_cat)):
        current_file = train_cat[i]
        new_file = "cat_train_" + f"{i:03}" + ".jpg"

        current_file_dir = cwd + '/training_data/train/cat/' + current_file
        new_file_dir = cwd + '/training_data/train/cat/' + new_file

        if current_file != new_file:
            os.rename(current_file_dir, new_file_dir)
            print(f"Renamed {current_file} to {new_file}")
        else:
            print(f"Skipping file {current_file}: already renamed")
            continue

    print("Renaming train data for: dog")
    for i in range(len(train_dog)):
        current_file = train_dog[i]
        new_file = "dog_train_" + f"{i:03}" + ".jpg"

        current_file_dir = cwd + '/training_data/train/dog/' + current_file
        new_file_dir = cwd + '/training_data/train/dog/' + new_file

        if current_file != new_file:
            os.rename(current_file_dir, new_file_dir)
            print(f"Renamed {current_file} to {new_file}")
        else:
            print(f"Skipping file {current_file}: already renamed")
            continue

    print("Renaming complete!")

