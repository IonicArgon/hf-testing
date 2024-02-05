# script to rename files in the training_data folder
import os

# test->humans are named h_te_xxx.jpg
# train->humans are named h_tr_xxx.jpg
# test->monkeys are named m_te_xxx.jpg
# train->monkeys are named m_tr_xxx.jpg

# get cwd
cwd = os.getcwd()

# list of files in respective folders
test_humans = os.listdir(cwd + '/training_data/test/humans')
train_humans = os.listdir(cwd + '/training_data/train/humans')
test_monkeys = os.listdir(cwd + '/training_data/test/monkeys')
train_monkeys = os.listdir(cwd + '/training_data/train/monkeys')


if __name__ == "__main__":
    # rename files
    print("Renaming test data for humans")
    for i in range(len(test_humans)):
        current_file = cwd + '/training_data/test/humans/' + test_humans[i]
        new_file = cwd + '/training_data/test/humans/h_te_' + f"{i+1:03}" + '.jpg'
        if not os.path.exists(new_file):
            print(f"Renaming {current_file} to {new_file}")
            os.rename(current_file, new_file)
        else:
            print(f"File {new_file} already exists")
    print("\n")

    print("Renaming training data for humans")
    for i in range(len(train_humans)):
        current_file = cwd + '/training_data/train/humans/' + train_humans[i]
        new_file = cwd + '/training_data/train/humans/h_tr_' + f"{i+1:03}" + '.jpg'
        if not os.path.exists(new_file):
            print(f"Renaming {current_file} to {new_file}")
            os.rename(current_file, new_file)
        else:
            print(f"File {new_file} already exists")
    print("\n")

    print("Renaming test data for monkeys")
    for i in range(len(test_monkeys)):
        current_file = cwd + '/training_data/test/monkeys/' + test_monkeys[i]
        new_file = cwd + '/training_data/test/monkeys/m_te_' + f"{i+1:03}" + '.jpg'
        if not os.path.exists(new_file):
            print(f"Renaming {current_file} to {new_file}")
            os.rename(current_file, new_file)
        else:
            print(f"File {new_file} already exists")
    print("\n")

    print("Renaming training data for monkeys")
    for i in range(len(train_monkeys)):
        current_file = cwd + '/training_data/train/monkeys/' + train_monkeys[i]
        new_file = cwd + '/training_data/train/monkeys/m_tr_' + f"{i+1:03}" + '.jpg'
        if not os.path.exists(new_file):
            print(f"Renaming {current_file} to {new_file}")
            os.rename(current_file, new_file)
        else:
            print(f"File {new_file} already exists")
    print("\n")

    print("Renaming complete")