import os

cwd = os.getcwd()

if __name__ == "__main__":
    master = {
        "giant_panda": {
            "eating": os.listdir(cwd + "/training_data/giant_panda/eating"),
            "sitting": os.listdir(cwd + "/training_data/giant_panda/sitting"),
            "standing": os.listdir(cwd + "/training_data/giant_panda/standing"),
        },
        "grizzly_bear": {
            "eating": os.listdir(cwd + "/training_data/grizzly_bear/eating"),
            "sitting": os.listdir(cwd + "/training_data/grizzly_bear/sitting"),
            "standing": os.listdir(cwd + "/training_data/grizzly_bear/standing"),
        },
        "polar_bear": {
            "eating": os.listdir(cwd + "/training_data/polar_bear/eating"),
            "sitting": os.listdir(cwd + "/training_data/polar_bear/sitting"),
            "standing": os.listdir(cwd + "/training_data/polar_bear/standing"),
        },
    }

    for animal in master:
        for action in master[animal]:
            for i, img in enumerate(master[animal][action]):
                curr_name = cwd + f"/training_data/{animal}/{action}/{img}"
                new_name = cwd + f"/training_data/{animal}/{action}/{animal}_{action}_{i:04}.jpg"

                if os.path.exists(new_name):
                    index = i
                    while os.path.exists(new_name):
                        index += 1
                        new_name = cwd + f"/training_data/{animal}/{action}/{animal}_{action}_{index:04}.jpg"

                os.rename(curr_name, new_name)
                print(f"Renamed {curr_name} to {new_name}")

    print("Renaming complete")
