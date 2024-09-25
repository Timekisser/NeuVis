import os
import random

def create_train_val_test_lists(directory, train_val_ratio=0.8):
    """
    Create train and test lists from .ply files in a directory.

    Args:
    directory (str): The directory containing the .ply files.
    train_val_ratio (float): The ratio of train/validation files to test files.

    Returns:
    dict: A dictionary containing two lists, 'train_val' and 'test', with file paths.
    """
    # List all .ply files in the directory
    ply_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.ply'):
                ply_files.append(os.path.join(root, file))

    # Shuffle the list
    random.shuffle(ply_files)

    # Split the list into train/val and test
    split_index = int(train_val_ratio * len(ply_files))
    train_val_files = ply_files[:split_index]
    test_files = ply_files[split_index:]

    return {'train_val': train_val_files, 'test': test_files}


def write_lists_to_files(file_lists, base_directory):
    """
    Write the train and test lists to files in a new directory.

    Args:
    file_lists (dict): A dictionary containing 'train_val' and 'test' lists.
    base_directory (str): The base directory where the new 'filelist' directory will be created.
    """
    # Create a new directory for the file lists
    filelist_dir = os.path.join(base_directory, 'filelist')
    os.makedirs(filelist_dir, exist_ok=True)

    # Write the train/val and test lists to files
    for list_name, file_list in file_lists.items():
        with open(os.path.join(filelist_dir, f'{list_name}.txt'), 'w') as f:
            for file_path in file_list:
                # Write the relative path to the file
                relative_path = os.path.relpath(file_path, base_directory)
                f.write(relative_path + '\n')


# 假设.ply文件位于以下目录
ply_directory_path = './data/ShapeNet_NeuVis/points/03691459'

# 创建训练/验证和测试文件列表
file_lists = create_train_val_test_lists(ply_directory_path)

# 假设您希望filelist子目录位于以下基础目录
base_directory_path = './data/ShapeNet_NeuVis/'

# 将文件列表写入到filelist子目录中的.txt文件
write_lists_to_files(file_lists, base_directory_path)




