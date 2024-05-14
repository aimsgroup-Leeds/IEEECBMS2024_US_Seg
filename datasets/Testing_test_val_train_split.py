def count_files_with_keyword(file_path, keyword):
    count = 0
    with open(file_path, 'r') as file:
        for line in file:
            if keyword in line:
                count += 1
    return count

train_path = 'train.txt'
val_path = 'val.txt'
test_path = 'test.txt'

keywords = ['benign', 'malignant', 'normal']

for keyword in keywords:
    train_count = count_files_with_keyword(train_path, keyword)
    val_count = count_files_with_keyword(val_path, keyword)
    test_count = count_files_with_keyword(test_path, keyword)

    print(f"Number of files with '{keyword}' in train set: {train_count}")
    print(f"Number of files with '{keyword}' in validation set: {val_count}")
    print(f"Number of files with '{keyword}' in test set: {test_count}")
