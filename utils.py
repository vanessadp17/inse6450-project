"""
Create mismatch ratios for the datasets
"""
def get_splits(dataset, seed, mismatch):
    if dataset == 'cifar10':
        if seed == 1:
            shuffled_list = [5, 3, 2, 6]
    elif dataset == 'cifar100':
        if seed == 1:
            shuffled_list = [0, 83, 10, 51, 61, 57, 53, 26, 45, 91, 13, 8, 90, 81, 5, 84, 20, 94, 40, 87, 6, 7, 14, 18, 24, 99, 79, 80, 75, 66, 1, 36, 65, 93, 78, 70, 92, 82, 62, 54]
    elif dataset == 'tinyimagenet':
        if seed == 1:
            shuffled_list = [67, 110, 31, 112, 14, 43, 96, 3, 46, 65,
                            159, 191, 136, 143, 187, 45, 121, 176, 4, 158,
                            16, 145, 171, 162, 127, 116, 48, 6, 113, 37,
                            62, 13, 70, 195, 153, 38, 80, 33, 161, 128,
                            193, 11, 129, 85, 106, 169, 105, 81, 148, 53,
                            102, 74, 44, 1, 192, 17, 79, 101, 194, 61,
                            156, 103, 39, 77, 114, 20, 15, 177, 141, 165,
                            66, 160, 107, 47, 135, 124, 55, 139, 125, 133]
    knownclass = shuffled_list[:mismatch]
    return knownclass