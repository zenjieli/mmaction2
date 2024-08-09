def run(classes, in_filepath, out_filepath):
    """Only keep selected classes in a split file
    """
    lines = []
    with open(in_filepath, 'r') as f:
        lines = f.readlines()

    classes = [class_name.lower() for class_name in classes]
    selected_lines = [line for line in lines if any(line.lower().startswith(class_name) for class_name in classes)]

    with open(out_filepath, 'w') as f:
        f.writelines(selected_lines)


if __name__ == '__main__':
    classes_to_keep = ['ApplyEyeMakeup', 'ApplyLipstick']
    in_filepath = 'data/ucf101/ucf101_val_split_0_rawframes.txt'
    out_filepath = 'data/ucf101/ucf101_val_split_0_rawframes_two.txt'
    run(classes_to_keep, in_filepath, out_filepath)