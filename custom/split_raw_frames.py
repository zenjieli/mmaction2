import os.path as osp
import os
import random


def find_sample_dirs(src_root, val_ratio=0.2) -> tuple[list[str], list[str]]:
    """Returns a list of training samples and a list of validation samples
    """
    # Get class dir names
    cls_dirs = os.listdir(src_root)
    cls_dirs.sort()  # Sort by class indices

    dirs = []
    for idx, cls_dir in enumerate(cls_dirs):
        sample_dirs = os.listdir(osp.join(src_root, cls_dir))
        for dir in sample_dirs:
            # Count the frame count
            frame_count = len([f for f in os.listdir(osp.join(src_root, cls_dir, dir)) if f.lower()[-4:] in ['.jpg', '.png']])

            dirs.append((osp.join(cls_dir, dir), str(frame_count), str(idx)))

    random.seed(42)
    random.shuffle(dirs)

    train_samples = dirs[:int(len(dirs)*(1-val_ratio))]
    val_samples = dirs[len(train_samples):]

    return train_samples, val_samples


def write_samples_to_file(filepath, lines):
    with open(filepath, 'w') as f:
        f.writelines(' '.join(line) + '\n' for line in lines)


def run(src_root, dst_dir, dataset_name, val_ratio=0.2):
    train_sample_dirs, val_sample_dirs = find_sample_dirs(src_root, val_ratio)

    write_samples_to_file(osp.join(dst_dir, f'{dataset_name}_train_rawframes.txt'), train_sample_dirs)
    write_samples_to_file(osp.join(dst_dir, f'{dataset_name}_val_rawframes.txt'), val_sample_dirs)


if __name__ == '__main__':
    src_root = osp.expanduser('~/data/action/fenix/rawframes')
    dst_dir = osp.expanduser('~/data/action/fenix')
    run(src_root, dst_dir, 'fenix')
