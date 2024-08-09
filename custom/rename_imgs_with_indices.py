import os
import os.path as osp
import shutil

def run(srcdir, dstdir):
    for dirpath, _, filenames in os.walk(srcdir):
        filenames.sort()
        for idx, filename in enumerate(filenames):
            full_dstdir = dirpath.replace(srcdir, dstdir)
            os.makedirs(full_dstdir, exist_ok=True)
            shutil.copyfile(osp.join(dirpath, filename), osp.join(full_dstdir, f'img_{idx+1:05d}.jpg')) #


if __name__ == '__main__':
    srcdir = osp.expanduser('~/data/action/fenix/rawframes/non-fall1')
    dstdir = osp.expanduser('~/data/action/fenix/rawframes/non-fall')
    run(srcdir, dstdir)