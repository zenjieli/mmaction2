import os
import os.path as osp
import PIL.Image as Image

def run(srcdir, dstdir):
    for dirpath, _, filenames in os.walk(srcdir):
        for filename in filenames:
            if filename.lower().endswith('.png'):
                img = Image.open(osp.join(dirpath, filename))
                full_dstdir = dirpath.replace(srcdir, dstdir)
                os.makedirs(full_dstdir, exist_ok=True)
                dstpath = osp.join(full_dstdir, filename.replace('.png', '.jpg'))
                img.save(dstpath)


if __name__ == '__main__':
    srcdir = osp.expanduser('~/Downloads/fenix')
    dstdir = osp.expanduser('~/Downloads/fenix_jpg')
    run(srcdir, dstdir)