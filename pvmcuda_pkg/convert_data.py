# GPU PVM Implementation
# (C) 2023 Filip Piekniewski 
# filip@piekniewski.info
import cv2
import numpy as np
import os
import zipfile
import sys
import argparse
import time


def drop_buf(myzip, filename, buf):
    zipi = zipfile.ZipInfo()
    zipi.filename = filename
    zipi.date_time = time.localtime()[:6]
    zipi.compress_type = zipfile.ZIP_DEFLATED
    zipi.external_attr = 0o777 << 16
    myzip.writestr(zipi, buf)


def downsample(filein, fileout, inter=cv2.INTER_NEAREST, fx=0.5, fy=0.5):
    myzipin = zipfile.ZipFile(os.path.expanduser(filein), "r")
    myzipout = zipfile.ZipFile(os.path.expanduser(fileout), "a", allowZip64=True)
    for (i, element) in enumerate(myzipin.namelist()):
        print(("[ %d/%d ] Processing " % (i, len(myzipin.namelist()))) + element + " " * 10 + "\r", end=' ')
        sys.stdout.flush()
        if element.endswith("jpg") or element.endswith("png"):
            handle = myzipin.open(element)
            buf = handle.read()
            img = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.CV_LOAD_IMAGE_COLOR)
            img1 = cv2.resize(img, dsize=(0, 0), fx=fx, fy=fy, interpolation=inter)
            ret, img2 = cv2.imencode(".png", img1)
            drop_buf(myzipout, element, np.getbuffer(img2))
    myzipin.close()
    myzipout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-i", "--input", help="Input file", type=str, default="")
    parser.add_argument("-o", "--output", help="Output file", type=str, default="")
    parser.add_argument("-D", "--downsample", help="By 0.5x", action="store_true")
    args = parser.parse_args()
    downsample(filein=args.input, fileout=args.output)

