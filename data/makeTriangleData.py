'''
    Copyright (c) 2021, Takashi Shirakawa. All rights reserved.
    e-mail: tkshirakawa@gmail.com

    Released under the MIT license.
    https://opensource.org/licenses/mit-license.php
'''


import sys, os
import cv2
import numpy as np


if sys.argv[1] == '-h':
    print('### Help for argv ###')
    print('  argv[1] : Path to a directory for output.')
    sys.exit(0)


dir_path = sys.argv[1]
format_ext = 'png'
pwidth = 512
pheight = 512
pmargin = 20
img_count = 600

np.random.seed(42)
pt = np.random.randint(pmargin, pwidth-pmargin, size=(img_count, 3, 2))
ch = np.random.randint(128, 256, size=(img_count, 3))            # Return random integers from low (inclusive) to high (exclusive).

print(dir_path)
os.makedirs(os.path.join(dir_path, 'image'), exist_ok=True)
os.makedirs(os.path.join(dir_path, 'mask'), exist_ok=True)


for i in range(img_count):
    img_base = np.full((pheight, pwidth, 3), fill_value=0, dtype=np.uint8)

    cv2.fillConvexPoly(img_base, pt[i], ch[i].tolist())
    img  = cv2.cvtColor(img_base, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)

    filename = '{:04}.{}'.format(i+1, format_ext)
    print(filename)

    cv2.imwrite(os.path.join(dir_path, 'image', filename), img)
    cv2.imwrite(os.path.join(dir_path, 'mask', filename), mask)

