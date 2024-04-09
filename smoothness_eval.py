import argparse
import os
from tqdm import tqdm

import numpy as np
import cv2 as cv
import decord
from torchvision.transforms import Resize

decord.bridge.set_bridge('torch')


parser = argparse.ArgumentParser()

parser.add_argument("--input", "-i", help="folder containing input videos")
parser.add_argument("--output", "-o", default="stats.txt", help="location of the output file")
parser.add_argument("--file_list", "-f", default="file_names.txt", help="location of the file list")
args = parser.parse_args()

file_names = []
with open(args.file_list, 'r') as f:
    for name in f:
        file_names.append(name[2:].rstrip('\n'))

with open(args.output, 'w') as out:
    out.write('flow_mean,flow_std,mse,mse_std\n')
    for file_name in tqdm(file_names):
        try:
            file = os.path.join(args.input, file_name)
            cap = cv.VideoCapture(file)

            ret, prev_frame = cap.read()
            prev_frame = cv.resize(prev_frame, (128, 128))
            prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

            ret, frame = cap.read()

            magnitudes = []
            mse = []
            while(ret):
                frame = cv.resize(frame, (128, 128))
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                flow = cv.calcOpticalFlowFarneback(
                                    prev_gray, 
                                    gray, 
                                    None,
                                    0.5, # pyr_scale
                                    7, # levels
                                    5, # winsize
                                    15, # iterations
                                    5, # poly_n
                                    1.2, # poly_sigma
                                    0 # flags
                                )
                
                magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
                magnitude = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
                magnitudes.append(cv.applyColorMap(magnitude.astype(np.uint8), cv.COLORMAP_JET))
                mse.append((frame - prev_frame) ** 2 )

                prev_gray = gray
                prev_frame = frame
                ret, frame = cap.read()
        except KeyboardInterrupt:
            break
        except:
            continue
        
        cap.release()
        magnitudes = np.array(magnitudes)
        mse = np.array(mse)
        out.write(f'{magnitudes.mean()},{magnitudes.std()},{mse.mean()},{mse.std()}\n')