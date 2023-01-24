#!/usr/bin/env python3

# Motion Perception Tool
# An efficient, real-time implementation of Adelson's spatiotemporal energy model,
# as a single Torch NN module. This is less flexible than motion_energy_split_kernel.py.
#
# Copyright (C) 2022 Aravind Battaje
# Email: aravind@oxidification.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser Public License for more details.
#
# You should have received a copy of the GNU Lesser Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import cv2
import numpy as np
from math import factorial, pi
from collections import deque
import torch

import sys
import argparse
import logging

import onnx
import onnxruntime as ort

if __name__ == "__main__":
    # Setup a command-line argument parser
    parser = argparse.ArgumentParser(
        description = 'Visualize spatiotemporal energy model (ONNX module) on webcam feed',
        # exit_on_error=True, needs Python 3.9
    )
    parser.add_argument(
        '-c', '--cam-id',
        default=0,
        type=int,
        help="""camera ID as input; typically 0 is internal webcam, 1 is external camera
        (default: 0); NOTE ignored if --file is specified"""
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='print extra information on the command line'
    )
    args = parser.parse_args()

    # Setup a logger
    logging.basicConfig(
        format = '%(levelname)s: %(message)s',
        level = logging.INFO if args.verbose else logging.WARN)

    motion_energy_model = onnx.load('motion_energy_module.onnx')
    onnx.checker.check_model(motion_energy_model)
    print(onnx.helper.printable_graph(motion_energy_model.graph))

    ort_session = ort.InferenceSession('motion_energy_module.onnx')

    # # The main object of this demo
    # motion_energy_module = MotionEnergyModule()

    # print('Starting motion energy visualization. To exit, press ESC on any visualization window.')
    
    # Try reading with DSHOW (works in most Windows)
    cap = cv2.VideoCapture(args.cam_id, cv2.CAP_DSHOW)
    ret_val, _ = cap.read()

    # Else, without DSHOW (works in most Linux/Mac)
    if ret_val is False:
        cap = cv2.VideoCapture(args.cam_id)
        ret_val, _ = cap.read()

    if ret_val is False:
        logging.error(
            f'{args.cam_id} is an invalid camera. '
            'Make sure camera is attached and usable in other programs.')
        sys.exit(1)

    img_grey_stack = deque(maxlen=3)

    # The main loop
    while True:
        ret_val, img_in = cap.read()

        if ret_val is False:
            # End of video reached
            break

        height_img_in, width_img_in, _ = img_in.shape

        if width_img_in > height_img_in:
            height_range_start = 0
            height_range_last = height_img_in
            width_range_start = (width_img_in - height_img_in) // 2
            width_range_last = width_range_start+height_img_in
        elif height_img_in >= width_img_in:
            width_range_start = 0
            width_range_last = width_img_in
            height_range_start = (height_img_in - width_img_in) // 2
            height_range_last = height_range_start+width_img_in
        img_in = img_in[
            height_range_start:height_range_last,
            width_range_start:width_range_last, :]
        img_in = cv2.resize(img_in, (240, 240))

        cv2.imshow('input_image', img_in)

        img_grey = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
        height_img, width_img = img_grey.shape
        
        img_grey_torch = torch.tensor(
            img_grey.reshape(1, 1, height_img, width_img),
            dtype=torch.float32, requires_grad=False)
        img_grey_stack.append(img_grey_torch)

        # Deque is complete
        if len(img_grey_stack) == 3:
            input_images = torch.cat(tuple(img_grey_stack), dim=0).numpy()

            # _, motion_energy_hsv = motion_energy_module(input_images)
            ort_outs = ort_session.run(None, {'input_images': input_images})

            motion_energy_rgb = cv2.cvtColor(
                ort_outs[1].astype(np.uint8),
                cv2.COLOR_HSV2BGR)

            cv2.imshow('motion_energy', motion_energy_rgb)
        
        if cv2.waitKey(1) == 27: 
            break  # esc to quit

    cv2.destroyAllWindows()
