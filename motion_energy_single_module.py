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

def gabor_kernel_separate_torch(frequency, theta=0, n_stds=3):
    def _sigma_prefactor(bandwidth):
        b = bandwidth
        # See http://www.cs.rug.nl/~imaging/simplecell.html
        return 1.0 / pi * torch.sqrt(torch.log(torch.tensor(2)) / 2.0) * \
            (2.0 ** b + 1) / (2.0 ** b - 1)
        
    sigma = _sigma_prefactor(1) / frequency

    width = torch.ceil(max(torch.abs(n_stds * sigma * torch.cos(theta)), torch.abs(n_stds * sigma * torch.sin(theta)), 1))

    y = torch.arange(-width, width + 1)
    x = torch.arange(-width, width + 1)
    
    rotxx = x * torch.cos(theta)
    rotyx = -x * torch.sin(theta)
    rotxy = y * torch.sin(theta)
    rotyy = y * torch.cos(theta)

    kernel_x = torch.zeros(len(x), dtype=torch.complex128)
    kernel_y = torch.zeros(len(y), dtype=torch.complex128)

    kernel_x[:] = torch.exp(-0.5 * (rotxx ** 2 / sigma ** 2 + rotyx ** 2 / sigma ** 2))
    kernel_y[:] = torch.exp(-0.5 * (rotxy ** 2 / sigma ** 2 + rotyy ** 2 / sigma ** 2))

    kernel_x /= torch.sqrt(torch.tensor(2 * pi)) * sigma
    kernel_y /= torch.sqrt(torch.tensor(2 * pi)) * sigma

    kernel_x *= torch.exp(1j * (2 * pi * frequency * rotxx))
    kernel_y *= torch.exp(1j * (2 * pi * frequency * rotxy))

    return kernel_x, kernel_y

class SplitConv2d(torch.nn.Module):
    def __init__(self, kernel_x, kernel_y):
        super().__init__()

        # NOTE unfortunately, native complex ops for conv are not currently
        # supported in PyTorch (<=1.11). So multiple "real" convolutions
        # must be performed on different parts of the complex number and
        # finally combined. However, some computational efficiences can still be achieved
        #   1. As image input only has real component, 
        #      the first pass (of the 2D conv split to 1D scheme)
        #      can be batched. So kernel_y only needs one conv operation
        #   2. The first pass results in a complex response, and the second pass
        #      can naively work with this to do 4 (real) conv operations
        #   3. Or to save a little bit more time, a variation of the 
        #      Karatsuba algorithm can be used, which needs 3 (real) conv ops
        self.conv_y = torch.nn.Conv2d(1, 2, kernel_y.real.shape[0], bias=False)
        kernel_y_real_and_imag = torch.stack((kernel_y.real, kernel_y.imag))
        self.conv_y.weight = torch.nn.Parameter(
            torch.tensor(kernel_y_real_and_imag.reshape(2, 1, -1, 1), dtype=torch.float32),
                requires_grad=False)
        self.conv_x_real = torch.nn.Conv2d(1, 1, kernel_x.real.shape[0], bias=False)
        self.conv_x_real.weight = torch.nn.Parameter(
            torch.tensor(kernel_x.real.reshape(1, 1, 1, -1), dtype=torch.float32),
                requires_grad=False)
        self.conv_x_imag = torch.nn.Conv2d(1, 1, kernel_x.imag.shape[0], bias=False)
        self.conv_x_imag.weight = torch.nn.Parameter(
            torch.tensor(kernel_x.imag.reshape(1, 1, 1, -1), dtype=torch.float32),
                requires_grad=False)

        # To be used for the Karatsuba
        self.conv_x_comb = torch.nn.Conv2d(1, 1, kernel_x.imag.shape[0], bias=False)
        self.conv_x_comb.weight = torch.nn.Parameter(
            torch.tensor(
                (kernel_x.real + kernel_x.imag).reshape(1, 1, 1, -1), dtype=torch.float32),
                requires_grad=False)

    def forward(self, incoming, expected_width=None):
        # If native complex was supported by PyTorch
        # intermediate = self.conv_y(incoming)
        # ret = self.conv_x(intermediate)
        # return ret

        intermediate = self.conv_y(incoming)
        intermediate_real = intermediate[:, 0].unsqueeze(1)
        intermediate_imag = intermediate[:, 1].unsqueeze(1)

        # Naive 4 conv ops on the second pass
        # resp_real = self.conv_x_real(intermediate_real) - self.conv_x_imag(intermediate_imag)
        # resp_imag = self.conv_x_real(intermediate_imag) + self.conv_x_imag(intermediate_real)

        # Variation of Karatsuba (3 conv ops)
        # https://en.wikipedia.org/wiki/Multiplication_algorithm#Complex_number_multiplication
        # https://github.com/pytorch/pytorch/issues/71108#issuecomment-1016889045
        k_a = self.conv_x_real(intermediate_real)
        k_b = self.conv_x_imag(intermediate_imag)
        k_c = self.conv_x_comb(intermediate_real + intermediate_imag)
        resp_real = k_a - k_b
        resp_imag = k_c - k_a - k_b

        # NOTE to save some more time, padding could be 
        # done after all kernels in one go.
        # But kernels can be of varying size.
        # Hence, padding to a common resolution now
        if expected_width is not None:
            # Expected pad difference to match image resolution
            # NOTE Careful using same pad_diff for all. 
            # That's assuming w and h of kernel is same
            pad_diff = expected_width - resp_real.shape[-1]
            padder = torch.nn.ZeroPad2d(int(pad_diff / 2))

            resp_real = padder(resp_real)
            resp_imag = padder(resp_imag)

        return resp_real, resp_imag

class MotionEnergyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Make kernels
        # NOTE below kernels work fine for 240 x 240 input
        self.spatial_kernel_params = [
            (0.3, 0),
            (0.3, 90),
            (0.3, 45),
            (0.3, 135)
        ]
        self.spatial_convs = []
        for param in self.spatial_kernel_params:
            freq, orin = param
            kernel_x, kernel_y = gabor_kernel_separate_torch(freq, theta=torch.deg2rad(torch.tensor(orin)))

            self.spatial_convs.append(
                SplitConv2d(kernel_x, kernel_y))


        dt = 1 / 30
        time_length = 90 / 1000 # ms to s
        self.num_frames = int(torch.ceil(torch.tensor(time_length / dt)))
        time_array = torch.linspace(0, time_length, self.num_frames+1)[1:]

        # Time functions due to George Mather
        # http://www.georgemather.com/Model.html
        slow_n = 6
        fast_n = 3
        beta = 0.99

        # k is scaling for time
        k = 125

        slow_t = (k * time_array)**slow_n * torch.exp(-k * time_array) * (
            1 / factorial(slow_n) - beta / factorial(slow_n + 2) * (k * time_array)**2)
        fast_t = (k * time_array)**fast_n * torch.exp(-k * time_array) * (
            1 / factorial(fast_n) - beta / factorial(fast_n + 2) * (k * time_array)**2)

        self.slow_t_conv = torch.nn.Conv2d(self.num_frames, 1, (1, 1), bias=False)
        self.slow_t_conv.weight = torch.nn.Parameter(
            torch.tensor(slow_t.reshape(1, -1, 1, 1),
            dtype=torch.float32))
        self.fast_t_conv = torch.nn.Conv2d(self.num_frames, 1, (1, 1), bias=False)
        self.fast_t_conv.weight = torch.nn.Parameter(
            torch.tensor(fast_t.reshape(1, -1, 1, 1),
            dtype=torch.float32))

        self.CONST_PI = torch.tensor(pi)

    def forward(self, input_images, expected_img_width=240):                           
        # Spatial convolutions
        stacked_gabor_resps = []
        for spatial_conv in self.spatial_convs:
            gabor_resp_even, gabor_resp_odd = spatial_conv(input_images, expected_img_width)
            # Will stack even, odd, even, odd....
            stacked_gabor_resps.extend(
                (gabor_resp_even.squeeze(), gabor_resp_odd.squeeze()))

            # Plain edge response (spatial convolutions)
            # edge_resp += (
            #     gabor_resp_even[0] +
            #     gabor_resp_odd[0]
            # ).squeeze()

        stacked_gabor_resps = torch.stack(stacked_gabor_resps)

        # Temporal convolutions 1x1 
        slow_time_resp = self.slow_t_conv(stacked_gabor_resps)
        fast_time_resp = self.fast_t_conv(stacked_gabor_resps)

        total_energy_x = torch.zeros((expected_img_width, expected_img_width))
        total_energy_y = torch.zeros_like(total_energy_x)

        for ix, param in enumerate(self.spatial_kernel_params):
            even_slow, even_fast = slow_time_resp[2*ix], fast_time_resp[2*ix]
            odd_slow, odd_fast = slow_time_resp[2*ix+1], fast_time_resp[2*ix+1]

            resp_negdir_1 =  odd_fast +  even_slow
            resp_negdir_2 = -odd_slow +  even_fast
            resp_posdir_1 = -odd_fast +  even_slow
            resp_posdir_2 =  odd_slow +  even_fast

            energy_negdir = (
                resp_negdir_1**2 + resp_negdir_2**2).squeeze()
            energy_posdir = (
                resp_posdir_1**2 + resp_posdir_2**2).squeeze()
            energy_thisdir = energy_negdir - energy_posdir

            orientation = torch.deg2rad(torch.tensor(param[1]))
            energy_in_image_space_x = energy_thisdir * torch.cos(torch.tensor(orientation))
            energy_in_image_space_y = energy_thisdir * torch.sin(torch.tensor(orientation))

            total_energy_x += energy_in_image_space_x
            total_energy_y += energy_in_image_space_y

            # Total energy in polar coordinates
            total_energy_mag = torch.sqrt(total_energy_x**2 + total_energy_y**2)
            total_energy_ang = my_atan2(total_energy_y, total_energy_x)
            
            # motion_energy = (total_energy_mag, total_energy_ang)

        motion_hue = ((total_energy_ang + self.CONST_PI) / (2 * self.CONST_PI) * 179)
        motion_val = torch.clip(total_energy_mag * 200, 0, 255)
        motion_sat = torch.ones_like(motion_hue) * 255

        motion_energy_xy = torch.stack([total_energy_x, total_energy_y], dim=-1)
        motion_energy_hsv = torch.stack([motion_hue, motion_sat, motion_val], dim=-1)

        return motion_energy_xy, motion_energy_hsv

# Because ONNX STILL doesn't support atan2
# Alternative from https://gist.github.com/nikola-j/b5bb6b141b8d9920318677e1bba70466
def my_atan2(y, x):
    ans = torch.atan(y / (x + 1e-6))
    ans += ((y > 0) & (x < 0)) * pi
    ans -= ((y < 0) & (x < 0)) * pi
    ans *= (1 - ((y > 0) & (x == 0)) * 1.0)
    ans += ((y > 0) & (x == 0)) * (pi / 2)
    ans *= (1 - ((y < 0) & (x == 0)) * 1.0)
    ans += ((y < 0) & (x == 0)) * (-pi / 2)
    return ans

if __name__ == "__main__":
    # Setup a command-line argument parser
    parser = argparse.ArgumentParser(
        description = 'Visualize spatiotemporal energy model (single module) on webcam feed',
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

    # The main object of this demo
    motion_energy_module = MotionEnergyModule()

    print('Starting motion energy visualization. To exit, press ESC on any visualization window.')
    
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

    img_grey_stack = deque(maxlen=motion_energy_module.num_frames)

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
        if len(img_grey_stack) == motion_energy_module.num_frames:
            with torch.no_grad():
                input_images  = torch.cat(tuple(img_grey_stack), dim=0)
                _, motion_energy_hsv = motion_energy_module(input_images)

                motion_energy_rgb = cv2.cvtColor(
                    motion_energy_hsv.numpy().astype(np.uint8),
                    cv2.COLOR_HSV2BGR)

                cv2.imshow('motion_energy', motion_energy_rgb)
        
        if cv2.waitKey(1) == 27: 
            break  # esc to quit

    torch.onnx.export(
        motion_energy_module, input_images,
        'motion_energy_module.onnx',
        input_names=['input_images'],
        verbose=True,
        opset_version=13)
    
    cv2.destroyAllWindows()
