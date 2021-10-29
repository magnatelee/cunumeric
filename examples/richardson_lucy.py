# Copyright 2021 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse

from benchmark import run_benchmark

float_type = "float32"

# A simplified implementation of Richardson-Lucy deconvolution


def run_richardson_lucy(
    shape, filter_shape, num_iter, warmup, timing, package
):
    if package == "cunumeric":
        from legate.timing import time

        import cunumeric as np

        convolve = np.convolve

        def timestamp():
            return time()

        def time_in_ms(start_ts, end_ts):
            return (end_ts - start_ts) / 1000.0

        def block():
            pass

    elif package == "cupy":
        import cupy as np
        import numpy
        from cupy import cuda

        def convolve(signal, psf, mode="same"):
            assert mode == "same"
            signal_shape = numpy.array(signal.shape)
            psf_shape = numpy.array(psf.shape)

            pad_shape = (
                signal_shape + psf_shape - (signal_shape % 2 != psf_shape % 2)
            )

            signal_slices = tuple(slice(0, size) for size in signal.shape)
            psf_slices = tuple(slice(0, size) for size in psf.shape)

            pad_signal = np.zeros(pad_shape, dtype=signal.dtype)
            pad_signal[signal_slices] = signal

            pad_psf = np.zeros(pad_shape, dtype=psf.dtype)
            pad_psf[psf_slices] = psf

            temp = np.fft.rfftn(pad_signal) * np.fft.rfftn(pad_psf)
            inverse = np.fft.irfftn(temp)

            offsets = psf_shape // 2 - (psf_shape % 2 == 0)
            out_slices = tuple(
                slice(offset, offset + size)
                for offset, size in zip(offsets, signal.shape)
            )
            return inverse[out_slices]

        def timestamp():
            ev = cuda.Event()
            ev.record()
            return ev

        def time_in_ms(start_ts, end_ts):
            return cuda.get_elapsed_time(start_ts, end_ts)

        def block():
            cuda.runtime.deviceSynchronize()

    image = np.random.rand(*shape).astype(float_type)
    psf = np.random.rand(*filter_shape).astype(float_type)
    im_deconv = np.full(image.shape, 0.5, dtype=float_type)
    psf_mirror = np.flip(psf)

    start = None
    for idx in range(num_iter + warmup):
        if idx == warmup:
            start = timestamp()
        conv = convolve(im_deconv, psf, mode="same")
        relative_blur = image / conv
        im_deconv *= convolve(relative_blur, psf_mirror, mode="same")

    stop = timestamp()

    if timing:
        block()
        print(f"Elapsed Time: {time_in_ms(start, stop)} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--iter",
        type=int,
        default=10,
        dest="I",
        help="number of iterations to run",
    )
    parser.add_argument(
        "-w",
        "--warmup",
        type=int,
        default=1,
        dest="warmup",
        help="warm-up iterations",
    )
    parser.add_argument(
        "-x",
        type=int,
        default=20,
        dest="X",
        help="number of elements in X dimension",
    )
    parser.add_argument(
        "-y",
        type=int,
        default=20,
        dest="Y",
        help="number of elements in Y dimension",
    )
    parser.add_argument(
        "-z",
        type=int,
        default=20,
        dest="Z",
        help="number of elements in Z dimension",
    )
    parser.add_argument(
        "-fx",
        type=int,
        default=4,
        dest="FX",
        help="number of filter weights in X dimension",
    )
    parser.add_argument(
        "-fy",
        type=int,
        default=4,
        dest="FY",
        help="number of filter weights in Y dimension",
    )
    parser.add_argument(
        "-fz",
        type=int,
        default=4,
        dest="FZ",
        help="number of filter weights in Z dimension",
    )
    parser.add_argument(
        "-t",
        "--time",
        dest="timing",
        action="store_true",
        help="perform timing",
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        type=int,
        default=1,
        dest="benchmark",
        help="number of times to benchmark this application (default 1 "
        "- normal execution)",
    )
    parser.add_argument(
        "--package",
        type=str,
        default="cunumeric",
        dest="package",
        help="NumPy package to use",
    )
    args = parser.parse_args()
    run_benchmark(
        run_richardson_lucy,
        args.benchmark,
        "Richardson Lucy",
        (
            (args.X, args.Y, args.Z),
            (args.FX, args.FY, args.FZ),
            args.I,
            args.warmup,
            args.timing,
            args.package,
        ),
    )
