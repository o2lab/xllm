#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

g++ \
  -g3 \
  -O0 \
  kineto_playground.cpp \
  -o main \
  -I/sw/eb/sw/CUDA/12.3.0/include \
  -I/sw/eb/sw/CUDA/12.3.0/extras/CUPTI/include \
  -I../third_party/fmt/include \
  -I/scratch/user/siweicui/test/kineto/libkineto/include/ \
  -L/usr/local/lib \
  -L/sw/eb/sw/CUDA/12.3.0/lib64 \
  -L/sw/eb/sw/CUDA/12.3.0/extras/CUPTI/lib64 \
  -lpthread \
  -lcuda \
  -lcudart \
  /scratch/user/siweicui/test/kineto/libkineto/build/libkineto.a \
  /sw/eb/sw/CUDA/12.3.0/extras/CUPTI/lib64/libcupti.so  \
  kplay_cu.o
