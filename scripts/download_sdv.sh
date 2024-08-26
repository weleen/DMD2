#!/bin/bash
OUTPUT_DIR=$1

pushd $OUTPUT_DIR
git lfs install
git clone git@hf.co:stabilityai/stable-video-diffusion-img2vid
popd