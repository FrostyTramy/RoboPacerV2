#!/bin/bash
# Usage: bash compile.sh <model_name>
# Runs inside the hailo-dfc Docker image. train.py already saved
# calib_data_nhwc.npy in the layout the DFC wants, so unlike some setups
# there's no NCHW->NHWC transpose step here.
set -e

NET_NAME=${1:-model}
MODELS_DIR=/workspace/models
ONNX_MODEL=$MODELS_DIR/${NET_NAME}.onnx
HAR_FILE=$MODELS_DIR/${NET_NAME}.har
CALIB_NHWC=$MODELS_DIR/${NET_NAME}_calib_data_nhwc.npy
OPT_HAR=$MODELS_DIR/${NET_NAME}_optimized.har
HW_ARCH=hailo8

echo "============================================================"
echo "Hailo DFC  model=${NET_NAME}  target=${HW_ARCH}"
echo "============================================================"

echo "[ 1/3 ] Parsing ONNX -> HAR ..."
hailo parser onnx "$ONNX_MODEL" \
    --hw-arch "$HW_ARCH" --net-name "$NET_NAME" --har-path "$HAR_FILE" -y

echo "[ 2/3 ] Optimizing (PTQ int8) ..."
hailo optimize "$HAR_FILE" \
    --hw-arch "$HW_ARCH" --calib-set-path "$CALIB_NHWC" --output-har-path "$OPT_HAR"

echo "[ 3/3 ] Compiling -> HEF ..."
cd "$MODELS_DIR"
hailo compiler "$OPT_HAR" --hw-arch "$HW_ARCH" --output-dir "$MODELS_DIR"

echo ""
echo "SUCCESS -> $MODELS_DIR/${NET_NAME}.hef"
