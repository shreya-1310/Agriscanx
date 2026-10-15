# AgriScanX

Experimental AI research project for RGB-to-multispectral image reconstruction and agricultural vegetation analysis.

## Project Overview

AgriScanX explores deep-learning approaches for estimating multispectral information from RGB imagery using the WeedyRice-RGBMS-DB dataset.

## Models

- RGB2MS_UNet: CBAM-based U-Net architecture in `rgb2ms_cbam_model.py`.
- EfficientUNetGenerator: compact RGB-to-single-channel model in `efficient_unet.py`.
- MobileNetV2UNet: separate experimental architecture in `model.py`.

## Dataset

WeedyRice-RGBMS-DB contains RGB imagery with corresponding Green, Red, Red Edge, and NIR TIFF files.

The corrected loader, `dataset_corrected.py`, explicitly assembles four separate spectral bands in this order:

Green, Red, Red Edge, NIR.

The original CBAM training notebook used a different pairing method that could select only one spectral TIFF per RGB image.

## Checkpoints

- `models/finetuned_epoch40.pth`: compact single-channel checkpoint.
- `best_model.pth`: CBAM model checkpoint, distributed separately because it exceeds GitHub's regular file-size limit.

The CBAM checkpoint is an experimental legacy artifact. Its four outputs should not be treated as validated physical multispectral measurements.

## Results and Limitations

Historical CBAM training logs report a best visible validation R2 of approximately 0.2028 at epoch 12.

These results are not verified four-band reconstruction performance because of the original target-loading issue.

Accurate spectral reconstruction, calibrated NDVI, and crop-health classification require additional evaluation and potentially retraining.

## Getting Started

See `requirements_train.txt` and `requirements_rpi.txt` for dependencies.

The original notebooks document training experiments. Some scripts target different model architectures and are not interchangeable with the supplied checkpoints.

A verified end-to-end inference command for the corrected four-band pipeline has not yet been established.

## Current Status

Research prototype. Model checkpoint loading and tensor inference have been tested, and the corrected dataset loader has passed a sample test.

Raspberry Pi deployment and scientifically validated crop-health predictions remain future work.




## Download the Experimental CBAM Model

The trained CBAM checkpoint (`best_model.pth`) is distributed
separately because it exceeds GitHub's regular Git file-size limit.

1. Open the [v0.1.0 release](https://github.com/shreya-1310/Agriscanx/releases/tag/v0.1.0).
2. Download `best_model.pth` from Assets.
3. Place it inside the `models/` folder.

Expected location:

    models/best_model.pth

### Test the Model

Install the necessary Python dependencies, including PyTorch,
NumPy and Pillow, in a compatible Python environment.

Run:

    python test_model.py --checkpoint models/best_model.pth

### Run RGB Image Inference

    python predict.py --image path/to/image.jpg --checkpoint models/best_model.pth

Predictions are saved in `outputs/prediction/`.

The four output channels are experimental and must not be
interpreted as validated spectral measurements or used to
calculate scientifically reliable NDVI.

