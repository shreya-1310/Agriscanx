"""
export_onnx.py
--------------
Export trained model -> ONNX FP32 -> INT8 quantized for Raspberry Pi 4.

Usage (Colab):
    !python export_onnx.py \
        --ckpt "/content/drive/MyDrive/rgb2ms_checkpoints/best_model.pth" \
        --out  "/content/drive/MyDrive/rgb2ms_models" \
        --img_size 512

Then copy rgb2ms_int8.onnx to your Raspberry Pi.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from model import MobileNetV2UNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",     required=True)
    p.add_argument("--out",      default="models")
    p.add_argument("--img_size", type=int, default=512,
                   help="Use 256 or 384 for faster Pi inference")
    p.add_argument("--opset",    type=int, default=13)
    return p.parse_args()


def export_fp32(model, img_size, path, opset):
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model, dummy, str(path),
        opset_version=opset,
        input_names=["rgb"],
        output_names=["ms"],
        dynamic_axes={
            "rgb": {0: "batch", 2: "H", 3: "W"},
            "ms":  {0: "batch", 2: "H", 3: "W"},
        },
        do_constant_folding=True,
        verbose=False,
    )
    print(f"FP32 ONNX saved -> {path}  ({path.stat().st_size/1e6:.1f} MB)")


def quantize_int8(fp32_path, int8_path):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
        per_channel=False,   # simpler, more compatible on ARM
        reduce_range=True,   # safer on ARM NEON
    )
    print(f"INT8 ONNX saved -> {int8_path}  ({int8_path.stat().st_size/1e6:.1f} MB)")


def verify(model, path, img_size):
    import onnxruntime as ort
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        pt_out = model(dummy).numpy()
    sess   = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(["ms"], {"rgb": dummy.numpy()})[0]
    diff = np.abs(pt_out - ort_out)
    print(f"  PyTorch vs ONNX: max_diff={diff.max():.6f}  mean_diff={diff.mean():.8f}")
    if diff.max() < 1e-3:
        print("  ✓ Outputs match")


def benchmark(path, img_size, n=20):
    import onnxruntime as ort
    sess  = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    dummy = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    for _ in range(3): sess.run(["ms"], {"rgb": dummy})   # warmup
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(["ms"], {"rgb": dummy})
        times.append(time.perf_counter() - t0)
    m, s = np.mean(times)*1000, np.std(times)*1000
    print(f"  Inference: {m:.1f} ± {s:.1f} ms  "
          f"(this machine — Pi 4 will be ~4-8× slower)")


def main():
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    fp32 = out_dir / "rgb2ms_fp32.onnx"
    int8 = out_dir / "rgb2ms_int8.onnx"

    # Load
    ckpt  = torch.load(args.ckpt, map_location="cpu")
    model = MobileNetV2UNet(out_channels=4, pretrained=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded epoch {ckpt.get('epoch','?')}  R²={ckpt.get('best_r2',0):.4f}\n")

    # Export
    print("Step 1: Export FP32 ONNX")
    export_fp32(model, args.img_size, fp32, args.opset)

    print("\nStep 2: Verify FP32")
    try:
        import onnxruntime
        verify(model, fp32, args.img_size)
    except ImportError:
        print("  onnxruntime not installed — install: pip install onnxruntime")

    print("\nStep 3: Quantize to INT8")
    try:
        quantize_int8(fp32, int8)
    except Exception as e:
        print(f"  INT8 failed: {e}")
        print("  Install: pip install onnxruntime onnxruntime-tools")

    print("\nStep 4: Benchmark FP32")
    try:
        benchmark(fp32, args.img_size)
        if int8.exists():
            print("Benchmark INT8")
            benchmark(int8, args.img_size)
    except Exception as e:
        print(f"  Skipped: {e}")

    print(f"\n✓ Deploy  {int8.name}  on Raspberry Pi 4.")
    print(f"  Copy it with:  scp {int8} pi@<pi-ip>:~/rgb2ms/")


if __name__ == "__main__":
    main()
