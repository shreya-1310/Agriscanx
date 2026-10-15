"""
rpi_inference.py
----------------
Raspberry Pi 4 deployment script.
Input : any RGB image (JPG/PNG from any camera)
Output: 4 predicted MS band images + NDVI + NDRE maps

Install on Pi:
    pip install onnxruntime numpy Pillow tifffile

Usage:
    # Single image
    python rpi_inference.py --model rgb2ms_int8.onnx --image field.jpg

    # Save scientific TIFFs too
    python rpi_inference.py --model rgb2ms_int8.onnx --image field.jpg --save_tiff

    # Live camera (needs OpenCV)
    python rpi_inference.py --model rgb2ms_int8.onnx --camera
"""

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort

# -----------------------------------------------------------------------
# IMPORTANT: Update these with your actual computed stats
# (copy from compute_stats.py output)
# -----------------------------------------------------------------------
RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MS_MEAN  = np.array([0.1500, 0.1200, 0.1800, 0.3500], dtype=np.float32)
MS_STD   = np.array([0.0800, 0.0700, 0.0900, 0.1500], dtype=np.float32)

BAND_NAMES    = ["Green", "Red", "RedEdge", "NIR"]
BAND_SUFFIXES = ["_G",    "_R",  "_RE",     "_NIR"]
INPUT_SIZE    = 512     # must match what was used in export_onnx.py --img_size


# -----------------------------------------------------------------------
class Predictor:
    def __init__(self, model_path: str):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4          # Pi 4 = 4 cores
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.sess = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        print(f"Loaded: {model_path}")

    def predict_file(self, image_path: str) -> np.ndarray:
        """RGB image path -> (4, H, W) float32 MS reflectance [0,1]."""
        img = Image.open(image_path).convert("RGB")
        return self._run(np.array(img, dtype=np.uint8))

    def predict_array(self, rgb_hwc: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 numpy -> (4, H, W) float32 MS reflectance."""
        return self._run(rgb_hwc)

    def _run(self, rgb: np.ndarray) -> np.ndarray:
        # Resize
        img = Image.fromarray(rgb.astype(np.uint8))
        img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0     # (H, W, 3)

        # Normalize RGB
        arr = (arr - RGB_MEAN) / RGB_STD

        # HWC -> BCHW
        inp = arr.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

        t0  = time.perf_counter()
        out = self.sess.run(["ms"], {"rgb": inp})[0][0]   # (4, H, W)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"Inference: {elapsed:.0f} ms")

        # Denormalise: predicted normalised values -> [0,1] reflectance
        out = out * MS_STD[:, None, None] + MS_MEAN[:, None, None]
        return np.clip(out, 0.0, 1.0)


# -----------------------------------------------------------------------
def ndvi(ms: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red). Range [-1, 1]."""
    nir, red = ms[3], ms[1]
    return np.clip((nir - red) / (nir + red + 1e-8), -1, 1)

def ndre(ms: np.ndarray) -> np.ndarray:
    """NDRE = (NIR - RedEdge) / (NIR + RedEdge). Range [-1, 1]."""
    nir, re = ms[3], ms[2]
    return np.clip((nir - re) / (nir + re + 1e-8), -1, 1)


# -----------------------------------------------------------------------
def save_results(ms: np.ndarray, stem: str, out_dir: Path, save_tiff=False):
    out_dir.mkdir(parents=True, exist_ok=True)

    for b, (name, suf) in enumerate(zip(BAND_NAMES, BAND_SUFFIXES)):
        band_u8 = (ms[b] * 255).astype(np.uint8)
        Image.fromarray(band_u8).save(out_dir / f"{stem}{suf}.png")
        if save_tiff:
            import tifffile
            tifffile.imwrite(
                str(out_dir / f"{stem}{suf}.tif"),
                (ms[b] * 65535).astype(np.uint16)
            )

    # Vegetation index maps  (scale [-1,1] -> [0,255] for PNG)
    for name, idx_arr in [("NDVI", ndvi(ms)), ("NDRE", ndre(ms))]:
        vis = ((idx_arr + 1.0) / 2.0 * 255).astype(np.uint8)
        Image.fromarray(vis).save(out_dir / f"{stem}_{name}.png")
        print(f"  {name}: min={idx_arr.min():.3f}  "
              f"max={idx_arr.max():.3f}  mean={idx_arr.mean():.3f}")

    print(f"Saved {len(BAND_NAMES)} MS bands + NDVI + NDRE -> {out_dir}/")


# -----------------------------------------------------------------------
def camera_mode(predictor: Predictor, out_dir: Path):
    try:
        import cv2
    except ImportError:
        print("Need OpenCV: pip install opencv-python-headless")
        return

    cap = cv2.VideoCapture(0)
    print("Camera ready. Press  S = capture & predict,  Q = quit")
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imshow("RGB  [S=capture  Q=quit]", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ms  = predictor.predict_array(rgb)
            save_results(ms, f"capture_{n:04d}", out_dir)
            n += 1
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


# -----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",     required=True,
                   help="Path to rgb2ms_int8.onnx")
    p.add_argument("--image",     default=None,
                   help="Input RGB image path")
    p.add_argument("--camera",    action="store_true",
                   help="Live webcam / Pi Camera mode")
    p.add_argument("--out",       default="output_ms")
    p.add_argument("--save_tiff", action="store_true",
                   help="Also save uint16 TIFFs (scientific use)")
    args = p.parse_args()

    predictor = Predictor(args.model)
    out_dir   = Path(args.out)

    if args.camera:
        camera_mode(predictor, out_dir)

    elif args.image:
        if not Path(args.image).exists():
            print(f"Not found: {args.image}")
            return
        ms = predictor.predict_file(args.image)
        print("\nMS band summary (reflectance):")
        for b, name in enumerate(BAND_NAMES):
            print(f"  {name:10s}: "
                  f"min={ms[b].min():.4f}  max={ms[b].max():.4f}  "
                  f"mean={ms[b].mean():.4f}")
        save_results(ms, Path(args.image).stem, out_dir, args.save_tiff)

    else:
        print("Provide --image <path>  or  --camera")


if __name__ == "__main__":
    main()
