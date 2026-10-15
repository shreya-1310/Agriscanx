# colab_setup.py
# Run this cell first in Google Colab to install all dependencies
# and mount Google Drive.
#
# Paste each block into a Colab cell and run in order.

SETUP_CELL = """
# Cell 1 — Mount Drive & install packages
from google.colab import drive
drive.mount('/content/drive')

!pip install -q tifffile onnx onnxruntime torch torchvision
print("Setup complete")
"""

STATS_CELL = """
# Cell 2 — Compute dataset normalization stats (run once)
# Replace the path with your actual Drive path
!python compute_stats.py \\
    --data "/content/drive/MyDrive/WeedyRice-RGBMS-DB (1)/WeedyRice-RGBMS-DB"
# Then copy the printed MS_MEAN / MS_STD into dataset.py and rpi_inference.py
"""

TRAIN_CELL = """
# Cell 3 — Train
!python train.py \\
    --data "/content/drive/MyDrive/WeedyRice-RGBMS-DB (1)/WeedyRice-RGBMS-DB" \\
    --out  "/content/drive/MyDrive/rgb2ms_checkpoints" \\
    --epochs 100 \\
    --batch  8 \\
    --img_size 512
# If Colab runs out of memory:  --batch 4 --img_size 384
"""

EVAL_CELL = """
# Cell 4 — Evaluate on test set
!python evaluate.py \\
    --data "/content/drive/MyDrive/WeedyRice-RGBMS-DB (1)/WeedyRice-RGBMS-DB" \\
    --ckpt "/content/drive/MyDrive/rgb2ms_checkpoints/best_model.pth" \\
    --out  "/content/drive/MyDrive/rgb2ms_eval" \\
    --save_vis
"""

EXPORT_CELL = """
# Cell 5 — Export to ONNX + INT8 for Raspberry Pi
!python export_onnx.py \\
    --ckpt     "/content/drive/MyDrive/rgb2ms_checkpoints/best_model.pth" \\
    --out      "/content/drive/MyDrive/rgb2ms_models" \\
    --img_size 512
# Output: rgb2ms_int8.onnx — copy this to your Raspberry Pi
"""

if __name__ == "__main__":
    print("Colab cells:\n")
    for name, cell in [("SETUP",  SETUP_CELL),
                        ("STATS",  STATS_CELL),
                        ("TRAIN",  TRAIN_CELL),
                        ("EVAL",   EVAL_CELL),
                        ("EXPORT", EXPORT_CELL)]:
        print(f"=== {name} ===")
        print(cell)
