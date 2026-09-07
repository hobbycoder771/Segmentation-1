"""

This script:
1. test trained YOLO model in one imagery file
2. Generates tested images in images_test/image_file_{row}_{col}.jpg

Usage:
    python batch_create_dataset.py
"""

from ultralytics import YOLO
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
import numpy as np

model = YOLO(
    r"model\best_yolov8m-seg.pt"
    #r"C:\Users\Gontzal\BILBOMATICA\keremberke\yolov8m-building-segmentation.pt"
    #r"model\runs\train_yolov8m_seg\weights\best.pt"
)

print(type(model))
print(model.task)
print(model.names)


TILE = 640
OVERLAP = 64
STRIDE = TILE - OVERLAP

image_name = "imagery_tile_1.tif"
image_file = f"data\\test\\{image_name}"

images_test_folder_path = r"data/test"

with rasterio.open(image_file) as src:

    print(src.width, src.height)

    for row in range(0, src.height - TILE + 1, STRIDE):
        for col in range(0, src.width - TILE + 1, STRIDE):

            window = Window(col, row, TILE, TILE)

            tile = np.ascontiguousarray(
                src.read([1, 2, 3], window=window).transpose(1, 2, 0)
            )

            # results = model.predict(source=tile, conf=0.25, verbose=False)
            results = model.predict(source=tile, conf=0.5, verbose=False)
            annotated = results[0].plot()

            # plt.imshow(annotated[:, :, ::-1])  # BGR -> RGB
            # plt.axis("off")
            # plt.show()
            results[0].save(
                filename=f"{images_test_folder_path}/{image_name[:-4]}_{row}_{col}.jpg"
            )
            result = results[0]

            print(result.boxes)


# print(f"row={row:4d}, col={col:4d}, tile shape={tile.shape}")

print("finish test")
