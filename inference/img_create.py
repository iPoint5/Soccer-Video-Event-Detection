import os
import cv2
import numpy as np


def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def resize_images_to_same_size(images):
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    target_height = min(heights)
    target_width = min(widths)
    resized = [cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA) for img in images]
    return resized


def stitch_2x2(image_paths, output_path=None, resize_to_min=True):
    """Stitch four images into a single 2x2 image.

    Args:
        image_paths (list[str]): List of exactly 4 image file paths in order [top-left, top-right, bottom-left, bottom-right].
        output_path (str|None): If given, save the stitched image to this path.
        resize_to_min (bool): Resize all images to the smallest width and height among them.

    Returns:
        np.ndarray: The stitched image in BGR format.
    """
    if len(image_paths) != 4:
        raise ValueError("image_paths must contain exactly 4 image file paths")

    images = [load_image(path) for path in image_paths]
    if resize_to_min:
        images = resize_images_to_same_size(images)

    top_row = np.hstack([images[0], images[1]])
    bottom_row = np.hstack([images[2], images[3]])
    stitched = np.vstack([top_row, bottom_row])

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, stitched)

    return stitched


if __name__ == "__main__":
    # 请在这里填写四张图片的文件路径，顺序为：
    # 左上, 右上, 左下, 右下
    image_paths = [
        r"E:\sn-spotting-main\Benchmarks\CALF\inference\outputs\9.png",
        r"E:\sn-spotting-main\Benchmarks\CALF\inference\outputs\10.png",
        r"E:\sn-spotting-main\Benchmarks\CALF\inference\outputs\11.png",
        r"E:\sn-spotting-main\Benchmarks\CALF\inference\outputs\12.png",
    ]

    output_path = r"E:\sn-spotting-main\Benchmarks\CALF\inference\outputs\stitched_output.png"
    resize_to_min = True  # 若不希望自动缩放为统一尺寸，可改为 False

    stitched_image = stitch_2x2(image_paths, output_path=output_path, resize_to_min=resize_to_min)
    print(f"拼接完成，已保存到：{output_path}")
