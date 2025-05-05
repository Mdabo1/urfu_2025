from pathlib import Path
import cv2
import numpy as np

__all__ = ["preprocess_image"]

def _is_office_doc(img: np.ndarray) -> bool:
    # Heuristic: bright, low-contrast, uniform white pages are office docs
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    mean_l, std_l, mean_b = float(l.mean()), float(l.std()), float(b.mean())
    white_ratio = float((l > 245).sum()) / l.size
    return (mean_l > 200 and std_l < 8 and mean_b < 5) or (white_ratio > 0.8)


def preprocess_image(image_path: Path) -> Path:
    """
    Preprocess historical manuscripts for Tesseract OCR, WITHOUT any deskew or rotation.
    Office documents are returned unchanged.
    """
    # Load image with OpenCV (ignores EXIF orientation)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    # Skip modern office scans
    if _is_office_doc(img):
        return image_path

    # Upscale if low resolution
    h, w = img.shape[:2]
    if max(h, w) < 1500:
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Contrast boost with CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Denoise: bilateral + median blur
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.medianBlur(gray, 3)

    # Adaptive Gaussian threshold (black text on white)
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=41, C=5
    )
    # Ensure black text on white background
    if float(th.mean()) < 128:
        th = cv2.bitwise_not(th)

    # Remove small pepper noise
    th = cv2.medianBlur(th, 3)

    # Close small gaps in strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Write result without any rotation or deskew
    out_path = image_path.with_name(f"{image_path.stem}_preprocessed.png")
    cv2.imwrite(str(out_path), clean)
    return out_path
