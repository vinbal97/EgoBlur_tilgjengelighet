# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Batch anonymization script for processing folders of images.

Usage example:
python anonymise_images.py \
    --face_model_path ego_blur_face.jit \
    --lp_model_path ego_blur_lp.jit \
    --input_folder_path /path/to/input/images \
    --output_folder_path /path/to/output/images
"""

import argparse
import os
from functools import lru_cache
from typing import List
import glob

import cv2
import numpy as np
import torch
import torchvision


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--face_model_path",
        required=False,
        type=str,
        default=None,
        help="Absolute EgoBlur face model file path",
    )

    parser.add_argument(
        "--face_model_score_threshold",
        required=False,
        type=float,
        default=0.2,
        help="Face model score threshold to filter out low confidence detections",
    )

    parser.add_argument(
        "--lp_model_path",
        required=False,
        type=str,
        default=None,
        help="Absolute EgoBlur license plate model file path",
    )

    parser.add_argument(
        "--lp_model_score_threshold",
        required=False,
        type=float,
        default=0.2,
        help="License plate model score threshold to filter out low confidence detections",
    )

    parser.add_argument(
        "--nms_iou_threshold",
        required=False,
        type=float,
        default=0.2,
        help="NMS iou threshold to filter out low confidence overlapping boxes",
    )

    parser.add_argument(
        "--scale_factor_detections",
        required=False,
        type=float,
        default=1,
        help="Scale detections by the given factor to allow blurring more area, 1.15 would mean 15% scaling",
    )

    parser.add_argument(
        "--input_folder_path",
        required=True,
        type=str,
        help="Absolute path to the folder containing images to anonymize",
    )

    parser.add_argument(
        "--output_folder_path",
        required=True,
        type=str,
        help="Absolute path to the folder where anonymized images will be saved",
    )

    parser.add_argument(
        "--image_extensions",
        required=False,
        type=str,
        nargs="+",
        default=["jpg", "jpeg", "png", "bmp", "tiff", "tif"],
        help="List of image file extensions to process",
    )

    parser.add_argument(
        "--skip_existing",
        required=False,
        action="store_true",
        default=True,
        help="Skip processing if output image already exists",
    )

    return parser.parse_args()


def create_output_directory(dir_path: str) -> None:
    """
    parameter dir_path: absolute path to the directory to create
    Simple logic to create output directories if they don't exist.
    """
    if not os.path.exists(dir_path):
        print(f"Directory {dir_path} does not exist. Attempting to create it...")
        os.makedirs(dir_path, exist_ok=True)
        if not os.path.exists(dir_path):
            raise ValueError(
                f"Directory {dir_path} didn't exist. Attempt to create also failed. Please provide another path."
            )


def validate_inputs(args: argparse.Namespace) -> argparse.Namespace:
    """
    parameter args: parsed arguments
    Run some basic checks on the input arguments
    """
    # input args value checks
    if not 0.0 <= args.face_model_score_threshold <= 1.0:
        raise ValueError(
            f"Invalid face_model_score_threshold {args.face_model_score_threshold}"
        )
    if not 0.0 <= args.lp_model_score_threshold <= 1.0:
        raise ValueError(
            f"Invalid lp_model_score_threshold {args.lp_model_score_threshold}"
        )
    if not 0.0 <= args.nms_iou_threshold <= 1.0:
        raise ValueError(f"Invalid nms_iou_threshold {args.nms_iou_threshold}")
    if not 0 <= args.scale_factor_detections:
        raise ValueError(
            f"Invalid scale_factor_detections {args.scale_factor_detections}"
        )

    # input/output paths checks
    if args.face_model_path is None and args.lp_model_path is None:
        raise ValueError(
            "Please provide either face_model_path or lp_model_path or both"
        )
    
    if not os.path.exists(args.input_folder_path):
        raise ValueError(f"Input folder {args.input_folder_path} does not exist.")
    
    if not os.path.isdir(args.input_folder_path):
        raise ValueError(f"{args.input_folder_path} is not a directory.")
    
    if args.face_model_path is not None and not os.path.exists(args.face_model_path):
        raise ValueError(f"{args.face_model_path} does not exist.")
    
    if args.lp_model_path is not None and not os.path.exists(args.lp_model_path):
        raise ValueError(f"{args.lp_model_path} does not exist.")
    
    # Create output directory if it doesn't exist
    create_output_directory(args.output_folder_path)
    
    # check we have write permissions on output folder
    if not os.access(args.output_folder_path, os.W_OK):
        raise ValueError(
            f"You don't have permissions to write to {args.output_folder_path}. Please grant adequate permissions, or provide a different output path."
        )

    return args


@lru_cache
def get_device() -> str:
    """
    Return the device type
    """
    return (
        "cpu"
        #if not torch.cuda.is_available()
        #else f"cuda:{torch.cuda.current_device()}"
    )


def read_image(image_path: str) -> np.ndarray:
    """
    parameter image_path: absolute path to an image
    Return an image in BGR format
    """
    bgr_image = cv2.imread(image_path)
    if bgr_image is None:
        raise ValueError(f"Could not read image from {image_path}")
    if len(bgr_image.shape) == 2:
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
    return bgr_image


def write_image(image: np.ndarray, image_path: str) -> None:
    """
    parameter image: np.ndarray in BGR format
    parameter image_path: absolute path where we want to save the visualized image
    """
    cv2.imwrite(image_path, image)


def get_image_tensor(bgr_image: np.ndarray) -> torch.Tensor:
    """
    parameter bgr_image: image on which we want to make detections

    Return the image tensor
    """
    bgr_image_transposed = np.transpose(bgr_image, (2, 0, 1))
    image_tensor = torch.from_numpy(bgr_image_transposed).to(get_device())

    return image_tensor


def get_detections(
    detector: torch.jit._script.RecursiveScriptModule,
    image_tensor: torch.Tensor,
    model_score_threshold: float,
    nms_iou_threshold: float,
) -> List[List[float]]:
    """
    parameter detector: Torchscript module to perform detections
    parameter image_tensor: image tensor on which we want to make detections
    parameter model_score_threshold: model score threshold to filter out low confidence detection
    parameter nms_iou_threshold: NMS iou threshold to filter out low confidence overlapping boxes

    Returns the list of detections
    """
    with torch.no_grad():
        detections = detector(image_tensor)

    boxes, _, scores, _ = detections  # returns boxes, labels, scores, dims

    nms_keep_idx = torchvision.ops.nms(boxes, scores, nms_iou_threshold)
    boxes = boxes[nms_keep_idx]
    scores = scores[nms_keep_idx]

    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()

    score_keep_idx = np.where(scores > model_score_threshold)[0]
    boxes = boxes[score_keep_idx]
    return boxes.tolist()


def scale_box(
    box: List[List[float]], max_width: int, max_height: int, scale: float
) -> List[List[float]]:
    """
    parameter box: detection box in format (x1, y1, x2, y2)
    parameter scale: scaling factor

    Returns a scaled bbox as (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    w = x2 - x1
    h = y2 - y1

    xc = x1 + w / 2
    yc = y1 + h / 2

    w = scale * w
    h = scale * h

    x1 = max(xc - w / 2, 0)
    y1 = max(yc - h / 2, 0)

    x2 = min(xc + w / 2, max_width)
    y2 = min(yc + h / 2, max_height)

    return [x1, y1, x2, y2]


def visualize(
    image: np.ndarray,
    detections: List[List[float]],
    scale_factor_detections: float,
) -> np.ndarray:
    """
    parameter image: image on which we want to make detections
    parameter detections: list of bounding boxes in format [x1, y1, x2, y2]
    parameter scale_factor_detections: scale detections by the given factor to allow blurring more area, 1.15 would mean 15% scaling

    Visualize the input image with the detections and save the output image at the given path
    """
    image_fg = image.copy()
    mask_shape = (image.shape[0], image.shape[1], 1)
    mask = np.full(mask_shape, 0, dtype=np.uint8)

    for box in detections:
        if scale_factor_detections != 1.0:
            box = scale_box(
                box, image.shape[1], image.shape[0], scale_factor_detections
            )
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        w = x2 - x1
        h = y2 - y1

        ksize = (image.shape[0] // 2, image.shape[1] // 2)
        image_fg[y1:y2, x1:x2] = cv2.blur(image_fg[y1:y2, x1:x2], ksize)
        cv2.ellipse(mask, (((x1 + x2) // 2, (y1 + y2) // 2), (w, h), 0), 255, -1)

    inverse_mask = cv2.bitwise_not(mask)
    image_bg = cv2.bitwise_and(image, image, mask=inverse_mask)
    image_fg = cv2.bitwise_and(image_fg, image_fg, mask=mask)
    image = cv2.add(image_bg, image_fg)

    return image


def anonymize_image(
    input_image_path: str,
    face_detector: torch.jit._script.RecursiveScriptModule,
    lp_detector: torch.jit._script.RecursiveScriptModule,
    face_model_score_threshold: float,
    lp_model_score_threshold: float,
    nms_iou_threshold: float,
    output_image_path: str,
    scale_factor_detections: float,
) -> bool:
    """
    parameter input_image_path: absolute path to the input image
    parameter face_detector: face detector model to perform face detections
    parameter lp_detector: license plate detector model to perform license plate detections
    parameter face_model_score_threshold: face model score threshold to filter out low confidence detection
    parameter lp_model_score_threshold: license plate model score threshold to filter out low confidence detection
    parameter nms_iou_threshold: NMS iou threshold
    parameter output_image_path: absolute path where the visualized image will be saved
    parameter scale_factor_detections: scale detections by the given factor to allow blurring more area

    Perform detections on the input image and save the output image at the given path.
    Returns True if successful, False otherwise.
    """
    try:
        bgr_image = read_image(input_image_path)
        image = bgr_image.copy()

        image_tensor = get_image_tensor(bgr_image)
        image_tensor_copy = image_tensor.clone()
        detections = []
        
        # get face detections
        if face_detector is not None:
            detections.extend(
                get_detections(
                    face_detector,
                    image_tensor,
                    face_model_score_threshold,
                    nms_iou_threshold,
                )
            )

        # get license plate detections
        if lp_detector is not None:
            detections.extend(
                get_detections(
                    lp_detector,
                    image_tensor_copy,
                    lp_model_score_threshold,
                    nms_iou_threshold,
                )
            )
        
        image = visualize(
            image,
            detections,
            scale_factor_detections,
        )
        write_image(image, output_image_path)
        return True
    except Exception as e:
        print(f"Error processing {input_image_path}: {str(e)}")
        return False


def get_image_files(input_folder: str, extensions: List[str]) -> List[str]:
    """
    Get all image files from the input folder with specified extensions.
    
    parameter input_folder: path to the input folder
    parameter extensions: list of file extensions to look for
    
    Returns list of image file paths
    """
    image_files = []
    
    for ext in extensions:
        # Add both lowercase and uppercase versions
        pattern = os.path.join(input_folder, f"*.{ext.lower()}")
        image_files.extend(glob.glob(pattern))
        pattern = os.path.join(input_folder, f"*.{ext.upper()}")
        image_files.extend(glob.glob(pattern))
    
    # Remove duplicates and sort
    image_files = sorted(list(set(image_files)))
    return image_files


def process_images_folder(
    input_folder_path: str,
    output_folder_path: str,
    face_detector: torch.jit._script.RecursiveScriptModule,
    lp_detector: torch.jit._script.RecursiveScriptModule,
    face_model_score_threshold: float,
    lp_model_score_threshold: float,
    nms_iou_threshold: float,
    scale_factor_detections: float,
    image_extensions: List[str],
    skip_existing: bool,
):
    """
    Process all images in the input folder and save anonymized versions to output folder.
    """
    image_files = get_image_files(input_folder_path, image_extensions)
    
    if not image_files:
        print(f"No image files found in {input_folder_path} with extensions {image_extensions}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    processed = 0
    skipped = 0
    failed = 0
    
    for image_file in image_files:
        # Get the filename without path
        filename = os.path.basename(image_file)
        output_path = os.path.join(output_folder_path, filename)
        
        # Check if output already exists and skip_existing is True
        if skip_existing and os.path.exists(output_path):
            print(f"Skipping {filename} - already exists in output folder")
            skipped += 1
            continue
        
        print(f"Processing {filename}...")
        
        success = anonymize_image(
            image_file,
            face_detector,
            lp_detector,
            face_model_score_threshold,
            lp_model_score_threshold,
            nms_iou_threshold,
            output_path,
            scale_factor_detections,
        )
        
        if success:
            processed += 1
            print(f"Successfully processed {filename}")
        else:
            failed += 1
    
    print(f"\nProcessing complete!")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    args = validate_inputs(parse_args())
    
    # Load models
    if args.face_model_path is not None:
        face_detector = torch.jit.load(args.face_model_path, map_location="cpu").to(
            get_device()
        )
        face_detector.eval()
    else:
        face_detector = None

    if args.lp_model_path is not None:
        lp_detector = torch.jit.load(args.lp_model_path, map_location="cpu").to(
            get_device()
        )
        lp_detector.eval()
    else:
        lp_detector = None

    # Process all images in the folder
    process_images_folder(
        args.input_folder_path,
        args.output_folder_path,
        face_detector,
        lp_detector,
        args.face_model_score_threshold,
        args.lp_model_score_threshold,
        args.nms_iou_threshold,
        args.scale_factor_detections,
        args.image_extensions,
        args.skip_existing,
    )