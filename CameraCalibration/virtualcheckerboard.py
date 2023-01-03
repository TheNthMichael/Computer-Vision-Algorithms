"""
Author: Michael Nickerson
Date: 01/03/2023
Description: This program creates a virtual checkerboard with the correct dimensions for camera calibration.
Reference: https://aliyasineser.medium.com/opencv-camera-calibration-e9a48bdd1844 used for image and calibration method.
"""
import argparse
import cv2
import math
import numpy as np
import sys

def calibration(images, square_size, width, height):
    """
    Calibrate the camera using images.
    Source: https://aliyasineser.medium.com/opencv-camera-calibration-e9a48bdd1844 
    """
    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return [ret, mtx, dist, rvecs, tvecs]

def calibrate_camera(square_size, width, height):
    """
    Calibrate the camera using a video capture.
    """
    images = []
    webcam = cv2.VideoCapture(0)

    while(True):
        ret, frame = webcam.read()
        if not ret:
            continue

        cv2.imshow('frame', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('e'):
            images.append(frame)

        if key == ord('q'):
            break
    
    webcam.release()
    return calibration(images, square_size, width, height)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Virtual Camera Calibration Checkerboard",
        description="Uses your computer's monitor to display an image usable for camera calibration."
    )

    subparser = parser.add_subparsers(help="Please choose wh if you have the width and height or ar if you have the aspect ratio and diagonal of your monitor.")

    group_width_height = subparser.add_parser("wh", help="Uses width and height of the monitor for camera calibration")
    group_width_height.set_defaults(mode = "wh")
    group_width_height.add_argument("--cmwidth", type=float, required=True)
    group_width_height.add_argument("--cmheight", type=float, required=True)
    group_width_height.add_argument("--pixelwidth", type=float, required=True)
    group_width_height.add_argument("--pixelheight", type=float, required=True)

    group_ar_diagonal = subparser.add_parser("ar", help="Uses aspect ratio and diagonal measurement of the monitor for camera calibration")
    group_ar_diagonal.set_defaults(mode = "ar")
    group_ar_diagonal.add_argument("--arwidth", type=float, required=True)
    group_ar_diagonal.add_argument("--arheight", type=float, required=True)
    group_ar_diagonal.add_argument("--cmdiagonal", type=float, required=True)
    group_ar_diagonal.add_argument("--pixelwidth", type=float, required=True)
    group_ar_diagonal.add_argument("--pixelheight", type=float, required=True)

    args = parser.parse_args()

    image_file = "chessboard.png"
    rows = 7
    columns = 10
    side_length_cm = 1.5

    width_cm = 0
    height_cm = 0
    width_pixels = 0
    height_pixels = 0

    if args.mode == "wh":
        # No conversions needed.
        width_cm = args.cmwidth
        height_cm = args.cmheight
        width_pixels = args.pixelwidth
        height_pixels = args.pixelheight
    elif args.mode == "ar":
        # Need to convert to width and height
        width_pixels = args.pixelwidth
        height_pixels = args.pixelheight
        aspect_ratio = args.arwidth / args.arheight
        height_cm = args.cmdiagonal / math.sqrt((aspect_ratio)**2 + 1)
        width_cm = height_cm * aspect_ratio
    else:
        print("Please run with either --wh or --ar flags enabled...")
        sys.exit(0)

    # chessboard.png is 10 squares in width and 7 squares in height.
    # The side length of each square should physically be 1.5 cm.
    # For a 2560x1440 monitor with 16:9 aspect ratio and 68.58cm diagonal,
    # width = 59.773cm, height = 33.622cm
    # We need at least 10 * 1.5cm (15cm) width and 7 * 1.5cm (10.5cm) height which is well in our screen space budget.
    if width_cm < columns * side_length_cm or height_cm < rows * side_length_cm:
        print("Sorry, your screen is too small to display this image as its real size.")
        print(f"(Requires minimum monitor dimensions of {columns * side_length_cm}cm x {rows * side_length_cm}cm)")
        sys.exit(0)

    # image size is dictated by pixel width and height, we need to find out what pixel width and height gives the displayed
    # image a physical size of 15cm x 10.5cm. Dimensions of image are 3000px x 2100px with a 10:7 aspect ratio.
    # 3000px * k = 15cm st k = cm / px
    img = cv2.imread(image_file)
    height, width, _ = img.shape
    print(f"image: {width}, {height}")
    print(f"monitor: {width_cm}, {height_cm}")
    k_x = width_cm / width_pixels
    k_y = height_cm / height_pixels
    print(f"k: {k_x}, {k_y}")
    dims = (int(columns * side_length_cm / k_x), int(rows * side_length_cm / k_y))

    print(dims)

    #sys.exit()

    img = cv2.resize(img, dims, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Virtual Camera Calibration", img)
    
    # Run camera calibration.
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(side_length_cm, columns - 1, rows - 1)
    K = mtx # Intrinsics
    D = dist # Extrinsics

    print(f"Intrinsics: {K}")
    print(f"Extrinsics: {D}")


    # Close the window.
    cv2.destroyAllWindows()
    

    