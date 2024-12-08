#!/usr/bin/env python
import cv2
import numpy as np


def resize(image, factor=0.5, iterations=2):
    # prefer more iterations with smaller steps
    _iter = image
    for _ in range(iterations):
        _iter = cv2.resize(
            _iter, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR
        )
    return _iter


def threshold(image):
    _, thresholded = cv2.threshold(image, 125, 255, cv2.THRESH_OTSU)
    return thresholded


def blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


def convolve(image1, image2):
    if len(image2.shape) == 2 and len(image1.shape) == 3:
        image2 = image2[..., np.newaxis]
        image2 = np.repeat(image2, 3, axis=2)
    # have to use higher precision because addition goes over 255 and then clip
    # to make white whatever is higher than 255 but preserve the other colors
    # normalization causes them to become darker
    res = np.clip(
        ((image1.astype(np.uint16) + image2.astype(np.uint16))),
        0,
        255,
    ).astype(np.uint8)
    return res


def grayscale(image):
    if len(image.shape) == 2 or image.shape[2] == 1:
        return image
    # Y=0.299R+0.587G+0.114B
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def bloom(image, intensity=0.10):
    grayscaled = grayscale(image)
    thresholded = threshold(grayscaled)
    downsampled = resize(thresholded, 0.5, 2)
    blurred = blur(downsampled)
    upscaled = resize(blurred, 2, 2)
    bloom = (upscaled * intensity).astype(np.uint8)
    blended = convolve(image, bloom)

    return blended


def bloom_per_channel(image, intensity=0.10):
    # quick and VERY dirty
    b, g, r = cv2.split(image)
    bb = bloom(b, intensity)
    bg = bloom(g, intensity)
    br = bloom(r, intensity)
    return cv2.merge([bb, bg, br])


if __name__ == "__main__":

    TRACK_INTENSITY = 0.1

    def on_trackbar(val):
        global TRACK_INTENSITY
        scaled_value = val / 100.0
        TRACK_INTENSITY = scaled_value

    image = cv2.imread("test2.jpg")
    # avoid silly dimension issues
    # image = image[:400]

    try:
        cv2.namedWindow("Bloom")
        cv2.createTrackbar("Bloom Intensity", "Bloom", 0, 100, on_trackbar)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                cv2.imwrite("_out.png", bloomed)
                print("+ Saved output to _out.png")
            bloomed = bloom_per_channel(image, TRACK_INTENSITY)
            cv2.imshow("Bloom", np.concatenate((image, bloomed), axis=1))
    except Exception as e:
        print(f"! Something went very wrong: {e}")

    cv2.destroyAllWindows()
