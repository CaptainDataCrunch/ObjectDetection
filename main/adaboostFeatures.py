__authors__ = "Isabel Litton, Vincent Pham, Henry Tom"
__team__ = "CaptainDataCrunch"

from adaboostImages import *

def feat_two_rectangles(gray_img, coords):
    """Calculates integral images for two windows A and B then finds their difference
    :param gray_img: array of pixels
    :param coords: tuple (block's upper left corner, block's bottom right corner)
    :return: I(B) - I(A) where I() = integral image
    """
    half_x = (coords[2] - coords[0]) / 2 - 1
    A = (coords[0], coords[1], coords[0] + half_x, coords[3])
    B = (coords[0] + half_x + 1, coords[1], coords[2], coords[3])

    A_sum = integralImage(gray_img, A)
    # print gray_img.shape
    # print A
    # print B
    # print A_sum
    B_sum = integralImage(gray_img, B)
    # print B_sum
    return (B_sum - A_sum)

def feat_three_rectangles_horizontal(gray_img, coords, height1=24, height2=14):
    """Calculates integral images for three windows A, B, and C
    :param gray_img: array of pixels
    :param coords: tuple (block's upper left corner, block's bottom right corner)
    :return: I(B) - (I(A) + I(C)) where I() = integral image
    """
    A = (coords[0], coords[1], coords[2], coords[1] + height1)
    B = (coords[0], coords[1] + height1 + 1, coords[2], coords[1] + height1 + height2)
    C = (coords[0], coords[1] + height1 + height2 + 1, coords[2], coords[3])

    A_sum = integralImage(gray_img, A)
    B_sum = integralImage(gray_img, B)
    C_sum = integralImage(gray_img, C)
    return (B_sum - (A_sum + C_sum))

def feat_three_rectangles_vertical(gray_img, coords, width1=52, width2=24):
    """Calculates integral images for three windows A, B, and C
    :param gray_img: array of pixels
    :param coords: tuple (block's upper left corner, block's bottom right corner)
    :return: I(B) - (I(A) + I(C)) where I() = integral image
    """
    A = (coords[0], coords[1], coords[0] + width1, coords[3])
    B = (coords[0] + width1 + 1, coords[1], coords[0] + width1 + width2, coords[3])
    C = (coords[0] + width1 + width2 + 1, coords[1], coords[2], coords[3])

    A_sum = integralImage(gray_img, A)
    B_sum = integralImage(gray_img, B)
    C_sum = integralImage(gray_img, C)
    return (B_sum - (A_sum + C_sum))

def feat_four_rectangles(gray_img, coords):
    """Calculates integral images for four windows in a square [A, B][C, D]
    :param gray_img: array of pixels
    :param coords: tuple (block's upper left corner, block's bottom right corner)
    :return: (I(A) + I(D)) - (I(B) + I(C)) where I() = integral image
    """
    half_x = (coords[2] - coords[0]) / 2 - 1
    half_y = (coords[3] - coords[1]) / 2 - 1

    A = (coords[0], coords[1], coords[0] + half_x, coords[1] + half_y)
    B = (coords[0] + half_x + 1, coords[1], coords[2], coords[1] + half_y)
    C = (coords[0], coords[1] + half_y + 1, coords[0] + half_x, coords[3])
    D = (coords[0] + half_x + 1, coords[1] + half_y + 1, coords[2], coords[3])

    A_sum = integralImage(gray_img, A)
    B_sum = integralImage(gray_img, B)
    C_sum = integralImage(gray_img, C)
    D_sum = integralImage(gray_img, D)
    return ((A_sum + D_sum) - (B_sum + C_sum))
