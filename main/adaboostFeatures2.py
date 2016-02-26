__authors__ = "Isabel Litton, Vincent Pham, Henry Tom"
__team__ = "CaptainDataCrunch"

from adaboostImages import *

def feat_two_rectangles(gray_img, block_num):
	"""Calculates integral images for two windows A and B then finds their difference
	:param gray_img: array of pixels
	:param block_num: tuple (block's upper left corner, block's bottom right corner)
	:return: I(B) - I(A) where I() = integral image
	"""
	half_x = (block_num[2]-block_num[0])/2 - 1
	A = (block_num[0], block_num[1], half_x, block_num[3])
	B = (half_x + 1, block_num[1], block_num[2], block_num[3])

	A_sum = integralImage(gray_img,A)
	B_sum = integralImage(gray_img,B)
	return (B_sum - A_sum)

def feat_three_rectangles_horizontal(gray_img, block_num, height1=24, height2=14):
	"""Calculates integral images for three windows A, B, and C
	:param gray_img: array of pixels
	:param block_num: tuple (block's upper left corner, block's bottom right corner)
	:return: I(B) - (I(A) + I(C)) where I() = integral image
	"""

	A = (block_num[0], block_num[1], block_num[2], block_num[13])
	B = (block_num[0], block_num[1]+height1+1, block_num[2],  block_num[1]+height1+height2)
	C = (block_num[0], block_num[1]+height1+height2+1, block_num[2], block_num[3])

	A_sum = integralImage(gray_img,A)
	B_sum = integralImage(gray_img,B)
	C_sum = integralImage(gray_img,C)
	return (B_sum - (A_sum + C_sum))

def feat_three_rectangles_vertical(gray_img, block_num, width1=52, width2=24):
	"""Calculates integral images for three windows A, B, and C
	:param gray_img: array of pixels
	:param block_num: tuple (block's upper left corner, block's bottom right corner)
	:return: I(B) - (I(A) + I(C)) where I() = integral image
	"""

	A = (block_num[0], block_num[1], block_num[0]+width1, block_num[3])
	B = (block_num[0]+width1+1, block_num[1], block_num[0]+width1+width2, block_num[3])
	C = (block_num[0]+width1+width2+1, block_num[1], block_num[2], block_num[3])

	A_sum = integralImage(gray_img,A)
	B_sum = integralImage(gray_img,B)
	C_sum = integralImage(gray_img,C)
	return (B_sum - (A_sum + C_sum))

def feat_four_rectangles(gray_img, block_num):
	"""Calculates integral images for four windows in a square [A, B][C, D]
	:param gray_img: array of pixels
	:param block_num: tuple (block's upper left corner, block's bottom right corner)
	:return: (I(A) + I(D)) - (I(B) + I(C)) where I() = integral image
	"""
	half_x = (block_num[2] - block_num[0])/2 - 1
	half_y = (block_num[3] - block_num[1])/2 - 1

	A = (block_num[0], block_num[1], half_x, half_y)
	B = (half_x + 1, block_num[1], block_num[2], half_y)
	C = (block_num[0], half_y + 1, half_x, block_num[3])
	D = (half_x + 1, half_y + 1, block_num[2], block_num[3])

	A_sum = integralImage(gray_img,A)
	B_sum = integralImage(gray_img,B)
	C_sum = integralImage(gray_img,C)
	D_sum = integralImage(gray_img,D)
	return ((A_sum + D_sum) - (B_sum + C_sum))