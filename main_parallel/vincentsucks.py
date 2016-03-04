from mpi4py import MPI
from adaboostImages2 import *
import numpy as np
import time

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
    B_sum = integralImage(gray_img, B)
    # print B_sum
    # print A_sum, B_sum
    diff = B_sum - A_sum
    print diff
    return (B_sum - A_sum)

timenow = time.time()
filename = 'bg_8.jpg'
gray_img = read_image(filename)
# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# rank = comm.Get_rank()

print feat_two_rectangles(gray_img, (10,10,50,50))
print time.time() - timenow

# def isabel():
#     if rank == 0:
#         A = feat_two_rectangles(gray_img, (10,10,50,50))
#         print A
#         comm.send(A, dest=2, tag=11)
#     if rank == 1:
#         B = feat_two_rectangles(gray_img, (10,20,50,60))
#         print B
#         comm.send(B, dest=2, tag=12)
#     if rank == 2:
#         data1 = comm.recv(source=0, tag=11)
#         data2 = comm.recv(source=1, tag=12)
#         print data1 + data2
#     return "Time:", time.time() - timenow

# def bree():
#     A = feat_two_rectangles(gray_img, (10,10,50,50))
#     B = feat_two_rectangles(gray_img, (10,20,50,60))
#     print A+B
#     return "Time:", time.time() - timenow

# print isabel()
