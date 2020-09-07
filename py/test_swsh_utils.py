#
# Grid Optimization Competion
# author: Jesse Holzer
# date: 2020-09-06
#
# to compile use
#
#   $ python setup.py build_ext --inplace
#

import numpy as np
from swsh_utils import solve_cy as swsh_solve
import time

def timeit(function):
    def timed(*args, **kw):
        start_time = time.time()
        result = function(*args, **kw)
        end_time = time.time()
        print('function: {}, time: {}'.format(function.__name__, end_time - start_time))
        return result
    return timed

@timeit
def solve(btar, n, b, x, br, tol):
    swsh_solve(btar, n, b, x, br, tol)

@timeit
def demo1(nh, na, ns, btar, n, b, tol):

    x = np.zeros(shape=(nh,na), dtype=int) # x[h,a] is the solution - number of steps activated in block a of switched shunt h
    br = np.zeros(shape=(nh,2), dtype=float) # column 0 is the residual, column 1 is the absolute value of resid, row h for switched shunt h

    print("nh: {}, na: {}, ns: {}".format(nh, na, ns))
    print('btar: {}'.format(btar))
    print('n: {}'.format(n))
    print('b: {}'.format(b))
    print('tol: {}'.format(tol))
    solve(btar, n, b, x, br, tol)
    print('x: {}'.format(x))
    print('br: {}'.format(br[:,0]))
    print('br abs: {}'.format(br[:,1]))
    print('br abs sorted: {}'.format(np.sort(br[:,1])))
    br_diff = np.amax(np.absolute(br[:,0] - (btar - np.sum(np.multiply(b, x), axis=1).flatten())))
    print('br diff: {}'.format(br_diff))
    print('tol: {}'.format(tol**0.5))
    solve(btar, n, b, x, br, tol**0.5)
    print('x: {}'.format(x))
    print('br: {}'.format(br[:,0]))
    print('br abs: {}'.format(br[:,1]))
    print('br abs sorted: {}'.format(np.sort(br[:,1])))
    br_diff = np.amax(np.absolute(br[:,0] - (btar - np.sum(np.multiply(b, x), axis=1).flatten())))
    print('br diff: {}'.format(br_diff))

@timeit
def demo2(nh, na, ns, tol):

    btar = np.random.randn(nh) # btar[h] is the target susceptance of switched shunt h
    n = np.random.randint(low=0, high=(ns + 1), size=(nh, na), dtype=int) # n[h][a] is the number of steps in block a of switched shunt h
    b = np.random.randn(nh,na) # b[h][a] is the susceptance of each step of block a of switched shunt h
    demo1(nh, na, ns, btar, n, b, tol)

# btar = np.random.randn(nh) # btar[h] is the target susceptance of switched shunt h
# n = np.random.randint(low=0, high=(ns + 1), size=(nh, na), dtype=int) # n[h][a] is the number of steps in block a of switched shunt h
# b = np.random.randn(nh,na) # b[h][a] is the susceptance of each step of block a of switched shunt h
# x = np.zeros(shape=(nh,na), dtype=int) # x[h,a] is the solution - number of steps activated in block a of switched shunt h
# br = np.zeros(shape=(nh,2), dtype=float) # column 0 is the residual, column 1 is the absolute value of resid, row h for switched shunt h

nh = 10 # 500 - max num swshs from our scenarios so far 9/6/2020
na = 8 # 8 - max number of blocks
ns = 11 # 11 - max number of steps
tol = 1e-8
demo2(nh, na, ns, tol)
