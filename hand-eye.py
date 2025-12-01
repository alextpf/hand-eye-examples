import cv2 as cv
import numpy as np
from .util import math_util


def get_rotation(x, transpose=True):

    y = np.reshape(x, (3, 3))

    if transpose:
        y = np.transpose(y)

    det_x = np.linalg.det(y)
    abs_det_x = abs(det_x)
    denom = pow(abs_det_x, 0.333333)
    num = np.sign(det_x)
    y = num/denom * y

    [u, s, vh] = np.linalg.svd(y)
    y = np.matmul(u, vh)

    return y


def handle_func(s):

    [u, s, vh] = np.linalg.svd(s)
    s = np.matmul(u, vh)
    r = np.linalg.det(s)

    if r < 0:  # pragma: nocover
        temp = np.diag(np.array([1, 1, -1]))
        temp = np.matmul(temp, vh)
        s = np.matmul(u, temp)

    return s


def get_transformation(r, t):

    s = np.reshape(r, (3, 3), order='C')

    s = handle_func(s)

    x = np.zeros((4, 4))
    x[0:3, 0:3] = s
    x[0:3, 3] = np.squeeze(t)
    x[3][3] = 1.0

    return x


def calibrate_hand_eye(
        r_gripper2base, t_gripper2base,
        r_target2cam, t_target2cam, algo="tsai"):

    """ Using OpenCV to solve AX = XB """

    if algo == "tsai":
        method = cv.CALIB_HAND_EYE_TSAI  # default
    elif algo == "daniilidis":
        method = cv.CALIB_HAND_EYE_DANIILIDIS
    elif algo == "park":
        method = cv.CALIB_HAND_EYE_PARK

    r_cam2gripper, t_cam2gripper = cv.calibrateHandEye(
        r_gripper2base, t_gripper2base,
        r_target2cam, t_target2cam, method=method)

    return [r_cam2gripper, t_cam2gripper]


def handeye2_shah(aa, bb, num_imgs):
    """
    hand eye calibration with 2 unknowns: AX = YB, with Shah's method
    see also http://faculty.cooper.edu/mili/Calibration/index.html
    """

    A = np.zeros((num_imgs*9, 18))  # noqa N806
    T = np.zeros((9, 9))  # noqa N806
    b = np.zeros((num_imgs*9, 1))

    for i in range(num_imgs):
        a = aa[i]
        b = bb[i]

        [r_a, t_a] = math_util.rt_from_mat(a)  # noqa N806
        [r_b, t_b] = math_util.rt_from_mat(b)  # noqa N806

        T = T + np.kron(r_b, r_a)  # noqa N806

    # Note: compared to Matlab's [U,S,V] = svd(A),
    # np's [u, s, vh] = np.linalg.svd(T):
    # u == U
    # S == diag(s)
    # vh == Hermitian(V) (if V is real ==> transpose(V))

    [u, s, vh] = np.linalg.svd(T)

    x = vh[0, 0:9]
    y = u[0:9, 0]

    X = get_rotation(x)  # noqa N806
    Y = get_rotation(y)  # noqa N806

    A = np.zeros((3*num_imgs, 6))  # noqa N806
    B = np.zeros((3*num_imgs, 1))  # noqa N806

    for i in range(num_imgs):
        tmp_a = aa[i]
        tmp_b = bb[i]
        A[i*3:(i+1)*3, 0:3] = -tmp_a[0:3, 0:3]
        A[i*3:(i+1)*3, 3:6] = np.eye(3)
        t = np.kron(np.transpose(tmp_b[0:3, 3]), np.eye(3))
        m = np.reshape(Y, (9, 1), order='F')
        t = np.matmul(t, m)
        u = np.expand_dims(tmp_a[0:3, 3], axis=2)
        B[i*3:(i+1)*3, :] = u - t

    # solve Ax=B linear system equation
    t = np.linalg.lstsq(A, B)

    new_x = np.zeros((4, 4))
    new_x[0:3, 0:3] = X
    new_x[0:3, 3] = np.squeeze(t[0][0:3])
    new_x[3, 3] = 1.0

    new_y = np.zeros((4, 4))
    new_y[0:3, 0:3] = Y
    new_y[0:3, 3] = np.squeeze(t[0][3:6])
    new_y[3, 3] = 1.0

    return [new_x, new_y]


def handeye2_li(aa, bb, num_imgs):
    """
    hand eye calibration with 2 unknowns: AX = YB, with Li's method
    see also http://faculty.cooper.edu/mili/Calibration/index.html
    """

    eye3 = np.eye(3, dtype=np.float32)

    A = np.zeros((num_imgs*12, 24), dtype=np.float32)  # noqa N806
    B = np.zeros((num_imgs*12, 1), dtype=np.float32)  # noqa N806

    for i in range(num_imgs):
        a = aa[i]
        b = bb[i]

        [r_a, t_a] = math_util.rt_from_mat(a)
        [r_b, t_b] = math_util.rt_from_mat(b)

        tmp = np.zeros((12, 24), dtype=np.float32)
        k1 = np.kron(r_a, eye3)
        k2 = np.kron(-eye3, np.transpose(r_b))
        k3 = np.kron(eye3, np.transpose(t_b))

        tmp[0:9, 0:9] = k1
        tmp[0:9, 9:18] = k2
        tmp[9:12, 9:18] = k3
        tmp[9:12, 18:21] = -r_a
        tmp[9:12, 21:24] = eye3

        A[12*i:12*(i+1)] = tmp
        B[12*i+9:12*(i+1)] = np.expand_dims(t_a, axis=2)

    # solve Ax=B linear system equation
    x = np.linalg.lstsq(A, B)

    # compute X
    new_x = get_transformation(x[0][0:9], x[0][18:21])

    # compute Y
    new_y = get_transformation(x[0][9:18], x[0][21:24])

    return [new_x, new_y]


def handeye1_liang(aa, bb):
    """
    Implementation of Liang's method for solving AX=XB
    See also:
    http://faculty.cooper.edu/mili/Calibration/index.html
    http://www.jzus.zju.edu.cn/oldversion/opentxt.php?doi=10.1631/jzus.A0820318
    """
    siz = len(aa)

    A = np.zeros((siz*9, 9))  # noqa N806

    for i in range(siz):
        a = np.asarray(aa[i])
        b = np.asarray(bb[i])

        [r_a, t_a] = math_util.rt_from_mat(a)
        [r_b, t_b] = math_util.rt_from_mat(b)

        tmp1 = np.kron(r_a, np.eye(3))
        tmp2 = np.kron(-np.eye(3), np.transpose(r_b))

        A[9*i:9*(i+1), :] = tmp1 + tmp2

    [u, s, vh] = np.linalg.svd(A)

    x = vh[8, 0:9]

    r = get_rotation(x, transpose=False)

    r = handle_func(r)

    c = np.zeros((3*siz, 3))
    d = np.zeros((3*siz, 1))
    iden = np.eye(3)

    for i in range(siz):
        a = np.asarray(aa[i])
        b = np.asarray(bb[i])

        [r_a, t_a] = math_util.rt_from_mat(a)
        [r_b, t_b] = math_util.rt_from_mat(b)

        tmp1 = iden - r_a
        tmp2 = t_a - np.matmul(r, t_b)
        tmp2 = np.expand_dims(tmp2, 1)

        c[3*i:3*(i+1), :] = tmp1
        d[3*i:3*(i+1), :] = tmp2

    # solve Ct=d linear system equation
    t = np.linalg.lstsq(c, d)

    # Put everything together to form X
    X = np.zeros((4, 4))  # noqa N806
    X[0:3, 0:3] = r
    X[0:3, 3] = np.squeeze(t[0][0:3])
    X[3, 3] = 1.0

    return X