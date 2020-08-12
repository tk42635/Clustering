import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import scipy.interpolate

import sys
from descartes import PolygonPatch
import alphashape
import cv2
import math 



# data import
data_raw = np.fromfile('./data2.dat')
data = data_raw.reshape((1680, 2))


# plot data
#plt.plot(data[:, 0], data[:, 1], 'b.')


# extract the boundary points
# write your code
#####################################      Part 1      ###########################################

#--------------------------     Method I: Delaunay Triangle    ---------------------------
def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list yet
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Repetive count!"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.simplices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges

result = alpha_shape(np.array(data), 5.9, True)
delaunay_x, delaunay_y =[], []
for pair in result:
    delaunay_x.append(data[pair[0]][0])
    delaunay_y.append(data[pair[0]][1])
#--------------------------------------------------------------------------------------------------

#---------------------     Method II: Alphashape method from library    ----------------------
#al = alphashape.optimizealpha(data)
al_shape2 = alphashape.alphashape(data, 0.1285)
hull_pts2 = al_shape2.exterior.coords.xy
x = hull_pts2[0]
y = hull_pts2[1]
#--------------------------------------------------------------------------------------------------


def fine_tune(x, y, beta):
    x_new, y_new = x, y
    for i in range(len(x) - 2, -1, -1):
        vec1 = np.array([(x[i + 1] - x[i]), (y[i + 1] - y[i])])
        vec2 = np.array([(x[i - 1] - x[i]), (y[i - 1] - y[i])])
        cos = vec1.dot(vec2) / (np.sqrt(vec1.dot(vec1)) * np.sqrt(vec2.dot(vec2)))
        angle = np.arccos(cos)
        if (angle < beta):
            x_new = np.delete(x_new, i)
            y_new = np.delete(y_new, i)
    return np.array(x_new), np.array(y_new)


x, y = fine_tune(x, y, 0.8)



# plot the results for boundary points
# write your code
plt.scatter(*zip(*data), s=6)
plt.scatter(delaunay_x, delaunay_y, s=10, color='red')  # Method I
plt.axis('equal')
plt.grid(True)



# extract the boundary lines
# write your code
#####################################      Part 2      ###########################################

#---------   Method I: Interpolate over boundary points to get a new polygon   ----------
def interpolate(x, y):
    num = len(x)
    nt = np.linspace(0, 1, num)
    t = np.zeros(x.shape)
    t[1:] = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
    t = np.cumsum(t)
    t /= t[-1]

    tck_x = scipy.interpolate.splrep(t, x)
    tck_y = scipy.interpolate.splrep(t, y)
    interpolate_x = scipy.interpolate.splev(nt, tck_x)
    interpolate_y = scipy.interpolate.splev(nt, tck_y)

    return interpolate_x, interpolate_y

interpolate_x, interpolate_y = interpolate(x, y)
#-----------------------------------------------------------------------------------------

#-------------------------   Method II: OpenCV approxPolyDP()   --------------------------
def approxPolyDP(x, y, close, approx_rate):
    curve=[]
    for i in range(len(x)):
        curve.insert(0, [x[i], y[i]])

    curve=np.array(curve)
    curve = np.reshape(curve, (curve.shape[0], 1, 2))
    curve = curve.astype(np.int32)

    approx = cv2.approxPolyDP(curve, approx_rate, True)

    poly1_x, poly1_y = [], []
    for each in approx:
        poly1_x.append(each[0][0])
        poly1_y.append(each[0][1])
    if close == True:
        poly1_x.append(approx[0][0][0])
        poly1_y.append(approx[0][0][1])

    return poly1_x, poly1_y

poly1_x, poly1_y = approxPolyDP(x, y, True, 4)
#-----------------------------------------------------------------------------------------



#####################################      Part 3      ###########################################

#--------------------------------   Get inner side   -------------------------------------
def cocave_hull():
    """
    Using alphashape() method from library to get both convex-hull points(alpha ≈ 0) and alpha shape points(high alpha value).
    Deleting those points in alpha shape that are within the certain radius of each point in convex-hull.
    Uning OpenCV approxPolyDP() to generate a fitted polygon.
    """
    al_shape = alphashape.alphashape(data, 0.02)
    hull_pts = al_shape.exterior.coords.xy
    x = hull_pts[0]
    y = hull_pts[1]
    x, y = fine_tune(x, y, 0.8)

    al_shape2 = alphashape.alphashape(data, 0.1285)
    hull_pts2 = al_shape2.exterior.coords.xy
    x5 = hull_pts2[0]
    y5 = hull_pts2[1]
    x5, y5 = fine_tune(x5, y5, 0.8)

    for i in range(len(x)):
        for j in range(len(x5) - 1, -1, -1):
            dist = np.sqrt((x[i] - x5[j])**2 + (y[i] - y5[j])**2)
            if (dist < 15):
                x5 = np.delete(x5, j)
                y5 = np.delete(y5, j)
    return x5, y5

inner_x, inner_y = cocave_hull()
inner_x, inner_y = approxPolyDP(inner_x, inner_y, False, 4)
idx_del = []

for i in range(len(inner_x) -2):
    vec1 = np.array([(inner_x[i + 1] - inner_x[i]), (inner_y[i + 1] - inner_y[i]), 0])
    vec2 = np.array([(inner_x[i + 2] - inner_x[i + 1]), (inner_y[i + 2] - inner_y[i + 1]), 0])
    cita = math.atan2(np.cross(vec1, vec2).dot([0, 0, -1]), vec1.dot(vec2))
    if (cita < 0):
        idx_del.append(i+2)
for i in idx_del[::-1]:
    inner_x = np.delete(inner_x, i)
    inner_y = np.delete(inner_y, i)
#--------------------------------------------------------------------------------------------------



# plot the results for boundary lines
# write your code


fig, outline1 = plt.subplots(sharex=True, sharey=True)
plt.axis('equal')
plt.grid(True)
fig, outline2 = plt.subplots(sharex=True, sharey=True)
plt.axis('equal')
plt.grid(True)
fig, inner_part = plt.subplots(sharex=True, sharey=True)
plt.axis('equal')
plt.grid(True)
outline1.scatter(*zip(*data), s=6)
outline2.scatter(*zip(*data), s=6)
inner_part.scatter(*zip(*data), s=6)
inner_part.plot(inner_x, inner_y, 'r--', linewidth=2, color='red')
outline1.scatter(delaunay_x, delaunay_y, s=10, color='red')
outline2.scatter(delaunay_x, delaunay_y, s=10, color='red')
outline2.plot(interpolate_x, interpolate_y,'r--',linewidth=2, color = 'green', label="Method I: interpolate")
outline1.plot(poly1_x, poly1_y, 'r--', linewidth=2, color='black', label="Method II: OpenCV approxPolyDP")

outline1.legend(loc='upper right')
outline2.legend(loc='upper right')
plt.show()
