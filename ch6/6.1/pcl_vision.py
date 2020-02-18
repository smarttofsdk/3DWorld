import cv2
import numpy as np
import vispy.scene
from vispy.scene import visuals
from scipy.ndimage import distance_transform_edt


def fill_img(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid')
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'. True cells set where data
                 value should be replaced.
                 If None (default), use: invalid  = np.isnan(data)

    Output:
        Return a filled array.
    """
    # import numpy as np
    # import scipy.ndimage as nd

    if invalid is None: invalid = data >= 65000

    ind = distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return data[tuple(ind)]


def point_cloud(depth):
    """Transform a depth image into a point cloud with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.

    depth is a 2-D ndarray with shape (rows, cols) containing
    depths from 1 to 254 inclusive. The result is a 3-D array with
    shape (rows, cols, 3). Pixels with invalid depth in the input have
    NaN for the z-coordinate in the result.

    """
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)

    z = depth
    # k1, k2, k3, p1, p2 = [-0.3224, 0.3324, -0.0024, 0.0015, 0]
    k1, k2, k3, p1, p2 = [-0.3662, 0.3662, -0.0030, 0.0021, 0]

    # x = z * (c - 157.35) / 333.923
    # y = z * (r - 123.41) / 339.364
    x = z * (c - 329.128) / 787.065
    y = z * (r - 243.278) / 787.515
    # x = z * (c - 165.75) / 195.37
    # y = z * (r - 128.65) / 198.59
    return np.dstack((x, y, z))


canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()

# create scatter object and fill in the data
scatter = visuals.Markers()
view.add(scatter)
scatter.set_data(np.array([[0, 0, 0]]), edge_color=None, face_color=(1, 1, 1, .5), size=5)
view.camera = 'turntable'  # or try 'arcball'
# add a colored 3D axis for orientation
axis = visuals.XYZAxis(parent=view.scene)

# load DATA
# data = np.fromfile(r"./201812_dep.bin", dtype=np.uint16)
# data = data.reshape(data.shape[0] // 240 // 320, 240, 320)
data = cv2.imread("20.png", -1)
imgs = data.copy()

# distort = np.array([-0.322419765862276, 0.332381170631306, -0.002392462237341, 0.001531739658127, 0.000000000000000])
distort = np.array([-0.366161, 0.366161, -0.00303016, 0.00215416, 0.000000000000000])
# matrix = np.array([[333.9237341, 0, 157.35706351], [0, 339.364650254459150, 123.416251713789165], [0, 0, 1]])
matrix = np.array([[787.065, 0, 329.128], [0, 787.515, 243.278], [0, 0, 1]])
for idx, img in enumerate(imgs):
    img = fill_img(img)
    img = cv2.undistort(img, matrix, distort)
    for _ in range(2):
        img = cv2.medianBlur(img, 3)
        img = cv2.medianBlur(img, 5)
        img = cv2.medianBlur(img, 7)
    img = np.where(img > 5000, 0, img)
    cv2.imshow('img', cv2.convertScaleAbs(img, None, 1 / 20))
    cv2.waitKey(0)
    pc = point_cloud(img).reshape((-1, 3))
    pc = pc[~np.all(pc == 0, axis=1)]
    scatter.set_data(pc, edge_color=None, face_color=(1, 0, 0, .5), size=10)
    canvas.on_draw(pc)