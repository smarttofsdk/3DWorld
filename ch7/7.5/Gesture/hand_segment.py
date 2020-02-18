import glob
import time

import cv2
import numpy as np
from shapely.geometry import *
from polygon import make_valid_polygon

fno = 0
img_show = None
img_dump = None


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def get_extend_line(p1: Point, p2: Point, ratio=10):
    """Creates a line extrapoled in p1->p2 direction"""
    a = p1
    b = Point(p1.x + ratio * (p2.x - p1.x), p1.y + ratio * (p2.y - p1.y))

    return LineString([a, b])


def fix_invalid_polygon(polygon: Polygon):
    if not polygon.is_valid:
        print("contour not valid, fix it")
        fixed_p = make_valid_polygon(polygon)
        # fixed_p = polygon.buffer(0)

        most_elem = 0
        most_elem_polygon = None
        if is_iterable(fixed_p):
            for p in fixed_p:
                if len(p.boundary.coords) > most_elem:
                    most_elem = len(p.boundary.coords)
                    most_elem_polygon = p
            return most_elem_polygon
        return fixed_p
    return polygon


def find_init_search_point(contour: np.ndarray, rect_size: tuple, rect_origin=(0, 0), bound_margin=3):
    """
    given the contour polygon and the boundary rectangle, find the proper point to start area mean shift 
    :rtype: tuple
    :param contour: specified contour polygon in numpy array shape=(x,1,2)
    :param rect_size: tuple(w,h)
    :param rect_origin: tuple(x,y)
    :param bound_margin: margin pixel 
    :return search point in tuple(center, radius, bound_near_status), if failed, return None,0, None
    """
    global img_dump

    enter_ts = time.time()

    # -- find poly_bound
    _contour = contour.reshape(-1, 2)
    cur_polygon = fix_invalid_polygon(Polygon(_contour))

    if len(cur_polygon.bounds) != 4:
        return None, 0, None

    tl_x, tl_y, br_x, br_y = cur_polygon.bounds
    b_tl_x, b_tl_y, b_br_x, b_br_y = (rect_origin[0], rect_origin[1], rect_size[0], rect_size[1])

    # -- check poly_bound with boundary rectangle
    dist_trbl = np.array((tl_y - b_tl_y, b_br_x - br_x, b_br_y - br_y, tl_x - b_tl_x))
    near_bound_trbl = dist_trbl < bound_margin  # (T, R, B, L)

    total_edge_near_bound = near_bound_trbl.sum()
    if total_edge_near_bound >= 2:
        print("too many edge near bound: %d" % sum(near_bound_trbl))
        return None, 0, None
    # elif total_edge_near_bound == 2:
    #     if near_bound_trbl[0] ^ near_bound_trbl[2] == 0:
    #         print("two parallel edge intersect!")
    #         return None, 0, None
    elif total_edge_near_bound == 0:
        # return the centroid if within the polygon or the polygon point nearest the centroid
        c = cur_polygon.centroid
        r = 10
        if not c.within(cur_polygon):
            # find nearest point on the edge of polygon
            _c = cur_polygon.exterior.interpolate(cur_polygon.exterior.project(c))
            r = c.distance(_c) + 10
            c = _c
        return (c.x, c.y), r, near_bound_trbl

    # compute the corner with shortest distance
    corner_valid = near_bound_trbl + np.roll(near_bound_trbl, 1)  # (TL, TR, BR, BL), only 0 is valid
    corner_points = MultiPoint([(tl_x, tl_y), (br_x, tl_y), (br_x, br_y), (tl_x, br_y)])

    min_corner = None
    min_dist = 99999999

    for i in range(4):
        if corner_valid[i] == 0:
            dist = corner_points[i].distance(cur_polygon)
            if min_dist > dist:
                min_dist = dist
                min_corner = corner_points[i]

            # if 2 bound near, check opposite corner
            if total_edge_near_bound == 2:
                oppsite_corner = corner_points[(i + 2) % 4]
                if oppsite_corner.distance(cur_polygon) > bound_margin:
                    print("two adjacent edge near bound: corner_dist=%.2f" % dist)
                    return None, 0, None

    if min_corner is None:
        return None, 0, None
    # the nearest point on the contour
    c = cur_polygon.exterior.interpolate(cur_polygon.exterior.project(min_corner))

    # check centroid and min_corner center
    c = LineString([c, cur_polygon.centroid]).centroid
    if not c.within(cur_polygon):
        # find nearest point on the edge of polygon
        c = cur_polygon.exterior.interpolate(cur_polygon.exterior.project(c))

    r = c.distance(cur_polygon.centroid)
    if r < 15:
        r = 15

    # print("find_init_search_point spends: %.2f s" % (time.time() - enter_ts))

    return (c.x, c.y), r, near_bound_trbl


def circle_area_mean_shift(contour: np.ndarray, init_p, init_r,
                           min_converge: float, area_ratio_threshold: float,
                           radius_step=1.0, max_loop=100, img_dump=None) -> tuple:
    """
    given the contour polygon and initial circle, do circle area mean shift
    until converge and area effective ratio is met
    :param contour: contour polygon in numpy array shape=(x,1,2)
    :param init_p: center of initial circle 
    :param init_r: center of initial radius 
    :param min_converge: minimum converge step in pixel
    :param area_ratio_threshold: effective area ratio threshold
    :param radius_step: radius increment step 
    :param max_loop: max loop count to mean shift
    :return: final circle denoted as tuple(center, radius), if error, return (None, 0)
    """
    enter_ts = time.time()
    cur_center = Point(init_p)
    cur_radius = init_r

    _contour = contour.reshape(-1, 2)

    cur_polygon = fix_invalid_polygon(Polygon(_contour))
    cur_bound = np.array(cur_polygon.bounds, np.int).reshape(2, -1)

    loop_cnt = 0
    last_inc = None

    while loop_cnt < max_loop:
        loop_cnt += 1
        # compute intersect poly
        cur_circle = cur_center.buffer(cur_radius)

        intersect_poly = cur_circle.intersection(cur_polygon)

        next_center = intersect_poly.centroid

        if next_center.distance(cur_center) >= min_converge:
            # not converge -> move to next center
            if img_dump is not None:
                cv2.arrowedLine(img_dump,
                                (int(cur_center.x + 0.5), int(cur_center.y + .5)),
                                (int(next_center.x + 0.5), int(next_center.y + .5)),
                                (0, 0, 255), 1)

            cur_center = next_center
            tmp_dist = cur_center.distance(cur_polygon)
            if cur_radius < tmp_dist:
                cur_radius = tmp_dist
                # print(cur_center)
        else:
            # converge condition pass-> check area ratio condition
            if intersect_poly.area > area_ratio_threshold * 1.1 * cur_circle.area:
                if last_inc is not None and not last_inc:
                    # avoid shake
                    break
                # cur_radius += radius_step
                next_radius = np.math.sqrt(
                    (intersect_poly.area - area_ratio_threshold * cur_circle.area) / 3.14 + cur_radius * cur_radius)
                cur_radius = next_radius if next_radius - cur_radius > radius_step else cur_radius + radius_step

                last_inc = True
            elif intersect_poly.area < area_ratio_threshold * 0.90 * cur_circle.area:
                if cur_radius < radius_step:
                    # avoid shake
                    break
                if last_inc is not None and last_inc:
                    break
                # cur_radius -= radius_step
                next_radius = np.math.sqrt(
                    cur_radius * cur_radius - (area_ratio_threshold * cur_circle.area - intersect_poly.area) / 3.14)
                cur_radius = next_radius if cur_radius - next_radius > radius_step else cur_radius - radius_step

                last_inc = False
            else:
                # effective area condition pass
                # print("ratio=%.2f ,dist=%.2f" % (
                #     intersect_poly.area / cur_circle.area, next_center.distance(cur_center)))
                break

                # print(cur_radius)

    # print("circle mean shift spends: %.2f s, loop=%d" % (time.time() - enter_ts, loop_cnt))
    return (int(cur_center.x + 0.5), int(cur_center.y + 0.5)), cur_radius


def update_contour(contour: np.ndarray, palm_circle_center: tuple, palm_circle_radius, rect_size: tuple,
                   init_point: tuple):
    """
    update specified contour with specified parm circle and boundary near status
    :param contour: specified contour
    :param palm_circle_center: 
    :param palm_circle_radius: 
    :param rect_size: tuple(w,h)
    """
    enter_ts = time.time()
    _contour = contour.reshape(-1, 2)
    polygon = fix_invalid_polygon(Polygon(_contour))
    palm_circle = Point(palm_circle_center).buffer(palm_circle_radius).simplify(1, False)

    bound_margin = 3

    img_bound = LinearRing([(bound_margin, bound_margin),
                            (rect_size[0] - bound_margin, bound_margin),
                            (rect_size[0] - bound_margin, rect_size[1] - bound_margin),
                            (bound_margin, rect_size[1] - bound_margin)])
    check_line = get_extend_line(Point(palm_circle_center), polygon.centroid, ratio=9999)

    regions = polygon.difference(palm_circle)
    intersect_poly = polygon.intersection(palm_circle)
    if intersect_poly.is_empty:
        print("palm_circle not intersect with polygon!\n")
        return None

    rem_poly = intersect_poly
    if not is_iterable(regions):
        regions = [regions]

    # only remove one polygon
    remove_flag = False
    for p in regions:
        if remove_flag:
            rem_poly = rem_poly.union(p)
            continue

        p_intersect_len = p.intersection(img_bound).length
        print("intersect_len: %.2f, radius=%.2f" % (p_intersect_len, palm_circle_radius))
        if p_intersect_len > palm_circle_radius / 2:
            # polygon = polygon.difference(p)
            remove_flag = True
        else:
            rem_poly = rem_poly.union(p)

    if not remove_flag and polygon.intersection(img_bound).length <= palm_circle_radius / 2:
        rem_poly = intersect_poly

        for p in regions:
            if palm_circle_center[0] == p.centroid.x and palm_circle_center[1] == p.centroid.y:
                break

            check_line = get_extend_line(Point(palm_circle_center), p.centroid, ratio=9999)
            cross_point = check_line.intersection(img_bound)
            # cross point with the bound
            cross_line = check_line.intersection(p)

            # print("ratio: %.2f" % (cross_line.length / (Point(palm_circle_center).distance(cross_point) + 0.00001)))
            if cross_line.length > 0.4 * Point(palm_circle_center).distance(cross_point):
                # polygon = polygon.difference(p)
                pass
            else:
                rem_poly = rem_poly.union(p)

    polygon = rem_poly
    if is_iterable(polygon):
        most_elem = 0
        most_elem_polygon = None
        for p in polygon:
            try:
                if len(p.boundary.coords) > most_elem:
                    most_elem = len(p.boundary.coords)
                    most_elem_polygon = p
            except NotImplementedError:
                print('NotImplementedError: Multi-part geometries do not provide a coordinate sequence')

        polygon = most_elem_polygon

    if polygon is None:
        return None

    return np.array(polygon.exterior, np.int).reshape(-1, 1, 2)


def pre_filter(img):
    """
     if (img_dist.type() == CV_16UC1) {
            img_dist.convertTo(img_f, CV_32FC1);
            /* fitering data within 1m */
            cv::threshold(img_f, img_f, 10000, 255, cv::THRESH_BINARY);
            img_f.convertTo(img_f, CV_8UC1);
        } else {
            img_f = img_dist;
            cv::threshold(img_f, img_f, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        }
    :param img: 
    :return: 
    """
    if img.dtype == np.uint16:
        img_f = img.astype(np.float32)
        ret, img_f = cv2.threshold(img_f, 8000, 255, cv2.THRESH_BINARY)
        img_src = img_f.astype(np.uint8)
    else:
        # img_src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ret, img_src = cv2.threshold(img_src, 35, 255, cv2.THRESH_BINARY)
        # Otsu's thresholding
        ret, img_src = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ----------- filer section --------------
    # img_src = cv2.adaptiveThreshold(img_src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)
    img_src_f = img_src
    # img_src_f = cv2.bitwise_not(img_src_f)
    # img_src_f = cv2.GaussianBlur(img_src_f, (3, 3), 0.2)
    # cv2.imshow("img_src",img_src_f)
    img_src_f = cv2.erode(img_src_f, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    img_src_f = cv2.dilate(img_src_f, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    img_src_f = cv2.dilate(img_src_f, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    img_src_f = cv2.erode(img_src_f, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

    # cv2.imshow("p0", img_src_f)
    # cv2.waitKey(0)

    # find max contour and fill hole
    i, contour, h = cv2.findContours(img_src_f, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    max_area = 0
    max_contour = None
    for c in contour:
        cur_area = cv2.contourArea(c)
        if max_area < cur_area:
            max_area = cur_area
            max_contour = c

    print(max_area)
    if max_area < 500:
        return img_src_f, None

    for c in contour:
        if not np.array_equal(c, max_contour):
            # use contours to fill hole
            cv2.drawContours(img_src_f, [c], 0, 255, -1)
    if max_contour is not None and len(max_contour) > 0:
        max_contour = cv2.approxPolyDP(max_contour, 1.5, True)
        print(" hand contour len=%d" % len(max_contour))
    # cv2.imshow("contour", img_src_f)
    return img_src_f, max_contour


def test():
    global fno, img_dump
    for file in glob.glob("raw_dump/img_dist_x10_c0_201705151041391.png"):
        fno += 1
        print("\n---> proc " + file)
        img = cv2.imread(file, -1)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.imread(file)
        # remove outline
        # img = img[3:, 3:]

        # pre filter the image
        img_src_f, max_contour = pre_filter(img)

        img_dump = cv2.cvtColor(img_src_f, cv2.COLOR_GRAY2BGR)

        img_wh = (img_src_f.shape[1], img_src_f.shape[0])
        # --- calc palm center
        init_point, init_r, bstatus = find_init_search_point(max_contour, img_wh)

        if init_point is None:
            print(" invalid pos!")
            continue
        # init_point = centroid
        # init_r = 2.0
        # print(init_point, init_r)
        c, r = circle_area_mean_shift(max_contour, init_point, init_r=init_r,
                                      min_converge=1.0, area_ratio_threshold=0.6, radius_step=2.0, img_dump=img_dump)
        print(c, r)

        # calc updated contour
        new_contour = update_contour(max_contour, c, r, img_wh, init_point)

        # --- draw contour
        contour_moment = cv2.moments(max_contour)
        centroid = (int(contour_moment["m10"] / contour_moment["m00"]),
                    int(contour_moment["m01"] / contour_moment["m00"]))
        cv2.circle(img_dump, c, int(r), (255, 128, 128), 1)  # parm circle
        cv2.drawContours(img_dump, [max_contour], 0, (255, 128, 0), 1, 4)
        cv2.circle(img_dump, centroid, 1, (128, 128, 0), 1)

        # --- draw convexHull
        hull = cv2.convexHull(max_contour, False)
        if max_contour is not None and len(max_contour) > 0:
            cv2.drawContours(img_dump, [hull], 0, (128, 255, 0), 2, 4)

        if new_contour is not None:
            cv2.drawContours(img_dump, [new_contour], 0, (50, 50, 255), 2)

        # ----------- visualize section --------------
        # img_dst = img_src_f
        # img_src = cv2.resize(img_src, (0, 0), fx=2, fy=2)
        # img_dst = cv2.resize(img_dst, (0, 0), fx=2, fy=2)
        img_show = np.concatenate((cv2.cvtColor(img_src_f, cv2.COLOR_GRAY2BGR), img_dump), axis=1)
        img_show = cv2.resize(img_show, (0, 0), fx=2, fy=2)
        cv2.imshow("i", img_show)
        cv2.waitKey()
        # break


if __name__ == "__main__":
    test()
