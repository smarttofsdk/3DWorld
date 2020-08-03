# Rohan Prinja
# COMP 776, Fall 2017
# Assignment: RANSAC

import numpy as np
import sys

from util import plot_epipolar_inliers

# for purposes of experimentation, we'll keep the sampling fixed every time the
# program is run
np.random.seed(0)


# -------------------------------------------------------------------------------

# Run RANSAC on a dataset for a given model type
#
# @param data                 M x K numpy array containing all M observations in
#                             the dataset, each with K dimensions
# @param inlier_threshold     given some error function for the model (e.g.,
#                             point-to-plane distance if the model is a 3D
#                             plane), label input data as inliers if their error
#                             is smaller than this threshold
# @param confidence_threshold our chosen value p that determines the minimum
#                             confidence required to stop RANSAC
# @param max_num_trials       initial maximum number of RANSAC iterations N
#
# @return best_F          the associated best model for the inliers
# @return inlier_mask         length M numpy boolean array with True indicating
#                             that a data point is an inlier and False
#                             otherwise; inliers can be recovered by taking
#                             data[inlier_mask]
def ransac(data, inlier_threshold, confidence_threshold, max_num_trials):
    max_iter = max_num_trials  # current maximum number of trials
    iter_count = 0  # current number of iterations

    best_inlier_count = 0  # initial number of inliers is zero
    best_inlier_mask = np.zeros(  # initially mark all samples as outliers
        len(data), dtype=np.bool)
    best_F = np.zeros((3, 3))  # dummy initial model

    # sample size S: 8 points are sampled for a Hartley model
    S = 8

    def make_hartley_vector(p1, p2):
        u1, v1 = p1
        u2, v2 = p2
        return np.array([u1 * u2, u1 * v2, u1, v1 * u2, v1 * v2, v1, u2, v2, 1])

    def make_hartley_matrix(left, right):
        m = len(left)
        A = np.zeros((m, 9))
        for i in xrange(m):
            A[i, :] = make_hartley_vector(left[i], right[i])
        return A

    def compute_F(A):
        u, s, vT = np.linalg.svd(A)
        return np.reshape(vT[-1], [3, 3], order='F')

    def enforce_rank_2(F):
        u, s, vT = np.linalg.svd(F)
        s[-1] = 0
        return u * np.diag(s) * vT

    def compute_normalization_mat(pts):
        translate = np.mean(pts, axis=0)
        translated = pts - translate
        distances = np.linalg.norm(translated, axis=1)
        scale = np.sqrt(2) / np.mean(distances)
        result = np.array([[scale, 0, scale * -translate[0]],
                           [0, scale, scale * -translate[1]],
                           [0, 0, 1]])
        return result

    def normalize_points(pts):
        # mean should be (0,0)
        res = pts - np.mean(pts, axis=0)

        # average dist to origin should be sqrt(2)
        distances = np.linalg.norm(res, axis=1)
        mean_dist = np.mean(distances)
        scale = mean_dist / np.sqrt(2)
        res /= scale

        return res

    def apply_transform(T, pts):
        ones = np.ones(len(pts))
        homogenized = np.column_stack((pts, ones))
        return np.dot(T, homogenized.T)

    def hartley(points):
        '''Run the entire Hartley algorithm from start to end'''
        T1 = compute_normalization_mat(points[:, :2])
        T2 = compute_normalization_mat(points[:, 2:])

        # left_img_pts = apply_transform(T1, points[:,:2])[:2,:].T
        # right_img_pts = apply_transform(T1, points[:,2:])[:2,:].T

        left_img_pts = normalize_points(points[:, :2])
        right_img_pts = normalize_points(points[:, 2:])

        # print np.mean(np.linalg.norm(left_img_pts, axis=1))
        # print np.mean(np.linalg.norm(right_img_pts, axis=1))
        # these two should be roughly sqrt(2)

        A = make_hartley_matrix(left_img_pts, right_img_pts)
        F = compute_F(A)
        F = enforce_rank_2(F)
        F = np.dot(T2.T, np.dot(F, T1))

        return F

    # --------------------------------------------------------------------------
    #
    # TODO: fill in the steps of F-matrix RANSAC, below
    #
    # --------------------------------------------------------------------------


    # continue while the maximum number of iterations hasn't been reached
    while iter_count < max_iter:
        iter_count += 1

        # -----------------------------------------------------------------------
        # 1) sample as many points from the data as are needed to fit the
        #    relevant model

        idxs = np.random.choice(len(data), S, replace=False)
        points = data[idxs]

        # -----------------------------------------------------------------------
        # 2) fit a model to the sampled data subset

        F = hartley(points)

        # -----------------------------------------------------------------------
        # 3) determine the inliers to the model; store the result as a boolean
        #    mask, with inliers referenced by data[inlier_mask]

        # Find the epipolar lines
        left = np.column_stack((data[:, :2],
                                np.ones(len(data)))).T
        epipolar_lines = np.dot(F, left)

        # Ensure that nx^2 + ny^2 = 1 for each epipolar line
        norms = np.linalg.norm(epipolar_lines[:2, :], axis=0)
        epipolar_lines /= norms

        # For each right point, find the distance to the epipolar line
        # obtained from its corresponding left point
        right = np.column_stack((data[:, 2:],
                                 np.ones(len(data)))).T
        assert right.shape == epipolar_lines.shape
        distances = np.sum(right * epipolar_lines, axis=0)
        inlier_mask = (distances < inlier_threshold)
        inlier_count = np.count_nonzero(inlier_mask)

        # -----------------------------------------------------------------------
        # 4) if this model is the best one yet, update the report and the
        #    maximum iteration threshold

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inlier_ratio = inlier_count / float(len(data))
            best_inlier_mask = inlier_mask
            best_F = F

            # if a perfect model was found, we're done
            if inlier_count == len(data):
                break

            # based on this inlier ratio, re-compute the maximum number of
            # iterations required to achieve the desired probability of a good
            # solution (under the assumption that at least this percentage of
            # samples are inliers)
            #
            # refer to the class slides: slide set 8, slide 71

            # compute (1 - e)^S
            prob_S_samples_are_inliers = np.power(best_inlier_ratio, S)

            # there's a chance that the number of inliers is very small (close
            # to zero); to stably calculate the denominator when solving for N,
            # we'll use the logsumexp trick:
            # log(1 - p) = log(exp(0) - exp(log(p)))
            #            = log(exp(log(p)) * (exp(0)/exp(log(p)) - 1))
            #            = log(p) + log(1 / p - 1)
            denom = (np.log(prob_S_samples_are_inliers) +
                     np.log(1. / prob_S_samples_are_inliers - 1.))

            if not np.isclose(denom, 0.):
                # solve for the new maximum number of trials N:
                # (1 - (1 - e)^S)^N = 1 - p
                # N * log(1 - (1 - e)^S) = log(1 - p)
                N = np.log(1. - confidence_threshold) / denom
                print
                N

                # use min here because the computed maximum could be greater
                # than max_num_trials
                # max_iter = min(max_iter, N)
                max_iter = N

    # ---------------------------------------------------------------------------
    # 5) run a final fit on the F matrix using the inliers

    inlier_data = data[best_inlier_mask]
    F = hartley(inlier_data)

    # sys.exit(0)

    # ---------------------------------------------------------------------------
    # print some information about the results of RANSAC

    inlier_ratio = best_inlier_count / float(len(data))

    print
    "Iterations:", iter_count
    print
    "Inlier Ratio: {:.3f}".format(inlier_ratio)
    print
    "Best Fit Model:"
    print
    "  [ {:7.4f}  {:7.4f}  {:7.4f} ]".format(*best_F[0])
    print
    "  [ {:7.4f}  {:7.4f}  {:7.4f} ]".format(*best_F[1])
    print
    "  [ {:7.4f}  {:7.4f}  {:7.4f} ]".format(*best_F[2])

    return best_F, best_inlier_mask


# -------------------------------------------------------------------------------

# program main
# @param args command-line arguments
def main(args):
    keypoints1 = np.loadtxt(args.keypoints1, delimiter=",")
    keypoints2 = np.loadtxt(args.keypoints2, delimiter=",")

    assert keypoints1.shape[1] == 2, "Keypoints should be Mx2 arrays."
    assert keypoints2.shape[1] == 2, "Keypoints should be Mx2 arrays."
    assert keypoints1.shape[0] == keypoints2.shape[0], (
        "Mismatch in the number of keypoints. Both arrays should have M rows, "
        "with the i-th keypoint in the first image corresponding to the i-th "
        "keypoint in the second image.")

    # to keep the design of the other RANSAC implementation, form an Mx4 data
    # array, where each row is a correspondence between the two images; just
    # like before, we'll sample rows in the data matrix
    data = np.column_stack((keypoints1, keypoints2))

    F, inlier_mask = ransac(data, args.inlier_threshold,
                            args.confidence_threshold, args.max_num_trials)

    plot_epipolar_inliers(
        args.image1, args.image2, keypoints1, keypoints2, F, inlier_mask)


# -------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Given matching keypoints in two images, compute a "
                    "fundamental matrix relating the two images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #
    # Input options
    #

    parser.add_argument("--image1", type=str, default="data/left_img.jpg",
                        help="first image file")

    parser.add_argument("--image2", type=str, default="data/right_img.jpg",
                        help="second image file")

    parser.add_argument(
        "--keypoints1", type=str,
        default="data/left_img_pts.csv",
        help="keypoints CSV file for the first image, with each line storing "
             "(x,y) pixel coordinates")

    parser.add_argument(
        "--keypoints2", type=str,
        default="data/right_img_pts.csv",
        help="keypoints CSV file for the second image, with each line storing "
             "(x,y) pixel coordinates")

    #
    # RANSAC options
    #

    parser.add_argument("--inlier_threshold", type=float, default=2.,
                        help="point-to-line distance threshold, in pixels, to use for RANSAC")

    parser.add_argument("--confidence_threshold", type=float, default=0.99,
                        help="stop RANSAC when the probability that a correct model has been "
                             "found reaches this threshold")

    parser.add_argument("--max_num_trials", type=float, default=10000,
                        help="maximum number of RANSAC iterations to allow")

    args = parser.parse_args()

    main(args)