from pyba.plot import draw_reference_frame
from typing import *
from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pyba.util import P_from_RtvecK


class Camera:
    def __init__(
        self,
        points2d: np.ndarray,
        cam_id: Optional[int] = None,
        intr: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        tvec: Optional[np.ndarray] = None,
        distort: Optional[np.ndarray] = None,
        image_path: Optional[dict] = None,
        heatmaps: Optional[np.ndarray] = None,
    ):
        """
        cam_id: optional integer index identifying this camera within a
            multi-camera rig. Used by downstream consumers (e.g. multi-view
            pose-correction algorithms) that need to address cameras by index.
            `CameraNetwork` sets it automatically; only consumers that index
            cameras by id (e.g. belief-propagation pose correction) require it.
            Default: None.
        fx, fy: focal length in pixels
        tvec: translation vector
        cx, cy: optical axis in pixels
        points2d: numpy array in pixels, TxJx2
        distort: list with 5 numbers
        image_path: Either a per-frame jpg/png template like
            'img_{img_id}.jpg', or a path to a single video file
            ('camera_0.mp4', '.avi') whose frames are read by index via
            cv2.VideoCapture.
        heatmaps: optional pre-computed per-joint probability maps for this
            camera. pyba treats the array as opaque (any shape) and never
            reads it; it is stored so that consumers like pictorial-structures
            pose correction can attach the network's raw heatmaps to their
            corresponding camera object. May also be set after construction
            via `cam.heatmaps = ...`. Default: None.
        """

        # fmt: off
        assert cam_id is None or isinstance(cam_id, (int, np.integer))
        assert points2d.ndim == 3 and points2d.shape[2] == 2
        assert R is None or R.ndim == 2 and R.shape[0] == 3 and R.shape[1] == 3
        assert tvec is None or tvec.ndim == 1 and tvec.shape[0] == 3
        assert distort is None or (distort.ndim == 1 and distort.shape[0] == 5)
        assert intr is None or intr.ndim == 2 and intr.shape[0] == 3 and intr.shape[1] == 3
        # fmt: on

        self.cam_id = None if cam_id is None else int(cam_id)
        self.image_path = image_path
        self._video_cap = None
        self._video_pos = 0
        if image_path is not None and image_path.lower().endswith(('.mp4', '.avi')):
            cap = cv2.VideoCapture(image_path)
            if cap.isOpened():
                self._video_cap = cap
        self._points2d = points2d

        self.intrinsic = intr
        self.tvec = tvec
        self.R = R
        self.distort = np.zeros(5, dtype=float) if distort is None else distort
        self.heatmaps = heatmaps

    @property
    def P(self):
        return P_from_RtvecK(self.R, self.tvec, self.intrinsic)

    @property
    def rvec(self):
        return cv2.Rodrigues(self.R)[0]

    def camera2world(self, pts: np.ndarray):
        return self.R.T @ (pts - self.tvec)

    @property
    def C(self):
        return -1 * self.R.T @ self.tvec

    @rvec.setter
    def rvec(self, rvec):
        self.R = cv2.Rodrigues(rvec)[0]

    @property
    def fx(self):
        return self.intrinsic[0, 0]

    @property
    def fy(self):
        return self.intrinsic[1, 1]

    @property
    def cx(self):
        return self.intrinsic[0, 2]

    @property
    def cy(self):
        return self.intrinsic[1, 2]

    @fx.setter
    def fx(self, fx):
        self.intrinsic[0, 0] = fx

    @fy.setter
    def fy(self, fy):
        self.intrinsic[1, 1] = fy

    @cx.setter
    def cx(self, cx):
        self.intrinsic[0, 2] = cx

    @cy.setter
    def cy(self, cy):
        self.intrinsic[1, 2] = cy

    @property
    def points2d(self):
        return self._points2d

    @points2d.setter
    def points2d(self, pts2d):
        self._points2d = pts2d

    def __getitem__(self, idx: int):
        return self.points2d[idx]

    def set_intrinsic(self, intrinsic):
        self.intrinsic = intrinsic

    def can_see(self, img_id, jid):
        return not np.any(np.isclose(self.points2d[img_id, jid], 0))

    def can_see_mask(self):
        return np.any(self.points2d == 0, axis=2)

    def get_njoints(self):
        return self.points2d.shape[1]

    def get_image(self, img_id: int):
        img = None
        if self._video_cap is not None:
            if img_id != self._video_pos:
                self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, img_id)
                self._video_pos = img_id
            ok, frame = self._video_cap.read()
            if ok:
                self._video_pos += 1
                img = frame
        else:
            try:
                img = cv2.imread(self.image_path.format(img_id=img_id))
            except:
                pass
        if img is None:
            img = np.zeros((480, 960), dtype=np.uint8)
        if img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    def has_calibration(self):
        return all(
            [self.intrinsic is not None, self.R is not None, self.tvec is not None]
        )

    def summarize(self):
        return {
            "R": self.R,
            "tvec": self.tvec,
            "distort": self.distort,
            "intr": self.intrinsic,
        }

    def plot_2d(self,
                img_id: int,
                points2d: Optional[np.ndarray] = None,
                bones: Optional[np.ndarray] = None,
                colors: Optional[List[Tuple]] = None) -> np.ndarray:
        """
        Plot 2D points (and optional bones) on the camera image for `img_id`.

        Parameters
        ----------
        img_id : int
            Index of the image/frame to draw on.
        points2d : np.ndarray, optional
            Array of shape (n_joints, 2) giving the 2D points to plot. If
            None, the camera's stored 2D points for this frame are used. To
            plot reprojected 3D points instead, use `plot_reprojections`.
        bones : np.ndarray, optional
            Array of joint-index pairs defining bones to draw.
        colors : list of tuples, optional
            Per-bone RGB colors.

        Returns
        -------
        np.ndarray
            The image with points and bones drawn on it.
        """
        img = self.get_image(img_id)

        if points2d is None:
            points2d = self.points2d[img_id]

        if bones is not None:
            for idx, b in enumerate(bones):
                if self.can_see(img_id, b[0]) and self.can_see(img_id, b[1]):
                    img = cv2.line(
                        img,
                        tuple(points2d[b[0]].astype(int)),
                        tuple(points2d[b[1]].astype(int)),
                        colors[idx] if colors is not None else (128, 0, 0),
                        5,
                    )

        for jid in range(self.get_njoints()):
            if self.can_see(img_id, jid):
                img = cv2.circle(
                    img, tuple(points2d[jid].T.astype(int)), 5, [0, 0, 128], 5
                )

        return img

    def plot_reprojections(self,
                           img_id: int,
                           points3d: np.ndarray,
                           bones: Optional[np.ndarray] = None,
                           colors: Optional[List[Tuple]] = None) -> np.ndarray:
        """
        Project a set of 3D points into this camera and plot them on the
        image for `img_id`.

        Parameters
        ----------
        img_id : int
            Index of the image/frame to draw on.
        points3d : np.ndarray
            Either a (n_joints, 3) array of 3D points for this frame, or a
            (T, n_joints, 3) array from which the row at `img_id` will be
            used.
        bones : np.ndarray, optional
            Array of joint-index pairs defining bones to draw.
        colors : list of tuples, optional
            Per-bone RGB colors.

        Returns
        -------
        np.ndarray
            The image with reprojected points and bones drawn on it.
        """
        if points3d.ndim == 3:
            points3d = points3d[img_id]
        elif points3d.ndim != 2:
            raise ValueError(f'Expected points3d to have shape (n_joints, 3) or '
                             f'(T, n_joints, 3), but got shape {points3d.shape}')
        # `project` expects a batch dimension (T, n_joints, 3)
        points2d = self.project(points3d[np.newaxis, ...]).squeeze(axis=0)
        return self.plot_2d(img_id, points2d=points2d, bones=bones, colors=colors)

    def project(self, points3d: np.ndarray) -> np.ndarray:
        """
        points3d: TxJx3
        returns: points2d: TxJx2
        """

        assert points3d.ndim == 3 and points3d.shape[2] == 3

        # original shape
        t, j = points3d.shape[0], points3d.shape[1]

        # opencv wants nx3 matrix
        # https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga1019495a2c8d1743ed5cc23fa0daff8c
        points3d = points3d.reshape(t * j, 3)  # (txj)x3
        points2d, _ = cv2.projectPoints(
            points3d, self.rvec, self.tvec, self.intrinsic, self.distort
        )

        # reshape back to txjx2
        points2d = points2d.reshape(t, j, 2)  # txjx2
        return points2d

    def reprojection_error(self, points3d: np.ndarray):
        err = self.points2d - self.project(points3d)
        err[self.can_see_mask(), :] = 0
        return err

    def draw(self, ax3d, size: float = 1, text: Optional[str] = None):
        x = self.camera2world(np.array([size, 0, 0]))
        y = self.camera2world(np.array([0, size, 0]))
        z = self.camera2world(np.array([0, 0, size]))

        draw_reference_frame(ax3d, center=self.C, x=x, y=y, z=z, text=text)
