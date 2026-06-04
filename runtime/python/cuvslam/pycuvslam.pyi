from collections.abc import Callable, Mapping, Sequence
import enum
from typing import Annotated, overload

import numpy
from numpy.typing import NDArray


def get_version() -> tuple[str, int, int, int]:
    """
    Get the version of cuVSLAM library.

    Returns a tuple with the semantic version string and major, minor and patch values.
    """

def set_verbosity(arg: int, /) -> None:
    """
    Set the verbosity level of the library.

    Available values: 0 (default) for no output, 1 for error messages, 2 for warnings, 3 for info messages.
    """

def warm_up_gpu() -> None:
    """
    Warm up the GPU (CUDA runtime).

    It is not necessary to call it before the first call to cuvslam, but it can help to reduce the first call latency.Throws an exception if CUDA, cusolver or cublas initialization fails.
    """

class Pose:
    """
    Transformation from one frame to another.

    Consists of rotation (quaternion in x, y, z, w order) and translation (3-vector).
    cuVSLAM uses OpenCV coordinate system convention: x is right, y is down, z is forward.
    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, rotation: Sequence[float], translation: Sequence[float]) -> None: ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...

    @property
    def rotation(self) -> Annotated[NDArray[numpy.float32], dict(shape=(4))]:
        """Rotation (quaternion in x, y, z, w order)"""

    @rotation.setter
    def rotation(self, arg: object, /) -> None: ...

    @property
    def translation(self) -> Annotated[NDArray[numpy.float32], dict(shape=(3))]:
        """Translation (3-vector)"""

    @translation.setter
    def translation(self, arg: object, /) -> None: ...

class Distortion:
    r"""
    Camera distortion model with parameters.

    Supports Pinhole (no distortion), Brown (radial and tangential), Fisheye (equidistant), and Polynomial distortion models.

    Common definitions used below:

    * principal point :math:`(c_x, c_y)`
    * focal length :math:`(f_x, f_y)`

    Supported values of distortion_model:

    **Pinhole (0 parameters):**
    No distortion

    **Fisheye (4 parameters):**
    Also known as equidistant distortion model for pinhole cameras.
    Coefficients k1, k2, k3, k4 are 100% compatible with ethz-asl/kalibr/pinhole-equi and OpenCV::fisheye.
    See: 'A Generic Camera Model and Calibration Method for Conventional, Wide-Angle, and Fish-Eye Lenses' by Juho Kannala and Sami S. Brandt for further information.
    Please note, this approach (pinhole+undistort) has a limitation and works only field-of-view below 180°
    For the TUMVI dataset FOV is ~190°.
    EuRoC and ORB_SLAM3 use a different approach (direct project/unproject without pinhole) and support >180°, so their coefficient is incompatible with this model.

    * 0-3: fisheye distortion coefficients :math:`(k_1, k_2, k_3, k_4)`

    Each 3D point :math:`(x, y, z)` is projected in the following way:
    :math:`(u, v) = (c_x, c_y) + diag(f_x, f_y) \cdot (distortedR(r) \cdot (x_n, y_n) / r)`

    where:

    :math:`distortedR(r) = \arctan(r) \cdot (1 + k_1 \cdot \arctan^2(r) + k_2 \cdot \arctan^4(r) + k_3 \cdot \arctan^6(r) + k_4 \cdot \arctan^8(r))`
    :math:`x_n = x/z`
    :math:`y_n = y/z`
    :math:`r = \sqrt{x_n^2 + y_n^2}`

    **Brown (5 parameters):**

    * 0-2: radial distortion coefficients :math:`(k_1, k_2, k_3)`
    * 3-4: tangential distortion coefficients :math:`(p_1, p_2)`

    Each 3D point :math:`(x, y, z)` is projected in the following way:
    :math:`(u, v) = (c_x, c_y) + diag(f_x, f_y) \cdot (radial \cdot (x_n, y_n) + tangential)`

    where:

    :math:`radial = (1 + k_1 \cdot r^2 + k_2 \cdot r^4 + k_3 \cdot r^6)`
    :math:`tangential.x = 2 \cdot p_1 \cdot x_n \cdot y_n + p_2 \cdot (r^2 + 2 \cdot x_n^2)`
    :math:`tangential.y = p_1 \cdot (r^2 + 2 \cdot y_n^2) + 2 \cdot p_2 \cdot x_n \cdot y_n`
    :math:`x_n = x/z`
    :math:`y_n = y/z`
    :math:`r = \sqrt{x_n^2 + y_n^2}`

    **Polynomial (8 parameters):**
    Coefficients are compatible with first 8 coefficients of OpenCV distortion model.

    * 0-1: radial distortion coefficients :math:`(k_1, k_2)`
    * 2-3: tangential distortion coefficients :math:`(p_1, p_2)`
    * 4-7: radial distortion coefficients :math:`(k_3, k_4, k_5, k_6)`

    Each 3D point :math:`(x, y, z)` is projected in the following way:
    :math:`(u, v) = (c_x, c_y) + diag(f_x, f_y) \cdot (radial \cdot (x_n, y_n) + tangential)`

    where:

    :math:`radial = \frac{1 + k_1 \cdot r^2 + k_2 \cdot r^4 + k_3 \cdot r^6}{1 + k_4 \cdot r^2 + k_5 \cdot r^4 + k_6 \cdot r^6}`
    :math:`tangential.x = 2 \cdot p_1 \cdot x_n \cdot y_n + p_2 \cdot (r^2 + 2 \cdot x_n^2)`
    :math:`tangential.y = p_1 \cdot (r^2 + 2 \cdot y_n^2) + 2 \cdot p_2 \cdot x_n \cdot y_n`
    :math:`x_n = x/z`
    :math:`y_n = y/z`
    :math:`r = \sqrt{x_n^2 + y_n^2}`
    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, model: Distortion.Model, parameters: Sequence[float] = []) -> None: ...

    @property
    def model(self) -> Distortion.Model:
        """Distortion model type, see :class:`Distortion.Model`"""

    @model.setter
    def model(self, arg: Distortion.Model, /) -> None: ...

    @property
    def parameters(self) -> list[float]:
        """Array of distortion parameters depending on model"""

    @parameters.setter
    def parameters(self, arg: Sequence[float], /) -> None: ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...

    class Model(enum.Enum):
        """Distortion model types for camera calibration"""

        Pinhole = 0
        """No distortion (0 parameters)"""

        Brown = 2
        """Brown distortion model with 3 radial and 2 tangential coefficients"""

        Fisheye = 1
        """Fisheye distortion model (equidistant) with 4 coefficients"""

        Polynomial = 3
        """
        Polynomial distortion model with 8 coefficients, order: (k1, k2, p1, p2, k3, k4, k5, k6)
        """

    Pinhole: Distortion.Model = Distortion.Model.Pinhole

    Brown: Distortion.Model = Distortion.Model.Brown

    Fisheye: Distortion.Model = Distortion.Model.Fisheye

    Polynomial: Distortion.Model = Distortion.Model.Polynomial

class Camera:
    """
    Camera calibration parameters.

    Describes intrinsic and extrinsic parameters of a camera and per-camera settings.
    For camera coordinate system, top left pixel has (0, 0) coordinate (y is down, x is right).
    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, size: Sequence[int], principal: Sequence[float], focal: Sequence[float], rig_from_camera: Pose = ..., distortion: Distortion = ..., border_top: int = 0, border_bottom: int = 0, border_left: int = 0, border_right: int = 0) -> None: ...

    @property
    def rig_from_camera(self) -> Pose:
        """
        Transformation from the camera coordinate frame to the rig coordinate frame
        """

    @rig_from_camera.setter
    def rig_from_camera(self, arg: Pose, /) -> None: ...

    @property
    def distortion(self) -> Distortion:
        """Distortion parameters, see :class:`Distortion`"""

    @distortion.setter
    def distortion(self, arg: Distortion, /) -> None: ...

    @property
    def border_top(self) -> int:
        """Top border to ignore in pixels (0 to use full frame)"""

    @border_top.setter
    def border_top(self, arg: int, /) -> None: ...

    @property
    def border_bottom(self) -> int:
        """Bottom border to ignore in pixels (0 to use full frame)"""

    @border_bottom.setter
    def border_bottom(self, arg: int, /) -> None: ...

    @property
    def border_left(self) -> int:
        """Left border to ignore in pixels (0 to use full frame)"""

    @border_left.setter
    def border_left(self, arg: int, /) -> None: ...

    @property
    def border_right(self) -> int:
        """Right border to ignore in pixels (0 to use full frame)"""

    @border_right.setter
    def border_right(self, arg: int, /) -> None: ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...

    @property
    def size(self) -> Annotated[NDArray[numpy.int32], dict(shape=(2))]:
        """Size of the camera (width, height)"""

    @size.setter
    def size(self, arg: object, /) -> None: ...

    @property
    def principal(self) -> Annotated[NDArray[numpy.float32], dict(shape=(2))]:
        """Principal point (cx, cy)"""

    @principal.setter
    def principal(self, arg: object, /) -> None: ...

    @property
    def focal(self) -> Annotated[NDArray[numpy.float32], dict(shape=(2))]:
        """Focal length (fx, fy)"""

    @focal.setter
    def focal(self, arg: object, /) -> None: ...

class ImuCalibration:
    """IMU Calibration parameters"""

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, rig_from_imu: Pose = ..., gyroscope_noise_density: float, accelerometer_noise_density: float, gyroscope_random_walk: float, accelerometer_random_walk: float, frequency: float) -> None: ...

    @property
    def rig_from_imu(self) -> Pose:
        """Transformation from IMU coordinate frame to the rig coordinate frame"""

    @rig_from_imu.setter
    def rig_from_imu(self, arg: Pose, /) -> None: ...

    @property
    def gyroscope_noise_density(self) -> float:
        """Gyroscope noise density in :math:`rad/(s*sqrt(hz))`"""

    @gyroscope_noise_density.setter
    def gyroscope_noise_density(self, arg: float, /) -> None: ...

    @property
    def accelerometer_noise_density(self) -> float:
        """Accelerometer noise density in :math:`m/(s^2*sqrt(hz))`"""

    @accelerometer_noise_density.setter
    def accelerometer_noise_density(self, arg: float, /) -> None: ...

    @property
    def gyroscope_random_walk(self) -> float:
        """Gyroscope random walk in :math:`rad/(s^2*sqrt(hz))`"""

    @gyroscope_random_walk.setter
    def gyroscope_random_walk(self, arg: float, /) -> None: ...

    @property
    def accelerometer_random_walk(self) -> float:
        """Accelerometer random walk in :math:`m/(s^3*sqrt(hz))`"""

    @accelerometer_random_walk.setter
    def accelerometer_random_walk(self, arg: float, /) -> None: ...

    @property
    def frequency(self) -> float:
        """IMU frequency in :math:`hz`"""

    @frequency.setter
    def frequency(self, arg: float, /) -> None: ...

    def __repr__(self) -> str: ...

class Rig:
    """Rig consisting of cameras and 0 or 1 IMU sensors"""

    def __init__(self, cameras: Sequence[Camera] = [], imus: Sequence[ImuCalibration] = []) -> None: ...

    @property
    def cameras(self) -> list[Camera]:
        """List of cameras in the rig, see :class:`Camera`"""

    @cameras.setter
    def cameras(self, arg: Sequence[Camera], /) -> None: ...

    @property
    def imus(self) -> list[ImuCalibration]:
        """
        List of IMU sensors in the rig (0 or 1 only), see :class:`ImuCalibration`
        """

    @imus.setter
    def imus(self, arg: Sequence[ImuCalibration], /) -> None: ...

    def __repr__(self) -> str: ...

class PoseStamped:
    """Pose with timestamp"""

    def __init__(self) -> None: ...

    @property
    def timestamp_ns(self) -> int:
        """Pose timestamp in nanoseconds"""

    @timestamp_ns.setter
    def timestamp_ns(self, arg: int, /) -> None: ...

    @property
    def pose(self) -> Pose:
        """Pose (transformation between two coordinate frames)"""

    @pose.setter
    def pose(self, arg: Pose, /) -> None: ...

    def __repr__(self) -> str: ...

class PoseWithCovariance:
    """Pose with covariance matrix"""

    def __init__(self) -> None: ...

    @property
    def pose(self) -> Pose:
        """Pose (transformation between two coordinate frames)"""

    @property
    def covariance(self) -> list[float]:
        """
        6x6 covariance matrix for the pose (row-major)
        The orientation parameters use a fixed-axis representation.
        The parameters: (rotation about X axis, rotation about Y axis, rotation about Z axis, x, y, z)
        """

    def __repr__(self) -> str: ...

class PoseEstimate:
    """
    Rig pose estimate from the tracker. The pose is world_from_rig where:

    The rig coordinate frame is user-defined and depends on the extrinsic parameters of the cameras.
    The world coordinate frame is an arbitrary 3D coordinate frame, corresponding to the rig frame at the start of tracking.
    """

    def __init__(self) -> None: ...

    @property
    def timestamp_ns(self) -> int:
        """Timestamp of the pose estimate in nanoseconds"""

    @property
    def world_from_rig(self) -> PoseWithCovariance | None:
        """Rig pose in the world coordinate frame"""

    def __repr__(self) -> str: ...

class Observation:
    """2D observation of a landmark in an image"""

    def __init__(self) -> None: ...

    @property
    def id(self) -> int:
        """Unique ID of the observed landmark"""

    @id.setter
    def id(self, arg: int, /) -> None: ...

    @property
    def u(self) -> float:
        """Horizontal pixel coordinate of the observation"""

    @u.setter
    def u(self, arg: float, /) -> None: ...

    @property
    def v(self) -> float:
        """Vertical pixel coordinate of the observation"""

    @v.setter
    def v(self, arg: float, /) -> None: ...

    @property
    def camera_index(self) -> int:
        """Index of the camera that made this observation"""

    @camera_index.setter
    def camera_index(self, arg: int, /) -> None: ...

    def __repr__(self) -> str: ...

class Landmark:
    """3D landmark point"""

    def __init__(self) -> None: ...

    @property
    def id(self) -> int:
        """Unique ID of the landmark"""

    @id.setter
    def id(self, arg: int, /) -> None: ...

    @property
    def coords(self) -> list[float]:
        """3D coordinates of the landmark in the world coordinate frame"""

    @coords.setter
    def coords(self, arg: Sequence[float], /) -> None: ...

    def __repr__(self) -> str: ...

class ImuMeasurement:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, timestamp_ns: int, linear_accelerations: Sequence[float], angular_velocities: Sequence[float]) -> None: ...

    @property
    def timestamp_ns(self) -> int:
        """Timestamp of the IMU measurement in nanoseconds"""

    @timestamp_ns.setter
    def timestamp_ns(self, arg: int, /) -> None: ...

    @property
    def linear_accelerations(self) -> list[float]:
        """Linear accelerations in :math:`m/s^2`"""

    @linear_accelerations.setter
    def linear_accelerations(self, arg: Sequence[float], /) -> None: ...

    @property
    def angular_velocities(self) -> list[float]:
        """Angular velocities in :math:`rad/s`"""

    @angular_velocities.setter
    def angular_velocities(self, arg: Sequence[float], /) -> None: ...

    def __repr__(self) -> str: ...

class Odometry:
    """Visual Inertial Odometry (VIO) Tracker"""

    def __init__(self, rig: Rig, cfg: Odometry.Config = ...) -> None: ...

    class MulticameraMode(enum.Enum):
        Performance = 0
        """Optimized for speed"""

        Precision = 1
        """Optimized for accuracy"""

        Moderate = 2
        """Balance between speed and accuracy"""

    Performance: Odometry.MulticameraMode = Odometry.MulticameraMode.Performance

    Precision: Odometry.MulticameraMode = Odometry.MulticameraMode.Precision

    Moderate: Odometry.MulticameraMode = Odometry.MulticameraMode.Moderate

    class OdometryMode(enum.Enum):
        Multicamera = 0
        """
        Uses multiple synchronized cameras, all cameras need to have frustum overlap with at least one another. Simplest case: stereo camera pair.
        """

        Inertial = 1
        """
        Uses stereo camera and IMU measurements. A single stereo-camera with a single IMU is supported.
        """

        RGBD = 2
        """
        Uses RGB-D camera for tracking. A single RGB-D camera is supported. RGB & Depth images must be aligned.
        """

        Mono = 3
        """Uses a single camera, tracking is accurate up to scale."""

    Multicamera: Odometry.OdometryMode = Odometry.OdometryMode.Multicamera

    Inertial: Odometry.OdometryMode = Odometry.OdometryMode.Inertial

    RGBD: Odometry.OdometryMode = Odometry.OdometryMode.RGBD

    Mono: Odometry.OdometryMode = Odometry.OdometryMode.Mono

    class RGBDSettings:
        """Settings for RGB-D odometry mode"""

        def __init__(self, *, depth_scale_factor: float = 1.0, depth_camera_id: int = -1, enable_depth_stereo_tracking: bool = False) -> None: ...

        @property
        def depth_scale_factor(self) -> float:
            """Scale factor for depth measurements"""

        @depth_scale_factor.setter
        def depth_scale_factor(self, arg: float, /) -> None: ...

        @property
        def depth_camera_id(self) -> int:
            """ID of the camera that the depth image is aligned with"""

        @depth_camera_id.setter
        def depth_camera_id(self, arg: int, /) -> None: ...

        @property
        def enable_depth_stereo_tracking(self) -> bool:
            """
            Whether to enable stereo tracking between depth-aligned camera and other cameras
            """

        @enable_depth_stereo_tracking.setter
        def enable_depth_stereo_tracking(self, arg: bool, /) -> None: ...

        def __repr__(self) -> str: ...

    class Config:
        def __init__(self, *, multicam_mode: Odometry.MulticameraMode = Odometry.MulticameraMode.Precision, odometry_mode: Odometry.OdometryMode = Odometry.OdometryMode.Multicamera, use_gpu: bool = True, async_sba: bool = True, use_motion_model: bool = True, use_denoising: bool = False, rectified_stereo_camera: bool = False, enable_observations_export: bool = True, enable_landmarks_export: bool = True, enable_final_landmarks_export: bool = False, max_frame_delta_s: float = 1.0, debug_dump_directory: str = '', debug_imu_mode: bool = False, rgbd_settings: Odometry.RGBDSettings = ...) -> None: ...

        @property
        def multicam_mode(self) -> Odometry.MulticameraMode:
            """See :class:`Odometry.MulticameraMode`"""

        @multicam_mode.setter
        def multicam_mode(self, arg: Odometry.MulticameraMode, /) -> None: ...

        @property
        def odometry_mode(self) -> Odometry.OdometryMode:
            """See :class:`Odometry.OdometryMode`"""

        @odometry_mode.setter
        def odometry_mode(self, arg: Odometry.OdometryMode, /) -> None: ...

        @property
        def use_gpu(self) -> bool:
            """Enable to use GPU acceleration"""

        @use_gpu.setter
        def use_gpu(self, arg: bool, /) -> None: ...

        @property
        def async_sba(self) -> bool:
            """Enable to run bundle adjustment asynchronously"""

        @async_sba.setter
        def async_sba(self, arg: bool, /) -> None: ...

        @property
        def use_motion_model(self) -> bool:
            """Enable to use motion model for pose prediction"""

        @use_motion_model.setter
        def use_motion_model(self, arg: bool, /) -> None: ...

        @property
        def use_denoising(self) -> bool:
            """Enable to apply denoising to input images"""

        @use_denoising.setter
        def use_denoising(self, arg: bool, /) -> None: ...

        @property
        def rectified_stereo_camera(self) -> bool:
            """Enable if stereo cameras are rectified and horizontally aligned"""

        @rectified_stereo_camera.setter
        def rectified_stereo_camera(self, arg: bool, /) -> None: ...

        @property
        def enable_observations_export(self) -> bool:
            """Enable to export landmark observations in images during tracking"""

        @enable_observations_export.setter
        def enable_observations_export(self, arg: bool, /) -> None: ...

        @property
        def enable_landmarks_export(self) -> bool:
            """Enable to export landmarks during tracking"""

        @enable_landmarks_export.setter
        def enable_landmarks_export(self, arg: bool, /) -> None: ...

        @property
        def enable_final_landmarks_export(self) -> bool:
            """
            Enable to export final landmarks. Also sets enable_landmarks_export and enable_observations_export.
            """

        @enable_final_landmarks_export.setter
        def enable_final_landmarks_export(self, arg: bool, /) -> None: ...

        @property
        def max_frame_delta_s(self) -> float:
            """Maximum time difference between frames in seconds"""

        @max_frame_delta_s.setter
        def max_frame_delta_s(self, arg: float, /) -> None: ...

        @property
        def debug_dump_directory(self) -> str:
            """Directory for debug data dumps. If empty, no debug data will be dumped"""

        @debug_dump_directory.setter
        def debug_dump_directory(self, arg: str, /) -> None: ...

        @property
        def debug_imu_mode(self) -> bool:
            """Enable IMU debug mode"""

        @debug_imu_mode.setter
        def debug_imu_mode(self, arg: bool, /) -> None: ...

        @property
        def rgbd_settings(self) -> Odometry.RGBDSettings:
            """Settings for RGB-D odometry mode. See :class:`Odometry.RGBDSettings`"""

        @rgbd_settings.setter
        def rgbd_settings(self, arg: Odometry.RGBDSettings, /) -> None: ...

    class State:
        """
        Odometry state snapshot. Contains pose, observations, landmarks, etc.
        Consumed by :meth:`Slam.track`.
        """

        def __init__(self) -> None: ...

        @property
        def frame_id(self) -> int:
            """Frame id of the state"""

        @frame_id.setter
        def frame_id(self, arg: int, /) -> None: ...

        @property
        def timestamp_ns(self) -> int:
            """Timestamp in nanoseconds"""

        @timestamp_ns.setter
        def timestamp_ns(self, arg: int, /) -> None: ...

        @property
        def delta(self) -> Pose:
            """Delta pose (Pose)"""

        @delta.setter
        def delta(self, arg: Pose, /) -> None: ...

        @property
        def keyframe(self) -> bool:
            """Whether this frame is a keyframe"""

        @keyframe.setter
        def keyframe(self, arg: bool, /) -> None: ...

        @property
        def warming_up(self) -> bool:
            """Whether tracker is in warming up phase"""

        @warming_up.setter
        def warming_up(self, arg: bool, /) -> None: ...

        @property
        def gravity(self) -> list[float] | None:
            """Optional gravity vector (if available)"""

        @gravity.setter
        def gravity(self, arg: Sequence[float] | None) -> None: ...

        @property
        def observations(self) -> list[Observation]:
            """List of 2D landmark observations (Observation)"""

        @observations.setter
        def observations(self, arg: Sequence[Observation], /) -> None: ...

        @property
        def landmarks(self) -> list[Landmark]:
            """List of 3D landmarks (Landmark)"""

        @landmarks.setter
        def landmarks(self, arg: Sequence[Landmark], /) -> None: ...

        @property
        def context(self) -> dict[int, "std::shared_ptr<cuvslam::Odometry::State::Context>"]:
            """Context of the state"""

        @context.setter
        def context(self, arg: Mapping[int, "std::shared_ptr<cuvslam::Odometry::State::Context>"], /) -> None: ...

        def __repr__(self) -> str: ...

    def track(self, timestamp: int, images: Sequence[Annotated[NDArray, dict(writable=False)]], masks: Sequence[Annotated[NDArray, dict(writable=False)]] | None = None, depths: Sequence[Annotated[NDArray, dict(writable=False)]] | None = None) -> PoseEstimate:
        """
        Track a rig pose using current image frame.

        Synchronously tracks current image frame and returns a PoseEstimate.

        By default, this function uses visual odometry to compute a pose.
        In Inertial mode, if visual odometry tracker fails to compute a pose, the function returns the position calculated from a user-provided IMU data.
        If after several calls of :meth:`track` visual odometry is not able to recover, then invalid pose will be returned.
        The track will output poses in the same coordinate system until a loss of tracking.

        All cameras must be synchronized. If a camera rig provides 'almost synchronized' frames, the timestamps should be within 1 millisecond.

        Images (masks, depth images, etc.) can be numpy arrays or tensors, both GPU (CUDA) and CPU.
        All data must be of the same type (either GPU or CPU).
        This is not the same as `odom_config.use_gpu` - if odometry uses GPU for computations,
        images etc. can still be either CPU or GPU arrays/tensors.

        The images etc. must be in the same order as cameras in the rig.
        If data for a camera is not available, pass empty array or tensor for that camera image.

        Parameters:
            timestamp: Image timestamp in nanoseconds
            images: List of numpy arrays containing the camera images
            masks: Optional list of numpy arrays containing masks for the images
            depths: Optional list of numpy arrays containing depth images

        Returns:
            :class:`PoseEstimate` object with the computed pose. If tracking fails, `is_valid` will be False.
        """

    def register_imu_measurement(self, sensor_index: int, imu_measurement: ImuMeasurement) -> None:
        """
        Register an IMU measurement.

        Requires Inertial mode. If visual odometry loses camera position, it briefly continues execution
        using user-provided IMU measurements while trying to recover the position.
        IMU sensors and cameras clocks must be synchronized, :meth:`track` and :meth:`register_imu_measurement` must be called in strict ascending order of timestamps.
        """

    def get_last_observations(self, camera_index: int) -> list[Observation]:
        """
        Get an array of observations from the last VO frame.

        Requires `enable_observations_export=True` in :class:`Odometry.Config`.
        """

    def get_last_landmarks(self) -> list[Landmark]:
        """
        Get an array of landmarks from the last VO frame.

        Landmarks are 3D points in the last camera frame.
        Requires `enable_landmarks_export=True` in :class:`Odometry.Config`.
        """

    def get_last_gravity(self) -> object:
        """
        Get gravity vector in the last VO frame.

        Returns `None` if gravity is not yet available.
        Requires Inertial mode (`odometry_mode=Odometry.OdometryMode.Inertial` in :class:`Odometry.Config`)
        """

    def get_final_landmarks(self) -> dict[int, list[float]]:
        """
        Get all final landmarks from all frames.

        Landmarks are 3D points in the odometry start frame.
        Requires `enable_final_landmarks_export=True` in :class:`Odometry.Config`.
        """

    def get_state(self) -> Odometry.State:
        """
        Get the current tracker state (pose, observations, landmarks, etc) as a State object.

        Returns:
            Odometry.State: The current tracker state snapshot.
        """

    def get_primary_cameras(self) -> list[int]:
        """
        Returns a list of primary cameras.

        Primary cameras are the ones where observations are always present.
        The list is required to initialize Slam.
        """

class Slam:
    """Simultaneous Localization and Mapping (SLAM)"""

    def __init__(self, rig: Rig, primary_cameras: Sequence[int], config: Slam.Config = ...) -> None: ...

    class DataLayer(enum.Enum):
        """Data layer for SLAM"""

        Landmarks = 0
        """Landmarks that are visible in the current frame"""

        Map = 1
        """Landmarks of the map"""

        LoopClosure = 2
        """Map's landmarks that are visible in the last loop closure event"""

        LocalizerMap = 5
        """Landmarks of the Localizer map (opened database)"""

        LocalizerLandmarks = 6
        """Landmarks that are visible in the localization"""

        LocalizerLoopClosure = 7
        """
        Landmarks that are visible in the final loop closure of the localization
        """

    class Landmarks:
        """Collection of landmarks with a capture timestamp."""

        def __init__(self) -> None: ...

        @property
        def timestamp_ns(self) -> int:
            """Timestamp of landmarks in nanoseconds"""

        @timestamp_ns.setter
        def timestamp_ns(self, arg: int, /) -> None: ...

        @property
        def landmarks(self) -> list[Slam.Landmark]:
            """List of landmarks, see :class:`Slam.Landmark`"""

        @landmarks.setter
        def landmarks(self, arg: Sequence[Slam.Landmark], /) -> None: ...

        def __repr__(self) -> str: ...

    Map: Slam.DataLayer = Slam.DataLayer.Map

    LoopClosure: Slam.DataLayer = Slam.DataLayer.LoopClosure

    LocalizerMap: Slam.DataLayer = Slam.DataLayer.LocalizerMap

    LocalizerLandmarks: Slam.DataLayer = Slam.DataLayer.LocalizerLandmarks

    LocalizerLoopClosure: Slam.DataLayer = Slam.DataLayer.LocalizerLoopClosure

    class Config:
        """SLAM configuration parameters"""

        @overload
        def __init__(self) -> None: ...

        @overload
        def __init__(self, *, map_cache_path: str = '', use_gpu: bool = True, sync_mode: bool = False, enable_reading_internals: bool = True, planar_constraints: bool = False, gt_align_mode: bool = False, map_cell_size: float = 0.0, max_landmarks_distance: float = 100.0, max_map_size: int = 300, throttling_time_ms: int = 0) -> None: ...

        @property
        def map_cache_path(self) -> str:
            """
            If empty, map is kept in memory only. Else, map is synced to disk (LMDB) at this path, allowing large-scale maps; if the path already exists it will be overwritten. To load an existing map, use LocalizeInMap(). To save map, use SaveMap().
            """

        @map_cache_path.setter
        def map_cache_path(self, arg: str, /) -> None: ...

        @property
        def use_gpu(self) -> bool:
            """Whether to use GPU acceleration"""

        @use_gpu.setter
        def use_gpu(self, arg: bool, /) -> None: ...

        @property
        def sync_mode(self) -> bool:
            """
            If true, localization and mapping run in the same thread as visual odometry
            """

        @sync_mode.setter
        def sync_mode(self, arg: bool, /) -> None: ...

        @property
        def enable_reading_internals(self) -> bool:
            """Enable reading internal data from SLAM"""

        @enable_reading_internals.setter
        def enable_reading_internals(self, arg: bool, /) -> None: ...

        @property
        def planar_constraints(self) -> bool:
            """Modify poses so camera moves on a horizontal plane"""

        @planar_constraints.setter
        def planar_constraints(self, arg: bool, /) -> None: ...

        @property
        def gt_align_mode(self) -> bool:
            """Special mode for visual map building with ground truth"""

        @gt_align_mode.setter
        def gt_align_mode(self, arg: bool, /) -> None: ...

        @property
        def map_cell_size(self) -> float:
            """Size of map cell (0 to auto-calculate from camera baseline)"""

        @map_cell_size.setter
        def map_cell_size(self, arg: float, /) -> None: ...

        @property
        def max_landmarks_distance(self) -> float:
            """Maximum distance from camera to landmark for inclusion in map"""

        @max_landmarks_distance.setter
        def max_landmarks_distance(self, arg: float, /) -> None: ...

        @property
        def max_map_size(self) -> int:
            """Maximum number of poses in SLAM pose graph (0 for unlimited)"""

        @max_map_size.setter
        def max_map_size(self, arg: int, /) -> None: ...

        @property
        def throttling_time_ms(self) -> int:
            """Minimum time between loop closure events in milliseconds"""

        @throttling_time_ms.setter
        def throttling_time_ms(self, arg: int, /) -> None: ...

        def __repr__(self) -> str: ...

    class LocalizationSettings:
        """Localization settings"""

        def __init__(self, *, horizontal_search_radius: float, vertical_search_radius: float, horizontal_step: float, vertical_step: float, angular_step_rads: float) -> None: ...

        @property
        def horizontal_search_radius(self) -> float:
            """Horizontal search radius in meters"""

        @horizontal_search_radius.setter
        def horizontal_search_radius(self, arg: float, /) -> None: ...

        @property
        def vertical_search_radius(self) -> float:
            """Vertical search radius in meters"""

        @vertical_search_radius.setter
        def vertical_search_radius(self, arg: float, /) -> None: ...

        @property
        def horizontal_step(self) -> float:
            """Horizontal step in meters"""

        @horizontal_step.setter
        def horizontal_step(self, arg: float, /) -> None: ...

        @property
        def vertical_step(self) -> float:
            """Vertical step in meters"""

        @vertical_step.setter
        def vertical_step(self, arg: float, /) -> None: ...

        @property
        def angular_step_rads(self) -> float:
            """Angular step around vertical axis in radians"""

        @angular_step_rads.setter
        def angular_step_rads(self, arg: float, /) -> None: ...

        def __repr__(self) -> str: ...

    class Metrics:
        """SLAM metrics"""

        def __init__(self) -> None: ...

        @property
        def timestamp_ns(self) -> int:
            """Timestamp of measurements in nanoseconds"""

        @timestamp_ns.setter
        def timestamp_ns(self, arg: int, /) -> None: ...

        @property
        def lc_status(self) -> bool:
            """Loop closure status"""

        @lc_status.setter
        def lc_status(self, arg: bool, /) -> None: ...

        @property
        def pgo_status(self) -> bool:
            """Pose graph optimization status"""

        @pgo_status.setter
        def pgo_status(self, arg: bool, /) -> None: ...

        @property
        def lc_selected_landmarks_count(self) -> int:
            """Count of landmarks selected for loop closure"""

        @lc_selected_landmarks_count.setter
        def lc_selected_landmarks_count(self, arg: int, /) -> None: ...

        @property
        def lc_tracked_landmarks_count(self) -> int:
            """Count of landmarks tracked in loop closure"""

        @lc_tracked_landmarks_count.setter
        def lc_tracked_landmarks_count(self, arg: int, /) -> None: ...

        @property
        def lc_pnp_landmarks_count(self) -> int:
            """Count of landmarks in PnP for loop closure"""

        @lc_pnp_landmarks_count.setter
        def lc_pnp_landmarks_count(self, arg: int, /) -> None: ...

        @property
        def lc_good_landmarks_count(self) -> int:
            """Count of landmarks in loop closure"""

        @lc_good_landmarks_count.setter
        def lc_good_landmarks_count(self, arg: int, /) -> None: ...

        def __repr__(self) -> str: ...

    class PoseGraphNode:
        """Node in a pose graph"""

        def __init__(self) -> None: ...

        @property
        def id(self) -> int:
            """Node identifier"""

        @id.setter
        def id(self, arg: int, /) -> None: ...

        @property
        def node_pose(self) -> Pose:
            """Node pose"""

        @node_pose.setter
        def node_pose(self, arg: Pose, /) -> None: ...

        def __repr__(self) -> str: ...

    class PoseGraphEdge:
        """Edge in a pose graph"""

        def __init__(self) -> None: ...

        @property
        def node_from(self) -> int:
            """Source node ID"""

        @node_from.setter
        def node_from(self, arg: int, /) -> None: ...

        @property
        def node_to(self) -> int:
            """Target node ID"""

        @node_to.setter
        def node_to(self, arg: int, /) -> None: ...

        @property
        def transform(self) -> Pose:
            """Transformation from source to target node"""

        @transform.setter
        def transform(self, arg: Pose, /) -> None: ...

        def __repr__(self) -> str: ...

        @property
        def covariance(self) -> Annotated[NDArray[numpy.float32], dict(shape=(36))]:
            """Covariance matrix of the transformation"""

        @covariance.setter
        def covariance(self, arg: object, /) -> None: ...

    class PoseGraph:
        """Pose graph structure"""

        def __init__(self) -> None: ...

        @property
        def nodes(self) -> list[Slam.PoseGraphNode]:
            """List of nodes in the graph"""

        @nodes.setter
        def nodes(self, arg: Sequence[Slam.PoseGraphNode], /) -> None: ...

        @property
        def edges(self) -> list[Slam.PoseGraphEdge]:
            """List of edges in the graph"""

        @edges.setter
        def edges(self, arg: Sequence[Slam.PoseGraphEdge], /) -> None: ...

        def __repr__(self) -> str: ...

    class Landmark:
        """
        Landmark with additional information.

        Contains unique identifier, weight, and 3D coordinates in the world frame.
        """

        def __init__(self) -> None: ...

        @property
        def id(self) -> int:
            """Unique ID of the landmark"""

        @id.setter
        def id(self, arg: int, /) -> None: ...

        @property
        def weight(self) -> float:
            """Landmark weight (selection/reliability metric)"""

        @weight.setter
        def weight(self, arg: float, /) -> None: ...

        @property
        def coords(self) -> list[float]:
            """3D coordinates of the landmark in the world coordinate frame"""

        @coords.setter
        def coords(self, arg: Sequence[float], /) -> None: ...

        def __repr__(self) -> str: ...

    class LocalizerProbe:
        """
        Localizer probe used during map localization.

        Contains input guess pose, result pose and weights, and success flag.
        """

        def __init__(self) -> None: ...

        @property
        def id(self) -> int:
            """Probe identifier"""

        @id.setter
        def id(self, arg: int, /) -> None: ...

        @property
        def guess_pose(self) -> Pose:
            """Input guess pose"""

        @guess_pose.setter
        def guess_pose(self, arg: Pose, /) -> None: ...

        @property
        def exact_result_pose(self) -> Pose:
            """Exact pose if localization succeeded"""

        @exact_result_pose.setter
        def exact_result_pose(self, arg: Pose, /) -> None: ...

        @property
        def weight(self) -> float:
            """Input weight for the probe"""

        @weight.setter
        def weight(self, arg: float, /) -> None: ...

        @property
        def exact_result_weight(self) -> float:
            """Weight of the resulting solution"""

        @exact_result_weight.setter
        def exact_result_weight(self, arg: float, /) -> None: ...

        @property
        def solved(self) -> bool:
            """True if the probe was solved successfully"""

        @solved.setter
        def solved(self, arg: bool, /) -> None: ...

        def __repr__(self) -> str: ...

    class LocalizerProbes:
        """Collection of localizer probes for a localization attempt."""

        def __init__(self) -> None: ...

        @property
        def timestamp_ns(self) -> int:
            """Timestamp of localizer try in nanoseconds"""

        @timestamp_ns.setter
        def timestamp_ns(self, arg: int, /) -> None: ...

        @property
        def size(self) -> float:
            """Size of search area"""

        @size.setter
        def size(self, arg: float, /) -> None: ...

        @property
        def probes(self) -> list[Slam.LocalizerProbe]:
            """List of localizer probes, see :class:`Slam.LocalizerProbe`"""

        @probes.setter
        def probes(self, arg: Sequence[Slam.LocalizerProbe], /) -> None: ...

        def __repr__(self) -> str: ...

    def track(self, state: Odometry.State, gt_pose: Pose | None = None) -> Pose:
        """
        Update SLAM pose based on Odometry state and past SLAM loop closures.

        This method should be called after each successful Odometry.track() call.
        Parameters:
            state: Odometry state containing all tracking data
            gt_pose: Optional ground truth pose. Should be provided if `gt_align_mode` is enabled, otherwise should be None.
        Returns:
            On success returns rig pose estimated by SLAM
        """

    def set_slam_pose(self, pose: Pose) -> None:
        """
        Set the rig SLAM pose to a value provided by a user.

        Parameters:
            pose: Rig pose estimated by customer
        """

    def get_all_slam_poses(self, max_poses_count: int = 0) -> list[PoseStamped]:
        """
        Get all SLAM poses for each frame.

        Parameters:
            max_poses_count: Maximum number of poses to return (0 for all)
        Returns:
            List of poses with timestamps
        This call could be blocked by slam thread.
        """

    def save_map(self, folder_name: str, callback: Callable) -> None:
        """
        Save SLAM database (map) to a folder.

        This folder will be created if it does not exist.
        **WARNING**: *Contents of the folder will be overwritten.*
        This method works asynchronously depending on the `sync_mode` parameter in `Slam.Config`.
        In both cases, the callback will be called with a flag indicating if the map was saved successfully.

        Parameters:
            folder_name: Folder name where SLAM database will be saved
            callback: Function to be called when save is complete (takes bool success parameter)
        """

    def localize_in_map(self, folder_name: str, guess_pose: Pose, images: Sequence[Annotated[NDArray, dict(writable=False)]], settings: Slam.LocalizationSettings, callback: Callable) -> None:
        """
        Localize in the existing database (map) asynchronously.

        Finds the position of the rig in existing SLAM database.
        If successful, sets the SLAM pose to the found position.
        This method works asynchronously depending on the `sync_mode` parameter in `Slam.Config`.
        In both cases, the callback will be called with localization result or error message.
        Parameters:
            folder_name: Folder name which stores saved SLAM database
            guess_pose: Proposed pose where the robot might be
            images: List of numpy arrays containing the camera images
            settings: Localization settings
            callback: Function to be called when localization is complete (takes <Pose | None> result and error message parameters)
        """

    def get_landmarks(self, layer: Slam.DataLayer) -> Slam.Landmarks:
        """
        Get landmarks for a given data layer of SLAM.

        Parameters:
            layer: Data layer to read (see `Slam.DataLayer`)
        Returns:
            Landmarks for a given data layer
        """

    def get_pose_graph(self) -> Slam.PoseGraph:
        """
        Get pose graph consisting of all keyframes and their connections including loop closures.

        Returns:
            Pose graph with nodes and edges
        """

    def get_localizer_probes(self) -> Slam.LocalizerProbes:
        """
        Get localizer probes from the most recent localization attempt.

        Returns:
            Collection of localizer probes with timestamp, search area size, and probe list
        """

    def get_slam_metrics(self) -> Slam.Metrics:
        """
        Get SLAM metrics.

        Returns:
            SLAM metrics
        """

    def get_loop_closure_poses(self) -> list[PoseStamped]:
        """
        Get list of last 10 loop closure poses with timestamps.

        Returns:
            List of poses with timestamps
        """

    @staticmethod
    def merge_maps(rig: Rig, databases: Sequence[str], output_folder: str) -> None:
        """
        Merge existing maps into one map.

        This method merges multiple maps into a single map. Maps must have the same world coordinate frame, e.g. using `set_slam_pose` from a ground truth source.

        This method cannot be used to merge maps from different locations, because pose graphs from input maps must be combined into a single graph.

        Parameters:
            rig: Camera rig configuration
            databases: Input array of directories with existing databases
            output_folder: Directory to save output database
        """
