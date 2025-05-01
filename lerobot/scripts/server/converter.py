import numpy as np

def mjcf_camera_to_view_params(pos, xyaxes, assumed_distance=2.0):
    """
    Converts a MuJoCo camera spec (pos, xyaxes) to azimuth, elevation, distance.
    
    Parameters:
    - pos: (3,) array-like, the camera position
    - xyaxes: (6,) array-like, first two columns of a rotation matrix
    - assumed_distance: distance to compute lookat point from view direction

    Returns:
    - lookat: (3,) np.array
    - azimuth: float, in degrees
    - elevation: float, in degrees
    - distance: float
    """

    pos = np.array(pos)
    x_axis = np.array(xyaxes[:3])
    y_axis = np.array(xyaxes[3:])

    # Compute the camera's Z axis via cross product of X and Y
    z_axis = np.cross(x_axis, y_axis)
    R = np.column_stack((x_axis, y_axis, z_axis))

    # Camera looks along -Z axis
    look_dir = -R[:, 2]
    lookat = pos + assumed_distance * look_dir

    # Compute view parameters
    diff = pos - lookat
    distance = np.linalg.norm(diff)
    azimuth = np.degrees(np.arctan2(diff[1], diff[0]))
    elevation = np.degrees(np.arctan2(diff[2], np.linalg.norm(diff[:2])))

    return lookat, azimuth, elevation, distance


# === Example usage ===
mjcf_pos = [1.465, -1.025, 0.936]
mjcf_xyaxes = [0.631, 0.776, -0.000, -0.340, 0.276, 0.899]

lookat, azimuth, elevation, distance = mjcf_camera_to_view_params(mjcf_pos, mjcf_xyaxes)

print("lookat:   ", lookat)
print("azimuth:  ", azimuth)
print("elevation:", elevation)
print("distance: ", distance)
