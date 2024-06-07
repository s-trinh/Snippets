import bpy
import bpy_extras
from mathutils import Matrix, Vector
import numpy as np
import os

#---------------------------------------------------------------
# 3x4 P matrix from Blender camera
#---------------------------------------------------------------

# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in 
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal), 
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio 
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal), 
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm


    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((alpha_u, skew,    u_0),
        (    0  , alpha_v, v_0),
        (    0  , 0,        1 )))
    return K

# Returns camera rotation and translation matrices from Blender.
# 
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates 
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_4x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam * location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv*R_world2bcam
    T_world2cv = R_bcam2cv*T_world2bcam

    # put into 4x4 matrix
    T = np.eye(4)
    T[0:3,0:3] = R_world2cv
    T[0:3,3] = T_world2cv

    return T

def get_pose_intrinsics_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    T = get_4x4_RT_matrix_from_blender(cam)
    return K, T

def inv_T(T):
    R = T[0:3,0:3]
    tvec = T[0:3,3]
    Rt = np.transpose(R)
    new_tvec = -Rt * tvec
    T_inv = np.eye(4)
    T_inv[0:3,0:3] = Rt
    T_inv[0:3,3:3] = new_tvec
    return T_inv

# ----------------------------------------------------------
if __name__ == "__main__":
    # Insert your camera name here
    cam = bpy.context.scene.objects['Camera']
    K, T_world2cam = get_pose_intrinsics_matrix_from_blender(cam)
    print("K")
    print(K)
    print("T_world2cam")
    print(T_world2cam)

    oos = bpy.context.scene.objects['oos']
    T_oos2world = np.asmatrix(oos.matrix_world)
    T_oos2cam = T_world2cam @ T_oos2world
    print('T_oos2world:\n', T_oos2world)
    print('T_oos2cam:\n', T_oos2cam)
