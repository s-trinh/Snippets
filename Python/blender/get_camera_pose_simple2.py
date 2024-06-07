import bpy
import bgl
import os

from mathutils import *
from math import *

# https://blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model/38189#38189
def get_calibration_matrix_K_from_blender(cameraName):
    camd = bpy.data.objects[cameraName].data

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
    u_0 = resolution_x_in_px*scale / 2
    v_0 = resolution_y_in_px*scale / 2
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((alpha_u, skew,    u_0),
        (    0  ,  alpha_v, v_0),
        (    0  ,    0,      1 )))
    return K

def get_camera_pose(cameraName, objectName):
    M = Matrix().to_4x4()
    M[1][1] = -1
    M[2][2] = -1

    cam = bpy.data.objects[cameraName]
    object_pose = bpy.data.objects[objectName].matrix_world

    #Normalize orientation with respect to the scale
    object_pose_normalized = object_pose.copy()
    object_orientation_normalized = object_pose_normalized.to_3x3().normalized()
    for i in range(3):
        for j in range(3):
            object_pose_normalized[i][j] = object_orientation_normalized[i][j]

    # print("object_pose_normalized:\n", object_pose_normalized)

    camera_pose = M*cam.matrix_world.inverted()*object_pose_normalized
    print("camera_pose:\n", camera_pose)

    return

get_camera_pose("Camera", "Object")
print('camera intrinsics:\n', get_calibration_matrix_K_from_blender("Camera"))
