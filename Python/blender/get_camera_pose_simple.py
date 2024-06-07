import bpy
import bgl
import os

from mathutils import *
from math import *

def get_camera_pose(cameraName, objectName):
  scn = bpy.context.scene
  width = scn.render.resolution_x * scn.render.resolution_percentage / 100.0
  height = scn.render.resolution_y * scn.render.resolution_percentage / 100.0
  cam = bpy.data.objects[cameraName]
  camData = cam.data
  ratio = width / height
  K = Matrix().to_3x3()

  px = (width / 2.0) / tan(camData.angle / 2.0)
  py = (height / 2.0) / tan(camData.angle / 2.0) * ratio
  u0 = (-2.0 * camData.shift_x) * (width / 2.0) + width / 2.0
  v0 = (2.0 * camData.shift_y) * (height / 2.0) * ratio + height / 2.0

  print ("px=", px)
  print ("py=", py)
  print ("u0=", u0)
  print ("v0=", v0)

  M = Matrix().to_4x4()
  M[1][1] = -1
  M[2][2] = -1

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
