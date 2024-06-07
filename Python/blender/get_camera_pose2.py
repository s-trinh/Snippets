import bpy
import bgl
import os

from mathutils import *
from math import *


prefix_pose = "/tmp/camera_pose/"
prefix_image = "/tmp/images/"

# Other method found on the net
# https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
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
def get_3x4_RT_matrix_from_blender(cam):
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

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
         ))
    return RT

def get_camera_pose(cameraName, objectName, scene, frameNumber):
  if not os.path.exists(prefix_pose):
    os.makedirs(prefix_pose)
    
  # init stuff
  #scn = bpy.context.scene
  scn = scene
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
  
  print("object_pose_normalized:\n", object_pose_normalized)
  
  camera_pose = M*cam.matrix_world.inverted()*object_pose_normalized
  print("camera_pose:\n", camera_pose)
  
  filename = prefix_pose + cameraName + "_%03d" % frameNumber + ".txt"
  with open(filename, 'w') as f:
    f.write(str(camera_pose[0][0]) + " ")
    f.write(str(camera_pose[0][1]) + " ")
    f.write(str(camera_pose[0][2]) + " ")
    f.write(str(camera_pose[0][3]) + " ")
    f.write("\n")

    f.write(str(camera_pose[1][0]) + " ")
    f.write(str(camera_pose[1][1]) + " ")
    f.write(str(camera_pose[1][2]) + " ")
    f.write(str(camera_pose[1][3]) + " ")
    f.write("\n")

    f.write(str(camera_pose[2][0]) + " ")
    f.write(str(camera_pose[2][1]) + " ")
    f.write(str(camera_pose[2][2]) + " ")
    f.write(str(camera_pose[2][3]) + " ")
    f.write("\n")

    f.write(str(camera_pose[3][0]) + " ")
    f.write(str(camera_pose[3][1]) + " ")
    f.write(str(camera_pose[3][2]) + " ")
    f.write(str(camera_pose[3][3]) + " ")
    f.write("\n")

  return


def my_handler(scene):
  frameNumber = scene.frame_current
  print("\n\nFrame Change", scene.frame_current)
  print("Frame Change2", bpy.context.scene.frame_current)

  print("\nCamera:")
  get_camera_pose("Camera", "Object", scene, frameNumber)


#bpy.app.handlers.frame_change_pre.append(my_handler)

# @url=https://blender.stackexchange.com/questions/17839/python-render-specific-frames
# @url=https://stackoverflow.com/questions/14982836/rendering-and-saving-images-through-blender-python
# Render from frame=1 to frame=249
# Note1: crash when frame=250
# Note2: check if frame=1 is equivalent to frame=1 or frame=0 when renderering animation
step_count = 249

scene = bpy.context.scene
for step in range(1, step_count):
  # Write images
  #bpy.data.scenes["Scene"].render.filepath = '/tmp/%04d.png' % step
  #bpy.ops.render.render( write_still=True )
  
  # Set render frame
  scene.frame_set(step)
  
  # Set filename and render
  if not os.path.exists(prefix_image):
    os.makedirs(prefix_image)
  scene.render.filepath = (prefix_image + '%04d.png') % step
  bpy.ops.render.render( write_still=True )
  
  # Get camera pose
  my_handler(scene)
