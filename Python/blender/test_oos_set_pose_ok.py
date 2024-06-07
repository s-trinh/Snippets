import bpy
from mathutils import *

def look_at(obj_camera, point):
    loc_camera = obj_camera.matrix_world.to_translation()

    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')

    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()

# Test
obj_camera = bpy.data.objects["Camera"]

M = Matrix().to_4x4()
gl2cv = Matrix().to_4x4()
gl2cv[1][1] = -1
gl2cv[2][2] = -1

# It works
M[0][0] = 0.9848077514785853
M[0][1] = -0.1736481863645372
M[0][2] = 0
M[0][3] = 0

M[1][0] = -0.07883463815756199
M[1][1] = -0.4470934270490684
M[1][2] = -0.8910064911750564
M[1][3] = 0

M[2][0] = 0.1547216612315784
M[2][1] = 0.8774700991269312
M[2][2] = -0.4539905645318131
M[2][3] = 2

print("M:\n",M)
print("M_inv:\n",M.inverted())

obj_camera.matrix_world = (gl2cv*M).inverted()
