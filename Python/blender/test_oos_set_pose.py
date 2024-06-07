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
#obj_other = bpy.data.objects["Cube"]



M = Matrix().to_4x4()
# above obj
M[0][0] = -1
M[0][1] = 0
M[0][2] = 0
M[0][3] = 0

M[1][0] = 0
M[1][1] = 0
M[1][2] = 1
M[1][3] = 0

M[2][0] = 0
M[2][1] = 1
M[2][2] = 0
M[2][3] = -2

# 0,0 obj
#M[0][0] = 0
#M[0][1] = 1
#M[0][2] = 0
#M[0][3] = 0

#M[1][0] = 0
#M[1][1] = 0
#M[1][2] = 1
#M[1][3] = 0

#M[2][0] = 1
#M[2][1] = 0
#M[2][2] = 0
#M[2][3] = -2

# above ply and obj (fixed)
M[0][0] = -1
M[0][1] = 0
M[0][2] = 0
M[0][3] = 0

M[1][0] = 0
M[1][1] = -1
M[1][2] = 0
M[1][3] = 0

M[2][0] = 0
M[2][1] = 0
M[2][2] = 1
M[2][3] = -2

print("M:\n",M)
print("M_inv:\n",M.inverted())

oos = bpy.data.objects["1"]
print("oos:\n", oos.matrix_world)

obj_camera.matrix_world = M.inverted()
#obj_camera.matrix_world = (M*oos.matrix_world).inverted()


#position = Vector((0.309443, 1.754940, 0.907981, 1))
#position = Vector((0, 2, 0, 1))
#M = Matrix().to_4x4()
#M[1][1] = 0
#M[2][2] = 0
#M[1][2] = -1
#M[2][1] = 1
#print(M)
#position_blender = M*position
#print(position_blender)

#obj_camera.location = (0.309443, 1.754940, 0.907981)
#obj_camera.location = (0.309443, 0.907981, 1.754940)

#obj_camera.location = (position_blender[0], position_blender[1], position_blender[2])

#obj_camera.location = (-0.309443, -0.907981, 1.754940)

#look_at(obj_camera, obj_other.matrix_world.to_translation())
#look_at(obj_camera, Vector((0,0,0)))
