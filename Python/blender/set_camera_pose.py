import bpy
from mathutils import *

obj_camera = bpy.data.objects["Camera"]

M = Matrix().to_4x4()
gl2cv = Matrix().to_4x4()
gl2cv[1][1] = -1
gl2cv[2][2] = -1

M[0][0] = 1
M[0][1] = 0
M[0][2] = 0
M[0][3] = 0

M[1][0] = 0
M[1][1] = 0
M[1][2] = -1
M[1][3] = 0

M[2][0] = 0
M[2][1] = 1
M[2][2] = 0
M[2][3] = 2

print("M:\n",M)
print("M_inv:\n",M.inverted())

obj_camera.matrix_world = (gl2cv*M).inverted()
