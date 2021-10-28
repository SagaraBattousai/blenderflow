import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mpt
import blenderflow as bf

def translate(tris, t):
  func = lambda a: a[0][1] > t or a[1][1] > t or a[2][1] > t
  return list(filter(func, tris))

i = bf.stl_model(b"Colon.stl")



# TODO: Fix later mate
# trisf = i.get_triangles_as_list()
s = 60
trisf = i.get_perspective(s)
tris = [[trisf[i], trisf[i + 1], trisf[i + 2]] for i in range(0, len(trisf), 3)]

fig = plt.figure()
#################################################
ax = mpt.Axes3D(fig, auto_add_to_figure=False, azim=-90, elev=0)
fig.add_axes(ax)
# ax.set_axis_off()

fig.canvas.manager.set_window_title(s)

ax.add_collection3d(mpt.art3d.Poly3DCollection(tris))
ax.auto_scale_xyz(trisf,trisf,trisf)
##################################################
plt.show()
