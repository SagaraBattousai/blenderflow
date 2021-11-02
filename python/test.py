import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mpt
import blenderflow as bf

def translate(tris, t):
  func = lambda a: a[0][1] > t or a[1][1] > t or a[2][1] > t
  return list(filter(func, tris))

def as_poly(trisf):
  return [[trisf[i], trisf[i + 1], trisf[i + 2]] for i in range(0, len(trisf), 3)]


i = bf.stl_model(b"Colon.stl")

# TODO: Fix later mate
trisf = i.get_triangles_as_list()

fig = plt.figure()

ax = mpt.Axes3D(fig, auto_add_to_figure=False, azim=-90, elev=0)
fig.add_axes(ax)
# ax.set_axis_off()
tri_arr = np.array(trisf)
print(np.amax(tri_arr))
nt = tri_arr/ np.amax(tri_arr)
print(np.amax(nt))
ax.add_collection3d(mpt.art3d.Poly3DCollection(as_poly(nt)))
ax.auto_scale_xyz(nt,nt,nt)

plt.show()
