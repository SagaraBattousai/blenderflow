import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mpt
import blenderflow as bf

i = bf.stl_model(b"Colon.stl")

# TODO: Fix later mate
trisf = i.get_triangles_as_list()
tris = [[trisf[i], trisf[i + 1], trisf[i + 2]] for i in range(0, len(trisf), 3)]
fig = plt.figure()
ax = mpt.Axes3D(fig,azim=-90, elev=0)
ax.add_collection3d(mpt.art3d.Poly3DCollection(tris))
ax.auto_scale_xyz(trisf,trisf,trisf)
plt.show()
