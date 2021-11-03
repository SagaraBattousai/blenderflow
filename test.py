# import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mpt
import blenderflow as bf

m = bf.Model("assets/Colon/Colon.stl")
polys = bf.normalise(m.polys)

fig = plt.figure()
ax = mpt.Axes3D(fig, auto_add_to_figure=False, azim=-90, elev=0)
fig.add_axes(ax)

ax.add_collection3d(mpt.art3d.Poly3DCollection(polys))
pf = polys.flatten()#.reshape(5844 *3, 3)
ax.auto_scale_xyz(pf,pf,pf)
plt.show()
