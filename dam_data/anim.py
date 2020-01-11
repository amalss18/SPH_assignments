from mayavi import mlab
import numpy as np
import time
i = 0

file = np.load("dataf" + str(i) + ".npz")
file1 = np.load("datas" + str(i) + ".npz")
file2 = np.load("datafp" + str(i) + ".npz")


coord = file["arr_0"]
coord1 = file1["arr_0"]
coord2 = file2["arr_0"]

# vel = []

# for j in range(coord2.shape[0]):
#     vel.append(np.linalg.norm(coord2[j]))

# fluid = mlab.points3d(coord.T[0], coord.T[1], np.zeros_like(coord.T[0]), vel)
# fluid = mlab.points3d(coord.T[0], coord.T[1], np.zeros_like(coord.T[0]), coord2)
fluid = mlab.points3d(coord.T[0], coord.T[1], np.zeros_like(coord.T[0]), color=(0,0,1))


solid = mlab.points3d(coord1.T[0], coord1.T[1], np.zeros_like(coord1.T[0]))

# solid.glyph.glyph_source.glyph_source.radius = 0.1
# fluid.glyph.glyph_source.glyph_source.radius = 0.1

@mlab.animate(delay=50)
def anim(i=0):
    f = mlab.gcf()
    while i <= 15400:
        file = np.load("dataf" + str(i) + ".npz")
        file1 = np.load("datas" + str(i) + ".npz")
        file2 = np.load("datafp" + str(i) + ".npz")
        coord = file["arr_0"]
        coord1 = file1["arr_0"]

        coord2 = file2["arr_0"]

        # vel = []
        # for j in range(coord2.shape[0]):
        #     vel.append(np.linalg.norm(coord2[j]))

        # fluid.mlab_source.set(x = coord.T[0], y = coord.T[1], z = np.zeros_like(coord.T[0]), scalars= np.array(vel))
        # fluid.mlab_source.set(x = coord.T[0], y = coord.T[1], z = np.zeros_like(coord.T[0]), scalars= np.array(coord2))
        fluid.mlab_source.set(x = coord.T[0], y = coord.T[1], z = np.zeros_like(coord.T[0]))


        solid.mlab_source.set(x = coord1.T[0], y = coord1.T[1], z = np.zeros_like(coord1.T[0]))

        # mlab.scalarbar(fluid)
        solid.glyph.glyph_source.glyph_source.radius = 0.1
        fluid.glyph.glyph_source.glyph_source.radius = 0.2
        yield
        f.scene.render()

        i = i + 50


anim()