import sys
import os

import numpy as np
from skimage import io
numcomp = 4
data = []
path = sys.argv[1]
for i in range(415):
	img_p = path + '/%s.jpg' % i
	src = io.imread(img_p)
	data.append(src.flatten())
X = np.array(data)
X_mean = np.mean(X, axis=0)
X_med = X - X_mean
U,s,V= np.linalg.svd(X_med.T, full_matrices=False)

reconstruct = []
for im in X_med:
	w = np.dot(im,U[:,:numcomp])
	M = np.dot(U[:,:numcomp],w)
	M = M + X_mean
	M -= np.min(M)
	M /= np.max(M)
	M = (M * 255).astype(np.uint8)
	reconstruct.append(M)
reconstruct = np.array(reconstruct)
idx = int(os.path.splitext(sys.argv[2])[0])
M = reconstruct[idx]
M = M.astype(np.uint8).reshape(600,600,3)
io.imsave('reconstruction.jpg', M)
