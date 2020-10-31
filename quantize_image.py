import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# %matplotlib inline

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from scipy.stats import multivariate_normal

import imageio
import cv2

# write your code here

def get_quantised_image_k(model):
    model.fit(X)
    cluster_assignments = model.predict(X)
    X_quant = np.array([model.cluster_centers_[c] for c in cluster_assignments])

    return X_quant.reshape((w, h, 3)).astype(np.uint8)


def get_quantised_image_em(model):
    model.fit(X)
    cluster_assignments = model.predict(X)
    X_quant = np.array([model.means_[c] for c in cluster_assignments])

    return X_quant.reshape((w, h, 3)).astype(np.uint8)

my_image = "test1.jpg"

file_name = "images/" + my_image
image = np.array(imageio.imread(file_name))
#
# plt.figure(figsize=(10,10))
# cv2.imshow("Original",image)
# cv2.waitKey(5)

w, h, _ = image.shape
X = image.reshape((w * h, 3))
k_means = KMeans(25)

gmm = GaussianMixture(n_components=20, max_iter=10, warm_start=True, init_params='random', tol=1e-8)
# print("Starting quant K...")
# new_image_k = get_quantised_image_k(k_means)
# print("End quant K...")
print("Starting quant EM...")
new_image_em = get_quantised_image_em(gmm)
print("End quant EM...")


# cv2.imshow("K means",new_image_k)
# cv2.waitKey(0)

cv2.imshow("EM alg",new_image_em)
cv2.waitKey(10)
