from numpy.linalg import svd
from numpy import dot, array
from sklearn.datasets import load_wine as load
from matplotlib.pyplot import scatter, show, figure
from mpl_toolkits.mplot3d import Axes3D


def pca(data, new_dim):
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    mean_columns = data.mean(axis=0)
    dif = data - mean_columns
    variance_matrix = dot(dif.T, dif) / data.shape[0]
    U, s, _ = svd(variance_matrix)
    percent = sum(s[: new_dim]) / sum(s)
    return dot(data, U[:, :new_dim]), percent


data = load().data
new_data, _ = pca(data, 10)
colors = array(['red', 'blue', 'green'])
scatter(new_data[:,0], new_data[:,1], c=colors[load().target])
show()
print()

fig = figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(new_data[:,0], new_data[:,1], new_data[:,2], c=colors[load().target])
show()