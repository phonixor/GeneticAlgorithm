# http://stackoverflow.com/questions/2982929/plotting-results-of-hierarchical-clustering-ontop-of-a-matrix-of-data-in-python

import scipy
import pylab
import scipy.cluster.hierarchy as sch

def prettyHierachy(data):

    # Compute and plot first dendrogram.
    fig = pylab.figure(figsize=(8,8))
    ax1 = fig.add_axes([0.09,0.1,0.2,0.6])
    Y = sch.linkage(data, method='centroid')
    print(Y)
    print(Y.shape)
    Z1 = sch.dendrogram(Y, orientation='right')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Compute and plot second dendrogram.
    ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
    Y = sch.linkage(data, method='single')
    Z2 = sch.dendrogram(Y)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Plot distance matrix.
    axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])

##    print(Z1['leaves'])
##    print(Z2['leaves'])
##    print(data.shape)

    # get the nr's of how the dendogram is ordered
    idx1 = Z1['leaves']
    idx2 = Z2['leaves']


##    print(data)
    data = data[idx1,:] # what do we do here?

##    print(data)
    data = data[:,idx2]

##    print(data.shape)


    im = axmatrix.matshow(data, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu) # http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    # Plot colorbar.
    axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
    pylab.colorbar(im, cax=axcolor)
    ##fig.show()
    ##fig.savefig('dendrogram.png')
    pylab.show()

if __name__ == '__main__':
##    # Generate random features and distance matrix.
##    x = scipy.rand(40)
##    D = scipy.zeros([40,40])
##    for i in range(40):
##        for j in range(40):
##            D[i,j] = abs(x[i] - x[j])
##
##    prettyHierachy(D)
    # Generate random features and distance matrix.
    x = scipy.rand(60)
    D = scipy.zeros([40,60])
    for i in range(40):
        for j in range(60):
            D[i,j] = abs(x[i] - x[j])

    prettyHierachy(D)


