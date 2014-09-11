# http://stackoverflow.com/questions/2982929/plotting-results-of-hierarchical-clustering-ontop-of-a-matrix-of-data-in-python
# note that it rapes everything!

import scipy
import pylab
import scipy.cluster.hierarchy as sch
import matplotlib

def prettyHierachy(data, ylabels=None):
    """ """
    # make them nicely indexable
    data=scipy.array(data)
    ylabels=scipy.array(ylabels)
##    print(ylabels)

    fig = pylab.figure()

    # dendogram
    dendAxes= fig.add_axes([0.05,0.05,0.2,0.9])
    Y=sch.linkage(data)
##    Y=sch.linkage(data)
##    dend = sch.dendrogram(Y, orientation='right', labels=ylabels)
    dend = sch.dendrogram(Y, orientation='right')
    dendAxes.set_xticks([]) # remove label thingies
    dendAxes.set_yticks([])


    # sort the data
    idx = dend['leaves'] # get how it is sorted


##    print(len(idx))
##    print(idx)
##    print(len(ylabels))
##    print(data.shape)

    sortedData = data[idx,:]
    sortedYlabels = ylabels[idx]

##    print(len(sortedYlabels))
##    print(sortedYlabels)

    # data grid
    dataMatrix = fig.add_axes([0.255,0.05,0.5,0.9])

    cdict = {'red': ((0.0, 1.0, 1.0),

                     (1.0, 0.0, 0.0)),
             'green': ((0.0, 1.0, 1.0),

                       (1.0, 1.0, 0.0)),
             'blue': ((0.0, 1.0, 1.0),

                     (1.0, 0.0, 0.0))}


    my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    im = dataMatrix.matshow(sortedData, aspect='auto', origin='lower', cmap=my_cmap) # http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps -- other colors: pylab.cm.YlGnBu
    dataMatrix.set_xticks([])
    dataMatrix.yaxis.tick_right()
##    dataMatrix.set_yticks(range(len(sortedYlabels)),range(len(sortedYlabels)))
##    dataMatrix.set_yticks(idx,sortedYlabels)
##    pylab.yticks(idx,sortedYlabels) # works... but wrong!!
##    pylab.yticks(idx,ylabels) # does not sort properly
    pylab.yticks(range(len(sortedYlabels)),sortedYlabels)

    pylab.show()






def correlationMatrix(data, ylabels=None):
    """ """

##    import matplotlib.pyplot as plt

    # make them nicely indexable
    data=scipy.array(data)
    ylabels=scipy.array(ylabels)

    fig = pylab.figure()

    # dendogram
    dendAxes= fig.add_axes([0.05,0.05,0.2,0.9])
    Y=sch.linkage(data) #http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
##    Y=sch.linkage(data)
##    dend = sch.dendrogram(Y, orientation='right', labels=ylabels)
    dend = sch.dendrogram(Y, orientation='right')
    dendAxes.set_xticks([]) # remove label thingies
    dendAxes.set_yticks([])


    # sort the data
    idx = dend['leaves'] # get how it is sorted
    sortedData = data[idx,:]
    sortedYlabels = ylabels[idx]


    # correlation matrix grid
    cm=scipy.corrcoef(sortedData)
    print(cm)


    # http://matplotlib.org/api/axes_api.html
    dataMatrix = fig.add_axes([0.255,0.05,0.5,0.9])

    cdict = {'red':  ((0.0, 1.0, 1.0),
                      (0.5, 0.0, 0.0),
                      (1.0, 0.0, 0.0)),
             'green':((0.0, 0.0, 0.0),
                      (0.5, 0.0, 0.0),
                      (1.0, 1.0, 1.0)),
             'blue': ((0.0, 0.0, 0.0),
                      (0.5, 0.0, 0.0),
                      (1.0, 0.0, 0.0))}


    my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)


    im = dataMatrix.matshow(cm, aspect='auto', origin='lower', cmap=my_cmap) # http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps -- other colors: pylab.cm.YlGnBu
    dataMatrix.set_xticks([])
    dataMatrix.yaxis.tick_right()
    pylab.yticks(range(len(sortedYlabels)),sortedYlabels)



##    fig.colorbar(im,ax=dataMatrix)

##    pylab.pcolor(scipy.rand(10,10),cmap=my_cmap)
##    pylab.matshow(cm)
##    pylab.pcolor(cm,cmap=my_cmap)
##    dataMatrix.pcolor(cm, cmap=my_cmap)
##    dataMatrix.matshow(cm, aspect='auto', origin='lower', cmap=my_cmap)
##    dataMatrix.colorbar()
##    fig.colorbar(my_cmap, ax=dataMatrix, shrink=0.9)

##    origin = 'lower'
##    #origin = 'upper'
##    delta = 0.025
##    x = y = scipy.arange(-3.0, 3.01, delta)
##    X, Y = scipy.meshgrid(x, y)
##    Z1 = pylab.mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
##    Z2 = pylab.mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
##    Z = 10 * (Z1 - Z2)
##    nr, nc = Z.shape
##    levels = [-1.5, -1, -0.5, 0, 0.5, 1]
##    # Illustrate all 4 possible "extend" settings:
##    extends = ["neither", "both", "min", "max"]
##    cmap = pylab.cm.get_cmap("winter")
##    cmap.set_under("magenta")
##    cmap.set_over("yellow")
##    # Note: contouring simply excludes masked or nan regions, so
##    # instead of using the "bad" colormap value for them, it draws
##    # nothing at all in them.  Therefore the following would have
##    # no effect:
##    #cmap.set_bad("red")
##
##    fig, axs = pylab.subplots(2,2)
##    for ax, extend in zip(axs.ravel(), extends):
##        cs = ax.contourf(X, Y, Z, levels, cmap=cmap, extend=extend, origin=origin)
##        fig.colorbar(cs, ax=ax, shrink=0.9)
##        ax.set_title("extend = %s" % extend)
##        ax.locator_params(nbins=4)





    pylab.show()




##    # dendogram
##    dendAxes= fig.add_axes([0.05,0.05,0.2,0.9])
##    Y=sch.linkage(data)
##    print(Y)
####    Y=sch.linkage(data)
####    dend = sch.dendrogram(Y, orientation='right', labels=ylabels)
##    dend = sch.dendrogram(Y, orientation='right')
##    dendAxes.set_xticks([]) # remove label thingies
##    dendAxes.set_yticks([])





##
##
##    # get the nr's of how the dendogram is ordered
##    idx1 = Z1['leaves']
##    idx2 = Z2['leaves']
##
##
##
##
##    #sort the data
##    sortedData = data[idx1,:]
##    sortedData = sortedData[:,idx2]
##
##
##
##    # data grid
##    dataMatrix = fig.add_axes([0.255,0.05,0.5,0.9])
##
##    cdict = {'red': ((0.0, 1.0, 1.0),
##
##                     (1.0, 0.0, 0.0)),
##             'green': ((0.0, 1.0, 1.0),
##
##                       (1.0, 1.0, 0.0)),
##             'blue': ((0.0, 1.0, 1.0),
##
##                     (1.0, 0.0, 0.0))}
##
##
##    my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)
##    im = dataMatrix.matshow(sortedData, aspect='auto', origin='lower', cmap=my_cmap) # http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps -- other colors: pylab.cm.YlGnBu
##    dataMatrix.set_xticks([])
##    dataMatrix.yaxis.tick_right()








##
##    # Generate random features and distance matrix.
##    x = scipy.rand(40)
##    D = scipy.zeros([40,40])
##    for i in range(40):
##        for j in range(40):
##            D[i,j] = abs(x[i] - x[j])
##
##    # Compute and plot first dendrogram.
##    fig = pylab.figure(figsize=(8,8))
##    ax1 = fig.add_axes([0.09,0.1,0.2,0.6])
##    Y = sch.linkage(D, method='centroid')
##    Z1 = sch.dendrogram(Y, orientation='right')
##    ax1.set_xticks([])
##    ax1.set_yticks([])
##
##    # Compute and plot second dendrogram.
##    ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
##    Y = sch.linkage(D, method='single')
##    Z2 = sch.dendrogram(Y)
##    ax2.set_xticks([])
##    ax2.set_yticks([])
##
##    # Plot distance matrix.
##    axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
##    idx1 = Z1['leaves']
##    idx2 = Z2['leaves']
##    D = D[idx1,:]
##    D = D[:,idx2]
##    im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)
##    axmatrix.set_xticks([])
##    axmatrix.set_yticks([])
##
##    # Plot colorbar.
##    axcolor = fig.add_axes([0.91,0.1,0.02,0.8])
##    pylab.colorbar(im, cax=axcolor)
##
##    # Display and save figure.
##    fig.show()


##
##
##def backup(): # no clue what this did again :P
##    data=scipy.array(data)
##
##    # Compute and plot first dendrogram (left/right).
##    fig = pylab.figure(figsize=(8,8))
##    ax1 = fig.add_axes([0.09,0.1,0.2,0.6])
##    X = sch.linkage(data)
####    Z1 = sch.dendrogram(X, orientation='right')
####    print(ylabels)
##    Z1 = sch.dendrogram(X, orientation='right', labels=xlabels)
##
##
####    Z1=sch.dendrogram(X, color_threshold=1, truncate_mode='lastp', labels=xlabels, distance_sort='descending')
####    Z1=sch.dendrogram(X, color_threshold=1, truncate_mode='lastp', orientation='right', labels=xlabels, distance_sort='descending')
##
##
##
##
##    ax1.set_xticks([])
##    ax1.set_yticks([])
##
##    # Compute and plot second dendrogram (up/down).
##    ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
##    Y = sch.linkage(scipy.transpose(data))
####    Z2 = sch.dendrogram(Y)
####    print(ylabels)
##    Z2 = sch.dendrogram(Y, labels=ylabels)
##    print(Z2)
##
##    ax2.set_xticks([])
##    ax2.set_yticks([])
##
##    # Plot distance matrix.
##    axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
##
##    # get the nr's of how the dendogram is ordered
##    idx1 = Z1['leaves']
##    idx2 = Z2['leaves']
##
##
##    #sort the data
##    sortedData = data[idx1,:]
##    sortedData = sortedData[:,idx2]
##
##
##    #create color map
####    cdict = {'red': ((0.0, 0.0, 1.0),
####                     (0.5, 0.0, 0.0),
####                     (1.0, 0.0, 0.0)),
####             'green': ((0.0, 0.0, 0.0),
####                       (0.5, 0.0, 0.0),
####                       (1.0, 1.0, 0.0)),
####             'blue': ((0.0, 0.0, 0.0),
####                      (0.5, 0.0, 0.0),
####                     (1.0, 0.0, 0.0))}
##    cdict = {'red': ((0.0, 1.0, 1.0),
##
##                     (1.0, 0.0, 0.0)),
##             'green': ((0.0, 1.0, 1.0),
##
##                       (1.0, 1.0, 0.0)),
##             'blue': ((0.0, 1.0, 1.0),
##
##                     (1.0, 0.0, 0.0))}
##
##    my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)
##
##
##
##    im = axmatrix.matshow(sortedData, aspect='auto', origin='lower', cmap=my_cmap) # http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps -- other colors: pylab.cm.YlGnBu
##    axmatrix.set_xticks([])
##    axmatrix.set_yticks([])
##
##    # Plot colorbar.
##    axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
##    pylab.colorbar(im, cax=axcolor)
##    ##fig.show()
##    ##fig.savefig('dendrogram.png')
##    pylab.show()
##
##
##
##




if __name__ == '__main__':
##    # Generate random features and distance matrix.
##    x = scipy.rand(40)
##    D = scipy.zeros([40,40])
##    for i in range(40):
##        for j in range(40):
##            D[i,j] = abs(x[i] - x[j])
##
##    prettyHierachy(D)
##    # Generate random features and distance matrix.
##    x = scipy.rand(60)
##    D = scipy.zeros([40,60])
##    for i in range(40):
##        for j in range(60):
##            D[i,j] = abs(x[i] - x[j])
##
##    prettyHierachy(D,ylabels=str(range(D.shape[0])))
##
##
    import initial_stuff

##    correlationMatrix([1,2,3],['test','test2','test3'])


