#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      phonixor
#
# Created:     03-07-2013
# Copyright:   (c) phonixor 2013
# Licence:     <your licence>
#-------------------------------------------------------------------------------
#!/usr/bin/env python

def main():
    pass

if __name__ == '__main__':
    import prettyHierarchy
    import scipy

    data=scipy.array([[1,2,3,4],[1,3,3,4]])
    prettyHierarchy.prettyHierachy(data)



    from pylab import *
    cdict = {'red': ((0.0, 0.0, 1.0),
                     (0.5, 0.0, 0.0),
                     (1.0, 0.0, 0.0)),
             'green': ((0.0, 0.0, 0.0),
                       (0.5, 0.0, 0.0),
                       (1.0, 1.0, 0.0)),
             'blue': ((0.0, 0.0, 0.0),
                      (0.5, 0.0, 0.0),
                     (1.0, 0.0, 0.0))}
    my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    pcolor(rand(10,10),cmap=my_cmap)
    colorbar()
    show()