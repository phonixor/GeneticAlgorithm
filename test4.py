#
#
# stolen from: http://stackoverflow.com/questions/6697259/interactive-matplotlib-plot-with-two-sliders
#

import pylab
import scipy
from matplotlib.widgets import Slider, Button, RadioButtons

##ax = pylab.subplot(111)
pylab.subplots_adjust(left=0.25, bottom=0.25)
t = scipy.arange(0.0, 1.0, 0.001)
a0 = 5
f0 = 3
s = a0*scipy.sin(2*scipy.pi*f0*t)
l, = pylab.plot(t,s, lw=2, color='red')
##pylab.axis([0, 1, -10, 10])

axcolor = 'lightgoldenrodyellow'
axfreq = pylab.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
axamp  = pylab.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)

sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=f0)
samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)

def update(val):
    amp = samp.val
    freq = sfreq.val
    l.set_ydata(amp*scipy.sin(2*scipy.pi*freq*t))
    pylab.draw()
sfreq.on_changed(update)
samp.on_changed(update)

resetax = pylab.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
def reset(event):
    sfreq.reset()
    samp.reset()
button.on_clicked(reset)

rax = pylab.axes([0.025, 0.5, 0.15, 0.15], axisbg=axcolor)
radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)
def colorfunc(label):
    l.set_color(label)
    pylab.draw()
radio.on_clicked(colorfunc)

pylab.show()