import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation
from matplotlib.patches import Polygon
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.stats import pareto
import math


class SubplotAnimation(animation.TimedAnimation):

    def __init__(self, seed, speed, nr_samples, interval):
        np.random.seed(seed)
        b = 3 
        self.samples = (np.random.pareto(b, nr_samples) + 1)
        mean, var, skew, kurt = pareto.stats(b, moments='mvsk')
        self.gt_mean = mean

        self.y_values = []
        self.confidence = []
        self.x_values = range(2, nr_samples, interval)
        for i in self.x_values:
            s = self.samples[:i]
            self.y_values.append(np.mean(s))
            self.confidence.append((np.std(s) / math.sqrt(len(s))) * 1.96)

        self.y_values = np.array(self.y_values)
        self.confidence = np.array(self.confidence)

        fig = plt.figure(figsize=(10,10))
        self.ax1 = fig.add_subplot(2, 2, (1,2))
        self.ax2 = fig.add_subplot(2, 2, 3)
        self.ax3 = fig.add_subplot(2, 2, 4)

        # history plot
        self.ax1.set_title('dancing bar history')
        self.ax1.set_xlabel('iteration')
        self.ax1.set_ylabel('estimated mean')
        self.ax1.set_xlim(0, nr_samples)
        self.ax1.set_ylim(np.min(self.y_values-self.confidence), np.max(self.y_values+self.confidence))

        self.ax1_primitives = []
        p = Polygon(self._history_polygon_xy(1), True, alpha=0.4, color='blue')
        self.ax1_primitives.append(p)
        self.ax1.add_patch(p)

        l = Line2D([], [], color='blue')
        self.ax1_primitives.append(l)
        self.ax1.add_line(l)

        self.ax1.axhline(y=mean, color='black', linestyle='--', linewidth=0.5)


         # bar plot
        self.ax2.set_title('dancing bar')
        self.ax2.set_ylabel('avg sales')
        self.ax2.set_xlim(-0.5, 1)
        self.ax2.set_xticks([0.25])
        self.ax2.set_xticklabels(['department XYZ'])
        self.ax2.set_ylim(0, np.max(self.y_values+self.confidence))

        self.ax2_primitives = []
        r = Rectangle((0,0), 0.5, self.y_values[1], alpha=0.4, color='blue')
        self.ax2_primitives.append(r)
        self.ax2.add_patch(r)

        self.ax2.axhline(y=mean, color='black', linestyle='--', linewidth=0.5)

        l = Line2D([0.25, 0.25], [self.y_values[1]-self.confidence[1], self.y_values[1]+self.confidence[1]], color='black')
        self.ax2_primitives.append(l)
        self.ax2.add_line(l)


        # pdf plot
        self.ax3.set_title('pareto pdf')
        x = np.linspace(pareto.ppf(0.01, b), pareto.ppf(0.99, b), 100)
        self.ax3.plot(x, pareto.pdf(x, b) + 1, 'blue', lw=1, alpha=0.6)


        animation.TimedAnimation.__init__(self, fig, interval=speed, blit=True, repeat=False)

    def _history_polygon_xy(self, i):
        x = np.reshape(np.array(self.x_values[:i]), (-1, 1))
        y = np.reshape(np.array(self.y_values[:i]), (-1, 1))
        ci = np.reshape(np.array(self.confidence[:i]), (-1, 1))
        xy_top = np.hstack((x, y + ci))
        xy_bot = np.hstack((np.flipud(x), np.flipud(y) - np.flipud(ci)))
        xy = np.vstack((xy_top, xy_bot))

        return xy

    def _draw_frame(self, framedata):
        i = framedata
        
        # history plot
        self.ax1_primitives[0].set_xy(self._history_polygon_xy(i))
        self.ax1_primitives[1].set_data(self.x_values[:i], self.y_values[:i])

        # bar plot
        self.ax2_primitives[0].set_height(self.y_values[i])
        self.ax2_primitives[1].set_data([0.25, 0.25], [self.y_values[i]-self.confidence[i], self.y_values[i]+self.confidence[i]])

        #dd = self.ax1.fill_between(x, y - ci, y + ci, alpha=0.2)

    def new_frame_seq(self):
        return iter(range(1, len(self.x_values)))


if __name__== "__main__":
    
    seed = int(sys.argv[1])
    speed = max(50, int(sys.argv[2]))
    ani = SubplotAnimation(seed, speed, 1000, 5)
    ani.save('example.mp4')
    #plt.show()

