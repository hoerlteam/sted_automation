import serial
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class ArduinoTempLogger():

    def __init__(self, outfile, device='/dev/cu.usbmodem621', br=9600, ax=None):
        # init serial connection
        self.ser = serial.Serial(device)
        time_str = b"T" + bytes(str(int(time.time())), 'utf-8')
        sys.stderr.write(str(time_str) + '\n')
        sys.stderr.flush()

        time.sleep(1)

        self.ser.write(time_str)
        self.ser.flush()

        # init logfile
        self.outfd = open(outfile, 'w+')
        self.outfd.write('time\ttemp\n')

        self.times = []
        self.temps = []

        self.ax = ax
        self.nToPlot = 3600;

    def update(self):
        l = self.ser.readline()
        #time_str = b'T' + bytes(str(int(time.time())), 'utf-8')
        #self.ser.write(time_str)
        self.times.append(float(l.strip().split(b';')[0]))
        self.temps.append(float(l.strip().split(b';')[1]))
        self.outfd.write(str(self.times[-1])+'\t'+str(self.temps[-1])+'\n')
        sys.stderr.write('read: '+ str(self.temps[-1]) + '\n')
        sys.stderr.flush()


    def updateAndPlot(self, i):
        self.update()
        xs = self.times[-self.nToPlot if len(self.times) > self.nToPlot else -len(self.times):]
        ys = self.temps[-self.nToPlot if len(self.temps) > self.nToPlot else -len(self.temps):]
        self.ax.clear()
        self.ax.plot(xs, ys)

    def close(self):
        self.outfd.close()
        self.ser.flush()
        self.ser.close()


def main(withPlot = False):
    if withPlot:
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        logger = ArduinoTempLogger(outfile="C:\\Users\\RESOLFT\\Desktop\\log.txt", device='COM14', ax=ax1)
    else:
        logger = ArduinoTempLogger(outfile="/Users/david/Desktop/log.txt")

    time.sleep(1)

    try:
        if withPlot:
            ani = animation.FuncAnimation(fig, logger.updateAndPlot, interval=10)
            plt.show()
        else:
            while True:
                logger.update()
    except KeyboardInterrupt:
        logger.close()
        sys.exit()
    logger.close()
    sys.exit()


if __name__ == '__main__':
    main(True)
