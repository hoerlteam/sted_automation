import os

dummy_measurements = [os.path.join(__file__.rsplit(os.sep, 1)[0], 'dummy_{}channel.msr'.format(i)) for i in range(5)]