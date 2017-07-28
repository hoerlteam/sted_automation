class RichData:
    """
    wrapper for data and settings of a measurement
    holds hardware ('global') settings and per measurement settings
    for each parameter set and a list of associated data
    """

    def __init__(self):
        self.globalSettings = []
        self.measurementSettings = []
        self.data = []

    # TODO: remove defaults?
    def append(self, globalSettings=None, measurementSettings=None, data=None):
        self.globalSettings.append(globalSettings)
        self.measurementSettings.append(measurementSettings)
        self.data.append(data)

    @property
    def numConfigurations(self):
        return len(self.data)

    def numImages(self, n):
        if n < self.numConfigurations:
            return len(self.data[n])
        else:
            return 0