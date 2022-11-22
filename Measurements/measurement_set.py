from measurements import get_all_measurements
import numpy as np
from numpy import array


class MeasurementSet:
    def __init__(self, dir_path, measurements):
        self.dir_path = dir_path
        self.measurements = measurements
        self.data = None

    def __repr__(self):
        return str(self.measurements)

    def get_mean_data(self):
        if self.data is None:
            mean_y = []
            for measurement in self.measurements:
                mean_y.append(measurement.get_data_td()[:, 1])
            mean_y = np.sum(mean_y, axis=0)/len(self.measurements)

            self.data = array([self.measurements[0].get_data_td()[:, 0], mean_y])

        return self.data.T


def get_measurement_sets():
    measurements = get_all_measurements()
    # unique measurement sets (files with same folder above)
    values = set(map(lambda measurement: measurement.dir_1above, measurements))

    # compare and group all measurements
    measurement_sets = []
    for dir_1above in values:
        grouped_measurements = [measurement for measurement in measurements if measurement.dir_1above == dir_1above]
        print(len(grouped_measurements), dir_1above)
        measurement_sets.append(MeasurementSet(dir_1above, grouped_measurements))

    return measurement_sets
