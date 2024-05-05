""" LidarPredictor.py

This file contains the implementation of the perceptions predictions algorithm
that is solely dependent on raw LiDAR point clouds.
"""

import cProfile

import numpy as np
np.set_printoptions(threshold=np.inf)

# interface
from perc22a.predictors.interface.PredictorInterface import Predictor

# data datatypes
from perc22a.data.utils.DataInstance import DataInstance
from perc22a.data.utils.DataType import DataType

# predict output datatype
from perc22a.predictors.utils.cones import Cones
from perc22a.predictors.utils.transform.transform import PoseTransformations

# visualization and core lidar algorithm functions
from perc22a.predictors.utils.vis.Vis3D import Vis3D
import perc22a.predictors.utils.lidar.visualization as vis
import perc22a.predictors.utils.lidar.filter as filter
import perc22a.predictors.utils.lidar.cluster as cluster
import perc22a.predictors.utils.lidar.color as color

# timer utilities
from perc22a.utils.Timer import Timer

# constants
from perc22a.predictors.lidar.constants import *

# general imports
import numpy as np
from typing import List

class LidarPredictor(Predictor):
    def __init__(self):
        # self.window = vis.init_visualizer_window()
        self.sensor_name = "lidar"
        self.transformer = PoseTransformations()
        self.timer = Timer()

        # self.use_old_vis = False 
        # if self.use_old_vis:
        #     self.window = vis.init_visualizer_window()
        # else:
        #     self.vis = Vis3D()

        return

    def profile_predict(self, data):
        profiler = cProfile.Profile()
        profiler.enable()
        ground_removal_time = self.predict(data)
        profiler.disable()
        return ground_removal_time, profiler

    def required_data(self):
        return [DataType.HESAI_POINTCLOUD]

    def _transform_points(self, points):
        points = points[:, :3]
        points = points[:, [1, 0, 2]]
        points[:, 0] = -points[:, 0]

        return points

    def predict(self, data) -> Cones:
        if DEBUG_TIME: self.timer.start("predict")
        if DEBUG_TIME: self.timer.start("\tinit-process")

        # coordinate frame points to perceptions coordinates system
        points = data[DataType.HESAI_POINTCLOUD]
        points = points[~np.any(points == 0, axis=1)]
        points = points[~np.any(np.isnan(points), axis=-1)]
        points = points[:, :3]
        points = self._transform_points(points)
        self.points = points

        # transfer to origin of car
        points = self.transformer.to_origin(self.sensor_name, points, inverse=False)

        if DEBUG_TIME: self.timer.end("\tinit-process")
        if DEBUG_TIME: self.timer.start("\tfilter")
        if DEBUG_TIME: self.timer.start("\t\tfov-range")

        points_ground_plane = filter.fov_range(
            points, 
            fov=180, 
            minradius=0, 
            maxradius=INIT_PC_MAX_RADIUS
        )

        if DEBUG_TIME: self.timer.end("\t\tfov-range")
        if DEBUG_TIME: self.timer.start("\t\tground-removal")

        points_filtered_ground = filter.remove_ground(
            points_ground_plane,
            debug=False
        )
       
        if DEBUG_TIME: to_ret = self.timer.end("\t\tground-removal")
        return to_ret

    def display(self):

        if self.use_old_vis:
            vis.update_visualizer_window(self.window, self.points_cluster_subset, self.cone_output_arr, self.cone_colors)
        else:
            self.vis.set_points(self.points_cluster_subset)
            self.vis.set_cones(self.cones)
            self.vis.update()

        return
