#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:00:59 2025

@author: pyare
"""
#%% 0. importing libraries
import open3d as o3d
#%% 1. reading the point cloud
point_cloud = o3d.io.read_point_cloud('/home/pyare/Pyare/PhD/LiDAR Data/Bhopal/Internship/Dec-24/Training/coding/sample_data/room_furnitures.ply')

o3d.visualization.draw_geometries([point_cloud])


#%% 2. Defining the oct tree
curr_max_depth = 6
octree = o3d.geometry.Octree(max_depth = curr_max_depth)
octree.convert_from_point_cloud(point_cloud)
o3d.visualization.draw_geometries([octree])