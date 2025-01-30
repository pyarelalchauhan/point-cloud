

#%% 0. Importing the libraries
import numpy as np
from scipy.spatial import KDTree
import networkx as nx
import matplotlib.pyplot as plt
import open3d as o3d


#%% 1.  Load point cloud (PLY format)
pcd = o3d.io.read_point_cloud("/home/pyare/Pyare/PhD/LiDAR Data/Bhopal/Internship/Dec-24/Training/coding/sample_data/room_furnitures.ply")
#translation = pcd.get_min_bound()
# pcd.translate(-translation)
# Visualization with Open3d
o3d.visualization.draw_geometries([pcd])

# Simulating the datasets

# sampling generate random 2D points with some clustering
np.random.seed(42)
n_points = 300
xyz = np.random.rand(300,2)

#%% 2. Construction of the graph and pruning function

def build_radius_graph(points, radius, max_neighbors):
    """ Build a graph by connecting points within a specified radius using KD-tree """
    
    # Convert points to numpy array if not already
    points = np.asarray(points)
    
    # Create KD-tree
    kdtree = KDTree(points)
    
    # Initialize graph
    graph = nx.Graph()
    # Add nodes with position attributes
    for i in range(len(points)):
        graph.add_node(i, pos=points[i])
    
    # Query the KD-tree for all points within radius
    pairs = kdtree.query_pairs(radius)
    # Add edges to the graph with distances as weights
    for i, j in pairs:
        dist = np.linalg.norm(points[i] - points[j])
        graph.add_edge(i, j, weight=dist)

    # If max_neighbors is specified, prune the graph
    if max_neighbors is not None:
        prune_to_k_neighbors(graph, max_neighbors)
    
    return graph


def prune_to_k_neighbors(graph, k):
    """ Prune the graph to keep only the k nearest neighbours for each node"""
    for node in graph.nodes():
        edges = [(node, neighbor, graph[node][neighbor]['weight'])
                 for neighbor in graph[node]]
        if len(edges) > k:
            # Sort edges by weight
            edges.sort(key=lambda x: x[2])
            # Remove edges beyond k nearest
            edges_to_remove = edges[k:]
            graph.remove_edges_from([(e[0], e[1]) for e in edges_to_remove])
    
simulation_graph = build_radius_graph(xyz, radius= 0.1, max_neighbors=4)

#%% 3. Retrieving the connected components
def get_connected_components(graph):
    """ Extract connected components from the graph"""
    components = list(nx.connected_components(graph))
    
    return [graph.subgraph(components).copy() for component in components]

#%% 4. Analyse the graph
def analyze_connected_components(graph):
    """
    Analyze connected components in the graph and return a list of clusters.
    """
    components = list(nx.connected_components(graph))
    analysis = {
        'num_components': len(components),
        'component_sizes':[len(c) for c in components],
        'largest_component_size':max(len(c) for c in components),
        'smallest_component_size':min(len(c) for c in components),
        'avg_component_size': np.mean([len(c) for c in components]),
        'isolated_points': sum(1 for c in components if len(c)==1)
        }
   
    return analysis

component_analysis = analyze_connected_components(simulation_graph)

print("\nComponent Analysis:")
for metric, value in component_analysis.items():
    print(f"{metric}:{value}")
    
#%% 5. [ Utility] Ploting Graphs
def plot_components(graph, points, radius, neighbors, figsize =(12,12), cmap ='tab20'):
    """
    Plot connected components with different colors
    """
    dim= points.shape[1]
    if dim not in [2,3]:
        raise ValueError("Plotting only supported for 2D or 3D points")
        
    # Get connected components
    components = list(nx.connected_components(graph))
    n_components = len(components)
    
    # Create color iterator
    colors = plt.colormaps[cmap](np.linspace(0, 1, max(n_components,1)))
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    # Plot each component with different color
    for idx, component in enumerate(components):
        color = colors[idx]
        
        # Get component edges
        component_edges = [(i,j) for i, j in graph.edges()
                           if i in component and j in component]
        
        # Plot Edges
        for edge in component_edges:
            edge_points = np.array([points[edge[0]], points[edge[1]]])
            if dim == 3:
                ax.plot(edge_points[:,0], edge_points[:,1], edge_points[:,2], color = color,alpha=0.5)
            else:
                ax.plot(edge_points[:,0], edge_points[:,1],
                        color=color, alpha=0.5)
                
        # plot points
        component_points = points[list(component)]
        if dim == 3:
            ax.scatter(component_points[:,0], component_points[:,1],
                       component_points[:,2], color = color, s = 50)
        else:
            ax.scatter(component_points[:,0], component_points[:,1],
                       color=color, s = 50)
            
    plt.title(f'Graph with {len(components)} connected components\n'
              f'{len(graph.nodes())} nodes and {len(graph.edges())} edges for radius of {radius} and {neighbors} neighbors')
    
    return fig, ax


#%% 6. Simulating datasets
#%% 6.1 Changing the radius
for radius_simulation in [0.05, 0.1, 0.5]:
    simulation_graph = build_radius_graph(xyz, radius=radius_simulation, max_neighbors=5)
    plot_components(simulation_graph, xyz, radius_simulation, 5)
    plt.show()
#%% 6.2 Changing the neighboring counts
for neighbors_simulation in [2,5,10]:
    simulation_graph = build_radius_graph(xyz, radius=0.1, max_neighbors= neighbors_simulation)
    plot_components(simulation_graph, xyz, 0.1, neighbors_simulation)
    plt.show()

#%% 7. Building the real-world graph

# Visualizing our input point cloud
o3d.visualization.draw_geometries([pcd])

# Get point cloud coordinates
xyz = np.asarray(pcd.points)
nn_d = np.mean(pcd.compute_nearest_neighbor_distance()[0])

# Build the Graph
graph = build_radius_graph(xyz, radius=nn_d*3, max_neighbors=10)

# Analyze the components
component_analysis = analyze_connected_components(graph)
print("\nComponent Analysis:")
for metric, value in component_analysis.items():
    print(f"{metric}:{value}")
    
#%% 8. Plotting our result point cloud
def plot_cc_o3d(graph, points, cmap='tab20'):
    """
    Plot connected components with different colours
    """
    # get connected components 
    components = list(nx.connected_components(graph))
    n_components = len(components)
    
    # Create a Color iterator
    colors = plt.colormaps[cmap](np.linspace(0, 1,max(20,1)))
    colors = np.vstack([colors,colors,colors,colors,colors,colors,colors,colors])
    rgb_cluster = np.zeros(np.shape(points))
    
    # plot each components with a different color
    for idx, component in enumerate(components):
        if len(component)<= 10:
            rgb_cluster[list(component)]= [0,0,0]
            idx-=1
            
        else:
            color = colors[idx][:3]
            rgb_cluster[list(component)]= color
            
    pcd_clusters = o3d.geometry.PointCloud()
    pcd_clusters.points = o3d.utility.Vector3dVector(xyz)
    pcd_clusters.colors = o3d.utility.Vector3dVector(rgb_cluster)
    return pcd_clusters

pcd_cluster = plot_cc_o3d(graph, xyz)
pcd_cluster.estimate_normals()
o3d.visualization.draw_geometries([pcd_cluster])