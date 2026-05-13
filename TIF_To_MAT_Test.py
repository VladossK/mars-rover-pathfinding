from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import heapq
import math
import csv
from random import randrange

# Read the TIFF file

def read_tiff(file_path):
    dataset = gdal.Open(file_path)
    elevation = dataset.ReadAsArray()
    return elevation

# Convert elevation data to a matrix

def tiff_to_matrix(file_path):
    elevation = read_tiff(file_path)
    return elevation

# Plot the matrix as a topographical map

def plot_topographical_map(matrix):
    # If the matrix has 3 dimensions, assume it's RGB data and convert it to grayscale
    if len(matrix.shape) == 3:
        matrix = np.mean(matrix, axis=0)

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='terrain')
    plt.colorbar(label='Elevation (m)')
    plt.title('Topographical Map')
    plt.xlabel('Pixels')
    plt.ylabel('Pixels')
    plt.show()


# Tiff file name

if __name__ == "__main__":
    tiff_file = "Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2_Test2.tif"
    elevation_matrix = tiff_to_matrix(tiff_file)
    plot_topographical_map(elevation_matrix)

filename = "Path_Data.csv"
with open(filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    
    # Write headers only once
    csv_writer.writerow(['Iteration', 'Start Point', 'End Point', 'Path Length'])

    # Dijktras Algorythm 
    for i in range(200):
        def dijkstra(elevation_matrix, start, end):
            rows = len(elevation_matrix)
            cols = len(elevation_matrix[0])
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]  # Including diagonal movements
        
            # Initialize distances with infinity and start vertex with 0
            distances = [[float('inf')] * cols for _ in range(rows)]
            distances[start[0]][start[1]] = 0
        
            # Initialize previous nodes to reconstruct the path
            previous = [[None] * cols for _ in range(rows)]
        
            # Priority queue to store vertices with their current minimum distance from the start
            pq = [(0, start)]
        
            while pq:
                dist, curr = heapq.heappop(pq)
        
                if curr == end:
                    # Reconstruct the path
                    path = []
                    while curr is not None:
                        path.append(curr)
                        curr = previous[curr[0]][curr[1]]
                    return path[::-1]  # Reverse the path to start from the start point
        
                for dx, dy in directions:
                    nx, ny = curr[0] + dx, curr[1] + dy
        
                    # Check if the next value is within bounds
                    if 0 <= nx < rows and 0 <= ny < cols:
                        # Check if the next value is suitable for use
                        if elevation_matrix[nx][ny] >= elevation_matrix[curr[0]][curr[1]]:
                            altitude_diff = elevation_matrix[nx][ny] - elevation_matrix[curr[0]][curr[1]]
                        else:
                            altitude_diff = elevation_matrix[curr[0]][curr[1]] - elevation_matrix[nx][ny]
                        angle = math.degrees(math.atan(altitude_diff / 200))
                        #print(elevation_matrix[nx][ny], elevation_matrix[curr[0]][curr[1]], altitude_diff, angle)
                        if angle <= 15: #Slope angle
                            alt_diff = abs(elevation_matrix[curr[0]][curr[1]] - elevation_matrix[nx][ny])
                            new_dist = dist + alt_diff + 1  # Adding 1 for moving to adjacent cell
        
                            if new_dist < distances[nx][ny]:
                                distances[nx][ny] = new_dist
                                previous[nx][ny] = curr  # Update the previous node
                                heapq.heappush(pq, (new_dist, (nx, ny)))
        
            return None  # No path found
        
        #Start points
        
        start_point = (randrange(1, 1999), randrange(1, 1999))  
        end_point = (randrange(1, 1999), randrange(1, 1999))   
        #print(start_point, end_point)
        
        
        path = dijkstra(elevation_matrix, start_point, end_point)
        path_length = sum(math.sqrt((path[i][0] - path[i + 1][0]) ** 2 + (path[i][1] - path[i + 1][1]) ** 2) for i in range(len(path) - 1))
    
        # Write data to CSV file within the loop
        csv_writer.writerow([i+1, start_point, end_point, path_length])
        
        print("Iteration", i+1)
        print(start_point, end_point)
        print(path_length)

print("Done")