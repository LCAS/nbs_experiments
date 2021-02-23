import yaml
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D


#1) Load topological map
path = '/home/pulver/ros_workspaces/rasberry_ws/src/RASberry/rasberry_navigation/maps/riseholme/riseholme_poly_act_rfid_sim.yaml'
dict = {}
stream = open(path, 'r')
data = yaml.load(stream)
index = 0
for element in data:
    name_wp = element['node']['name']
    dict[name_wp] = index
    index += 1

param_list = [x for x in range(0, index)]

# def build_matrix(input_array, max_value):
#     max_index = np.max(input_array[:,0])
#     matrix_size = int(max_index*100)
#     matrix = np.ones(shape=(matrix_size, matrix_size))
#     matrix *= max_value

#     for w_info_gain in range(0, matrix_size):
#         for w_travel_distance in range(matrix_size, 0, -1):
#             # print("\n[{},{}]".format(w_info_gain/100.0, w_travel_distance/100.0))
#             if (w_info_gain + w_travel_distance == matrix_size):
#                 index_arr = np.where(input_array[:,0] == (w_info_gain/100.0))
#                 # print(index_arr[0][0])
#                 value = input_array[index_arr, 2][0][0]
#                 # print(value)
#                 matrix[w_info_gain, w_travel_distance-1] = value

#     return matrix


def sortData(input_array):
    input_array = input_array[input_array[:, 0].argsort()]
    num_mini_batch = int(input_array.shape[0] / len(param_list)) + 1
    for i in range(0, num_mini_batch):
        # Create the minimatch to sort
        mini_batch = input_array[i*len(param_list):len(param_list)+i*len(param_list)]
        # Sort it first based on the first column and the on the second
        mini_batch = mini_batch[mini_batch[:, 1].argsort()]
        # mini_batch = mini_batch[mini_batch[:, 1].argsort()
        # Update the original array
        input_array[i*len(param_list):len(param_list)+i*len(param_list)] = mini_batch 
        np.savetxt('/tmp/sorted_' + str(i) + '.txt', mini_batch)
    np.savetxt('/tmp/sorted_array.txt', input_array)
    return input_array

def mjrFormatter(x, pos):
    return "${{{0}}}$".format(param_list[pos])

def mjrFormatter_no_TeX(x, pos):
    return "2^{0}".format(x)

mcdm_r1 = np.genfromtxt('/home/pulver/Desktop/topoNBS/nbs/only_rfid/17-02-2021-17-41-07/pf_vs_gt_0.csv', skip_header=True, delimiter=',', dtype=str)  
mcdm_r2 = np.genfromtxt('/home/pulver/Desktop/topoNBS/nbs/only_rfid/17-02-2021-18-01-01/pf_vs_gt_0.csv', skip_header=True, delimiter=',', dtype=str)  
mcdm_r3 = np.genfromtxt('/home/pulver/Desktop/topoNBS/nbs/only_rfid/17-02-2021-18-02-03/pf_vs_gt_0.csv', skip_header=True, delimiter=',', dtype=str)  
mcdm_r4 = np.genfromtxt('/home/pulver/Desktop/topoNBS/nbs/only_rfid/17-02-2021-18-21-12/pf_vs_gt_0.csv', skip_header=True, delimiter=',', dtype=str)  
mcdm_r5 = np.genfromtxt('/home/pulver/Desktop/topoNBS/nbs/only_rfid/17-02-2021-18-22-13/pf_vs_gt_0.csv', skip_header=True, delimiter=',', dtype=str)  

mcdm_list = [mcdm_r1, mcdm_r2, mcdm_r3, mcdm_r4, mcdm_r5]

final_mcdm_matrix = np.zeros(shape=(len(param_list), len(param_list)))

counter = 0

mcdm_mean_list = []
mcdm_best_list = []
# print(dict.keys)
max_value = 0
for mcdm in mcdm_list:
    # Keep only [PF_prediction, ground_truth]
    # mcdm = np.delete(mcdm, [2], axis=1)
    # print(mcdm)
    #TODO: create a square matrix in which we set equal to 1 the cell defined in the log
    for i in range(0, mcdm.shape[0]):
        # print(mcdm[i, 0])
        row_index = dict[mcdm[i, 0]]
        col_index = dict[mcdm[i, 1]] 
        final_mcdm_matrix[row_index][col_index] += 1
        if (final_mcdm_matrix[row_index][col_index] > max_value):
            max_value = final_mcdm_matrix[row_index][col_index]
    # print("Matrix: ", mcdm_matrix.shape)
    # Reshape into a 2D matrix
    # mcdm_matrix = np.reshape(mcdm_matrix, (len(param_list), len(param_list) ))
    # print("Reshaped: ", mcdm_matrix.shape)
    # final_mcdm_matrix = final_mcdm_matrix +  mcdm_matrix
    
    # np.savetxt('/tmp/clean_mcdm' + str(counter) + '.txt', mcdm_matrix)
    # counter += 1
# exit(0)

print("Max value: " , max_value)


# Normalize the matrix
# final_mcdm_matrix /= len(mcdm_list)



np.savetxt('/tmp/clean_mcdm.txt', final_mcdm_matrix)


# Print some statistics data
# print("[mcdm] Best mean: ", np.min(mcdm_mean_list))
# print("[mcdm] mean[std]: {}[{}]".format(np.mean(mcdm_mean_list), np.std(mcdm_mean_list)))
# print("[mcdm] Top10 - mean[std]: {}[{}]".format(np.mean(mcdm_best_list), np.std(mcdm_best_list)))


# Plot the matrix
fig = plt.figure()
ax2 = fig.add_subplot(1,1,1)
ax2.set_aspect('equal')
plt.imshow(final_mcdm_matrix, interpolation='nearest', cmap=plt.cm.ocean)
plt.colorbar()
plt.ylabel("Waypoint")
plt.xlabel("Waypoint")
plt.title("Confusion Matrix over waypoint prediction")
# plt.yticks(np.arange(0, len(param_list), 1.0))
# ax2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter))
# plt.xticks(np.arange(0, len(param_list), 1.0))
# ax2.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter))
plt.show()
