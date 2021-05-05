import numpy as np
from habitat import logger
import matplotlib.pyplot as plt

def SO2Rotation(theta):
    return(np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]]))


def SE2Transformation(p, theta):
    T = np.eye(3)
    R = SO2Rotation(theta)
    T[:2, :2] = R
    T[:2, 2] = np.array(p).flatten()
    return(T)

def checkSquare(index,mat, val):
    for x in range(-1,2):
        for y in range(-1,2):
            if mat[index[0]+x,index[1]+y] == val:
                return(True)
    return(False)

def setSquare(index,mat, val):
    for x in range(-1,2):
        for y in range(-1,2):
            mat[index[0]+x,index[1]+y] = val
    return(mat)



class Verify:
    def __init__(self, cell_size=.125):
        self.cell_size = cell_size
        self.primitive_lib = np.zeros((3, 3, 1))

    def gen_primitive_lib(self, velocities, steers, dt=1):
        velocities /= self.cell_size
        controls = np.meshgrid(velocities, steers)
        controls = np.vstack((controls[0].flatten(), controls[1].flatten())).T
        lib = np.zeros((3, 3, controls.shape[0]))
        for i in range(controls.shape[0]):
            velocity, steer = controls[i, 0], controls[i, 1]
            if steer != 0:
                velocity = 0
            lib[:, :, i] = SE2Transformation(
                np.array([[0, 1]]).T*velocity*dt, steer)
        self.primitive_lib = lib

    def verify_safety(self, infos, T, action, threshold=.3, verbose=False):
        if (self.primitive_lib == 0).all():
            print("Error: Please initialize primitive library")
            return(False)

        if infos:
            # print(infos[0]['top_down_map']['agent_map_coord'],infos[0]['top_down_map']['agent_angle'])
            # for j in range(self.primitive_lib.shape[2]):
            #     print(self.primitive_lib[:,:,j])

            collision_map = infos[0]['top_down_map']['map']
            out_map = np.copy(collision_map)
            branching_factor = self.primitive_lib.shape[2]
            # print("Init Index: ", infos[0]['top_down_map']['agent_map_coord'])
            agent_pose = (-infos[0]['top_down_map']['agent_map_coord'][1],infos[0]['top_down_map']['agent_map_coord'][0])
            currentpose = SE2Transformation(agent_pose , infos[0]['top_down_map']['agent_angle'])
            # print("Current Pose")
            # print(currentpose)
            if action == 1:
                startPose = np.dot(currentpose, self.primitive_lib[:, :, 1])
            elif action == 2:
                startPose = np.dot(currentpose, self.primitive_lib[:, :, 0])
            elif action == 3:
                startPose = np.dot(currentpose, self.primitive_lib[:, :, 2])
            else:
                startPose = currentpose
            # print("Start Pose")
            # print(startPose)
            num_collisions = 0
            open_set = startPose.reshape((3, 3, 1))
            for t in range(T):
                new_open_set = []
                # print("Timestep: ", t)
                for i in range(open_set.shape[2]):
                    for j in range(self.primitive_lib.shape[2]):
                        new_pose = np.dot(
                            open_set[:, :, i], self.primitive_lib[:, :, j])
                        p = new_pose[:2, 2].astype(int).flatten()
                        index = (p[1], -p[0])
                        # print(index)
                        
                        if index[0] < 1 or index[0] >= collision_map.shape[0] - 1 or index[1] < 1 or index[1] >= collision_map.shape[1] - 1:
                            num_collisions += branching_factor**(T-t-1)
                        elif collision_map[index] == 0:
                            num_collisions += branching_factor**(T-t-1)
                            out_map[index] = 7
                            # out_map = setSquare(index, out_map, 7)
                            # print(index)
                            # logger.info(f"in collision")
                        else:
                            new_open_set.append(new_pose)
                            # out_map = setSquare(index, out_map, 10)
                            out_map[index] = 10

                if len(new_open_set) == 0:
                    break
                elif len(new_open_set) == 1:
                    open_set = new_open_set[0].reshape((3, 3, 1))
                else:
                    open_set = np.dstack(new_open_set)
            prob = num_collisions/(branching_factor**T)
            
            # plt.matshow(collision_map)
            # plt.show()
            
            # plt.matshow(out_map)
            # plt.show()
            
            if verbose:
                print("Collision probability: ", prob)
            return(prob < threshold, True, out_map)
        else:
            return(True, False, np.zeros((1,1)))
