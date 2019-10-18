#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# {student Ernestas Simutis}
# {student 19960408-T399}
# {student ernestas@kth.se}

from dubins import *
import numpy as np
import math

class AStarPlanner:

    """

    A* grid planning

    Inspired by Atsushi Sakai (https://github.com/AtsushiSakai/PythonRobotics)

    """

    def __init__(self, ox, oy, reso, obsr):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        reso: grid resolution [m]
        rr: robot radius[m]
        """

        self.reso = reso
        self.calc_obstacle_map(ox, oy, obsr)
        self.motion = self.get_motion_model()

    class Node:
        def __init__(self, x, y, cost, pind):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.pind = pind

        def __str__(self):
            return str(self.x) + " x\n" + str(self.y) + " y\n" + str(self.cost) + " cost\n" + str(self.pind)

    def planning(self, sx, sy, gx, gy):
        """
        A star path search

        input:
            sx: start x position [m]
            sy: start y position [m]
            gx: goal x position [m]
            gx: goal x position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        nstart = self.Node(self.calc_xyindex(sx, self.minx),
                           self.calc_xyindex(sy, self.miny), 0.0, -1)
        ngoal = self.Node(self.calc_xyindex(gx, self.minx),
                          self.calc_xyindex(gy, self.miny), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(nstart)] = nstart

        while 1:
            # print(open_set)
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(
                open_set, key=lambda o: open_set[o].cost + self.calc_heuristic(ngoal, open_set[o]))
            current = open_set[c_id]
            # print('CURRENT')
            # print(current)

            if current.x == ngoal.x and current.y == ngoal.y:
                ngoal.pind = current.pind
                ngoal.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # print('EXPANDED')
                # print(node)
                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    # print('not safe')
                    # exit()
                    continue

                if n_id in closed_set:
                    # print('closed')
                    continue

                if n_id not in open_set:
                    # print('discover')
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # print('best path until now')
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(ngoal, closed_set)

        return rx, ry

    def calc_final_path(self, ngoal, closedset):
        # generate final course
        rx, ry = [self.calc_grid_position(ngoal.x, self.minx)], [
            self.calc_grid_position(ngoal.y, self.miny)]
        pind = ngoal.pind
        while pind != -1:
            n = closedset[pind]
            rx.append(self.calc_grid_position(n.x, self.minx))
            ry.append(self.calc_grid_position(n.y, self.miny))
            pind = n.pind

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.sqrt((n1.x - n2.x) ** 2 + (n1.y - n2.y) ** 2)
        return d

    def calc_grid_position(self, index, minp):
        """
        calc grid position

        :param index:
        :param minp:
        :return:
        """
        pos = index * self.reso + minp
        return pos

    def calc_xyindex(self, position, min_pos):
        return round((position - min_pos) / self.reso)

    def calc_grid_index(self, node):
        return (node.y - self.miny) * self.xwidth + (node.x - self.minx)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.minx)
        py = self.calc_grid_position(node.y, self.miny)

        # print(f'px: {px}, py: {py}')
        # print(f'minx: {self.minx}, miny: {self.miny}, maxx: {self.maxx}, maxy: {self.maxy}')

        if px < self.minx:
            return False
        elif py < self.miny:
            return False
        elif px > self.maxx:
            return False
        elif py > self.maxy:
            return False

        # collision check
        if self.obmap[int(node.x)][int(node.y)]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy, obsr):
        self.minx = .0
        self.miny = .0
        self.maxx = 20.0
        self.maxy = 10.0

        self.xwidth = int(round((self.maxx - self.minx) / self.reso)) + 1
        self.ywidth = int(round((self.maxy - self.miny) / self.reso)) + 1

        # obstacle map generation
        self.obmap = [
            # false all
            [False for i in range(self.ywidth)] for i in range(self.xwidth)
        ]

        for ix in range(self.xwidth):
            x = self.calc_grid_position(ix, self.minx)
            for iy in range(self.ywidth):
                y = self.calc_grid_position(iy, self.miny)
                for iox, ioy, ior in zip(ox, oy, obsr):
                    d = math.sqrt((iox - x) ** 2 + (ioy - y) ** 2)
                    st = 0.5 # safe treshold
                    if d < (st + ior): # true if obstacle too close
                        self.obmap[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]
        return motion


def solution(car):
    target_x, target_y = car.xt, car.yt
    min_x, min_y, max_x, max_y = car.xlb, car.ylb, car.xub, car.yub
    obstacles = np.array(car.obs)
    x, y = car.x0, car.y0  # initial
    grid_size = .2  # [m]

    ox, oy, obsr = list(obstacles[:, 0]), list(obstacles[:, 1]),  list(obstacles[:, 2])
    a_star = AStarPlanner(ox, oy, grid_size, obsr)
    rx, ry = a_star.planning(x, y, target_x, target_y)

    theta = t = phi = 0 
    controls, times = [], [0]

    for point_x, point_y in zip(reversed(rx), reversed(ry)):
        delta_x = point_x - x
        delta_y = point_y - y
        dis = math.sqrt(delta_x**2 + delta_y**2)
        while dis > 1:
            delta_x = point_x - x
            delta_y = point_y - y
            dis = math.sqrt(delta_x**2 + delta_y**2)
           
            phi = math.atan2(delta_y, delta_x) - theta
            if phi < -math.pi or phi > math.pi:
                break
            if phi < -math.pi/4:
                phi = -math.pi/4
            elif phi > math.pi/4:
                phi = math.pi/4

            x, y, theta = step(car, x, y, theta, phi)
            controls.append(phi)
            t += 0.01
            times.append(t)

    return controls, times
