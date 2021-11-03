#!/usr/bin/env python3

import numpy as np
import pandas as pd
from math import sqrt
from queue import PriorityQueue
#import matplotlib.pyplot as plt 
import time

def generate_gridworld(length, width, probability):
    grid = np.random.choice([0,1], length * width, p = [1 - probability, probability]).reshape(length, width)
    #grid[0][0] = 0
    #grid[-1][-1] = 0
    return grid

# Calculates h(x) using one of a range of heuristics
def hureisticValue(point1, point2):
    x1,y1=point1
    x2,y2=point2
    return abs(x1 - x2) + abs(y1 - y2)

class Cell:
    def __init__(self, row, col, dim):
        self.row=row
        self.col=col
        self.dim=dim
        self.neighbors=[]
        self.visited=False
        self.blocked=9999
        self.terrain=9999
        self.prob_containing = 1 / (dim**2)
        self.prob_finding = (.5 * self.prob_containing + .8 * self.prob_containing + .2 * self.prob_containing) / 3
		

    def getPos(self):
        return self.col, self.row
    #return a list of neighbors of the current cell
    def findneighbors(self):
        dim = self.dim
        y, x = self.getPos()
        neighbors = [(y2, x2) for x2 in range(x-1, x+2)                     
                              for y2 in range(y-1, y+2)
                              if (-1 < x < dim and
                                  -1 < y < dim and
                                  (x != x2 or y != y2) and
                                  (0 <= x2 < dim) and
                                  (0 <= y2 < dim))]
        return neighbors                    
    def __lt__(self, other):
        return False

#generate knowledge embeded with cell class
def generate_knowledge(grid,revealed = False):
    cell_list = []
    rows = len(grid)
    cols = len(grid[0])
    #calculate initial_prob of each cell
    g_index = np.where(grid !=1)
    initial_prob = 1/len(g_index[0])
    for i in range(rows):
        cell_list.append([])
        for j in range(cols):
            cellOBJ = Cell(i,j,rows)
            if revealed:
                cellOBJ.blocked = grid[i][j] 
            cellOBJ.prob= initial_prob
            cell_list[i].append(cellOBJ)
    return cell_list

def A_star(curr_knowledge, start, end):
    # Initializes the g(x), f(x), and h(x) values for all squares
    g = {(x, y):float("inf") for y, eachRow in enumerate(curr_knowledge) for x, eachcolumn in enumerate(eachRow)}
    g[start] = 0
    f = {(x, y):float("inf") for y, eachRow in enumerate(curr_knowledge) for x, eachcolumn in enumerate(eachRow)}
    f[start] = hureisticValue(start, end)
    h = {(x, y): hureisticValue((x, y), end) for y, eachRow in enumerate(curr_knowledge) for x, eachcolumn in enumerate(eachRow)}
    parent = {}
    visited={start} # it is a set which provide the uniqueness, means it is ensure that not a single cell visit more than onece.
    tiebreaker = 0
	# Creates a priority queue using a Python set, adding start cell and its distance information
    pq = PriorityQueue()
    pq.put((f[start], tiebreaker, start))
    #count cell being processed
    cell_count = 0
    # A* algorithm, based on assignment instructions
    while not pq.empty():
		# Remove the node in the priority queue with the smallest f value
        n = pq.get()
        cell_count += 1
        successors = []
		# curr_pos is a tuple (x, y) where x represents the column the square is in, and y represents the row
        curr_pos = n[2]
        visited.remove(curr_pos)
		# if goal node removed from priority queue, shortest path found
        if curr_pos == end:
            shortest_path = []
            path_pointer = end
            while path_pointer != start:
                shortest_path.append(path_pointer)
                path_pointer = parent[path_pointer]
            shortest_path.append(start)
            shortest_path = shortest_path[::-1]
            return [shortest_path, cell_count]	
		# Determine which neighbors are valid successors
        #add unblocked or unknown cells
        if curr_pos[0] > 0 and curr_knowledge[curr_pos[1]][curr_pos[0] - 1].blocked != 1: # the current node has a neighbor to its left which is unblocked
            left_neighbor = (curr_pos[0] - 1, curr_pos[1])
            if g[left_neighbor] > g[curr_pos] + 1: # if neighbor is undiscovered
                successors.append(left_neighbor)
				
        if curr_pos[0] < len(curr_knowledge[0])  - 1 and curr_knowledge[curr_pos[1]][curr_pos[0] + 1].blocked != 1: # the current node has a neighbor to its right which is unblocked
            right_neighbor = (curr_pos[0] + 1, curr_pos[1])
            if g[right_neighbor] > g[curr_pos] + 1:
                #if neighbor is undiscovered
                successors.append(right_neighbor)
   
        if curr_pos[1] > 0 and curr_knowledge[curr_pos[1] - 1][curr_pos[0]].blocked != 1: # the current node has a neighbor to its top which is unblocked
            top_neighbor = (curr_pos[0], curr_pos[1] - 1)
            if g[top_neighbor] > g[curr_pos] + 1: # if neighbor is undiscovered
                successors.append(top_neighbor)
                
        if curr_pos[1] < len(curr_knowledge) - 1 and curr_knowledge[curr_pos[1] + 1][curr_pos[0]].blocked != 1: # the current node has a neighbor to its bottom which is unblocked
            bottom_neighbor = (curr_pos[0], curr_pos[1] + 1)
            if g[bottom_neighbor] > g[curr_pos] + 1: # if neighbor is undiscovered
                successors.append(bottom_neighbor)
			
        # Update shortest paths and parents for each valid successor and add to priority queue, per assignment instructions
        for successor in successors:
            g[successor] = g[curr_pos] + 1
            parent[successor] = curr_pos
            if successor not in visited:
                tiebreaker += 1
                pq.put((g[successor] + h[successor], -tiebreaker, successor))
                visited.add(successor)
		# if priority queue is empty at any point, then unsolvable
        if pq.empty():
            return False

# Handles processing of Repeated A*, restarting that algorithm if a blocked square is found in the determined shortest path
def algorithmA(grid, start, end, has_four_way_vision):
    # The assumed state of the gridworld at any point in time. For some questions, the current knowledge is unknown at the start
    curr_knowledge = generate_knowledge(grid)
    prob_table = generate_prob_table(grid)
    # If the grid is considered known to the robot, operate on that known grid
	# Else, the robot assumes a completely unblocked gridworld and will have to discover it as it moves
    complete_path = [start]
	# Run A* once on grid as known, returning False if unsolvable
    shortest_path = A_star(curr_knowledge, start, end)
    if not shortest_path:
        return False
    is_broken = False
    cell_count = shortest_path[1]
    while True:
		# Move pointer square by square along path
        for sq in shortest_path[0]:
            x = sq[0]
            y = sq[1]
			# If blocked, rerun A* and restart loop
            if grid[y][x] == 1:
                # If the robot can only see squares in its direction of movement, update its current knowledge of the grid to include this blocked square
                if not has_four_way_vision:
                    curr_knowledge[y][x].blocked = 1
                shortest_path = A_star(curr_knowledge, prev_sq, end)                
                if not shortest_path:
                    return False
                is_broken = True
                cell_count += shortest_path[1]
                break
			# If new square unblocked, update curr_knowledge. Loop will restart and move to next square on presumed shortest path
            else:
                if sq != complete_path[-1]:
                    complete_path.append(sq)
                # If the robot can see in all compass directions, update squares adjacent to its current position
                if has_four_way_vision:
                     if x != 0:
                         curr_knowledge[y][x - 1].blocked = grid[y][x - 1]
                     if x < len(curr_knowledge[0]) - 1:
                         curr_knowledge[y][x + 1].blocked = grid[y][x + 1]
                     if y != 0:
                         curr_knowledge[y - 1][x].blocked = grid[y - 1][x]
                     if y < len(curr_knowledge) - 1:
                         curr_knowledge[y + 1][x].blocked = grid[y + 1][x]
            prev_sq = sq
        if not is_broken:
            break
        is_broken = False
    return [complete_path, cell_count]

"""generate grid with terrains
1:blocked, 2:flat, 3:hill, 4:forest,
5:flat with target, 6:hill with target, 7:forest with target"""
def generate_terrain(dim):
    initial_grid = generate_gridworld(dim,dim,.3)
    solvable = False
    while not solvable:
        x1, y1, x2, y2 = np.random.choice(dim,4)        
        while initial_grid[y1][x1] == 1 or initial_grid[y2][x2] == 1:
            x1, y1, x2, y2 = np.random.choice(dim,4)
        start = (x1,y1)
        end = (x2,y2)
        curr_knowledge = generate_knowledge(initial_grid,revealed = True)
        solvable = A_star(curr_knowledge, start, end)
    initial_grid[y1][x1] = np.random.choice([2,3,4],1)
    initial_grid[y2][x2] = np.random.choice([5,6,7],1)
    g_index = np.where(initial_grid ==0)
    initial_grid[g_index] = np.random.choice([2,3,4],len(g_index[0]))
    print([start,end])    
    print(solvable[0])
    return initial_grid, start
    
def find_max_prob(curr_knowledge,curr_pos,end):
    dim = len(curr_knowledge)
    max_prob = 0
    for y in range(dim):
        for x in range(dim):
            if curr_knowledge[y][x].prob_containing > max_prob:
                max_prob = curr_knowledge[y][x].prob_containing
                end = (x,y)
            #if probability same, then compare their distance
            elif curr_knowledge[y][x].prob_containing == max_prob:
                if hureisticValue(curr_pos,(x,y)) < hureisticValue(curr_pos,end):
                    end = (x,y)
                #if distance same, then random choice
                elif hureisticValue(curr_pos,(x,y)) == hureisticValue(curr_pos,end) and np.random.choice([0,2]) > 1:
                    end = (x,y)
    return end

def examine_sq(cell, grid):
    x, y = cell
    cell_type = grid[y][x]
    # The presence of the target is determined.
    # The agent is not aware of this
    contains_target = (lambda type: True if type >= 5 else False)(cell_type)
    found_target = False
    # Depending on our knowledge of the grid, the possibility of a false negative must be simulated
    if cell_type == 5:
        found_target = np.random.random() < .8
    elif cell_type == 6:
        found_target = np.random.random() < .5
    elif cell_type == 7:
        found_target = np.random.random() < .2
    return found_target, contains_target
	
def update_containing_probs(curr_knowledge, grid, curr_cell, cell_type):
    n = len(curr_knowledge)**2
    x, y = curr_cell
    # If the discovered cell is blocked, that cell clearly cannot contain the target,
    # and the remaining cells now become slightly more likely to contain it
    if cell_type == 1:
        for row in curr_knowledge:
            for cell in row:
                cell.prob_containing /= (1 - (1 / n))
        curr_knowledge[y][x].prob_containing = 0
        curr_knowledge[y][x].blocked = True
    # The cell MAY contain the target
    else:
        # The agent examines the square to see if the target is present
        found_target, contains_target = examine_sq(curr_cell, grid)
		# There are no false positives, so the agent found the target
        if found_target:
            for row in curr_knowledge:
                for cell in row:
                    cell.prob_containing = 0
            curr_knowledge[y][x].prob_containing = 1
            return "found"
        else:
            # The agent did not find the target, so because of the possibility of false negatives,
			# probabilities for all squares must be adjusted accordingly
            if cell_type == 2 or cell_type == 5:
                prob = curr_knowledge[y][x].prob_containing
                for row in curr_knowledge:
                    for cell in row:
                        if cell != curr_knowledge[y][x]:
                            cell.prob_containing /= .2 * prob + (1 - prob)
                curr_knowledge[y][x].prob_containing = .2 * prob / (.2 * prob + (1 - prob))
            elif cell_type == 3 or cell_type == 6:
                prob = curr_knowledge[y][x].prob_containing
                for row in curr_knowledge:
                    for cell in row:
                        if cell != curr_cell:
                            cell.prob_containing /= .5 * prob + (1 - prob)
                curr_knowledge[y][x].prob_containing = .5 * prob / (.5 * prob + (1 - prob))
            else:
                prob = curr_knowledge[y][x].prob_containing
                for row in curr_knowledge:
                    for cell in row:
                        if cell != curr_cell:
                            cell.prob_containing /= .8 * prob + (1 - prob)
                curr_knowledge[y][x].prob_containing = .8 * prob / (.8 * prob + (1 - prob))

    return curr_knowledge
            
def agent_6(grid, start):
    curr_knowledge = generate_knowledge(grid)
    found_target = False
    curr_pos = start
    end = start
    # The agent finds the optimal square to plan toward until it finds the target
    while not found_target:
        # Calculate the most likely square to contain the target, with ties broken by distance and random selection
        end = find_max_prob(curr_knowledge,curr_pos,end)
        shortest_path = A_star(curr_knowledge, curr_pos, end)
        print("curr_pos is {}, end is {}".format(curr_pos, end))
        #print(shortest_path)
		# Traverse the returned shortest path, sensing squares and updating probabilities as the agent moves
        for cell in shortest_path[0]:
            x, y = cell
            curr_knowledge[y][x].terrain = grid[y][x]
            curr_knowledge = update_containing_probs(curr_knowledge, grid, cell, grid[y][x])
            if curr_knowledge == 'found':
                print("Found target at: ({}, {})".format(x,y))
                return cell
            elif grid[y][x] == 1:
                curr_pos = prev_cell
                break
            prev_cell = cell

#test    
grid, start = generate_terrain(5)
print("Full grid:")
print(grid)
print("Start: ")
print(start)
agent_6(grid, start)
