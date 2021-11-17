#!/usr/bin/env python3

import numpy as np
from queue import PriorityQueue
import random
import matplotlib.pyplot as plt 

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
        self.b=0
        self.e=0
        self.h=0
        self.area_prob = 0
        self.n=len(self.neighbors)
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
    
    def sensing(self, knowledge):
        neighbors = knowledge[self.row][self.col].findneighbors()
        b = 0
        e = 0
        h = 0
        #area_prob = self.prob_finding
        for cell in neighbors:
            x, y = cell
            if knowledge[y][x].blocked == 1:
                b += 1
            elif knowledge[y][x].blocked == 0:
                e += 1
            elif knowledge[y][x].blocked == 9999:
                h += 1
        self.b = b
        self.e = e
        self.h = h
        self.n = len(neighbors)
                  
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
    
def find_max_prob(curr_knowledge, curr_pos, agent7=False):
    dim = len(curr_knowledge)
    max_prob = 0
    candidate = [curr_pos]
    end = curr_pos
    for y in range(dim):
        for x in range(dim):
            if agent7:
                prob = curr_knowledge[y][x].prob_finding
            else:
                prob = curr_knowledge[y][x].prob_containing
            #if find max probability, put it as candidate
            if prob > max_prob:
                max_prob = prob
                end = (x,y)
                candidate = [end]
            #if probability same, then compare their distance
            elif prob == max_prob:
                if hureisticValue(curr_pos,(x,y)) < hureisticValue(curr_pos,end):
                    end = (x,y)
                    candidate = [end]
                #if distance same, then random choice
                elif hureisticValue(curr_pos,(x,y)) == hureisticValue(curr_pos,end):
                    candidate.append((x,y))
    end = random.sample(candidate,1)[0]
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
    x, y = curr_cell
    prob = curr_knowledge[y][x].prob_containing
    # If the discovered cell is blocked, that cell clearly cannot contain the target,
    # and the remaining cells now become slightly more likely to contain it
    if cell_type == 1:
        for row in curr_knowledge:
            for cell in row:
                cell.prob_containing /= (1 - prob)
        curr_knowledge[y][x].prob_containing = 0
        curr_knowledge[y][x].blocked = 1
    # The cell MAY contain the target
    else:
        curr_knowledge[y][x].blocked = 0
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
                for row in curr_knowledge:
                    for cell in row:
                        if cell != curr_knowledge[y][x]:
                            cell.prob_containing /= .2 * prob + (1 - prob)
                curr_knowledge[y][x].prob_containing = .2 * prob / (.2 * prob + (1 - prob))
            elif cell_type == 3 or cell_type == 6:
                for row in curr_knowledge:
                    for cell in row:
                        if cell != curr_cell:
                            cell.prob_containing /= .5 * prob + (1 - prob)
                curr_knowledge[y][x].prob_containing = .5 * prob / (.5 * prob + (1 - prob))
            else:
                for row in curr_knowledge:
                    for cell in row:
                        if cell != curr_cell:
                            cell.prob_containing /= .8 * prob + (1 - prob)
                curr_knowledge[y][x].prob_containing = .8 * prob / (.8 * prob + (1 - prob))

    return curr_knowledge
            
def update_finding_probs(curr_knowledge, unreachable=False):
    dim = len(curr_knowledge)
    n = len(curr_knowledge)**2
    for y in range(dim):
        for x in range(dim):
            if unreachable:
                curr_knowledge[y][x].prob_containing /= (1 - (1 / n))
            # The square has flat terrain
            if curr_knowledge[y][x].terrain == 2 or curr_knowledge[y][x].terrain == 5:
                curr_knowledge[y][x].prob_finding = .8 * curr_knowledge[y][x].prob_containing
            # The square has hilly terrain
            elif curr_knowledge[y][x].terrain == 3 or curr_knowledge[y][x].terrain == 6:
                curr_knowledge[y][x].prob_finding = .5 * curr_knowledge[y][x].prob_containing
	        # The square has forest terrain
            elif curr_knowledge[y][x].terrain == 4 or curr_knowledge[y][x].terrain == 7:
                curr_knowledge[y][x].prob_finding = .2 * curr_knowledge[y][x].prob_containing
            # The square has never been visited, but because the probability it contains the target may have changed,
            # The probability the agent can find it must too
            else:
                curr_knowledge[y][x].prob_finding = (.8 * curr_knowledge[y][x].prob_containing + .5 * curr_knowledge[y][x].prob_containing + .2 * curr_knowledge[y][x].prob_containing) / 3
    return curr_knowledge
	
def agent_6(grid, start):
    curr_knowledge = generate_knowledge(grid)
    found_target = False
    curr_pos = start
    end = start
    exam_count = 0
    move_count = 0
    # The agent finds the optimal square to plan toward until it finds the target
    while not found_target:
        # Calculate the most likely square to contain the target, with ties broken by distance and random selection
        end = find_max_prob(curr_knowledge,curr_pos)
        shortest_path = A_star(curr_knowledge, curr_pos, end)
        #print("curr_pos is {}, end is {}".format(curr_pos, end))
        #the end is unreachable, update curr_knowledge;
        if not shortest_path:
            n = len(curr_knowledge)**2
            x,y = end
            curr_knowledge[y][x].prob_containing = 0
            for row in curr_knowledge:
                for cell in row:
                    cell.prob_containing /= (1 - (1 / n))
            continue
        #print(shortest_path)
		# Traverse the returned shortest path, sensing squares and updating probabilities as the agent moves
        for cell in shortest_path[0]:
            move_count += 1
            x, y = cell
            curr_knowledge[y][x].terrain = grid[y][x]
            curr_knowledge = update_containing_probs(curr_knowledge, grid, cell, grid[y][x])
            exam_count += 1
            if curr_knowledge == 'found':
                print("Found target at: ({}, {})".format(x,y))
                return [cell, move_count, exam_count, move_count + exam_count]
            elif grid[y][x] == 1:
                curr_pos = prev_cell
                exam_count -= 1
                break
            new_end = find_max_prob(curr_knowledge, curr_pos)
            #if the agent found a cell with higher prob, replan
            if new_end != end and curr_knowledge[new_end[1]][new_end[0]].prob_containing > curr_knowledge[end[1]][end[0]].prob_containing:
                curr_pos = cell
                break
            prev_cell = cell

def agent_7(grid, start):
    curr_knowledge = generate_knowledge(grid)
    found_target = False
    curr_pos = start
    end = start
    exam_count = 0
    move_count = 0
    # The agent finds the optimal square to plan toward until it finds the target
    while not found_target:
        # Calculate the most likely square to contain the target, with ties broken by distance and random selection
        end = find_max_prob(curr_knowledge, curr_pos, True)
        shortest_path = A_star(curr_knowledge, curr_pos, end)
        #print("curr_pos is {}, end is {}".format(curr_pos, end))
        #the end is unreachable, update curr_knowledge;
        if not shortest_path:
            n = len(curr_knowledge)**2
            x,y = end
            curr_knowledge[y][x].prob_containing = 0
            curr_knowledge[y][x].prob_finding = 0
            curr_knowledge = update_finding_probs(curr_knowledge, unreachable=True)
            continue
        #print(shortest_path)
		# Traverse the returned shortest path, sensing squares and updating probabilities as the agent moves
        for cell in shortest_path[0]:
            move_count += 1
            x, y = cell
            curr_knowledge[y][x].terrain = grid[y][x]
            curr_knowledge = update_containing_probs(curr_knowledge, grid, cell, grid[y][x])
            exam_count += 1
            if curr_knowledge == 'found':
                print("Found target at: ({}, {})".format(x,y))
                return [cell, move_count, exam_count, move_count + exam_count]
            elif grid[y][x] == 1:
                curr_pos = prev_cell
                exam_count -= 1
                break
            curr_knowledge = update_finding_probs(curr_knowledge)
            new_end = find_max_prob(curr_knowledge, curr_pos, True)
            #if the agent found a cell with higher prob, replan
            if new_end != end and curr_knowledge[new_end[1]][new_end[0]].prob_finding > curr_knowledge[end[1]][end[0]].prob_finding:
                curr_pos = cell
                break
            prev_cell = cell
 
#find end with max probability, tie break uses utility, which compute the weighted distance and blocks around the target cell.
def find_optimal_end(curr_knowledge, curr_pos):
    dim = len(curr_knowledge)
    end = curr_pos
    max_prob = 0
    max_utility = 0
    candidate = []
    for y in range(dim):
        for x in range(dim):
            """if curr_knowledge[y][x].prob_containing == 0:
                continue"""
            curr_knowledge[y][x].sensing(curr_knowledge)
            prob = curr_knowledge[y][x].prob_finding
            dist = hureisticValue(curr_pos,(x,y))
            #if cell is on the edge, add extra blocks
            if curr_knowledge[y][x].n == 3:
                block = curr_knowledge[y][x].b + 5
            elif curr_knowledge[y][x].n == 5:
                block = curr_knowledge[y][x].b + 3
            else:
                block = curr_knowledge[y][x].b
            utility = dist//10 + block//2
            #if find max probability, put it as candidate
            if prob > max_prob:
                max_prob = prob
                max_utility = utility
                end = (x,y)
                candidate = [end]
            #if probability same, then compare their utility
            elif prob == max_prob:
                if utility < max_utility:
                    max_utility = utility
                    end = (x,y)
                    candidate = [end]
                elif utility == max_utility:
                    candidate.append((x,y))
                #print(max_utility , utility)
    #print(candidate)
    end = random.sample(candidate,1)[0]
    return end            
 
def agent_8(grid, start):
    curr_knowledge = generate_knowledge(grid)
    found_target = False
    curr_pos = start
    end = start
    exam_count = 0
    move_count = 0
    # The agent finds the optimal square to plan toward until it finds the target
    while not found_target:
        # Calculate the most likely square to contain the target, with ties broken by distance and random selection
        end = find_optimal_end(curr_knowledge, curr_pos)
        shortest_path = A_star(curr_knowledge, curr_pos, end)
        #print("curr_pos is {}, end is {}".format(curr_pos, end))
        #the end is unreachable, update curr_knowledge;
        if not shortest_path:
            x,y = end
            curr_knowledge[y][x].prob_containing = 0
            curr_knowledge[y][x].prob_finding = 0
            curr_knowledge = update_finding_probs(curr_knowledge, unreachable=True)
            continue
        #print(shortest_path)
		# Traverse the returned shortest path, sensing squares and updating probabilities as the agent moves
        for cell in shortest_path[0]:
            move_count += 1
            x, y = cell
            curr_knowledge[y][x].terrain = grid[y][x]
            curr_knowledge = update_containing_probs(curr_knowledge, grid, cell, grid[y][x])
            exam_count += 1
            if curr_knowledge == 'found':
                print("Found target at: ({}, {})".format(x,y))
                return [cell, move_count, exam_count, move_count + exam_count]
            elif grid[y][x] == 1:
                curr_pos = prev_cell
                exam_count -= 1
                break
            curr_knowledge = update_finding_probs(curr_knowledge)
            new_end = find_optimal_end(curr_knowledge, curr_pos)
            ##if the agent found a cell with higher prob, replan
            if new_end != end and curr_knowledge[new_end[1]][new_end[0]].prob_finding > curr_knowledge[end[1]][end[0]].prob_finding:
                curr_pos = cell
                break
            prev_cell = cell

def plot_move_exam():
    trails = list(range(20))
    Agent6_move = {trail: 0 for trail in trails}
    Agent6_exam = {trail: 0 for trail in trails}
    Agent7_move = {trail: 0 for trail in trails}
    Agent7_exam = {trail: 0 for trail in trails}
    for trail in trails:
        print("running trial {}".format(str(trail)))
        grid, start = generate_terrain(51)
        result_6 = agent_6(grid, start)
        result_7 = agent_7(grid, start)
        Agent6_move[trail] = result_6[1]
        Agent6_exam[trail] = result_6[2]
        Agent7_move[trail] = result_7[1]
        Agent7_exam[trail] = result_7[2]
    plt.title("movement/examinations distribution")
    plt.xlabel("trails")
    plt.ylabel("movement/examinations")
    plt.xticks(trails) 
    plt.plot(trails, Agent6_move.values(),"b", label="Agent6_move")
    plt.plot(trails, Agent6_exam.values(),"g", label="Agent6_exam")
    plt.plot(trails, Agent7_move.values(),"r", label="Agent7_move")
    plt.plot(trails, Agent7_exam.values(),"y", label="Agent7_exam")
    plt.legend(loc='best')
    

def plot_performence():
    trails = list(range(20))
    Agent6 = {trail: 0 for trail in trails}
    Agent7 = {trail: 0 for trail in trails}
    Agent8 = {trail: 0 for trail in trails}
    for trail in trails:
        print("running trial {}".format(str(trail)))
        grid, start = generate_terrain(51)
        Agent6[trail] = agent_6(grid, start)[3]
        Agent7[trail] = agent_7(grid, start)[3]
        Agent8[trail] = agent_8(grid, start)[3]
    ind = np.arange(20)
    plt.figure(figsize=(12,5))
    width = 0.3
    plt.title("Performence by total actions")
    plt.xlabel("trails")
    plt.ylabel("total actions")
    plt.bar(ind - width, Agent6.values() , width, label="Agent6")
    plt.bar(ind, Agent7.values(), width, label="Agent7")
    plt.bar(ind + width, Agent8.values(), width, label="Agent8")
    plt.xticks(ind + width/3, trails)
    plt.legend(loc='best')
    plt.show()
    


