import time
import random
import copy
import itertools
import heapq
import argparse
import warnings
import inspect
import multiprocessing
import numpy as np

inf = float('inf')
debug = True

class Solution:
    def __init__(self):
        self.routes = [] # a list of routes. Each route is a list of tasks.

    def __eq__(self, __value: object):
        if isinstance(__value, Solution):
            return self.routes == __value.routes
        return False
    
    def copy(self):
        sol = Solution()
        sol.routes = copy.deepcopy(self.routes)
        return sol


class Solver:
    '''
    The instance of a Solver matches a corresponding problem instance.

    Data must be manually read and placed inside the Solver. Then, the user must manually call `calc_dist_map` to calculate the distance map.
    
    After that, the user may call `solve()` to solve the problem within given time.

    Solutions:
        Solutions are in the form of tuples of task ID and inversion indicator. An inversion indicator of True indicates that the task should be inverted when fetched from the list.
        
        An ID of 0 represents a dummy task that is used to separate different routes.
        
        Together they are contained in a list.
    '''

    def __init__(self, name: str = "DEFAULT", 
                 n: int = 0, 
                 depot: int = 0, 
                 task_cnt: int = 0, 
                 non_task_cnt: int = 0, 
                 capacity: int = 0, 
                 t_cost: int = 0, 
                 seed: int = 233,
                 psize: int = 30,
                 ubtrial: int = 50,
                 prob_ls: float = 0.2,
                 ms_p: int = 2,
                 gen_cnt: float = inf):
        self.name = name
        self.n = n
        self.depot = depot
        self.task_cnt = task_cnt
        self.non_task_cnt = non_task_cnt
        self.capacity = capacity
        self.t_cost = t_cost
        self.dist_map = np.full((n + 1, n + 1), np.inf) # distance map mapping indices of (src, dst) to costs
        self.node_edge_list = [[] for _ in range(n + 1)] # list mapping indices to list of adjacent edges in the form of [(dst, cost)]
        self.edge_list = []  # in the form of (src, dst, cost), directed
        self.task_list = [(0, 0, 0, 0)]  # in the form of (vertex1, vertex2, cost, demand)
        self.weight_dict = {}  # mapping tuple of (src, dst) to a direct cost
        self.seed = seed  # random seed used
        random.seed(seed)

        self.candidate_list = [] # candidate list for results, sorted in ascending order of favorability

        self.psize = psize # population size
        self.ubtrial = ubtrial # maximum trials for generating initial solutions
        self.opsize = psize * 6 # number of offsprings produced in each generation
        self.prob_ls = prob_ls # probability for conducting a local search
        self.ms_p = ms_p # number of routes involved in Merge-Split operator
        self.gen_cnt = gen_cnt # number of generations involved
        self.fitness_param = 2 # the fitness parameter is used to determined the weight of constraint violation when evaluating the fitness of a solution
        self.ls_trad_iter_cnt = 100 # number of traditional iterations for a local search procedure
        self.ls_ms_iter_cnt = 100 # number of MS operations for a local search procedure

        self.perf_counter_alpha = 0.4
        self.perf_counter_beta = 0.6

    def show_debug_info(self):
        print(f'[solver] name={self.name}, n={self.n}, depot={self.depot}, task_cnt={self.task_cnt}, non_task_cnt={self.non_task_cnt}, capacity={self.capacity}, t_cost={self.t_cost}')

    def run_dijkstra(self, src: int):
        '''
        run dijkstra on source node and update the distance map
        '''
        marked_list = np.zeros((self.n + 1), dtype=int)
        self.dist_map[(src, src)] = 0
        q = [(0, src)]

        while q:
            v_cost, v = heapq.heappop(q)
            if (marked_list[v] == 0):
                marked_list[v] = 1
                for target, target_cost in self.node_edge_list[v]:
                    if marked_list[target] == 1:
                        continue
                    cost0 = v_cost + target_cost
                    if cost0 < self.dist_map[(src, target)]:
                        self.dist_map[(src, target)] = cost0
                        heapq.heappush(q, (cost0, target))

    def calc_dist_map(self):
        '''
        precompute distance map using dijkstra
        '''
        for i in range(1, self.n + 1):
            self.run_dijkstra(i)

    def expand_task(self, task: tuple):
        '''
        expand a task of tuples of (id, inv) into the form (src, dst, cost, demand)
        '''
        target = self.task_list[task[0]]
        if task[1]:
            return (target[1], target[0], target[2], target[3])
        else:
            return (target[0], target[1], target[2], target[3])

    def get_route_cost(self, _route: list):
        '''
        get the cost of a route, represented by a list of tuple of (task id, inverted).
        '''

        if not _route:
            return 0

        route = list(map(self.expand_task, _route))
        result = self.dist_map[(self.depot, route[0][0])] + sum(map(lambda x : x[2], route))
        cursor = 1
        while cursor < len(route):
            result += self.dist_map[(route[cursor - 1][1], route[cursor][0])]
            cursor += 1
        result += self.dist_map[(route[cursor - 1][1], self.depot)]
        return result

    def get_cost(self, sol: Solution):
        '''
        get the cost of a solution
        '''
        return sum(map(self.get_route_cost, sol.routes))

    def get_raw_sol_cost(self, routes: list):
        '''
        get the cost of a solution
        '''
        return sum(map(self.get_route_cost, routes))

    def test_constraints(self, sol: Solution):
        '''
        Calculate how much the constraint is violated

        Return:
            (`count`, `exceeded_num`)
        '''
        count = 0
        exceeded = 0
        for _route in sol.routes:
            demand = sum([i[3] for i in map(self.expand_task, _route)])
            if demand > self.capacity:
                count += 1
                exceeded += demand - self.capacity
        return count, exceeded

    def test_raw_constraints(self, routes: list):
        '''
        calculate how much the constraint is violated

        Return:
            (`count`, `exceeded_num`)
        '''
        count = 0
        exceeded = 0
        for _route in routes:
            demand = sum([i[3] for i in map(self.expand_task, _route)])
            if demand > self.capacity:
                count += 1
                exceeded += demand - self.capacity
        return count, exceeded

    def get_sol_task_cnt(self, sol: Solution):
        return len(set([j[0] for i in sol.routes for j in i]))

    def test_demand(self, sol: Solution):
        '''
        test if the demand is satisfied.
        '''
        return self.task_cnt == self.get_sol_task_cnt(sol)

    def feasible(self, sol: Solution):
        return self.test_constraints(sol)[0] == 0 and self.test_demand(sol)

    def get_fitness(self, sol: Solution):
        '''
        get the fitness of a solution, based on the constraints it has violated and the cost it induces. The lower, the better.
        '''
        return self.get_cost(sol) + self.fitness_param * self.test_constraints(sol)[1]

    def get_presence_cost(self, route: list, idx: int):
        '''
        get the cost of presence of a task inside a route. The cost of the task itself is excluded
        '''
        target = route[idx][0]

        start_idx = self.depot
        mid1_idx = self.task_list[target][0]
        mid2_idx = self.task_list[target][1]
        end_idx = self.depot

        if idx - 1 >= 0:
            if route[idx - 1][1]:
                start_idx = self.task_list[route[idx - 1][0]][0]
            else:
                start_idx = self.task_list[route[idx - 1][0]][1]
        if route[idx][1]: # inverted:
            mid1_idx = self.task_list[target][1]
            mid2_idx = self.task_list[target][0]
        if idx + 1 < len(route):
            if route[idx + 1][1]:
                end_idx = self.task_list[route[idx + 1][0]][1]
            else:
                end_idx = self.task_list[route[idx + 1][0]][0]

        cost = self.dist_map[(start_idx, mid1_idx)] + self.dist_map[(mid2_idx, end_idx)] - self.dist_map[(start_idx, end_idx)]
        return cost

    def get_empty_insertion_cost(self, task: tuple):
        '''
        assume the task is placed in a separated route. Calculate the cost in both direction.
        '''
        cost = self.dist_map[(self.depot, task[0])] + self.dist_map[(task[1], self.depot)]
        inv_cost = self.dist_map[(self.depot, task[1])] + self.dist_map[(task[0], self.depot)]
        return cost, inv_cost

    def get_insertion_cost(self, route: list, idx: int, task: tuple):
        '''
        given the starting index before which the task will be inserted, calculate the augmentation of cost, excluding the cost of the task itself.

        The task is given in the form of (v1, v2, cost, demand).

        Return a tuple (cost, inv_cost), indicating the cost in both directions.
        '''

        start_idx = self.depot
        end_idx = self.depot

        if idx > 0:
            if route[idx - 1][1]:
                start_idx = self.task_list[route[idx - 1][0]][0]
            else:
                start_idx = self.task_list[route[idx - 1][0]][1]

        if idx < len(route):
            if route[idx][1]:
                start_idx = self.task_list[route[idx][0]][1]
            else:
                start_idx = self.task_list[route[idx][0]][0]

        cost = self.dist_map[(start_idx, task[0])] + self.dist_map[(task[1], end_idx)] - self.dist_map[(start_idx, end_idx)]
        inv_cost = self.dist_map[(start_idx, task[1])] + self.dist_map[(task[0], end_idx)] - self.dist_map[(start_idx, end_idx)]
        return cost, inv_cost

    def __builtin_sol_inspect(self, sol: Solution, origin: Solution):
        '''
        inspect whether a solution is valid, and whether its origin is valid.
        '''
        if not self.test_demand(sol):
            print(f'Unsatisfied solution. Is_original_satisfied: {self.test_demand(origin)}')
            print(f"Unsatisfied solution. From: {inspect.currentframe().f_back.f_code.co_name}")
            print(f"self.task_cnt={self.task_cnt}, sol.task_cnt={self.get_sol_task_cnt(sol)}, origin.task_cnt={self.get_sol_task_cnt(origin)}")
            return False
        return True

    def __ps_rule1(self, task):
        '''
        rules accepting an argument of tuple ((task_id, inverted), (vertex1, vertex2, cost, demand))
        '''
        return -self.dist_map[(task[1][0], self.depot)]

    def __ps_rule2(self, task):
        '''
        rules accepting an argument of tuple ((task_id, inverted), (vertex1, vertex2, cost, demand))
        '''
        return self.dist_map[(task[1][0], self.depot)]

    def __ps_rule3(self, task):
        '''
        rules accepting an argument of tuple ((task_id, inverted), (vertex1, vertex2, cost, demand))
        '''
        return -task[1][3] / task[1][2]

    def __ps_rule4(self, task):
        '''
        rules accepting an argument of tuple ((task_id, inverted), (vertex1, vertex2, cost, demand))
        '''
        return task[1][3] / task[1][2]

    def __ps_rule5(self, capacity: int):
        '''
        rules accepting an argument of tuple ((task_id, inverted), (vertex1, vertex2, cost, demand))
        '''
        if capacity > 0:
            def rule(task):
                return -self.dist_map[(task[1][0], self.depot)]
        else:
            def rule(task):
                return self.dist_map[(task[1][0], self.depot)]
        return rule

    def path_scanning(self, tasks0: list):
        '''
        given a list of tasks, do path-scanning to generate a list of five resultant lists of routes based on five rules
        '''
        '''
        do path-scanning based on rules and return a list of routes
        '''
        rules = [self.__ps_rule1, self.__ps_rule2, self.__ps_rule3, self.__ps_rule4, self.__ps_rule5]
        results = []
        for rule in rules:
            tasks = tasks0.copy()
            random.shuffle(tasks)

            result = []
            current_route = []
            capacity = 0
            
            while tasks:
                task_list = list(map(lambda x : (x, self.task_list[x]), tasks))
                candidates = list(filter(lambda x : capacity + x[1][3] <= self.capacity, task_list))
                if not candidates:
                    result.append(current_route)
                    current_route = []
                    capacity = 0
                    continue

                candidates = list(map(lambda x : ((x[0], False), x[1]), candidates)) \
                            + list(map(lambda x : ((x[0], True), (x[1][1], x[1][0], x[1][2], x[1][3])), candidates))
                
                def candidate_dist_extractor(candidate: tuple):
                    if not current_route:
                        return self.dist_map[(self.depot, candidate[1][0])]
                    else:
                        return self.dist_map[(self.task_list[current_route[-1][0]][1], candidate[1][0])]
                    
                candidates.sort(key=candidate_dist_extractor)
                ref = candidate_dist_extractor(candidates[0])
                candidates2 = list(filter(lambda x : candidate_dist_extractor(x) == ref, candidates))
                if rule != self.__ps_rule5:
                    candidates2.sort(key=rule)
                else:
                    candidates2.sort(key=rule(capacity))

                candidate = candidates2[0]
                capacity += candidate[1][3]
                current_route.append(candidate[0])
                tasks.remove(candidate[0][0])

            if current_route:
                result.append(current_route)
            results.append(result)

        return results

    def gen_instance(self):
        '''
        generate an offspring instance that doesn't depend on any other parents.

        Uses the path-scanning method to split the tasks.

        Return:
            A `Solution` object
        '''
        sol = Solution()
        tasks = [i for i in range(1, self.task_cnt + 1)]
        random.shuffle(tasks)

        results = self.path_scanning(tasks)
        random.shuffle(results)

        sol.routes = results[0].copy()
        return sol

    def crossover(self, s1: Solution, s2: Solution):
        '''
        accept two parents, namely `s1` and `s2`, and produce a random offspring by RBX (route-based crossover)
        '''

        rt1 = random.randint(0, len(s1.routes) - 1)
        rt2 = random.randint(0, len(s2.routes) - 1)

        removed_tasks = [i for i in map(lambda x : x[0], s1.routes[rt1])]
        new_tasks = [i for i in map(lambda x : x[0], s2.routes[rt2])]
        duplicated_tasks = [i for i in filter(lambda x : x not in removed_tasks, new_tasks)] # now stores the list of id of duplicated tasks
        unserved_tasks = [i for i in filter(lambda x : x not in new_tasks, removed_tasks)] # now stores the list of id of unserved tasks

        sx = s1.copy()
        sx.routes[rt1] = copy.deepcopy(s2.routes[rt2])

        # I. remove the duplicated
        for duplicated in duplicated_tasks:
            '''
            record the duplicated position, check for cost and prune the one with larger cost
            '''
            rm_route_idx_x = -1
            rm_route_idx_y = -1
            rm_cost = -inf
            for i in range(len(sx.routes)):
                for j in range(len(sx.routes[i])):
                    if sx.routes[i][j][0] == duplicated:
                        cost = self.get_presence_cost(sx.routes[i], j)
                        if cost > rm_cost:
                            rm_route_idx_x = i
                            rm_route_idx_y = j
                            rm_cost = cost
            
            if len(sx.routes[rm_route_idx_x]) == 1:
                sx.routes.pop(rm_route_idx_x)
            else:
                sx.routes[rm_route_idx_x].pop(rm_route_idx_y)

        # DEBUG
        stemp = sx.copy()

        # II. insert the unserved tasks
        random.shuffle(unserved_tasks)
        for x in unserved_tasks:
            pending = self.task_list[x]
            demand = pending[3]
            demand_list = list(filter(lambda x: self.capacity - x[1] >= demand, zip(range(len(sx.routes)), map(lambda x : sum(map(lambda y : self.task_list[y[0]][3], x)), sx.routes))))

            ist_cost = inf
            is_inverted = False
            ist_idx_x = -1
            ist_idx_y = -1

            for entry in demand_list:
                for j in range(len(sx.routes[entry[0]]) + 1):
                    cost, inv_cost = self.get_insertion_cost(sx.routes[entry[0]], j, pending)
                    if cost < ist_cost or inv_cost < ist_cost:
                        ist_idx_x = entry[0]
                        ist_idx_y = j
                        if cost < inv_cost:
                            ist_cost = cost
                            is_inverted = False
                        else:
                            ist_cost = inv_cost
                            is_inverted = True
            
            empty_cost, empty_inv_cost = self.get_empty_insertion_cost(pending)
            if ist_cost == inf or ist_cost > empty_cost or ist_cost > empty_inv_cost:
                route = []
                if empty_cost < empty_inv_cost:
                    route.append((x, False))
                else:
                    route.append((x, True))
                sx.routes.append(route)
            else:
                sx.routes[ist_idx_x].insert(ist_idx_y, (x, is_inverted))

        if not self.__builtin_sol_inspect(sx, s1): exit(1)
        if not self.__builtin_sol_inspect(sx, s2): exit(1)

        return sx

    def ulusoy_split(self, sol0: Solution):
        '''
        split the target list using Ulusoy's Splitting Procedure
        '''
        # TODO: implementation
        return sol0.copy()

    def do_single_insertion(self, sol0: Solution):
        '''
        do a single insertion operation on the target solution
        '''
        sol = sol0.copy()
        total_cnt = sum([len(route) for route in sol.routes])
        src_idx = random.randint(0, total_cnt - 1)
        candidate = None
        for i in range(len(sol.routes)):
            len0 = len(sol.routes[i])
            if src_idx < len0:
                candidate = sol.routes[i].pop(src_idx)
                if not sol.routes[i]:
                    del sol.routes[i]
                break
            else:
                src_idx -= len0

        t_cap = self.task_list[candidate[0]][3]

        route_indices = list(range(0, len(sol.routes) + 1))
        random.shuffle(route_indices)
        for route_idx in route_indices:
            if route_idx < len(sol.routes):
                r_cap = self.capacity - self.get_route_cost(sol.routes[route_idx])
                if t_cap <= r_cap:
                    tar_idx = random.randint(0, len(sol.routes[route_idx]))
                    cost, inv_cost = self.get_insertion_cost(sol.routes[route_idx], tar_idx, self.task_list[candidate[0]])
                    if cost < inv_cost:
                        sol.routes[route_idx].insert(tar_idx, candidate)
                    else:
                        sol.routes[route_idx].insert(tar_idx, (candidate[0], True))
                    break
            else:
                sol.routes.append([candidate])
                break
        
        if not self.__builtin_sol_inspect(sol, sol0): exit(1)

        return sol
        
    def do_double_insertion(self, sol0: Solution):
        '''
        do a double insertion operation on the target solution
        '''
        sol = sol0.copy()
        src_routes = list(filter(lambda x: len(x) >= 2, sol.routes))
        if not src_routes:
            return sol
        
        # shuffle and pick up the first candidate
        random.shuffle(src_routes)
        route = src_routes[0]
        ridx = random.randint(0, len(route) - 2)
        task1 = route.pop(ridx + 1)
        task2 = route.pop(ridx)

        total_cap = self.task_list[task1[0]][3] + self.task_list[task2[0]][3]
        total_cnt = len(sol.routes)
        if random.randint(0, total_cnt) < total_cnt:
            tar_routes = list(filter(lambda x: self.capacity - self.get_route_cost(x) >= total_cap, sol.routes))
            if tar_routes:
                random.shuffle(tar_routes)
                route = tar_routes[0]
                tar_pos = random.randint(0, len(route))

                cost, inv_cost = self.get_insertion_cost(route, tar_pos, self.task_list[task1[0]])
                if cost < inv_cost:
                    route.insert(tar_pos, task1)
                else:
                    route.insert(tar_pos, (task1[0], True))

                tar_pos += 1
                cost, inv_cost = self.get_insertion_cost(route, tar_pos, self.task_list[task2[0]])
                if cost < inv_cost:
                    route.insert(tar_pos, task2)
                else:
                    route.insert(tar_pos, (task2[0], True))

                return sol
        # else append to a new route
        route = []
        cost, inv_cost = self.get_empty_insertion_cost(self.task_list[task1[0]])
        if cost < inv_cost:
            route.append(task1)
        else:
            route.append((task1[0], True))
        
        cost, inv_cost = self.get_insertion_cost(route, 1, self.task_list[task2[0]])
        if cost < inv_cost:
            route.append(task2)
        else:
            route.append((task2[0], True))
        sol.routes.append(route)
        
        if not self.__builtin_sol_inspect(sol, sol0): exit(1)

        return sol

    def do_swap(self, sol0: Solution):
        '''
        do a swap operation on the target solution and evaluates the effectiveness using function `f`. Swap two tasks inside a route only.
        '''
        sol = sol0.copy()
        src_routes = list(filter(lambda x: len(x) >= 2, sol.routes))
        if not src_routes:
            return sol
        
        src_route = src_routes[0]
        idx1, idx2 = sorted(random.sample(range(0, len(src_route)), k=2), reverse=True)
        task1 = src_route.pop(idx1)
        task2 = src_route.pop(idx2)

        cost, inv_cost = self.get_insertion_cost(src_route, idx2, self.task_list[task1[0]])
        if cost < inv_cost:
            src_route.insert(idx2, task1)
        else:
            src_route.insert(idx2, (task1[0], True))

        cost, inv_cost = self.get_insertion_cost(src_route, idx1, self.task_list[task2[0]])
        if cost < inv_cost:
            src_route.insert(idx1, task2)
        else:
            src_route.insert(idx1, (task2[0], True))
        
        if not self.__builtin_sol_inspect(sol, sol0): exit(1)

        return sol

    def do_merge_split(self, sol0: Solution, local_param: float):
        '''
        do a Merge-Split operation on the target solution, return a list of results
        '''
        results = [] # store the list of solutions after applying MS
        maxlen = len(sol0.routes)
        ubcomb = maxlen * (maxlen - 1) / 2
        pending_index_list = []

        if ubcomb <= self.ls_ms_iter_cnt:
            pending_index_list = [list(i) for i in itertools.combinations(range(maxlen), r=min(maxlen, self.ms_p))]
        else:
            indices = random.sample(range(maxlen), k=min(maxlen, self.ms_p))
            pending_index_list.append(indices)
        
        for indices in pending_index_list:
            sol = sol0.copy()
            indices.sort(reverse=True)
            tasks = []
            for idx in indices:
                tasks = tasks + [i[0] for i in sol.routes[idx]]
                del sol.routes[idx]

            ps_results = self.path_scanning(tasks)

            # sort and return
            ps_results.sort(key=lambda x : self.get_raw_sol_cost(x) + local_param * self.test_raw_constraints(x)[1])
            sol.routes = sol.routes + ps_results[0]

            if not self.__builtin_sol_inspect(sol, sol0): exit(1)
            results.append(sol)

        return results
        
    def local_search(self, sol0: Solution, ancestors: list):
        '''
        the local search procedure that searches around the given solution `sol`
        '''
        best_candidate = ancestors[0]
        tc_best = self.get_cost(best_candidate)

        tc_s = self.get_cost(sol0)
        tv_s = self.test_constraints(sol0)[1]
        localparam = tc_best / self.capacity * (1.0 + tc_best / tc_s + tv_s / self.capacity)

        def get_test_func(_localparam):
            return lambda x : self.get_cost(x) + _localparam * self.test_constraints(x)[1]

        sol = sol0.copy()
        candidate_list = []
        last_val = 0
        for _ in range(self.ls_trad_iter_cnt):
            candidate_list = [sol]
            candidate_list.append(self.do_single_insertion(sol))
            candidate_list.append(self.do_double_insertion(sol))
            candidate_list.append(self.do_swap(sol))
            candidate_list.sort(key=get_test_func(localparam))
            sol = candidate_list[0]

        candidate_list = candidate_list + self.do_merge_split(sol, localparam)
        
        candidate_list.sort(key=get_test_func(localparam))
        return candidate_list[0]

    def produce(self, ancestors: list):
        '''
        produce an offspring based on the list of parents and the best_candidate. Return a new solution

        Procedure:

            Setup the fitness parameter

        '''
        s1, s2 = random.choices(ancestors, k=2)
        offspring = self.crossover(s1, s2)
        r = random.random()
        if r < self.prob_ls:
            offspring = self.local_search(offspring, ancestors)
        return offspring

    def solve(self, rtime: float = 120, parallel: bool = False):
        '''
        try solving the problem in time

        Procedure:

            Record the starting time

            Preparation:
                    
                    Initialize the population

            Do algorithm until timeout:

                    Produce offspring and improve until opsize is reached

                    Sort the solutions using stochastic ranking

                    Pick the first psize offspring
                    
                    Pick the best valid solution and append it to the candidate list
        '''
        start = time.perf_counter()
        pop = [] # the population

        # initialize the population
        size = 0
        while size < self.psize:
            ntrial = 1
            sol = self.gen_instance()
            while sol in pop and ntrial <= self.ubtrial:
                sol = self.gen_instance()
                ntrial += 1
            if sol in pop:
                break
            pop.append(sol)
            size += 1

        size = len(pop)
        self.candidate_list.append(random.choice(pop))

        time_threshold = -1
        # do algorithm until timeout

        while time.perf_counter() <= start + rtime - time_threshold:
            loop_start = time.perf_counter()
            pop_t = list(pop)

            if parallel:
                with multiprocessing.Pool(processes=8) as pool:
                    tasks = [pop_t for _ in range(self.opsize)]
                    results = pool.map(self.produce, tasks)
                    pop_t = pop_t + list(filter(lambda x : x not in pop_t, results))
            else:
                results = []
                for _ in range(self.opsize):
                    results.append(self.produce(pop_t))
                pop_t = pop_t + list(filter(lambda x : x not in pop_t, results))

            pop_t.sort(key=self.get_fitness)
            pop = pop_t[:size]

            for sol in pop:
                if self.feasible(sol):
                    self.candidate_list.append(sol)
                    break
            loop_time = time.perf_counter() - loop_start
            if time_threshold == -1:
                time_threshold = loop_time
            else:
                time_threshold = self.perf_counter_alpha * time_threshold + self.perf_counter_beta * loop_time
        
        self.pop = pop

    def task2str(self, task: tuple):
        '''
        transform a task tuple (task_id, inv) into its string representation
        '''
        target = self.task_list[task[0]]
        if task[1]: # inversed
            return f'({target[1]},{target[0]})' 
        else:
            return f'({target[0]},{target[1]})' 
    
    def route2str(self, route: list):
        '''
        transform a route into its string representation
        '''
        return f'0,{",".join(map(self.task2str, route))},0'
    
    def sol2str(self, sol: Solution):
        '''
        transform a solution into its string representation
        '''
        return ','.join(map(self.route2str, sol.routes))

    def print_result(self):
        '''
        print the result of calculation
        '''
        print(f's {self.sol2str(self.candidate_list[-1])}')
        print(f'q {int(self.get_cost(self.candidate_list[-1]))}')


def read_carp_data(filename: str, psize: int):
    '''
    read CARP data from given source and initialize a Solver instance
    '''
    lines = []
    with open(filename, 'r') as f:
        lines = f.readlines()

    params = []
    for line in lines[:8]:
        params.append(line.split(":")[1].strip())

    name = params[0]
    n, depot, task_cnt, non_task_cnt, _, capacity, t_cost = map(
        int, params[1:8])

    solver = Solver(name, n, depot, task_cnt, non_task_cnt, capacity, t_cost, psize=psize)

    for line in lines[9:]:
        if (line.startswith("END")):
            break
        # import edges
        src, dst, cost, demand = line.split()
        src = int(src)
        dst = int(dst)
        cost = int(cost)
        demand = int(demand)
        solver.node_edge_list[src].append((dst, cost))
        solver.node_edge_list[dst].append((src, cost))
        solver.edge_list.append((src, dst, cost))
        solver.edge_list.append((dst, src, cost))
        if (demand > 0):
            solver.task_list.append((src, dst, cost, demand))
        solver.weight_dict[(src, dst)] = cost
        solver.weight_dict[(dst, src)] = cost
    return solver


def test(solver: Solver):
    '''
    test routines for the solver
    '''
    # print(f'printing tasks')
    # for edge in solver.task_list:
    #     print(f'\t{edge}')
    # print(solver.dist_map)
    # print(solver.task_list)
    # inst = solver.gen_instance()
    # print(solver.sol2str(inst))
    # print(int(solver.get_cost(inst)))
    # sample_sol = Solution()
    # sample_route1 = [(1, False), (2, False)]
    # sample_route2 = [(5, True), (4, True), (3, True)]
    # sample_sol.routes = [sample_route1, sample_route2]
    # print(solver.sol2str(sample_sol))
    # print(solver.get_cost(sample_sol))
    # for item in solver.candidate_list:
        # print(solver.sol2str(item))
    
    # print(f'Printing population:')
    # for item in solver.pop:
    #     print(f'\t{solver.sol2str(item)}, satisfied: ')


if __name__ == "__main__":
    time_start = time.perf_counter()

    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "file", help="file name", type=str)
    argParser.add_argument(
        "-t", "--time", help="max time (in seconds) the program can use before termination", type=int)
    argParser.add_argument(
        "-s", "--seed", help="random seed used in the program", type=int)
    argParser.add_argument(
        "-p", "--population", help="size of the population", type=int, default=30)
    argParser.add_argument(
        "-f", "--fetch", help="fetch information about the dataset and display them without solving the problem", action="store_const", const=True)
    cmd_args = argParser.parse_args()

    filename = cmd_args.file
    seed = cmd_args.seed
    runtime = cmd_args.time
    psize = cmd_args.population
    solve = not cmd_args.fetch

    ### DEBUG
    # filename = "sample.dat"
    # seed = 2333
    # runtime = 10
    ### DEBUG

    solver = read_carp_data(filename, psize=psize)
    solver.seed = seed
    solver.calc_dist_map()

    if solve:
        remaining_time = runtime - (time.perf_counter() - time_start) - 0.01
        solver.solve(remaining_time, parallel=False)
        solver.print_result()
        print()

    solver.show_debug_info()
    print(f'[perf] Took {time.perf_counter() - time_start} seconds')
    test(solver)