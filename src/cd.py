#!/usr/bin/env python3
#
# Copyright (C) 2023 Alexandre Jesus <https://adbjesus.com>, Carlos M. Fonseca <cmfonsec@dei.uc.pt>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from typing import TextIO, Optional, Any
from collections.abc import Iterable, Hashable

from dataclasses import dataclass

import logging

Objective = Any

class Node:
    def __init__(self,
                 node_id: int
                 ) -> None:
        super().__init__()
        self.__node_id: int = node_id
    
    def get_id(self) -> int:
        return self.__node_id
    
    def __str__(self) -> str:
        return str(self.get_id())
    
    def __repr__(self) -> str:
        return str(self)
    
    def __hash__(self) -> int:
        return hash(self.get_id())
    
    def __eq__(self, __value: object) -> bool:
        return self.get_id() == __value.get_id()

    @property
    def cid(self) -> Hashable:
        return self.get_id()

class Edge:
    def __init__(self,
                 i: int,
                 j: int
                 ) -> None:
        super().__init__()
        self.__i: int = i if i < j else j
        self.__j: int = j if i < j else i
    
    def get_i(self) -> int:
        return self.__i
    
    def get_j(self) -> int:
        return self.__j
    
    def __str__(self) -> str:
        return "<"+str(self.get_i())+","+str(self.get_j())+">"
    
    def __repr__(self) -> str:
        return str(self)
    
    def __hash__(self) -> int:
        return hash(str(self))
    
    def __eq__(self, __value: object) -> bool:
        return self.get_i() == __value.get_i() and self.get_j() == __value.get_j()

    @property
    def cid(self) -> Hashable:
        return self.get_i(), self.get_j()
    
@dataclass
class Component:
    u: int
    v: int

    @property
    def cid(self) -> Hashable:
        return self.u, self.v
    
# class Component:
#     @property
#     def cid(self) -> Hashable:
#         raise NotImplementedError

class LocalMove:
    ...

class Solution:
    def __init__(self,
                problem: Problem,
                community_structure: list[int]) -> None:
        self.problem = problem
        self.community_structure = community_structure
        self.community_mappings = {}
        self.N = len(self.community_structure)

        for i in range(self.N):
            community_id = self.community_structure[i]
            if not (community_id in self.community_mappings):
                self.community_mappings[community_id] = []
            self.community_mappings[community_id].append(i)

        self.community_edges = {}
        for i in range(self.N):
            community_id = self.community_structure[i]
            if not (community_id in self.community_edges):
                self.community_edges[community_id] = []
            row = self.problem.graph[i]
            neighs = []
            for j in range(len(row)):
                weight = row[j]
                if weight != 0 and self.community_structure[j] == community_id:
                    neighs.append(Edge(i, j))
            self.community_edges[community_id].extend(neighs)

    def output(self) -> str:
        """
        Generate the output string for this solution
        """
        output_string = ""

        K = len(self.community_mappings)
        for i in range(K):
            output_string += " ".join(self.community_mappings[i])
            output_string += "\n"

        return output_string

    def copy(self) -> Solution:
        """
        Return a copy of this solution.

        Note: changes to the copy must not affect the original
        solution. However, this does not need to be a deepcopy.
        """

        return Solution(problem=self.problem, community_structure=[val for val in self.community_structure])
        
    def is_feasible(self) -> bool:
        """
        Return whether the solution is feasible or not
        """

        return len(self.community_structure) == self.N

    def objective(self) -> Optional[float]:
        """
        Return the objective value for this solution if defined, otherwise
        should return None
        """

        if self.is_feasible():
            obj = 0.0
            K = len(self.community_mappings)
            for i in range(K):
                neighs = self.community_edges[i]
                obj += sum([self.problem.graph[edge.get_i()][edge.get_j()] for edge in neighs])/2.0
            return -obj
        else:
            return None

    def lower_bound(self) -> Optional[Objective]:
        """
        Return the lower bound value for this solution if defined,
        otherwise return None
        """
        bound = 0.0
        for i in range(self.N):
            for j in range(i+1, self.N):
                if self.community_structure[i] == self.community_structure[j]:
                    bound += self.problem.graph[i][j]
                else:
                    if self.problem.graph[i][j] > 0:
                        bound += self.problem.graph[i][j]
        return -bound

    def add_moves(self) -> Iterable[Component]:
        """
        Return an iterable (generator, iterator, or iterable object)
        over all components that can be added to the solution
        """
        if len(self.community_structure) == 0:
            yield Component(0, 0)
        else:
            new_node_id = len(self.community_structure)
            yield Component(new_node_id, max(self.community_structure)+1)
            for k in range(max(self.community_structure)):
                yield Component(new_node_id, k)

    def local_moves(self) -> Iterable[LocalMove]:
        """
        Return an iterable (generator, iterator, or iterable object)
        over all local moves that can be applied to the solution
        """
        raise NotImplementedError

    def random_local_move(self) -> Optional[LocalMove]:
        """
        Return a random local move that can be applied to the solution.

        Note: repeated calls to this method may return the same
        local move.
        """
        raise NotImplementedError

    def random_local_moves_wor(self) -> Iterable[LocalMove]:
        """
        Return an iterable (generator, iterator, or iterable object)
        over all local moves (in random order) that can be applied to
        the solution.
        """
        raise NotImplementedError
            
    def heuristic_add_move(self) -> Optional[Component]:
        """
        Return the next component to be added based on some heuristic
        rule.
        """
        raise NotImplementedError

    def add(self, component: Component) -> None:
        """
        Add a component to the solution.

        Note: this invalidates any previously generated components and
        local moves.
        """
        node_id, community_id = component.u, component.v
        self.community_structure[node_id] = community_id
        if not (community_id in self.community_mappings):
            self.community_mappings[community_id] = []
        self.community_mappings[community_id].append(node_id)

        if not (community_id in self.community_edges):
            self.community_edges[community_id] = []
        row = self.problem.graph[node_id]
        neighs = []
        for j in range(len(row)):
            weight = row[j]
            if weight != 0 and self.community_structure[j] == community_id:
                neighs.append(Edge(node_id, j))
        self.community_edges[community_id].extend(neighs)

    def step(self, lmove: LocalMove) -> None:
        """
        Apply a local move to the solution.

        Note: this invalidates any previously generated components and
        local moves.
        """
        raise NotImplementedError

    def objective_incr_local(self, lmove: LocalMove) -> Optional[Objective]:
        """
        Return the objective value increment resulting from applying a
        local move. If the objective value is not defined after
        applying the local move return None.
        """
        raise NotImplementedError

    def lower_bound_incr_add(self, component: Component) -> Optional[Objective]:
        """
        Return the lower bound increment resulting from adding a
        component. If the lower bound is not defined after adding the
        component return None.
        """
        raise NotImplementedError

    def perturb(self, ks: int) -> None:
        """
        Perturb the solution in place. The amount of perturbation is
        controlled by the parameter ks (kick strength)
        """
        raise NotImplementedError

    def components(self) -> Iterable[Component]:
        """
        Returns an iterable to the components of a solution
        """
        raise NotImplementedError

class Problem:
    def __init__(self, graph: list[list[float]]) -> None:
        self.nnodes = len(graph)
        self.graph = graph
        
    @classmethod
    def from_textio(cls, f: str) -> Problem:
        """
        Create a problem from a text I/O source `f`
        """
        with open(f, "r") as file:
            lines = file.readlines()
            n = int(lines[0])
            graph: list[list[float]] = [[0 for j in range(n)] for i in range(n)]
            t = 1
            for i in range(n):
                l = list(map(float, lines[t].split()))[1:]
                for j in range(len(l)):
                    graph[i][j] = l[j]
                    graph[j][i] = l[j]
                t += 1 

        return cls(graph)


    def empty_solution(self) -> Solution:
        """
        Create an empty solution (i.e. with no components).
        """
        return Solution(self, [])


if __name__ == '__main__':
    from api.solvers import *
    from time import perf_counter
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--log-level',
                        choices=['critical', 'error', 'warning', 'info', 'debug'],
                        default='warning')
    parser.add_argument('--log-file', type=argparse.FileType('w'), default=sys.stderr)
    parser.add_argument('--csearch',
                        choices=['beam', 'grasp', 'greedy', 'heuristic', 'as', 'mmas', 'none'],
                        default='none')
    parser.add_argument('--cbudget', type=float, default=5.0)
    parser.add_argument('--lsearch',
                        choices=['bi', 'fi', 'ils', 'rls', 'sa', 'none'],
                        default='none')
    parser.add_argument('--lbudget', type=float, default=5.0)
    parser.add_argument('--input-file', default=sys.stdin)
    parser.add_argument('--output-file', default=sys.stdout)
    args = parser.parse_args()

    logging.basicConfig(stream=args.log_file,
                        level=args.log_level.upper(),
                        format="%(levelname)s;%(asctime)s;%(message)s")

    p = Problem.from_textio(args.input_file)
    s: Optional[Solution] = p.empty_solution()

    start = perf_counter()

    if s is not None:
        if args.csearch == 'heuristic':
            s = heuristic_construction(s)
        elif args.csearch == 'greedy':
            s = greedy_construction(s)
        elif args.csearch == 'beam':
            s = beam_search(s, 10)
        elif args.csearch == 'grasp':
            s = grasp(s, args.cbudget, alpha = 0.01)
        elif args.csearch == 'as':
            ants = [s]*100
            s = ant_system(ants, args.cbudget, beta = 5.0, rho = 0.5, tau0 = 1 / 3000.0)
        elif args.csearch == 'mmas':
            ants = [s]*100
            s = mmas(ants, args.cbudget, beta = 5.0, rho = 0.02, taumax = 1 / 3000.0, globalratio = 0.5)

    if s is not None:
        if args.lsearch == 'bi':
            s = best_improvement(s, args.lbudget)
        elif args.lsearch == 'fi':
            s = first_improvement(s, args.lbudget) 
        elif args.lsearch == 'ils':
            s = ils(s, args.lbudget)
        elif args.lsearch == 'rls':
            s = rls(s, args.lbudget)
        elif args.lsearch == 'sa':
            s = sa(s, args.lbudget, 30)

    end = perf_counter()

    if s is not None:
        print(s.output(), file=args.output_file)
        if s.objective() is not None:
            logging.info(f"Objective: {s.objective():.3f}")
        else:
            logging.info(f"Objective: None")
    else:
        logging.info(f"Objective: no solution found")

    logging.info(f"Elapsed solving time: {end-start:.4f}")

