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

from dataclasses import dataclass
import logging

def evaluate(inputfile, outputfile):
    # n = int(inputfile.readline().strip())
    # w = []
    # for i in range(n):
    #     aux = list(map(int, inputfile.readline().strip().split()))
    #     assert len(aux) == n-i, (i, len(aux), n-i)
    #     row = []
    #     for j in range(i):
    #         row.append(w[j][i])
    #     row += aux
    #     w.append(row)

    with open(inputfile, "r") as file:
        lines = file.readlines()
        n = int(lines[0])
        w: list[list[float]] = [[0 for j in range(n)] for i in range(n)]
        t = 1
        for i in range(n):
            l = list(map(float, lines[t].split()))[1:]
            for j in range(len(l)):
                w[i][j] = l[j]
                w[j][i] = l[j]
            t += 1 

    # lines = outputfile.read().strip().split("\n")
    # cliques = []
    # nodes = set()
    # for line in lines:
    #     try:
    #         cliques.append(list(map(int, line.strip().split())))
    #     except ValueError:
    #         logging.warning(f"Ignoring line (failed to parse): {line}")
    #         continue
    #     for v in cliques[-1]:
    #         if v > n or v < 1:
    #             logging.error(f"Invalid node value {v}")
    #             return None
    #         if v in nodes:
    #             logging.error(f"Repeated node value {v}")
    #             return None
    #         nodes.add(v)
    # if len(nodes) != n:
    #     logging.error(f"Got {len(nodes)} unique nodes, expected {n}")
    #     return None
    
    with open(outputfile, "r") as file:
        lines = file.readlines()
        cliques = []
        nodes = set()
        for line in lines:
            try:
                cliques.append(list(map(int, line.split())))
            except ValueError:
                logging.warning(f"Ignoring line (failed to parse): {line}")
                continue
            for v in cliques[-1]:
                if v > n - 1 or v < 0:
                    logging.error(f"Invalid node value {v}")
                    return None
                if v in nodes:
                    logging.error(f"Repeated node value {v}")
                    return None
                nodes.add(v)
        if len(nodes) != n:
            logging.error(f"Got {len(nodes)} unique nodes, expected {n}")
            return None

    obj = 0
    for ind, c in enumerate(cliques):
        aux = 0
        for i in range(len(c)):
            for j in range(i, len(c)):
                aux += w[c[i]-1][c[j]-1]
        obj += aux

    return obj

if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file')
    parser.add_argument('--output-file')
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stderr, level="INFO",
                        format="%(levelname)s;%(message)s")

    obj = evaluate(args.input_file, args.output_file)
    print("Objective value:", obj)


