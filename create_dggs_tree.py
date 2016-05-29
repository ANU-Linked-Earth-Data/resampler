#!/usr/bin/env python3

import h5py
import itertools

def getNames(start, end):
    result = []
    if start == 0:
       names = ['N','O','P','Q','R','S']
       result.append(names)
       start += 1

    for i in range(start, end+1):
        names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        result.append(names)
    return result

def create_dggs_hierarchy(start, end, cell=""):
    assert cell != "" or start == 0

    f_dggs = h5py.File("hello.hdf5", "w")
    if cell != "":
        cell = "/" + "/".join(cell)

    for tup in itertools.product(*getNames(start, end)):
        name = "/".join((cell,) + tup)
        f_dggs.create_group(name)

if __name__ == '__main__':
    print(
        'Creating tree for all nodes from N000000 to N099999 (as well as '
        'their parents)'
    )
    create_dggs_hierarchy(2,6, 'N0')
