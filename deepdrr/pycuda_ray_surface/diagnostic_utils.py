#============================================================================
#Copyright (c) 2023, Raymond Leung
#All rights reserved.
#
#This source code is licensed under the BSD-3-clause license.
#See the LICENSE.md file in the root directory for details.
#
#Purpose: Facilitate binary to float conversion and display BHV node content
#Details:
# - This module provides methods for type conversion between float and binary.
# - To examine attribute values within "struct BVHNode", supply it with the
#   bytestream of a BVHNode in the form of an numpy.int32 array. This may
#   be obtained using
#   |   ints_per_node = int(struct_size(bytes_in_BVHNode, d_szQuery) / 4)
#   |   h_leafNodes = np.zeros(n_triangles * ints_per_node, dtype=np.int32)
#   |   cuda.memcpy_dtoh(h_leafNodes, d_leafNodes)                     
#   then iterating over individual BVHNode's using
#   |   for t in range(n_triangles):
#   |       words = h_leafNodes[t*sz_BVHNode:(t+1)*sz_BVHNode]
#   |       display_node_contents(words, t, params['USE_EXTRA_BVH_FIELDS'])
#   This shows the content within each BVHNode one node at a time.
# - If the PyCUDA code is compiled with params['USE_EXTRA_BVH_FIELDS'] set
#   to True (this introduces extra diagnostic parameters into struct BVHNode),
#   the address of the parent, current and left/right descendant nodes will
#   be reported, allowing the binary radix tree structure to be traversed.
#============================================================================
import numpy as np
import struct
from codecs import decode


def bin_to_float(b, bytes=8):
    """ Convert binary string into IEEE 754 binary float32/64 format. """
    if bytes==8: #case for float64
        bf = int_to_bytes(int(b, 2), 8)
        return struct.unpack('>d', bf)[0]
    else:        #case for float32
        bf = int_to_bytes(int(b, 2), 4)
        return struct.unpack('>f', bf)[0]

def int_to_bytes(n, length):
    """ Int/long to byte string. """
    return decode('%%0%dx' % (length << 1) % n, 'hex')[-length:]

def float_to_bin(value, bytes=8):
    """ Convert IEEE 754 float to 32/64-bit binary string. """
    if bytes==8:
        [d] = struct.unpack(">Q", struct.pack(">d", value))
        return '{:064b}'.format(d) 
    else:
        [d] = struct.unpack(">L", struct.pack(">f", value))
        return '{:032b}'.format(d)

def display_node_contents(words, node_id, include_extra_fields=False):
    """ Interpret values in the word-stream according to BVHNode structure """
    #`words` represents a single BVHNode as an int32 numpy array
    offset = 9 if include_extra_fields else 0
    xMin = '%.6g' % bin_to_float(np.binary_repr(words[0],32),4)
    xMax = '%.6g' % bin_to_float(np.binary_repr(words[1],32),4)
    yMin = '%.6g' % bin_to_float(np.binary_repr(words[2],32),4)
    yMax = '%.6g' % bin_to_float(np.binary_repr(words[3],32),4)
    zMin = '%.6g' % bin_to_float(np.binary_repr(words[4],32),4)
    zMax = '%.6g' % bin_to_float(np.binary_repr(words[5],32),4)
    childL = hex(int(np.binary_repr(words[7],32) + np.binary_repr(words[6],32), 2))
    childR = hex(int(np.binary_repr(words[9],32) + np.binary_repr(words[8],32), 2))
    leaf_node = sum(words[6:10])==0

    if include_extra_fields:
        parent = hex(int(np.binary_repr(words[11],32) + np.binary_repr(words[10],32), 2))
        myself = hex(int(np.binary_repr(words[13],32) + np.binary_repr(words[12],32), 2))
        idxSelf, idxChildL, idxChildR, isLeafChildL, isLeafChildR = words[14:19]

    triangleID, atomic, rangeL, rangeR = words[offset+10:offset+14]
    contents = f'[{node_id}] x:[{xMin},{xMax}], y:[{yMin},{yMax}], z:[{zMin},{zMax}]\n'

    if include_extra_fields:
        contents += f'self: {myself}, parent: {parent}\n'
        if not leaf_node:
            label = '(root node)' if atomic == 0 and rangeR < 0 else ''
            typeL = 'leaf' if isLeafChildL else 'internal'
            typeR = 'leaf' if isLeafChildR else 'internal'
            contents += f'indices: {idxSelf}(self), {idxChildL}(L-{typeL}), ' \
                        f'{idxChildR}(R-{typeR})\nchildL: {childL}{label}, childR: {childR}\n'
    if leaf_node:
        contents += f'triangleID: {triangleID}\n'
    else:
        contents += f'atomic: {atomic}, rangeL: {rangeL}, rangeR: {rangeR}\n'

    print(contents)
