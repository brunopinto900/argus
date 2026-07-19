# A fixed-bounds 3-D Euclidean signed (unsigned, in this project) distance
# field, published whenever argus_mapping_node finishes a computeEsdf() pass.
# Mirrors argus_esdf::VoxelGrid's own layout exactly so the receiving side
# can rebuild a queryable VoxelGrid via VoxelGrid::setDistanceField() with
# no reinterpretation.

std_msgs/Header header

geometry_msgs/Point origin   # world-frame position of voxel (0,0,0)'s min corner
int32[3] dims                # voxel counts per axis [x, y, z]
float64 voxel_size           # voxel edge length [m]

# Unsigned distance-to-nearest-obstacle [m] per voxel, row-major with x
# fastest-varying: index = x + dims[0] * (y + dims[1] * z) — same formula as
# argus_esdf::VoxelGrid::linearIndex(). Length must equal dims[0]*dims[1]*dims[2].
float32[] distances
