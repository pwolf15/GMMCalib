1.       GMMCalib returns a 3D point cloud with associatiens of every point to cluster.

2.       Given the CAD Model: Move the dusters to better match the CAD Model

3.       So far: We assume that intrinsics are very accurate!

4.       What is the influence here?

5.       → constraint is perfect

6.       → cannot match the point cloud correctly because real-word geomer is not perfectly perpendicular.

7.       → Estimate the intrinsics.

8.       How to constraint GMM?

9.       Their normals The normals of the initial point cloud must be the same.

10.   → More general because the above is only valid for planes.

11.   → The points form a surface and can only be moved on the surface.

12.   How to deal with noise?

13.   Intrinsics: Add the K- Matrix into the transformation representing the intrinsic matrix for both the LiDAR and camera features.

14.   One Intrinsic estimation should move all points simultaneously