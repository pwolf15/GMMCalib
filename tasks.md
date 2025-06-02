/ generate new dataset with known location and extrinsics
/ add noise to generation (https://gitlab.lrz.de/av2.0/carla/-/blob/carla_0.9.15_tum/Docs/ref_sensors.md#lidar-sensor)
/ sensor also outputs timestamp, frame, and transform
/ get exact position of object
/ log conditions of data generation

/ dataset access
* scannet request
* shapenet request
* partnet request

/ cleanup code
/ avoid rereading files for iterations
/ add back move points to mesh

# GMM Calib Summary

* GMMCalib deals with the problem of LIDAR-LIDAR calibration for autonomous vehicles. 
* It extends joint registration of multiple point clouds (JRMPC), by using Gaussian Mixture models to register multiple point clouds more robustly.
* Registration is the process of estimating rigid transformations (rotation + translation) that align different point clouds into a shared reference frame. 
* Traditional methods like ICP, suffer from reference-target bias and poor performance under partial views or noise.
* GMM-based registration models all point clouds as noisy samples drawn from a shared latent structure -- represented by centroids of a Gaussian Mixture Model
* GMM-Calib takes this further: it not only aligns point clouds, but estimates relative sensor extrinsics, refining each sensor's transformation to the shared latent model.

## Approaches for incorporating the CAD model into GMMCalib

* A key extension suggested by the original GMMCalib paper is to improve calibration accuracy by using a geometric prior - for example, a CAD model of the object.
* Normally, GMMCalib initializes with a randomly sampled point cloud or simple goemetry.
* But, if we know the true object shape (from a CAD model), we can use it to guide or constrain the optimization.

### Fixing Centroids

* Simplest approach: freeze GMM centroids to match the CAD mesh (or a downsampled version)
* Problem: ignores real-world deviations (e.g., sensor noise, occlusion, or object deformation), and may reduce flexibility too much.

### Soft geometric constraints

* Constrain GMM updates to stay near the CAD model (e.g., remain on surface, preserve normals)
* allows flexibility while discouraging unrealistic deformations
* Works well when paired with partiail visiblity or known surfaces (like seats, flat tops)

### CAD-Deform Integration

* CAD-Deform deforms a CAD model to fit partial, noisy scans using smoothness, edge sharpness, and semantic priors
* Its robust to occlusion, clutter, and partial views
* In GMMCalib, you could:
    1. Use GMMCalib to estimat initial sensor-to-GMM transforms
    2. Aggregate the latent GMM shape (centroids)
    3. Fit the CAD model to this shape using CAD Deform
    4. Use the deformed CAD model as a new prior to rerun GMM calib, either as a fixed shape or strongly anchored constraint

## Alternatives

* create first for each observation, left and right
* use those for alignment