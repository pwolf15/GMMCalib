# gmmcalib
* core logic for joint registration of point clouds
* jgmm: performs joint GMM registration
* generatePCDs: load/simulate point clouds
* create_gt: initialize GMM centroids using uniform PC within cube and apply offset
* v: point clouds
* jgmm: joint registration across all point clouds by iteratively aligning to GMM model
    * X: updated GMM model
    * TV: transformed views of input point clouds
        * transformed versions of all input clouds, after aplying latest rot + translation Updates
    * AllT: transformation matrices over iterations (complete history of rotation + translation)
        * list of tuples (rotation, translation)
    * pk: probabilities associated with each GMM component
* transformation estimation
    * T_1, T_2: compute transformation between two sets of point clouds (from two sensors)
    * T_calib: extrinsic calibration; calculate relative transformation between two sets
    * T_final: averages calibration transformations
* flow
    1. jgmm: probabilistic registration
    2. GMM centroids = latent representation of of object
    3. transformatioin estimation: alignments computed to determine sensor calibration
* observation: single capture from given sensor
* in default example, there are 3 observations per sensor (6 total across 2 sensors)
* each corresponding observation for the sensors is assumed at same time instant

# model generator
* generates GMM 
* initialize means, covs, weights

# transform PCDs
* transform point cloud data (pcds)
* handle rotation, translation, scaling for pc alignment

# generate PCDs
* load synthetic data
# generate_gt
* generate/define ground truth
* compare estimated transformation w/ known values

# todo

* the first thing I need to do, is to determine the ROI between the 2 overlapping fields
/ add visualize option for GMM calib (open3d vs. plotly)
/ 1. need to isolate problem with data (ROI vs. transformation)
/ 2. plot registration of sets after n iterations
/ 3. plot final GMM means (optional plot initial X)

- [X] for set of PCDs, can I plot over time
- [X] for 2 sets of PCDs, can I plot over time.
- [X] can I plot PCD in global space?
- [ ] assuming reduced ROI, can I remove the constraint? 
    / does the result (what is the result?) improve
    / calibration vs. not calibration
- [X] can I plot the centroids over each frame of optimization?
- CI/CD
- [X] display final calibration
- [X] remove code that moves centroid in JRMPC (because this already moves them)
- [ ] understand output calibration results: 
    / normal
    / normal (fix centroids)
    / cube 
    / cube (fix centroids)
- [ ] should receive 0 calibration error
- [ ] relax code constraints to allow them to move along the plane
- [ ] read 
- [ ] really understand the code

/ 2.12
/ pre-processing code to allow you to select subset of fields
/ reduce ROI so it only includes cubes
/ try with and without noise (should do better with noise counter-intuitively)
/ try with and without moving centroids

100 iterations, cube config

Config: /workspace/config/cube_config.yaml
Data: /workspace/cube_data
ROI: [-10, -7, -5] -> [-6, -4, 0]
Transform Sensor 1: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
Transform Sensor 2: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
Final calibration (T_final): [[ 9.91463413e-01  1.30383242e-01  7.14803838e-04  6.40506252e-01]
 [-1.30385178e-01  9.91451599e-01  4.84057823e-03 -1.06807078e+00]
 [-7.75631267e-05 -4.89245604e-03  9.99988029e-01 -2.71920272e-02]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]