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
- [X] for set of PCDs, can I plot over time
- [X] for 2 sets of PCDs, can I plot over time.
- [ ] can I plot PCD in global space?
    / using transformation from config.yaml seems wrong
    / can I compute using ICP, are the results improved
    / plotting without transformation looks better
- [ ] compute ICP -> compute rotation + translation per sensor pair
    / does it vary per frame
- [ ] assuming reduced ROI, can I remove the constraint? 
    / does the result (what is the result?) improve
    / calibration vs. not calibration
- [ ] can I plot the centroids over each frame of optimization?
- CI/CD



- [ ] can I plot region + ROI
- [ ] plot before + after ROI (generatePCDs)
- [ ] set ROI for each point cloud
    - [ ] you want to plot the globally aligned point clouds 1st
- [ ] remove code that moves centroid in JRMPC (because this already moves them)
- [ ] understand output calibration results
- [ ] should receive 0 calibration error
- [ ] relax code constraints to allow them to move along the plane
- [ ] read 
- [ ] really understand the code