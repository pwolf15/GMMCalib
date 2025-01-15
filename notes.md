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

