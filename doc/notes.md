# GMMCalib

## Problem

* LiDAR-LiDAR calibration
* LiDAR calibration determines the relative pose (rotation and translation) between multiple LiDAR sensors
* probabilistic: model uncertainty about data using probability distributions
* registration: determine rigid transformation (rotation and translation) between point sets / clouds
* automatic: does not require human intervention
* target-based: requires presence of reference object to help constrain or drive optimization
* extrinsic calibration: rotation and translation of sensor relative to reference frame or another sensor
* Gaussian mixture model: probabilistic model which uses set of Gaussians to model latent space
* joint registration: perform point set registration without selecting target point set

### new ideas

* distance-based metric for evaluation
* local minima problem of local registration

### SOTA registration

* iterative closest point (ICP)
* limitations of ICP
    * requires spatial pose difference be small; it's sensitive to initialization and parametrization
    * one of the point clouds must serve as reference frame; this is an algorithmic bias
* local registration method: given rough initial pose, methods like ICP can determine fine / accurate alignment
* latent shape: point cloud in reference frame; latent means that it's not directly observed or known
* drawbacks of joint registration
* reconstruction: reproducing accurate shape or model given set of observatoins
* point-to-point ICP
* point-to-plane ICP
* generalized ICP
* geometric prior: known geometric model (prior knowledge of scene)


### related work

#### extrinsic sensor calibration

* motion-based: solve spatial relationship between sensors; frames follow rigid body motion; solve hand-eye problem
* appearance-based: match features in environment; registration problem
* GMM follows appearance-based approach
* Procrustes alignment: minimize error when applying rotation and translation to each point in point cloud

#### non-probabilistic registration

* Iterative closest point: non-probabilistic registration
    * point-to-point: direct point-to-point correspondences; robust if correspondences knwon
        * Zhang: camera-to-LIDAR 
        * target-less 3D lIDAR calirbation
    * point-to-plane
        * good with planar objects
        * algorithm for calibrating depth camera to 2D laser rangefinder
        * IRLS for plane-fitting
        * Hu: target-less extrinsic self-calirbation for LiDAR and stereo cameras
        * CROON: targetless
        * LiDAR-to-LiDAR using ICPN
    * generalized ICP
        * multiple 3D LiDAR: merge measurements into a calirbated point cloud
        * LiDAR-Link: target-less, non-overlapping LiDAR

#### probabilistic registration

* address handling of outliers and noise
* EM-ICP: joint registration of multiple registrations using EM algorithm
* NDT algorithm: Join registration on data in voxel grid registratioin
* EM-ICP, NDT require reference point cloud
* GMMReg: probabilistic registration
* JRMPC: no reference point cloud
* HGMR, MLMD focus on efficiency
* direct registration: don't operate on data directly; require pre-processing for coarse pose estimation

### methodology

* good initial estimate by aligning respective LiDAR in vehicle reference frame
* expected positiona nd orientation of sensors reltative to vehicle frame is given

#### Gmm-based joint registration

* given N observations (half from first sensor; half from second), determine transformations applied to each observation / point set as well as compute spatial relationship between corresponding sensors
* model GMM parameters
    * theta: M components with following parametrs
        * probability: confidence that each point belongs to given Gaussian distribtuion
        * covariance: uncertainty of each mixture component
        * mean: Gausssain component
* transformations: N poses (R,t) between observer frame and calibrated frame
* minimize expected log likelihood
    * Q-function: choose model parameters theta to maximize average/expected value of log probability that the model would jointly genrate
        * O: observed data/point clouds
        * Z: hidden labels Z; which Gaussian component each component point came from
        * expectation taken over all possible label assignments Z under current estimate of parameters
        * E-step: estimate label probabilities (distirbution over Z)
        * M-step pick new theta to mazimize this expected log-likelihood

### LiDAR-toLIDAR calibration

* GMM introduces reference frame R, since both sensor frames transformed into arbitrary reference frame
* differs from ICP: directly solves for transformation from source to target point cloud
* we determine transformation matrix for each observation

### experiment design

* CARLA + real-world
* 3 cubes
* sensors front left and right relative to vehicle origin
* 3 cubes with edges of 0.5 meters

#### CARLA  simulation

* road modeled as flat surface to generate ground level points
* 3 cubes oriented differently
    * placedd in overlapping field of view of both sensors
* sensors
    * hfov: 360
    * vfov: 25 in simulation, 45 in real-world
    * range: 50/120 m
    * channels: 50/128
    * sensor rate: 10 Hz
    * precision +/- 0.01 m

#### real-world

* three cubes in front of vehicle at 10 m distnace, in FoV of sensors
* additional cube at 16 m for validation purposes
* Ouster OS1-128 LiDARs mounted on EDGAR
* no significant weather conditions

### results

#### evaluation

* simulated 100 calibration errors
* point sets from right LiDAR randomly transformed by a roll, pitch, and yaw angle error between +/-3 degrees and translation error of +/- 0.1 m
* true calibration error between frames is known
* ideally inverse of computed extrinsics and known extrinsics is identity 
* in reality you have small delta which gives results
* converting 3x3 rotation matrix into Euler angles, can run into singularity (gimbacl lock) if pitch is +/- 90 degrees
* to avoid this, assume true calibration errors and residual rotations are less than 90 degrees
    * safe assumption if sensors not wildly misaligned
* transformation error = inv(estimated transform from L2 to L1) * ground truth transform from L2 to L1
* compute ICP registration on same point cloud pairs
* delta transformation can be mislead interpretation of accuracy or robustness
* final registration result may be accurate
* 6 DoF for alignment; so not adequate
* distance based metric
    * mean distance error per point with known point correspondences
    * just run this over point clouds in sensor frame
    * take known correspondence, throw it through the ground truth transform and estimate, look at difference and average
    * compute distance metric using local points (roi)
    * compute distance metric using entire scan

#### simulation

* 106 observations per sensor
* 212 total observations
* average point cloud size 1850 points
* total point cloud size (not just ROI) is 12500 points
* compute mean euler angle and translation errors
* calculate rotation (euler) angle and translation (meters) using equation (4)
* correlation between translation and angular errors indicates a compensating effect rather than a calibration errors
* box plots using 100 calibration erros
* 100 observations per sensor => 10000 data points in box plots
* distance-based metric evaluaed on entire point cloud

* fig 7: compute estimated and ground truth points
* form point-wise euclidean error
* bin errors by x-distance of each point
* compute mean and standard eeviation of ek
* slope is angular misalignment
* intercept is remaining translation error
* error vs. range (less accurate farther from the sensor)

#### real world

* 104 observations per sensor
* 1400 pts per observation
* 131000 pts in entire point cloud

#### discussion
* relative large translations in calibration matrix may be due to high angular errors and can be considered compensatory parameters
* additional distance metric helps evaluate robustness

### conclusion

* approach overcomes limitations of ICP
* could be promising for sensor system with non-overlapping FoV
* geometric priors can be used as plausibility check and a constraint to opimzie the extrinsic
* especially useful for real-world scenarios with intrinsic properties of LiDAR sensors

# definitions

* L2_L1_T: takes from L2 frame to L1 frame
* L2_R_T: takes from L2 frame to R frame
* L1_R_T: take from L1 frame to R frame

# equations

## question 1: EM objective

* O: all observed points from every LiDAR scan
* Z: hidden "which-Gaussian generated this point labels" labels
* theta: unkonwns: GMM parameters and every LiDAR pose
    * GMM weights, means, covariances and rigid-body poses
* Z is unkonwn, so formula takes expectation over Z under current parameter guess
* log P(O,Z|theta) is the complete data log-likelihood
    * how likely that parameters theta produced observed points and latent component assignments
* E-step: compute conditional expectation using current theta
* M-step: choose new parameters theta_n+1 that maximize expected log-likelihood f
* in plain english: take average over unknown mixture assignments of the log-likelihood that our model with parameters theta would have produced observed points and those assignments. Then pick theta that maximizes that average

## equation 2: ground-truth extrinsic

* take point in LiDAR-2 frame to LiDAR-1 frame
* take point in LiDAR-2 to R to LiDAR-1
* (L1_R_T)^-1 takes from R to L1
* (L2_R_T) takes from L2 to R
* this is ground-truth extrinsic

## equation 3: computing extrinsics

* computes extrinsic from LiDAR-1 to LiDAR-2 for observation/scan-pair j using reference algorithm r
* this is used to show that many algorithms are compared

## equation 4: registration-error transform

* L2_1_T: true extrinsics (2->1)
* L2_1_T_r_j: estimated extrinsics (2->1)
* (L2_1_T_r,j)^-1: take point from L1 back to L2 using estimated transform
* (L2_1_T): take point from L2 to L1 using ground truth
* if estimate were correct, delta_T would be identity. Any residual rotation/translation is 6-DoF calibration error

## equation 5: distance-based quality metric

* x_jk: kth LiDAR-2 pt in jth-validation observation set O_j
* first term: x_jk in LiDAR-1 after applying ground truth
* second term: inverse of estimated transform; estimated transform actually computes L2->L1
* subtract => 3d error vector for correspondence
* average over all N_j correspondences -> 1 3D-mean error vector for scan perror
* stack mean error for every scan pair
* plain english: each column tells average algorithm r missed by (delta x, delta y, delta z) in that scan
* take norm of columns score how algorithm aligns
* N_2 -> N is second half of observations (those in L2 frame)
* we are just re-indexing the LiDAR-2 observations

# transformation syntax (T)

* see craig robotics pg. 34
* A_B_T: describe frame {B} relative to frame {A}
* it is a transform mapping. It maps B_P -> A_P
* it is a transforma operator
* this paper's syntax is actually wrong
    * {}^{L_2}_{L_1}T: L1 (source) to L2 (target) in Craig's
    * {}^{L_2}_{L_1}T: L2 (source) to L1 (target) in this paper
    * see pg. 4: "Given {}^{L_1}_{R}T and {}^{L_2}_{R}T, which are transformations of the sensor frames to the common reference frame."
    * also see fig. 2 and fig. 5

# code notes

``` python

    # T_1: L1 to world; think that this is the estimate of transform from source (LiDAR) to target (GMM)
    # T_2: L2 to world
    # T_calib: L1 to L2
    # T_2^-1: world to L2
    # T_2^-1 * T_1 will transform points in sensor 1 frame to pts in sensor 2 frame
    T_calib = [np.dot(np.linalg.inv(T_2[i]), T_1[i]) for i in range(len(T_1))]
    T_final = transformPCDs.mean_transform(T_calib)

```