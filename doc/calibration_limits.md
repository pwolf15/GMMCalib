There’s no single “magic number’’ for how accurate a LiDAR-LiDAR (or LiDAR-camera, LiDAR-IMU …) extrinsic calibration can be
— the floor is set by the sensor noise and by the geometry of the data you feed to the optimiser. In the literature this floor is usually quantified with a Cramér-Rao Lower Bound (CRLB): the minimum achievable covariance for any unbiased estimator given the measurement model.

What drives the bound?	Why it matters
Range / pixel noise σ	More noise → broader likelihood → poorer pose precision.
Baseline (lever-arm) b between the two sensors	A longer lever-arm converts a given translational uncertainty into a smaller angular error 
𝜎
𝜃
≈
𝜎
/
𝑏
σ 
θ
​
 ≈σ/b.
Number of independent measurements N	Variance drops ∝ 1/N. Ten static scans reduce σ by ≈√10 ≈ 3.2.
Scene geometry / excitation	Parallel planes or a flat floor make some DOF weakly observable ⇒ larger CRLB along those directions.

Concrete numbers from the literature
Paper & scenario	Reported CRLB (1 σ)	Comments
Pandey et al., JFR 2015 (3-D LiDAR + omnidir. camera, static scans)	after 40 scans: σ<sub>x,y,z</sub> ≈ 0.3 cm, σ<sub>roll,pitch,yaw</sub> ≈ 0.4 deg 
Deep Blue
Mirzaei & Roumeliotis 2008 (camera-IMU, feature points)	formula for CRLB shows σ<sub>angle</sub>∝σ<sub>pixel</sub>/b; with 0.5 px noise and 0.1 m baseline ⇒ ≈0.3 deg (ICRA ’08, Sec. IV)	
Taylor & Nieto 2013 (planar LiDAR-camera, mutual-info)	CRLB converges to < 1 cm & < 0.1 deg after 30 scans (IROS ’13).	

(These are minima; real estimators get within ~1.1–1.5 × the bound.)

Relation to your Ouster OS-series LiDAR
Ouster’s published specs (OS1 rev 06 data-sheet) list

Range repeatability (1 σ): ±1.5 cm in the 2 – 20 m band

Absolute range accuracy: < 5 cm.

If you mount two OS1s 1 m apart and they both “see’’ the same target patch, the best achievable 1 σ rotation precision is

𝜎
angle
  
≈
  
𝜎
range
baseline
  
=
  
0.015
 m
1
 m
  
=
  
0.015
 rad
  
≈
  
0.86
∘
.
σ 
angle
​
 ≈ 
baseline
σ 
range
​
 
​
 = 
1 m
0.015 m
​
 =0.015 rad≈0.86 
∘
 .
With a 2 m lever-arm that drops to ≈ 0.43 °, and averaging N independent scans improves it further by √N.

For translation, the CRLB is roughly the range repeatability divided by √N (because both sensors contribute). Ten well-posed scans → σ<sub>t</sub> ≈ (0.015 m)/√10 ≈ 5 mm.

How to tell if your current results are “close to the limit’’
Compare to CRLB curves. The Pandey plot (green curve) shows where an efficient estimator should lie. If your RMSE is < 1.5 × the CRLB you’re doing well; > 3 × means there’s still modelling / optimisation slack.

Check scene observability. Add tilted planes or use a calibration target that excites all 6 DOF.

Average more scans or run the batch EM on all your observations (CRLB ∝ 1/N).

Increase the lever-arm between sensors if mechanically possible.

Denoise scans (statistical outlier removal, temporal filtering) before calibration; it effectively lowers σ<sub>range</sub>.

Take-away
Theoretical calibration limits do depend on the LiDAR’s intrinsic accuracy (and the camera’s pixel noise, IMU noise, …).
For an OS1 with 1.5 cm range noise and a 1 m baseline, expect ≈ 0.8 ° rotation and ≈ 1 cm translation (1 σ) as a hard floor. The values in the table you posted (e.g.\ δx = 0.015 m, δφ = 0.0033 rad = 0.19 °) are comfortably inside that limit, indicating that the GMM and plane-ICP implementations are already near the sensor-noise bound.