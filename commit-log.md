# v1 - First commit
21 Aug 2026
Initial commit of working lab 1 code 
- Forward kinematics
- Camera calibration
- Transformation calibration 

# v2 - Canonical restructure 
### 16 Sep 2026
### Restructured to a sensible structure
- Folder contains src/ data/ docs/
- All calibration parameters and results etc are now in data/
- All code in src/

# v3 - Camera calibration fix
### 17 Sep 2026
### Fixed camera calibration issues
- Code writes calibration report showing detected and reprojected corners

# v4 - Modularized, constants separated
### 17 Sep 2026
### Modularized lab 1 code and put all constants in separate file
- Lab 1 code uses the same unified modularization as other labs
- All tuneable constants including board size etc put now in lab_config.py

## v4.1 - Minor fixes
### 17 Sep 2026
### Validation script now prints validation using both calibrated T_ee_cam matrices
- get_transform_3D.py saves the matrix as T_ee_cam_normal.npy
- ransac_calibration.py saves as _ransac.npy
- validation script uses both and compares
