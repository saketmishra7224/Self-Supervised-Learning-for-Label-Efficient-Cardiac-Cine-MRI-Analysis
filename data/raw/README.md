# ACDC Dataset (Automated Cardiac Diagnosis Challenge)

## 1. Overview & Official Source
- **Dataset Name**: Automated Cardiac Diagnosis Challenge (ACDC)
- **Challenge / Event**: MICCAI 2017 Cardiac Segmentation & Diagnosis Challenge
- **Official Webpage**: [https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html)
- **Official Data Repository**: Human Heart Project (CREATIS Laboratory, INSA Lyon)
- **Repository URL**: [https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb)
- **Download API Endpoint**: `https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/folder/63721d7073e9f0047faa0525/download`

## 2. Acquisition Method
- **Method**: Atomic patient-wise streaming via the official Girder REST API.
- **Authentication**: Publicly accessible; no manual credentials or paywalls required.
- **Download Date**: 2026-09-05 22:23:10
- **Archive Name**: `training.zip`
- **Archive Location**: `data/raw/training.zip`
- **Archive Size**: 1555.75 MB (1,631,323,057 bytes)
- **Extraction Location**: `data/raw/ACDC/`

## 3. Dataset Contents & Structure
The primary training cohort contains **100 patients** (patient001 to patient100) evenly distributed across 5 clinical groups (20 subjects each):
1. **NOR**: Normal cardiac function
2. **MINF**: Previous Myocardial Infarction
3. **DCM**: Dilated Cardiomyopathy
4. **HCM**: Hypertrophic Cardiomyopathy
5. **ARV**: Abnormal Right Ventricle

### Directory Organization:
```text
data/raw/
  |-- training.zip                             # Original downloaded archive
  |-- dataset_download_metadata.json           # Machine-readable provenance record
  |-- README.md                                # This document
  `-- ACDC/
        `-- training/
              |-- patient001/
              |     |-- Info.cfg               # Group, Height, Weight, ED, ES
              |     |-- patient001_4d.nii.gz   # 4D Cine volume (X, Y, Z, T)
              |     |-- patient001_frame01.nii.gz
              |     |-- patient001_frame01_gt.nii.gz
              |     |-- patient001_frame12.nii.gz
              |     `-- patient001_frame12_gt.nii.gz
              |-- patient002/
              `-- ... patient100/
```

## 4. Verification Results
- **Patients Found**: 100
- **Image Volumes / Files**: 300
- **Segmentation Masks**: 200
- **Verification Status**: `VERIFIED_VALID`
- **Corrupted Files**: 0
- **Dimensions Compatibility**: Verified (ED and ES images match corresponding ground truth masks).
- **Label Encoding**: Raw ACDC format (0=BG, 1=RV, 2=Myo, 3=LV), ready for project remapping.
