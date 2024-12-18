# High-resolution maps of Arctic surface skin temperature and type
retrieved from airborne thermal infrared imagery collected during
the HALO–(AC)³ campaign

This repository reproduces the analysis and retrievals outlined in Müller et al., in submission to AMT. The repo is divided in three major parts:
1. Surface type retrieval using state-of-the-art supvervised statistical learning; for the final version a random forest classifier is used, but many different models are evaluated
2. Radiative transfer simulations with libRadtran, simulating the cloud-free surface temperature at theflight level of the airplane
3. Comparing/Validating the results with different satellite products (Surface temp.: MODIS; Ice frac.: MODIS-AMSR2) and reanalysis products (ERA5 and CARRA)

   
