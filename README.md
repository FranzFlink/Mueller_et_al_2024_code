# High-resolution maps of Arctic surface skin temperature and type retrieved from airborne thermal infrared imagery collected during the HALO–(AC)³ campaign

This repository reproduces the figures, data and code outlined in Müller et al., (in submission to AMT). 
### Abstract 

Two retrieval methods for the determination of Arctic surface skin temperature and surface type based on radiance measurements from the thermal infrared (TIR) imager VELOX (Video airbornE Longwave Observations within siX channels) are introduced. VELOX captures TIR radiances in terms of brightness temperatures in the atmospheric window for wavelengths from 7.7 μm to 12 μm in six spectral channels. It was deployed on the High Altitude and LOng Range research aircraft (HALO) during the HALO–(AC)³ airborne field campaign conducted in the framework of the Arctic Amplification: Climate Relevant Atmospheric and SurfaCe Processes and Feedback Mechanisms (AC)³ research program. The measurements were taken over the Fram Strait and the central Arctic in March and April 2022. To derive the surface skin temperature, radiative transfer simulations assuming cloud-free atmospheric conditions were performed, quantifying the influence of water vapour on the measured brightness temperature. Since this influence was negligible, it was possible to apply a single-channel retrieval of the surface skin temperature. The derived surface skin temperatures were compared with data from the MODerate-resolution Imaging Spectroradiometer (MODIS). Furthermore, a pixel-by-pixel surface classification into types of open water, sea-ice water mixture, thin sea ice, and snow-covered sea ice was developed using a random forest algorithm. When the resulting sea-ice concentrations are compared with satellite data, a mean absolute error (MAE) of 5 % is obtained. In addition, the classified pixels where aggregated into segments of the same surface type, providing different segment size distributions for all surface types. When grouped by the distance to the sea ice edge, the segment size distribution shows a shift, favoring fewer but larger floes in the direction of the pack ice. 

[Figure7](plots/publish/figure07.png!)

Overview of surface classification and segmentation results for the pushbroom-like image captured on 04. April 2022 from 13:36:14
to 13:38:31 UTC. a) Broadband brightness temperature TB,1 (7.7 μm–12 μm) as pushbroom-like image b) Initial surface type classification
using the random forest algorithm (RFA), identifying open-water, thin ice, and snow-covered ice. c) Initial segmentation using the segmentanything
model (SAM), with numbered segments representing the ten largest areas for illustration. d) Final surface type classification: the
most common surface type within each segment from (c) was assigned, and a surface skin temperature threshold was used to sort the icewater
mix class (IWM) form OW. (e) Final segmentation, where new segments were assigned to all connected regions of the same surface
type derived from (d), with the largest segments again highlighted by their respective numbers.











To produce the data, all scripts in the `run` directory need to be executed
