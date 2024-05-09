import ac3airborne
import intake
from ac3airborne.tools import flightphase
import os

ac3cloud_username = os.environ['AC3_USER']
ac3cloud_password = os.environ['AC3_PASSWORD']

credentials = dict(user=ac3cloud_username, password=ac3cloud_password)

# load intake catalog and flight segments
cat = intake.open_catalog('/projekt_agmwend/home_rad/Joshua/LIM_intake/catalog.yaml')
meta = ac3airborne.get_flight_segments()

mission = 'HALO-AC3'
platform = 'P5'
flight_id = 'HALO-AC3_P5_RF09'

ds_gps_ins = cat[mission][platform]['BBR_KT19'][flight_id].to_dask()


# for flight_id, flight in meta['HALO-AC3']['HALO'].items():        
#     campaign, airplane, rf = flight_id.split('_')
#     if len(flight['co-location']) > 0: 
#         print(flight['co-location'])
#         ds_gps = cat[campaign][airplane]['GPS_INS'][flight_id](**credentials).to_dask()

