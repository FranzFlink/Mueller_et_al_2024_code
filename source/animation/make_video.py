import npyscreen
import os
from make_frames import make_frames


processed_path = '/projekt_agmwend/home_rad/Joshua/Mueller_et_al_2024/data/cluster/output_sam'
processed_files = [os.path.join(processed_path, f) for f in os.listdir(processed_path) if f.endswith('.nc')]
processed_files.sort()
display_files = [f.split('/')[-1] for f in processed_files]

if __name__ == '__main__':
    for processed_file in processed_files:
        ### check if the file is already processed
        processed_file_stump = processed_file.split('/')[-1].split('_full.nc')[0]
        outpath = f'../../plots/ani/{processed_file_stump}'
        if not os.path.isdir(f'../../plots/ani/{processed_file_stump}'):
            print('File already processed, skipping')
            print('Processing ' + processed_file)
            outpath = make_frames(processed_file, FPS=60)
        print('Processing complete')
        os.chdir(outpath)
        outpath = f'/projekt_agmwend/home_rad/Joshua/Mueller_et_al_2024/plots/ani/videos/{processed_file_stump}_output.mp4'

        print('Making video @ ' + outpath)
        os.system(f'ffmpeg -framerate 5 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p {outpath}')
        print('Video complete')
