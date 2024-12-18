import npyscreen
import os
from make_frames import make_frames


processed_path = '/projekt_agmwend/home_rad/Joshua/Mueller_et_al_2024/data/cluster/output_sam'

processed_files = [os.path.join(processed_path, f) for f in os.listdir(processed_path) if f.endswith('.nc')]

processed_files.sort()

display_files = [f.split('/')[-1] for f in processed_files]

class selectFile(npyscreen.Form):
    def create(self):
        self.title        = self.add(npyscreen.TitleText, name='Select a file for processing into pushbroom format:')
        self.file = self.add(npyscreen.TitleSelectOne, scroll_exit=True, max_height=20, name=' ', values = display_files)


def myFunction(*args):
    F = selectFile(name = "Select file for processing")
    F.edit()

    selected_file = F.file.get_selected_objects()[0]

    selected_index = display_files.index(selected_file)

    print('Selected files:\n' + processed_files[selected_index] + '\n' + processed_files[selected_index])

    processed_file = processed_files[selected_index]

    outpath = make_frames(processed_file, FPS=60)


    print('Processing complete', flush=True)

    os.chdir(outpath)

    video_path = outpath + '_output.mp4'    
    print('Making video @ ' + outpath, flush=True)
    os.system(f'ffmpeg -framerate 5 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p {video_path}')
    print('Video complete', flush=True)

if __name__ == '__main__':

    print(npyscreen.wrapper_basic(myFunction))

