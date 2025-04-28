<div align='center'>

# StereoSense
<img src="res/logo.png" width="150" />

<strong>Complete flow for 3D reconstruction form stereo cameras</strong>  

</div>

#### Conda installation
```bash
# clone project
git clone https://github.com/lus105/StereoSense.git
# change directory
cd StereoSense
# update conda
conda update -n base conda
# create conda environment
conda create --name StereoSense python=3.11
# activate conda environment
conda activate StereoSense
# install requirements
pip install -r requirements.txt
```
<strong>Note:</strong> for using complete flow (main.py), install [basler pylon.](https://www.baslerweb.com/en/software/pylon/?srsltid=AfmBOooUIwLYSjNfoSDrSVLIKNl0GcDOSuO1PzaT0-Hp7pFtrDHgTb_H)

#### Instructions (use with basler cameras)
1. Gather calibration data with basler cameras: ```python src/stereo_grab_basler.py ```
2. Run stereo camera calibration: ```notebooks/1.0_Calibrate.ipynb ```. Change constants to your specific ones.
3. Download [model](https://drive.google.com/file/d/1OhkN9eJKYKqpmAayoDoskqX-ZyZSvjs-/view?usp=sharing) and place inside models/ directory.
4. Run ```python main.py ```
5. Once the configs are loaded, press 'c' to capture frames. Results will be saved in output/ directory.

#### Instructions (use without camera)
1. Create your own camera calibration files (refer to ```notebooks/1.0_Calibrate.ipynb ```)
2. Download [model](https://drive.google.com/file/d/1OhkN9eJKYKqpmAayoDoskqX-ZyZSvjs-/view?usp=sharing) and place inside models/ directory.
3. Grab sample images (left and right) and place inside data/samples directory.
4. Run ```notebooks/2.0_Stereo_inference.ipynb ```

#### Notes

The input size of the model is 800x640 (hxw).

#### Expected result

<div align='center'>
<img src="res/image_left.png" width="400" />
<img src="res/disparity_map.png" width="400" />
<img src="res/pcl.png" width="400" />
</div>