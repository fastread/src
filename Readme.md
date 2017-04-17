What is MAR?
-----
MAR stands for machine assisted reading, it is a tool to support primary study selection in systematic literature review.

Latest Versions:

- On Github repo: [https://github.com/ai-se/MAR](https://github.com/ai-se/MAR).
- In the Seacraft repository: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.203136.svg)](https://doi.org/10.5281/zenodo.203136)

Cite as:
``` 
@misc{zhe yu_2016, 
      title={ai-se/MAR: MAR v1.0.2}, 
      DOI={10.5281/zenodo.203136}, 
      publisher={Zenodo}, 
      author={Zhe Yu}, 
      year={2016}, 
      month={Dec}}
```

Setting up MAR
-----

1. Setting up Python:
  + We use anaconda by continuum.io (see [Why?](https://www.continuum.io/why-anaconda))
    - We won't need the entire distribution. [Download](http://conda.pydata.org/miniconda.html) a Python 2.7 version & install a minimal version of anaconda.
  + Make sure you select add to PATH during install.
  + Next, navigate to MAR, and run `setup.bat`. This will install all the dependencies needed to run the tool.
  + If the above does not work well. Remember you only need a Python 2.7 and three packages listed in `requirements.txt` installed.

2. Running script:
  + Navigate to *MAR > src* and run `index.py`.
  + If all is well, you'll be greeted by this:
  ![](https://github.com/ai-se/MAR/blob/master/tutorial/screenshots/run.png?raw=yes)

4. The Interface:
  + Fire up your browser and go to [`http://localhost:5000/hello/`](http://localhost:5000/hello/). You'll see a page like below:
  ![](https://github.com/ai-se/MAR/blob/master/tutorial/screenshots/start.png?raw=yes)
    
Use MAR
-----

1. Get data ready:
  + Put your candidate list (a csv file) in *MAR > workspace > data*.
  + The candidate list can be as the same format as the example file *MAR > workspace > data > Hall.csv* or a csv file exported from [IEEExplore](http://ieeexplore.ieee.org/).
  
2. Load the data:
  + Click **Choose File** button to select your csv file in *MAR > workspace > data*. Wait a few seconds for the first time. Once the data is successfully loaded, you will see the following:
  ![](https://github.com/ai-se/MAR/blob/master/tutorial/screenshots/load.png?raw=yes)
  
3. Begin reviewing studies:
  - choose from **Relevant**, **Irrelevant**, or **Undetermined** for each study and hit **Submit**.
  - hit **Next** when you want a to review more.
  - statistics are displayed as **Documents Coded: x/y (z)**, where **x** is the number of relevant studies retrieved, **y** is the number of studies reviewed, and **z** is the total number of candidate studies.
  - when **x** is greater than or equal to 1, an SVM model will be trained after hitting **Next**.
  - rather than **Random** sampling, you can now select **certain** or **uncertain** for reviewing studies. **certain** returns the studies that the model thinks are most possible to be relevant while **uncertain** returns the studies that model is least confident to classify.
  - keep reviewing studies until you think most relevant ones have been retrieved.
  
4. Auto review:
  + If your data contains true label, like Hall.csv does, another button called **Auto Review** will be enabled. By clicking it, it automatically labels all your current studies (depending on the selection **Random**, **certain** or **uncertain**).

4. Plot the curve:
  + Click **Plot** button will plot a **Relevant studies retrieved** vs. **Studies reviewed** curve.
  + Check **Auto Plot** so that every time you hit next, a curve will be automatically generated.
  + You can also find the figure in *MAR > src > static > image*.
  ![](https://github.com/ai-se/MAR/blob/master/tutorial/screenshots/plot.png?raw=yes)
  
5. Export csv:
  + Click **Export** button will generate a csv file with your coding in *MAR > workspace > coded*.

6. Restart:
  + Click **Restart** button will give you a fresh start and loose all your previous effort on the current data.
  
7. Remember to click **Next** button:
  + User data will be saved when and only when you hit **Next** button, so please don't forget to hit it before you want to stop reviewing.
  
Version Logs
-----
Dec 5, 2016. **v1.0.0** The very first, basic version is released.

Dec 6, 2016. **v1.0.1** Add one feature under testing, which can predict when the review process should stop.
![](https://github.com/ai-se/MAR/blob/master/tutorial/screenshots/Curve_Prediction.png?raw=yes)

Dec 14, 2016. **v1.0.2**: 
 - Exported csv file: timestamp added, auto-sorted. 
 - "How to Read Less: Better Machine Assisted Reading Methods for Systematic Literature Reviews" is submitted to IST journal and uploaded to [arxiv](https://arxiv.org/abs/1612.03224).
