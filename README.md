What is FASTREAD?
-----
FASTREAD is a tool to support primary study selection in systematic literature review.

Latest Versions:

- On Github repo: [https://github.com/fastread/src](https://github.com/fastread/src).
- In the Seacraft repository: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.579740.svg)](https://doi.org/10.5281/zenodo.579740)

Cite as:
``` 
@misc{zhe_yu_2017_546749,
  author       = {Zhe Yu and
                  Tim Menzies},
  title        = {fastread/src: FASTREAD},
  month        = apr,
  year         = 2017,
  doi          = {10.5281/zenodo.546749},
  url          = {https://doi.org/10.5281/zenodo.555970}
}
```

Setting up FASTREAD
-----

1. Setting up Python:
  + We use anaconda by continuum.io (see [Why?](https://www.continuum.io/why-anaconda))
    - We won't need the entire distribution. [Download](http://conda.pydata.org/miniconda.html) a Python 2.7 version & install a minimal version of anaconda.
  + Make sure you select add to PATH during install.
  + Next, run `setup.bat`. This will install all the dependencies needed to run the tool.
  + If the above does not work well. Remember you only need a Python 2.7 and three packages listed in `requirements.txt` installed. So `pip install -r requirements.txt` will work.

2. Running script:
  + Navigate to *src* and run `index.py`.
  + If all is well, you'll be greeted by this:
  ![](https://github.com/fastread/src/blob/master/tutorial/screenshots/run.png?raw=yes)

3. The Interface:
  + Fire up your browser and go to [`http://localhost:5000/hello/`](http://localhost:5000/hello/). You'll see a page like below:
  ![](https://github.com/fastread/src/blob/master/tutorial/screenshots/start.png?raw=yes)
    
Use FASTREAD
-----

1. Get data ready:
  + Put your candidate list (a csv file) in *workspace > data*.
  + The candidate list can be as the same format as the example file *workspace > data > Hall.csv* or a csv file exported from [IEEExplore](http://ieeexplore.ieee.org/).
  
2. Load the data:
  + Click **Choose File** button to select your csv file in *workspace > data*. Wait a few seconds for the first time. Once the data is successfully loaded, you will see the following:
  ![](https://github.com/fastread/src/blob/master/tutorial/screenshots/load.png?raw=yes)
  
3. Begin reviewing studies:
  - choose from **Relevant**, **Irrelevant**, or **Undetermined** for each study and hit **Submit**.
  - hit **Next** when you want a to review more.
  - statistics are displayed as **Documents Coded: x/y (z)**, where **x** is the number of relevant studies retrieved, **y** is the number of studies reviewed, and **z** is the total number of candidate studies.
  - when **x** is greater than or equal to 1, an SVM model will be trained after hitting **Next**.
  - rather than **Random** sampling, you can now select **certain** or **uncertain** for reviewing studies. **certain** returns the studies that the model thinks are most possible to be relevant while **uncertain** returns the studies that model is least confident to classify.
  - keep reviewing studies until you think most relevant ones have been retrieved.
  
3.5. Auto review:
  + If your data contains true label, like Hall.csv does, another button called **Auto Review** will be enabled. By clicking it, it automatically labels all your current studies (depending on the selection **Random**, **certain** or **uncertain**).

4. Plot the curve:
  + Click **Plot** button will plot a **Relevant studies retrieved** vs. **Studies reviewed** curve.
  + Check **Auto Plot** so that every time you hit next, a curve will be automatically generated.
  + You can also find the figure in *src > static > image*.
  ![](https://github.com/fastread/src/blob/master/tutorial/screenshots/plot.png?raw=yes)
  
5. Export csv:
  + Click **Export** button will generate a csv file with your coding in *workspace > coded*.

6. Restart:
  + Click **Restart** button will give you a fresh start and loose all your previous effort on the current data.
  
7. Remember to click **Next** button:
  + User data will be saved when and only when you hit **Next** button, so please don't forget to hit it before you want to stop reviewing.
  
Re-Use FASTREAD
-----

Exported primary study selection results (e.g. *workspace > coded > Hall.csv* ) can be re-applied to bootstrap new selections (e.g. *workspace > data > Wahono.csv* ).

1. Get data ready:
  + Make sure previously exported data is in *workspace > coded*.
  + Or create csv file with the same format as *workspace > coded > Hall.csv* and put it in *workspace > coded*.
  + Put your candidate list (a csv file) for the new selection in *workspace > data*.

2. Load the data for current selection:
  + Click **target: Choose File** button to select your csv file in *workspace > data*. Wait a few seconds for the first time. Once the data is successfully loaded, you will see the following:
  ![](https://github.com/fastread/src/blob/master/tutorial/screenshots/load1.png?raw=yes)

3. Load old data:
  + Click **import: Choose File** button to select your csv file in *workspace > coded*. Wait a few seconds for the first time. Once the data is successfully loaded, you will see the following:
  ![](https://github.com/fastread/src/blob/master/tutorial/screenshots/load2.png?raw=yes)
  
4. Begin reviewing studies:
  - Select **certain** instead of **Random**
  - choose from **Relevant**, **Irrelevant**, or **Undetermined** for each study and hit **Submit**.
  - hit **Next** when you want a to review more.
  - statistics are displayed as **Documents Coded: x/y (z)**, where **x** is the number of relevant studies retrieved, **y** is the number of studies reviewed, and **z** is the total number of candidate studies.
  - If the selection criteria of target and imported review are the same, stay with **certain** selection until finished. Otherwise, change to **reuse** once **x** is greater or equal to 5.
  - keep reviewing studies until you think most relevant ones have been retrieved.

  
Version Logs
-----
Dec 5, 2016. **v1.0.0** The very first, basic version is released.

May 14, 2017. **v1.1.0** Features of UPDATE/REUSE are edited to allow FASTREAD import previously exported data to bootstrap a new review.

Jun 29, 2017. **v1.2.0** Estimate the number of **Relevant** studies in the pool. Enabling **Estimation** will slow down the training, but provide the following benefits:
 - number of **Relevant** studies will be estimated, thus helps to decide when to stop; 
 - probability scores will be more accurate;
 - probability scores can then be used to decide whether to choose **reuse** or **certain** when Re-Using FASTREAD.
 
