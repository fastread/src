What is FASTREAD?
-----
FASTREAD is a tool to support primary study selection in systematic literature review.

Latest Versions:

- On Github repo: [https://github.com/fastread/src](https://github.com/fastread/src).
- In the Seacraft repository: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.852663.svg)](https://doi.org/10.5281/zenodo.852663)

Cite as:
``` 
@article{Yu2019,
title = "FAST2: An intelligent assistant for finding relevant papers",
journal = "Expert Systems with Applications",
volume = "120",
pages = "57 - 71",
year = "2019",
author = "Zhe Yu and Tim Menzies",
keywords = "Active learning, Literature reviews, Text mining, Semi-supervised learning, Relevance feedback, Selection process"
}


@Article{Yu2018,
author="Yu, Zhe
and Kraft, Nicholas A.
and Menzies, Tim",
title="Finding better active learners for faster literature reviews",
journal="Empirical Software Engineering",
year="2018",
month="Mar",
day="07",
issn="1573-7616",
doi="10.1007/s10664-017-9587-0",
url="https://doi.org/10.1007/s10664-017-9587-0"
}
```

Setting up FASTREAD
-----

1. Setting up Python:
  + We use anaconda by continuum.io (see [Why?](https://www.continuum.io/why-anaconda))
    - We won't need the entire distribution. [Download](http://conda.pydata.org/miniconda.html) a Python 2.7 (or Python 3) & install a minimal version of anaconda.
  + Make sure you select add to PATH during install.
  + Next, run `setup.bat`. This will install all the dependencies needed to run the tool. Or:
  + If the above does not work well. Remember you only need a Python 2.7 (now also support Python 3) and three packages listed in `requirements.txt` installed. So `pip install -r requirements.txt` will work.

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
  + Click **Target: Choose File** button to select your csv file in *workspace > data*. Wait a few seconds for the first time. Once the data is successfully loaded, you will see the following:
  ![](https://github.com/fastread/src/blob/master/tutorial/screenshots/load.png?raw=yes)

2.5. [Better ways to start the review](#better-ways-to-start-the-review)
  
3. Begin reviewing studies:
  - Check the box before **Enable Estimation**.
  - A simple search with two or three keywords can help find **Relevant** studies fast before any training starts.
![](https://github.com/fastread/src/blob/master/tutorial/screenshots/BM25.png?raw=yes)
  - Choose from **Relevant**, **Irrelevant**, or **Undetermined** for each study and hit **Submit**.
  - Hit **Next** when you want a to review more.
  - Statistics are displayed as **Documents Coded: x/y (z)**, where **x** is the number of relevant studies retrieved, **y** is the number of studies reviewed, and **z** is the total number of candidate studies.
  - When **x** is greater than or equal to 1, an SVM model will be trained after hitting **Next**. From now on, different query strategies can be chosen.
  - It is suggested to keep using **Uncertain** until the highest probability score for **Certain** is greater than 0.9 or no **Relevant** studies can be found throught **Uncertain** (switch to **Certain** at that point of time).
  - keep reviewing studies until you think most relevant ones have been retrieved. (If **Estimation** is enabled, stop when **x** is close to or greater than 0.95 (or 0.90) of the estimated number of **Relevant** studies.)

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

### Double checking previous labels:

Now we allow users to recheck their previously labeled results and change their decisions. Therefore human errors/concept drift can be handled. Model learned so far is also used to suggest which labels are most suspicious. 
Two options added:
  - **Labeled Pos**: recheck the studies previously labeled as **Relevant**. Sorted by the level of disagreement between current model prediction and human label.
  - **Labeled Neg**: recheck the studies previously labeled as **Irrelevant**. Sorted by the level of disagreement between current model prediction and human label.
  ![](https://github.com/fastread/src/blob/master/tutorial/screenshots/recheck.png?raw=yes)
It is recommended to recheck the top 10 suspicious labels every 50 studies reviewed.
  - **Latest**: change latest submitted labels.
  

### Run simulations of FASTREAD on labeled datasets:
[simulate.py](https://github.com/fastread/src/blob/master/src/simulate.py)


  
Version Logs
-----
Dec 5, 2016. **v1.0.0** The very first, basic version is released.

May 14, 2017. **v1.1.0** Features of UPDATE/REUSE are edited to allow FASTREAD import previously exported data to bootstrap a new review.

Jun 29, 2017. **v1.2.0** Estimate the number of **Relevant** studies in the pool. Enabling **Estimation** will slow down the training, but provide the following benefits:
 - number of **Relevant** studies will be estimated, thus helps to decide when to stop; 
 - probability scores will be more accurate.

Aug 01, 2017. **v1.3.0** Core algorithm updated to utilize both Weighting and aggressive undersampling.

Aug 28, 2017. **v1.4.0** Integrated as FAST2.

Nov 15, 2017. **v1.5.0** Allow user to change their decision on previous labels. Machine suggestions used to efficiently handle human errors or concept drift.

Jan 28, 2020. **v1.6.0** Added Support for Python 3. Show latest labeled items so that it is easy for the users to change their decisions. 
 
