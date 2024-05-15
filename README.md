This repository contains (most of) the code used during my masters project. Any more questions please feel free to contact me.

Data:

The data is not in this repository, but can be found on the CPOM server. The Kmedoid labels for SAR data May-October 2018 (I believe they are split into months, and then one with every month's data but subsampled)
is located at "/home/spowell/SAR_Kmedoids/label_kmedoids2Sam.mat". I believe Wenxuan should also have a copy of this. The path to the SAR data is already stated in the below files but if required can be found at
"/cpnet/li2_cpdata/SATS/RA/S3A/L2/THEMATIC/BC005/SI". The OLCI data is not currently on the server and so what I did was just manually search for an image of interest for plotting purposes on copernicus. 
The corresponding track can be found trivially since they are already collocated.

Files :
- S3_Tandem.py - Author - Connor Nelson

    File used for downloading some of the OLCI tandem data onto the CPOM server. Requires the SI polygons and a few other files to function properly, all the files required to run can be found on the server at "/home/spowell/S3TandemData/S3Tandem".
    For other queries regarding this file please contact Connor.
- SAR_Kmedoids.py
  
    Please run this file on the CPOM server. It loads existing SAR altimetry data from the CPOM server, disregarding echoes which are not flagged as sea ice or lead. It then calculates several
    waveform parameters as defined in https://tc.copernicus.org/articles/17/809/2023/tc-17-809-2023.pdf and uses these parameters to perform kmedoid classification. In the array that loads the data,
    the print statements that are commented out are for finding the start and end index of a particular track you are interested in (usually for plotting purposes)
- SAR_training.py
  
    File used for training a simple model based off of Kmedoid labels. This assumes you have already manually plotted the labels over OLCI images before hand to identify which classes are leads.
    Crucial also to identify classes which are clearly sea ice as the counter examples or else the model will get confused. Ensure that you are loading in the correct data which corresponds to your kmedoid labels (change indexes in the loop as appropriate).
    File also uses model to make some simple predictions on some new data you give it.
- Remove_outliers.py
  
  File to remove outliers from the data. Includes some fiddly indexing used once I resent Wenxuan some of the data with outliers removed. 'start' and 'end' are simply the indices where a particular track (that you are interested in for plotting over OLCI) begins and finishes.
  In my code these were found when originally loading the data in the SAR_Kmedoids file as explained above.
  The saved array 'no_outliers_indices' is crucial to have so that you can plot the resulting kmedoid labels correctly.
- Find_OLCI.py
  
  File which essentially runs the same indexing as described in Remove_outliers.py
- Overlay_S3_track.ipynb
  
    A very messy notebook (apologies) wich is mainly used for plotting Kmedoid labels (a lot of the code is obsolete). If you have a given OLCI image you are interested in and have located the corresponding track,
    this code will plot the centre points of the tracks on top of the OLCI image. To plot Kmedoid labels, the notebook can automatically remove points not flagged as lead or sea ice, but to remove the outliers which are also discounted in the kmedoid process,
    you must have a 'no_outliers_indices' file which essentially tells you which echoes the labels actually correspond to. This was pretty fiddly to perform so be careful and ensure you are plotting the labels correctly or else you will mess up the visual
    analysis of the labels ontop of the OLCI data beneath. If you have questions about this file (and I suspect you might given the state of it) don't hesitate to contact me
- OLCI_unsupervised_test.ipynb

    Notebook which tested running kmeans on OLCI data, but performance wasn't too great so this was disregarded fairly early on in my project.
