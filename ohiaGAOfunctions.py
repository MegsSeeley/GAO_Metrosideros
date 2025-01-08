import rasterio
import glob
import numpy as np
from rasterio.enums import Resampling
import rasterio.warp
import matplotlib.pyplot as plt
from osgeo import gdal
from matplotlib.lines import Line2D
import pandas as pd



# Import plot site data, apply shade, ndvi, and canopy height mask. Export result as new file.


def filterSites(site, ndvi_thresh=0.7, height_thresh=2):
    # Import Plot Data
    file = rasterio.open(glob.glob('ohiaPlots/'+site+'/orig/*_refl.tif')[0])
    refl = file.read()
    
    #Mask NDVI
    red = file.read(30)
    nir = file.read(51)
    ndvi = np.where(nir==0, 0, ((nir-red)/(nir+red)))
    mask = ndvi < ndvi_thresh
    mask_reshape = np.broadcast_to(mask,(refl.shape))
    refl = np.where(mask_reshape, -9999, refl)
    
    #Mask Height
    pathMask = glob.glob('ohiaPlots/'+site+'/orig/*tch.tif')
    mask = rasterio.open(pathMask[0])
    mask = mask.read()
    mask = mask < height_thresh
    mask_reshape = np.broadcast_to(mask.T,(refl.shape))
    refl = np.where(mask_reshape,-9999, refl)
    
    #Mask Shade
    pathMask = glob.glob('ohiaPlots/'+site+'/mask_crop/*shademask*.tif')
    mask = rasterio.open(pathMask[0])
    mask = mask.read()
    mask = mask !=1
    mask_reshape = np.broadcast_to(mask.T,(refl.shape))
    refl = np.where(mask_reshape, -9999, refl)

    # Export cleaned site
    with rasterio.open('ohiaPlots/'+site+'/orig/'+site+'_filtered.tif', 'w', **file.meta) as dst:
        dst.write(refl)
        
# Import GAO data cube as numpy, remove water bands, optionally brightness-normalize data

def importData(site, start=5, norm = True):
    # Define bands to keep
    bands = np.arange(start,214)
    bands = np.delete(bands, np.where((bands>=144) & (bands<166)))
    bands = np.delete(bands, np.where((bands>=99) & (bands<110)))
    
    # Reshape data (flatten to 2d) and remove no data
    for b in bands:
        tmpdat = site.read(indexes=[b])
        tmpdat = tmpdat.reshape(tmpdat.shape[0],-1)
        tmpdat = tmpdat[tmpdat != -9999]
        tmpdat = tmpdat[tmpdat != np.nan]
        tmpdat = tmpdat.reshape(1,tmpdat.shape[0])
        if (b == start):
            fullDat = tmpdat
        else:
            fullDat = np.append(fullDat, tmpdat, axis =0)
    
    del tmpdat
    
    if norm == True:
    # Normalize spectra
        outdat = fullDat / np.linalg.norm(fullDat, axis=0) 
    else:
        outdat = fullDat
    return outdat
    
# Fill in water bands with nan for plotting

def specFill(indat):
    outdat=indat[0:92]
    
    emp = np.empty(14)
    emp[:]=np.nan
    
    outdat=np.append(outdat, emp)
    
    outdat = np.append(outdat, indat[94:127])

    emp = np.empty(22)
    emp[:]=np.nan
    
    outdat=np.append(outdat, emp)

    outdat = np.append(outdat, indat[128:])
   
    return outdat
    
# Get mean and sd to plot spectra

def getMeanSD(site):
    for b in range(site.shape[0]):

        meanrefl = np.nanmean(site[b,:])
        sd = np.nanstd(site[b,:])

        s1= meanrefl.astype(np.float)+sd.astype(np.float)
        s2 = meanrefl.astype(np.float)-sd.astype(np.float)

        if (b == 0):
            meandat = meanrefl
            sd1dat = s1
            sd2dat = s2
        else:
            meandat = np.append(meandat, meanrefl)
            sd1dat = np.append(sd1dat, s1)
            sd2dat = np.append(sd2dat, s2)
    return meandat, sd1dat, sd2dat
    
    
# Calculate PCA. To apply the pca to another site, replace sameSite arguement with alternative dataset

def pca(dat, sameSite = 'yes'):
    C = np.cov(dat)
    D,V = np.linalg.eig(C)
    if sameSite == 'yes':
        reflPCA = np.dot(V.T, dat)
    else:
        reflPCA = np.dot(V.T, sameSite)
    return reflPCA, V

# Print the variance explained by each PC

def explVar(pcaDat):
    reflPCA_df = pd.DataFrame(pcaDat.T)
    varExp = reflPCA_df.var().div(float(reflPCA_df.var().sum()))

    print('Explained Variance: PC1 ' + str(round(varExp[0]*100, 2)) 
          + '%, PC2 ' + str(round(varExp[1]*100, 2)) 
          + '%, PC3 ' + str(round(varExp[2]*100, 2))
          + '%, PC4 ' + str(round(varExp[3]*100, 2))
          + '%, PC5 ' + str(round(varExp[4]*100, 2))
          + '%, PC6 ' + str(round(varExp[5]*100, 2))
          +"%")
          
# Plot 3D plot of 3 PCs (PC argument specifies which ones)

def plotPC(reflPCA, PC, col, custom_legend = "default", labs = ['YL', 'YH', 'ML', 'MH', 'OL', 'OH']):
    fig = plt.figure(1,figsize = (8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(reflPCA[PC[0],:],reflPCA[PC[1],:],reflPCA[PC[2],:], marker='o', c = col)

    if custom_legend == "default":
        custom_lines = [Line2D([0], [0], marker='o', color='red', label='Scatter',
                                  linestyle='None', markersize=4),
                        Line2D([0], [0], marker='o', color='darkred', label='Scatter',
                                  linestyle='None', markersize=4),
                        Line2D([0], [0], marker='o', color='blue', label='Scatter',
                                  linestyle='None', markersize=4),
                        Line2D([0], [0], marker='o', color='midnightblue', label='Scatter',
                                   linestyle='None', markersize=4),
                        Line2D([0], [0], marker='o', color='limegreen', label='Scatter',
                                  linestyle='None', markersize=4),
                        Line2D([0], [0], marker='o', color='forestgreen', label='Scatter',
                                  linestyle='None', markersize=4)]
    else:
        custom_lines = custom_legend

    ax.legend(custom_lines, labs, bbox_to_anchor=(0, 1), loc='upper right')
    plt.xlabel('Principal Component - ' + str(PC[0]),fontsize=10)
    plt.ylabel('Principal Component - ' + str(PC[1]),fontsize=10)
    ax.set_zlabel('Principal Component - '+ str(PC[2]),fontsize=10)

    plt.show()