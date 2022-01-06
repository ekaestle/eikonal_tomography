#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:38:31 2020

@author: emanuel
"""

import numpy as np
rndst = np.random.RandomState(seed=11) # this way, always the same random numbers are drawn when restarting the program
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import cm
import matplotlib.patheffects as path_effects
import pyproj
import os
import cartopy.crs as ccrs
import cartopy.feature as cf
import FMM
from scipy.interpolate import griddata,RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from itertools import combinations
from PIL import Image,ImageOps

#%%

if not os.path.exists("syntest_desert"):
    os.makedirs("syntest_desert")

# load satellite image
try:
    image = Image.open('Desert_art.jpg')
except:
    import urllib.request
    print("Downloading Image from ESA... (C opyright Copernicus Sentinel data (2015)/ESA, CC BY-SA 3.0 IGO )")
    urllib.request.urlretrieve("http://www.esa.int/var/esa/storage/images/esa_multimedia/images/2015/07/desert_art/15536691-1-eng-GB/Desert_art.jpg","./Desert_art.jpg")
    image = Image.open('Desert_art.jpg')
gray_image = ImageOps.grayscale(image)
imgarray = np.array(gray_image).astype(float)
imgarray = gaussian_filter(imgarray,sigma=3)
imgarray[imgarray<50.] = 50.
imgarray[imgarray>210.] = 210.
#plt.figure()
#plt.pcolormesh(imgarray)

#load anisotropic patches
patches = Image.open('patches.png')
patches = ImageOps.grayscale(patches)
patches = np.array(patches).astype(float)
#patches = gaussian_filter(patches,sigma=1)
uniquevals,counts = np.unique(patches,return_counts=True)
uniquevals = uniquevals[counts.argsort()[::-1]]
patches[np.abs(patches-uniquevals[0])<=5] = uniquevals[0]
patches[np.abs(patches-uniquevals[1])<=5] = uniquevals[1]
patches[np.abs(patches-uniquevals[2])<=5] = uniquevals[2]
patches[(patches!=uniquevals[0])*(patches!=uniquevals[1])*(patches!=uniquevals[2])] = 0.
patches[patches==uniquevals[1]] = 1.
patches[patches==uniquevals[2]] = 2. 
lons = np.linspace(3,20,np.shape(patches)[1])
lats = np.linspace(41,50,np.shape(patches)[0])
patches_interpolator = RegularGridInterpolator((lons,lats), patches.T,method='nearest',
                                               bounds_error=False,fill_value=0.)
# transform patches into python paths
from svg.path import parse_path
paths = []
lineno = 0
with open("patches.svg","r") as f:
    #test = f.readlines()
    for i,line in enumerate(f.readlines()):
        if "transform" in line:
            scale = float(line.split("(")[-1].split(")")[0])
            lineno = i
        if 'd="m ' in line.lower():
        #if 'd="M ' in line:
            if i == lineno+2:
                scale = scale
                #print("using scale")
            else:
                scale = 1.
            path_string = line.split('"')[1]
            path_data = parse_path(path_string)
            path = [(path_data[0].point(0.).real*scale,path_data[0].point(0.).imag*scale)]
            for element in path_data[1:]:
                for j in np.linspace(0,1,10):
                    pnt = (element.point(j).real*scale,element.point(j).imag*scale)
                    if pnt == path[-1]:
                        continue
                    path.append(pnt)
            path = np.vstack(path)
            paths.append(path)
            
minx = np.min(np.vstack(paths)[:,0])
maxx = np.max(np.vstack(paths)[:,0])
miny = np.min(np.vstack(paths)[:,1])
maxy = np.max(np.vstack(paths)[:,1])
# save to file
pathslonlat = []
for path in paths:
    pathx = np.interp(path[:,0],np.linspace(minx,maxx,len(lons)),lons)
    pathy = np.interp(path[:,1],np.linspace(miny,maxy,len(lats)),lats)
    pathslonlat.append(np.column_stack((pathx,pathy)))
np.save("syntest_desert/aniso_patches_outlines.npy",pathslonlat)
    
# plt.figure()
# for path in pathslonlat:
#     plt.plot(path[:,0],path[:,1])

#%% create velocity model from desert image
imgarray = imgarray.astype(float)
lons = np.linspace(0,21,np.shape(imgarray)[1])
lats = np.linspace(39,52,np.shape(imgarray)[0])
image_interpolator = RegularGridInterpolator((lons,lats), imgarray.T)


lons = np.linspace(0.,21,294)
lats = np.linspace(39,52,260)
LON,LAT = np.meshgrid(lons,lats)
syndata = np.column_stack((LON.flatten(),LAT.flatten(),np.zeros_like(LON.flatten())))
syndata[:,2] = image_interpolator((LON,LAT)).flatten()
syndata[:,2] -= np.min(syndata[:,2])
syndata[:,2] /= np.max(syndata[:,2])
syndata_base = syndata[:,2].copy() # in range 0 - 1
patches = patches_interpolator((LON,LAT))

# add anisotropy of the velocity field
phi2 = np.zeros_like(LON)
a2 = np.zeros_like(LON)

a2[patches==1] = 2.
a2[patches==2] = 2.
phi2[patches==1] = -90.
phi2[patches==2] = 180.

# save to file
np.save("syntest_desert/phi2syn.npy",phi2)
np.save("syntest_desert/a2syn.npy",a2)

# create a test plot
if True:
    # testplot
    plt.ioff()
    proj = ccrs.Mercator()
    fig = plt.figure(figsize=(12,10))
    axm = fig.add_subplot(111,projection=proj)
    cbar = axm.contourf(LON,LAT,syndata[:,2].reshape(np.shape(LON)),levels=30,cmap=cm.GMT_haxby_r,transform = ccrs.PlateCarree())
    #axm.plot(stat_select[:,1],stat_select[:,0],'kv',transform = ccrs.PlateCarree())
    for path in pathslonlat:
        axm.plot(path[:,0],path[:,1],'w--',transform = ccrs.PlateCarree())
    # the angles argument takes angles in degrees, counter-clockwise from horizontal
    q = axm.quiver(LON[::5,::5],LAT[::5,::5],a2[::5,::5],a2[::5,::5],angles=phi2[::5,::5],width=0.005,headwidth=0,headlength=0,headaxislength=0,scale=140,pivot='middle',transform=ccrs.PlateCarree())
    qk = axm.quiverkey(q, X=0.8, Y=0.05, U=2, label='2% anisotropy', labelpos='E',edgecolor='w',linewidth=3)
    qk.text.set_path_effects([path_effects.withStroke(linewidth=3,foreground='w')])
    qk2 = axm.quiverkey(q, X=0.8, Y=0.05, U=2, label='2% anisotropy', labelpos='E',linewidth=0.1)
    axm.coastlines(resolution='50m')
    axm.add_feature(cf.BORDERS.with_scale('50m'))
    axm.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey')
    axm.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey')
    plt.savefig("syntest_desert/synmodel_input.png",dpi=150)
    plt.close(fig)


# read the station locations in lat lon coordinates from the real alparray
# dataset as presented in the article to the publication
input_filelocation = "example_dataset_rayleigh.txt"
indata = np.loadtxt(input_filelocation)

# calculate the synthetic data for different anomaly strenghts
# use all available station pairs      
for anomaly_strength in [5,15,25]:
    
    syndata[:,2] = 3.8 * (1+(syndata_base-0.5)/(50./anomaly_strength))

    np.save("syntest_desert/synmodel_desert_%d.npy" %anomaly_strength,syndata)
    
    print("creating synthetic data for model with anomaly strength",anomaly_strength)
    print("FMM path calculation will take a while...")
    print("receivers outside map boundary are ignored (warnings are not a problem)")

    
    #%%    
    coords = indata[:,:4]   
    stations = np.vstack((indata[:,:2],indata[:,2:4]))
    stations = np.unique(stations,axis=0)
    stations = stations[(stations[:,0]>np.min(LAT))*(stations[:,0]<np.max(LAT))*
                        (stations[:,1]>np.min(LON))*(stations[:,1]<np.max(LON))]
    stat_select = stations[rndst.choice(np.arange(len(stations)),size=int(len(stations)*0.5),replace=False)]       
       
    # projecting the data
    central_lon = np.around(np.mean(lons),1)
    central_lat = np.around(np.mean(lats),1)
    
    g = pyproj.Geod(ellps='WGS84')
    p = pyproj.Proj("+proj=tmerc +datum=WGS84 +lat_0=%f +lon_0=%f" %(central_lat,central_lon))
    
    stats_xy = p(stat_select[:,1],stat_select[:,0])
    srcxy = (stats_xy[0][0],stats_xy[1][0])
    rcvxy = np.column_stack((stats_xy[0][1:],stats_xy[1][1:]))
    
    X,Y = p(LON,LAT)
    VEL = syndata[:,2].reshape(np.shape(X))
    
    # we need a regular grid
    xfine = np.linspace(np.max(X[:,0]),np.min(X[:,-1]),len(X[0])*2)
    yfine = np.linspace(np.max(Y[0]),np.min(Y[-1]),len(Y[:,0])*2)
    Xfine,Yfine = np.meshgrid(xfine,yfine)
    VELfine = griddata((X.flatten(),Y.flatten()),VEL.flatten(),(Xfine,Yfine),method='cubic')
    LONfine,LATfine = p(Xfine,Yfine,inverse=True)
    velfu = RegularGridInterpolator((xfine,yfine),VELfine.T)
    
    # for convenience, make the same with the anisotropic grid
    PHIfine = griddata((X.flatten(),Y.flatten()),phi2.flatten(),(Xfine,Yfine),method='linear')
    Afine = griddata((X.flatten(),Y.flatten()),a2.flatten(),(Xfine,Yfine),method='linear')
    
    phifu = RegularGridInterpolator((xfine,yfine),PHIfine.T)
    anisofu = RegularGridInterpolator((xfine,yfine),Afine.T)
        
    # set this to True if you want to see a test plot of the ray paths in the model
    if False:
        xnew,ynew,ttimefield = FMM.calculate_ttime_field(Xfine[0],Yfine[:,0],VELfine,
                                                         (srcxy[0],srcxy[1]),
                                                         refine_source_grid=True,
                                                         pts_refine=5)
        
        xshift = xfine[0]-xnew[0]
        yshift = yfine[0]-ynew[0]
        ttimefu = RegularGridInterpolator((xnew+xshift,ynew+yshift),ttimefield.T,bounds_error=False,fill_value=np.nan)
        ttimes = ttimefu((rcvxy[:,0],rcvxy[:,1]))/1000.
        measurements = np.column_stack((stat_select[1:][~np.isnan(ttimes)],ttimes[~np.isnan(ttimes)]))
        measurements = np.vstack((measurements,[stat_select[0,0],stat_select[0,1],0.0]))
        dists = np.sqrt((rcvxy[:,0]-srcxy[0])**2+(rcvxy[:,1]-srcxy[1])**2)
        vels = dists/1000./ttimes
        print("mean vel:",np.nanmean(vels),"maxvel:",np.nanmax(vels),"minvel:",np.nanmin(vels))
        
        Xnew,Ynew = np.meshgrid(xnew,ynew)
        
        paths = FMM.shoot_ray(xnew,ynew,ttimefield,(srcxy[0],srcxy[1]),rcvxy)
        pathttimes = []
        for path in paths:
            if np.isnan(path).any():
                pathttimes.append(np.nan)
                continue
            if (path[:,0]>np.max(xfine)).any() or (path[:,0]<np.min(xfine)).any() or (path[:,1]>np.max(yfine)).any() or (path[:,1]<np.min(yfine)).any():
                pathttimes.append(np.nan)
                continue
            centers = path[:-1]+np.diff(path,axis=0)/2.
            isovels = velfu(centers)
            pathdirs = np.arctan2(np.diff(path[:,1]),np.diff(path[:,0]))/np.pi*180.
            pathphi = phifu(centers)
            pathaniso = anisofu(centers)
            
            pathvels = isovels * (1+pathaniso/100.*np.cos(2*(pathphi-pathdirs)/180.*np.pi))
                
        #    plt.figure()
        #    ax = plt.gca()
        #    plt.pcolormesh(Xfine,Yfine,VELfine)
        #    plt.quiver(Xfine[::50,::50],Yfine[::50,::50],Afine[::50,::50],Afine[::50,::50],
        #               angles=PHIfine[::50,::50],headwidth=0,headlength=0,headaxislength=0,scale=180,pivot='middle')
        #    plt.plot(path[:,0],path[:,1])
        #    ax.set_aspect('equal')
        #    plt.show()
            
            pathdists = np.sqrt(np.sum(np.diff(path,axis=0)**2,axis=1))
            #raise Exception("this introduces an error! better sum(pathdists*1./pathvels)")
            pathtimes = pathdists/pathvels
            pathttimes.append(np.sum(pathtimes))
        pathttimes = np.array(pathttimes)/1000.
        
        print("max ttime difference (isotropic vs anisotropic):",np.nanmax(np.abs(pathttimes-ttimes)))
        
        aniso_measurements = np.column_stack((stat_select[1:][~np.isnan(pathttimes)],pathttimes[~np.isnan(pathttimes)]))
      
        
        plt.figure()
        plt.pcolormesh(Xfine/1000.,Yfine/1000.,VELfine)
        plt.plot(stats_xy[0]/1000.,stats_xy[1]/1000.,'kv',ms=6)
        plt.plot(rcvxy[83,0]/1000.,rcvxy[83,1]/1000.,'rv',ms=6)
        plt.plot(srcxy[0]/1000.,srcxy[1]/1000.,'yv',ms=4)
        plt.contour(Xnew/1000.,Ynew/1000.,ttimefield,levels=30)
        plt.show()
        
        
        # testplot
        proj = ccrs.Mercator()
        fig = plt.figure(figsize=(10,8))
        axm = fig.add_subplot(111,projection=proj)
        cbar = axm.pcolormesh(LONfine,LATfine,VELfine,cmap=cm.GMT_haxby_r,transform = ccrs.PlateCarree())
        axm.scatter(measurements[:,1],measurements[:,0],c=measurements[:,2],vmin=0.,vmax=np.max(measurements[:,2]),transform = ccrs.PlateCarree())
        for pathidx in rndst.choice(np.arange(len(paths)),20,replace=False):
            path = paths[pathidx]
            if np.isnan(path).any():
                continue
            centers = path[:-1]+np.diff(path,axis=0)/2.
            isovels = velfu(centers)
            pathdirs = np.arctan2(np.diff(path[:,1]),np.diff(path[:,0]))/np.pi*180.
            pathphi = phifu(centers)
            pathaniso = anisofu(centers)    
            pathanomalies = pathaniso*np.cos(2*(pathphi-pathdirs)/180.*np.pi)       
            lonpath,latpath = p(centers[:,0],centers[:,1],inverse=True)
            #plt.plot(lonpath,latpath,color='black',transform = ccrs.PlateCarree())
            cbar2 = plt.scatter(lonpath,latpath,c=pathanomalies,vmin=-3,vmax=3,cmap=plt.cm.PiYG,s=1,transform = ccrs.PlateCarree())
        LONnew,LATnew = p(Xnew,Ynew,inverse=True)
        axm.contour(LONnew,LATnew,ttimefield/1000.,levels=40,vmin=0.,vmax=np.max(measurements[:,2]),transform = ccrs.PlateCarree())
        axm.coastlines(resolution='50m')
        axm.add_feature(cf.BORDERS.with_scale('50m'))
        axm.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey')
        axm.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey')
        plt.colorbar(cbar,shrink=0.3)
        axmi = axm.inset_axes((0.02,0.02,0.02,0.4))
        plt.colorbar(cbar2,cax=axmi)
        plt.show()
        
        measurements = measurements[:-1] # remove central station
        # end of testplot creation
        
    #%%
    # LOOP FOR THE TOTAL DATASET
    with open(input_filelocation,"r") as f:
        header = f.readline()
        header += f.readline()
        header += f.readline()
        header += f.readline()
        
    data = indata[:,:4]
    sortind = np.lexsort((data[:,2],data[:,1],data[:,0]))
    statcoords = data[sortind]
    
    az,baz,dist_real = g.inv(statcoords[:,1], statcoords[:,0], statcoords[:,3], statcoords[:,2]) 
    
    srcxy = p(statcoords[:,1],statcoords[:,0])
    rcvxy = p(statcoords[:,3],statcoords[:,2])
    dist_proj = np.sqrt((srcxy[0]-rcvxy[0])**2 + (srcxy[1]-rcvxy[1])**2)
    
    distortion_factor = dist_proj/dist_real
    
    sourcestat = np.array([0.,0.])
    syn_ttimes = np.ones(len(statcoords))*np.nan
    for i,statpair in enumerate(statcoords):
        
        if i%1000==0:
            print(i,"/",len(statcoords))
            
        if (statpair[:2]!=sourcestat).any():
            srcxy = p(statpair[1],statpair[0])
            
            try:
                xnew,ynew,ttimefield = FMM.calculate_ttime_field(Xfine[0],Yfine[:,0],VELfine,
                                                             (srcxy[0],srcxy[1]),
                                                             refine_source_grid=True,
                                                             pts_refine=5)
            except:
                #print("source is not within the coordinate limits")
                continue        
        
            xshift = xfine[0]-xnew[0]
            yshift = yfine[0]-ynew[0] 
            
            sourcestat = statpair[:2]
        
        rcvxy = p(statpair[3],statpair[2])
        paths = FMM.shoot_ray(xnew,ynew,ttimefield,(srcxy[0],srcxy[1]),rcvxy)
        if len(paths) != 1:
            raise Exception("")
        path = paths[0]
        if np.isnan(path).any():
            continue
        if (path[:,0]>np.max(xfine)).any() or (path[:,0]<np.min(xfine)).any() or (path[:,1]>np.max(yfine)).any() or (path[:,1]<np.min(yfine)).any():
            continue
        centers = path[:-1]+np.diff(path,axis=0)/2.
        isovels = velfu(centers)
        pathdirs = np.arctan2(np.diff(path[:,1]),np.diff(path[:,0]))/np.pi*180.
        pathphi = phifu(centers)
        pathaniso = anisofu(centers)
        
        pathvels = isovels * (1+pathaniso/100.*np.cos(2*(pathphi-pathdirs)/180.*np.pi))            
        
        pathdists = np.sqrt(np.sum(np.diff(path,axis=0)**2,axis=1))
        pathtimes = pathdists/pathvels
        
        pathttime = np.sum(pathtimes)/1000.
        
        syn_ttimes[i] = pathttime/distortion_factor[i]
    
    vel_errors = rndst.normal(loc=0.,scale=0.02,size=len(syn_ttimes))
    #vel_errors = 0.
    ttimes_w_error = syn_ttimes * (1 - 1/3.8*vel_errors)
    
    
    outdata_aniso = np.ones_like(indata)*np.nan
    outdata_aniso[:,:4] = statcoords
    for col_idx in range(np.shape(indata)[1]):
        if col_idx <= 3:
            continue
        nonan_idx = ~np.isnan(indata[sortind,col_idx])
        outdata_aniso[nonan_idx,col_idx] = ttimes_w_error[nonan_idx]
        
    np.savetxt("syntest_desert/synthetic_measurements_desert_model_%s.txt" 
               %(anomaly_strength),outdata_aniso,header=header[:-1],comments='')
