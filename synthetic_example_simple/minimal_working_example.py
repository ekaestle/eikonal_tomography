#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:38:31 2020

@author: emanuel
"""

import numpy as np
rndst = np.random.RandomState(seed=1112) # this way, always the same random numbers are drawn when restarting the program
import matplotlib.pyplot as plt
plt.ioff()
from mpl_toolkits.basemap import cm
import pyproj
import cartopy.crs as ccrs
import cartopy.feature as cf
import FMM
from scipy.interpolate import RegularGridInterpolator, Rbf, griddata
from scipy.ndimage import gaussian_filter
from itertools import combinations

#%% setting up a synthetic model
minlon = 4
maxlon = 12
minlat = 44
maxlat = 49
meanlon = np.mean([maxlon,minlon])
meanlat = np.mean([minlat,maxlat])

# we define a model in lats and lons and project it to x and y coordinates
g = pyproj.Geod(ellps='WGS84')
p = pyproj.Proj("+proj=tmerc +datum=WGS84 +lat_0=%f +lon_0=%f" %(meanlat,meanlon))

LON,LAT = np.meshgrid(np.linspace(minlon,maxlon,500),np.linspace(minlat,maxlat,500))
X,Y = p(LON,LAT)
# make it a regular grid in x,y
x = np.linspace(np.min(X),np.max(X),300)
y = np.linspace(np.min(Y),np.max(Y),300)
X,Y = np.meshgrid(x,y)
LON,LAT = p(X,Y,inverse=True)
VEL = np.ones(np.shape(LON))*3.7
VEL[(LON>5.5)*(LON<8.5)*(LAT>45.3)*(LAT<45.7)] = 4.0
VEL[(LON>10)*(LON<10.2)*(LAT>45.1)*(LAT<47)] = 3.4
VEL[(LON>6.8)*(LON<7.2)*(LAT>45)*(LAT<47.6)] = 3.4
for lon,lat,rad,v in [[8,47,50000,4.0],[6,48,80000,3.4],[10,46,30000,4.0],
                    [8,45.5,18000,3.4],[9,46.3,34000,3.4]]:
    x,y = p(lon,lat)
    VEL[((X-x)**2+(Y-y)**2) < (rad**2)] = v

# smooth the model
VEL = gaussian_filter(VEL,5)    


# add stations
# number of stations
nstations = 100

# station locations
sourcex = rndst.uniform(minlon+0.2,maxlon-0.2,nstations)
sourcey = rndst.uniform(minlat+0.2,maxlat-0.2,nstations)
sources = np.column_stack((sourcex,sourcey))
pairs = np.array(list(combinations(sources,2)))
sortind = np.lexsort((pairs[:,1][:,0],pairs[:,0][:,1],pairs[:,0][:,0]))
pairs = pairs[sortind]
stations = np.unique(np.vstack((pairs[:,0],pairs[:,1])),axis=0)

#%% make a testplot

if True:
    # testplot
    proj = ccrs.Mercator()
    fig = plt.figure(figsize=(12,10))
    axm = fig.add_subplot(111,projection=proj)
    cbar = axm.contourf(LON,LAT,VEL,levels=30,cmap=cm.GMT_haxby_r,transform = ccrs.PlateCarree())
    #axm.plot(stat_select[:,1],stat_select[:,0],'kv',transform = ccrs.PlateCarree())
    axm.plot(stations[:,0],stations[:,1],'rv',transform=ccrs.PlateCarree())
    axm.plot(stations[20,0],stations[20,1],'gv',ms=10,transform=ccrs.PlateCarree())
    axm.coastlines(resolution='50m')
    axm.add_feature(cf.BORDERS.with_scale('50m'))
    axm.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey')
    axm.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey')
    gridlines = axm.gridlines(draw_labels=True)
    plt.colorbar(cbar,shrink=0.5)
    #plt.savefig("synthetic_model_simple.png",bbox_inches='tight',dpi=100)
    plt.show()
    #plt.close(fig)

#%%
# choose a source station
sourceindex = 20
source = stations[sourceindex]

# receivers are all stations minus the source station
receivers = np.delete(stations,sourceindex,axis=0)

# get stations in x and y coordinates
srcxy = p(source[0],source[1])
rcvxy = p(receivers[:,0],receivers[:,1])
rcvxy = np.column_stack(rcvxy)


# X,Y and srcxy are in units of meter, make sure that the VEL array is also in m/s
# get the traveltimefield with the fast marching method
xnew,ynew,ttimefield = FMM.calculate_ttime_field(X[0],Y[:,0],VEL*1000.,srcxy)
# xnew and ynew is in the same units as x and y but slightly shifted so that
# the source is exactly at a grid node

# interpolate the traveltimefield to get the traveltimes at the receiver locations
intp2d = RegularGridInterpolator((xnew,ynew),ttimefield.T)

ttimes = intp2d(rcvxy)

# just for illustration: calculate the paths from the source to all receivers
paths = FMM.shoot_ray(xnew,ynew,ttimefield,srcxy,rcvxy)




#%% make a testplot
if True:
    # testplot
    proj = ccrs.Mercator()
    fig = plt.figure(figsize=(12,10))
    axm = fig.add_subplot(111,projection=proj)
    cbar = axm.contourf(LON,LAT,VEL,levels=30,cmap=cm.GMT_haxby_r,transform = ccrs.PlateCarree())
    for path in paths:
        lonpath,latpath = p(path[:,0],path[:,1],inverse=True)
        axm.plot(lonpath,latpath,'k',linewidth=0.8,transform=ccrs.PlateCarree())
    ttime_cbar = axm.scatter(receivers[:,0],receivers[:,1],c=ttimes,transform=ccrs.PlateCarree())
    axm.plot(source[0],source[1],'rv',ms=10,transform=ccrs.PlateCarree())
    axm.coastlines(resolution='50m')
    axm.add_feature(cf.BORDERS.with_scale('50m'))
    axm.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey')
    axm.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey')
    gridlines = axm.gridlines(draw_labels=True)
    plt.colorbar(ttime_cbar,shrink=0.5,label='traveltimes')
    #plt.savefig("synthetic_model_simple_rays.png",bbox_inches='tight',dpi=100)
    plt.show()
   # plt.close(fig)
   
   
#%%

method = 'rbf'

#dists = np.sqrt((rcvxy[:,0]-srcxy[0])**2+(rcvxy[:,1]-srcxy[1])**2)
#meanvel = dists/ttimes
#ttime_residuals = dists/meanvel

# add also the information at the source station (traveltime = 0)
traveltimes = np.append(0,ttimes)
stations_xy = np.vstack((srcxy,rcvxy))


if method == 'linear':
    ttime_field = griddata(np.column_stack((stations_xy[:,0],stations_xy[:,1])),
                              traveltimes,(X,Y),method='linear')
    
elif method == 'cubic':
    ttime_field = griddata(np.column_stack((stations_xy[:,0],stations_xy[:,1])),
                              traveltimes,(X,Y),method='cubic') 
    
elif method == 'rbf':
    rbfi = Rbf(stations_xy[:,0],stations_xy[:,1],traveltimes,
               function='linear',smooth=0.) 
    ttime_field = rbfi(X,Y)
    
  

#%% make a testplot
if True:
    # testplot
    proj = ccrs.Mercator()
    fig = plt.figure(figsize=(12,10))
    axm = fig.add_subplot(111,projection=proj)
    cbar = axm.contourf(LON,LAT,ttime_field,levels=30,
                        vmin=0,vmax=np.max(ttimes),
                        transform = ccrs.PlateCarree())
    for path in paths:
        lonpath,latpath = p(path[:,0],path[:,1],inverse=True)
        axm.plot(lonpath,latpath,'k',linewidth=0.8,transform=ccrs.PlateCarree())
    ttime_cbar = axm.scatter(stations_xy[:,0],stations_xy[:,1],c=traveltimes,
                             edgecolors='r',vmin=0,vmax=np.max(ttimes),
                             transform=ccrs.PlateCarree())
    axm.plot(source[0],source[1],'rv',ms=10,transform=ccrs.PlateCarree())
    axm.coastlines(resolution='50m')
    axm.add_feature(cf.BORDERS.with_scale('50m'))
    axm.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey')
    axm.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey')
    gridlines = axm.gridlines(draw_labels=True)
    plt.colorbar(ttime_cbar,shrink=0.5,label='traveltimes')
    #plt.savefig("synthetic_model_simple_rays.png",bbox_inches='tight',dpi=100)
    plt.show()
   # plt.close(fig)
   
   
#%% get the phase velocity field

xgridspacing = np.diff(X[0])[0]
ygridspacing = np.diff(Y[:,0])[0]

gradient_field = np.gradient(ttime_field)    
gradient_field[1] /= xgridspacing # x gradient
gradient_field[0] /= ygridspacing # y gradient  

# the velocity field is the inverse of the gradient field
velocity_field = 1./np.sqrt(gradient_field[0]**2+gradient_field[1]**2)


#%% make a testplot
if True:
    colorlevels = np.linspace(np.min(VEL),np.max(VEL),30)
    
    # testplot
    proj = ccrs.Mercator()
    fig = plt.figure(figsize=(16,8))
    axm = fig.add_subplot(121,projection=proj)
    cbar = axm.contourf(LON,LAT,VEL,levels=colorlevels,cmap=cm.GMT_haxby_r,
                        transform = ccrs.PlateCarree())
    #for path in paths:
    #    lonpath,latpath = p(path[:,0],path[:,1],inverse=True)
    #    axm.plot(lonpath,latpath,'k',linewidth=0.8,transform=ccrs.PlateCarree())
    #cbar = axm.scatter(receivers[:,0],receivers[:,1],c=ttimes,
    #                   edgecolors='r',vmin=np.min(VEL),vmax=np.max(VEL),
    #                   cmap=cm.GMT_haxby_r,transform=ccrs.PlateCarree())
    axm.plot(source[0],source[1],'rv',ms=10,transform=ccrs.PlateCarree())
    axm.coastlines(resolution='50m')
    axm.add_feature(cf.BORDERS.with_scale('50m'))
    axm.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey')
    axm.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey')
    gridlines = axm.gridlines(draw_labels=True)
    gridlines.right_labels=False
    plt.colorbar(cbar,shrink=0.5,label='velocities')
    
    axm = fig.add_subplot(122,projection=proj)
    cbar = axm.contourf(LON,LAT,velocity_field/1000.,levels=colorlevels,
                        cmap=cm.GMT_haxby_r,transform = ccrs.PlateCarree())
    #for path in paths:
    #    lonpath,latpath = p(path[:,0],path[:,1],inverse=True)
    #    axm.plot(lonpath,latpath,'k',linewidth=0.8,transform=ccrs.PlateCarree())
    #cbar = axm.scatter(receivers[:,0],receivers[:,1],c=ttimes,
    #                   edgecolors='r',vmin=0,vmax=np.max(ttimes),
    #                   cmap=cm.GMT_haxby_r,transform=ccrs.PlateCarree())
    axm.plot(source[0],source[1],'rv',ms=10,transform=ccrs.PlateCarree())
    axm.coastlines(resolution='50m')
    axm.add_feature(cf.BORDERS.with_scale('50m'))
    axm.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey')
    axm.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey')
    gridlines = axm.gridlines(draw_labels=True)
    gridlines.right_labels=False
    plt.colorbar(cbar,shrink=0.5,label='velocities')
    #plt.savefig("synthetic_model_simple_rays.png",bbox_inches='tight',dpi=100)
    plt.show()
   # plt.close(fig)
    
