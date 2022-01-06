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
import matplotlib.patheffects as path_effects
import pyproj
import cartopy.crs as ccrs
import cartopy.feature as cf
import FMM
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from itertools import combinations

#%%
minlon = 4
maxlon = 12
minlat = 44
maxlat = 49
meanlon = np.mean([maxlon,minlon])
meanlat = np.mean([minlat,maxlat])

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

VEL = gaussian_filter(VEL,5)    


# anisotropy of the velocity field
phi2 = np.zeros_like(LON)
a2 = np.zeros_like(LON)
phi2[(LON>meanlon)*(LAT>meanlat)] = 0.
a2[(LON>meanlon)*(LAT>meanlat)] = 3.
phi2[(LON>meanlon)*(LAT<meanlat)] = -90.
a2[(LON>meanlon)*(LAT<meanlat)] = 2.
phi2[(LON<meanlon)*(LAT<meanlat)] = 30.
a2[(LON<meanlon)*(LAT<meanlat)] = 2.

# station locations
sourcex = rndst.uniform(minlon+0.2,maxlon-0.2,100)
sourcey = rndst.uniform(minlat+0.2,maxlat-0.2,100)
sources = np.column_stack((sourcex,sourcey))
pairs = np.array(list(combinations(sources,2)))
sortind = np.lexsort((pairs[:,1][:,0],pairs[:,0][:,1],pairs[:,0][:,0]))
pairs = pairs[sortind]
stations = np.unique(np.vstack((pairs[:,0],pairs[:,1])),axis=0)

#%%

if True:
    # testplot
    proj = ccrs.Mercator()
    fig = plt.figure(figsize=(12,10))
    axm = fig.add_subplot(111,projection=proj)
    cbar = axm.contourf(LON,LAT,VEL,levels=30,cmap=cm.GMT_haxby_r,transform = ccrs.PlateCarree())
    #axm.plot(stat_select[:,1],stat_select[:,0],'kv',transform = ccrs.PlateCarree())
    # the angles argument takes angles in degrees, counter-clockwise from horizontal
    q = axm.quiver(LON[::15,::15],LAT[::15,::15],a2[::15,::15],a2[::15,::15],
                   angles=phi2[::15,::15],width=0.003,headwidth=0,headlength=0,
                   headaxislength=0,scale=150,pivot='middle',transform=ccrs.PlateCarree())
    axm.plot(stations[:,0],stations[:,1],'rv',transform=ccrs.PlateCarree())
    qk = axm.quiverkey(q, X=0.8, Y=0.05, U=2, label='2% anisotropy', labelpos='E',edgecolor='w',linewidth=3)
    qk.text.set_path_effects([path_effects.withStroke(linewidth=3,foreground='w')])
    qk2 = axm.quiverkey(q, X=0.8, Y=0.05, U=2, label='2% anisotropy', labelpos='E',linewidth=0.1)
    axm.coastlines(resolution='50m')
    axm.add_feature(cf.BORDERS.with_scale('50m'))
    axm.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey')
    axm.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey')
    gridlines = axm.gridlines(draw_labels=True)
    plt.colorbar(cbar,shrink=0.5)
    plt.savefig("synthetic_model_simple.png",bbox_inches='tight',dpi=100)
    #plt.show()
    plt.close(fig)

#%%
x = np.linspace(np.min(X),np.max(X),500)
y = np.linspace(np.min(Y),np.max(Y),500)
Xreg,Yreg = np.meshgrid(x,y)

velfu = RegularGridInterpolator((X[0],Y[:,0]),VEL.T)
phifu = RegularGridInterpolator((X[0],Y[:,0]),phi2.T,method='nearest')
anisofu = RegularGridInterpolator((X[0],Y[:,0]),a2.T)

data = np.zeros(len(pairs))

srcxy = p(pairs[:,0][:,0],pairs[:,0][:,1])
rcvxy = p(pairs[:,1][:,0],pairs[:,1][:,1])

eikonal_paths = []

print("Calculating rays with the fast marching method, this will take a while...")
sourcestat = None
for i in range(len(pairs)):
    
    if i%100 == 0:
        print(i,"/",len(pairs))
    
    src = (srcxy[0][i],srcxy[1][i])
    rcv = (rcvxy[0][i],rcvxy[1][i])
    
    if not np.array_equal(src,sourcestat):
        
        xnew,ynew,ttimefield = FMM.calculate_ttime_field(X[0],Y[:,0],VEL*1000.,src,
                                                     refine_source_grid=True,
                                                     pts_refine=5)       

        xshift = X[0][0]-xnew[0]
        yshift = Y[:,0][0]-ynew[0] 
        
        sourcestat = src
    
            
    paths = FMM.shoot_ray(xnew,ynew,ttimefield,src,rcv)
    if len(paths) != 1:
        raise Exception("")
    path = paths[0]
    if np.isnan(path).any():
        raise Exception("nan in path")

    centers = path[:-1]+np.diff(path,axis=0)/2.
    isovels = velfu(centers)
    pathdirs = np.arctan2(np.diff(path[:,1]),np.diff(path[:,0]))/np.pi*180.
    pathphi = phifu(centers)
    pathaniso = anisofu(centers)
    
    pathvels = isovels * (1+pathaniso/100.*np.cos(2*(pathphi-pathdirs)/180.*np.pi))            
    
    pathdists = np.sqrt(np.sum(np.diff(path,axis=0)**2,axis=1))
    az,baz,dist_real = g.inv(pairs[i][0,0],pairs[i][0,1],pairs[i][1,0],pairs[i][1,1])
    #print(dist_real,np.sum(pathdists),np.sqrt(np.sum((np.array(src)-np.array(rcv))**2)))
    #pause

    pathtimes = pathdists/pathvels
    
    data[i] = np.sum(pathtimes)/1000.
    
    eikonal_paths.append(path)

#%%
# we create two datasets from the data vector, one with good data and one with
# less good data. we assign these two datasets the periods 5s and 25s.
# the period influences only how the eikonal tomography code calculates
# the wavelength theshold. no actual finite frequency simulation.

# add error
data_w_error1 = data + np.random.normal(scale=data/200.)
data_w_error2 = data + np.random.normal(scale=data/80.)

# remove some measurements
pairidx = rndst.choice(np.arange(len(data)),int(len(data)*0.05),replace=False)
data_w_error1[pairidx] = np.nan
pairidx = rndst.choice(np.arange(len(data)),int(len(data)*0.2),replace=False)
data_w_error2[pairidx] = np.nan

# add one bad stations
bad_station_index = np.random.choice(np.arange(len(stations)),1,replace=False)
bad_measurements = 0
for i in range(len(data)):
    if (pairs[i][0,0] in stations[bad_station_index].flatten() or 
        pairs[i][1,0] in stations[bad_station_index].flatten()):
        bad_measurements += 1
        data_w_error1[i] += np.random.uniform(-10,10)

# add four bad stations
bad_station_index = np.random.choice(np.arange(len(stations)),4,replace=False)
bad_measurements = 0
for i in range(len(data)):
    if (pairs[i][0,0] in stations[bad_station_index].flatten() or 
        pairs[i][1,0] in stations[bad_station_index].flatten()):
        bad_measurements += 1
        data_w_error2[i] += np.random.uniform(-10,10)    

with open("example_dataset_simple_anisotropic.dat","w") as f:
    f.write("# example dataset from synthetic model\n")
    f.write("# header line 2\n")
    f.write("# Periods: 5.0 25.0 \n")
    f.write("# lat1 lon1 lat2 lon2 ttime5s ttime25s\n")
    for i in range(len(data)):
        f.write("%7.3f\t%7.3f\t%7.3f\t%7.3f\t%10.5f\t%10.5f\n" 
                %(pairs[i][0,1],pairs[i][0,0],pairs[i][1,1],pairs[i][1,0],
                  data_w_error1[i],data_w_error2[i]))

#%%
if True:
    # testplot
    proj = ccrs.Mercator()
    fig = plt.figure(figsize=(12,10))
    axm = fig.add_subplot(111,projection=proj)
    cbar = axm.contourf(LON,LAT,VEL,levels=30,cmap=cm.GMT_haxby_r,transform = ccrs.PlateCarree())
    #axm.plot(stat_select[:,1],stat_select[:,0],'kv',transform = ccrs.PlateCarree())
    # the angles argument takes angles in degrees, counter-clockwise from horizontal
    q = axm.quiver(LON[::15,::15],LAT[::15,::15],a2[::15,::15],a2[::15,::15],
                   angles=phi2[::15,::15],width=0.003,headwidth=0,headlength=0,
                   headaxislength=0,scale=150,pivot='middle',transform=ccrs.PlateCarree())
    for path in eikonal_paths:
        lonpath,latpath = p(path[:,0],path[:,1],inverse=True)
        axm.plot(lonpath,latpath,'k',linewidth=0.3,transform=ccrs.PlateCarree())
    axm.plot(stations[:,0],stations[:,1],'rv',transform=ccrs.PlateCarree())
    axm.plot(stations[bad_station_index,0],stations[bad_station_index,1],'yv',ms=14,transform=ccrs.PlateCarree())    
    qk = axm.quiverkey(q, X=0.8, Y=0.05, U=2, label='2% anisotropy', labelpos='E',edgecolor='w',linewidth=3)
    qk.text.set_path_effects([path_effects.withStroke(linewidth=3,foreground='w')])
    qk2 = axm.quiverkey(q, X=0.8, Y=0.05, U=2, label='2% anisotropy', labelpos='E',linewidth=0.1)
    axm.coastlines(resolution='50m')
    axm.add_feature(cf.BORDERS.with_scale('50m'))
    axm.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey')
    axm.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey')
    gridlines = axm.gridlines(draw_labels=True)
    plt.colorbar(cbar,shrink=0.5)
    plt.savefig("synthetic_model_simple_rays.png",bbox_inches='tight',dpi=100)
    #plt.show()
    plt.close(fig)