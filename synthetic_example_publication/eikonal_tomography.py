#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: emanuel

Run this program with Python 3 installed from a console window as
    python eikonal_tomography.py

You can speed up the calculation by using multiple CPUs if mpi4py is installed
For example, run the program on 4 cores with the command
    mpirun -np 4 python eikonal_tomography.py

"""
"""
###############################################################################
## USER DEFINED PARAMETERS ##
"""

# input file
# input file format
# line1: arbitrary header line, should start with a #
# line2: arbitrary header line, should start with a #
# line3: # Periods: PERIOD1 PERIOD2 PERIOD3 ...
# line4: arbitrary header line, should start with a #
# line5-N: LAT1 LON1 LAT2 LON2 TRAVELTIME[PERIOD1] TRAVELTIME[PERIOD2] ...
# no measurement for a certain station pair is indicated by a nan entry
#
# input_filelocation can be either a single path or a list to several files
# that are treated subsequently, e.g.
# input_filelocation = "/path/to/input_file.txt"
# input_filelocation = ["/path/to/input_file.txt", "../input_file2.dat"]
input_filelocation = ["syntest_desert/synthetic_measurements_desert_model_25.txt",
                      "syntest_desert/synthetic_measurements_desert_model_15.txt",
                      "syntest_desert/synthetic_measurements_desert_model_5.txt",
                      ]

# a logfile and the calculated models will be stored in this output folder
output_location = "./syntest_desert" # will be created if it does not exist yet

# output files are named automatically and start with a freely chosen basename
outfilename_base = ["result_syntest_desert_model25",
                    "result_syntest_desert_model15",
                    "result_syntest_desert_model5",
                    ]                    

# minimum wavelength threshold. Data closer than min_wavelength around the
# source station is removed. Recommended: 1-2
min_wavelength = 1.0

# at very long periods (= long wavelengths), too much of the mapped area might
# get removed because of the min_wavelength threshold. The max_distance value
# gives the maximum radius around that central station that is removed
max_distance = 150. # in km; set to a very high value to switch off

# areas of the velocity map are discarded that are too far away from the next
# measurement/station. This threshold is controlled by min wavelength.
# The min_distance parameter is a lower threshold to avoid removal of large
# parts of the map region at very short periods 
min_distance = 50. # in km; set to 0 to switch off


# size of gridcells in km
xgridspacing = 5.
ygridspacing = 5.

# In which format are the input source/receiver coordinates?
# currently, the plotting routines assume latlon data meaning that for xy data,
# the plot will be mapped to an arbitrary region.
coordinates = 'latlon' # choose from 'xy', 'latlon'

# only important if there is no period defined in the header of the input file.
# otherwise ignored. Period information is used to calculate the wavelength
data_period = 123

# the travel-time fields are interpolated using different functions
# choose between 'linear', 'cubic', 'splineintension', 'rbf-linear',
# 'rbf-multiquadric', 'rbf-thinplate', ... for more see
# (https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Rbf.html)
# splineintension works only with Generic Mapping Tools installed and 
# GMTcommands like surface callable from a console. splineintension needs an
# area limit for the interpolation. The plotarea (see below) limits are used
method = ['rbf-linear',] # recommended 'rbf-linear'
# by adding more values to the list, several models are created with different parameters

# rejecting travel-time measurements if they are causing a curvature in the 
# travel-time field that is more than curvature_threshold standard deviations
# away from the mean
# recommended value: 3; to switch off, set to a very high value
curvature_threshold = [3.,] # max. allowed std deviation
# by adding more values to the list, several models are created with different parameters

# interpolation smoothing. Works The traveltime field is smoothed so that there are
# no velocities higher than 'smoothing_threshold' times the average velocity.
# only has an effect in combination with 'rbf-...' otherwise ignored
# recommended value: 3; to switch off, set to a very high value (999.)
smoothing_threshold = [3.,]
# by adding more values to the list, several models are created with different parameters

# if a traveltime field yields velocities that are greater/smaller than
# (1 +- rejection_threshold) * average velocity, these areas of the field are 
# removed
# recommended value: 1.5 (= max 50% deviation from the mean)
# to switch off, set to a very high value (999.)
rejection_threshold = [1.5,]
# by adding more values to the list, several models are created with different parameters
     
# Apply a gaussian filter of width wavelength/2, truncated at wavelength/2
# Recommended by Lehujeur et al. (2020) "On the validity of Eikonal tomography"
# this only affects the phase velocities, not the travel directions
gaussian_filtering = False # standard is False

# the final, isotropic model can be calculated as the mean or the mode of the 
# model collection. standard is 'mean'
average_type = 'mean' # choose between 'mean' and 'mode'

# minimum number of measurements in each gricell for the calculation of the average
# (gridcells where less measurements are availabe are masked)
minimum_no_of_models = 50
    


# PARAMETERS USED FOR THE ISOTROPIC AND ANISOTROPIC MODEL DETERMINATION

# maximum allowed azimuthal gap
max_azimuthal_gap = 60. # in degrees



# PARAMETERS USED FOR THE AZIMUTHAL ANISOTROPY FITTING

# area radius for the anisotropy calculation
# phase velocities and their directions are averaged within that radius
avg_area = 30. # in km

# the phase velocity measurements from different azimuthal directions are put
# into bins and counted. The azimuthal gap must not be larger than
# max_azimuthal_gap degrees. Every bin where less than models_per_bins (see 
# below) values are found is counted as gap
azimuthal_bin_size = 15 # in degrees, recommended between 5 - 20 degrees

# minimum number of independent velocity measurements that have to be in each bin.
# independent means coming from a different central station
models_per_bin = 5 

# you can choose to fit also A1 and PHI1, which represents a type of anisotropy
# whis is fast in one direction and slow in the opposite one. This does not make
# any physical sense (contradicts the reciprocity of the wave equation) but
# can happen if there are measurement errors, non-homogenous noise source distributions
# or in case of large velocity gradients (recommended to be True).
include_A1 = True # choose between True and False


# MISCELLANEOUS AND MODEL PLOTTING

# estimate variance reduction (on a straight ray path, may not be meaningful
# in case of bent rays or rays of finite width). Just for information.
print_variance_reduction = True

# set the plotting area in degrees (minlon,maxlon,minlat,maxlat); can also be set to 'auto'
plotarea = (4,19,42,50)

# you can create an example plot for each individual central station.
# currently only works if the interpolation method is one of the rbf methods
create_example_plots = False

"""
## END OF USER DEFINED PARAMETERS ##
###############################################################################
"""

from mpi4py import MPI
import os, time, subprocess, warnings
import numpy as np
from scipy.interpolate import RBFInterpolator,RegularGridInterpolator,griddata
from scipy.spatial import Delaunay, cKDTree
from scipy.stats import binned_statistic, gaussian_kde
from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter
import matplotlib
if not bool(os.environ.get('DISPLAY', None)):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from matplotlib.collections import LineCollection
import matplotlib.patheffects as path_effects
from mpl_toolkits.basemap import cm
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cf
import pyproj


def gaussian_filt(U,sigma,truncate):
    V=U.copy()
    V[np.isnan(U)]=0
    VV=gaussian_filter(V,sigma=sigma,truncate=truncate)
    
    W=0*U.copy()+1
    W[np.isnan(U)]=0
    WW=gaussian_filter(W,sigma=sigma,truncate=truncate)
    
    return VV/WW


def mode(samples,return_kde=False):
    
    if len(samples) == 0:
        if return_kde:
            return np.nan, None
        else:
            return np.nan
    
    if len(samples) < 10:
        if return_kde:
            return np.mean(samples), None
        else:
            return np.mean(samples)
    
    gkde = gaussian_kde(samples,bw_method=0.6)                            
    # sample the gaussian kde at the following test velocities
    testvels = np.linspace(np.min(samples),np.max(samples),10)
    smoothhist = gkde(testvels)
    
    if return_kde:
        hist_kde = np.column_stack((testvels,smoothhist))
    
    # resample around maximum
    if smoothhist.argmax()==0:
        idxmax=1
    elif smoothhist.argmax()==9:
        idxmax=8
    else:
        idxmax=smoothhist.argmax()
    testvels = np.linspace(testvels[idxmax-1],testvels[idxmax+1],10)
    smoothhist = gkde(testvels)                                                         
    # take the maximum
    mode =  testvels[smoothhist.argmax()]
    
    if return_kde:
        hist_kde = np.vstack((hist_kde, np.column_stack((testvels,smoothhist))))
        hist_kde = hist_kde[hist_kde[:,0].argsort()]
        return mode, hist_kde
    else:
        return mode
    
    

if __name__ == '__main__':

    logfile = os.path.join(output_location,"logfile_eikonal_tomography.txt")
      
    # Initialize MPI
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size() 
    
    # check whether the input file is a string or a list of strings. Expected
    # type is a list of input files that are processed in a loop
    if type(input_filelocation) is not type([]):
        input_filelocation = [input_filelocation]
    if type(outfilename_base) is not type([]):
        outfilename_base = [outfilename_base]
    if len(outfilename_base) != len(input_filelocation):
        raise Exception("input_filelocation and outfilename_base have to be "+
                        "lists of the same length (for each input file, "+
                        " define a separate output filename).")
    outfilename_baselist = outfilename_base.copy()
    input_filelocationlist = input_filelocation.copy()
    
    # process all input files
    for fileidx in range(len(input_filelocation)):
        input_filelocation = input_filelocationlist[fileidx]
        outfilename_base = outfilename_baselist[fileidx]
    
        # initialize variables that are shared between mpi processes
        periods = []
        data = []
        stats = []
        X = []
        Y = []
        p = []
        wavelength_threshold = []
        meanvel = []
        
        if mpi_rank == 0:
            
            # check if output folder exists. Otherwise create.
            if not os.path.exists(output_location):
                os.makedirs(output_location)
                
            with open(logfile, 'a') as f:
                print("\nStarting Eikonal tomography\n",file=f)
                print(time.ctime(),file=f)
                print("Working on "+input_filelocation,file=f)
                print("Working on "+input_filelocation) # print also to console                
                    
            with open(input_filelocation,"r") as f:
                line1 = f.readline()
                if not(line1[0] == '#'):
                       periods = [str(data_period)]
                       with open(logfile, 'a') as f:
                           print("Period: %.1fs" %data_period,file=f)
                else:
                    line2 = f.readline()
                    line3 = f.readline()
                    periods = line3.split()[2:]
                    with open(logfile, 'a') as f:
                        print("Periods:",periods,file=f)
                
            # READ INPUT DATA        
            indata = np.loadtxt(input_filelocation)#[:5000]
            coords = indata[:,:4]
                
            stations = np.vstack((indata[:,:2],indata[:,2:4]))
            stats,uidx = np.unique(stations,return_inverse=True,axis=0)
            
            if coordinates == 'latlon':
                central_lon = np.around(np.mean(indata[:,(1,3)]),1)
                central_lat = np.around(np.mean(indata[:,(0,2)]),1)
                
                # Cartopy is not useful for creating geodetic lines
                g = pyproj.Geod(ellps='WGS84')
                p = pyproj.Proj("+proj=tmerc +datum=WGS84 +lat_0=%f +lon_0=%f" %(central_lat,central_lon))
            
                srcxy = p(indata[:,1],indata[:,0])
                rcvxy = p(indata[:,3],indata[:,2])
                statsxy = p(stats[:,1],stats[:,0])
                az,baz,dist_real = g.inv(indata[:,1], indata[:,0], indata[:,3], indata[:,2]) 
                
                dist_proj = np.sqrt((srcxy[0]-rcvxy[0])**2 + (srcxy[1]-rcvxy[1])**2)
                distortion_factor = dist_proj/dist_real
                if np.max(distortion_factor)>1.03 or np.min(distortion_factor)<0.97:
                    with open(logfile, 'a') as f:
                        print("WARNING: significant distortion by the coordinate projection!",file=f)
                
                #correct traveltimes for distortion (very small effect)
                indata[:,4:-1] = (indata[:,4:-1].T*distortion_factor).T
                
                # regular grid in projected coordinates
                minx,maxx,miny,maxy = (min(statsxy[0])/1000.,
                                       max(statsxy[0])/1000.,
                                       min(statsxy[1])/1000.,
                                       max(statsxy[1])/1000.)   
                x = np.arange(minx, maxx+xgridspacing, xgridspacing)
                y = np.arange(miny, maxy+ygridspacing, ygridspacing)    
                X,Y = np.meshgrid(x,y)
                LON,LAT = p(X*1000.,Y*1000.,inverse=True)
                
                np.save(os.path.join(output_location,"lons.npy"),LON)
                np.save(os.path.join(output_location,"lats.npy"),LAT)
                
                
            elif coordinates == 'xy':
                
                srcxy = indata[:,(0,1)]
                rcvxy = indata[:,(2,3)]
                statsxy = stats
                dist_real = np.sqrt((srcxy[:,0]-rcvxy[:,0])**2 + (srcxy[:,1]-rcvxy[:,1])**2)*1000. # in meters
                minx,maxx,miny,maxy = (min(statsxy[:,0]),
                                       max(statsxy[:,0]),
                                       min(statsxy[:,1]),
                                       max(statsxy[:,1]))   
                x = np.arange(minx, maxx+xgridspacing, xgridspacing)
                y = np.arange(miny, maxy+ygridspacing, ygridspacing)    
                X,Y = np.meshgrid(x,y)
                
                # some dummy values so I can reuse the geographical plotting routines:
                central_lon = np.mean(x)/1000.
                central_lat = np.mean(y)/1000.
                # Cartopy is not useful for creating geodetic lines
                g = pyproj.Geod(ellps='WGS84')
                p = pyproj.Proj("+proj=tmerc +datum=WGS84 +lat_0=%f +lon_0=%f" %(central_lat,central_lon))
                LON,LAT = (X/1000.,Y/1000.)
    
                np.save(os.path.join(output_location,"x.npy"),X)
                np.save(os.path.join(output_location,"y.npy"),Y)
        
            if plotarea == 'auto':
                plotarea = (np.min(LON),np.max(LON),np.min(LAT),np.max(LAT))
        
            # check whether bin parameter is okay
            no_bins = 360./azimuthal_bin_size
            if len(stats)/no_bins < models_per_bin:
                print("Warning! For the anisotropic parameter determination, you have defined",
                      int(no_bins),"azimuthal bins (azimuthal_bin_size = ",azimuthal_bin_size,"). " +
                      "The total number of stations in the dataset is",len(stats),". " +
                      "This leaves on average only",int(len(stats)/no_bins),"independent "+
                      "measurements per bin which is lower than the models_per_bin parameter:",
                      models_per_bin,". Consider lowering the models_per_bin parameter " +
                      "to get meaningful results.")
        
        # share information among MPI processes
        periods = mpi_comm.bcast(periods,root=0)
        stats = mpi_comm.bcast(stats,root=0)
        X = mpi_comm.bcast(X,root=0)
        Y = mpi_comm.bcast(Y,root=0)
        p = mpi_comm.bcast(p,root=0)  
        kdt_grid = cKDTree(np.column_stack((X.flatten(),Y.flatten())))
        
        # create a list of jobs (separate job for each interpolation method,
        # each rejection, smoothing and curvature threshold)
        jobs = []
        if type(method) is not type([]):
            method = [method]
        if type(rejection_threshold) is not type([]):
            rejection_threshold = [rejection_threshold]
        if type(smoothing_threshold) is not type([]):
            smoothing_threshold = [smoothing_threshold]
        if type(curvature_threshold) is not type([]):
            curvature_threshold = [curvature_threshold]
        for mt in method:
            for rt in rejection_threshold:
                for st in smoothing_threshold:
                    for ct in curvature_threshold:
                        jobs.append([mt,rt,st,ct])
        
        # loop over all jobs
        for job in jobs:
            
            method = job[0]
            rejection_threshold = job[1]
            smoothing_threshold = job[2]
            curvature_threshold = job[3]
        
            # loop over all periods
            for pidx,period in enumerate(periods):
                
                #if not float(period) in [6.5]:
                #    continue
                  
                if mpi_rank == 0:
                    print("\n###########################",flush=True)
                    print("Working on period", period)
                    print("Interpolation method:",method)
                    print("curvature threshold: %.2f" %curvature_threshold)
                    print("rejection threshold: %.2f" %rejection_threshold)
                    print("smoothing_threshold: %.2f" %smoothing_threshold) 
                    with open(logfile, 'a') as f:
                        print("\n\n###########################", file=f, flush=True)
                        print("Working on period", period, file=f, flush=True)
                        print("Interpolation method:",method,file=f)
                        print("curvature threshold: %.2f" %curvature_threshold,file=f)
                        print("rejection threshold: %.2f" %rejection_threshold,file=f)
                        print("smoothing_threshold: %.2f" %smoothing_threshold,file=f)
                        
                    input_list = []
                    
                    # SELECT INPUT DATA        
                    data = indata.copy()
                    data = data[:,(0,1,2,3,pidx+4)]
                    
                    # remove nan entries
                    idx = ~np.isnan(data[:,-1])
                    data = data[idx]
            
                    # calculate mean velocities and mean wavelength
                    velocities = dist_real[idx]/data[:,-1]/1000.
                    meanvel = np.mean(velocities)
                    mean_wavelength = float(period)*meanvel
                    
                    # only use data with a minimum distance of min_wavelength
                    # Eikonal tomography is not reliable in proximity to the source station
                    wavelength_threshold = min_wavelength*mean_wavelength
                    if wavelength_threshold > max_distance:
                        wavelength_threshold = max_distance
                        with open(logfile, 'a') as f:
                            print("Warning: taking also station pairs that are closer than %.1f wavelengths" %min_wavelength,file=f)
                    subidx = dist_real[idx]/1000.>=wavelength_threshold
                    data = data[subidx] 
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with open(logfile,'a') as f:
                            print("discarded %d data points which are closer than min_wavelength from the central station (%.1f %%)." %(np.sum(~subidx),np.sum(~subidx)/np.sum(idx)*100),file=f)        
                    
                    # Set True to create a plot of the path geometries
                    # (automatically limited to 5000 paths)
                    if False and coordinates=='latlon':
                        plt.figure(figsize=(12,10))
                        plt.ion()
                        # uses a WGS84 ellipsoid as default to create the transverse mercator projection
                        proj = ccrs.TransverseMercator(central_longitude=central_lon,central_latitude=central_lat)
                        axm = plt.axes(projection=proj)
                        station_coords = np.unique(np.vstack((data[:,0:2],data[:,2:4])),axis=0)
                        segments = np.hstack((np.split(np.column_stack(srcxy)[idx][subidx],len(data)),
                                              np.split(np.column_stack(rcvxy)[idx][subidx],len(data))))
                        velocities = dist_real[idx][subidx]/data[:,4]/1000.
                        lc = LineCollection(segments[:5000], linewidths=0.3)
                        lc.set(array=velocities[:5000],cmap='jet_r')
                        # PlateCarree is the projection of the input coordinates, i.e. "no" projection (lon,lat)
                        # it is also possible to use ccrs.Geodetic() here 
                        axm.plot(station_coords[:,1],station_coords[:,0],'rv',ms = 2,transform = ccrs.PlateCarree())
                        axm.add_collection(lc)
                        axm.coastlines(resolution='50m')
                        axm.add_feature(cf.BORDERS.with_scale('50m'))
                        axm.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey')
                        axm.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey')
                        plt.colorbar(lc,fraction=0.05,shrink=0.5,label='velocity')
                        plt.title("Period: %s" %period)
                        plt.draw()
                   
                # share the input data among MPI processes
                data = mpi_comm.bcast(data,root=0)
                wavelength_threshold = mpi_comm.bcast(wavelength_threshold,root=0)
                meanvel = mpi_comm.bcast(meanvel,root=0)
                    
        
        #%% STARTING EIKONAL TOMOGRAPHY CALCULATIONS
                                    
                models = []
                directions = []
                discarded = 0
                total = 0
                stationcount = np.zeros(len(stats))
                stationerrors = np.zeros(len(stats))
                # loop through all possible central stations
                station_indices = np.arange(len(stats))[mpi_rank::mpi_size]
                for k in (station_indices[mpi_rank::mpi_size]):
                    
                    station = stats[k]
                    #if k!=416:#320:
                    #    continue
                    
                    # select data for this particular central station
                    idx11 = np.isin(data[:,0],station[0])
                    idx12 = np.isin(data[:,1],station[1])
                    idx21 = np.isin(data[:,2],station[0])
                    idx22 = np.isin(data[:,3],station[1])
                    
                    data_selection = np.vstack((data[:,(2,3,4)][idx11*idx12],
                                                data[:,(0,1,4)][idx21*idx22],        
                                                np.array([station[0],station[1],0]))) 
                    
                    # should have at least 10 stations to calculate a traveltime field
                    if len(data_selection)<10:
                        continue
        
                    #with open(logfile, 'a') as f:
                    #    print("MPI rank %d: %d/%d" %(mpi_rank,k,len(stats[mpi_rank::mpi_size])),file=f,flush=True)
                    if mpi_size>1:
                        print("mpi rank",mpi_rank,"   ",k,"/",len(stats))
                    else:
                        print(k,"/",len(stats))
                    
                    # Check which stations are involved and whether some stations appear twice
                    station_idx = []
                    for teststat in data_selection[:-1,:2]:
                        idx_x = np.isin(stats[:,0],teststat[0])
                        idx_y = np.isin(stats[:,1],teststat[1])                
                        idx = idx_x*idx_y
                        idx = idx.nonzero()[0]
                        if len(idx)!=1:
                            raise Exception("Are there stations with identical lat/lon values?")
                        station_idx.append(idx[0])
                    stationcount[station_idx] += 1
    
                    # Check that each station pair appears exactly once in the dataset
                    for teststat in data_selection[:,:2]:
                        idx_x = np.isin(data_selection[:,0],teststat[0])
                        idx_y = np.isin(data_selection[:,1],teststat[1])
                        idx = idx_x*idx_y
                        idx = idx.nonzero()[0]
                        if len(idx)!=1:
                            raise Exception("There is more than one measurement for the station couple",data_selection[idx],". This will result in a singular matrix when doing the Radial Basis Spline interpolation.")
                    
                    # xcoords and ycoords give the station locations
                    if coordinates == 'latlon':
                        xcoords,ycoords = p(data_selection[:,1],data_selection[:,0])
                        xcoords /= 1000.
                        ycoords /= 1000.
                    else:
                        xcoords = data_selection[:,0]
                        ycoords = data_selection[:,1]
                        
                    xstat = xcoords[-1]
                    ystat = ycoords[-1]            
        
                    # exclude points that are closer than the wavelength threshold
                    # to the central station
                    dist_matrix = np.sqrt((X-xstat)**2+(Y-ystat)**2)
                    exclude_gridpoints_idx = dist_matrix<wavelength_threshold
                           
                    # convert the travel times to travel-time residuals
                    dists = np.sqrt((xcoords[:-1]-xstat)**2+(ycoords[:-1]-ystat)**2)
                    ttime_residuals = np.append(data_selection[:-1,2]-dists/meanvel,0)
                    ttime_residuals_pre = ttime_residuals.copy() # only for plotting reasons
                    mean_ttimes = dist_matrix/meanvel
                       
                    # the quality selection list shows the indices of the stations
                    # that are NOT rejected
                    quality_selection = np.append(np.arange(len(xcoords)-1,dtype=int),-1)
                    
                    pointcoords = np.column_stack((xcoords,ycoords))
                    
                    # nearest neighbor tree for all stations
                    kdt = cKDTree(pointcoords)
                                  
                    # Create a convex hull around the region that is covered by stations
                    hull = Delaunay(pointcoords)
                    inhull = (hull.find_simplex(np.column_stack((X.flatten(),Y.flatten())))>=0).reshape(np.shape(X))
    
    
                    ###############################################################
                    # STEP 1: rejecting stations that coincide with a strong 
                    # gradient in the slowness field (this is not exactly the curvature)
                    rbfi = RBFInterpolator(pointcoords,ttime_residuals,
                                           kernel='linear',smoothing=0)
                    residual_field = np.reshape(rbfi(np.column_stack((X.flatten(),
                                                Y.flatten()))),X.shape)                    
                    
                    # gaussian curvature of a surface (yields similar results)
                    # Zy, Zx = np.gradient(residual_field)
                    # Zy /= ygridspacing
                    # Zx /= xgridspacing
                    # Zxy, Zxx = np.gradient(Zx)
                    # Zxy /= ygridspacing
                    # Zxx /= xgridspacing
                    # Zyy, _ = np.gradient(Zy)
                    # Zyy /= ygridspacing
                    # K = (Zxx * Zyy - (Zxy ** 2)) /  (1 + (Zx ** 2) + (Zy **2)) ** 2
                    # grad2 = np.abs(K)
                    
                    ttime_field = residual_field+mean_ttimes
                    gradient_field = np.gradient(ttime_field) 
                    # normalize the x and y gradients that result from np.gradient
                    gradient_field[1] /= xgridspacing # x gradient
                    gradient_field[0] /= ygridspacing # y gradient
                    # grad1 = np.sqrt(gradient_field[0]**2 + gradient_field[1]**2)

                    # angles = np.ma.masked_array(data=np.arctan2(gradient_field[0],
                    #                                             gradient_field[1]),
                    #                             mask=np.isnan(grad1))                
                    # # curvature from angles
                    # tmp = np.abs(np.diff(angles,axis=1))
                    # tmp[tmp>np.pi] = 2*np.pi - tmp[tmp>np.pi]
                    # tmp = np.column_stack((tmp[:,0],tmp,tmp[:,-1]))
                    # xdiff = (tmp[:,:-1] + tmp[:,1:]) / (2.*xgridspacing)
                    # tmp = np.abs(np.diff(angles,axis=0))
                    # tmp[tmp>np.pi] = 2*np.pi - tmp[tmp>np.pi]
                    # tmp = np.vstack((tmp[0],tmp,tmp[-1]))
                    # ydiff = (tmp[:-1,:] + tmp[1:,:]) / (2.*ygridspacing)
                    # curvature = np.max((xdiff,ydiff),axis=0)
                    # curvature[~inhull] = 0.
                    # curvature[exclude_gridpoints_idx] = 0.
                    grad2 = np.gradient(np.sqrt(gradient_field[0]**2+gradient_field[1]**2))
                    grad2 = np.sqrt(grad2[0]**2+grad2[1]**2)
                    grad2[~inhull] = 0.
                    grad2[exclude_gridpoints_idx] = 0.
                    # gradient at the station location (nearest neighbor interpolation)
                    nn = kdt_grid.query(np.column_stack((xcoords,ycoords)),3)
                    curv_at_statloc = np.sum(grad2.flatten()[nn[1]],axis=1)
                    bad_stations = np.where(curv_at_statloc>np.mean(curv_at_statloc)+curvature_threshold*np.std(curv_at_statloc))[0]
                    # remove bad stations from the quality selection
                    quality_selection = list(set(quality_selection)-set(bad_stations))
                    
                    # make sure that the central station didn't get rejected
                    if not -1 in quality_selection:
                        quality_selection.append(-1)
    
                    if mpi_size==1:
                        print("    outlier rection: discarded %.1f %%" %(len(bad_stations)/len(xcoords)*100))
    
                    # make sure there are at least 10 stations left
                    if len(quality_selection)<10:
                        continue
                    
                    # for statistics, count how many data points have been rejected
                    total += len(xcoords)
                    discarded += len(bad_stations)
    
                    # Create a convex hull around the region that is covered by stations
                    # This is repeated, because some stations have been rejected
                    hull = Delaunay(np.column_stack((xcoords[quality_selection],ycoords[quality_selection])))
                    inhull = (hull.find_simplex(np.column_stack((X.flatten(),Y.flatten())))>=0).reshape(np.shape(X))
                    
                    
                    ###############################################################
                    # STEP 2: Interpolating the travel times with the chosen method
                    
                    if method == 'linear':
                        residual_field = griddata(np.column_stack((xcoords[quality_selection],
                                                                   ycoords[quality_selection])),
                                                  ttime_residuals[quality_selection],
                                                  (X,Y),method='linear')
                        
                    elif method == 'cubic':
                        residual_field = griddata(np.column_stack((xcoords[quality_selection],
                                                                   ycoords[quality_selection])),
                                                  ttime_residuals[quality_selection],
                                                  (X,Y),method='cubic')                    
                        
                    elif method == 'splineintension':
                        # add 10% on each side
                        xadd = (plotarea[1]-plotarea[0])*0.1
                        yadd = (plotarea[3]-plotarea[2])*0.1
                        interpolation_limits = p(np.array([plotarea[0]-xadd,plotarea[0]-xadd,
                                                           plotarea[1]+xadd,plotarea[1]+xadd]),
                                                 np.array([plotarea[2]-yadd,plotarea[2]-yadd,
                                                           plotarea[3]+yadd,plotarea[3]+yadd]))
                        x0 = np.min(interpolation_limits[0]/1000.)
                        x1 = np.max(interpolation_limits[0]/1000.)
                        y0 = np.min(interpolation_limits[1]/1000.)
                        y1 = np.max(interpolation_limits[1]/1000.)
                        if np.sum((xcoords[quality_selection]<x1)*(xcoords[quality_selection]>x0)*
                               (ycoords[quality_selection]<y1)*(ycoords[quality_selection]>y0)) < 10:
                            continue
                        np.savetxt("ttime.ascii",np.column_stack((xcoords[quality_selection],
                                                                  ycoords[quality_selection],
                                                                  ttime_residuals[quality_selection])))                        
                        #x0 = np.min(X)
                        #x1 = np.max(X)
                        #y0 = np.min(Y)
                        #y1 = np.max(Y)
                        # Tension parameter can be changed here (see GMT surface documentation)
                        dump = subprocess.check_output("surface ttime.ascii -Gsmoothgrid.nc -T0 -I%f -R%d/%d/%d/%d" %(np.mean([xgridspacing,ygridspacing]),x0,x1,y0,y1),shell=True)                        
                        dump = subprocess.check_output("grd2xyz smoothgrid.nc > smoothgrid.ascii",shell=True)
                        smoothgrid = np.loadtxt("smoothgrid.ascii")
                        xgrid = np.unique(smoothgrid[:,0])
                        ygrid = np.unique(smoothgrid[:,1])
                        rgi = RegularGridInterpolator((xgrid,ygrid),smoothgrid[:,2].reshape((len(ygrid),len(xgrid)))[::-1,:].T,bounds_error=False,fill_value=np.nan)
                        residual_field = rgi((X.flatten(),Y.flatten())).reshape(np.shape(X))

                        # Lin et al. (2009) do the spline interpolation twice with
                        # different tension parameters. If the result differs too
                        # much, it means that the result is not well defined and
                        # should be discarded.
                        dump = subprocess.check_output("surface ttime.ascii -Gsmoothgrid.nc -T0.25 -I%f -R%d/%d/%d/%d" %(np.mean([xgridspacing,ygridspacing]),x0,x1,y0,y1),shell=True)
                        dump = subprocess.check_output("grd2xyz smoothgrid.nc > smoothgrid.ascii",shell=True)
                        smoothgrid = np.loadtxt("smoothgrid.ascii")
                        xgrid = np.unique(smoothgrid[:,0])
                        ygrid = np.unique(smoothgrid[:,1])
                        rgi = RegularGridInterpolator((xgrid,ygrid),smoothgrid[:,2].reshape((len(ygrid),len(xgrid)))[::-1,:].T,bounds_error=False,fill_value=np.nan)
                        residual_field_extra = rgi((X.flatten(),Y.flatten())).reshape(np.shape(X))

                        residual_field[np.abs(residual_field-residual_field_extra)>5.] = np.nan

                    elif 'rbf' in method:
                        function = method.split("-")[1]
                        # Finding the best smoothing parameter so that the
                        # velocities do not exceed the smoothing threshold
                        # this step takes most of the calculation time because of
                        # the repeated interpolation procedure
                        if 'multiquadric' in method:
                            smoothstep = 0.001
                            # very low smoothing values are enough for the multiquadric method
                        else:
                            smoothstep = 1.
                        smoothing = 0.
                        iterations = 0
                        while True:
                            # limit the number of iterations so the loop doesn't get 
                            # stuck in cases where smoothing is not possible
                            if iterations>35:
                                print("stopping")
                                break

                            rbfi = RBFInterpolator(np.column_stack(
                                (xcoords[quality_selection],
                                 ycoords[quality_selection])),
                                ttime_residuals[quality_selection],
                                kernel=function,smoothing=smoothing)  
                            residual_field = np.reshape(rbfi(
                                np.column_stack((X.flatten(),Y.flatten()))),
                                X.shape)
                            if iterations == 0:
                                residual_field_pre = residual_field.copy()
                           
                            ttime_field = residual_field+mean_ttimes
                            gradient_field = np.gradient(ttime_field)    
                            gradient_field[1] /= xgridspacing # x gradient
                            gradient_field[0] /= ygridspacing # y gradient  
                            # the velocity field is the inverse of the gradient field
                            velocity_field = 1./np.sqrt(gradient_field[0]**2+gradient_field[1]**2)
                            #velocity_field = np.ma.masked_array(data=1./np.sqrt(gradient_field[0]**2+gradient_field[1]**2),
                            #                                    mask=np.logical_or(~inhull,exclude_gridpoints_idx))

                            if gaussian_filtering:
                                velocity_field = gaussian_filt(velocity_field,(mean_wavelength/4./ygridspacing,mean_wavelength/4./xgridspacing),1)
                            velocity_field = np.ma.masked_array(data=velocity_field,
                                                mask=np.logical_or(~inhull,exclude_gridpoints_idx))

                            # if there are velocities greater than the smoothing_
                            # threshold times the mean velocity, increase smoothing
                            if np.max(velocity_field)>smoothing_threshold*meanvel: 
                                iterations += 1
                                smoothing += iterations*smoothstep
                                continue
                            else:
                                break                
                        if mpi_size==1:
                            print("    smoothing parameter: %.3f" %(smoothing))
                    
    
                    ttime_field = residual_field+mean_ttimes
                    gradient_field = np.gradient(ttime_field)    
                    gradient_field[1] /= xgridspacing # x gradient
                    gradient_field[0] /= ygridspacing # y gradient  
                    # the velocity field is the inverse of the gradient field
                    velocity_field = 1./np.sqrt(gradient_field[0]**2+gradient_field[1]**2)
                    #velocity_field = np.ma.masked_array(data=1./np.sqrt(gradient_field[0]**2+gradient_field[1]**2),
                    #                                    mask=np.logical_or(~inhull,exclude_gridpoints_idx))
                    # gaussian filter sigma=(ysigma,xsigma)
                    if gaussian_filtering:
                        velocity_field = gaussian_filt(velocity_field,(mean_wavelength/4./ygridspacing,mean_wavelength/4./xgridspacing),1)
                    velocity_field = np.ma.masked_array(data=velocity_field,
                                                        mask=np.logical_or(~inhull,exclude_gridpoints_idx))
                    
                    
                    # ####################################################
                    # STEP 2.5: Discard areas with high curvature
                    # from scipy.ndimage import laplace    
                    # # curvature estimation from laplacian
                    # curvature = np.abs(laplace(ttime_field))
                    # # define a curvature threshold
                    # max_curv = np.nanmin(curvature[dist_matrix<10*np.sqrt(xgridspacing**2+ygridspacing**2)])
                    
                    # velocity_field = np.ma.masked_where(curvature > max_curv,
                    #                                     velocity_field)
                                        
        
                    ###############################################################
                    # STEP 3: Discard all points in the final velocity field that
                    # exceed the mean velocity times the rejection threshold
                    maxvel = rejection_threshold*meanvel
                    minvel = 1./rejection_threshold*meanvel
                    if mpi_size==1:
                        print("    velocity outlier rejection: discarded %.2f %%" %(
                            (np.sum(velocity_field>maxvel)+np.sum(velocity_field<minvel))/np.sum(~velocity_field.mask)*100))
                    velocity_field = np.ma.masked_where(velocity_field>maxvel,velocity_field)
                    velocity_field = np.ma.masked_where(velocity_field<minvel,velocity_field)
                    
                    
                    # within the convex hull, remove also model points that are 
                    # farther away than the wavelength threshold. However, if the
                    # wavelength threshold is less than min_distance, take
                    # min_distance as threshold (otherwise, at high freqs.
                    # too many points would get removed)
                    kdt = cKDTree(np.column_stack((xcoords[quality_selection][:-1],ycoords[quality_selection][:-1])))
                    nn_points = kdt.query(np.column_stack((X.flatten(),Y.flatten())))
                    nn_dists = nn_points[0].reshape(np.shape(X))
                    dist_threshold = np.max((wavelength_threshold,min_distance))
                    velocity_field = np.ma.masked_where(nn_dists>dist_threshold,
                                                        velocity_field)
                    
    
                    # just for statistics: station errors
                    predictions = rbfi(np.column_stack((xcoords[:-1],ycoords[:-1])))
                    errs = predictions - ttime_residuals_pre[:-1]# data_selection[:-1,2]            
                    stationerrors[station_idx] += np.abs(errs)
        
    
                    if velocity_field.mask.all():
                        continue
        
                    # the angles give the direction of the steepest gradient
                    # that is the direction in which the wave travels
                    # this is needed for the calculation of the azimuthal anisotropy
                    angles = np.ma.masked_array(data=np.arctan2(gradient_field[0],gradient_field[1]),
                                                mask=velocity_field.mask)
                    
                    directions.append(angles)
                    models.append(velocity_field)
        
    #%% TESTPLOT (optional)
                    if create_example_plots: #testplots, currently works only with RBF options
                        if not 'rbf' in method:
                            print("creating a testplot currently only works properly with rbf interpolation option")
                        from geographiclib.geodesic import Geodesic
                        tmax = np.ceil(np.max(data_selection[:,2]))
                        maxv = meanvel*1.5
                        minv = meanvel*0.5
                        
                        # uses a WGS84 ellipsoid as default to create the transverse mercator projection
                        #proj = ccrs.TransverseMercator(central_longitude=central_lon,central_latitude=central_lat)
                        proj = ccrs.Mercator()
                        azi = np.linspace(0,360,100)
                        circle_coords = []
                        for az in azi:
                            geoline = Geodesic.WGS84.Line(data_selection[-1,0],data_selection[-1,1],az)
                            line = geoline.Position(wavelength_threshold*1000)
                            circle_coords.append([line['lon2'],line['lat2']])
                        circle_coords = np.array(circle_coords)
                        
                        removed_stat_idx = list(set(np.append(np.arange(len(xcoords)-1,dtype=int),-1)) - set(quality_selection))
                        
                        if not "rbf" in method:
                            residual_field_pre = residual_field
                        
                        ttime_field_pre = mean_ttimes+residual_field_pre
                        gradient_field = np.gradient(ttime_field_pre)    
                        gradient_field[1] /= xgridspacing # x gradient
                        gradient_field[0] /= ygridspacing # y gradient  
                        phase_vel_pre = 1./np.sqrt(gradient_field[0]**2+gradient_field[1]**2)

                        ttime_field_post = mean_ttimes + residual_field
 
                        gradient_field = np.gradient(ttime_field_post)  
                        gradient_field[1] /= xgridspacing # x gradient
                        gradient_field[0] /= ygridspacing # y gradient  
                        grad2post = np.gradient(np.sqrt(gradient_field[0]**2+gradient_field[1]**2))
                        grad2post = np.sqrt(grad2post[0]**2+grad2post[1]**2)
                        
                        fig = plt.figure(figsize=(15,8))   
                        gs = gridspec.GridSpec(2,3, width_ratios=[1, 1, 1], height_ratios=[1,1],
                                               wspace=0.05,hspace=0.01)
                        extent = plotarea#(2,19,41.5,51)
                        axm1 = fig.add_subplot(gs[0],projection=proj)
                        axm1.set_extent(extent)
                        cbar = axm1.scatter(data_selection[:,1],data_selection[:,0],c=data_selection[:,2],vmin=0,vmax=tmax,s=7,transform = ccrs.PlateCarree(),label='travel-time measurement')
                        cbar = axm1.contour(LON,LAT,ttime_field_pre,levels=np.linspace(0,tmax,20),transform = ccrs.PlateCarree())
                        axm1.scatter([],[],s=50,marker='o',facecolors='none',edgecolors='r',label='one wavelength')
                        axm1.plot(circle_coords[:,0],circle_coords[:,1],'r',transform = ccrs.PlateCarree())
                        axm1.coastlines(resolution='50m',rasterized=True)
                        axm1.add_feature(cf.BORDERS.with_scale('50m'),rasterized=True)
                        axm1.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey',rasterized=True)
                        axm1.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey',rasterized=True)
                        gl = axm1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                            #xlocs=[6,10,14,18],ylocs=[43,45,47,49],
                                            linewidth=2, color='gray', alpha=0.5, linestyle='--')
                        gl.top_labels = False
                        gl.bottom_labels = False
                        gl.right_labels = False
                        gl.xlines = False
                        gl.ylines = False
                        plt.legend(loc='upper right',fontsize=9)
                        plt.title("a",loc='left',fontweight='bold')
                        
                        
                        axm2 = fig.add_subplot(gs[1],projection=proj)
                        axm2.set_extent(extent)
                        #cbar = axm2.scatter(data_selection[:,1],data_selection[:,0],c=data_selection[:,2],vmin=0,vmax=tmax,s=5,transform = ccrs.PlateCarree())
                        cbar = axm2.pcolormesh(LON,LAT,grad2,cmap=plt.cm.Reds,
                                               vmin=0,vmax=0.2,
                                               transform = ccrs.PlateCarree(),
                                               shading='nearest',rasterized=True)
                        axm2.plot(data_selection[:,1],data_selection[:,0],'v',#,marker=r'$\bigtriangledown$',
                                  color='black',ms=2,transform = ccrs.PlateCarree(),label='station')
                        axm2.plot(data_selection[removed_stat_idx,1],data_selection[removed_stat_idx,0],
                                 'v',color='mediumseagreen',ms=3,transform = ccrs.PlateCarree(),label='rejected station')
                        axm2.scatter([],[],s=50,marker='o',facecolors='none',edgecolors='r',label='one wavelength')
                        axm2.plot(circle_coords[:,0],circle_coords[:,1],'r',transform = ccrs.PlateCarree())
                        axm2.coastlines(resolution='50m',rasterized=True)
                        axm2.add_feature(cf.BORDERS.with_scale('50m'),rasterized=True)
                        axm2.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey',rasterized=True)
                        axm2.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey',rasterized=True)
                        gl = axm2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                            #xlocs=[6,10,14,18],ylocs=[43,45,47,49],
                                            linewidth=2, color='gray', alpha=0.5, linestyle='--')
                        gl.top_labels = False
                        gl.bottom_labels = False
                        gl.right_labels = False
                        gl.left_labels = False
                        gl.xlines = False
                        gl.ylines = False
                        plt.legend(loc='upper right',fontsize=9)
                        plt.title("b",loc='left',fontweight='bold')
    
                        axm3 = fig.add_subplot(gs[2],projection=proj)
                        axm3.set_extent(extent)
                        #cbar = axm3.scatter(data_selection[:,1],data_selection[:,0],c=data_selection[:,2],vmin=0,vmax=tmax,s=5,transform = ccrs.PlateCarree())
                        cbar = axm3.pcolormesh(LON,LAT,phase_vel_pre,cmap=cm.GMT_haxby_r,
                                               vmin=minv,vmax=maxv,transform = ccrs.PlateCarree(),
                                               shading='nearest',rasterized=True)
                        axm3.coastlines(resolution='50m',rasterized=True)
                        axm3.add_feature(cf.BORDERS.with_scale('50m'),rasterized=True)
                        axm3.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey',rasterized=True)
                        axm3.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey',rasterized=True)
                        gl = axm3.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                            #xlocs=[6,10,14,18],ylocs=[43,45,47,49],
                                            linewidth=2, color='gray', alpha=0.5, linestyle='--')
                        gl.top_labels = False
                        gl.bottom_labels = False
                        gl.right_labels = False
                        gl.left_labels = False
                        gl.xlines = False
                        gl.ylines = False
                        plt.title("c",loc='left',fontweight='bold')
    
                                        
                        axm4 = fig.add_subplot(gs[3],projection=proj)
                        axm4.set_extent(extent)
                        cbar1 = axm4.scatter(data_selection[quality_selection,1],
                                            data_selection[quality_selection,0],
                                            c=data_selection[quality_selection,2],
                                            vmin=0,vmax=tmax,s=7,transform = ccrs.PlateCarree())
                        cbar = axm4.contour(LON,LAT,ttime_field_post,levels=np.linspace(0,tmax,20),transform = ccrs.PlateCarree())
                        #axm4.plot(circle_coords[:,0],circle_coords[:,1],'r',transform = ccrs.PlateCarree())
                        axm4.coastlines(resolution='50m',rasterized=True)
                        axm4.add_feature(cf.BORDERS.with_scale('50m'),rasterized=True)
                        axm4.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey',rasterized=True)
                        axm4.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey',rasterized=True)
                        gl = axm4.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                            #xlocs=[6,10,14,18],ylocs=[43,45,47,49],
                                            linewidth=2, color='gray', alpha=0.5, linestyle='--')
                        gl.top_labels = False
                        gl.right_labels = False
                        gl.xlines = False
                        gl.ylines = False
                        plt.title("d",loc='left',fontweight='bold')
                        cax = fig.add_axes([0.15, 0.06, 0.2, 0.01])
                        plt.colorbar(cbar1,cax=cax,orientation='horizontal',label='travel time [s]')                    
                        
                        
                        axm5 = fig.add_subplot(gs[4],projection=proj)
                        axm5.set_extent(extent)
                        #cbar = axm5.scatter(data_selection[:,1],data_selection[:,0],c=data_selection[:,2],vmin=0,vmax=tmax,s=5,transform = ccrs.PlateCarree())
                        cbar = axm5.pcolormesh(LON,LAT,grad2post,cmap=plt.cm.Reds,vmin=0,vmax=0.2,
                                               transform = ccrs.PlateCarree(),
                                               shading='nearest',rasterized=True)
                        axm5.plot(data_selection[quality_selection,1],data_selection[quality_selection,0],'v',#,marker=r'$\bigtriangledown$',
                                  color='black',ms=2,transform = ccrs.PlateCarree(),label='station')
                        axm5.coastlines(resolution='50m',rasterized=True)
                        axm5.add_feature(cf.BORDERS.with_scale('50m'),rasterized=True)
                        axm5.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey',rasterized=True)
                        axm5.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey',rasterized=True)
                        gl = axm5.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                            #xlocs=[6,10,14,18],ylocs=[43,45,47,49],
                                            linewidth=2, color='gray', alpha=0.5, linestyle='--')
                        gl.top_labels = False
                        gl.right_labels = False
                        gl.left_labels = False
                        gl.xlines = False
                        gl.ylines = False
                        plt.title("e",loc='left',fontweight='bold')
                        cax = fig.add_axes([0.45, 0.06, 0.15, 0.01])
                        plt.colorbar(cbar,cax=cax,orientation='horizontal',label=r'slowness gradient [$d\nu/dxdy$]')
    
                        axm6 = fig.add_subplot(gs[5],projection=proj)
                        axm6.set_extent(extent)
                        #cbar = axm6.scatter(data_selection[:,1],data_selection[:,0],c=data_selection[:,2],vmin=0,vmax=tmax,s=5,transform = ccrs.PlateCarree())
                        cbar = axm6.pcolormesh(LON,LAT,velocity_field,cmap=cm.GMT_haxby_r,
                                               vmin=minv,vmax=maxv,transform = ccrs.PlateCarree(),
                                               shading='nearest',rasterized=True)
                        axm6.coastlines(resolution='50m',rasterized=True)
                        axm6.add_feature(cf.BORDERS.with_scale('50m'),rasterized=True)
                        axm6.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey',rasterized=True)
                        axm6.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey',rasterized=True)
                        gl = axm6.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                            #xlocs=[6,10,14,18],ylocs=[43,45,47,49],
                                            linewidth=2, color='gray', alpha=0.5, linestyle='--')
                        gl.top_labels = False
                        gl.right_labels = False
                        gl.left_labels = False
                        gl.xlines = False
                        gl.ylines = False
                        plt.title("f",loc='left',fontweight='bold')
                        cax = fig.add_axes([0.71, 0.06, 0.15, 0.01])
                        plt.colorbar(cbar,cax=cax,orientation='horizontal',label='phase velocity [km/s]')
                        plt.savefig(os.path.join(
                            output_location,
                            "exampleplot_%ss_central_stat_%f_%f.png" %(
                                period,station[0],station[1])),
                            dpi=100,bbox_inches='tight')
                        plt.close(fig)
                        #plt.show()
                        
    #%% DONE PROCESSING INNER LOOP
                
                # gathering information from MPI parallel processes
                all_models = []
                all_models = mpi_comm.gather(models, root=0)
                all_directions = []
                all_directions = mpi_comm.gather(directions, root=0)        
        
                # average traveltime errors of stations
                stationerrors[stationcount>0] = stationerrors[stationcount>0]/stationcount[stationcount>0]
                all_stationerrors = []
                all_stationerrors = mpi_comm.gather(stationerrors,root=0)
                
                #
                discarded_stations = []
                total_stations = []
                discarded_stations = mpi_comm.gather(discarded)
                total_stations = mpi_comm.gather(total)
                
    
    #%% SUMMING AND AVERAGING MODELS  
    
                outfilename = os.path.join(
                    output_location,"%s_%ss_%s_%.2f_%.2f_%.2f.npy" %(
                        outfilename_base,period,method,curvature_threshold,
                        rejection_threshold,smoothing_threshold))        
    
                if mpi_rank==0:
                    
                    no_models = np.sum([len(mods) for mods in all_models])
                    
                    with open(logfile, 'a') as f:
                        print("\nAll phase-velocity maps calculated. Starting averaging and plotting procedure.\n",file=f,flush=True)     
                        if no_models < minimum_no_of_models:
                            print("Less than %d models - aborting.\n" %minimum_no_of_models,file=f,flush=True)
                            print("Less than %d models - aborting.\n" %minimum_no_of_models)
                            continue             
                        
                    total = np.sum(total_stations)
                    discarded = np.sum(discarded_stations)
                    with open(logfile,'a') as f:
                        print("Discarded station pair data: %.1f %%" %(discarded/total*100),file=f)                
                        
                        
                    models = []
                    for modcollection in all_models:
                        if len(modcollection) == 0:
                            continue      
                        for mod in modcollection:
                            model = mod.data
                            model[mod.mask] = np.nan
                            models.append(model)                     
                    models = np.stack(models,axis=0)

                    directions = []
                    for directcollection in all_directions:
                        if len(directcollection) == 0:
                            continue
                        for direct in directcollection:
                            direction = direct.data
                            direction[direct.mask] = np.nan
                            directions.append(direction)
                    directions = np.stack(directions,axis=0)
                    
                    stationerrors = np.sum(all_stationerrors,axis=0)/mpi_comm.Get_size()
                    
                    
                    #%%
                    """
                    # special test
                    lon = 7.413
                    lat = 45.33111
                    ivreaidx = np.where(np.sqrt((LAT-lat)**2+(LON-lon)**2)==np.min(np.sqrt((LAT-lat)**2+(LON-lon)**2)))
                    modvals = []
                    for mod in models:
                        modvals.append(mod[ivreaidx])
                    modvals = np.array(modvals)
                    modvals = modvals[~np.isnan(modvals)]
                    plt.figure(figsize=(10,5))
                    plt.text(2.1,22,"gridcell location:\nlat=45.33 lon=7.41",
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='white', boxstyle='round'))
                    gkde = gaussian_kde(modvals,bw_method=0.6)
                    smoothhist = gkde(np.linspace(min(modvals),max(modvals),100))
                    plt.plot(np.linspace(min(modvals),max(modvals),100),smoothhist/np.max(smoothhist)*10,color='black',label='KDE (scaled)')
                    kdemode = np.linspace(min(modvals),max(modvals),100)[smoothhist.argmax()]
                    kdemode = mode(modvals)
                    plt.hist(modvals,bins=100,color='lightgrey')
                    plt.plot([np.mean(modvals),np.mean(modvals)],[0,20],'b--',label='mean')
                    plt.plot([np.median(modvals),np.median(modvals)],[0,20],'green',label='median')
                    plt.plot([kdemode,kdemode],[0,20],'r',label='mode from KDE')
                    #plt.plot([np.mean(modvals)-1*np.std(modvals),np.mean(modvals)-1*np.std(modvals)],[0,5],'--')
                    #plt.plot([np.mean(modvals)+1*np.std(modvals),np.mean(modvals)+1*np.std(modvals)],[0,5],'--')
                    modvals2 = modvals[(modvals>np.mean(modvals)-1*np.std(modvals))*(modvals<np.mean(modvals)+1*np.std(modvals))]
                    plt.plot([np.mean(modvals2),np.mean(modvals2)],[0,20],'b-',label='mean after outlier removal')                
                    #plt.plot([np.median(modvals2),np.median(modvals2)],[0,5],label='median after outlier removal')                
                    plt.ylabel("no of samples")
                    plt.xlabel("phase velocity [km/s]")
                    plt.legend()
                    #plt.savefig("mode_estimation.pdf",bbox_inches='tight')
                    plt.show()                
                    
                    #%%
                    bin_edges = np.arange(-180,180+30,30)
                    bin_edges = bin_edges/180.*np.pi
                    modvals = []
                    dirvals = []
                    for mi in range(len(models)):
                        modvals.append(models[mi][ivreaidx])
                        dirvals.append(directions[mi][ivreaidx])
                    modvals = np.array(modvals)
                    dirvals = np.array(dirvals)
                    modvals = modvals[~np.isnan(modvals)]  
                    dirvals = dirvals[~np.isnan(dirvals)]
                    binned_count = binned_statistic(dirvals,modvals,statistic='count',bins=bin_edges)
                    binned_mean = binned_statistic(dirvals,modvals,statistic='mean',bins=bin_edges)
                    binned_mode = binned_statistic(dirvals,modvals,statistic=mode,bins=bin_edges)

                    plt.figure()
                    plt.hist(modvals,bins=100,color='lightgrey')
                    plt.hist(binned_mode.statistic,bins=30,color='grey')
                    plt.plot([np.mean(modvals),np.mean(modvals)],[0,20],'b--',label='mean')
                    plt.plot([np.nanmean(binned_mean.statistic),np.nanmean(binned_mean.statistic)],
                             [0,20],'magenta',label='mean after azimuthal binning')
                    samplemode, smooth_hist = mode(binned_mode.statistic[~np.isnan(binned_mode.statistic)],return_kde=True)
                    plt.plot(smooth_hist[:,0],smooth_hist[:,1],'k',label='kde')
                    plt.plot([samplemode,samplemode],
                             [0,20],'g--',label='mode from bin modes after azimuthal binning')
                    samplemode, smooth_hist = mode(binned_mean.statistic[~np.isnan(binned_mean.statistic)],return_kde=True)
                    plt.plot([samplemode,samplemode],
                             [0,20],'r-.',label='mode from bin means after azimuthal binning')
                    plt.plot([np.nanmean(binned_mode.statistic), np.nanmean(binned_mode.statistic)],
                             [0,20],'y',lw=3,label='mean from bin modes after azimuthal binning')
                    
                    kdemode = mode(modvals)
                    plt.plot([kdemode,kdemode],[0,20],'r',label='mode from KDE of full histogram')
                    
                    plt.ylabel("no of samples")
                    plt.xlabel("phase velocity [km/s]")
                    plt.legend()
                    plt.show()
                    pause
                    """

                    
                    #%% REMOVE OUTLIERS
    
                    # delete all points that deviate by more than two standard deviations
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        meanmodel = np.nanmean(models,axis=0)
                        modelstd = np.nanstd(models,axis=0)
        
                        rejected_modelarea = 0
                        total_area = 0
                        for mi,mod in enumerate(models):
                            rejected = np.abs(mod-meanmodel)>2*np.nanmean(modelstd)
                            rejected_modelarea += np.sum(rejected)
                            total_area += np.sum(~np.isnan(mod))
                            directions[mi][rejected] = np.nan
                            mod[rejected] = np.nan
                        print("The two std criterion rejected %.1f %% of the modeled area" %(rejected_modelarea/total_area*100))
            
                        meanmodel = np.nanmean(models,axis=0)
                        modelstd = np.nanstd(models,axis=0) 
    
                   
                    #%% CHECK AZIMUTHAL COVERAGE, at least 1 model per bin
                    # azimuthal bins
                    bin_edges = np.arange(-180,180+30,30)
                    bin_edges = bin_edges/180.*np.pi
                    bad_azimuthal_coverage_and_lower_than_min_no_mods = np.ones_like(X,dtype=bool)
                    rejected_bad_coverage = 0
                    total_area = 0
                    for xi in range(len(X[0])):
                        for yi in range(len(Y[:,0])):
                            nonans = ~np.isnan(models[:,yi,xi])
                            if np.sum(nonans) < minimum_no_of_models:
                                continue
                            binned_count = binned_statistic(directions[nonans,yi,xi],models[nonans,yi,xi],statistic='count',bins=bin_edges)
                            azimuthal_gap = 0
                            for binno in range(1,len(bin_edges)):
                                models_in_bin = len(np.unique(np.where(binned_count.binnumber==binno)[0]))
                                if models_in_bin == 0:
                                    azimuthal_gap += azimuthal_bin_size
                            if azimuthal_gap <= max_azimuthal_gap:
                                bad_azimuthal_coverage_and_lower_than_min_no_mods[yi,xi] = False
                            else:
                                rejected_bad_coverage += 1
                            total_area += 1
                    print("The azimuthal coverage criterion rejected %.1f %% of isotropic model area." %(rejected_bad_coverage/total_area*100))
                    
                    # take only modelpoints where the average is calculated from 50 or more models
                    # take only modelpoints where the azimuthal coverage is good
                    meanmodel[bad_azimuthal_coverage_and_lower_than_min_no_mods] = np.nan
                    modelstd[bad_azimuthal_coverage_and_lower_than_min_no_mods] = np.nan
                    
                    #%% CALCULATE MODE (if requested)     
                    # calculate the mode of all models
                    bin_edges = np.arange(-180,180+10,10)
                    bin_edges = bin_edges/180.*np.pi
                    if average_type == 'mode':
                        modemodel = np.ones_like(meanmodel)*np.nan
                        for xi in range(len(X[0])):
                            for yi in range(len(Y[:,0])):
                                if bad_azimuthal_coverage_and_lower_than_min_no_mods[yi,xi]:
                                    continue
                                nonans = ~np.isnan(models[:,yi,xi])
                                if np.sum(nonans) < minimum_no_of_models:
                                    continue
                                
                                #samplemode_without_binning = mode(models[nonans,yi,xi])
                                #binned_mode = binned_statistic(directions[nonans,yi,xi],
                                #                               models[nonans,yi,xi],
                                #                               statistic=mode,bins=bin_edges)
                                binned_mean = binned_statistic(directions[nonans,yi,xi],
                                                               models[nonans,yi,xi],
                                                               statistic='mean',bins=bin_edges)

                                modemodel[yi,xi] = mode(binned_mean.statistic[~np.isnan(binned_mean.statistic)])
                                #modemodel[yi,xi] = np.nanmean(binned_mean.statistic)
                        
                        #fig = plt.figure(figsize=(14,6))
                        #plt.subplot(121)
                        #plt.pcolormesh(X,Y,meanmodel,cmap=cm.GMT_haxby_r,
                        #                      vmin=0.9*meanvel,
                        #                      vmax=1.1*meanvel)
                        #plt.subplot(122)
                        #plt.pcolormesh(X,Y,modemodel,cmap=cm.GMT_haxby_r,
                        #                      vmin=0.9*meanvel,
                        #                      vmax=1.1*meanvel)
                        #plt.savefig(outfilename.replace(".npy","_mode_vs_mean_models.png"),bbox_inches='tight')
                        ##plt.show()
                        #plt.close(fig)
                        
                        if np.isnan(modemodel).all():
                            print("not enough data at period "+period+" - aborting.")
                            continue
                            
                    if np.isnan(meanmodel).all():
                        print("not enough data at period "+period+" - aborting.")  
                        continue                      
    
    
    #%% EXTRACT THE ANISOTROPY
                    # azimuthal bins
                    if 360%azimuthal_bin_size != 0:
                        new_azimuthal_bin_size = np.array([1,2,4,5,6,8,10,12,15,20,25,30,35,40,45])
                        new_azimuthal_bin_size = new_azimuthal_bin_size[np.abs(new_azimuthal_bin_size-azimuthal_bin_size).argmin()]
                        print(azimuthal_bin_size,"is not a factor of 360! Changing to:",new_azimuthal_bin_size)
                        azimuthal_bin_size = new_azimuthal_bin_size
                    bin_edges = np.arange(-180,180+azimuthal_bin_size,azimuthal_bin_size)
                    bin_edges = bin_edges/180.*np.pi
                    
                    for avg_area in [avg_area]:
                        print("\n AVG AREA:",avg_area)
                        # Smith and Dahlen 1973, anisotropy follows the following form
                        def aniso_fitting_function1(azi,A2,A4,PHI2,PHI4):
                            return A2*np.cos(2*(azi-PHI2)) + A4*np.cos(4*(azi-PHI4))
                        
                        def aniso_fitting_function2(azi,A1,A2,A4,PHI1,PHI2,PHI4):
                            return A1*np.cos(azi-PHI1)+A2*np.cos(2*(azi-PHI2)) + A4*np.cos(4*(azi-PHI4))
                        
                        downsamplingx = int(np.ceil(avg_area/xgridspacing))
                        downsamplingy = int(np.ceil(avg_area/ygridspacing))
                        if downsamplingx<=2 or downsamplingy<=2:
                            with open(logfile, 'a') as f:
                                print("area averaging not working properly",file=f)
                            
                        # subtract the mean in each cell to reduce tradeoffs
                        # between isotropic velocity variations within the avg_area
                        # and anisotropy strengths
                        if average_type == 'mode':
                            relative_models = (models-modemodel)/modemodel*100.
                        else:
                            relative_models = (models-meanmodel)/meanmodel*100
                                  
                        Xlow = X[::downsamplingy,::downsamplingx]
                        Ylow = Y[::downsamplingy,::downsamplingx]
                        LATlow = LAT[::downsamplingy,::downsamplingx]
                        LONlow = LON[::downsamplingy,::downsamplingx]
                        A1 = np.ones_like(Xlow)*np.nan
                        A2 = np.ones_like(Xlow)*np.nan
                        A4 = np.ones_like(Xlow)*np.nan
                        PHI1 = np.ones_like(Xlow)*np.nan
                        PHI2 = np.ones_like(Xlow)*np.nan
                        PHI4 = np.ones_like(Xlow)*np.nan
                        if include_A1:
                            PARAM_STD = np.ones(np.shape(Xlow)+(6,))*np.nan
                        else:
                            PARAM_STD = np.ones(np.shape(Xlow)+(4,))*np.nan
                        cnt=0
                        BINNED_CNT = np.ones(np.shape(Xlow)+(len(bin_edges)-1,))*np.nan
                        BINNED_STD = np.ones(np.shape(Xlow)+(len(bin_edges)-1,))*np.nan
                        BINNED_MEAN = np.ones(np.shape(Xlow)+(len(bin_edges)-1,))*np.nan
                        errorsum = 0.
                        errorN = 0
                        anisoplot=True
                        for xi in range(len(Xlow[0])):
                            for yi in range(len(Ylow[:,0])):
                                x = Xlow[yi,xi]
                                y = Ylow[yi,xi]
                                dist_matrix = np.sqrt((x-X)**2+(y-Y)**2)
                                indices = np.where(dist_matrix<=avg_area)
                                submodels = relative_models[:,indices[0],indices[1]]
                                subdirs = directions[:,indices[0],indices[1]]
                                if np.sum(~np.isnan(submodels).all(axis=1))<minimum_no_of_models:
                                    continue
                                nonans = ~np.isnan(submodels.flatten())
                                binned_count = binned_statistic(subdirs.flatten()[nonans],submodels.flatten()[nonans],statistic='count',bins=bin_edges)
                                # we will need the association between models and 
                                # their bins several times. Therefore store in list
                                modelidx_in_bin = []
                                for binno in range(1,len(bin_edges)):
                                    modelidx_in_bin.append(np.where(binned_count.binnumber==binno)[0])
                                # get the number of INDIVIDUAL models in each bin
                                # individual means, coming from a different central station
                                model_indices = np.repeat(np.arange(len(models)),np.shape(subdirs)[1]).reshape(np.shape(subdirs))
                                cnt_unique = np.zeros(len(bin_edges)-1)
                                for binno in range(len(bin_edges)-1):
                                    cnt_unique[binno] = len(np.unique(model_indices.flatten()[nonans][modelidx_in_bin[binno]]))  
                                bad_bins = np.where(cnt_unique<models_per_bin)[0]
                                azimuthal_gap = len(bad_bins)*azimuthal_bin_size
                                if azimuthal_gap > max_azimuthal_gap: # more than 60 degrees without value (maybe only exclude if neighboring bins are empty?)
                                    continue 
                                binned_means = np.ones(len(bin_edges)-1)*np.nan
                                binned_stds = np.ones(len(bin_edges)-1)*np.nan
                                for binno in range(len(bin_edges)-1):
                                    if len(modelidx_in_bin[binno])==0 or binno in bad_bins:
                                        continue
                                    if average_type == 'mode':
                                        binned_means[binno] = mode(submodels.flatten()[nonans][modelidx_in_bin[binno]])
                                    else:
                                        binned_means[binno] = np.mean(submodels.flatten()[nonans][modelidx_in_bin[binno]])
                                    binned_stds[binno] = np.std(submodels.flatten()[nonans][modelidx_in_bin[binno]])
                                azimuths = (bin_edges[:-1]+0.5*np.diff(bin_edges))/np.pi*180
                                cnt+=1
                                #isotropic_vel = np.nanmean(binned_means.statistic)
                                #aniso_data = (binned_means.statistic-isotropic_vel)/isotropic_vel*100.
                                aniso_data = binned_means
                                dirs_rad = azimuths/180.*np.pi
                                dirs_rad = dirs_rad[~np.isnan(aniso_data)]
                                azimuths = azimuths[~np.isnan(aniso_data)]
                                aniso_data = aniso_data[~np.isnan(aniso_data)]
                                try:
                                    # amplitude boundaries are important. Otherwise, it may choose negative amplitudes which correspond to a 180deg shift for the PHI1 component (no problem) and a 90deg shift for the PHI2 component (problem!)
                                    if include_A1:
                                        popt,pcov = curve_fit(aniso_fitting_function2,dirs_rad,aniso_data,bounds=([0,0,0,0,0,0],[20,20,20,2*np.pi,np.pi,np.pi/2.])) 
                                        datafit = aniso_fitting_function2(dirs_rad,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5])             
                                    else:
                                        # solving only for A2 and A4 and their angles
                                        popt,pcov = curve_fit(aniso_fitting_function1,dirs_rad,aniso_data,bounds=([0,0,0,0],[20,20,np.pi,np.pi/2.]))
                                        datafit = aniso_fitting_function1(dirs_rad,popt[0],popt[1],popt[2],popt[3])
                                        popt = [0.,popt[0],popt[1],0.,popt[2],popt[3]]
                                except:
                                    print("Warning: fitting anisotropic parameters not successful.")
                                    continue
                                errorsum += np.sum((datafit-aniso_data)**2)
                                errorN += len(aniso_data)
                                
                                A1[yi,xi] = popt[0]
                                A2[yi,xi] = popt[1]
                                A4[yi,xi] = popt[2]
                                PHI1[yi,xi] = popt[3]
                                PHI2[yi,xi] = popt[4]
                                PHI4[yi,xi] = popt[5]
                                PARAM_STD[yi,xi,:] = np.sqrt(np.diag(pcov))
                                BINNED_CNT[yi,xi,:] = cnt_unique
                                BINNED_STD[yi,xi,:] = binned_stds
                                BINNED_MEAN[yi,xi,:] = binned_means
    
                                # create a plot of the anisotropic fit (optional)
                                lonplot,latplot = p(Xlow[yi,xi]*1000.,Ylow[yi,xi]*1000.,inverse=True)
                                if xi>int(len(Xlow[0])/2) and yi>int(len(Ylow[:,0])/2) and popt[1]>1.1*np.nanmean(A2) and anisoplot:
                                    outfilename_anisofit = outfilename.replace(".npy","_%.3f_%.3f_anisofit.png" %(latplot,lonplot))
                                    fig=plt.figure(figsize=(8,6))
                                    #plt.errorbar(azimuths,
                                    #             rel_aniso,yerr=[
                                    #             np.abs((binned_means.statistic+binned_stds.statistic-isotropic_vel)/isotropic_vel*100.-rel_aniso),
                                    #             np.abs((binned_means.statistic-binned_stds.statistic-isotropic_vel)/isotropic_vel*100.-rel_aniso)],fmt='o')
                                    plt.plot(azimuths,aniso_data,'ko',label='mean velocity')
                                    plt.plot(dirs_rad/np.pi*180.,datafit,'--',label='velocity fit')
                                    rms = np.sqrt(np.sum((datafit-aniso_data)**2)/len(aniso_data))
                                    if include_A1:
                                        txt = plt.text(90,-4.5,"A1=%.1f PHI1=%.1f$^{\circ}$\nA2=%.1f PHI2=%d$^{\circ}$\nA4=%.1f PHI4=%d$^{\circ}$\nrms=%.2f" %(popt[0],popt[3]/np.pi*180,popt[1],popt[4]/np.pi*180,popt[2],popt[5]/np.pi*180.,rms),
                                                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='white', boxstyle='round'))
                                    else:
                                        txt = plt.text(90,-4.5,"A2=%.1f PHI2=%d$^{\circ}$\nA4=%.1f PHI4=%d$^{\circ}$\nrms=%.2f" %(popt[1],popt[4]/np.pi*180,popt[2],popt[5]/np.pi*180.,rms),
                                                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='white', boxstyle='round'))
                                    plt.legend(loc='upper right')
                                    plt.ylim(-5,5)
                                    plt.xlabel("direction (counter-clockwise from east)")
                                    plt.ylabel("relative amplitude [%]")
                                    plt.savefig(outfilename_anisofit,bbox_inches='tight')
                                    plt.close(fig)
                                    anisoplot=False
                                    
        
                        if errorN>0:
                            rms = np.sqrt(errorsum/errorN)
                        else:
                            rms = np.nan
                        print("A1=%.2f A2=%.2f A4=%.2f" %(np.nanmean(A1),np.nanmean(A2),np.nanmean(A4)))
                        #print("A2/A1=%.4f" %(np.nanmean(A2)/np.nanmean(A1)))   
                        print("rms=%.2f" %rms)
                            
                        with open(logfile, 'a') as f:
                            print("Anisotropy (amplitudes in percent, directions in radians, counter-clockwise from the east):",file=f)
                            print("A1 = %.2f" %np.nanmean(np.abs(A1)),file=f)
                            print("A2 = %.2f" %np.nanmean(np.abs(A2)),file=f)
                            print("A4 = %.2f" %np.nanmean(np.abs(A4)),file=f) 
                            print("PHI1 = %.2f" %np.nanmean(np.abs(PHI1)),file=f)
                            print("PHI2 = %.2f" %np.nanmean(np.abs(PHI2)),file=f)
                            print("PHI4 = %.2f" %np.nanmean(np.abs(PHI4)),file=f)
                            print("rms = %.3f" %rms,file=f,flush=True)
                                       
                        if False:#not np.isnan(A1).all():
                            plt.ioff()
                            fig=plt.figure()
                            #plt.errorbar(azimuths,
                            #             rel_aniso,yerr=[
                            #             np.abs((binned_means.statistic+binned_stds.statistic-isotropic_vel)/isotropic_vel*100.-rel_aniso),
                            #             np.abs((binned_means.statistic-binned_stds.statistic-isotropic_vel)/isotropic_vel*100.-rel_aniso)],fmt='o')
                            plt.plot(azimuths,aniso_data,'o')
                            plt.plot(dirs_rad/np.pi*180.,datafit)
                            plt.plot([],[],label='A1=%.2f' %np.nanmean(np.abs(A1)))
                            plt.legend()
                            plt.close(fig)
                            #plt.ylim(-6,6)
                            #plt.show()
        
                        # calculate variance reduction (optional)
                        if print_variance_reduction:
                            if average_type == 'mode':
                                velfu = RegularGridInterpolator((X[0],Y[:,0]),modemodel.T)
                            else:
                                velfu = RegularGridInterpolator((X[0],Y[:,0]),meanmodel.T)
                            phifu = RegularGridInterpolator((Xlow[0],Ylow[:,0]),PHI2.T,method='nearest')
                            anisofu = RegularGridInterpolator((Xlow[0],Ylow[:,0]),A2.T)
                            syn_ttimes_iso = np.ones(len(data))*np.nan
                            syn_ttimes_aniso = np.ones(len(data))*np.nan
                            syn_ttimes_constvel = np.ones(len(data))*np.nan
                            srcxy = p(data[:,1],data[:,0])
                            rcvxy = p(data[:,3],data[:,2])
                            srcxy = (srcxy[0]/1000.,srcxy[1]/1000.)
                            rcvxy = (rcvxy[0]/1000.,rcvxy[1]/1000.)
                            for i,statpair in enumerate(data):
                                    
                                #if i%5000==0:
                                #    print(i,"/",len(data))
                            
                                distance = np.sqrt((srcxy[0][i]-rcvxy[0][i])**2 + (srcxy[1][i]-rcvxy[1][i])**2)
                                stepsize = np.min([xgridspacing,ygridspacing])/2.
                                x_regular = np.interp(np.linspace(0,1,int(distance/stepsize)),[0,1],[srcxy[0][i],rcvxy[0][i]])
                                y_regular = np.interp(np.linspace(0,1,int(distance/stepsize)),[0,1],[srcxy[1][i],rcvxy[1][i]])
                                
                                path_straight = np.column_stack((x_regular,y_regular))
                    
                                if np.isnan(path_straight).any():
                                    continue
                                if ((path_straight[:,0]>np.max(Xlow[0])).any() or 
                                    (path_straight[:,0]<np.min(Xlow[0])).any() or 
                                    (path_straight[:,1]>np.max(Ylow[:,0])).any() or 
                                    (path_straight[:,1]<np.min(Ylow[:,0])).any()):
                                    continue
                                
                                centers = path_straight[:-1]+np.diff(path_straight,axis=0)/2.
                                isovels = velfu(centers)
                                pathdirs = np.arctan2(np.diff(path_straight[:,1]),np.diff(path_straight[:,0]))/np.pi*180.
                                pathphi = phifu(centers)/np.pi*180.
                                pathaniso = anisofu(centers)
                                pathaniso[np.isnan(pathaniso)] = 0.
                                pathphi[np.isnan(pathphi)] = 0.
                                
                                
                                pathvels = isovels * (1+pathaniso/100.*np.cos(2*(pathphi-pathdirs)/180.*np.pi))            
                                
                                pathdists = np.sqrt(np.sum(np.diff(path_straight,axis=0)**2,axis=1))
                                pathtimes = pathdists/pathvels
                                pathttime = np.sum(pathtimes)
                                
                                syn_ttimes_iso[i] = np.sum(pathdists/isovels)
                                syn_ttimes_aniso[i] = pathttime
                                syn_ttimes_constvel[i] = np.sum(pathdists/meanvel)
                            
                            np.savetxt(os.path.join(os.path.dirname(outfilename),
                                        "straight_ray_phasetraveltimes_period_%ss.txt" %(period)),
                                       np.column_stack((data,syn_ttimes_iso,syn_ttimes_aniso)),
                                       fmt="%.5f %.5f %.5f %.5f %.4f %.4f %.4f",
                                       header="lat1 lon1 lat2 lon2 ttime syn_iso syn_aniso")
                            valid_idx = (~np.isnan(syn_ttimes_iso)) * (~np.isnan(syn_ttimes_aniso))
                            print("valid paths for calculation of variance reduction:",np.sum(valid_idx))
                            var_red_iso = 1 - np.sum((syn_ttimes_iso[valid_idx]-data[valid_idx,4])**2) / np.sum((syn_ttimes_constvel[valid_idx]-data[valid_idx,4])**2)
                            var_red_aniso = 1 - np.sum((syn_ttimes_aniso[valid_idx]-data[valid_idx,4])**2) / np.sum((syn_ttimes_constvel[valid_idx]-data[valid_idx,4])**2)
                    
                            #outfilename_variance_red_dataset = outfilename.replace(".npy","_variance_reduction_dataset.npy")
                            #np.save(outfilename_variance_red_dataset,np.column_stack((srcxy[valid_idx],rcvxy[valid_idx],syn_ttimes_iso[valid_idx],syn_ttimes_aniso[valid_idx])))
                            print("Variance reduction isotropic: %.3f anisotropic: %.3f" %(var_red_iso,var_red_aniso))
                            #with open("var_red_test.txt","a") as f:
                            #    f.write("%s %.1f %.3f %.3f\n" %(period,avg_area,var_red_iso,var_red_aniso))
        
                            with open(logfile,'a') as f:
                                print("Variance reduction isotropic: %.3f anisotropic: %.3f\n" %(var_red_iso,var_red_aniso),file=f)        
                        
                        
    #%%             SAVING MODELS      
            
                    outfilename_meanmodel = outfilename.replace(".npy","_meanmodel.npy")
                    np.save(outfilename_meanmodel,meanmodel)
                    if average_type == 'mode':
                        outfilename_modemodel = outfilename.replace(".npy","_modemodel.npy")
                        np.save(outfilename_modemodel,modemodel)
                    outfilename_modelstd = outfilename.replace(".npy","_modelstd.npy")
                    np.save(outfilename_modelstd,modelstd)
                    if include_A1:
                        outfilename_aniso = outfilename.replace(".npy","_aniso_amp1.npy")
                        np.save(outfilename_aniso,A1)
                        outfilename_aniso = outfilename.replace(".npy","_aniso_phi1.npy")
                        np.save(outfilename_aniso,PHI1)                                
                    outfilename_aniso = outfilename.replace(".npy","_aniso_amp2.npy")
                    np.save(outfilename_aniso,A2)
                    outfilename_aniso = outfilename.replace(".npy","_aniso_phi2.npy")
                    np.save(outfilename_aniso,PHI2)  
                    outfilename_aniso = outfilename.replace(".npy","_aniso_amp4.npy")
                    np.save(outfilename_aniso,A4)
                    outfilename_aniso = outfilename.replace(".npy","_aniso_phi4.npy")
                    np.save(outfilename_aniso,PHI4)  
                    outfilename_staterrs = outfilename.replace(".npy","_staterrors.npy")
                    np.save(outfilename_staterrs,np.column_stack((stats,stationerrors)))
                    outfilename_aniso = outfilename.replace(".npy","_binned_count.npy")
                    np.save(outfilename_aniso,BINNED_CNT)  
                    outfilename_aniso = outfilename.replace(".npy","_binned_std.npy")
                    np.save(outfilename_aniso,BINNED_STD)
                    outfilename_aniso = outfilename.replace(".npy","_binned_mean.npy")
                    np.save(outfilename_aniso,BINNED_MEAN)
                    azimuths = (bin_edges[:-1]+0.5*np.diff(bin_edges))/np.pi*180
                    np.save(outfilename.replace(".npy","_bin_azimuths.npy"),azimuths)
                    outfilename_aniso = outfilename.replace(".npy","_anisoparams_std.npy")
                    np.save(outfilename_aniso,PARAM_STD)
                    
        #%% PLOTTING            
                    fig = plt.figure(figsize=(16,7))
                    gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1, 1], width_ratios=[2.5,1],hspace=0.05,wspace=0.1)
                    # uses a WGS84 ellipsoid as default to create the transverse mercator projection
                    #proj = ccrs.TransverseMercator(central_longitude=central_lon,central_latitude=central_lat)
                    # currently, no lat/lon labels are supported for other projections than Mercator
                    proj = ccrs.Mercator()
                    
                    
                    # ISOTROPIC VELOCITY PLOT
                    #     vmin = 0.85*meanvel
                    #     vmax = 1.12*meanvel
                    vmin=np.nanmin(meanmodel)
                    vmax=np.nanmax(meanmodel)
                        
                    ax1 = fig.add_subplot(gs[:,:1],projection=proj)
                    if average_type == 'mode':
                        contour = ax1.contourf(LON,LAT,modemodel,cmap=cm.GMT_haxby_r,
                                              levels=np.linspace(vmin,vmax,100),
                                              extend='both',
                                              transform = ccrs.PlateCarree())
                    else:
                        contour = ax1.contourf(LON,LAT,meanmodel,cmap=cm.GMT_haxby_r,
                                               levels=np.linspace(vmin,vmax,100),
                                               extend='both',
                                               transform = ccrs.PlateCarree())
                    for c in contour.collections:
                        c.set_rasterized(True)
                    ax1.text(0.03,0.03,"Period = "+period+"s",fontsize=14,
                             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                             transform = ax1.transAxes)
                    # PlateCarree is the projection of the input coordinates, i.e. "no" projection (lon,lat)
                    # it is also possible to use ccrs.Geodetic() here 
                    #ax1.plot(stats[:,1],stats[:,0],'rv',ms = 2,transform = ccrs.PlateCarree())
                    ax1.coastlines(resolution='50m',rasterized=True)
                    ax1.add_feature(cf.BORDERS.with_scale('50m'),rasterized=True)
                    ax1.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey',rasterized=True)
                    ax1.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey',rasterized=True)
                    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
                    gl.top_labels = False
                    gl.right_labels = False
                    gl.xlines = False
                    gl.ylines = False
                    cbaxes = fig.add_axes([0.61, 0.24, 0.007, 0.5]) 
                    plt.colorbar(contour,cax=cbaxes,#orientation='horizontal',
                                 label='phase velocity [km/s]')
                    
                    # ANISOTROPY FAST AXIS PLOT
                    if not np.isnan(A2).all():
                        aniso_legend = int(np.ceil(np.mean([np.nanmin(A2),np.nanmax(A2)])))
                        # the angles argument takes angles in degrees, counter-clockwise from horizontal
                        q = ax1.quiver(LONlow,LATlow,A2,A2,angles=PHI2/np.pi*180,
                                       headwidth=0,headlength=0,headaxislength=0,
                                       pivot='middle',width=0.005,#scale=100.,#width=0.005,
                                       color='yellow',edgecolor='k',#scale=80.,width=0.0035,
                                       linewidth=0.5,transform=ccrs.PlateCarree(),zorder=2)
                        qk = ax1.quiverkey(q, X=0.8, Y=0.05, U=aniso_legend,
                                           label='%d%% anisotropy' %aniso_legend,
                                           labelpos='E',#edgecolor='w',linewidth=0.5,
                                           fontproperties=dict(size=12))
                        qk.text.set_path_effects([path_effects.withStroke(linewidth=2,foreground='w')])                
                    
                    
                    # MODEL STD PLOT
                    ax2 = fig.add_subplot(gs[0,1],projection=proj)
                    contour = ax2.contourf(LON,LAT,modelstd,cmap=plt.cm.autumn_r,
                                          levels=np.linspace(0,0.5,50),extend='max',
                                          transform = ccrs.PlateCarree())
                    for c in contour.collections:
                        c.set_rasterized(True)
                    #ax2.plot(stats[:,1],stats[:,0],'wv',ms = 2,transform = ccrs.PlateCarree())
                    ax2.coastlines(resolution='50m',rasterized=True)
                    ax2.add_feature(cf.BORDERS.with_scale('50m'),rasterized=True)
                    ax2.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey',rasterized=True)
                    ax2.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey',rasterized=True)
                    colorbar = plt.colorbar(contour,fraction=0.05,shrink=0.5,
                                            ticks=[0.,0.1,0.2,0.3,0.4,0.5],label='model std [km/s]')
      
                    
                    # STATION ERROR PLOT
                    ax4 = fig.add_subplot(gs[1,1],projection=proj)
                    stats = stats[stationerrors.argsort()]
                    stationerrors=stationerrors[stationerrors.argsort()]
                    cbar = ax4.scatter(stats[stationerrors>0.,1],stats[stationerrors>0.,0],
                                       c=stationerrors[stationerrors>0.],s=10,cmap=plt.cm.autumn_r,
                                       vmin=0.,vmax=5.0,transform = ccrs.PlateCarree())
                    ax4.coastlines(resolution='50m',rasterized=True)
                    ax4.add_feature(cf.BORDERS.with_scale('50m'),rasterized=True)
                    ax4.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey',rasterized=True)
                    ax4.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey',rasterized=True)
                    plt.colorbar(cbar,fraction=0.05,shrink=0.5,ticks=[0,1,2,3,4,5],
                                 label='mean ttime error [s]')   
                    
                    if plotarea != 'auto':
                        ax1.set_extent(plotarea,crs=ccrs.PlateCarree())
                        ax2.set_extent(plotarea,crs=ccrs.PlateCarree())
                        ax4.set_extent(plotarea,crs=ccrs.PlateCarree())
                    
                    #plt.show()
                    #pause
                    plt.savefig(outfilename.replace(".npy",".png"),bbox_inches='tight')
                    plt.close(fig)

