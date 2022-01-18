#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 15:57:31 2018

@author: emanuel
"""

import numpy as np
import matplotlib.pyplot as plt
import skfmm
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline,RegularGridInterpolator#,interpn
#from scipy.integrate import solve_ivp
import datetime

# velocity field v has to be regular of shape (ny,nx), (lats,lons)
#scikit-fmm only works for regular Cartesian grids, but grid cells may have
# a different (uniform) length in each dimension

def main():
    print("test run")
    X, Y = np.meshgrid(np.linspace(-2000,2000,301), np.linspace(0,4000,401))
    dx = X[0][1]-X[0][0]
    dy = Y[:,0][1]-Y[:,0][0]
    source = (-1500,2500)
    receivers = [(1497,1940),[1400,1200]]
    #receivers = [np.array([1497,1950]),np.array([1200,1500])]
    v = np.ones_like(X)*3.5
    v[Y<1300] = 4.5
    v[(abs(X)<300)*(Y>1000)*(Y<3100)] *= 0.01
    v[(X>1000)*(X<1200)*(Y>2400)*(Y<2900)] = 5.5
    wavelength = 50
    v_smooth = smooth_velocity_field(v,dx,dy,wavelength)

    t0 = datetime.datetime.now()
    for j in range(100):    
        xnew,ynew,ttimefield = calculate_ttime_field(X[0],Y[:,0],
                                    v_smooth,source,interpolate=True,
                                    refine_source_grid=True)
        path_list = shoot_ray(xnew,ynew,ttimefield,source,receivers)
    print("100 runs take",(datetime.datetime.now()-t0).total_seconds(),"s")

    X,Y = np.meshgrid(xnew,ynew)
    plt.figure()
    plt.pcolormesh(X,Y,v_smooth,vmin=-1.0,vmax=6.0,cmap=plt.cm.gray)
    plt.contour(X,Y,ttimefield,levels=np.linspace(0,1300,50))
    plt.plot(X,Y,'k.',marker='+',markersize=0.4)
    plt.plot(source[0],source[1],'ro')
    for receiver in receivers:
        plt.plot(receiver[0],receiver[1],'bo')
    for path in path_list:
        plt.plot(path[:,0],path[:,1],'k',linewidth=3)
    plt.colorbar(label='travel time [s]')
    plt.xlabel('xdistance [km]')
    plt.ylabel('ydistance [km]')
    plt.text(0,2500,'v=0.01 km/s',rotation=90,color='white')
    plt.show()
 
# smoothing is not used in the actual ttime field calculation below
def smooth_velocity_field(v,dx,dy,wavelength):
    # 0.5: standard deviation sigma relates to one side of the gaussian bell
    # 1./3: the entire wavelength should fit in 3 standard deviations
    sigmax = 0.5*wavelength/dx*1./3
    sigmay = 0.5*wavelength/dy*1./3
    v_smooth = gaussian_filter(v,sigma=[sigmay,sigmax],truncate=3.0,mode='nearest')
    return v_smooth
   

def calculate_ttime_field(x,y,v,source,interpolate=True,refine_source_grid=True,pts_refine=5):
        
    if source[0] > np.max(x) or source[0] < np.min(x) or source[1] > np.max(y) or source[1] < np.min(y):
        raise Exception("Source must be within the coordinate limits!")
   
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    
    xsource = source[0]
    ysource = source[1]
    sourceidx = (np.argmin(np.abs(y-ysource)),np.argmin(np.abs(x-xsource)))
    
    # create a new x and y axis because the source point normally doesn't coincide with the gridpoint locations
    xshift = x[sourceidx[1]]-source[0]
    yshift = y[sourceidx[0]]-source[1]
    xnew = x-xshift
    ynew = y-yshift
    
    Xnew,Ynew = np.meshgrid(xnew,ynew)
    
    if interpolate or refine_source_grid:
        v_func = RectBivariateSpline(x,y,v.T)
        v = (v_func(xnew,ynew)).T

    # creating a finer grid around the source and doing the traveltime calculations
    # on this fine grid before expanding to the entire grid
    # gives slightly better results
    create_testplots = False
    if refine_source_grid:
        fine_factor = 20
        xfine = np.linspace(xsource-pts_refine*dx,xsource+pts_refine*dx,fine_factor*2*pts_refine+1)
        yfine = np.linspace(ysource-pts_refine*dy,ysource+pts_refine*dy,fine_factor*2*pts_refine+1)
        
        # limit the fine subaxis to the valid range
        xfine = xfine[np.abs(xfine-np.min(xnew)).argmin():np.abs(xfine-np.max(xnew)).argmin()+1]
        yfine = yfine[np.abs(yfine-np.min(ynew)).argmin():np.abs(yfine-np.max(ynew)).argmin()+1]
                
        Xfine,Yfine = np.meshgrid(xfine,yfine)
        # we start with a perfectly circular traveltimefield which is centered
        # around the source at a distance of 1/4th of the minimum sampling
        # distance.
        radius = np.min((dx,dy))/4.
        phi_fine = np.sqrt((Xfine-xsource)**2+(Yfine-ysource)**2)
        v_fine = (v_func(xfine,yfine)).T # v_func(tuple(np.meshgrid(xfine,yfine))) # (v_func(xfine,yfine)).T
        ttime_fine = skfmm.travel_time(phi_fine-radius, v_fine, dx=[dy/fine_factor,dx/fine_factor])
        ttime_fine += radius/np.mean(v_fine[phi_fine<radius])
        
        if create_testplots:# for testing       
            plt.figure()
            plt.pcolormesh(Xnew,Ynew,v,cmap=plt.cm.seismic_r,shading='nearest')
            plt.pcolormesh(xfine,yfine,v_fine,shading='nearest',cmap=plt.cm.seismic_r)
            cont = plt.contour(xfine,yfine,ttime_fine,levels=np.linspace(0,np.max(ttime_fine),90))
            #plt.contour(xfine,yfine,ttime_fine1,levels=np.linspace(0,np.max(ttime_fine),90),linestyles='dashed')
            plt.plot(source[0],source[1],'rv')
            plt.gca().set_aspect('equal')
            plt.colorbar(cont)
            plt.show()
        
        # from the traveltimes on the fine grid we have to find an iso-
        # velocity contour that can serve as input for the larger grid
        # this iso-velocity contour is normally smoother compared to the one
        # we would get from a calculation without the grid refinement                
        phi_coarse = np.ones_like(Xnew)*np.max(ttime_fine)
        yidx0 = np.max([0,sourceidx[0]-pts_refine])
        yidx1 = np.min([sourceidx[0]+pts_refine+1,len(ynew)])
        xidx0 = np.max([0,sourceidx[1]-pts_refine])
        xidx1 = np.min([sourceidx[1]+pts_refine+1,len(xnew)])
        phi_coarse[yidx0:yidx1, xidx0:xidx1] = ttime_fine[::fine_factor,::fine_factor]
        
        if create_testplots:
            plt.figure()
            plt.pcolormesh(Xnew,Ynew,v,shading='nearest',cmap=plt.cm.seismic_r)
            levels=np.linspace(0,np.max(ttime_fine),30)
            cont=plt.contour(Xnew,Ynew,phi_coarse,linestyles='dashed',levels=levels)
            plt.contour(Xfine,Yfine,ttime_fine,linestyles='solid',levels=levels)
            plt.colorbar(cont)
            plt.gca().set_aspect('equal')
            plt.show()
        
        # minimum traveltime to the border of the fine-grid region
        borderxmax = sourceidx[1]+pts_refine if sourceidx[1]+pts_refine < len(xnew) else 0
        borderymax = sourceidx[0]+pts_refine if sourceidx[0]+pts_refine < len(ynew) else 0
        minborder_ttime = np.min((phi_coarse[sourceidx[0], sourceidx[1]-pts_refine],
                                  phi_coarse[sourceidx[0], borderxmax],
                                  phi_coarse[sourceidx[0]-pts_refine, sourceidx[1]],
                                  phi_coarse[borderymax, sourceidx[1]]))
        phi_coarse -= minborder_ttime
        
        if create_testplots:
            plt.figure()
            plt.pcolormesh(Xnew,Ynew,v,shading='nearest',cmap=plt.cm.seismic_r)
            cont=plt.contour(Xnew,Ynew,phi_coarse,linestyles='dashed',levels=[-1,0,1])
            plt.contour(Xfine,Yfine,ttime_fine,linestyles='solid',
                        levels=[minborder_ttime-1,minborder_ttime,minborder_ttime+1])
            plt.colorbar(cont)
            plt.gca().set_aspect('equal')
            plt.show()
        
        try:
            ttime_field = skfmm.travel_time(phi_coarse, v, dx=[dy,dx])
        except:
            print("had to reduce order")
            ttime_field = skfmm.travel_time(phi_coarse, v,order=1, dx=[dy,dx])
        ttime_field += minborder_ttime
        ttime_field[yidx0:yidx1, xidx0:xidx1] = ttime_fine[::fine_factor,::fine_factor]
                   
    else:
        phi = (Xnew-xsource)**2 + (Ynew-ysource)**2# - (dx/1000.)**2
        phi[sourceidx[0],sourceidx[1]] *= -1
        try:
            ttime_field = skfmm.travel_time(phi, v, dx=[dy,dx])
        except:
            #print("had to reduce order")
            ttime_field = skfmm.travel_time(phi, v,order=1, dx=[dy,dx])

    if create_testplots: # for testing
        plt.figure()
        ax = plt.gca()
        plt.pcolormesh(Xnew,Ynew,v,shading='nearest',cmap=plt.cm.seismic_r)
        levels = np.linspace(0,np.max(ttime_field),80)
        plt.contour(xnew,ynew,ttime_field,levels=levels,label='after grid refinement')
        #plt.legend(loc='upper right')
        #plt.contour(xnew,ynew,ttime_field2,linestyles='dashed',levels=levels,label='before grid refinement')
        #plt.contour(xnew,ynew,np.sqrt(phi)/3.8,linestyles='dotted',levels=80,label='homogeneous velocity contours')
        ax.set_aspect('equal')
        plt.show()
        
        
    return xnew,ynew,ttime_field


# this is the same as above, but the input and output grids are identical
# the ttime field is less exact but the calculations are faster and the further
# use is simpler.
def calculate_ttime_field_samegrid(X,Y,v,source,refine_source_grid=True,pts_refine=5):
        
    xsource = source[0]
    ysource = source[1]
    x = X[0]
    y = Y[:,0]  
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    
    if xsource > np.max(x) or xsource < np.min(x) or ysource > np.max(y) or ysource < np.min(y):
        raise Exception("Source must be within the coordinate limits!")
    
    if refine_source_grid:
        v_func = RectBivariateSpline(x,y,v.T)

    # creating a finer grid around the source and doing the traveltime calculations
    # on this fine grid before expanding to the entire grid
    # gives slightly better results
    if refine_source_grid:
        fine_factor = 20
        xfine = np.linspace(xsource-pts_refine*dx,xsource+pts_refine*dx,fine_factor*2*pts_refine+1)
        yfine = np.linspace(ysource-pts_refine*dy,ysource+pts_refine*dy,fine_factor*2*pts_refine+1)
        
        Xfine,Yfine = np.meshgrid(xfine,yfine)
        # we start with a perfectly circular traveltimefield which is centered
        # around the source at a distance of 1/4th of the minimum sampling
        # distance.
        radius = np.min((dx,dy))/4.
        phi_fine = np.sqrt((Xfine-xsource)**2+(Yfine-ysource)**2)
        v_fine = (v_func(xfine,yfine)).T # v_func(tuple(np.meshgrid(xfine,yfine))) # (v_func(xfine,yfine)).T
        ttime_fine = skfmm.travel_time(phi_fine-radius, v_fine, dx=[dy/fine_factor,dx/fine_factor])
        ttime_fine += radius/np.mean(v_fine[phi_fine<radius])
        # this last line is not really necessary:
        ttime_fine[phi_fine<radius] = phi_fine[phi_fine<radius]/np.mean(v_fine[phi_fine<radius])
        
        # # for testing
        # plt.figure()
        # ax = plt.gca()
        # plt.pcolormesh(X,Y,v,shading='nearest',cmap=plt.cm.seismic_r)
        # plt.pcolormesh(Xfine,Yfine,v_fine,shading='nearest',cmap=plt.cm.seismic_r)
        # plt.plot(X.flatten(),Y.flatten(),'k.')
        # plt.contour(xfine,yfine,ttime_fine,levels=np.linspace(0,50,80))
        # ttime_fine2 = skfmm.travel_time(phi_fine, v_fine, dx=[dy/fine_factor,dx/fine_factor])
        # plt.contour(xfine,yfine,ttime_fine2,linestyles='dashed',levels=np.linspace(0,50,80))
        # plt.plot(xsource,ysource,'rv')
        # ax.set_aspect('equal')
        # plt.show()
        
        # from the traveltimes on the fine grid we have to find an iso-
        # velocity contour that can serve as input for the larger grid
        # this iso-velocity contour is normally smoother compared to the one
        # we would get from a calculation without the grid refinement                
        ttime_fine_func = RegularGridInterpolator((xfine,yfine),ttime_fine.T,
                                                  method='nearest',bounds_error=False)
        phi_coarse = ttime_fine_func((X,Y))
        # find the largest closed contour [TOO SLOW]
        # plt.ioff()
        # levels=np.linspace(0,np.max(phi_fine)/np.mean(v_fine),20)
        # cset = plt.contour(xfine, yfine, ttime_fine,levels=levels)
        # # Iterate over all the contour segments and find the largest
        # stop = False
        # for i, segs in enumerate(cset.allsegs[::-1]):
        #     for j, seg in enumerate(segs):
        #         # First make sure it's closed
        #         if (seg[0]-seg[-1]).any():
        #             continue
        #         else:
        #             stop = True
        #             break
        #     if stop:
        #         contour_ttime = levels[::-1][i]
        #         break

        # minimum traveltime to the border of the fine-grid region
        sourceidx = np.array([np.argmin(np.abs(y-ysource)),np.argmin(np.abs(x-xsource))])
        # ignore the contour levels that are too close to the border of the domain
        # 0.9 as safety margin
        contour_ttime = 0.9*np.min((np.nanmax(phi_coarse[sourceidx[0]:,sourceidx[1]] if len(y)-sourceidx[0] > pts_refine else np.max(ttime_fine)),
                                    np.nanmax(phi_coarse[:sourceidx[0],sourceidx[1]] if sourceidx[0] >= pts_refine else np.max(ttime_fine)),
                                    np.nanmax(phi_coarse[sourceidx[0],sourceidx[1]:] if len(x)-sourceidx[1] > pts_refine else np.max(ttime_fine)),
                                    np.nanmax(phi_coarse[sourceidx[0],:sourceidx[1]] if sourceidx[1] >= pts_refine else np.max(ttime_fine))))
        phi_coarse -= contour_ttime
        phi_coarse[np.isnan(phi_coarse)] = np.nanmax(phi_coarse)
        
        # # for testing
        # plt.figure()
        # ax = plt.gca()
        # #phi_coarse[np.isnan(phi_coarse)] = np.nanmax(phi_coarse)
        # cbar = plt.tricontourf(X.flatten(),Y.flatten(),phi_coarse.flatten(),
        #                         cmap=plt.cm.seismic,levels=np.arange(-50,51,1))
        # plt.plot(X.flatten(),Y.flatten(),'k.')
        # plt.contour(xfine,yfine,ttime_fine,levels=np.linspace(0,50,50))
        # plt.contour(x,y,phi_coarse,linestyles='dashed',levels=np.linspace(0,50,50))
        # plt.plot(xsource,ysource,'rv')
        # ax.set_aspect('equal')
        # plt.colorbar(cbar)
        # plt.show()
        
        try:
            ttime_field = skfmm.travel_time(phi_coarse, v, dx=[dy,dx])
        except:
            print("had to reduce order")
            ttime_field = skfmm.travel_time(phi_coarse, v,order=1, dx=[dy,dx])
        ttime_field += contour_ttime
        ttime_field[phi_coarse<=0] = phi_coarse[phi_coarse<=0]+contour_ttime
        
    else:
        phi = (X-xsource)**2 + (Y-ysource)**2
        phi[sourceidx[0],sourceidx[1]] *= -1
        if np.min(np.abs(source[0]-x)) > dx/10 and np.min(np.abs(source[1]-y)) > dy/10.:
            print("Warning: refine_source_grid should be set to 'True' if the source "+
                  "location does not coincide exactly with one of the points of the grid.")
        try:
            ttime_field = skfmm.travel_time(phi, v, dx=[dy,dx])
            #ttime_field1 = skfmm.travel_time(phi, v,order=1, dx=[dy,dx]) 
            #print("max error:",np.max(np.abs(ttime_field-ttime_field1)))
        except:
            #print("had to reduce order")
            ttime_field = skfmm.travel_time(phi, v,order=1, dx=[dy,dx])

    # # for testing
    # plt.figure()
    # ax = plt.gca()
    # plt.plot(X.flatten(),Y.flatten(),'k.')
    # plt.contour(x,y,ttime_field,levels=np.linspace(0,np.max(ttime_field),80),label='with grid refinement')
    # phi = (X-xsource)**2 + (Y-ysource)**2# - (dx/1000.)**2
    # phi[sourceidx[0],sourceidx[1]] *= -1
    # ttime_field2 = skfmm.travel_time(phi, v, dx=[dy,dx])
    # plt.contour(x,y,ttime_field2,linestyles='dashed',
    #             levels=np.linspace(0,np.max(ttime_field),80),label='without grid refinement')
    # #plt.contour(xnew,ynew,np.sqrt(phi)/3.8,linestyles='dotted',levels=80,label='homogeneous velocity contours')
    # plt.plot(xsource,ysource,'rv')
    # ax.set_aspect('equal')
    # plt.legend(loc='upper right')
    # plt.colorbar()
    # plt.show()
        
    return ttime_field


def shoot_ray(x,y,ttimefield,source,receivers,stepsize=0.33):
    """

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    ttimefield : TYPE
        DESCRIPTION.
    source : TYPE
        DESCRIPTION.
    receivers : TYPE
        DESCRIPTION.
    stepsize : float, optional
        Size of the ray tracing step, relative w.r.t. the average grid sampling distance. The default is 0.33, meaning 1/3rd of a cell size.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # ray will be shot from the receiver to the source
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    step_dist = np.sqrt(dx**2+dy**2)*stepsize

    grad = np.gradient(ttimefield)
    spl_x = RectBivariateSpline(x,y,grad[1].T)
    spl_y = RectBivariateSpline(x,y,grad[0].T*dx/dy)
    def descent(t,xy):
        return [-spl_x.ev(xy[0],xy[1]),-spl_y.ev(xy[0],xy[1])]
     
    """
    # for testing - interesting artefacts in the reconstructed velocity field
    # I am not sure where they come from. Maybe because of the use of carte-
    # sian coordinates for a problem that is better described in polar coords?
    ttimefu = RectBivariateSpline(x,y,ttimefield.T)
    expected_ttimes = ttimefu.ev(np.array(receivers)[:,0],np.array(receivers)[:,1])
    velfield = 1./np.sqrt((grad[0]/dy)**2+(grad[1]/dx)**2)
    
    plt.figure()
    plt.pcolormesh(x,y,velfield,vmin=3.5,vmax=4.1)
    plt.contour(x,y,ttimefield,levels=80)
    plt.plot(receivers[0][0],receivers[0][1],'rv')
    plt.plot(path_list[0][:,0],path_list[0][:,1],'.')
    plt.show()
    """

    if len(np.shape(receivers)) == 1:
        receivers = [receivers]    

    path_list = []
    for receiver in receivers:
        if receiver[0]>np.max(x) or receiver[0]<np.min(x) or receiver[1]>np.max(y) or receiver[1]<np.min(y):
            print("Warning: receiver location outside map boundary")
            path = np.array([[np.nan,np.nan]])
            path_list.append(path)
            continue
        step_inc = 1.
        path = [[receiver[0],receiver[1]]]
        total_dist = np.sqrt((receiver[1]-source[1])**2 + (receiver[0]-source[0])**2)
        N = int(total_dist/step_dist*3)
        if N>100000:
            print("Ray tracing warning: path is sampled with more than 100k steps, consider using a larger stepsize")
        for i in range(N):
            x0,y0 = path[-1]
            if np.isinf(x0):
                raise Exception("ray tracing error")
            dist = np.sqrt((x0-source[0])**2 + (y0-source[1])**2)
            gradx,grady = descent(1,[x0,y0])
            step_distance = np.sqrt(gradx**2+grady**2)
            step_inc = step_dist/step_distance
            if dist/(step_distance*step_inc) <= 1.:
                break

            path.append([x0+gradx*step_inc,y0+grady*step_inc])
            
            # in vincinity of the source, the traveltimefield is not very exact, just make a straight line jump
            if np.abs(x0-source[0])<dx/1.5 and np.abs(y0-source[1])<dy/1.5:
                break
        else:
            print("ERROR: ray tracing is not converging towards the source!")
            print("source:",source,"receiver:",receiver)
            raise Exception()
        path.append([source[0],source[1]])
        path_list.append(np.array(path)[::-1])
    return path_list
    #ALTERNATIVE setting it up as initial value problem. is slower
#    def event(t,xy):
#        if np.abs(source[0]-xy[0])<dx/2. and np.abs(source[1]-xy[1])<dy/2.:
#            return 0
#        else:
#            return (source[0]-xy[0])**2+(source[1]-xy[1])**2
#    event.terminal = True
#    sol = solve_ivp(descent,[0,5000],[receiver[0],receiver[1]],events=event,dense_output=True)
#    return sol.y.T
    
if __name__ == "__main__":
    main()
