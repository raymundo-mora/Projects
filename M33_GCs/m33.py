'''
Author: RaymundoMora 
@linkedin: https://www.linkedin.com/in/raymundo-mora/ 
@github: https://github.com/raymundo-mora 

Untitled-2 (c) 2021
Description: This file contains all of the helper functions created to work
with our LBT data.

Created: 2021-11-19T03:06:51.277Z
Modified 2021-22-19T03:06:51.337Z:
''' 

import os
from os.path import exists
import warnings
import re

import pandas as pd
import numpy as np

from skimage import restoration

from astropy.io import fits
from astropy import wcs
from astropy.wcs.utils import skycoord_to_pixel
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, Galactic, FK4, FK5
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.table import Table, vstack
from astropy.wcs.utils import pixel_to_skycoord

from photutils.isophote import Ellipse
from photutils.isophote import EllipseGeometry


import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf 
from matplotlib.colors import LogNorm

from scipy.optimize import curve_fit
#######################################################################################################
#######################################################################################################
#######################################################################################################
def ccd_dict(LBT_filter='r-SLOAN') -> dict:
    """
    Creates a dictionary of all images in a LBT filter in our data. Images are 
    separated by field -> chip. 

    Args:
    LBT_filter : (str), optional
        DESCRIPTION. The default is 'r-SLOAN'. Takes the name of one of the 
        six filters in the M33 data.

    Returns:
    ccd_dict : (dict) 
        Dictonary of all images in every field + chip in the filter. 

    """

    # Initialize the ccd dictionary .
    ccd_dict = {}   

    # Root directory the LBT data.    
    startpath = '/home/ray/m33/astrometry_solved/'

    # Root directory of the data in specified filter. 
    startpath = startpath+LBT_filter
    
    # Get the path to each file in the directory and add it to the 'ccd_dict'
    for root, dirs, files in os.walk(startpath):
            for f in files:
                if f.endswith('.fits')== False:
                    continue
                else:
                    key = root[40:46]+' '+root[47:52]
                    ccd_dict[key] = []
                    for f in files:
                        if f.endswith('.fits'):
                            ccd_dict[key].append((root+'/'+f))
   
    # Print an example of key:value pair in 'ccd_dict' to make usage 
    # and retrieval easier. 
    print('_'*80)
    print("Dictionary created")
    print("Example Key:Value pair:")
    print('{',list(ccd_dict.keys())[0],':',ccd_dict[list(ccd_dict.keys())[0]],'}')

    # Sort the Keys in 'ccd_dict' by Field # and Chip # 
    ccd_dict = sorted(ccd_dict.items())
    ccd_dict = dict(ccd_dict)
    
    return ccd_dict
#######################################################################################################
#######################################################################################################
#######################################################################################################
def mosaic_info(file_list: list) -> None:
    """
    Returns information about a mosaic fits file that is given to the 
    function.

    Args:
        file (list of strings): the location + file of the fits image
        of interest
        

    Returns:
        None 
    """
    # Open the fits file of interest
    for file in file_list:
        hdulist = fits.open(file, mode='readonly', ignore_missing_end=True)
        
        # Get the information of interst from the fits file
        name = hdulist[0].header['FILENAME']
        im_type = hdulist[0].header['OBJECT']
        date = hdulist[0].header['DATE_OBS'][:10]
        instrument = hdulist[0].header['INSTRUME']
        band = hdulist[0].header['FILTER']
        lbcfwhm = hdulist[0].header['LBCFWHM']

        # Print the information of interest
        print('File: ', name)
        print('Date:', date)
        print('Filter: ', band)
        print('Type:', im_type)
        print('Camera:', instrument)
        print('FWHM: ', lbcfwhm)
        print('_'*40)
        hdulist.close() # close the hdulist to take less memomry space
    
    return None
#######################################################################################################
#######################################################################################################
#######################################################################################################
def fits_info(file_list: list):
    
    """
    Returns information about a fits file that is given to the 
    function.

    Args:
        file (list of strings): the location + file of the fits image
        of interest
        

    Returns:
        None
    """
    # We generally consider this bad practice, this is to avoid 
    # printing an excess amount of information about the opening of 
    # of our fits files that we do not care about.   
    warnings.filterwarnings("ignore")
    # Open the fits file of interest
    for file in file_list:
        hdulist = fits.open(file, mode='readonly', ignore_missing_end=True)
        
        # Get the information of interst from the fits file
        
        im_type = hdulist[0].header['OBJECT']
        date = hdulist[0].header['DATE_OBS'][:10]
        band = hdulist[0].header['FILTER']

        # Print the information of interest

        print('Date:', date)
        print('Filter: ', band)
        print('Type:', im_type)
        print('_'*40)
        hdulist.close() # close the hdulist to take less memomry space
    
    return None
#######################################################################################################
#######################################################################################################
#######################################################################################################
def world2pix(image: str, RA: float, DEC: float) -> np.array:
    """Takes RA and DEC of an object (in degrees) and 
    returns the pixel position (x,y) of that object in 
    the provided image.

    Args:
        image (str): Name of the ccd file where the object is located   
        RA (float): RA of your object in degrees
        DEC (float): DEC of your object in degrees
    Returns:
        [np.array]: np.array of length 2 with the x and y pixel positions
        of the target in the given image. 
    """
    # Load the FITS hdulist using astropy.io.fits
    hdulist = fits.open(image, mode='readonly', ignore_missing_end=True)

    # Parse the WCS keywords in the primary HDU
    w = wcs.WCS(hdulist[0].header)

    # Convert the same coordinates back to pixel coordinates.
    # print('wcs to pix',pixcrd2)
    coord = SkyCoord(ra=RA,dec=DEC,unit='deg',frame=FK5)
    pix_coord = skycoord_to_pixel(coord, w)
    hdulist.close() # close the hdulist to take less memomry space
    
    # conver pix_coord to a numpy array
    pix_coord = np.array(pix_coord)
    return pix_coord   
#######################################################################################################
#######################################################################################################
#######################################################################################################
def psf(r: list, *pars: np.array) -> np.array:
    """   Creates the Point Spread Function with the given parameters. This PSF function is
    the same one described in Wang & Ma (2013)

    Args:
        r (list): list of floats, the radii at which to evaluate the psf,
        unit: pixels
        
        *pars (np.array): tuple of floats that define the parameters of the psf 
        unit: pixels

    Returns:
        A numpy.array of the PSF evaluated at all points given in r with the 
        parameters provided by *pars
    """
    # Parameters describing the PSF 
    r0, alpha, beta, off = pars
    
    return (1+ (r/r0)**(alpha))**(-beta/alpha) # PSF from Wang & Ma (2013)
#######################################################################################################
#######################################################################################################
#######################################################################################################
def king(r: list, *pars: np.array) -> np.array:
    """
    King profile, from King (1962).

    Notes:
        -- The `*` notation tells Python to expand the `pars` list/array.
           `king(r, *pars)` is equivalent to `king(r, k, rc, rt, off)`.

        -- `k` is the scale factor (or amplitude) of the profile.
    Args:
        r (list): list of floats, the radii at which to evaluate the king profile,
        unit: pixels
    
    *pars (np.array): np.array that defines the parameters of the King profile 
    unit: pixels

    Returns:
        A numpy.array of the King profile evaluated at all points given in r with the 
        parameters provided by *pars
    """
    k, rc, rt, off = pars
    d1 = np.sqrt(1 + (r / rc) ** 2)
    d2 = np.sqrt(1 + (rt / rc) ** 2)
    return k * (1 / d1 - 1 / d2) ** 2 + off
#######################################################################################################
#######################################################################################################
#######################################################################################################
def is_in_image(RA: list,DEC: list,ID: list ,fits_image: str) -> pd.DataFrame:
    """Takes a list of objects and looks for them in 'fits_image' returns 
    a pd.DataFrame of objects found in 'fits_image' with information about 
    the target.  

    Args:
        RA (list): list of RAs of objects of interest
        DEC (list): list of DECs of objects of interest
        ID (list): list of IDs of objects of interest
        fits_image (str): name of fits where objects will be searched for

    Returns:
        pd.DataFrame: df with columns=('id','chip','ra','dec','x','y','file','tag')
    """
    ID = pd.Series(ID)
    RA = pd.Series(RA)
    DEC = pd.Series(DEC)
    # We generally consider this bad practice, this is to avoid 
    # printing an excess amount of information about the opening of 
    # of our fits files that we do not care about.   
    warnings.filterwarnings("ignore")
    
    # Going to use this counter for indeces in case an object is in
    # more than one ccd.
    count = 0

    # Load the FITS hdulist using astropy.io.fits
    hdulist = fits.open(fits_image, mode='readonly', ignore_missing_end=True)

    image_data = fits.getdata(fits_image)
    # Parse the WCS keywords in the primary HDU
    w = wcs.WCS(hdulist[0].header)
    hdulist.close()
    xsize= image_data.shape[1]+1
    ysize= image_data.shape[0]+1

    match_df = pd.DataFrame(columns=('id','chip','ra','dec','x','y','file','tag'))
    coord = SkyCoord(ra=RA,dec=DEC,unit='deg',frame=FK5)
    pix_coord = skycoord_to_pixel(coord, w)
    x = pix_coord[0]
    y = pix_coord[1]
    field = re.search("[fF]ield\d",fits_image).group()
    chip = re.search("[Cc]hip\d",fits_image).group()
    tag = re.search("[fF]\d[Cc]\d",fits_image).group()
    for i in range(len(x)):
        x_int = int(x[i])
        y_int = int(y[i])
        in_x =  x_int in range(xsize)
        in_y =  y_int in range(ysize)
        if in_x and in_y:
            match_df.loc[count] = ([ID[ID.index[i]],
                                field+' '+chip,
                                RA[ID.index[i]],
                                DEC[ID.index[i]],
                                x[i],
                                y[i],
                                fits_image,
                                tag]
                               )
            count += 1
    return match_df
#######################################################################################################
#######################################################################################################
#######################################################################################################
def is_in_filter(source_df,ra_label,dec_label,ID='Name',LBT_filter='r-SLOAN'):    
    """Takes a list of targets and returns a pd.DataFrame of targets found in
    the specified filter. 

    Args:
        source_df (pd.DataFrame): pd.DataFrame
        rame of your sources containing 
                                  ra & dec in degrees and an ID column.
        ra_label (str): Name of your RA column.
        dec_label (str): Name of your DEC column.
        ID (str, optional): Name of your ID column. Defaults to 'Name'
        LBT_filter (str, optional): Filter of interest. Defaults to 'r-SLOAN'.

    Returns:
        pd.DataFrame: df containing relevant information of the targets of interest
        provided. 
    """
    
    from m33 import world2pix
    # Just to make writing a little easier.
    df = source_df    
    
    # Going to use this counter for indeces in case an object is in
    # more than one ccd.
    count = 0
    
    # Get all images in our desired filter.
    images = ccd_dict(LBT_filter)    
   
    # Get the list of all our RAs and DECs of interest.
    ra = df[ra_label]
    dec = df[dec_label]
    ID = df[ID]
    # Initiate a data frame for all the matches we get 
    # and make so that we can easily record the data we are interested in. 
    target_df = pd.DataFrame([],columns=('ccd',
                                 'ID',
                                 'frame_x',
                                 'frame_y',
                                 'x_coord',
                                 'y_coord',
                                 'chip',
                                 'ra',
                                 'dec',
                                 'tag',
                                 'cutout_data')
                         )
   
    # Iterate through the images in each field to see if we find any of our 
    # targets in the field. Record if there is a match and save information of 
    # interest in a pd.DataFrame. 
    for chip in images:
        # In case we have multiple exposures of the same field 
        # in the same filter, just use the first one. 
        ccd = images[chip][0]
        field = re.search("[fF]ield\d",ccd).group()
        chip_num = re.search("[Cc]hip\d",ccd).group()
        tag = re.search("[fF]\d[Cc]\d",ccd).group()
        
        
    
        # Look for our target in the current image. 
        match_df = is_in_image(ra, dec,ID, ccd)    
        # If we have targets in the current image, do the following. 
        if len(match_df) !=0:
            for i in match_df.index:
            
                # Load the data of the current image.
                image_data = fits.getdata(ccd)
                # Get the pixel position of our found target in the current image.
                position = world2pix(ccd,match_df['ra'][i],match_df['dec'][i])
                # Set the size of the 2D cutout we want to get centered at our 
                # target.
                size= (100,100)
                
                
                cutout = Cutout2D(image_data, position, size)
                coord = cutout.to_cutout_position(position)
                
                target_df.loc[count] = (field+' '+chip_num,
                                        match_df['id'][i],
                                        position[0],
                                        position[1],
                                        coord[0],
                                        coord[1],
                                        ccd,
                                        match_df['ra'][i],
                                        match_df['dec'][i],
                                        match_df['tag'][i],
                                        cutout.data)
                count += 1
    return target_df
#######################################################################################################
#######################################################################################################
#######################################################################################################
def display_target(ra: float,dec: float,name: str, inspect=False) -> list:
    """Returns plots of the frame and a 100 by 100 cutout
    of where the target is found. 

    Args:
        ra (float): RA of the target in degrees
        dec (float): DEC of the target in degrees
        name (str): name or ID of the target
    Returns: list: List of figures of where our target was found if inspect == True. If 
             inspect == False returns empty list. 
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    plots = []
    # Load up the M33 images 
    images = ccd_dict()
    
    # Loop through the M33 images and see in which one if any our target is in
    for chip in images:
        for exposure in images[chip]:
            match_df = is_in_image([ra],[dec] ,[name], exposure)

            #keep track of what field and chip the source was found in 
            if 'SLOAN' in exposure:
                field = exposure[40:46]
                chip_num = exposure[47:52]
            
            if 'Bessel' in exposure:
                field = exposure[41:47]
                chip_num = exposure[48:53]
            
            # Create a figure if we found our object in the current image we are looping 
            # through.    
            for i in match_df.index:
                    # Load the FITS hdulist using astropy.io.fits
                    hdulist = fits.open(exposure, mode='readonly', ignore_missing_end=True)
                    # Load Header
                    w = wcs.WCS(hdulist[0].header)
                    cmap = 'gnuplot2'
                    # Figure 1    
                    fig, ax = plt.subplots(1,2)
                    fig.suptitle(f"{name} \n {exposure}")  
                    image_data = fits.getdata(exposure)

                    # Get 100X100 cutout of our image centered at our target
                    coord = SkyCoord(ra=ra,dec=dec,unit='deg',frame=FK5)        
                    position = skycoord_to_pixel(coord, w)
                    size= (100,100)
                    cutout = Cutout2D(image_data, position, size)
                    
                    ## Figure 1 Axis 0   
                    
                    # Add circle centered at our taget                                      
                    coord = cutout.to_cutout_position(position)
                    circle = plt.Circle(coord, 5,color ='red', fill=False)
                    ax[0].add_patch(circle)
                    
                    # Display our Data
                    im1 = ax[0].imshow(cutout.data,origin='lower',vmin=0,vmax=10000,cmap=cmap)
                    
                    ### Figure 1 Axis 0 Colorbar
                    divider = make_axes_locatable(ax[0])
                    cax = divider.append_axes("bottom", size="5%", pad=0.1)
                    fig.colorbar(im1, cax=cax,orientation='horizontal')     
                    
                    ## Figure 1 Axis 1
                    coord = SkyCoord(ra=ra,dec=dec,unit='deg',frame=FK5)  
                    position = skycoord_to_pixel(coord, w)    
                    size = (100,100)
                    
                    # Add square centered at our target                  
                    rectangle = plt.Rectangle(position,color='red',height= 100,width= 100, fill=False)
                    ax[1].add_patch(rectangle)
                    
                    # Display image where our target was found
                    sigma_clip = 1*np.std(image_data)
                    mdd = np.mean(image_data)
                    im2 = ax[1].imshow(image_data,origin='lower',vmin=0,vmax=10000,cmap=cmap)
                    
                    ### Figure 1 Axis 1 Colorbar
                    divider = make_axes_locatable(ax[1])
                    cax = divider.append_axes("right", size="5%", pad=0.1)
                    fig.colorbar(im2, cax=cax) 
                    
                    #### Tidy Up
                    fig.tight_layout()
                    
                    if inspect == True: 
                        plots.append(fig)
    return plots   
###############################################################################
###############################################################################
###############################################################################
def hwhm(pars: np.ndarray) -> float:
    """Gets the HWWM of of the psf defined by 'pars' as described in psf() from the 
    PSF function in Wang & Ma (2013). 

    Args:
        pars (np.ndarray): pars describing the PSF in psf()

    Returns:
        float: hwhm radius in pixels
    """
    r = [2]# bad initial guess! 
    counter = 1 # Simple/non-sophisticated way to converge to the answer without
                # oscilating around it indefinetely
    accuracy = 0.00005 # tolerance in error between true value and 0.5
    while((abs(psf(r,*pars)- 0.5)) > accuracy): 
        if (psf(r,*pars)- 0.5) > 0:
            r[0] = r[0]+r[0]/counter # if we underestimate r subtract r/counter
            counter += 1
        elif (psf(r,*pars)- 0.5) < 0:
            r[0] = r[0]-r[0]/counter # if we overestimate r add r/counter
            counter +=1
    return r[0]        
###############################################################################
###############################################################################
###############################################################################
def bkg_subtraction(data: np.ndarray,inspect=False) -> (np.ndarray, list,float):
    """Estimates the 2-Dimensional background of an image data provided and returns
    the background subtracted data and a list of all the figures relevant to the subtraction. 

    Args:
        data (np.ndarray): Data pertaning to the image we will estimate and subtract the
                           background from. 
        inspect (bool): Whether or not to generate relevant plots and return them in a list. 
    Returns:
        np.ndarray: background subtracted data.
        list: list of relevant plots if 'inspect' == True. 
    """
    import matplotlib.pyplot as plt
    from photutils.background import Background2D
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.backends.backend_pdf import PdfPages
    
    # initialize the figure list
    fig_list = []
    
    # bkg = Background2D(data, (20, 20),filter_size=(3,3),exclude_percentile=14)
    bkg = Background2D(data, (25, 25),filter_size=(5,5),exclude_percentile=10)
    background_subtracted_data = data - bkg.background
    
    bkg_range = np.max(bkg.background) - np.min(bkg.background)
    # Create figures to visualize our background subtraction  
    if inspect == True:
        
        # Figure 1
        fig,ax = plt.subplots(1,2)

        fig.suptitle(f"Background Range: {int(bkg_range)} counts\n Background Median: {int(np.median(bkg.background))}")
        
        ## Figure 1 Axis 0
        ax[0].set_title('Data')
        im1 = ax[0].imshow(data, origin='lower')
        bkg.plot_meshes(axes=ax[0],outlines=True)
              
        #### Figure 1 Axis 0 Colorbar  
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im1, cax=cax)        
                
        ## Figure 1 Axis 1
        ax[1].set_title('Background')
        im2 = ax[1].imshow(bkg.background, origin='lower')
        bkg.plot_meshes(axes=ax[1],outlines=True)
        
        #### Figure 1 Axis 1 Colorbar
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im2, cax=cax)      

        #### Tidy Up
        fig.tight_layout()
        fig_list.append(fig)        
                
        # Figure 2
        minmin = np.min((np.min(bkg.background),np.min(background_subtracted_data)))
        maxmax = np.max((np.max(bkg.background),np.max(background_subtracted_data)))
        
        fig, ax = plt.subplots(1,2)

        ## Figure 2 Axis 0
        ax[0].set_title('Background')
        im1 = ax[0].imshow(bkg.background,vmin=minmin,vmax=maxmax,origin='lower')
        
        ## Figure 2 Axis 1 
        ax[1].set_title('Data - Background')
        im2 = ax[1].imshow(background_subtracted_data,vmin=minmin,vmax=maxmax,origin='lower')

        ### Figure 2 Colorbar
        p0 = ax[0].get_position().get_points().flatten()
        p1 = ax[1].get_position().get_points().flatten()
        cax = fig.add_axes([p0[0], 0.15, p1[2]-p0[0], 0.02])
        plt.colorbar(im2, cax=cax, orientation='horizontal')
        
        #### Tidy Up
        fig_list.append(fig)
 
        # Figure 3
        minmin = np.min((np.min(data),np.min(background_subtracted_data)))
        maxmax = np.max((np.max(data),np.max(background_subtracted_data)))
        
        fig,ax = plt.subplots(1,2)
        
        ## Figure 3 Axis 0 
        ax[0].set_title('Data')
        im1 = ax[0].imshow(data,vmin=minmin,vmax=maxmax,origin='lower')
        
        ## Figure 3 Axis 1
        ax[1].set_title('Data - Background')
        im2 = ax[1].imshow(background_subtracted_data,vmin=minmin,vmax=maxmax,origin='lower')

        ### Figure 3 Colorbar
        p0 = ax[0].get_position().get_points().flatten()
        p1 = ax[1].get_position().get_points().flatten()
        cax = fig.add_axes([p0[0], 0.15, p1[2]-p0[0], 0.02])
        plt.colorbar(im2, cax=cax, orientation='horizontal')
        
        #### Tidy Up
        fig_list.append(fig)
        
    return background_subtracted_data, fig_list
#######################################################################################################
#######################################################################################################
#######################################################################################################
def make_isophote(data: float,x0: float,y0: float,ID,inspect = False) -> tuple:
    """Generates an 'Isophotelist' instance from 
    photutils.isophote.Ellipse.fit_image() for the data provided
    for an object centered at x0, y0. 

    Args:
        data (np.ndarray): 2D np.ndarray, your image data.
        x0 (float): x coordiante of your object in 'data'.
        y0 (float): y coordinate of your object in 'data'.
        ID (str/float): Object ID. 
        inspect (bool, optional): If True, generates plots describing the
        Isophotelist instance. Defaults to False.

    Returns:
        tuple: instance, photutils.isophote.Ellipse.fit_image().
               list,  
    """

    
    fig_list = [] # Initiate figure list. 
    max_sma = 50 # sma size at which to stop trying to make a fit 
    # initial parameters for EllipseGeometry.
    step = 1.0 #stepsize between isophote fittings, in pixels.
    sma = 1.5 # Initial sma to try 'ellipse.fit_image'.
    eps = 0.1 # Initial eps to try 'ellipse.fit_image'.
    pa = 0.1 # Initial pa to try 'ellipse.fit_image'.

    geometry = EllipseGeometry(x0=x0, y0=y0,sma=sma, eps=eps,
                                pa=pa) # Initialize 'geometry' for 'ellipse'.
    ellipse = Ellipse(data,geometry) 
    isolist = [] # Create an enpty object for isolist so we have something to check
                 # for length if 'ellipse.fit_image' does not work. 

    # Attempt to fit an isophote to our data, if it fails attempt again with a larger
    # sma. Stop when sma > max_ma or sucessful fit. 
    while(len(isolist)==0) and sma < max_sma:     
            geometry = EllipseGeometry(x0=x0, y0=y0,sma=sma, eps=eps,
                                pa=pa)
            ellipse = Ellipse(data,geometry)        
            isolist = ellipse.fit_image(linear=True,step=step,maxgerr=4)
            
            sma+= 1
            # If we have a sucessfull fit and inspect == True then
        # Generate Figures        

    if len(isolist) != 0 and inspect == True:

            # Figure 1 
            # counts v sma
            fig = plt.figure()
            plt.title(str(ID)+' isophotes')
            plt.xlabel('sma (pixels)')
            plt.ylabel('intensity (counts')
            
            x = isolist.sma
            y = isolist.intens
            plt.scatter(x, y)
            
            #### Tidy Up
            plt.tight_layout()
            fig_list.append(fig)
            
            
            # Figure 2 
            fig, ax = plt.subplots(1,2)
            fig.suptitle(str(ID)+' isophotes')
            
            ## Figure 2 Axis 0 
            ax[0].imshow(data,origin='lower')
            
            # Generate cirlces that show the isophotes
            smas = np.linspace(1.5, 25.5, 8)
            for sma in smas:
                iso = isolist.get_closest(sma)
                x,y = iso.sampled_coordinates()
                ax[0].plot(x,y, color='white')
                
            ## Figure 2 Axis 1 
            ax[1].imshow(data,origin='lower')
            
            # Generate circles that show the isophotes 
            smas = np.linspace(1.5, 25.5, 4)
            for sma in smas:
                iso = isolist.get_closest(sma)
                x,y = iso.sampled_coordinates()
                ax[1].plot(x,y, color='white')
            
            #### Tidy Up 
            plt.tight_layout()
            fig_list.append(fig)
        
        
            # Figure 3 
            fig, ax = plt.subplots(2,2) 

            ## Figure 3 Axis 0
            # eps v sma
            ax[0,0].errorbar(isolist.sma, isolist.eps, yerr=isolist.ellip_err,fmt='o')
            ax[0,0].set_xlabel('Semimajor Axis Length (pix)')
            ax[0,0].set_ylabel('Ellipticity')

            ## Figure 3 Axis 1
            # pa v sma
            ax[0,1].errorbar(isolist.sma, isolist.pa / np.pi * 180.,
                         yerr=isolist.pa_err / np.pi * 80., fmt='o', markersize=4)
            ax[0,1].set_xlabel('Semimajor Axis Length (pix)')
            ax[0,1].set_ylabel('PA (deg)')

            ## Figure 3 Axis 2 
            # x0 v sma
            ax[1,0].errorbar(isolist.sma, isolist.x0, yerr=isolist.x0_err, fmt='o',
                        markersize=4)
            ax[1,0].set_xlabel('Semimajor Axis Length (pix)')
            ax[1,0].set_ylabel('x0')

            ## Figure 3 Axis 3
            # y0 v sma
            ax[1,1].errorbar(isolist.sma, isolist.y0, yerr=isolist.y0_err, fmt='o',
                            markersize=4)
            ax[1,1].set_xlabel('Semimajor Axis Length (pix)')
            ax[1,1].set_ylabel('y0')
            
            #### Tidy Up
            plt.tight_layout()
            fig_list.append(fig)            
                    
    return isolist, fig_list   
#######################################################################################################
#######################################################################################################
#######################################################################################################
def make_psf(starlist: pd.DataFrame,RA_label: str,DEC_label: str,id_label: str,
             chip: str,inspect=False) -> tuple:
    """Takes a pd.DataFrame containing the location of reference stars used to 
    construct the PSF of an image. The psf is constructed by obtaining isophote 
    information from each star using photutils. Since the isophote information 
    is obtained using consistent sma intervals we get the average intensity at 
    each sma and then normalize it to construct our psf. 

    Args:
        starlist (pd.DataFrame): Contains the location for each reference star used
        to construct our PSF. 
        RA_label (str): Name of the RA column in 'starlist'.
        DEC_label (str): Name of the DEC column in 'starlist'.
        ID (str): Name of the column used to identify each object in 'starlist'. 
        chip (str): Tag of the field+chip of interest (e.g. f1c1). 
        inspect (bool, optional): If True creates and returns list of figures 
        that describe the psf construction. Defaults to False.

    Returns:
        tuple: pars, contains a list of parameters that describe the psf. 
               fig_list, contains a list of figures. Empty if inspect== False.
               seeing, a list of the psf FWHM in pixels and arcseconds. 
    """
    from m33 import psf

    ra_col = starlist[RA_label]
    dec_col = starlist[DEC_label]
    id_col = starlist[id_label]
        
    # If a reference star is in multiple exposures get only the exposure of 
    # interest. 
    in_filter = is_in_filter(starlist, RA_label, DEC_label,id_label)
    stars = in_filter[in_filter['tag'] == chip]
    # Initialize a DF for the isophotes of each star. 
    isophotes = pd.DataFrame(columns=('ccd','ra','dec','isolist'))

    # The index for each star will be its ID.
    isophotes.index.name = 'ID'

    # Initilize the lists to contain the sma and intensities of our reference stars.
    psf_sma = []
    psf_intens = []

    for i in stars.index:
        data = stars['cutout_data'][i]
        x0 = stars['x_coord'][i]
        y0 = stars['y_coord'][i]
        ID = stars['ID'][i]
        ccd = stars['ccd'][i]
        ra = stars['ra'][i]
        dec = stars['dec'][i]

        # Preprocess our reference stars by performing a background subtraction.
        data = bkg_subtraction(data)[0]
        
        # Generate an isolist for our reference stars. 
        iso = make_isophote(data,x0,y0,ID)[0] 
        
        # If isolist generation was successful add it
        if len(iso) != 0:
            isophotes.loc[ID] = (ccd,
                                 ra,
                                 dec,
                                 iso
                                 )
    
    npsf = len(isophotes)
    min_list = 500
    for i in isophotes.index:
        if len(isophotes['isolist'][i].sma) < min_list:
            min_list = len(isophotes['isolist'][i].sma)

    star_sum_intens = np.full(min_list,0.0)
    for i in isophotes.index:
        
        star_sum_intens += isophotes['isolist'][i].intens[:min_list]

    star_avg_intens = star_sum_intens/len(isophotes.index)

    y = (star_avg_intens - min(star_avg_intens)) / (max(star_avg_intens) - min(star_avg_intens))
    x = isophotes['isolist'][i].sma[:min_list]

    guess = [1, 2, 3.8, 0.4]  # bad initial guess!


    pars, pars_cov = curve_fit(psf, x, y, p0=guess)
    
    fig_list = []

    fwhm_pix = hwhm(pars)*2
    fwhm_arc = hwhm(pars)*2*0.2255

    # Generate plots if inspect == True
    if inspect == True:
            
        psf_count = 0
        for i in isophotes.index:
            if (psf_count % 4) == 0:
                
                # Figure 1 
                fig, ax = plt.subplots(2,2)
                fig.suptitle(f"Surface Brightness Profiles of {isophotes['ccd'].unique()[0]} PSF stars")
                
                fig.text(0.5, 0.00, 'Radius (pixels)', ha='center')
                fig.text(0.0, 0.5, 'Inensity (ADUs)', va='center', rotation='vertical')



                sma = isophotes['isolist'][i].sma
                intens = isophotes['isolist'][i].intens

                psf_sma.append(sma)
                psf_intens.append(intens)
                
                ## Figure 1 Axis 0 
                ax[0,0].plot(sma,intens) 
                ax[0,0].set_title(f"{isophotes['ra'][i]} {isophotes['dec'][i]}")

                fig.tight_layout(pad=0.2, w_pad=1, h_pad=1.5)
                
                fig_list.append(fig)
                
                
            ## Figure 1 Axis 1 
            if (psf_count % 4) == 1:

                sma = isophotes['isolist'][i].sma
                intens = isophotes['isolist'][i].intens

                psf_sma.append(sma)
                psf_intens.append(intens)
                ax[0,1].plot(sma,intens) 
                ax[0,1].set_title(f"{isophotes['ra'][i]} {isophotes['dec'][i]}")

            ## Figure 1 Axis 2 
            if (psf_count % 4) == 2:

                sma = isophotes['isolist'][i].sma
                intens = isophotes['isolist'][i].intens

                psf_sma.append(sma)
                psf_intens.append(intens)
                ax[1,0].plot(sma,intens) 
                ax[1,0].set_title(f"{isophotes['ra'][i]} {isophotes['dec'][i]}")
    
            #### Tidy Up
            plt.tight_layout()
            ## Figure 1 Axis 3 
            if (psf_count % 4) == 3:

                sma = isophotes['isolist'][i].sma
                intens = isophotes['isolist'][i].intens

                psf_sma.append(sma)
                psf_intens.append(intens)
                ax[1,1].plot(sma,intens) 
                ax[1,1].set_title(f"{isophotes['ra'][i]} {isophotes['dec'][i]}")

            psf_count += 1             
                
        # Figure 2 
        fig = plt.figure()
        plt.title('Averaged Intenities')
        plt.plot(x,np.full(len(x),0))
        plt.scatter(x,star_avg_intens)
        
        
        xfitrange = np.linspace(min(x),max(x),1000)
        yfit = psf(xfitrange, *pars)
        
        #### Tidy Up
        plt.tight_layout()
        fig_list.append(fig)
        
        # Figure 3 
        fig = plt.figure()
        plt.title('PSF')
        plt.plot(x,np.full(len(x),0))
        plt.scatter(x, y)
        
        plt.plot(xfitrange,yfit)
        
        #### Tidy Up
        plt.tight_layout()
        fig_list.append(fig)
        
        # Figure 4 
        fig = plt.figure()
        plt.title('PSF Overlay')
        plt.xlabel('Radius (pixels)')
        plt.ylabel('Intensity')
        for i, star in enumerate(psf_sma):
            plt.plot(psf_sma[i],psf_intens[i])
        
        #### Tidy Up
        plt.tight_layout()
        fig_list.append(fig)
        
        # Figure 5 
        fig = plt.figure()
        plt.title('PSF Overlay')
        plt.xlabel('Radius (pixels)')
        plt.ylabel('Normalized Intensity')
        for i, star in enumerate(psf_sma):
            normy = (psf_intens[i] - min(psf_intens[i])) / (max(psf_intens[i]) - min(psf_intens[i]))
            plt.plot(psf_sma[i],normy, c='deepskyblue')

        
        ynew = psf(xfitrange,*pars)
        xnew = xfitrange
        plt.plot(xnew,ynew,c='r',label='master psf')
        plt.legend()

        #add HWHM 
        half_r = hwhm(pars)
        half_max = psf(half_r,*pars)
        plt.plot([-5,half_r],np.full(2,psf([half_r],*pars)),c='g',label='HWHM')# horizontal
        plt.plot(np.full(2,half_r),[0.25,0.75],c='g')# vertical
        plt.xlim(0,13)
        plt.legend()
        #### Tidy Up
        plt.tight_layout()
        fig_list.append(fig)
    
    seeing = [fwhm_pix,fwhm_arc]
                    
    
    return pars, fig_list, seeing
#######################################################################################################
#######################################################################################################
#######################################################################################################
def generate_psf(lbt_filter:str, ccd: str) -> pd.DataFrame:
    """Utiizes 'make_psf' to streamline the process for making a PSF 
    for each chip and saves relevant data in a csv file under
    'datasets/psf_params.csv'. 
    REQUIRES: That the psf to be created has a list of reference stars
    for the chip under 'datasets/M33_psf_starlist.csv'. 
    

    Args:
        lbt_filter (str): name of the filter of the image we want to
        make a psf for.
        ccd (str): 

    Returns:
        pd.DataFrame: The df that was saved as 'psf_params.csv' 
    """
    dir_exists = exists('datasets/')
    if dir_exists == False:
        os.mkdir('datasets/')
    file_exists = exists('datasets/psf_params.csv')
    if file_exists == False:
        df = pd.DataFrame(columns=('filter','ccd','r0','alpha',
                                   'beta','off','fwhm','seeing'))
        df.to_csv('datasets/psf_params.csv',index_label='chip')
        
    psf_stars = pd.read_csv('datasets/M33_psf_starlist.csv')
    psf_stars = psf_stars[psf_stars['Filter'] == lbt_filter]
    psf_stars = psf_stars[psf_stars['FIELD+CHIP'] == ccd]
    
    psf = make_psf(psf_stars, 'RA', 'DEC', 'ID', ccd)
    
    pars = psf[0]
    figs = psf[1]
    seeing = psf[2]
    
    ID = lbt_filter[0]+ccd
  
    r0 = pars[0]
    alpha = pars[1]
    beta = pars[2]
    off = pars[3]
    
    
    psf_params = pd.read_csv('datasets/psf_params.csv', index_col='chip')
    psf_params.loc[ID] = (lbt_filter,ccd,r0,alpha,beta,off,seeing[0],seeing[1])
      
    psf_params.to_csv('datasets/psf_params.csv',index_label='chip')
    
    return psf_params
#######################################################################################################
#######################################################################################################
#######################################################################################################
"""
@author: luca (lucabeale@gmail.com)
"""

def deconvolve_PSF(data, r0, alpha, beta, Niterations=10, inspect=False):
    """
    Takes in an image (FITS or pkl) and deconvolves it from its PSF which is 
    built by the PSF parameters provided (i.e. r0, alpha, beta)    

    Parameters
    ----------
    data : (pkl)
        DESCRIPTION. Image that is to be deconvolved from its PSF. 
    r0 : (float)
        DESCRIPTION. In pixels. r0 that describes the PSF of the image and 
        will be plugged into 'PSF2D'. 
    alpha : (float)
        DESCRIPTION. 'alpha' that describes the PSF of the image and will be 
        plugged into 'PSF2D'. 
    beta : (flaot)
        DESCRIPTION. 'beta' that describes the PSF of the image and will be 
        plugged into 'PSF2D'
    Niterations : (int), optional
        DESCRIPTION. The default is 10. The number of iterations that will be 
        plugged into 'richardson_lucy' deconvolution function. 
    inspect : (bool), optional
        DESCRIPTION. The default is False. Whether or not plots of the raw 
        data and the deconvolved data will be displayed for comparison. 

    Returns
    -------
    (np.ndarray)
        DESCRIPTION. An array with the same shape of 'data' that is the 
        original data deconvolved with its PSF. 

    """
    # helper functions
    def PSF2D(xy, *pars):
        I0, r0, a, b = pars
        x, y = xy
        r = np.sqrt(x ** 2 + y ** 2)
        return I0 * (1 + (r / r0) ** a) ** (-b / a)
    
    def make2Dkernel(model, model_pars, scale_factor):
        # define kernel out to M times the scale length (in pixels) of the profile
        kernel_radius = np.ceil(10 * scale_factor) // 2 * 2  # rounds up to nearest even integer
        # make kernel grid centered on (0, 0)
        xspan, yspan = np.arange(-kernel_radius, kernel_radius + 1, 1), np.arange(-kernel_radius, kernel_radius + 1, 1)
        xgrid, ygrid = np.meshgrid(xspan, yspan)
        # populate kernel
        kernel = model((xgrid, ygrid), *model_pars)
        # normalize -- final kernel must have sum(kernel) = 1
        kernel /= kernel.sum()
        return kernel

    # make kernel
    PSF_pars = [1, r0, alpha, beta]  # I0 is irrelevant since kernel is normalized anyway
    kernel_PSF = make2Dkernel(PSF2D, PSF_pars, r0)

    # deconvolve data
    data_deconvolved = restoration.richardson_lucy(data, kernel_PSF, iterations=Niterations, clip=False)
    iters = Niterations
    
    
    if inspect == True:
        fig, ax = plt.subplots(1,2,sharey='row')
        
        MAX = np.max(data)
        ax[0].set_title('Data')
        ax0 = ax[0].imshow(data, vmin = 0, vmax = MAX, origin = 'lower')
        plt.colorbar(ax0, ax= ax[0])

        ax[1].set_title('Deconvolved')
        ax1 = ax[1].imshow(data_deconvolved,vmin = 0,vmax = MAX, origin = 'lower')
        plt.colorbar(ax1)
        plt.suptitle(f'Niterations: {Niterations}')

    return data_deconvolved
#######################################################################################################
#######################################################################################################
#######################################################################################################
def fit_king(x,y, inspect=False, xlog=False):

    from scipy.optimize import curve_fit
    # fitting
    guess = [max(y), 1.5*x[1], max(y)*0.5, min(y)]  # bad initial guess!

    pars, pars_cov = curve_fit(king, x, y, p0=guess)  # pars_cov is a matrix of variances
    errs = np.sqrt(np.diag(pars_cov))  # sqrt of diagonal entries gives σ, the formal uncertainty on each parameter

    # When both optional parameters are true then the function returns a 
    # plot of log10(intensity) vs log10(radii). 
    if inspect == True and xlog == True:
        xfitrange = np.linspace(min(x),max(x),1000)
        yfit = king(xfitrange, *pars)
        
        
        plt.figure()
        plt.scatter(np.log10(x),np.log10(y),label='Data')
        plt.plot(np.log10(xfitrange),np.log10(yfit),label='King fit',color='r',)
     
        plt.xlim([0,max(np.log10(x))])
        plt.xlabel('log (radii)')
        plt.ylabel('log (ADU)')
        plt.legend()
        
    # If only 'inspect' = True , then it returns a plot of log10(ADU) v 
    # radii [pixels]. 
    elif inspect == True and xlog == False:
        xfitrange = np.linspace(min(x),max(x),1000)
        yfit = king(xfitrange, *pars)
        
        
        plt.figure()
        plt.scatter(x,np.log10(y),label='Data')
        plt.plot(xfitrange,np.log10(yfit),label='King fit',color='r',)
        plt.xlabel("Radii (pixels)")
        plt.ylabel("log (ADU)")
        plt.xlim([min(x),max(x)])
        plt.legend()
    
    return pars
#######################################################################################################
#######################################################################################################
#######################################################################################################
def px2distance(pixels):
    """
    Convert the distance between two objects in M33 on the LBT data from 
    pixels to parsecs.     
    """
    lbt_pixel_scale = 0.2255 #arcescond/pixel
    arcsec_per_radian = 206265 #arcsec/radian
    distance_to_m33 =834875.6# in parsecs

    distance = (pixels * lbt_pixel_scale * distance_to_m33) / arcsec_per_radian
    
    return distance
#######################################################################################################
#######################################################################################################
#######################################################################################################
def get_parameters(rc,rt,log=False):

    
    
    import numpy as np
    from m33_dictionary import px2distance
    
    rc = px2distance(rc)
    rt = px2distance(rt)
    
    if log==True:
        rc = np.log10(rc)
        rt = np.log10(rt)
    return rc,rt