#!/usr/bin/python
#
#Python Class file for Microscope.
#
#Written by CD Phatak, ANL, 20.Feb.2015.
#
# modified to keep only relevant functions for demonstrating forward model - CD, ANL, 15.Sep.2019.

import numpy as np
import scipy.constants as physcon
import scipy.ndimage as ndimage
from skimage import io as skimage_io
from skimage import color as skimage_color
from matplotlib import colors as mt_cols


class Microscope(object):

    def __init__(self, E=200.0e3, Cs=1.0e6, Cc=5.0e6, theta_c=6.0e-4, Ca=0.0e6, phi_a=0, def_spr=120.0,verbose=False):
        
        #initialize with either default values or user supplied values - properties that can be changed
        self.E = E#200.0e3
        self.Cs = Cs#1.0e6
        self.Cc = Cc#5.0e6
        self.theta_c = theta_c#6.0e-4
        self.Ca = Ca#0.0e6
        self.phi_a = phi_a#0
        self.def_spr = def_spr#120.0
        self.defocus = 0.0 #nm
        self.aperture = 1.0
        
        #properties that are derived and cannot be changed directly.
        epsilon = 0.5 * physcon.e / physcon.m_e / physcon.c**2
        self.lam = physcon.h * 1.0e9 / np.sqrt(2.0 * physcon.m_e * physcon.e) / np.sqrt(self.E + epsilon * self.E**2)
        self.gamma = 1.0 + physcon.e * self.E / physcon.m_e / physcon.c**2
        self.sigma = 2.0 * np.pi * physcon.m_e * self.gamma * physcon.e * self.lam * 1.0e-18 / physcon.h**2
        
        if verbose:
            print( "Creating a new microscope object with the following properties:")
            print( "Quantities preceded by a star (*) can be changed using optional arguments at call.")
            print( "-------------------------------------------------------------------------")
            print( "*Accelerating voltage         E: [V]      ",self.E)
            print( "*Spherical Aberration        Cs: [nm]     ",self.Cs)
            print( "*Chromatic Aberration        Cc: [nm]     ",self.Cc)
            print( "*Beam Coherence         theta_c: [rad]    ",self.theta_c)
            print( "*2-fold astigmatism          Ca: [nm]     ",self.Ca)
            print( "*2-fold astigmatism angle phi_a: [rad]    ",self.phi_a)
            print( "*defocus spread         def_spr: [nm]     ",self.def_spr)
            print( "Electron wavelength      lambda: [nm]     ",self.lam)
            print( "Relativistic factor       gamma: [-]      ",self.gamma)
            print( "Interaction constant      sigma: [1/V/nm] ",self.sigma)
            print( "-------------------------------------------------------------------------")

    def setAperture(self,qq,del_px, sz):
        #This function will set the objective aperture
        #the input size of aperture sz is given in nm.
        ap = np.zeros(qq.shape)
        sz_q = qq.shape
        #Convert the size of aperture from nm to nm^-1 and then to px^-1
        ap_sz = sz/del_px
        ap_sz /= float(sz_q[0])
        ap[qq <= ap_sz] = 1.0
        #Smooth the edge of the aperture
        ap = ndimage.gaussian_filter(ap,sigma=2)
        self.aperture = ap
        return 1

    def getChiQ(self,qq,del_px):
        #this function will calculate the phase transfer function.
        
        #convert all the properties to pixel values
        lam = self.lam / del_px
        def_val = self.defocus / del_px
        spread = self.def_spr / del_px
        cs = self.Cs / del_px
        ca = self.Ca / del_px
        phi = 0

        #compute the required prefactor terms
        p1 = np.pi * lam * (def_val + ca * np.cos(2.0 * (phi - self.phi_a)))
        p2 = np.pi * cs * lam**3 * 0.5
        p3 = 2.0 * (np.pi * self.theta_c * spread)**2

        #compute the phase transfer function
        u = 1.0 + p3 * qq**2
        chiq = -p1 * qq**2 + p2 * qq**4
        return chiq

    def getDampEnv(self,qq,del_px):
        #this function will calculate the complete damping envelope: spatial + temporal
        
        #convert all the properties to pixel values
        lam = self.lam / del_px
        def_val = self.defocus / del_px
        spread = self.def_spr / del_px
        cs = self.Cs / del_px

        #compute prefactors
        p3 = 2.0 * (np.pi * self.theta_c * spread)**2
        p4 = (np.pi * lam * spread)**2
        p5 = np.pi**2 * self.theta_c**2 / lam**2
        p6 = cs * lam**3
        p7 = def_val * lam

        #compute the damping envelope
        u = 1.0 + p3 * qq**2
        es_arg = 1.0/(2.0*u) * p4 * qq**4
        et_arg = 1.0/u * p5 * (p6 * qq**3 - p7 * qq)**2
        dampenv = np.exp(es_arg-et_arg)
        return dampenv

    def getTransferFunction(self,qq,del_px):
        #This function will generate the full transfer function in reciprocal space-
        chiq = self.getChiQ(qq,del_px)
        dampenv = self.getDampEnv(qq,del_px)
        tf = (np.cos(chiq) - 1j * np.sin(chiq)) * dampenv * self.aperture
        return tf

    def PropagateWave(self, ObjWave, qq, del_px):
        #This function will propagate the object wave function to the image plane
        #by convolving with the transfer function of microscope and returns the 
        #complex real-space ImgWave

        #get the transfer function
        tf = self.getTransferFunction(qq, del_px)
        
        #Compute Fourier transform of ObjWave and convolve with tf
        f_ObjWave = np.fft.fftshift(np.fft.fftn(ObjWave))
        f_ImgWave = f_ObjWave * tf
        ImgWave = np.fft.ifftn(np.fft.ifftshift(f_ImgWave))

        return ImgWave

    def BackPropagateWave(self, ObjWave, qq, del_px):
        #This function will propagate the object wave function to the image plane
        #by convolving with the transfer function of microscope and returns the 
        #complex real-space ImgWave

        #get the transfer function
        tf = self.getTransferFunction(qq, del_px)
        
        #Compute Fourier transform of ObjWave and convolve with tf
        f_ObjWave = np.fft.fftshift(np.fft.fftn(ObjWave))
        f_ImgWave = f_ObjWave * np.conj(tf)
        ImgWave = np.fft.ifftn(np.fft.ifftshift(f_ImgWave))

        return ImgWave
    
    def getImage(self, ObjWave, qq, del_px):
        #This function will produce the image at the set defocus using the 
        #methods in this class.

        #Get the Propagated wave function
        ImgWave = self.PropagateWave(ObjWave, qq, del_px)
        Image = np.abs(ImgWave)**2

        return Image


# Plot phase gradient

def Plot_ColorMap(Bx = np.random.rand(256,256), By = np.random.rand(256,256), \
                  hsvwheel = False, filename = 'Vector_ColorMap.jpeg'):
    # first get the size of the input data
    [dimx,dimy] = Bx.shape
    #inset colorwheel size - 100 px
    csize = 100
    #co-ordinate arrays for colorwheel.
    line = np.arange(csize) - float(csize/2)
    [X,Y] = np.meshgrid(line,line,indexing = 'xy')
    th = np.arctan2(Y,X)
    h_col = (th + np.pi)/2/np.pi
    rr = np.sqrt(X**2 + Y**2)
    msk = np.zeros(rr.shape)
    msk[np.where(rr <= csize/2)] = 1.0
    rr *= msk
    rr /= np.amax(rr)
    val_col = np.ones(rr.shape) * msk
    

    #Compute the maximum in magnitude BB = sqrt(Bx^2 + By^2)
    mmax = np.amax(np.sqrt(Bx**2 + By**2))
    # Normalize with respect to max.
    Bx /= float(mmax)
    By /= float(mmax)
    #Compute the magnitude and scale between 0 and 1
    Bmag = np.sqrt(Bx**2 + By**2)
    
    if hsvwheel:
        # Here we will proceed with using the standard HSV colorwheel routine.
        # Get the Hue (angle) as By/Bx and scale between [0,1]
        hue = (np.arctan2(By,Bx) + np.pi)/2/np.pi
        # Array to hold the colorimage.
        color_im = np.zeros([dimx, dimy, 3])
        #First the Hue.
        color_im[0:dimx,0:dimy,0] = hue
        # Then the Sat.
        color_im[0:dimx,0:dimy,1] = Bmag
        # Then the Val.
        color_im[0:dimx,0:dimy,2] = np.ones([dimx,dimy])
        # Convert to RGB image.
        rgb_image = mt_cols.hsv_to_rgb(color_im)
    else:
        #Here we proceed with custom RGB colorwheel.
        #Arrays for each RGB channel
        red = np.zeros([dimx,dimy])
        gr = np.zeros([dimx,dimy])
        blue = np.zeros([dimx,dimy])
    
        #Scale the magnitude between 0 and 255
        cmag = Bmag #* 255.0
        #Compute the cosine of the angle
        cang =  Bx / cmag
        #Compute the sine of the angle
        sang = np.sqrt(1.0 - cang**2)
        #first the green component
        qq = np.where((Bx < 0.0) & (By >= 0.0))
        gr[qq] = cmag[qq] * np.abs(cang[qq])
        qq = np.where((Bx >= 0.0) & (By < 0.0))
        gr[qq] = cmag[qq] * np.abs(sang[qq])
        qq = np.where((Bx < 0.0) & (By < 0.0))
        gr[qq] = cmag[qq]
        # then the red
        qq = np.where((Bx >= 0.0) & (By < 0.0))
        red[qq] = cmag[qq]
        qq = np.where((Bx >=0.0) & (By >= 0.0))
        red[qq] = cmag[qq] * np.abs(cang[qq])
        qq = np.where((Bx < 0.0) & (By < 0.0))
        red[qq] = cmag[qq] * np.abs(sang[qq])
        # then the blue
        qq = np.where(By >= 0.0)
        blue[qq] = cmag[qq] * np.abs(sang[qq])
        # Store the color components in the RGB image
        rgb_image = np.zeros([dimx+csize,dimy,3])
        rgb_image[0:dimx,0:dimy,0] = red
        rgb_image[0:dimx,0:dimy,1] = gr
        rgb_image[0:dimx,0:dimy,2] = blue
    
        #Recompute cmag, cang, sang for the colorwheel representation.
        mmax = np.amax([np.abs(X),np.abs(Y)])
        X /= mmax
        Y /= mmax
        cmag = np.sqrt(X**2 + Y**2) #* 255.0
        cang =  X / cmag
        sang = np.sqrt(1.0 - cang**2)
        # Arrays for colorwheel sizes
        red = np.zeros([csize,csize])
        gr = np.zeros([csize,csize])
        blue = np.zeros([csize,csize])
        #first the green component
        qq = np.where((X < 0.0) & (Y >= 0.0))
        gr[qq] = cmag[qq] * np.abs(cang[qq])
        qq = np.where((X >= 0.0) & (Y < 0.0))
        gr[qq] = cmag[qq] * np.abs(sang[qq])
        qq = np.where((X < 0.0) & (Y < 0.0))
        gr[qq] = cmag[qq]
        # then the red
        qq = np.where((X >= 0.0) & (Y < 0.0))
        red[qq] = cmag[qq]
        qq = np.where((X >=0.0) & (Y >= 0.0))
        red[qq] = cmag[qq] * np.abs(cang[qq])
        qq = np.where((X < 0.0) & (Y < 0.0))
        red[qq] = cmag[qq] * np.abs(sang[qq])
        # then the blue
        qq = np.where(Y >= 0.0)
        blue[qq] = cmag[qq] * np.abs(sang[qq])

        #Store in the colorimage
        rgb_image[dimx:,dimy/2-csize/2:dimy/2+csize/2,0] = red * msk
        rgb_image[dimx:,dimy/2-csize/2:dimy/2+csize/2,1] = gr * msk
        rgb_image[dimx:,dimy/2-csize/2:dimy/2+csize/2,2] = blue * msk

    # Now we have the RGB image. Save it and then return it.
    # skimage_io.imsave(filename,rgb_image)

    return rgb_image


