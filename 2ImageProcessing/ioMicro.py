import numpy as np
import cv2
def tile_registration(fls, ch_dapi, tileidx, overlap=10):
    """
    Input:
    fls - list of file names
    tileidx - list of tile xy id e.g. 16 elements from [0,0] to [3,3]
    overlap - percentile of overlap from settings
    Estimation based on pairwise registration of horizontal/vertical neighboring tiles
    """
    import skimage
    import tifffile
    im = skimage.io.imread(fls[1])
    sizes = im.shape
    dimXY = sizes[1]
    dimZ  = sizes[0]
    pairs = list()
    offsets = list()
    # Estimate between neighboring tiles
    for i1 in range(len(tileidx)): # for each x position
        for i2 in range(len(tileidx)): #for each y position
            if (tileidx[i2, 0] - tileidx[i1, 0] == 1 and tileidx[i1, 1] == tileidx[i2, 1]): #if the tiles are next 
                print("Registering between (" + str(tileidx[i1, 0])+ ", " + str(tileidx[i1, 1]) + ") and (" + str(tileidx[i2, 0])+ ", "+ str(tileidx[i2, 1]) +")")
                # neighboring tiles horizontal
                im1 = tifffile.imread(fls[i1])
                im2 = tifffile.imread(fls[i2])
                pairs.append([i1, i2])
                im1_ = im1[:, dimXY-round(dimXY*overlap/100):dimXY]
                im2_ = im2[:, 0:round(dimXY*overlap/100)]
                txyz, txyzs = get_txyz(im1_,im2_,sz_norm=20,sz = 300,nelems=5,plt_val=False)
                txyz[1] = txyz[1] - round(dimXY*(1-overlap/100))
                print(txyz)
                offsets.append(txyz)
            elif (tileidx[i2, 1] - tileidx[i1, 1] == 1 and tileidx[i1, 0] == tileidx[i2, 0]):
                print("Registering between (" + str(tileidx[i1, 0])+ ", " + str(tileidx[i1, 1]) + ") and (" + str(tileidx[i2, 0])+ ", " + str(tileidx[i2, 1]) +")")
                # neighboring tiles vertical
                im1 = tifffile.imread(fls[i1])
                im2 = tifffile.imread(fls[i2])
                pairs.append([i1, i2])
                im1_ = im1[:, :, dimXY-round(dimXY*overlap/100):dimXY]
                im2_ = im2[:, :, 0:round(dimXY*overlap/100)]
                txyz, txyzs = get_txyz(im1_,im2_,sz_norm=20,sz = 300,nelems=5,plt_val=False)
                txyz[2] = txyz[2] - round(dimXY*(1-overlap/100))   
                print(txyz)
                offsets.append(txyz)
    return pairs, offsets
def norm_perc(img,percm=1,percM=99.9):
    p1,p99 = np.percentile(img,percm),np.percentile(img,percM)
    return np.array(np.clip((img-p1)/(p99-p1),0,1),dtype=np.float32)
def norm_zstack(imT,gb_sm=5,gb_big=200,resc=4):
    from tqdm import tqdm
    return np.array([(cv2.blur(im_,(gb_sm,gb_sm))/cv2.blur(im_,(gb_big,gb_big)))[::resc,::resc] for im_ in tqdm(imT)])
def resize(im,shape_ = [50,2048,2048]):
    """Given an 3d image <im> this provides a quick way to resize based on nneighbor sampling"""
    z_int = np.round(np.linspace(0,im.shape[0]-1,shape_[0])).astype(int)
    x_int = np.round(np.linspace(0,im.shape[1]-1,shape_[1])).astype(int)
    y_int = np.round(np.linspace(0,im.shape[2]-1,shape_[2])).astype(int)
    return im[z_int][:,x_int][:,:,y_int]



def norm_slice(im,s=50):
    import cv2
    import numpy as np
    im_=im.astype(np.float32)
    return np.array([im__-cv2.blur(im__,(s,s)) for im__ in im_],dtype=np.float32)

def norm_slice_new(im,s1=50,s2=300):
    import cv2
    import numpy as np
    im_32=im.astype(np.float32)
    im_ = np.array([cv2.divide(zplane,cv2.blur(zplane,(s2,s2))) for zplane in im_32],dtype=np.float32)
    im_norm = np.array([im__-cv2.blur(im__,(s1,s1)) for im__ in im_],dtype=np.float32)
    im_norm = im_norm*(np.max(im_32)/np.max(im_norm))
    return np.array(im_norm,dtype=np.float32)

def get_new_ims(im_dapi0_,im_dapi1_,txyz,ib=0,sz=100):
    dic_ims0 = get_tiles(im_dapi0_,size=sz)
    keys = list(dic_ims0.keys())
    best = np.argsort([np.std(dic_ims0[key]) for key in keys])[::-1]
    start = np.array(keys[best[ib]])*sz
    szs = im_dapi0_.shape
    szf = np.array(im_dapi0_.shape)
    szs = np.min([szs,[sz]*3],axis=0)
    start1,end1 = start+txyz,start+szs+txyz
    start2,end2 = start,start+szs
    start2[start1<0]-=start1[start1<0]
    start1[start1<0]=0
    end2[end1>szf]-=end1[end1>szf]-szf[end1>szf]
    end1[end1>szf]=szf[end1>szf]
    im1=im_dapi1_[start1[0]:end1[0],start1[1]:end1[1],start1[2]:end1[2]]
    im0=im_dapi0_[start2[0]:end2[0],start2[1]:end2[1],start2[2]:end2[2]]
    return im0,im1

def get_txyz(im_dapi0,im_dapi1,sz_norm=20,sz = 200,nelems=5,plt_val=False):
    """
    Given two 3D images im_dapi0,im_dapi1, this normalizes them by subtracting local background (gaussian size sz_norm)
    and then computes correlations on <nelemes> blocks with highest  std of signal of size sz
    It will return median value and a list of single values.
    """
    import numpy as np
    im_dapi0_ = norm_slice(im_dapi0,sz_norm)
    im_dapi1_ = norm_slice(im_dapi1,sz_norm)
    dic_ims0 = get_tiles(im_dapi0_,size=sz)
    dic_ims1 = get_tiles(im_dapi1_,size=sz)
    keys = list(dic_ims0.keys())
    best = np.argsort([np.std(dic_ims0[key]) for key in keys])[::-1]
    txyzs = []
    im_cors = []
    for ib in range(min(nelems,len(best))):
        im0 = dic_ims0[keys[best[ib]]][0].copy()
        im1 = dic_ims1[keys[best[ib]]][0].copy()
        im0-=np.mean(im0)
        im1-=np.mean(im1)
        from scipy.signal import fftconvolve
        im_cor = fftconvolve(im0[::-1,::-1,::-1],im1, mode='full')
        txyz = np.unravel_index(np.argmax(im_cor), im_cor.shape)-np.array(im0.shape)+1
        if plt_val:
            plt.figure()
            plt.imshow(np.max(im_cor,0))
            print(txyz)      
        im_cors.append(im_cor)
        txyzs.append(txyz)
    txyz = np.median(txyzs,0).astype(int)
    return txyz,txyzs
def imshow_color(im1, im2, vmin=50, vmax=200, axis=0):
    imf = np.concatenate([np.clip(im1[...,np.newaxis], vmin, vmax),
                          np.clip(im2[...,np.newaxis], vmin, vmax),
                          np.clip(im1[...,np.newaxis], vmin, vmax)],axis=-1)

    tifffile.imshow(np.max(imf,axis=axis),cmap='gray')

def get_tiles(im_3d,size=256):
    import numpy as np
    sz,sx,sy = im_3d.shape
    Mz = int(np.ceil(sz/float(size)))
    Mx = int(np.ceil(sx/float(size)))
    My = int(np.ceil(sy/float(size)))
    ims_dic = {}
    for iz in range(Mz):
        for ix in range(Mx):
            for iy in range(My):
                ims_dic[(iz,ix,iy)]=ims_dic.get((iz,ix,iy),[])+[im_3d[iz*size:(iz+1)*size,ix*size:(ix+1)*size,iy*size:(iy+1)*size]] 
    return ims_dic

def rescale_gaus(im0,resc=4,gauss=False):
    sx,sy,sz = im0.shape
    im0_ = im0[np.arange(0,sx,1./resc).astype(int)]
    im0_ = im0_[:,np.arange(0,sy,1./resc).astype(int)]
    im0_ = im0_[:,:,np.arange(0,sz,1./resc).astype(int)]
    if gauss:
        sigma=resc
        sz0=sigma*4
        X,Y,Z = (np.indices([2*sz0+1]*3)-sz0)
        im_ker = np.exp(-(X*X+Y*Y+Z*Z)/2/sigma**2)

        im0_ = fftconvolve(im0_,im_ker, mode='same')
    return im0_

def convolve_gauss(im0,sigma=2):
    from scipy.signal import fftconvolve
    sz0=sigma*4
    X,Y,Z = (np.indices([2*sz0+1]*3)-sz0)
    im_ker = np.exp(-(X*X+Y*Y+Z*Z)/2/sigma**2)
    im_ker = im_ker/np.sum(im_ker)
    return fftconvolve(im0,im_ker, mode='same')


def apply_drift(im0, txyz):
    import numpy as np
    txyz_ = txyz.copy()
    szs = np.array(im0.shape)
    szf = np.array(im0.shape)
    start1,end1 = txyz_,szs+txyz_
    start2,end2 = np.array([0,0,0]), szs
    start2[start1<0]-=start1[start1<0]
    start1[start1<0]=0
    end2[end1>szf]-=end1[end1>szf]-szs[end1>szf]
    end1[end1>szf]=szf[end1>szf]
    im_ = np.zeros(im0.shape, dtype='uint')
    im_[start2[0]:end2[0],start2[1]:end2[1],start2[2]:end2[2]]=im0[start1[0]:end1[0],start1[1]:end1[1],start1[2]:end1[2]]
    return im_

def expand_dapi_seg(im,npix=5,zinterp=2,xyinterp=4):
    A = im[::zinterp,::xyinterp,::xyinterp].copy()
    A_nonzero = A>0
    from scipy import ndimage as nd
    #A_dil = nd.binary_dilation(A_nonzero,nd.generate_binary_structure(3, 1),iterations=npix)
    A_dil = np.array([nd.binary_dilation(slice_,nd.generate_binary_structure(2, 1),iterations=npix) for slice_ in A_nonzero])
    X = np.array(np.where(A_dil.astype(np.float32)-A_nonzero)).T
    Xincell = np.array(np.where(A_nonzero)).T
    from scipy.spatial import cKDTree as KDTree
    tree = KDTree(Xincell)
    dists,ielems = tree.query(X)
    A[tuple(X.T)] = A[tuple(Xincell[ielems].T)]
    return A
def segment_margin_calc_dist(stitched_im_fl, resize_factor,nuc_positions):
    ## use opencv to segment the embryonic margin and calculate the distance of cells to the margin
    import numpy as np
    import cv2 as cv
    import os,glob

    #ix,iy,sx,sy = -1,-1,-1,-1 <- initialise these outside the function
    #print(ix)
    global ix,iy,sx,sy
    class CoordinateStore:
        def __init__(self):
            self.points = []
        def draw_lines(self,event, x, y, flags, param):
            global ix,iy,sx,sy
            # if the left mouse button was clicked, record the starting
            if event == cv.EVENT_LBUTTONDOWN:
                # draw circle of 2px
                cv.circle(img_, (x, y), 3, (0, 0, 127), -1)
                self.points.append((x,y))
                if ix != -1: # if ix and iy are not first points, then draw a line
                    cv.line(img_, (ix, iy), (x, y), (0, 0, 127), 2, cv.LINE_AA)
                else: # if ix and iy are first points, store as starting points
                    sx, sy = x, y
                ix,iy = x, y
            elif event == cv.EVENT_LBUTTONDBLCLK:
                ix, iy = -1, -1 # reset ix and iy
                if flags == 33: # if alt key is pressed, create line between start and end points to create polygon
                    cv.line(img_, (x, y), (sx, sy), (0, 0, 127), 2, cv.LINE_AA)
    outline = CoordinateStore()

    # read image from path and add callback
    img = cv.imread(stitched_im_fl[0],cv.IMREAD_UNCHANGED)
    img_ = img[::resize_factor,::resize_factor] #resize image to fit screen
    img_ = img_/255 #normalise to values between 0 and 1
    img_[np.where(img_==0)] = 1 #make area not covered by tiles white so segmented area can be found as a black contour
    cv.namedWindow('embryo segmentation')
    cv.setMouseCallback('embryo segmentation',outline.draw_lines)
    while True:
        cv.imshow('embryo segmentation',img_)
        if cv.waitKey(2) & 0xFF == 27: #show until Esc is pressed
            cv.destroyAllWindows()
            outline.points = np.array(outline.points)
            break
    outline_points = outline.points * resize_factor
    mask = np.ones(img.shape, dtype="uint8") * 255
    # Draw the contours on the mask
    cv.drawContours(mask,[outline_points],0,(0,0,0),-1)
    thresh = 100
    ret,thresh_img = cv.threshold(mask, thresh, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, contours, -1, (0,255,0), 10)
    cnt = contours[1]
    dists = []
    pxsize = 0.104 #micro meter pixel size
    for nucleus in nuc_positions[:,[2,1]]:
        #cv.circle(img,(int(nucleus[0]),int(nucleus[1])),30,(0,0,255),-1)
        dist = cv.pointPolygonTest(cnt,nucleus,True)
        dist = dist * pxsize
        dists.append(dist)
    dists = np.array(dists)
    cv.imwrite(r'\\biopz-jumbo.storage.p.unibas.ch\rg-as04$\_Members\elkhol0000\Desktop\trash\test_img.png',img)
    return dists

def define_dorsal(stitched_im_fl, resize_factor,nuc_positions,dists):
    ## use opencv to define the most dorsal position
        ## to change the function to take into account the pacman shape:
        ##      - define start and end of empty area (mouth of pacman) with GUI together with dorsal side (draw line from AP to mouseclick)
        ##      - calculate the slice degree of the mouth and substract it from the position of all cells
        ##      - normalise to set the maximum back to 360
        ##      - deduct values >180 from 360 to get the degrees away from dorsal
    
    import numpy as np
    import cv2 as cv
    import os,glob
    import math
    import copy
    class CoordinateStore:
        def __init__(self):
            self.dorsal = []
            self.pacman_slice = []
        def draw_point(self,event, x, y, flags, param):
            # if the left mouse button was clicked, record the starting
            if event == cv.EVENT_LBUTTONDOWN:
                # draw circle of 2px
                if len(self.dorsal) != 0 and len(self.pacman_slice) <2:
                    cv.line(img_,(int(AP[2]/10),int(AP[1]/10)),(x,y),(0, 0, 127),5)
                    self.pacman_slice.append([x,y]) 
                if len(self.dorsal) == 0:
                    cv.circle(img_, (x, y), 10, (0, 0, 127), -1)
                    self.dorsal = [x,y]    

    dorsal = CoordinateStore()

    # read image from path and add callback
    img = cv.imread(stitched_im_fl[0],cv.IMREAD_UNCHANGED)
    img_ = img[::resize_factor,::resize_factor] #resize image to fit screen
    img_ = img_/255 #normalise to values between 0 and 1
    img_[np.where(img_==0)] = 1 #make area not covered by tiles white so segmented area can be found as a black contour
    
    AP = nuc_positions[np.where(dists == np.max(dists))][0] # position of nucleus with maximal distance from the margin, i.e. the animal pole
    
    cache = copy.deepcopy(img_)
    cv.namedWindow('embryo segmentation')
    cv.setMouseCallback('embryo segmentation',dorsal.draw_point)
    while True:
        cv.imshow('embryo segmentation',img_)
        if cv.waitKey(2) & 0xFF == 27: #show until Esc is pressed
            cv.destroyAllWindows()
            break
            
    DP = np.array(dorsal.dorsal) *resize_factor
    rel_DP = DP - AP[1:3]
    rel_dors = math.degrees(math.atan(rel_DP[1]/rel_DP[0]))
    
    pacman_slice = np.array([pos for pos in dorsal.pacman_slice],dtype='float64') #absolute position of slice borders
    pacman_slice *= resize_factor # resizes absolute position of slice borders
    
    rel_pacman_slice = pacman_slice - AP[1:3] #slice borders relativ to animal pole
    print(rel_pacman_slice)
    slice_xy = rel_pacman_slice[:,1] /rel_pacman_slice[:,0] # y/x ratio of slice border points
    rel_slice = np.array([math.degrees(math.atan(pos)) for pos in slice_xy]) # slice border point positions in degrees
    #rel_slice = [math.atan(pos) for pos in slice_xy] # slice border point positions in degrees
    for posID,pos in enumerate(rel_pacman_slice):
        if pos[0] <0:
            rel_slice[posID] -= 180
        if pos[1] <0:
            rel_slice[posID] += 180
        if pos[1]/pos[0] < 0:
            rel_slice[posID] += 180
        if pos[0] > 0 and pos[1] >0:
            rel_slice[posID] += 180
    
    #q3_q4 = rel_pacman_slice[:,1] < 0 #slice border points which are in the q3 or q4 relative to the animal pole
    #rel_slice[q3_q4] = rel_slice[q3_q4] + 180 #corrected
        
    #for posID,pos in enumerate(rel_slice):
    #    if pos < 90:
    #        rel_slice[posID] = rel_slice[posID] -180
   #     if pos > 180: 
     #       rel_slice[posID] = rel_slice[posID] -180
    
    rel_nuc = nuc_positions[:,1:3] - AP[1:3] + np.amax(nuc_positions[:,1:3],0)/100000 # nuclear position relativ to animal pole
    nucxy = rel_nuc[:,0] /rel_nuc[:,1] # y/x ratio
    rel_nuc_ = np.array([math.degrees(math.atan(nuc)) for nuc in nucxy]) #rel. nuclear position in polar degrees
    q3_q4 = rel_nuc[:,1] < 0 
    rel_nuc_[q3_q4] = rel_nuc_[q3_q4] + 180

    DV_nuc = rel_nuc_ - rel_dors
    DV_slice = rel_slice - rel_dors

        

    

    
    if rel_DP[0] < 0:
        DV_nuc += 180
        
    
    
    
    DV_nuc[DV_nuc<0] += 360
    DV_nuc[DV_nuc>360] -= 360
    DV_slice[DV_slice<0] += 360
    DV_slice[DV_slice>360] -= 360
    
    slice_deg = (np.max(DV_slice) - np.min(DV_slice))
    #print(np.max(rel_slice))
   # #print(np.min(rel_slice))
   # print(np.max(DV_slice))
   # print(np.min(DV_slice))

    in_slice_ind = np.logical_and(DV_nuc>np.min(DV_slice), DV_nuc<np.max(DV_slice))
    after_slice_ind = DV_nuc>np.max(DV_slice)
    DV_nuc[in_slice_ind] = np.min(DV_slice)
    DV_nuc[after_slice_ind] = DV_nuc[after_slice_ind] - slice_deg

    #DV_nuc[DV_nuc>np.max(DV_slice)] -= slice_deg     
    DV_nuc *= (360/(360-slice_deg))

    
    DV_nuc[DV_nuc>180] =  360 - DV_nuc[DV_nuc>180]
                          
    return DP,DV_nuc
def get_local_max(im_dif,th_fit,delta=2,delta_fit=3,dbscan=True,return_centers=False,mins=None):
    import numpy as np
    """Given a 3D image <im_dif> as numpy array, get the local maxima in cube -<delta>_to_<delta> in 3D.
    Optional a dbscan can be used to couple connected pixels with the same local maximum. 
    (This is important if saturating the camera values.)
    Returns: Xh - a list of z,x,y and brightness of the local maxima
    """
    
    z,x,y = np.where(im_dif>th_fit)
    zmax,xmax,ymax = im_dif.shape
    in_im = im_dif[z,x,y]
    keep = np.ones(len(x))>0
    for d1 in range(-delta,delta+1):
        for d2 in range(-delta,delta+1):
            for d3 in range(-delta,delta+1):
                keep &= (in_im>=im_dif[(z+d1)%zmax,(x+d2)%xmax,(y+d3)%ymax])
    z,x,y = z[keep],x[keep],y[keep]
    h = in_im[keep]
    Xh = np.array([z,x,y,h]).T
    if dbscan and len(Xh)>0:
        from scipy import ndimage
        im_keep = np.zeros(im_dif.shape,dtype=bool)
        im_keep[z,x,y]=True
        lbl, nlbl = ndimage.label(im_keep,structure=np.ones([3]*3))
        l=lbl[z,x,y]#labels after reconnection
        ul = np.arange(1,nlbl+1)
        il = np.argsort(l)
        l=l[il]
        z,x,y,h = z[il],x[il],y[il],h[il]
        inds = np.searchsorted(l,ul)
        Xh = np.array([z,x,y,h]).T
        Xh_ = []
        for i_ in range(len(inds)):
            j_=inds[i_+1] if i_<len(inds)-1 else len(Xh)
            Xh_.append(np.mean(Xh[inds[i_]:j_],0))
        Xh=np.array(Xh_)
        z,x,y,h = Xh.T
    im_centers=[]
    if delta_fit!=0 and len(Xh)>0:
        z,x,y,h = Xh.T
        z,x,y = z.astype(int),x.astype(int),y.astype(int)
        im_centers = [[],[],[],[]]
        for d1 in range(-delta_fit,delta_fit+1):
            for d2 in range(-delta_fit,delta_fit+1):
                for d3 in range(-delta_fit,delta_fit+1):
                    if (d1*d1+d2*d2+d3*d3)<=(delta_fit*delta_fit):
                        im_centers[0].append((z+d1))
                        im_centers[1].append((x+d2))
                        im_centers[2].append((y+d3))
                        im_centers[3].append(im_dif[(z+d1)%zmax,(x+d2)%xmax,(y+d3)%ymax])

        im_centers_ = np.array(im_centers)
        im_centers_[-1] -= np.min(im_centers_[-1],axis=0)
        zc = np.sum(im_centers_[0]*im_centers_[-1],axis=0)/np.sum(im_centers_[-1],axis=0)
        xc = np.sum(im_centers_[1]*im_centers_[-1],axis=0)/np.sum(im_centers_[-1],axis=0)
        yc = np.sum(im_centers_[2]*im_centers_[-1],axis=0)/np.sum(im_centers_[-1],axis=0)
        Xh = np.array([zc,xc,yc,h]).T
    if return_centers:
        return Xh,np.array(im_centers)
    return Xh   



#import napari
import torch
import numpy as np,pickle,glob,os
import cv2
from scipy.signal import convolve,fftconvolve
from tqdm import tqdm
import matplotlib.pylab as plt
from scipy.spatial import cKDTree

def get_p99(fl_dapi,resc=4):
    im = read_im(fl_dapi)
    im_ = np.array(im[-1][im.shape[1]//2],dtype=np.float32)
    img = (cv2.blur(im_,(2,2))-cv2.blur(im_,(50,50)))[::resc,::resc]
    p99 = np.percentile(img,99.9)
    p1 = np.percentile(img,1)
    img = np.array(np.clip((img-p1)/(p99-p1),0,1),dtype=np.float32)
    plt.figure()
    plt.imshow(img,cmap='gray')
    return p99
def resize(im,shape_ = [50,2048,2048]):
    """Given an 3d image <im> this provides a quick way to resize based on nneighbor sampling"""
    z_int = np.round(np.linspace(0,im.shape[0]-1,shape_[0])).astype(int)
    x_int = np.round(np.linspace(0,im.shape[1]-1,shape_[1])).astype(int)
    y_int = np.round(np.linspace(0,im.shape[2]-1,shape_[2])).astype(int)
    return im[z_int][:,x_int][:,:,y_int]

import scipy.ndimage as ndimage
def get_final_cells_cyto(im_polyA,final_cells,icells_keep=None,ires = 4,iresf=10,dist_cutoff=10):
    """Given a 3D im_polyA signal and a segmentation fie final_cells """
    incell = final_cells>0
    med_polyA = np.median(im_polyA[incell])
    med_nonpolyA = np.median(im_polyA[~incell])
    im_ext_cells = im_polyA>(med_polyA+med_nonpolyA)/2


    X = np.array(np.where(im_ext_cells[:,::ires,::ires])).T
    Xcells = np.array(np.where(final_cells[:,::ires,::ires]>0)).T
    from sklearn.neighbors import KDTree

    kdt = KDTree(Xcells[::iresf], leaf_size=30, metric='euclidean')
    icells_neigh = final_cells[:,::ires,::ires][Xcells[::iresf,0],Xcells[::iresf,1],Xcells[::iresf,2]]
    dist,neighs = kdt.query(X, k=1, return_distance=True)
    dist,neighs = np.squeeze(dist),np.squeeze(neighs)

    final_cells_cyto = im_ext_cells[:,::ires,::ires]*0
    if icells_keep is not None:
        keep_cyto = (dist<dist_cutoff)&np.in1d(icells_neigh[neighs],icells_keep)
    else:
        keep_cyto = (dist<dist_cutoff)
    final_cells_cyto[X[keep_cyto,0],X[keep_cyto,1],X[keep_cyto,2]] = icells_neigh[neighs[keep_cyto]]
    final_cells_cyto = resize(final_cells_cyto,im_polyA.shape)
    return final_cells_cyto
def slice_pair_to_info(pair):
    sl1,sl2 = pair
    xm,ym,sx,sy = sl2.start,sl1.start,sl2.stop-sl2.start,sl1.stop-sl1.start
    A = sx*sy
    return [xm,ym,sx,sy,A]
def get_coords(imlab1,infos1,cell1):
    xm,ym,sx,sy,A,icl = infos1[cell1-1]
    return np.array(np.where(imlab1[ym:ym+sy,xm:xm+sx]==icl)).T+[ym,xm]
def cells_to_coords(imlab1,return_labs=False):
    """return the coordinates of cells with some additional info"""
    infos1 = [slice_pair_to_info(pair)+[icell+1] for icell,pair in enumerate(ndimage.find_objects(imlab1))
    if pair is not None]
    cms1 = np.array([np.mean(get_coords(imlab1,infos1,cl+1),0) for cl in range(len(infos1))])
    if len(cms1)==0: cms1 = []
    else: cms1 = cms1[:,::-1]
    ies = [info[-1] for info in infos1]
    if return_labs:
        return imlab1.copy(),infos1,cms1,ies
    return imlab1.copy(),infos1,cms1
def resplit(cells1,cells2,nmin=100):
    """intermediate function used by standard_segmentation.
    Decide when comparing two planes which cells to split"""
    imlab1,infos1,cms1 = cells_to_coords(cells1)
    imlab2,infos2,cms2 = cells_to_coords(cells2)
    if len(cms1)==0 or len(cms2)==0:
        return imlab1,imlab2,[],0
    #find centers 2 within the cells1 and split cells1
    cms2_ = np.round(cms2).astype(int)
    cells2_1 = imlab1[cms2_[:,1],cms2_[:,0]]
    imlab1_cells = [0]+[info[-1] for info in infos1]
    cells2_1 = [imlab1_cells.index(cl_) for cl_ in cells2_1]#reorder coords
    #[for e1,e2 in zip(np.unique(cells2_1,return_counts=True)) if e1>0]
    dic_cell2_1={}
    for cell1,cell2 in enumerate(cells2_1):
        dic_cell2_1[cell2] = dic_cell2_1.get(cell2,[])+[cell1+1]
    dic_cell2_1_split = {cell:dic_cell2_1[cell] for cell in dic_cell2_1 if len(dic_cell2_1[cell])>1 and cell>0}
    cells1_split = list(dic_cell2_1_split.keys())
    imlab1_cp = imlab1.copy()
    number_of_cells_to_split = len(cells1_split)
    for cell1_split in cells1_split:
        count = np.max(imlab1_cp)+1
        cells2_to1 = dic_cell2_1_split[cell1_split]
        X1 = get_coords(imlab1,infos1,cell1_split)
        X2s = [get_coords(imlab2,infos2,cell2) for cell2 in cells2_to1]
        from scipy.spatial.distance import cdist
        X1_K = np.argmin([np.min(cdist(X1,X2),axis=-1) for X2 in X2s],0)

        for k in range(len(X2s)):
            X_ = X1[X1_K==k]
            if len(X_)>nmin:
                imlab1_cp[X_[:,0],X_[:,1]]=count+k
            else:
                #number_of_cells_to_split-=1
                pass
    imlab1_,infos1_,cms1_ = cells_to_coords(imlab1_cp)
    return imlab1_,infos1_,cms1_,number_of_cells_to_split

def converge(cells1,cells2):
    imlab1,infos1,cms1,labs1 = cells_to_coords(cells1,return_labs=True)
    imlab2,infos2,cms2 = cells_to_coords(cells2)
    
    if len(cms1)==0 or len(cms2)==0:
        return imlab1,imlab2
    #find centers 2 within the cells1 and split cells1
    cms2_ = np.round(cms2).astype(int)
    cells2_1 = imlab1[cms2_[:,1],cms2_[:,0]]
    imlab1_cells = [0]+[info[-1] for info in infos1]
    cells2_1 = [imlab1_cells.index(cl_) for cl_ in cells2_1]#reorder coords
    #[for e1,e2 in zip(np.unique(cells2_1,return_counts=True)) if e1>0]
    dic_cell2_1={}
    for cell1,cell2 in enumerate(cells2_1):
        dic_cell2_1[cell2] = dic_cell2_1.get(cell2,[])+[cell1+1]
        
    dic_cell2_1_match = {cell:dic_cell2_1[cell] for cell in dic_cell2_1 if cell>0}
    cells2_kp = [e_ for e in dic_cell2_1_match for e_ in dic_cell2_1_match[e]]
    modify_cells2 = np.setdiff1d(np.arange(len(cms2)),cells2_kp)
    imlab2_ = imlab2*0
    for cell1 in dic_cell2_1_match:
        for cell2 in dic_cell2_1_match[cell1]:
            xm,ym,sx,sy,A,icl = infos2[cell2-1]
            im_sm = imlab2[ym:ym+sy,xm:xm+sx]
            imlab2_[ym:ym+sy,xm:xm+sx][im_sm==icl]=labs1[cell1-1]
    count_cell = max(np.max(imlab2_),np.max(labs1))
    for cell2 in modify_cells2:
        count_cell+=1
        xm,ym,sx,sy,A,icl = infos2[cell2-1]
        im_sm = imlab2[ym:ym+sy,xm:xm+sx]
        imlab2_[ym:ym+sy,xm:xm+sx][im_sm==icl]=count_cell
    return imlab1,imlab2_
def final_segmentation(fl_dapi,
                        analysis_folder=r'X:\DCBB_human__11_18_2022_Analysis',
                        plt_val=True,
                        rescz = 4,trimz=2, resc=4,p99=None):
    segm_folder = analysis_folder+os.sep+'Segmentation'
    if not os.path.exists(segm_folder): os.makedirs(segm_folder)
    
    save_fl  = segm_folder+os.sep+os.path.basename(fl_dapi).split('.')[0]+'--'+os.path.basename(os.path.dirname(fl_dapi))+'--dapi_segm.npz'
    
    if not os.path.exists(save_fl):
        im = read_im(fl_dapi)
        #im_mid_dapi = np.array(im[-1][im.shape[1]//2],dtype=np.float32)
        im_dapi = im[-1,::rescz][trimz:-trimz]
        
        im_seg_2 = standard_segmentation(im_dapi,resc=resc,sz_min_2d=100,sz_cell=20,use_gpu=True,model='cyto2',p99=p99)
        shape = np.array(im[-1].shape)
        np.savez_compressed(save_fl,segm = im_seg_2,shape = shape)

        

    if plt_val:
        fl_png = save_fl.replace('.npz','__segim.png')
        if not os.path.exists(fl_png):
            im = read_im(fl_dapi)
            im_seg_2 = np.load(save_fl)['segm']
            shape =  np.load(save_fl)['shape']
            
            im_dapi_sm = resize(im[-1],im_seg_2.shape)
            img = np.array(im_dapi_sm[im_dapi_sm.shape[0]//2],dtype=np.float32)
            masks_ = im_seg_2[im_seg_2.shape[0]//2]
            from cellpose import utils
            outlines = utils.masks_to_outlines(masks_)
            p1,p99 = np.percentile(img,1),np.percentile(img,99.9)
            img = np.array(np.clip((img-p1)/(p99-p1),0,1),dtype=np.float32)
            outX, outY = np.nonzero(outlines)
            imgout= np.dstack([img]*3)
            imgout[outX, outY] = np.array([1,0,0]) # pure red
            fig = plt.figure(figsize=(20,20))
            plt.imshow(imgout)
            
            fig.savefig(fl_png)
            plt.close('all')
            print("Saved file:"+fl_png)
def standard_segmentation(im_dapi,resc=2,sz_min_2d=400,sz_cell=25,use_gpu=True,model='cyto2',p99=None):
    """Using cellpose with nuclei mode"""
    from cellpose import models, io,utils
    model = models.Cellpose(gpu=use_gpu, model_type=model)
    #decided that resampling to the 4-2-2 will make it faster
    #im_dapi_3d = im_dapi[::rescz,::resc,::resc].astype(np.float32)
    chan = [0,0]
    masks_all = []
    flows_all = []
    from tqdm import tqdm
    for im in tqdm(im_dapi):
        im_ = np.array(im,dtype=np.float32)
        img = (cv2.blur(im_,(2,2))-cv2.blur(im_,(50,50)))[::resc,::resc]
        p1 = np.percentile(img,1)
        if p99 is None:
            p99 = np.percentile(img,99.9)
        img = np.array(np.clip((img-p1)/(p99-p1),0,1),dtype=np.float32)
        masks, flows, styles, diams = model.eval(img, diameter=sz_cell, channels=chan,
                                             flow_threshold=10,cellprob_threshold=-20,min_size=50,normalize=False)
        masks_all.append(utils.fill_holes_and_remove_small_masks(masks,min_size=sz_min_2d))#,hole_size=3
        flows_all.append(flows[0])
    masks_all = np.array(masks_all)

    sec_half = list(np.arange(int(len(masks_all)/2),len(masks_all)-1))
    first_half = list(np.arange(0,int(len(masks_all)/2)))[::-1]
    indexes = first_half+sec_half
    masks_all_cp = masks_all.copy()
    max_split = 1
    niter = 0
    while max_split>0 and niter<2:
        max_split = 0
        for index in tqdm(indexes):
            cells1,cells2 = masks_all_cp[index],masks_all_cp[index+1]
            imlab1_,infos1_,cms1_,no1 = resplit(cells1,cells2)
            imlab2_,infos2_,cms2_,no2 = resplit(cells2,cells1)
            masks_all_cp[index],masks_all_cp[index+1] = imlab1_,imlab2_
            max_split += max(no1,no2)
            #print(no1,no2)
        niter+=1
    masks_all_cpf = masks_all_cp.copy()
    for index in tqdm(range(len(masks_all_cpf)-1)):
        cells1,cells2 = masks_all_cpf[index],masks_all_cpf[index+1]
        cells1_,cells2_ = converge(cells1,cells2)
        masks_all_cpf[index+1]=cells2_
    return masks_all_cpf

def get_dif_or_ratio(im_sig__,im_bk__,sx=20,sy=20,pad=5,col_align=-2):
    size_ = im_sig__.shape
    imf = np.ones(size_,dtype=np.float32)
    #resc=5
    #ratios = [np.percentile(im_,99.95)for im_ in im_sig__[:,::resc,::resc,::resc]/im_bk__[:,::resc,::resc,::resc]]
    for startx in tqdm(np.arange(0,size_[2],sx)[:]):
        for starty in np.arange(0,size_[3],sy)[:]:
            startx_ = startx-pad
            startx__ = startx_ if startx_>0 else 0
            endx_ = startx+sx+pad
            endx__ = endx_ if endx_<size_[2] else size_[2]-1

            starty_ = starty-pad
            starty__ = starty_ if starty_>0 else 0
            endy_ = starty+sy+pad
            endy__ = endy_ if endy_<size_[3] else size_[3]-1

            padx_end = pad+endx_-endx__
            pady_end = pad+endy_-endy__
            padx_st = pad+startx_-startx__
            pady_st = pad+starty_-starty__

            ims___ = im_sig__[:,:,startx__:endx__,starty__:endy__]
            imb___ = im_bk__[:,:,startx__:endx__,starty__:endy__]

            txy = get_txy_small(np.max(imb___[col_align],axis=0),np.max(ims___[col_align],axis=0),sz_norm=5,delta=3,plt_val=False)
            tzy = get_txy_small(np.max(imb___[col_align],axis=1),np.max(ims___[col_align],axis=1),sz_norm=5,delta=3,plt_val=False)
            txyz = np.array([tzy[0]]+list(txy))
            #print(txyz)
            from scipy import ndimage
            for icol in range(len(imf)):
                imBT = ndimage.shift(imb___[icol],txyz,mode='nearest',order=0)
                im_rat = ims___[icol]/imBT
                #im_rat = ims___[icol]-imBT*ratios[icol]
                im_rat = im_rat[:,padx_st:-padx_end,pady_st:-pady_end]

                imf[icol,:,startx__+padx_st:endx__-padx_end,starty__+pady_st:endy__-pady_end]=im_rat
                if False:
                    plt.figure()
                    plt.imshow(np.max((im_rat),0))
                    plt.figure()
                    plt.imshow(np.max((imb___[icol,:,pad:-pad,pad:-pad]),0))
    return imf

def get_txy_small(im0_,im1_,sz_norm=10,delta=3,plt_val=False):
    im0 = np.array(im0_,dtype=np.float32)
    im1 = np.array(im1_,dtype=np.float32)
    if sz_norm>0:
        im0 -= cv2.blur(im0,(sz_norm,sz_norm))
        im1 -= cv2.blur(im1,(sz_norm,sz_norm))
    im0-=np.mean(im0)
    im1-=np.mean(im1)
    im_cor = convolve(im0[::-1,::-1],im1[delta:-delta,delta:-delta], mode='valid')
    #print(im_cor.shape)
    if plt_val:
        plt.figure()
        plt.imshow(im_cor)
    txy = np.array(np.unravel_index(np.argmax(im_cor), im_cor.shape))-delta
    return txy
def resize_slice(slices,shape0,shapef,fullz=True):
    slices_ = []
    for sl,sm,sM in zip(slices,shape0,shapef):
        start = sl.start*sM//sm
        end = sl.stop*sM//sm
        slices_.append(slice(start,end))
    if fullz:
        slices_[0]=slice(0,shapef[0])
    return tuple(slices_)

    
def get_txyz_small(im0_,im1_,sz_norm=10,delta=3,plt_val=False):
    im0 = np.array(im0_,dtype=np.float32)
    im1 = np.array(im1_,dtype=np.float32)
    im0 = norm_slice(im0,sz_norm)
    im1 = norm_slice(im1,sz_norm)
    im0-=np.mean(im0)
    im1-=np.mean(im1)
    from scipy.signal import fftconvolve
    im_cor = fftconvolve(im0[::-1,::-1,::-1],im1, mode='full')
    if plt_val:
        plt.figure()
        plt.imshow(np.max(im_cor,0))
        #print(txyz)
    min_ = np.array(im0.shape)-delta
    min_[min_<0]=0
    max_ = np.array(im0.shape)+delta+1
    im_cor-=np.min(im_cor)
    im_cor[tuple([slice(m,M,None)for m,M in zip(min_,max_)])]*=-1
    txyz = np.unravel_index(np.argmin(im_cor), im_cor.shape)-np.array(im0.shape)+1
    #txyz = np.unravel_index(np.argmax(im_cor_),im_cor_.shape)+delta_
    return txyz

def get_txyz_small(im0_,im1_,sz_norm=10,plt_val=False,return_cor=False):
    im0 = np.array(im0_,dtype=np.float32)
    im1 = np.array(im1_,dtype=np.float32)
    if sz_norm>0:
        im0 = norm_slice(im0,sz_norm)
        im1 = norm_slice(im1,sz_norm)
    im0-=np.mean(im0)
    im0/=np.std(im0)
    im1-=np.mean(im1)
    im1/=np.std(im1)
    from scipy.signal import fftconvolve
    im_cor = fftconvolve(im0[::-1,::-1,::-1],im1, mode='full')
    
        #print(txyz)
    imax = np.unravel_index(np.argmax(im_cor), im_cor.shape)
    cor = im_cor[tuple(imax)]/np.prod(im0.shape)
    txyz = imax-np.array(im0.shape)+1
    if plt_val:
        import matplotlib.pylab as plt
        plt.figure()
        plt.imshow(np.max(im_cor,0))
        import napari
        from scipy.ndimage import shift
        viewer = napari.view_image(im0)
        viewer.add_image(shift(im1,-txyz,mode='nearest'))
    
    if return_cor:
        return txyz,cor
    return txyz


def get_local_max(im_dif,th_fit,im_raw=None,dic_psf=None,delta=1,delta_fit=3,dbscan=True,return_centers=False,mins=None):
    """Given a 3D image <im_dif> as numpy array, get the local maxima in cube -<delta>_to_<delta> in 3D.
    Optional a dbscan can be used to couple connected pixels with the same local maximum. 
    (This is important if saturating the camera values.)
    Returns: Xh - a list of z,x,y and brightness of the local maxima
    """
    
    z,x,y = np.where(im_dif>th_fit)
    zmax,xmax,ymax = im_dif.shape
    in_im = im_dif[z,x,y]
    keep = np.ones(len(x))>0
    for d1 in range(-delta,delta+1):
        for d2 in range(-delta,delta+1):
            for d3 in range(-delta,delta+1):
                keep &= (in_im>=im_dif[(z+d1)%zmax,(x+d2)%xmax,(y+d3)%ymax])
    z,x,y = z[keep],x[keep],y[keep]
    h = in_im[keep]
    Xh = np.array([z,x,y,h]).T
    if dbscan and len(Xh)>0:
        from scipy import ndimage
        im_keep = np.zeros(im_dif.shape,dtype=bool)
        im_keep[z,x,y]=True
        lbl, nlbl = ndimage.label(im_keep,structure=np.ones([3]*3))
        l=lbl[z,x,y]#labels after reconnection
        ul = np.arange(1,nlbl+1)
        il = np.argsort(l)
        l=l[il]
        z,x,y,h = z[il],x[il],y[il],h[il]
        inds = np.searchsorted(l,ul)
        Xh = np.array([z,x,y,h]).T
        Xh_ = []
        for i_ in range(len(inds)):
            j_=inds[i_+1] if i_<len(inds)-1 else len(Xh)
            Xh_.append(np.mean(Xh[inds[i_]:j_],0))
        Xh=np.array(Xh_)
        z,x,y,h = Xh.T
    im_centers=[]
    if delta_fit!=0 and len(Xh)>0:
        z,x,y,h = Xh.T
        z,x,y = z.astype(int),x.astype(int),y.astype(int)
        im_centers = [[],[],[],[],[]]
        Xft = []
        
        for d1 in range(-delta_fit,delta_fit+1):
            for d2 in range(-delta_fit,delta_fit+1):
                for d3 in range(-delta_fit,delta_fit+1):
                    if (d1*d1+d2*d2+d3*d3)<=(delta_fit*delta_fit):
                        im_centers[0].append((z+d1))
                        im_centers[1].append((x+d2))
                        im_centers[2].append((y+d3))
                        im_centers[3].append(im_dif[(z+d1)%zmax,(x+d2)%xmax,(y+d3)%ymax])
                        if im_raw is not None:
                            im_centers[4].append(im_raw[(z+d1)%zmax,(x+d2)%xmax,(y+d3)%ymax])
                        Xft.append([d1,d2,d3])
              
        Xft = np.array(Xft)
        im_centers_ = np.array(im_centers)
        bk = np.min(im_centers_[3],axis=0)
        im_centers_[3] -= bk
        a = np.sum(im_centers_[3],axis=0)
        habs = np.zeros_like(bk)
        if im_raw is not None:
            habs = im_raw[z%zmax,x%xmax,y%ymax]
          
        if dic_psf is not None:
            keys = list(dic_psf.keys())
            ### calculate spacing
            im0 = dic_psf[keys[0]]
            space = np.sort(np.diff(keys,axis=0).ravel())
            space = space[space!=0][0]
            ### convert to reduced space
            zi,xi,yi = (z/space).astype(int),(x/space).astype(int),(y/space).astype(int)

            keys_ =  np.array(keys)
            sz_ = list(np.max(keys_//space,axis=0)+1)

            ind_ = tuple(Xft.T+np.array(im0.shape)[:,np.newaxis]//2-1)

            im_psf = np.zeros(sz_+[len(ind_[0])])
            for key in keys_:
                coord = tuple((key/space).astype(int))
                im__ = dic_psf[tuple(key)][ind_]
                im_psf[coord]=(im__-np.mean(im__))/np.std(im__)
            im_psf_ = im_psf[zi,xi,yi]
            im_centers__ = im_centers_[3].T.copy()
            im_centers__ = (im_centers__-np.mean(im_centers__,axis=-1)[:,np.newaxis])/np.std(im_centers__,axis=-1)[:,np.newaxis]
            hn = np.mean(im_centers__*im_psf_,axis=-1)
        else:
            sigma = 1
            norm_G = np.exp(-np.sum(Xft*Xft,axis=-1)/2./sigma/sigma)
            norm_G = norm_G/np.sum(norm_G)
            hn = np.sum(im_centers_[-1]*norm_G[...,np.newaxis],axis=0)
        
        zc = np.sum(im_centers_[0]*im_centers_[3],axis=0)/np.sum(im_centers_[3],axis=0)
        xc = np.sum(im_centers_[1]*im_centers_[3],axis=0)/np.sum(im_centers_[3],axis=0)
        yc = np.sum(im_centers_[2]*im_centers_[3],axis=0)/np.sum(im_centers_[3],axis=0)
        Xh = np.array([zc,xc,yc,bk,a,habs,hn,h]).T
    if return_centers:
        return Xh,np.array(im_centers)
    return Xh
from scipy.spatial.distance import cdist
def get_set(fl):
     if '_set' in fl: 
        return int(fl.split('_set')[-1].split(os.sep)[0].split('_')[0])
     else:
        return 0
from dask.array import concatenate
def concat(ims):
    shape = np.min([im.shape for im in ims],axis=0)
    ims_ = []
    for im in ims:
        shape_ = im.shape
        tupl = tuple([slice((sh_-sh)//2, -(sh_-sh)//2 if sh_>sh else None) for sh,sh_ in zip(shape,shape_)])
        ims_.append(im[tupl][np.newaxis])
    
    return concatenate(ims_)
class analysis_smFISH():
    def __init__(self,data_folders = [r'X:\DCBB_human__11_18_2022'],
                 save_folder = r'X:\DCBB_human__11_18_2022_Analysis',
                 H0folder=  r'X:\DCBB_human__11_18_2022\H0*',exclude_H0=True):
        self.Qfolders = [fld for data_folder in data_folders 
                             for fld in glob.glob(data_folder+os.sep+'H*')]
        self.H0folders = glob.glob(H0folder)
        self.save_folder = save_folder
        if not os.path.exists(self.save_folder): os.makedirs(self.save_folder)
        if exclude_H0:
            self.Qfolders = [fld for fld in self.Qfolders if fld not in self.H0folders]
        self.fls_bk = np.sort([fl for H0fld in self.H0folders for fl in glob.glob(H0fld+os.sep+'*.zarr')])
        print("Found files:"+str(len(self.fls_bk)))
        print("Found hybe folders:"+str(len(self.Qfolders)))
    def set_set(self,set_=''):
        self.set_ = set_
        self.fls_bk_ = [fl for fl in self.fls_bk if set_ in fl]
    def set_fov(self,ifl,set_=None):
        if set_ is not None:
            self.set_set(set_)
        self.fl_bk = self.fls_bk_[ifl]
    def set_hybe(self,iQ):
        self.Qfolder = [qfld for qfld in self.Qfolders if self.set_ in qfld][iQ]
        self.fl = self.Qfolder+os.sep+os.path.basename(self.fl_bk)
    def get_background(self,force=False):
        ### define H0
        print('### define H0 and load background')
        if not (getattr(self,'previous_fl_bk','None')==self.fl_bk) or force:
            print("Background file: "+self.fl_bk)
            path0 =  self.fl_bk
            im0,x0,y0=read_im(path0,return_pos=True)
            self.im_bk_ = np.array(im0,dtype=np.float32)
            self.previous_fl_bk = self.fl_bk
    def get_signal(self):
        print('### load signal')
        print("Signal file: "+self.fl)
        path =  self.fl
        im,x,y=read_im(path,return_pos=True)
        self.ncols,self.szz,self.szx,self.szy = im.shape
        self.im_sig_ = np.array(im,dtype=np.float32)
    def compute_drift(self,sz=200):
        im0 = self.im_bk_[-1]
        im = self.im_sig_[-1]
        txyz,txyzs = get_txyz(im0,im,sz_norm=40,sz = sz,nelems=5,plt_val=False)
        self.txyz,self.txyzs=txyz,txyzs
        self.dic_drift = {'txyz':self.txyz,'Ds':self.txyzs,'drift_fl':self.fl_bk}
        print("Found drift:"+str(self.txyz))
    def get_aligned_ims(self):
        txyz = self.txyz
        Tref = np.round(txyz).astype(int)
        slices_bk = tuple([slice(None,None,None)]+[slice(-t_,None,None) if t_<=0 else slice(None,-t_,None) for t_ in Tref])
        slices_sig = tuple([slice(None,None,None)]+[slice(t_,None,None) if t_>=0 else slice(None,t_,None) for t_ in Tref])
        self.im_sig__ = np.array(self.im_sig_[slices_sig],dtype=np.float32)
        self.im_bk__ = np.array(self.im_bk_[slices_bk],dtype=np.float32)
    def subtract_background(self,ssub=40,s=10,plt_val=False):
        print("Reducing background...")
        self.im_ratio = get_dif_or_ratio(self.im_sig__,self.im_bk__,sx=ssub,sy=ssub,pad=5,col_align=-2)
        self.im_ration = np.array([norm_slice(im_,s=s) for im_ in self.im_ratio])
        if plt_val:
            import napari
            napari.view_image(self.im_ration,contrast_limits=[0,0.7])
    def get_Xh(self,th = 4,s=30,dic_psf=None,normalized=False):
        resc=  5
        self.Xhs = []
        for im_raw in self.im_sig_[:-1]:
            im_ = norm_slice(im_raw,s=s)
            th_ = np.std(im_[::resc,::resc,::resc])*th
            self.Xhs.append(get_local_max(im_,th_,im_raw=im_raw,dic_psf=dic_psf))
                   
    def check_finished_file(self):
        file_sig = self.fl
        save_folder = self.save_folder
        fov_ = os.path.basename(file_sig).split('.')[0]
        hfld_ = os.path.basename(os.path.dirname(file_sig))
        self.base_save = self.save_folder+os.sep+fov_+'--'+hfld_
        self.Xh_fl = self.base_save+'--'+'_Xh_RNAs.pkl'
        return os.path.exists(self.Xh_fl)
    def save_fits(self,icols=None,plt_val=True):
        if plt_val:
            if icols is None:
                icols =  range(self.ncols-1)
            for icol in icols:

                fig = plt.figure(figsize=(40,40))
                im_t = self.im_ration[icol]
                if False:
                    Xh = self.Xhs[icol]
                    H = Xh[:,-1]
                    vmax = np.median(np.sort(H)[-npts:])
                vmax = self.dic_th.get(icol,1)
                plt.imshow(np.max(im_t,0),vmin=0,vmax=vmax,cmap='gray')
                #plt.show()
                fig.savefig(self.base_save+'_signal-col'+str(icol)+'.png')
                plt.close('all')
        pickle.dump([self.Xhs,self.dic_drift],open(self.Xh_fl,'wb'))
def get_best_trans(Xh1,Xh2,th_h=1,th_dist = 2,return_pairs=False):
    mdelta = np.array([np.nan,np.nan,np.nan])
    if len(Xh1)==0 or len(Xh2)==0:
        if return_pairs:
            return mdelta,[],[]
        return mdelta
    X1,X2 = Xh1[:,:3],Xh2[:,:3]
    h1,h2 = Xh1[:,-1],Xh2[:,-1]
    i1 = np.where(h1>th_h)[0]
    i2 = np.where(h2>th_h)[0]
    if len(i1)==0 or len(i2)==0:
        if return_pairs:
            return mdelta,[],[]
        return mdelta
    i2_ = np.argmin(cdist(X1[i1],X2[i2]),axis=-1)
    i2 = i2[i2_]
    deltas = X1[i1]-X2[i2]
    dif_ = deltas
    bins = [np.arange(m,M+th_dist*2+1,th_dist*2) for m,M in zip(np.min(dif_,0),np.max(dif_,0))]
    hhist,bins_ = np.histogramdd(dif_,bins)
    max_i = np.unravel_index(np.argmax(hhist),hhist.shape)
    #plt.figure()
    #plt.imshow(np.max(hhist,0))
    center_ = [(bin_[iM_]+bin_[iM_+1])/2. for iM_,bin_ in zip(max_i,bins_)]
    keep = np.all(np.abs(dif_-center_)<=th_dist,-1)
    center_ = np.mean(dif_[keep],0)
    for i in range(5):
        keep = np.all(np.abs(dif_-center_)<=th_dist,-1)
        center_ = np.mean(dif_[keep],0)
    mdelta = center_
    keep = np.all(np.abs(deltas-mdelta)<=th_dist,1)
    if return_pairs:
        return mdelta,Xh1[i1[keep]],Xh2[i2[keep]]
    return mdelta
    
def norm_im_med(im,im_med):
    if len(im_med)==2:
        return (im.astype(np.float32)-im_med[0])/im_med[1]
    else:
        return im.astype(np.float32)/im_med
def read_im(path,return_pos=False):
    import zarr,os
    from dask import array as da
    dirname = os.path.dirname(path)
    fov = os.path.basename(path).split('_')[-1].split('.')[0]
    #print("Bogdan path:",path)
    file_ = dirname+os.sep+fov+os.sep+'data'
    #image = zarr.load(file_)[1:]
    image = da.from_zarr(file_)[1:]

    shape = image.shape
    #nchannels = 4
    xml_file = os.path.dirname(path)+os.sep+os.path.basename(path).split('.')[0]+'.xml'
    if os.path.exists(xml_file):
        txt = open(xml_file,'r').read()
        tag = '<z_offsets type="string">'
        zstack = txt.split(tag)[-1].split('</')[0]
        
        tag = '<stage_position type="custom">'
        x,y = eval(txt.split(tag)[-1].split('</')[0])
        
        nchannels = int(zstack.split(':')[-1])
        nzs = (shape[0]//nchannels)*nchannels
        image = image[:nzs].reshape([shape[0]//nchannels,nchannels,shape[-2],shape[-1]])
        image = image.swapaxes(0,1)
    shape = image.shape
    if return_pos:
        return image,x,y
    return image



def linear_flat_correction(ims,fl=None,reshape=True,resample=4,vec=[0.1,0.15,0.25,0.5,0.75,0.9]):
    #correct image as (im-bM[1])/bM[0]
    #ims=np.array(ims)
    if reshape:
        ims_pix = np.reshape(ims,[ims.shape[0]*ims.shape[1],ims.shape[2],ims.shape[3]])
    else:
        ims_pix = np.array(ims[::resample])
    ims_pix_sort = np.sort(ims_pix[::resample],axis=0)
    ims_perc = np.array([ims_pix_sort[int(frac*len(ims_pix_sort))] for frac in vec])
    i1,i2=np.array(np.array(ims_perc.shape)[1:]/2,dtype=int)
    x = ims_perc[:,i1,i2]
    X = np.array([x,np.ones(len(x))]).T
    y=ims_perc
    a = np.linalg.inv(np.dot(X.T,X))
    cM = np.swapaxes(np.dot(X.T,np.swapaxes(y,0,-2)),-2,1)
    bM = np.swapaxes(np.dot(a,np.swapaxes(cM,0,-2)),-2,1)
    if fl is not None:
        folder = os.path.dirname(fl)
        if not os.path.exists(folder):
            os.makedirs(folder)
        pickle.dump(bM,open(fl,'wb'))
    return bM 
def compose_mosaic(ims,xs_um,ys_um,ims_c=None,um_per_pix=0.108333,rot = 0,return_coords=False):
    dtype = np.float32
    im_ = ims[0]
    szs = im_.shape
    sx,sy = szs[-2],szs[-1]
    ### Apply rotation:
    theta=-np.deg2rad(rot)
    xs_um_ = np.array(xs_um)*np.cos(theta)-np.array(ys_um)*np.sin(theta)
    ys_um_ = np.array(ys_um)*np.cos(theta)+np.array(xs_um)*np.sin(theta)
    ### Calculate per pixel
    xs_pix = np.array(xs_um_)/um_per_pix
    xs_pix = np.array(xs_pix-np.min(xs_pix),dtype=int)
    ys_pix = np.array(ys_um_)/um_per_pix
    ys_pix = np.array(ys_pix-np.min(ys_pix),dtype=int)
    sx_big = np.max(xs_pix)+sx+1
    sy_big = np.max(ys_pix)+sy+1
    dim = [sx_big,sy_big]
    if len(szs)==3:
        dim = [szs[0],sx_big,sy_big]

    if ims_c is None:
        if len(ims)>25:
            try:
                ims_c = linear_flat_correction(ims,fl=None,reshape=False,resample=1,vec=[0.1,0.15,0.25,0.5,0.65,0.75,0.9])
            except:
                imc_c = np.median(ims,axis=0)
        else:
            ims_c = np.median(ims,axis=0)

    im_big = np.zeros(dim,dtype = dtype)
    sh_ = np.nan
    for i,(im_,x_,y_) in enumerate(zip(ims,xs_pix,ys_pix)):
        if ims_c is not None:
            if len(ims_c)==2:
                im_coef,im_inters = np.array(ims_c,dtype = 'float32')
                im__=(np.array(im_,dtype = 'float32')-im_inters)/im_coef
            else:
                ims_c_ = np.array(ims_c,dtype = 'float32')
                im__=np.array(im_,dtype = 'float32')/ims_c_*np.median(ims_c_)
        else:
            im__=np.array(im_,dtype = 'float32')
        im__ = np.array(im__,dtype = dtype)
        im_big[...,x_:x_+sx,y_:y_+sy]=im__
        sh_ = im__.shape
    if return_coords:
        return im_big,xs_pix+sh_[-2]/2,ys_pix+sh_[-1]/2
    return im_big
import cv2

def get_tiles(im_3d,size=256,delete_edges=False):
    sz,sx,sy = im_3d.shape
    if not delete_edges:
        Mz = int(np.ceil(sz/float(size)))
        Mx = int(np.ceil(sx/float(size)))
        My = int(np.ceil(sy/float(size)))
    else:
        Mz = np.max([1,int(sz/float(size))])
        Mx = np.max([1,int(sx/float(size))])
        My = np.max([1,int(sy/float(size))])
    ims_dic = {}
    for iz in range(Mz):
        for ix in range(Mx):
            for iy in range(My):
                ims_dic[(iz,ix,iy)]=ims_dic.get((iz,ix,iy),[])+[im_3d[iz*size:(iz+1)*size,ix*size:(ix+1)*size,iy*size:(iy+1)*size]] 
    return ims_dic
def norm_slice(im,s=50):
    im_=im.astype(np.float32)
    return np.array([im__-cv2.blur(im__,(s,s)) for im__ in im_],dtype=np.float32)


def get_txyz(im_dapi0,im_dapi1,sz_norm=40,sz = 200,nelems=5,plt_val=False):
    """
    Given two 3D images im_dapi0,im_dapi1, this normalizes them by subtracting local background (gaussian size sz_norm)
    and then computes correlations on <nelemes> blocks with highest  std of signal of size sz
    It will return median value and a list of single values.
    """
    im_dapi0 = np.array(im_dapi0,dtype=np.float32)
    im_dapi1 = np.array(im_dapi1,dtype=np.float32)
    im_dapi0_ = norm_slice(im_dapi0,sz_norm)
    im_dapi1_ = norm_slice(im_dapi1,sz_norm)
    dic_ims0 = get_tiles(im_dapi0_,size=sz,delete_edges=True)
    dic_ims1 = get_tiles(im_dapi1_,size=sz,delete_edges=True)
    keys = list(dic_ims0.keys())
    best = np.argsort([np.std(dic_ims0[key]) for key in keys])[::-1]
    txyzs = []
    im_cors = []
    for ib in range(min(nelems,len(best))):
        im0 = dic_ims0[keys[best[ib]]][0].copy()
        im1 = dic_ims1[keys[best[ib]]][0].copy()
        im0-=np.mean(im0)
        im0/=np.std(im0)
        im1-=np.mean(im1)
        im1/=np.std(im1)
        from scipy.signal import fftconvolve
        im_cor = fftconvolve(im0[::-1,::-1,::-1],im1, mode='full')
        if plt_val:
            plt.figure()
            plt.imshow(np.max(im_cor,0))
            #print(txyz)
        txyz = np.unravel_index(np.argmax(im_cor), im_cor.shape)-np.array(im0.shape)+1
        
        im_cors.append(im_cor)
        txyzs.append(txyz)
    txyz = np.median(txyzs,0).astype(int)
    return txyz,txyzs

class drift_refiner():
    def __init__(self,data_folder=r'\\132.239.200.33\Raw_data\DCBB_MER250__12_2_2022',
                 analysis_folder = r'\\132.239.200.33\Raw_data\DCBB_MER250__12_2_2022_Analysis'):
        """
        Example use:
        
        drift_  = drift_refiner()
        drift_.get_fov(0,'set1')
        Hs = len(drift_.raw_fls)
        ifovs = len(drift_.dapi_fls_)
        for ifov in tqdm(np.arange(ifovs)):
            drift_  = drift_refiner()
            drift_.get_fov(ifov,'set1')
            for iR in np.arange(Hs):
                analysis_folder_ = drift_.analysis_folder+os.sep+'distortion'
                if not os.path.exists(analysis_folder_):os.makedirs(analysis_folder_)
                fl = analysis_folder_+os.sep+os.path.basename(drift_.raw_fls[0]).split('.')[0]+'--'+drift_.set_+'--iR'+str(iR)+'.npy'
                if not os.path.exists(fl):
                    drift_.load_images(iR)
                    drift_.normalize_ims(zm=30,zM=50)
                    drift_.get_Tmed(sz_=300,th_cor=0.6,nkeep=9)
                    try:
                        P1_,P2_ = drift_.get_P1_P2_plus();
                        P1__,P2__ = drift_.get_P1_P2_minus();
                        P1f,P2f = np.concatenate([P1_,P1__]),np.concatenate([P2_,P2__])
                    except:
                        P1f,P2f = [],[]

                    if False:
                        import napari
                        viewer = napari.view_image(drift_.im2n,name='im2',colormap='green')
                        viewer.add_image(drift_.im1n,name='im1',colormap='red')
                        viewer.add_points(P2_,face_color='g',size=10)
                        viewer.add_points(P1_,face_color='r',size=10) 
                        drift_.check_transf(P1f,P2f)
                    try:
                        print("Error:",np.percentile(np.abs((P1f-P2f)-np.median(P1f-P2f,axis=0)),75,axis=0))
                        P1fT = drift_.get_Xwarp(P1f,P1f,P2f-P1f,nneigh=50,sgaus=20)
                        print("Error:",np.percentile(np.abs(P1fT-P2f),75,axis=0))
                    except:
                        pass

                    print(fl)
                    np.save(fl,np.array([P1f,P2f]))
        
        """         
                 
        
        self.data_folder = data_folder
        self.analysis_folder = analysis_folder
        self.dapi_fls = np.sort(glob.glob(analysis_folder+os.sep+'Segmentation'+os.sep+'*--dapi_segm.npz'))
    def get_fov(self,ifov=10,set_='set1',keepH = ['H'+str(i)+'_' for i in range(1,9)]):
        self.set_ = set_
        self.ifov = ifov
        self.dapi_fls_ = [fl for fl in self.dapi_fls if set_+'-' in os.path.basename(fl)]
        self.dapi_fl = self.dapi_fls_[ifov]
        
        fov = os.path.basename(self.dapi_fl).split('--')[0]
        
        self.allHfolders = glob.glob(self.data_folder+os.sep+'H*')
        self.raw_fls = [[fld+os.sep+fov+'.zarr' for fld in self.allHfolders 
                         if (tag in os.path.basename(fld)) and (self.set_ in os.path.basename(fld))][0] for tag in keepH]
    def load_segmentation(self):
        dapi_fl  = self.dapi_fl
        im_segm = np.load(dapi_fl)['segm']
        shape = np.load(dapi_fl)['shape']
        cellcaps = [resize_slice(pair,im_segm.shape,shape) for icell,pair in enumerate(ndimage.find_objects(im_segm)) if pair is not None]
        cm_cells = np.array([[(sl.start+sl.stop)//2 for sl in sls]for sls in cellcaps])
        self.cellcaps=cellcaps
        self.cm_cells=cm_cells
    def load_images(self,ifl2,ifl1=None,icol=-1):
        if ifl1 is None:
            ifl1 = len(self.raw_fls)//2
        self.reloadfl1=True
        if hasattr(self,'im1'):
            if self.fl1==self.raw_fls[ifl1]:
                self.reloadfl1=False
        
        self.fl1,self.fl2 = self.raw_fls[ifl1],self.raw_fls[ifl2]
        print("Loading images:",self.fl1,self.fl2)
        
        if self.reloadfl1:
            self.im1 = np.array(read_im(self.fl1)[icol],np.float32)
        self.im2 = np.array(read_im(self.fl2)[icol],np.float32)
        
        self.sh = np.array(self.im1.shape)
    def normalize_ims(self,zm=5,zM=50):
        im01 = self.im1
        if self.reloadfl1 or not hasattr(self,'im1n'):
            self.im1n = np.array([cv2.blur(im_,(zm,zm))-cv2.blur(im_,(zM,zM)) for im_ in im01])
        im02 = self.im2
        self.im2n = np.array([cv2.blur(im_,(zm,zm))-cv2.blur(im_,(zM,zM)) for im_ in im02])
    def apply_drift(self,cell,Tmed,sh=None):
        if sh is None:
            sh = np.array(self.im1.shape)
        xm,xM = np.array([(sl.start,sl.stop)for sl in cell]).T
        xm1,xM1 = xm-Tmed,xM-Tmed
        xm2,xM2 = xm,xM
        bad = xm1<0
        xm2[bad]=xm2[bad]-xm1[bad]
        xm1[bad]=0
        bad = xM2>sh
        xM1[bad]=xM1[bad]-(xM2-sh)[bad]
        xM2[bad]=sh[bad]
        return tuple([slice(x_,x__) for x_,x__ in zip(xm1,xM1)]),tuple([slice(x_,x__) for x_,x__ in zip(xm2,xM2)])
    
    def get_cell_caps(self,sz_ = 40):
        sh = self.sh
        szz,szx,szy = np.ceil(sh/sz_).astype(int)
        cellcaps = [(slice(iz*sz_,(iz+1)*sz_),slice(ix*sz_,(ix+1)*sz_),slice(iy*sz_,(iy+1)*sz_))
                      for ix in range(szx) for iy in range(szy) for iz in range(szz)]
        return cellcaps
    def filter_cor(self,P1,h1,P2,h2,cor_th=0.75):
        h1_ = h1.copy().T
        h1_ = h1_-np.nanmean(h1_,axis=0)
        h1_ = h1_/np.nanstd(h1_,axis=0)

        h2_ = h2.copy().T
        h2_ = h2_-np.nanmean(h2_,axis=0)
        h2_ = h2_/np.nanstd(h2_,axis=0)
        cors = np.nanmean(h1_*h2_,axis=0)
        keep=cors>cor_th
        P1_ = P1[keep]
        P2_ = P2[keep]
        return P1_,P2_


    def get_Xwarp(self,x_ch,X,T,nneigh=10,sgaus=20,szero=10):
        #X,T = cm_cells[keep],txyzs[keep]
        #T = T+Tmed
        from scipy.spatial import cKDTree
        tree = cKDTree(X)


        dists,inds = tree.query(x_ch,nneigh);
        ss=sgaus
        Tch = T[inds].copy()
        #Tch[:,-1]=0
        #dists[:,-1] = ss*szero
        M = np.exp(-dists*dists/(2*ss*ss))
        TF = np.sum(Tch*M[...,np.newaxis],axis=1)/np.sum(M,axis=1)[...,np.newaxis]
        #TF = Tch[:,0]#np.median(Tch,axis=1)
        TF = np.round(TF).astype(int)


        XF = x_ch+TF
        XF[XF<0]=0
        sh = self.sh
        for idim in range(len(sh)):
            XF[XF[:,idim]>=sh[idim],idim]=sh[idim]-1

        return XF
    def get_Tmed(self,sz_=300,th_cor=0.75,nkeep=5):
        """Assuming that self.imn1 and self.imn2 are loaded and normalized, this takes """
        cellcaps = self.get_cell_caps(sz_ = sz_)#self.cellcaps
        cm_cells = np.array([[(sl.start+sl.stop)//2 for sl in sls]for sls in cellcaps])
        txyzs,cors = [],[]
        icells = np.argsort([np.std(self.im1n[cell]) for cell in cellcaps])[::-1][:nkeep]
        for icell in tqdm(icells):
            cell = cellcaps[icell]
            imsm1,imsm2 = self.im1n[cell],self.im2n[cell]
            txyz,cor = get_txyz_small(imsm1,imsm2,sz_norm=0,plt_val=False,return_cor=True)
            txyzs.append(txyz)
            cors.append(cor)
        cors = np.array(cors)
        txyzs = np.array(txyzs)
        keep = cors>th_cor
        print("Keeping fraction of cells: ",np.mean(keep))
        self.Ts = txyzs[keep]
        self.Tmed = np.median(txyzs[keep],axis=0).astype(int)
        
    def get_max_min(self,P,imn,delta_fit=5,ismax=True,return_ims=False):
        XI = np.indices([2*delta_fit+1]*3)-delta_fit
        keep = (np.sum(XI*XI,axis=0)<=(delta_fit*delta_fit))
        XI = XI[:,keep].T
        XS = (P[:,np.newaxis]+XI[np.newaxis])
        shape = self.sh
        XSS = XS.copy()
        XS = XS%shape
        #XSS = XS.copy()
        is_bad = np.any((XSS!=XS),axis=-1)


        sh_ = XS.shape
        XS = XS.reshape([-1,3])
        im1n_local = imn[tuple(XS.T)].reshape(sh_[:-1])
        #print(is_bad.shape,im1n_local.shape)
        im1n_local[is_bad]=np.nan
        im1n_local_ = im1n_local.copy()
        im1n_local_[is_bad]=-np.inf
        XF = XSS[np.arange(len(XSS)),np.nanargmax(im1n_local_,axis=1)]
        #im1n_med = np.min(im1n_local,axis=1)[:,np.newaxis]
        #im1n_local_ = im1n_local.copy()
        #im1n_local_ = np.clip(im1n_local_-im1n_med,0,np.inf)
        if return_ims:
            return XF,im1n_local
        return XF
    def get_XB(self,im_,th=3):
        #im_ = self.im1n
        std_ = np.std(im_[::5,::5,::5])
        #im_base = im_[1:-1,1:-1,1:-1]
        #keep=(im_base>im_[:-2,1:-1,1:-1])&(im_base>im_[1:-1,:-2,1:-1])&(im_base>im_[1:-1,1:-1,:-2])&\
        #    (im_base>im_[2:,1:-1,1:-1])&(im_base>im_[1:-1,2:,1:-1])&(im_base>im_[1:-1,1:-1,2:])&(im_base>std_*3);#&(im_[:1]>=im_[1:])
        #XB = np.array(np.where(keep)).T+1

        keep = im_>std_*th
        XB = np.array(np.where(keep)).T
        from tqdm import tqdm
        for delta_fit in tqdm([1,2,3,5,7,10,15]):
            XI = np.indices([2*delta_fit+1]*3)-delta_fit
            keep = (np.sum(XI*XI,axis=0)<=(delta_fit*delta_fit))
            XI = XI[:,keep].T
            XS = (XB[:,np.newaxis]+XI[np.newaxis])
            shape = self.sh
            XS = XS%shape

            keep = im_[tuple(XB.T)]>=np.max(im_[tuple(XS.T)],axis=0)
            XB = XB[keep]
        return XB
    def get_P1_P2_plus(self):
        if self.reloadfl1 or not hasattr(self,'P1_plus'):
            P10 = self.get_XB(self.im1n,th=3)
            P1,h1 = self.get_max_min(P10,self.im1n,delta_fit=15,ismax=True,return_ims=True)
            P1,h1 = self.get_max_min(P1,self.im1n,delta_fit=7,ismax=True,return_ims=True)
            self.P1_plus,self.h1_plus = P1,h1
        P1,h1 = self.P1_plus,self.h1_plus
        Tmed = self.Tmed.astype(int)
        P20 = P1+Tmed
        P2,h2 = self.get_max_min(P20,self.im2n,delta_fit=15,ismax=True,return_ims=True)
        P2,h2 = self.get_max_min(P2,self.im2n,delta_fit=7,ismax=True,return_ims=True)
        P1_,P2_ = self.filter_cor(P1,h1,P2,h2,cor_th=0.75)
        print(len(P1_)/len(P1))
        return P1_,P2_
    def get_P1_P2_minus(self):
        if self.reloadfl1 or not hasattr(self,'P1_minus'):
            P10 = self.get_XB(-self.im1n,th=2)
            P1,h1 = self.get_max_min(P10,-self.im1n,delta_fit=15,ismax=True,return_ims=True)
            P1,h1 = self.get_max_min(P1,-self.im1n,delta_fit=7,ismax=True,return_ims=True)
            self.P1_minus,self.h1_minus = P1,h1
        P1,h1 = self.P1_minus,self.h1_minus
        Tmed = self.Tmed.astype(int)
        P20 = P1+Tmed
        P2,h2 = self.get_max_min(P20,-self.im2n,delta_fit=15,ismax=True,return_ims=True)
        P2,h2 = self.get_max_min(P2,-self.im2n,delta_fit=7,ismax=True,return_ims=True)
        #P2 = get_max_min(self,P2,self.im2n,delta_fit=10,ismax=True)
        #P2 = get_max_min(self,P2,self.im2n,delta_fit=5,ismax=True)
        P1_,P2_ = self.filter_cor(P1,h1,P2,h2,cor_th=0.75)
        print(len(P1_)/len(P1))
        return P1_,P2_
    def check_transf(self,P1_,P2_,nneigh=30,sgaus=20):
        shape = self.sh
        Tmed = self.Tmed.astype(int)
        X_,Y_,Z_=np.indices([1,shape[1],shape[2]])
        X_+=shape[0]//2
        x_ch = np.array([X_,Y_,Z_]).reshape([3,-1]).T
        X=P1_
        T=P2_-P1_
        XF = self.get_Xwarp(x_ch,X,T,nneigh=nneigh,sgaus=sgaus)
        im2_ = self.im2n[tuple(XF.T)].reshape([shape[1],shape[2]])
        im1_ = self.im1n[tuple(x_ch.T)].reshape([shape[1],shape[2]])
        import napari
        from scipy.ndimage import shift
        viewer=napari.view_image(np.array([shift(self.im2n[shape[0]//2+Tmed[0]],-Tmed[1:]),im1_,im2_]))
        viewer.add_points(P1_[:,1:],face_color='g',size=10)
        P2__ = P2_+np.median(P1_-P2_,axis=0)
        viewer.add_points(P2__[:,1:],face_color='r',size=10)
        
        
        
        
import glob,os,numpy as np
import pickle
from tqdm import tqdm
from scipy.spatial import cKDTree
def get_all_pos(analysis_folder = r'\\132.239.200.33\Raw_data\DCBB_MER250__12_2_2022_Analysis',
                data_folder = r'\\132.239.200.33\Raw_data\DCBB_MER250__12_2_2022\H1_set*',set_='set1',ext='.zarr',force=False):
    
    fl_pos = analysis_folder+os.sep+'pos_'+set_+'.pkl'
    if os.path.exists(fl_pos) and not force:
        dic_coords = pickle.load(open(fl_pos,'rb'))
    else:
        dic_coords = {}
        allflds = glob.glob(data_folder)
        for fld in allflds:
            if set_ in fld:
                for fl in tqdm(glob.glob(fld+os.sep+'*'+ext)):
                    dic_coords[get_ifov(fl)]=get_pos(fl)
        pickle.dump(dic_coords,open(fl_pos,'wb'))
    return dic_coords
def get_pos(path):
    xml_file = os.path.dirname(path)+os.sep+os.path.basename(path).split('.')[0]+'.xml'
    x,y=0,0
    if os.path.exists(xml_file):
        txt = open(xml_file,'r').read()
        tag = '<stage_position type="custom">'
        x,y = eval(txt.split(tag)[-1].split('</')[0])
    return x,y
def get_ifov(fl):return os.path.basename(fl).split('Point')[1].split('_')[0].split('.')[0]
def get_ifov_raw(fl):return os.path.basename(fl).split('Point')[1].split('.')[0]
def get_icol(fl):return int(os.path.basename(fl).split('ch')[1].split('.')[0].split('_')[0])-1 #maybe correct!!!
def get_H(fl):return int(os.path.basename(fl).split('hyb')[1].split('_')[0])
def get_H_raw(fl):return int(os.path.basename(fl).split('hyb')[1].split('_')[0])
def get_iH_npy(fl): return int(os.path.basename(fl).split('--iR')[-1].split('.')[0])
def get_iH_npz(fl): return int(os.path.basename(fl).split('hyb')[1].split('_')[0])
def get_Xwarp(x_ch,X,T,nneigh=50,sgaus=100):
    from scipy.spatial import cKDTree
    tree = cKDTree(X)


    dists,inds = tree.query(x_ch,nneigh);
    dists = dists[:,:len(X)]
    inds = inds[:,:len(X)]
    ss=sgaus
    Tch = T[inds].copy()
    
    M = -dists*dists/(2*ss*ss)
    M = M-np.max(M,axis=-1)[...,np.newaxis]
    M = np.exp(M)
    #M = dists<sgaus
    TF = np.sum(Tch*M[...,np.newaxis],axis=1)/np.sum(M,axis=1)[...,np.newaxis]
    #bad = np.any(np.isnan(TF),axis=1)
    #TF[bad] = np.median(TF[~bad],axis=0)
    XF = x_ch+TF
    return XF
def get_closest_nan(X,bad):
    from scipy.spatial.distance import cdist
    zsu_ = X.astype(float)
    zsu__ = zsu_.copy()
    zsu__[bad]=np.inf
    M = cdist(zsu_,zsu__)
    inds = np.argmin(M,axis=-1)
    return inds[bad]
class decoder():
    def __init__(self,analysis_folder = r'\\132.239.200.33\Raw_data\DCBB_MER250__12_2_2022_Analysis',force=False):
        """
        Use as:
        dec = decoder(analysis_folder = r'\\132.239.200.33\Raw_data\DCBB_MER250__12_2_2022_Analysis')
        dec.get_set_ifov(ifov=2,set_='set1',keepH = [1,2,3,4,5,6,7,8],ncols=3)
        print("Loading the fitted molecules")
        dec.get_XH()
        print("Correcting for distortion acrossbits")
        dec.apply_distortion_correction()
        dec.load_library()
        dec.XH = dec.XH[dec.XH[:,-4]>0.25]
        dec.get_inters(distance_th=3)
        dec.pick_best_brightness(nUR_cutoff = 3,resample = 10000)
        dec.pick_best_score(nUR_cutoff = 3)
        """
        self.analysis_folder=analysis_folder
        
        self.files_map = self.analysis_folder+os.sep+'files_map.npz'
        if not os.path.exists(self.files_map) or force:
            self.remap_files()
        else:
            try:
                result = np.load(self.files_map)
                #self.files,self.dapi_fls = result['files'],result['dapi_fls']
                self.files,self.rawfiles = result['files'],result['rawfiles']
                if len(self.files)==0: self.remap_files()
            except KeyError: self.remap_files()
        self.save_folder = self.analysis_folder+os.sep+'Decoded'
        if not os.path.exists(self.save_folder): os.makedirs(self.save_folder)
            
            
            
    def remap_files(self): #get list of spot files and DAPI segmentation files
        self.files=np.sort(glob.glob(self.analysis_folder+os.sep+'Result' + os.sep + 'called_spots'+os.sep+ '*.npz'))
        #self.dapi_fls = np.sort(glob.glob(self.analysis_folder+os.sep+'Segmentation'+os.sep+'*--dapi_segm.npz'))
        self.dapi_files = np.sort(glob.glob(self.analysis_folder+os.sep + 'rawdata' + os.sep + 'dapi_tifs' + os.sep +'*.tif')) 
        self.rawfiles=np.sort(glob.glob(self.analysis_folder+os.sep+'rawdata'+os.sep+ '*.nd2'))
        np.savez(self.files_map,files=self.files, rawfiles=self.rawfiles, dapi_files = self.dapi_files)
    def get_set_ifov(self,ifov=0,set_='',keepH = [1,2,3,4,5,6,7,8,9,10,11,12],ncols=2):
        """map all the complete files in self.dic_fls_final for the hybes H<*> in keepH.
        Put the files for fov <ifov> in self.fls_fov"""
        
        self.set_ = set_ #''
        self.ifov = ifov #number of the FoV
        
        self.keepH = keepH #hybs to keep for that FoV
        self.files_set = [fl for fl in self.files if (set_ in os.path.basename(fl))]
        self.raw_files_set = [fl for fl in self.rawfiles if (set_ in os.path.basename(fl))]
        self.ncols=ncols
        
        def refine_fls(fls,keepH):
            fls_ = [fl for fl in fls if get_H(fl) in keepH]
            Hs = [fl for fl in fls_ if get_H(fl)]
            fls_ = np.array(fls_)[np.argsort(Hs)]
            return fls_
        def refine_fls_raw(fls,keepH):
            fls_ = [fl for fl in fls if get_H_raw(fl) in keepH]
            Hs = [fl for fl in fls_ if get_H_raw(fl)]
            fls_ = np.array(fls_)[np.argsort(Hs)]
            return fls_
        dic_fls = {}
        
        for fl in self.files_set:
            ifov_ = get_ifov(fl)
            if ifov_ not in dic_fls: dic_fls[ifov_]=[]
            dic_fls[ifov_].append(fl)
        
        
        dic_fls_final = {ifv:refine_fls(dic_fls[ifv],keepH) for ifv in dic_fls}
        self.dic_fls_final = {ifv:dic_fls_final[ifv] for ifv in np.sort(list(dic_fls_final.keys())) 
                              if len(dic_fls_final[ifv])==len(keepH)*ncols}
        self.fls_fov = self.dic_fls_final.get(ifov,[])
        

        self.is_complete=False
        self.out_of_range = False
        if len(self.fls_fov)>0:
            self.fov = os.path.basename(fl).split('_')[1].split('_')[0]
            self.save_file_dec = self.save_folder+os.sep+self.fov+'--'+self.set_+'_decoded.npz'
            self.save_file_cts = self.save_folder+os.sep+self.fov+'--'+self.set_+'_cts.npz'
            if os.path.exists(self.save_file_cts):
                self.is_complete=True
        else:
            self.out_of_range = True
            self.is_complete=True
        
        dic_rawfls = {}
        for fl in self.raw_files_set:
            ifov_ = get_ifov_raw(fl)
            if ifov_ not in dic_rawfls: dic_rawfls[ifov_]=[]
            dic_rawfls[ifov_].append(fl)
        
        dic_rawfls_final = {ifv:refine_fls_raw(dic_rawfls[ifv],keepH) for ifv in dic_rawfls}
        self.dic_rawfls_final = {ifv:dic_rawfls_final[ifv] for ifv in np.sort(list(dic_rawfls_final.keys())) 
                              if len(dic_rawfls_final[ifv])==len(keepH)}
        self.dic_raw_fls_fov = {get_H_raw(fl): fl for fl in self.dic_rawfls_final.get(ifov)}
        self.fov = 'Point' + self.ifov
        
    def compute_drift_jakob(self): #Jakob
        import nd2
        im1 = nd2.imread(self.dic_raw_fls_fov[1],dask=True)[:,0] #read in DAPI
        szz = 50 #resample
        izs = np.arange(szz//2,len(im1)-szz//2,szz) # range 
        print("Computing drift:")
        iz = izs[np.argmax([np.std(np.array(im1[iz_])) for iz_ in izs])]
        im1_ = np.array(im1[(iz-szz//2):(iz+szz//2)],dtype=np.float32)
        iz
        dic_drift = {}
        for hyb in self.dic_raw_fls_fov:
            fl = self.dic_raw_fls_fov.get(hyb)
            im2 = nd2.imread(fl,dask=True)[:,0]
            im2_ = np.array(im2[(iz-szz//2):(iz+szz//2)],dtype=np.float32)
            txyz,txyzs = get_txyz(im1_,im2_,sz_norm=15,sz = 250,nelems=7,plt_val=False)
            print('hyb ' + str(hyb) + str(txyz))
            dic_drift[hyb] = -txyz
        self.dic_drift = dic_drift
        drift_fl = self.analysis_folder + os.sep + 'drift_correction' + os.sep + 'drift_correction_' + self.ifov +'.npz'
        with open(drift_fl, 'wb') as f:
            pickle.dump(self.dic_drift, f)
        print('Saved drift dictionary at ' + drift_fl)
        
    def get_XH(self,hyb_1st_bit = 1,apply_drift=True):
        """given self.fls_fov this loads each fitted file and keeps: zc,xc,yc,hn,h,icol into self.XH
        Also saves """
        #self.drift_set=np.array([[0,0,0]for i_ in range(len(self.fls_fov)*self.ncols)])

        for iFl in tqdm(np.arange(len(self.fls_fov))):
            fl = self.fls_fov[iFl]
            icol = get_icol(fl)
            iH = get_H(fl)
            Xh= np.load(fl)['arr_0']
            if iFl==0:
                XH = np.zeros([0, Xh.shape[1]+2])
            if apply_drift:
                Xh[:, :3] = Xh[:, :3]+self.dic_drift[iH]# plus or minus
 
            R = (iH-hyb_1st_bit)*self.ncols+icol #2 for 2 colours
            icolRs = np.ones([Xh.shape[0],2])
            icolRs[:,0]=icol
            icolRs[:,1]=R
            Xf = np.column_stack((Xh,icolRs))
            XH = np.concatenate([XH,Xf],axis=0)
        self.XH=XH
     
        
    def get_counts_per_cell(self,nbad=0):
        keep_good = np.sum(self.res_pruned==-1,axis=-1)<=nbad
        Xcms = self.Xcms[keep_good]
        icodes = self.icodes[keep_good]
        Xred = np.round((Xcms/self.shape)*self.shapesm).astype(int)
        good = ~np.any((Xred>=self.shapesm)|(Xred<0),axis=-1)
        Xred = Xred[good]
        icodes = icodes[good]
        cts_all = []
        for ikeep in np.arange(len(self.gns_names)):
            Xred_ = Xred[icodes==ikeep]
            icells,cts = np.unique(self.im_segm[tuple(Xred_.T)],return_counts=True)
            dic_cts = {icell:ct for icell,ct in zip(icells,cts)}
            ctsf = [dic_cts.get(icell,0) for icell in self.icells]
            cts_all.append(ctsf)
        cts_all = np.array(cts_all)
        return cts_all

    def plot_drift_image(self,hyb, ch,viewer = None, withdask = True):
        import nd2, napari
        from dask import array as da
        drift = self.dic_drift[hyb].astype(int)
        fl = self.dic_raw_fls_fov.get(hyb)
        fov = self.ifov
        if withdask:
            im = nd2.imread(fl,dask=True)
        else:
            im = nd2.imread(fl,dask=False)  
        #drift = -np.array(tzxy)
        rtxyz_col = [drift[0],0]+list(drift[1:])
        im_ = np.roll(im,rtxyz_col,axis=[0,1,2,3])
        if viewer is None: viewer = napari.Viewer()
        viewer.add_image(im_[:,ch],name = 'fov' +fov + '_hyb' + str(hyb) + '_ch' + str(ch) ,contrast_limits=[0,600])
        return viewer,im_[:,ch]
    
    def plot_image(self,hyb, ch,viewer = None, withdask = True):
        import nd2, napari
        from dask import array as da
        fl = self.dic_raw_fls_fov.get(hyb)
        fov = self.ifov
        if withdask:
            im = nd2.imread(fl,dask=True)
        else:
            im = nd2.imread(fl,dask=False)  
        if viewer is None: viewer = napari.Viewer()
        viewer.add_image(im[:,ch],name = 'fov' +fov + '_hyb' + str(hyb) + '_ch' + str(ch) ,contrast_limits=[0,600])
        return viewer
        
    def plot_points(self,bit,th=200,th_g=0, viewer = None, colour = 'y', ih=-3, ic=-4):
        if viewer is None: viewer = napari.Viewer()
        fov = self.ifov
        XH_ = self.XH[self.XH[:,-1]==bit]
        hs = XH_[:,ih] #-3: deconv space, -5: raw space
        gs = XH_[:,ic] #-4: deconv space, -6: raw space
        keep = np.logical_and(hs>th, gs>th_g)
        X = XH_[keep,:3]
        viewer.add_points(X,name='fov' + fov + "_points_bit"+str(bit) + '_th' + str(th),face_color=np.array([0,0,0,0]),edge_color=colour,size=10)
        return viewer
    def plot_dec_points(self,fov,gene,viewer = None, colour = 'y'):
        if viewer is None: viewer = napari.Viewer()
        keep  = (self.icodes==self.gns_names.index(gene)).astype(bool)
        RNAs = self.Xcms[keep]
        viewer.add_points(RNAs,name='fov' + fov + '_' + gene,face_color=np.array([0,0,0,0]),edge_color=colour,size=10)
        return viewer
            
    #def map_to_raw_data(self,)
         #plot corrected points from self.XH on translated images from dic_drift on raw data. 
    def load_library(self,lib_fl = r'codebook_DCBB250.csv',nblanks=300):
        code_txt = np.array([ln.replace('\n','').split(',') for ln in open(lib_fl,'r') if ',' in ln])
        gns = code_txt[1:,0]
        code_01 = code_txt[1:,2:].astype(int)
        codes = np.array([np.where(cd)[0] for cd in code_01])
        codes_ = [list(np.sort(cd)) for cd in codes]
        nbits = np.max(codes)+1


        ### get extrablanks
        from itertools import combinations
        X_codes = np.array((list(combinations(range(nbits),4))))
        X_code_01 = []
        for cd in X_codes:
            l_ = np.zeros(nbits)
            l_[cd] = 1
            X_code_01.append(l_)
        X_code_01 = np.array(X_code_01,dtype=int)
        from scipy.spatial.distance import cdist
        eblanks = np.where(np.min(cdist(code_01,X_code_01,metric='hamming'),0)>=4/float(nbits))[0]
        if nblanks>0:
            eblanks = eblanks[np.linspace(0,len(eblanks)-1,nblanks).astype(int)]
        codes__ = [list(e)for e in X_codes[eblanks]] + codes_
        gns__ = ['blanke'+str(ign+1).zfill(4) for ign in range(len(eblanks))] + list(gns)
        
        bad_gns = np.array(['blank' in e for e in gns__])
        good_gns = np.where(~bad_gns)[0]
        bad_gns = np.where(bad_gns)[0]

        
        
        self.lib_fl = lib_fl ### name of coding library
        self.nbits = nbits ### number of bits
        self.gns_names = gns__  ### names of genes and blank codes
        self.bad_gns = bad_gns ### indices of the blank codes
        self.good_gns = good_gns ### indices of the real gene codes
        self.codes__ = codes__ ### final extended codes of form [bit1,bit2,bit3,bit4]
        self.codes_01 = np.concatenate([code_01,X_code_01[eblanks]],axis=0) ### final extended codes of form [0,1,0,0,1...]
        
        dic_bit_to_code = {}
        for icd,cd in enumerate(self.codes__): 
            for bit in cd:
                if bit not in dic_bit_to_code: dic_bit_to_code[bit]=[]
                dic_bit_to_code[bit].append(icd)
        self.dic_bit_to_code = dic_bit_to_code  ### a dictinary in which each bit is mapped to the index of a code
    def load_library_v2(self,lib_fl = r'codebook_DCBB250.csv',nblanks=300):
        code_txt = np.array([ln.replace('\n','').split(',') for ln in open(lib_fl,'r') if ',' in ln])
        gns = code_txt[1:,0]
        code_01 = code_txt[1:,2:].astype(int)
        bits = np.array([int(bitname.split('bit')[1]) for bitname in code_txt[0,2:]])-1
        codes = bits[np.array([np.where(cd)[0] for cd in code_01])]
        codes_ = [list(np.sort(cd)) for cd in codes]
        nbits = len(bits)

        if nblanks>0:
        ### get extrablanks
            from itertools import combinations
            X_codes = np.array((list(combinations(range(nbits),4))))
            X_codes = np.array([code for code in X_codes if np.all(code % 2 == 0) or np.all(code % 2 != 0)]) #only keep codes encoded in one color 
            X_code_01 = []
            for cd in X_codes:
                l_ = np.zeros(nbits)
                l_[cd] = 1
                X_code_01.append(l_)
            X_code_01 = np.array(X_code_01,dtype=int)
            from scipy.spatial.distance import cdist
            eblanks = np.where(np.min(cdist(code_01,X_code_01,metric='hamming'),0)>=2/float(nbits))[0]

            Mblank = cdist(code_01,X_code_01[eblanks],metric='hamming')
            Mcode = cdist(code_01,code_01,metric='hamming')
            Mcode_mean = np.mean(np.mean(Mcode,axis=0))
            blank_mean_dist = np.mean(Mblank,axis=0)
            eblanks = eblanks[np.argsort(np.abs(blank_mean_dist - Mcode_mean))]
            
            eblanks_even = [eblank for eblank in eblanks if np.all(X_codes[eblank] %2 == 0)][:nblanks//2]
            eblanks_odd  = [eblank for eblank in eblanks if np.all(X_codes[eblank] %2 == 1)][:nblanks//2]
            eblanks = np.array(eblanks_even + eblanks_odd) ### here
            codes__ = [list(e)for e in X_codes[eblanks]] + codes_
            gns__ = ['blanke'+str(ign+1).zfill(4) for ign in range(len(eblanks))] + list(gns)
            code01__ = np.concatenate([X_code_01[eblanks],code_01],axis=0)
        else:
            codes__ = codes_
            gns__ = list(gns)
            code01__ = code_01

        bad_gns = np.array(['blank' in e for e in gns__])
        good_gns = np.where(~bad_gns)[0]
        bad_gns = np.where(bad_gns)[0]

        self.lib_fl = lib_fl ### name of coding library
        self.nbits = nbits ### number of bits
        self.gns_names = gns__  ### names of genes and blank codes
        self.bad_gns = bad_gns ### indices of the blank codes
        self.good_gns = good_gns ### indices of the real gene codes
        self.codes__ = codes__ ### final extended codes of form [bit1,bit2,bit3,bit4]
        self.codes_01 =  code01__### final extended codes of form [0,1,0,0,1...]

        dic_bit_to_code = {}
        for icd,cd in enumerate(self.codes__): 
            for bit in cd:
                if bit not in dic_bit_to_code: dic_bit_to_code[bit]=[]
                dic_bit_to_code[bit].append(icd)
        self.dic_bit_to_code = dic_bit_to_code  ### a dictinary in which each bit is mapped to the index of a code
        
    def get_inters(self,dinstance_th=2,enforce_color=False):
        """Get an initial intersection of points and save in self.res"""
        res =[]
        if enforce_color:
            icols = self.XH[:,-2]
            XH = self.XH
            for icol in tqdm(np.unique(icols)):
                inds = np.where(icols==icol)[0]
                Xs = XH[inds,:3]
                Ts = cKDTree(Xs)
                res_ = Ts.query_ball_tree(Ts,dinstance_th)
                res += [inds[r] for r in res_]
        else:
            XH = self.XH
            Xs = XH[:,:3]
            Ts = cKDTree(Xs)
            res = Ts.query_ball_tree(Ts,dinstance_th)
        self.res = res
     
    def get_icodes(self,nmin_bits=4,method = 'top4',redo=False,norm_brightness=None):    
        #### unfold res which is a list of list with clusters of loc.


        res = self.res

        import time
        start = time.time()
        res = [r for r in res if len(r)>=nmin_bits]
        #rlens = [len(r) for r in res]
        #edges = np.cumsum([0]+rlens)
        res_unfolder = np.array([r_ for r in res for r_ in r])
        #res0 = np.array([r[0] for r in res for r_ in r])
        ires = np.array([ir for ir,r in enumerate(res) for r_ in r])
        print("Unfolded molecules:",time.time()-start)

        ### get scores across bits
        import time
        start = time.time()
        RS = self.XH[:,-1].astype(int)
        brighness = self.XH[:,-3]
        brighness_n = brighness.copy()
        if norm_brightness is not None:
            colors = self.XH[:,norm_brightness]#self.XH[:,-1] for bits
            med_cols = {col: np.median(brighness[col==colors])for col in np.unique(colors)}
            for col in np.unique(colors):
                brighness_n[col==colors]=brighness[col==colors]/med_cols[col]
        scores = brighness_n[res_unfolder]

        bits_unfold = RS[res_unfolder]
        nbits = len(np.unique(RS))
        scores_bits = np.zeros([len(res),nbits])
        arg_scores = np.argsort(scores)
        scores_bits[ires[arg_scores],bits_unfold[arg_scores]]=scores[arg_scores]

        import time
        start = time.time()
        ### There are multiple avenues here: 
        #### nearest neighbors - slowest
        #### best dot product - reasonable and can return missing elements - medium speed
        #### find top 4 bits and call that a code - simplest and fastest

        if method == 'top4':
            n_on_bits = nmin_bits
            codes = self.codes__
            vals = np.argsort(scores_bits,axis=-1)
            bcodes = np.sort(vals[:,-n_on_bits:],axis=-1)
            base = [nbits**ion for ion in np.arange(n_on_bits)[::-1]]
            bcodes_b = np.sum(bcodes*base,axis=1)
            codes_b = np.sum(np.sort(codes,axis=-1)*base,axis=1)
            icodesN = np.zeros(len(bcodes_b),dtype=int)-1
            for icd,cd in enumerate(codes_b):
                icodesN[bcodes_b==cd]=icd
            bad = np.sum(scores_bits>0,axis=-1)<n_on_bits
            
            icodesN[bad]=-1
            igood = np.where(icodesN>-1)[0]
            inds_spotsN =  np.zeros([len(res),nbits],dtype=int)-1
            inds_spotsN[ires[arg_scores],bits_unfold[arg_scores]]=res_unfolder[arg_scores]
            res_prunedN = np.array([inds_spotsN[imol][codes[icd]] for imol,icd in enumerate(icodesN) if icd>-1])
            scores_prunedN = np.array([scores_bits[imol][codes[icd]] for imol,icd in enumerate(icodesN) if icd>-1])
            icodesN = icodesN[igood]
#        if method == 'top4':
#             codes = self.codes__
#             vals = np.argsort(scores_bits,axis=-1)
#             bcodes = np.sort(vals[:,-4:],axis=-1)
#             base = [nbits**3,nbits**2,nbits**1,nbits**0]
#             bcodes_b = np.sum(bcodes*base,axis=1)
#             codes_b = np.sum(np.sort(codes,axis=-1)*base,axis=1)
#             icodesN = np.zeros(len(bcodes_b),dtype=int)-1
#             for icd,cd in enumerate(codes_b):
#                 icodesN[bcodes_b==cd]=icd
#             bad = np.sum(scores_bits>0,axis=-1)<4
#             icodesN[bad]=-1
#             igood = np.where(icodesN>-1)[0]
#             inds_spotsN =  np.zeros([len(res),nbits],dtype=int)-1
#             inds_spotsN[ires[arg_scores],bits_unfold[arg_scores]]=res_unfolder[arg_scores]
#             res_prunedN = np.array([inds_spotsN[imol][codes[icd]] for imol,icd in enumerate(icodesN) if icd>-1])
#             scores_prunedN = np.array([scores_bits[imol][codes[icd]] for imol,icd in enumerate(icodesN) if icd>-1])
#             icodesN = icodesN[igood]
        elif method == 'dot':
            icodesN = np.argmax(np.dot(scores_bits[:],self.codes_01.T),axis=-1)
            inds_spotsN =  np.zeros([len(res),nbits],dtype=int)-1
            inds_spotsN[ires[arg_scores],bits_unfold[arg_scores]]=res_unfolder[arg_scores]
            res_prunedN = np.array([inds_spotsN[imol][codes[icd]] for imol,icd in enumerate(icodesN) if icd>-1])
            scores_prunedN = np.array([scores_bits[imol][codes[icd]] for imol,icd in enumerate(icodesN) if icd>-1])

        print("Computed the decoding:",time.time()-start)
        if False:
            import time
            start = time.time()
            ### should check this
            mean_scores = np.mean(scores_prunedN,axis=-1)
            ordered_mols = np.argsort(mean_scores)[::-1]
            
            keep_mols = []
            visited = np.zeros(len(self.XH))
            for imol in tqdm(ordered_mols):
                r = np.array(res_prunedN[imol])
                r_ = r[r>=0]
                if np.all(visited[r_]==0):
                    keep_mols.append(imol)
                    visited[r_]=1
            keep_mols = np.array(keep_mols)
            self.scores_prunedN = scores_prunedN[keep_mols]
            self.res_prunedN = res_prunedN[keep_mols]
            self.icodesN = icodesN[keep_mols]
            print("Computed best unique assigment:",time.time()-start)
        else:
            self.scores_prunedN = scores_prunedN
            self.res_prunedN = res_prunedN
            self.icodesN = icodesN
        
        

        self.XH_pruned = self.XH[self.res_prunedN]
        np.savez_compressed(self.decoded_fl,XH_pruned=self.XH_pruned,icodesN=self.icodesN,gns_names = np.array(self.gns_names))
        #XH_pruned -> 10000000 X 4 X 10 [z,x,y,bk...,corpsf,h,col,bit] 
        #icodesN -> 10000000 index of the decoded molecules in gns_names
        #gns_names
    def load_decoded(self, decoded_fl=None):
        import time
        start= time.time()
        if (decoded_fl==None):
            self.decoded_fl = self.save_folder+os.sep+'decoded_'+self.fov.split('.')[0]+'--'+self.set_+'.npz'
        else:
            self.decoded_fl = decoded_fl
        self.XH_pruned = np.load(self.decoded_fl)['XH_pruned']
        self.icodesN = np.load(self.decoded_fl)['icodesN']
        self.gns_names = np.load(self.decoded_fl)['gns_names']
        print("Loaded decoded:",start-time.time())
        

    def get_scores(dec,plt_val=True):
        """Cobines brightness and inter-distance into a single score called dec.scoreA"""
        XH_T = dec.XH_pruned.copy()
        
        bits_ = XH_T[:,:,-1].astype(int)
        bits,ndec = np.unique(bits_,return_counts=True)
        median_brightness = np.median(XH_T[:,:,-3])
        for bit in bits:
            is_bit = bit==bits_
            median_brightness_per_bit = np.median(XH_T[is_bit,-3])
            XH_T[is_bit,-3]=XH_T[is_bit,-3]*(median_brightness/median_brightness_per_bit)
        
        H = np.median(XH_T[...,-3],axis=1)
        Hd = np.std(XH_T[...,-3],axis=1)/H
        
        D = dec.XH_pruned[...,:3]-np.mean(dec.XH_pruned[...,:3],axis=1)[:,np.newaxis]
        D = np.mean(np.linalg.norm(D,axis=-1),axis=-1)
        score = np.array([H,-D])
        scoreA = np.argsort(np.argsort(score,axis=-1),axis=-1)+1
        scoreA = np.sum(np.log(scoreA)-np.log(len(D)),axis=0)
        dec.scoreA = scoreA
        if plt_val:
            bad_igns = [ign for ign,gn in enumerate(dec.gns_names) if 'blank' in gn.lower()]
            good_igns = [ign for ign,gn in enumerate(dec.gns_names) if 'blank' not in gn.lower()]
            is_good_gn = np.in1d(dec.icodesN,good_igns)
            #is_gn = dec.icodesN==(list(dec.gns_names).index('Ptbp1'))
            plt.figure()
            plt.hist(scoreA[is_good_gn],density=True,bins=100,alpha=0.5,label='all genes')
            #plt.hist(scoreA[is_gn],density=True,bins=100,alpha=0.5,label='Ptbp1')
            plt.hist(scoreA[~is_good_gn],density=True,bins=100,alpha=0.5,label='blanks');
            plt.legend()
    
    def get_new_scores(dec,plt_val=True):
            """Evaluate decoding quality based on correlation with Gaussian on raw image (Cor_) and inter-distance (Xdist_)"""
            XH_pruned = dec.XH_pruned.copy()
            
            Cor_ = np.median(XH_pruned[:, :, -6],axis=1)

            Xcms = np.mean(XH_pruned[:,:,:3],axis=1)
            Xdif = XH_pruned[:,:,:3]-Xcms[:,:3][:,np.newaxis]
            Xdists = np.linalg.norm(Xdif,axis=-1)
            Xdists_ = np.max(Xdists,axis=1)
            
            dec.Cor_ = Cor_ 
            dec.Xdist_ = Xdists_
            if plt_val:
                #bad_igns = [ign for ign,gn in enumerate(dec.gns_names) if 'blank' in gn.lower()]
                good_igns = [ign for ign,gn in enumerate(dec.gns_names) if 'blank' not in gn.lower()]
                is_good_gn = np.in1d(dec.icodesN,good_igns)
                #is_gn = dec.icodesN==(list(dec.gns_names).index('Ptbp1'))
                plt.figure()
                plt.hist(Cor_[is_good_gn],density=True,bins=100,alpha=0.5,label='all genes')
                plt.hist(Cor_[~is_good_gn],density=True,bins=100,alpha=0.5,label='blanks');
                plt.title('Raw Correlation')
                plt.legend() 
                plt.figure()
                plt.hist(Xdists_[is_good_gn],density=True,bins=100,alpha=0.5,label='all genes')
                plt.hist(Xdists_[~is_good_gn],density=True,bins=100,alpha=0.5,label='blanks');
                plt.title('Distance to center')
                plt.legend()                    
            
    def plot_1gene(self,gene='Gad1',viewer = None,per_max=97,smax=20,smin=1):
        icodesN,XH_pruned = self.icodesN,self.XH_pruned
        Cor_=self.Cor_
        Xdist_ = self.Xdist_
        
        th=self.th
        gns_names = list(self.gns_names)
        icodesf = icodesN
        Xcms = np.mean(XH_pruned,axis=1)
        H = Xcms[:,-3]
        X = Xcms[:,:3]
        size = smin+np.clip(H/np.percentile(H,per_max),0,1)*(smax-smin)
        
        if viewer is None:
            import napari
            viewer = napari.Viewer()

        icode = gns_names.index(gene)
        is_code = icode==icodesf
        viewer.add_points(X[is_code],size=size[is_code],face_color='r',name=gene)

        is_gn = self.icodesN==(list(self.gns_names).index(gene))
        keep_gn = Cor_[is_gn]>th[0] & Xdist_[is_gn]<th[1]
        Xcms = np.mean(self.XH_pruned,axis=1)
        viewer.add_points(Xcms[is_gn][keep_gn][:,:3],size=10,face_color='g',name=gene)
        return viewer
    def plot_multigenes(self,genes=['Gad1','Sox9'],colors=['r','g','b','m','c','y','w'],smin=3,smax=10,viewer = None,
                        drift=[0,0,0],resc=[1,1,1],per_max=100,is2d=False,blending='translucent'):
        icodesN,XH_pruned = self.icodesN,self.XH_pruned
        Cor_=self.Cor_
        Xdist_ = self.Xdist_
        
        th=self.th
        gns_names = list(self.gns_names)
        
        Xcms = np.mean(XH_pruned,axis=1)
        keep = (Cor_>th[0]) & (Xdist_< th[1])
       
        if is2d:
            X = (Xcms[:,1:3][keep]-drift[1:])/resc[1:]
        else:
            X = (Xcms[:,:3][keep]-drift)/resc  
        H = Cor_[keep] - Xdist_[keep]
        H -= np.min(H)
        icodesf = icodesN[keep]
        size = smin+np.clip(H/np.percentile(H,per_max),0,1)*(smax-smin)
        
        if viewer is None:
            import napari
            viewer = napari.Viewer()
        for igene,gene in enumerate(genes):
            color= colors[igene%len(colors)]
            icode = gns_names.index(gene)
            is_code = icode==icodesf
            viewer.add_points(X[is_code],size=size[is_code],face_color=np.array([0,0,0,0]),edge_color=color,name=gene,blending=blending)

        return viewer    
        

    def compute_z_distortion(dec,th=0.4):
        def get_closest_inds(zsu,bad):
            from scipy.spatial.distance import cdist
            zsu_ = zsu.copy()[:,np.newaxis].astype(float)
            zsu__ = zsu_.copy()
            zsu__[bad]=np.inf
            M = cdist(zsu_,zsu__)
            inds = np.argmin(M,axis=-1)
            return inds[bad]
        iblank = [ign for ign,gn in enumerate(dec.gns_names) if 'blank' in gn]
        keep = ~np.in1d(dec.icodesN,iblank)
        Cor_ = np.median(dec.XH_pruned[:, :, -6],axis=1)
        keep&=Cor_>th
        dec.XH_good = dec.XH_pruned[keep]
        #dec.XH_good = dec.XH_pruned[dec.scoreA>th]
        import scipy.ndimage as nd
        XHG = dec.XH_good.copy()

        RG = XHG[:,:,-1].astype(int)
        iRs = np.unique(RG)
        ## iterate mean field
        for iit in range(4):## number of iterations for mean field
            direc = [1,-1][iit%2]
            for iR in tqdm(iRs[::direc]):
                has_iR = np.any(RG==iR,axis=1)
                XHGiR = XHG[has_iR]
                RGiR  = XHGiR[...,-1].astype(int)
                XHFinR = XHGiR.copy()
                XHFiR = XHGiR.copy()
                is_subiR =(RGiR==iR) 
                XHFiR[~is_subiR]=np.nan
                XHFinR[is_subiR]=np.nan
                drift = np.nanmean(XHFiR[:,:,:3],axis=1)-np.nanmean(XHFinR[:,:,:3],axis=1)
                X = np.nanmean(XHFiR[:,:,:3],axis=1)
                z = X[:,0].astype(int)

                uXi,cuXi = np.unique(z,return_counts=True)
                mn = nd.mean(drift[:,0],z,uXi)
                # construct the bucket to hold the array
                elems = np.zeros(np.max(uXi)+1)
                elems[uXi]=mn

                drift_ = XHG[has_iR]*0
                drift_[is_subiR,0] = elems[z]
                XHG[has_iR] = XHG[has_iR]-drift_
                
        ### extract the best z for each bit    
        zs = dec.XH_good[:,:,0].astype(int)
        zsu = np.unique(dec.XH[:,0].astype(int))
        m,M = np.min(zsu),np.max(zsu)
        zsu = np.arange(M-m+1).astype(int)+m
        dif = (dec.XH_good[:,:,0]-XHG[:,:,0])
        zfs = []
        for iR in iRs:
            zs_ = zs[RG==iR]
            d = dif[RG==iR]
            zf = nd.mean(d,zs_,zsu)
            nf = nd.sum(np.ones_like(d),zs_,zsu)
            bad = (nf<5)
            inds = get_closest_inds(zsu,bad)    
            zf[bad]=zf[inds]
            #plt.plot(zf)
            zfs.append(zf)
        zfs = np.array(zfs)
        zfs[np.isnan(zfs)] = 0 #set invalid bit correction to 0
        dec.zsu = zsu
        dec.zfs = zfs
        return zsu,zfs
        
    def apply_z_distortion_onXH(dec):
        #if not getattr(dec,'has_z_distortion',False):
        zsu,zfs=dec.zsu,dec.zfs
        Rs = dec.XH[:,-1].astype(int)
        iRs = np.unique(Rs)
        print('Applying z correction for self.XH')
        for iR in tqdm(iRs):
            is_iR = Rs==iR
            z_corr = np.interp(dec.XH[is_iR,0],zsu,zfs[iR])
            dec.XH[is_iR,0]-=z_corr
         #   dec.has_z_distortion=True
        
    def to_Xi(dec,X):
        Xr = np.round(X[...,:3]/dec.resc).astype(int)-dec.m0
        max_ = dec.drift0.shape
        Xi = Xr[:,-1]+Xr[:,-2]*max_[-1]+Xr[:,-3]*max_[-1]*max_[-2]
        return Xi
    def apply_corr(dec,XHG):
        XHG_ = XHG.copy()
        Rs = XHG[...,-1].astype(int)
        for iR in dec.dic_correction:
            isR = Rs==iR
            X = XHG[isR,:3]
            XiR = dec.to_Xi(X)
            MuXi = np.max(dec.uXi)+1
            vals = np.zeros([MuXi,3])
            vals[dec.uXi]=dec.dic_correction[iR].T
            XHG_[isR,:3]=XHG[isR,:3]-vals[XiR]
        return XHG_
    def compute_xyz_distortion(dec,th=0.4,resc=100,min_pts=5):
        #resc = 100
        dec.resc = resc
        XiA = np.round(dec.XH[:,:3]/resc).astype(int)
        m0 = np.min(XiA,axis=0)
        M = np.max(XiA,axis=0)
        drift0 = np.zeros(M-m0+1)
        dec.drift0=drift0
        dec.m0=m0
        Xr_ = XiA-m0
        max_ = dec.drift0.shape
        uXi = np.unique(Xr_[:,-1]+Xr_[:,-2]*max_[-1]+Xr_[:,-3]*max_[-1]*max_[-2])
        dec.uXi = uXi
        
        iblank = [ign for ign,gn in enumerate(dec.gns_names) if 'blank' in gn]
        keep = ~np.in1d(dec.icodesN,iblank)
        Cor_ = np.median(dec.XH_pruned[:, :, -6],axis=1)
        keep&=Cor_>th
        dec.XH_good = dec.XH_pruned[keep]
        
        #dec.XH_good = dec.XH_pruned[dec.scoreA>th]
        
        import scipy.ndimage as nd
        XHG = dec.XH_good.copy()

        RG = XHG[:,:,-1].astype(int)
        iRs = np.unique(RG)
        direc = 1
        dec.dic_correction={}
        ## iterate mean field
        for iit in range(6):
            direc = [1,-1][iit%2]
            for iR in tqdm(iRs[::direc]):
                has_iR = np.any(RG==iR,axis=1)
                XHGiR = XHG[has_iR]
                RGiR  = XHGiR[...,-1].astype(int)
                XHFinR = XHGiR.copy()
                XHFiR = XHGiR.copy()
                is_subiR =(RGiR==iR) 
                XHFiR[~is_subiR]=np.nan
                XHFinR[is_subiR]=np.nan
                drift = np.nanmean(XHFiR[:,:,:3],axis=1)-np.nanmean(XHFinR[:,:,:3],axis=1)
                X = np.nanmean(XHFiR[:,:,:3],axis=1)

                Xr = np.round(X/resc).astype(int)-m0
                max_ = dec.drift0.shape
                Xi = Xr[:,-1]+Xr[:,-2]*max_[-1]+Xr[:,-3]*max_[-1]*max_[-2]

                mns = []
                for id_ in range(drift.shape[-1]):
                    mn = nd.mean(drift[:,id_],Xi,uXi)
                    sm = nd.sum(np.ones_like(Xi),Xi,uXi)
                    mn[sm<min_pts]=np.nan
                    uX = np.array(np.unravel_index(uXi,max_)).T
                    inds = get_closest_nan(uX,np.isnan(mn))
                    mn[np.isnan(mn)] = mn[inds]
                    mns.append(mn)
                mns = np.array(mns)
                mns[np.isnan(mns)] = 0 #for special case where all mns are nan
                dec.dic_correction[iR]=mns
                dec.uX = uX
                if False:
                    z_,x_,y_ = np.unravel_index(uXi,max_)
                    plt.figure()
                    plt.scatter(x_,y_,c=mn)
                    plt.axis('equal')
                    plt.colorbar()
            XHG = dec.apply_corr(XHG)
            dec.XHG=XHG
            
    def get_correction(dec):
        import scipy.ndimage as nd
        dec.uX = np.array(np.unravel_index(dec.uXi,dec.drift0.shape)).T
        for iR in dec.dic_correction:
            is_iR = dec.XH_good[...,-1]==iR
            drift = dec.XH_good[is_iR,:3]-dec.XHG[is_iR,:3]

            Xi = dec.to_Xi(dec.XH_good[is_iR,:3])
            #nd.mean(drift[:,:],Xi,dec.uXi)
            #inds = get_closest_nan(dec.uX,np.isnan(mn))
            mns = []
            for id_ in range(3):
                sn = nd.sum(np.ones_like(drift[:,id_]),Xi,dec.uXi)
                mn = nd.mean(drift[:,id_],Xi,dec.uXi)
                bad = sn<5
                inds = get_closest_nan(dec.uX,bad)
                mn[bad]=mn[inds]
                mns.append(mn)
            mns = np.array(mns)
            dec.dic_correction[iR]=mns
    
    def get_drift(self):
        self.dic_drift = {}
        drift_fl = self.analysis_folder + os.sep + 'drift_correction' + os.sep + 'drift_correction_Point' + self.ifov +'.npz'
        rtxyz = np.load(drift_fl,allow_pickle=True)['rtxyz']
        fls = np.load(drift_fl)['fls']
        for ifl,fl in enumerate(fls):
            self.dic_drift[get_H_raw(fl)] = -np.array(rtxyz[ifl])

    def get_drift_temp(self):
        with open(self.analysis_folder + os.sep + 'drift_correction' + os.sep + 'drift_correction_Point' + self.ifov +'.npz', 'rb') as f:
            self.dic_drift = pickle.load(f)
            
    def apply_distortion_correction(self):
        """
        This modifies self.XH to add the correction for distortion (and drift) for each hybe
        """
        fls_dist = glob.glob(self.analysis_folder+os.sep+'distortion\*.npy') #one file for each fov and hyb
        fls_dist_ = np.sort([fl for fl in fls_dist if get_ifov(fl)==self.ifov and self.set_ in os.path.basename(fl)])
        self.dic_fl_distortion = {get_iH_npy(fl):fl for fl in fls_dist_} #dictonary with hyb: distortion file
        self.dic_pair = {}
        for iH in range(len(self.keepH)): #for each hyb
            Xf1,Xf2 = [],[] #initialise
            fl = self.dic_fl_distortion.get(iH,None) #get the filename with the distortions for this hyb
            if fl is not None:
                Xf1,Xf2 = np.load(fl) #load the file
            for icol in range(self.ncols): #for each color
                self.dic_pair[iH*self.ncols+icol]=[Xf1,Xf2] #dictionary with bit: 

        if not hasattr(self,'XH_'):
            self.XH_ = self.XH.copy()
        self.XH = self.XH_.copy()
        for iR in tqdm(np.unique(self.XH[:,-1])):
            IR = int(iR)
            Xf1,Xf2= self.dic_pair[IR]
            Rs = self.XH[:,-1]
            keep = Rs==IR
            X = self.XH[keep,:3]
            if len(Xf1):
                XT = get_Xwarp(X,Xf2,Xf1-Xf2,nneigh=50,sgaus=100)
            else:
                XT = X-self.drift_set[IR]
            self.XH[keep,:3] = XT
    
    def apply_distortion_correction_v2(self, hyb_1st_bit = 1):
        """
        This modifies self.XH to add the correction for distortion (and drift) for each hybe
        """
        fls_dist = glob.glob(self.analysis_folder+os.sep+'drift_distortion\*.npz') #one file for each fov and hyb
        fls_dist_ = np.sort([fl for fl in fls_dist if get_ifov(fl)==self.ifov  in os.path.basename(fl)])
        self.dic_fl_distortion = {get_iH_npz(fl):fl for fl in fls_dist_} #dictonary with hyb: distortion file
        self.dic_pair = {}
        for iH in self.keepH: #for each hyb
            Xf1,Xf2 = [],[] #initialise
            fl = self.dic_fl_distortion.get(iH,None) #get the filename with the distortions for this hyb
            if fl is not None:
                Xf1 = np.load(fl)['P1f'] #load the file
                Xf2 = np.load(fl)['P2f']
            for icol in range(self.ncols): #for each color
                self.dic_pair[(iH - hyb_1st_bit)*self.ncols+icol]=[Xf1,Xf2] #dictionary with bit: 

        #if not hasattr(self,'XH_'):
        #    self.XH_ = self.XH.copy()
        #self.XH = self.XH_.copy()
        for iR in tqdm(np.unique(self.XH[:,-1])):
            IR = int(iR)
            Xf1,Xf2= self.dic_pair[IR]
            Rs = self.XH[:,-1]
            keep = Rs==IR
            X = self.XH[keep,:3]
            if len(Xf1):
                XT = get_Xwarp(X,Xf2,Xf1-Xf2,nneigh=10,sgaus=20)
            else:
                XT = X-self.drift_set[IR]
            self.XH[keep,:3] = XT
            
            

    def pick_best_brightness(self,nUR_cutoff = 3,resample = 10000):
        """Pick the best code based on normalized brightness for a subsample of <resample> molecule-clusters.
        This is used to find the statistics for the distance and brightness distribution"""
        XH = self.XH # all called spots: x,y,z,brightness,color,bit
        res =self.res # list of spots and the spots in their vicinity from get_inters
        codes = self.codes__ # bits used by each code


        RS = XH[:,-1].astype(int) # the bit of each deceted spot
        HS = XH[:,-3] #the intensity of each detected spot
        colS = XH[:,-2].astype(int) # the channel of origin of each spot
        colSu = np.unique(colS) # all channels used
        #Ru_ = Ru[3] 
        meds_col = np.array([np.median(HS[colS==cu]) for cu in colSu]) #the median intensity of spots called in each channel
        self.meds_col = meds_col 
        HN = HS/meds_col[colS] #the intensity of each spot normalised by the median of all spots in that channel

        ncodes = len(codes) #number of codes

        bucket = np.zeros(ncodes) # one row per code
        nbits_per_code = np.array([len(cd) for cd in codes]) #number of bits per code (always 4)

        icodes = []
        res_pruned = []

        Nres = len(res) # number of spots
        resamp = max(Nres//resample,1) # N for each Nth element to get resampling factor

        for r in tqdm(res[::resamp]): #for each Nth spot 
            scores = HN[r] # the median normalised spot intensity

            dic_sc = {r:sc for r,sc in zip(r,scores)} # dict: for each spot the corresponding median normalised intensity
            isort = np.argsort(scores) # indeces of colocalising spots ordered by their norm. intensity
            r = np.array(r)[isort] # spots colocalising with spot r ordered by their norm. intensity
            scores = scores[isort] # norm. spot intensity of close spots orderd by intensity
            R = RS[r] # the bits of the colocalising spots
            dic_u = {R_:r_ for r_,R_ in zip(r,R)} # dic of bit of colocalising spot to colocalising spot
            if len(dic_u)>=nUR_cutoff: # if the number of colocalising spots is at least the threshold
                bucket_ = np.zeros(ncodes) #zero for each code
                for R_ in dic_u: # for each bit of colocalising spot
                    if R_ in self.dic_bit_to_code: # if the bit is used in the codebook
                        icds = self.dic_bit_to_code[R_] # codes that uses that bit
                        bucket_[icds]+=dic_sc[dic_u[R_]]  # add the norm. intensity to codes that use that bit
                bucket_/=nbits_per_code # average norm. intensity of spots for each code
                best_code = np.argmax(bucket_) # code with the highest average norm. intensity across bits
                icodes.append(best_code) # for each spot with at least nUR_cutoff colocalising spots the code it fits to best 
                res_pruned.append([dic_u.get(R_,-1) for R_ in codes[best_code]]) # for each bit in the best code get the colocalising spots, if there is no colocalising spot 
                
        self.icodes = icodes # for each spot with at least nUR_cutoff colocalising spots the code with the highest norm. intensity across bits
        self.res_pruned = res_pruned # for each spot the colocalising spots that are in a bit in the code with the best fit
                                     # res_pruned has -1 where there was no spot detected in the bits of the best code

        
    def get_brightness_distance_distribution(self):
        XH = self.XH
        all_dists = []
        all_brightness = []
        for rs,icd in zip(tqdm(self.res_pruned),self.icodes):
            if icd in self.good_gns:
                rs = np.array(rs)
                rs = rs[rs>-1]
                X = XH[rs]
                h = X[:,-3]
                col = X[:,-2].astype(int)
                dists_ = np.linalg.norm(np.mean(X[:,:3],axis=0)-X[:,:3],axis=-1)
                all_dists.extend(dists_)
                all_brightness.extend(h/self.meds_col[col])
        all_brightness = np.sort(all_brightness)[:,np.newaxis]
        all_dists = np.sort(all_dists)[::-1,np.newaxis]
        self.tree_br = cKDTree(all_brightness)
        self.tree_dist = cKDTree(all_dists)
    def get_score_brightness(self,x):
        return (self.tree_br.query(x[:,np.newaxis])[-1]+1)/(len(self.tree_br.data)+1)
    def get_score_distance(self,x):
        return (self.tree_dist.query(x[:,np.newaxis])[-1]+1)/(len(self.tree_dist.data)+1)
    def pick_best_score(self,nUR_cutoff = 3,resample=1):
        """Pick the best code for each molecular cluster based on the fisher statistics 
        for the distance and brightness distribution"""
        self.get_brightness_distance_distribution()
        res =self.res
        XH = self.XH
        codes = self.codes__

        RS = XH[:,-1].astype(int) #bit info
        HS = XH[:,-3] #brightness info
        colS = XH[:,-2].astype(int) #color info
        colSu = np.unique(colS)  #colors used
        meds_col = np.array([np.median(HS[colS==cu]) for cu in colSu]) #median intensity per channel

        self.HN = HS/meds_col[colS] # normalize brighnesses per each color

        ncodes = len(codes) #number of codes in codebook

        bucket = np.zeros(ncodes)
        nbits_per_code = np.array([len(cd) for cd in codes])

        icodes = [] 
        res_pruned = []
        scores_pruned = []
        spots_used = []
        for r in tqdm(res[::resample]): # for each detected spot
            hn = self.HN[r] #all brightnesses for i.e 4 spots within 2 pixels of another spot
            X = XH[r,:3] #positions for them
            dn = np.linalg.norm(X-np.mean(X,axis=0),axis=-1) #distance from the centroid
            sc_dist = self.get_score_distance(dn) # scoring distances from centroid
            sc_br = self.get_score_brightness(hn) # scoring brihnesses
            scores = sc_dist*sc_br ## combine scores by multiplying
            dic_sc = {r:sc for r,sc in zip(r,scores)} # dict with spot: score for each spot in vicinity
            isort = np.argsort(scores) # order of scores 
            r = np.array(r)[isort] #ordering the spots in the vicinity by score
            #scores = scores[isort]
            R = RS[r] #bits of spots in vicinity (ordered by score)
            dic_u = {R_:r_ for r_,R_ in zip(r,R)}
            if len(dic_u)>=nUR_cutoff: # if number of co-localised spot for that spot is at least the cut-off
                bucket_ = np.zeros(ncodes)
                for R_ in dic_u:
                    if R_ in self.dic_bit_to_code:
                        icds = self.dic_bit_to_code[R_]
                        bucket_[icds]+=dic_sc[dic_u[R_]]
                bucket_/=nbits_per_code
                best_code = np.argmax(bucket_)
                icodes.append(best_code)
                rf = [dic_u.get(R_,-1) for R_ in codes[best_code]]
                res_pruned.append(rf)
                scores_pruned.append([dic_sc.get(r_,-1000) for r_ in rf])
        self.res_pruned = np.array(res_pruned) # list of lists but with only the subsampled points corresponding to the best match (overwrites res_pruned defined by pick_best_brightness)
        self.icodes = np.array(icodes) # index in the codebook of best predicted spot (overwrites icodes defined by pick_best_brightness)
        self.scores_pruned = np.array(scores_pruned) # this 
        #X1f,X2f = self.dic_pair.get(0,[[[0,0,0]],[[0,0,0]]])
        #driftf = np.mean(np.array(X1f)-X2f,axis=0)
        X = self.XH[:,:3]#-driftf #adjust the spot location with the drift
        res_ = np.array(self.res_pruned)
        keep = (res_>=0)[...,np.newaxis]
        self.Xcms = np.sum(X[res_]*keep,axis=1)/np.sum(keep,axis=1) # center of masses
    def load_segmentation(self):
        dapi_fl  = [fl for fl in self.dapi_fls if self.set_ in os.path.basename(fl) and self.fov in os.path.basename(fl)][0]
        im_segm = np.load(dapi_fl)['segm']
        shape = np.load(dapi_fl)['shape']
        objs = ndimage.find_objects(im_segm)
        self.icells,self.cellcaps_,self.cellcaps = [],[],[]
        if len(objs): 
            self.icells,self.cellcaps_,self.cellcaps = zip(*[(icell+1,pair,resize_slice(pair,im_segm.shape,shape)) for icell,pair in enumerate(ndimage.find_objects(im_segm)) if pair is not None])
        #self.icells,self.cellcaps_,self.cellcaps = zip(*[(icell+1,pair,resize_slice(pair,im_segm.shape,shape)) for icell,pair in enumerate(ndimage.find_objects(im_segm)) if pair is not None])
        #self.icells = [icell+1 for icell,pair in enumerate(ndimage.find_objects(im_segm)) if pair is not None]
        cm_cells = np.array([[(sl.start+sl.stop)//2 for sl in sls]for sls in self.cellcaps])
        #self.cellcaps=cellcaps
        self.cm_cells=cm_cells
        self.im_segm = im_segm
        self.shapesm = self.im_segm.shape
        self.shape = shape
        self.vols = [int(np.sum(self.im_segm[cell_cap]==icell)*np.prod(self.shape/self.shapesm)) for icell,cell_cap in zip(self.icells,self.cellcaps_)]
    def get_ptb_aso(self,icol_aso=0,icol_ptb=1,th_cor_ptb=0.5,th_ptb=2500):
        """
        This gets the ptb counts and the average aso level per cell assuming the data is already fitted
        use as:
        
        from ioMicro import *

        dec = decoder(analysis_folder = r'\\132.239.200.33\Raw_data\DCBB_MER250__12_2_2022_Analysis')
        for set_ in ['set1','set2','set3','set4']:
            for ifov in tqdm(np.arange(400)):
                ### computation
                dec.get_set_ifov(ifov=ifov,set_=set_,keepH = [1,2,3,4,5,6,7,8],ncols=3)
                dec.save_file_cts_ptb = dec.save_file_cts.replace('.npz','_ptb-aso.npz')
                if not os.path.exists(dec.save_file_cts_ptb) and not dec.out_of_range:
                    dec.load_segmentation()
                    dec.get_ptb_aso(icol_aso=0,icol_ptb=1,th_cor_ptb=0.5,th_ptb=2500)
                    np.savez(dec.save_file_cts_ptb,aso_mean=dec.aso_mean,cm_cells=dec.cm_cells)
        
        """
    
    
        self.dic_ptb = {get_ifov(fl):fl for fl in self.files_set if 'ptb' in os.path.basename(fl).lower()}
        self.ptb_fl = self.dic_ptb.get(self.ifov,None)
        self.dic_aso = {get_ifov(fl):fl for fl in self.files_set if 'aso' in os.path.basename(fl).lower()}
        self.aso_fl = self.dic_aso.get(self.ifov,None)
        
        #load ptb file and correct drift
        Xhs,dic_drift = pickle.load(open(self.ptb_fl,'rb'))
        Xh = Xhs[icol_ptb]
        self.txyz=dic_drift['txyz']
        Xh[:,:3] = Xh[:,:3]-dic_drift['txyz']

        #filter based on correlation with PSF and brightness
        keep = Xh[:,-2]>th_cor_ptb
        keep &= Xh[:,-1]>th_ptb
        Xh = Xh[keep]
        
        #plotting
        if False:
            from matplotlib import cm as cmap
            import napari
            cts_ = np.clip(Xh[:,-1],0,15000)
            cts_ = cts_/np.max(cts_)
            sizes = 1+cts_*5
            colors = cmap.coolwarm(cts_)
            napari.view_points(Xh[:,1:3],face_color=colors,size=sizes)

        ### count per cell
        Xcms = Xh[:,:3]
        Xred = np.round((Xcms/self.shape)*self.shapesm).astype(int)
        good = ~np.any((Xred>=self.shapesm)|(Xred<0),axis=-1)
        Xred = Xred[good]
        icells,cts = np.unique(self.im_segm[tuple(Xred.T)],return_counts=True)
        dic_cts = {icell:ct for icell,ct in zip(icells,cts)}
        ctsf = [dic_cts.get(icell,0) for icell in self.icells]
        self.ptbp_cts = ctsf

        if False:
            import napari
            viewer = napari.view_points(Xred,size=2)
            viewer.add_labels(self.im_segm)

        #do same for aso
        Xhs,dic_drift = pickle.load(open(self.aso_fl,'rb'))
        Xh = Xhs[0]
        self.txyz=dic_drift['txyz']
        Xh[:,:3] = Xh[:,:3]-dic_drift['txyz']
        Xcms = Xh[:,:3]
        Xred = np.round((Xcms/self.shape)*self.shapesm).astype(int)
        good = ~np.any((Xred>=self.shapesm)|(Xred<0),axis=-1)
        Xred = Xred[good]
        labels = self.im_segm[tuple(Xred.T)]
        Xh = Xh[good]
        from scipy import ndimage
        self.aso_mean = ndimage.mean(Xh[:,5],labels=labels,index=self.icells)

    def get_intersV2(self,dinstance_th=2,enforce_color=True):
        """Get an initial intersection of points and save in self.res_unfolder,lens"""


        res =[]
        if enforce_color:
            icols = self.XH[:,-2].astype(int)
            XH = self.XH
            for icol in tqdm(np.unique(icols)):
                inds = np.where(icols==icol)[0]
                Xs = XH[inds,:3]
                Ts = cKDTree(Xs)
                res_ = Ts.query_ball_tree(Ts,dinstance_th)
                res += [inds[r] for r in res_]
        else:
            XH = self.XH
            Xs = XH[:,:3]
            Ts = cKDTree(Xs)
            res = Ts.query_ball_tree(Ts,dinstance_th)
        print("Calculating lengths of clusters...")
        lens = np.array(list(map(len,res)))
        Mlen = np.max(lens)
        print("Unfolding indexes...")
        res_unfolder = np.concatenate(res)
        self.res_unfolder=res_unfolder
        self.lens=lens
    
    def get_icodesV2(dec,nmin_bits=4,delta_bits=None,iH=-3,redo=False,norm_brightness=False,nbits=24,is_unique=False):
        """
        This is an updated version that includes uniqueness
        """
        
        import time
        start = time.time()
        lens = dec.lens
        res_unfolder = dec.res_unfolder
        Mlen = np.max(lens)
        print("Calculating indexes within cluster...")
        res_is = np.tile(np.arange(Mlen), len(lens))
        res_is = res_is[res_is < np.repeat(lens, Mlen)]
        print("Calculating index of molecule...")
        ires = np.repeat(np.arange(len(lens)), lens)
        #r0 = np.array([r[0] for r in res for r_ in r])
        print("Calculating index of first molecule...")
        r0i = np.concatenate([[0],np.cumsum(lens)])[:-1]
        r0 = res_unfolder[np.repeat(r0i, lens)]
        print("Total time unfolded molecules:",time.time()-start)

        ### torch
        ires = torch.from_numpy(ires.astype(np.int64))
        res_unfolder = torch.from_numpy(res_unfolder.astype(np.int64))
        res_is = torch.from_numpy(res_is.astype(np.int64))
        
        
        
        ### get score for brightness 
        def get_scoresH():
            H = torch.from_numpy(dec.XH[:,-3])
            Hlog = H#np.log(H)
            mnH = Hlog.mean()
            stdH = Hlog.std()
            distribution = torch.distributions.Normal(mnH, stdH)
            scoreH = distribution.cdf(Hlog)
            return scoreH[res_unfolder]
        ### get score for inter-distance between molecules
        def get_scoresD():
            X = dec.XH[:,:3]
            XT = torch.from_numpy(X)
            XD = XT[res_unfolder]-XT[r0]
            meanD = -torch.mean(torch.abs(XD),axis=-1)
            distribution = torch.distributions.Normal(meanD.mean(), meanD.std())
            scoreD = distribution.cdf(meanD)
            return scoreD
        def get_combined_scores():
            scoreH = get_scoresH()
            scoreD = get_scoresD()
            ### combine scores. Note this score is for all the molecules un-ravelled from their clusters
            scoreF = scoreD*scoreH
            return scoreF
        
        import time
        start = time.time()
        print("Computing score...")
        if iH is None:
            scoreF = get_combined_scores()
        else:
            scoreF = torch.from_numpy(dec.XH[:,iH])[res_unfolder]
        print("Total time computing score:",time.time()-start)

        ### organize molecules in blocks for each cluster
        def get_asort_scores():
            val = torch.max(scoreF)+2
            scoreClu = torch.zeros([len(lens),Mlen],dtype=torch.float64)+val
            scoreClu[ires,res_is]=scoreF
            asort = scoreClu.argsort(-1)
            scoreClu = torch.gather(scoreClu,dim=-1,index=asort)
            scoresF2 = scoreClu[scoreClu<val-1]
            return asort,scoresF2
        def get_reorder(x,val=-1):
            if type(x) is not torch.Tensor:
                x = torch.from_numpy(np.array(x))
            xClu = torch.zeros([len(lens),Mlen],dtype=x.dtype)+val
            xClu[ires,res_is] = x
            xClu = torch.gather(xClu,dim=-1,index=asort)
            xf = xClu[xClu>val]
            return xf

        import time
        start = time.time()
        print("Computing sorting...")
        asort,scoresF2 = get_asort_scores()
        res_unfolder2 = get_reorder(res_unfolder,val=-1)
        del asort
        del scoreF
        print("Total time sorting molecules by score:",time.time()-start)
        
        
        
        import time
        start = time.time()
        print("Finding best bits per molecules...")

        Rs = dec.XH[:,-1].astype(np.int64)
        Rs = torch.from_numpy(Rs)
        Rs_U = Rs[res_unfolder2]

        score_bits = torch.zeros([len(lens),nbits],dtype=scoresF2.dtype)-1
        score_bits[ires,Rs_U]=scoresF2

        
        #codes_lib = torch.from_numpy(np.array(dec.codes__))
        codes_lib = torch.from_numpy(np.array(dec.codes__,dtype=np.int64))
        
        if is_unique:
            codes_lib_01 = torch.zeros([len(codes_lib),nbits],dtype=score_bits.dtype)
            for icd,cd in enumerate(codes_lib):
                codes_lib_01[icd,cd]=1

            print("Finding best code...")
            batch = 10000
            icodes_best = torch.zeros(len(score_bits),dtype=torch.int64)
            from tqdm import tqdm
            for i in tqdm(range((len(score_bits)//batch)+1)):
                score_bits_ = score_bits[i*batch:(i+1)*batch]
                if len(score_bits_)>0:
                    icodes_best[i*batch:(i+1)*batch] = torch.argmax(torch.matmul(score_bits_,codes_lib_01.T),dim=-1)
        
            if delta_bits is not None:
                argsort_bits = torch.argsort(score_bits,dim=-1,descending=True)[:,:(nmin_bits+delta_bits)]
                score_bits_ = score_bits*0
                score_bits_.scatter_(1, argsort_bits, 1)
                keep_all_bits = torch.all(score_bits_.gather(1,codes_lib[icodes_best])>0.5,-1)
            else:
                keep_all_bits = torch.all(score_bits.gather(1,codes_lib[icodes_best])>=0,-1)
            
            score_bits = score_bits[keep_all_bits]
            icodes_best_ = icodes_best[keep_all_bits]
            icodesN=icodes_best_
            
            indexMols_ = torch.zeros([len(lens),nbits],dtype=res_unfolder2.dtype)-1
            indexMols_[ires,Rs_U]=res_unfolder2
            indexMols_ = indexMols_[keep_all_bits]
            indexMols_ = indexMols_.gather(1,codes_lib[icodes_best_])
            # make unique
            indexMols_,rinvMols = get_unique_ordered(indexMols_)
            icodesN = icodesN[rinvMols]
        else:
            indexMols_ = torch.zeros([len(lens),nbits],dtype=res_unfolder2.dtype)-1
            indexMols_[ires,Rs_U]=res_unfolder2
            def get_inclusive(imols,code_lib):
                iMol,iScore = torch.where(torch.all(imols[...,code_lib]>0,dim=-1))
                return imols[iMol].gather(1,code_lib[iScore]),iScore
            batch = 10000
            from tqdm import tqdm
            indexMolsF_ = torch.zeros([0,codes_lib.shape[-1]],dtype=torch.int64)
            icodesN = torch.zeros([0],dtype=torch.int64)
            for i in tqdm(range((len(indexMols_)//batch)+1)):
                indexMols__ = indexMols_[i*batch:(i+1)*batch]
                if len(indexMols__)>0:
                    indexMolsF__,icodesN_ = get_inclusive(indexMols__,codes_lib)
                    indexMolsF_ = torch.concatenate([indexMolsF_,indexMolsF__])
                    icodesN = torch.concatenate([icodesN,icodesN_])
            indexMols_ = indexMolsF_
            indexMols_,rinvMols = get_unique_ordered(indexMols_)
            icodesN = icodesN[rinvMols]
        XH = torch.from_numpy(dec.XH)
        XH_pruned = XH[indexMols_]
        
        dec.XH_pruned=XH_pruned.numpy()
        dec.icodesN=icodesN.numpy()
        
        #np.savez_compressed(dec.decoded_fl,XH_pruned=dec.XH_pruned,icodesN=dec.icodesN,gns_names = np.array(dec.gns_names),is_unique=is_unique)
        
        print("Total time best bits per molecule:",time.time()-start)
        
        
    def get_score_withRef(dec,scoresRef,plt_val=False,gene=None,iSs=None,th_min=-np.inf):
        H = np.median(dec.XH_pruned[...,-3],axis=1)
        D = dec.XH_pruned[...,:3]-np.mean(dec.XH_pruned[...,:3],axis=1)[:,np.newaxis]
        D = np.mean(np.linalg.norm(D,axis=-1),axis=-1)
        score = np.array([H,-D]).T
        keep_color = [dec.XH_pruned[:,0,-2]==icol for icol in np.arange(dec.ncols)]
        scoreA = np.zeros(len(H))
        for icol in range(dec.ncols):
            scoresRef_ = scoresRef[icol]
            score_ = score[keep_color[icol]]
            from scipy.spatial import KDTree
            scoreA_ = np.zeros(len(score_))
            if iSs is None: iSs = np.arange(scoresRef_.shape[-1])
            for iS in iSs:
                dist_,inds_ = KDTree(scoresRef_[:,[iS]]).query(score_[:,[iS]])
                scoreA_+=np.log((inds_+1))-np.log(len(scoresRef_))
            scoreA[keep_color[icol]]=scoreA_
        dec.scoreA =scoreA
        if plt_val:
            bad_igns = [ign for ign,gn in enumerate(dec.gns_names) if 'blank' in gn.lower()]
            good_igns = [ign for ign,gn in enumerate(dec.gns_names) if 'blank' not in gn.lower()]
            is_good_gn = np.in1d(dec.icodesN,good_igns)
            
            plt.figure()
            kp = scoreA>th_min
            plt.hist(scoreA[(is_good_gn)&kp],density=True,bins=100,alpha=0.5,label='all genes')
            if gene is not None:
                is_gn = dec.icodesN==(list(dec.gns_names).index(gene))
                plt.hist(scoreA[(is_gn)&kp],density=True,bins=100,alpha=0.5,label=gene)
            plt.hist(scoreA[(~is_good_gn)&kp],density=True,bins=100,alpha=0.5,label='blanks');
            plt.legend()
    def plot_statistics(dec):
        if hasattr(dec,'im_segm_'):
            ncells = len(np.unique(dec.im_segm_))-1
        else:
            ncells = 1
        keep = (dec.Cor_>dec.th[0]) & (dec.Xdist_<dec.th[1])
        icds,ncds = np.unique(dec.icodesN[keep],return_counts=True)
        good_igns = [ign for ign,gn in enumerate(dec.gns_names) if 'blank' not in gn.lower()]
        kp = np.in1d(icds,good_igns)
        ncds = ncds/ncells
        plt.figure()
        plt.xlabel('Genes')
        plt.plot(icds[kp],ncds[kp],label='genes')
        plt.plot(icds[~kp],ncds[~kp],label='blank')
        plt.ylabel('Number of molecules in the fov')
        plt.title(str(np.round(np.mean(ncds[~kp])/np.mean(ncds[kp]),3)))
        plt.legend()
    def apply_fine_drift(dec,plt_val=True,npts=50000):
        bad_igns = [ign for ign,gn in enumerate(dec.gns_names) if 'blank' in gn.lower()]
        good_igns = [ign for ign,gn in enumerate(dec.gns_names) if 'blank' not in gn.lower()]
        is_good_gn = np.in1d(dec.icodesN,good_igns)
        allR = dec.XH_pruned[:,:,-1].astype(int)
        XHG = dec.XH_pruned[is_good_gn]
        
        RG = XHG[:,:,-1].astype(int)
        iRs=np.unique(RG)
        dic_fine_drift = {}
        for iR in tqdm(iRs):
            XHGiR = XHG[np.any(RG==iR,axis=1)]
            RGiR  = XHGiR[...,-1].astype(int)
            mH = np.median(XHGiR[:,:,-3],axis=1)
            XHF = XHGiR[np.argsort(mH)[::-1][:npts]]
            RF  = XHF[...,-1].astype(int)
            XHFinR = XHF.copy()
            XHFiR = XHF.copy()
            XHFiR[~(RF==iR)]=np.nan
            XHFinR[(RF==iR)]=np.nan
            drift = np.mean(np.nanmean(XHFiR[:,:,:3],axis=1)-np.nanmean(XHFinR[:,:,:3],axis=1),axis=0)
            dic_fine_drift[iR]=drift
        drift_arr = np.zeros([np.max(allR)+1,3])
        for iR in iRs:
            drift_arr[iR]=dic_fine_drift[iR]
        if plt_val:
            ncols = len(np.unique(XHG[:,:,-2]))
            X1 = np.array([dic_fine_drift[iR] for iR in iRs[0::ncols]])
            X3 = np.array([dic_fine_drift[iR] for iR in iRs[(ncols-1)::ncols]])

            plt.figure()
            plt.plot(X1[:,0],X3[:,0],'o',label='z-color0-2')
            plt.plot(X1[:,1],X3[:,1],'o',label='x-color0-2')
            plt.plot(X1[:,2],X3[:,2],'o',label='y-color0-2')

            plt.xlabel("Drift estimation color 1 (pixels)")
            plt.ylabel("Drift estimation color 2 (pixels)")
            plt.legend()
        dec.drift_arr = drift_arr
        R = dec.XH_pruned[:,:,-1].astype(int)#
        dec.XH_pruned[:,:,:3] -= drift_arr[R]
        
def load_ct_data(analysis_folder = r'\\132.239.200.33\Raw_data\DCBB_MER250__12_2_2022_Analysis',
                 data_folder = r'\\132.239.200.33\Raw_data\DCBB_MER250__12_2_2022\H1_set*',set_='set1',um_per_pixel = 0.108333,
                tag_cts = 'cts_all',tag_fl = r'\Decoded\*_cts.npz'):
    dic_coords = get_all_pos(analysis_folder = analysis_folder,data_folder =data_folder,set_=set_)

    fls = glob.glob(analysis_folder+os.sep+tag_fl)
    ctM = None
    cm_cells,ifovs = [],[]
    cm_tags = []
    for fl in tqdm(np.sort(fls)):
        if set_ in fl:
            dic = np.load(fl)
            gns_names = dic['gns_names']
            ctM_ = dic[tag_cts]
            cm_cells_ = dic['cm_cells']
            cm_tags_= [[get_ifov(fl)]+list(np.array(cm,dtype=int)) for cm in cm_cells_]
            cm_tags+=cm_tags_
            cm_cells.extend(cm_cells_)
            ifovs += [get_ifov(fl)]*len(cm_cells_)
            if ctM is None: ctM=ctM_ 
            else: ctM = np.concatenate([ctM,ctM_],axis=1)

    ifovs = np.array(ifovs)
    cm_cells = np.array(cm_cells)
    cm_cells = cm_cells[:,1:]
    abs_pos = np.array([dic_coords[ifov] for ifov in ifovs])
    abs_pos = abs_pos[:,::-1]*np.array([1,-1])
    
    cm_cellsf = cm_cells*um_per_pixel+abs_pos
    return ctM,gns_names,cm_cellsf,cm_tags
def load_ct_data_ptb_aso(analysis_folder = r'\\132.239.200.33\Raw_data\DCBB_MER250__12_2_2022_Analysis',
                 data_folder = r'\\132.239.200.33\Raw_data\DCBB_MER250__12_2_2022\H1_set*',set_='set1',um_per_pixel = 0.108333,
                tags_cts = ['ptbp_cts','aso_mean'],tag_fl = r'\Decoded\*_cts_ptb-aso.npz'):
    dic_coords = get_all_pos(analysis_folder = analysis_folder,data_folder =data_folder,set_=set_)

    fls = glob.glob(analysis_folder+os.sep+tag_fl)
    ctM = None
    cm_cells,ifovs = [],[]
    for fl in tqdm(np.sort(fls)):
        if set_ in fl:
            dic = np.load(fl)
            gns_names = tags_cts
            ctM_ = np.array([dic[tag_cts] for tag_cts in tags_cts])
            cm_cells_ = dic['cm_cells']
            cm_cells.extend(cm_cells_)
            ifovs += [get_ifov(fl)]*len(cm_cells_)
            if ctM is None: ctM=ctM_ 
            else: ctM = np.concatenate([ctM,ctM_],axis=1)

    ifovs = np.array(ifovs)
    cm_cells = np.array(cm_cells)
    cm_cells = cm_cells[:,1:]
    abs_pos = np.array([dic_coords[ifov] for ifov in ifovs])
    abs_pos = abs_pos[:,::-1]*np.array([1,-1])
    
    cm_cellsf = cm_cells*um_per_pixel+abs_pos
    return ctM,gns_names,cm_cellsf
def example_rerun():
    #from ioMicro import *
    set_='set1'
    ifov=0
    dec = decoder(analysis_folder = r'\\132.239.200.33\Raw_data\DCBB_MER250__12_2_2022_Analysis')
    for ifov in tqdm(np.arange(320)):
        
        dec.get_set_ifov(ifov=ifov,set_=set_,keepH = [1,2,3,4,5,6,7,8],ncols=3)
        save_fl = dec.save_file_cts.replace('_cts.npz','_ctsV2.npz')
        if not os.path.exists(save_fl):
            dec.load_segmentation()
            dec.load_library()
            dic = np.load(dec.save_file_dec)
            for key in list(dic.keys()):
                setattr(dec,key,dic[key])
            
            ###             perform some refinement
            
            dec.cts_all_pm = dec.get_counts_per_cell(nbad=0)
            dec.cts_all = dec.get_counts_per_cell(nbad=1)
            np.savez(save_fl,
                     cts_all_pm = dec.cts_all_pm,cts_all = dec.cts_all,
                     gns_names=dec.gns_names,cm_cells=dec.cm_cells,vols=dec.vols)
def get_unique_ordered(vals):
        #vals = torch.from_numpy(vals)
        vals,_ = torch.sort(vals,dim=-1)
        del _
        vals,rinv = unique(vals,dim=0)
        return vals,rinv
        
def unique(x, dim=None):
        """Unique elements of x and indices of those unique elements
        https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

        e.g.

        unique(tensor([
            [1, 2, 3],
            [1, 2, 4],
            [1, 2, 3],
            [1, 2, 5]
        ]), dim=0)
        => (tensor([[1, 2, 3],
                    [1, 2, 4],
                    [1, 2, 5]]),
            tensor([0, 1, 3]))
        """
        unique, inverse = torch.unique(
            x, sorted=True, return_inverse=True, dim=dim)
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                            device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        return unique, inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
        
def combine_scoresRef(scoresRef,scoresRefT):
    return [np.sort(np.concatenate([scoresRef[icol],scoresRefT[icol]]),axis=0)
     for icol in np.arange(len(scoresRef))]
def get_score_per_color(dec):
    H = np.median(dec.XH_pruned[...,-3],axis=1)
    D = dec.XH_pruned[...,:3]-np.mean(dec.XH_pruned[...,:3],axis=1)[:,np.newaxis]
    D = np.mean(np.linalg.norm(D,axis=-1),axis=-1)
    score = np.array([H,-D]).T
    score = np.sort(score,axis=0)
    return [score[dec.XH_pruned[:,0,-2]==icol] for icol in np.arange(dec.ncols)]
def full_deconv(im_,s_=500,pad=100,psf=None,parameters={'method': 'wiener', 'beta': 0.001, 'niter': 50},gpu=True,force=False):
    im0=np.zeros_like(im_)
    sx,sy = im_.shape[1:]
    ixys = []
    for ix in np.arange(0,sx,s_):
        for iy in np.arange(0,sy,s_):
            ixys.append([ix,iy])
    
    for ix,iy in tqdm(ixys):#ixys:#tqdm(ixys):
        imsm = im_[:,ix:ix+pad+s_,iy:iy+pad+s_]
        imt = apply_deconv(imsm,psf=psf,parameters=parameters,gpu=gpu,plt_val=False,force=force)
        start_x = ix+pad//2 if ix>0 else 0
        end_x = ix+pad//2+s_
        start_y = iy+pad//2 if iy>0 else 0
        end_y = iy+pad//2+s_
        #print(start_x,end_x,start_y,end_y)
        im0[:,start_x:end_x,start_y:end_y] = imt[:,(start_x-ix):(end_x-ix),(start_y-iy):(end_y-iy)]
    return im0

def _wiener_3d(self, image):
    """Monkey pathc to compute the 3D wiener deconvolution

    Parameters
    ----------
    image: torch.Tensor
        3D image tensor

    Returns
    -------
    torch.Tensor of the 3D deblurred image

    """
    import torch
    from sdeconv.core import SSettings
    from sdeconv.deconv.wiener import pad_3d,laplacian_3d,unpad_3d
    
    image_pad, psf_pad, padding = pad_3d(image, self.psf / torch.sum(self.psf), self.pad)

    fft_source = torch.fft.fftn(image_pad)
    
    container = SSettings.instance()
    if not hasattr(container,'dic_psf'): container.dic_psf = {}
    if container.dic_psf.get('den'+str(fft_source.shape),None) is None:        
        psf_roll = torch.roll(psf_pad, int(-psf_pad.shape[0] / 2), dims=0)
        psf_roll = torch.roll(psf_roll, int(-psf_pad.shape[1] / 2), dims=1)
        psf_roll = torch.roll(psf_roll, int(-psf_pad.shape[2] / 2), dims=2)

        fft_psf = torch.fft.fftn(psf_roll)
        fft_laplacian = torch.fft.fftn(laplacian_3d(image_pad.shape))
        den = fft_psf * torch.conj(fft_psf) + self.beta * fft_laplacian * torch.conj(fft_laplacian)
        
        
        
        container.dic_psf['den'+str(fft_source.shape)] = den
        container.dic_psf['fft_psf'+str(fft_source.shape)] = fft_psf
    else:
        den = container.dic_psf['den'+str(fft_source.shape)].to(SSettings.instance().device)
        fft_psf = container.dic_psf['fft_psf'+str(fft_source.shape)].to(SSettings.instance().device)
    
    out_image = torch.real(torch.fft.ifftn((fft_source * torch.conj(fft_psf)) / den))
    if image_pad.shape != image.shape:
        return unpad_3d(out_image, padding)
    return out_image
    
def apply_deconv(imsm,psf=None,plt_val=False,parameters = {'method':'wiener','beta':0.001,'niter':50},gpu=False,force=False,pad=None):
    r"""Applies deconvolution to image <imsm> using sdeconv: https://github.com/sylvainprigent/sdeconv/
    Currently assumes 60x objective with ~1.4 NA using SPSFGibsonLanni. Should be modified to find 
    
    Recomendations: the default wiener method with a low beta is the best for very fast local fitting. Albeit larger scale artifacts.
    For images: recommend the lucy method with ~30 iterations.
    
    This wraps around pytoch.
    
    To install:
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
    pip install sdeconv
    Optional: decided to modify the __init__ file of the SSettingsContainer in 
    C:\Users\BintuLabUser\anaconda3\envs\cellpose\Lib\site-packages\sdeconv\core\_settings.py
    
    import os
    gpu = True
    if os.path.exists("use_gpu.txt"):
        gpu = eval(open("use_gpu.txt").read())
    self.device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu) else "cpu")
    to toggle the GPU on or off. By default it just uses the GPU if GPU detected by pytorch"""
    
    #import sdeconv,os
    #fl = os.path.dirname(sdeconv.__file__)+os.sep+'core'+os.sep+'use_gpu.txt'
    #fid = open(fl,'w')
    #fid.write('True')
    #fid.close()
    import torch
    from sdeconv.core import SSettings
    obj = SSettings.instance()
    obj.device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu) else "cpu")
    if force:
        if hasattr(obj,'dic_psf'): del obj.dic_psf
    # map to tensor
    
    imsm_ = torch.from_numpy(np.array(imsm,dtype=np.float32))
    if psf is None:
        from sdeconv.psfs import SPSFGibsonLanni
        #psf_generator = SPSFGaussian((1,1.5, 1.5), imsm_.shape)
        psf_generator = SPSFGibsonLanni(M=60,shape=imsm_.shape)
        psf = psf_generator().to(obj.device)
    else:
        psff = np.zeros(imsm_.shape,dtype=np.float32)
                
        slices = [(slice((s_psff-s_psf_full_)//2,(s_psff+s_psf_full_)//2),slice(None)) if s_psff>s_psf_full_ else
         (slice(None),slice((s_psf_full_-s_psff)//2,(s_psf_full_+s_psff)//2))
          
          for s_psff,s_psf_full_ in zip(psff.shape,psf.shape)]
        sl_psff,sl_psf_full_ = list(zip(*slices))
        psff[sl_psff]=psf[sl_psf_full_]
        psf = torch.from_numpy(np.array(psff,dtype=np.float32)).to(obj.device)
        
    method = parameters.get('method','wiener')
    if pad is None:
        pad = int(np.min(list(np.array(imsm.shape)-1)+[50]))
    if method=='wiener':
        from sdeconv.deconv import SWiener
        beta = parameters.get('beta',0.001)
        filter_ = SWiener(psf, beta=beta, pad=pad)
        #monkey patch _wiener_3d to allow recycling the fft of the psf components
        filter_._wiener_3d = _wiener_3d.__get__(filter_, SWiener)
    elif method=='lucy':
        from sdeconv.deconv import SRichardsonLucy
        niter = parameters.get('niter',50)
        filter_ = SRichardsonLucy(psf, niter=niter, pad=pad)
    elif method=='spitfire':
        from sdeconv.deconv import Spitfire
        filter_ = Spitfire(psf, weight=0.6, reg=0.995, gradient_step=0.01, precision=1e-6, pad=pad)
    out_image = filter_(imsm_)
    out_image = out_image.cpu().detach().numpy().astype(np.float32)
    if plt_val:
        import napari
        viewer = napari.view_image(out_image)
        viewer.add_image(imsm)
    return out_image
    
    
def get_local_maxfast_tensor(im_dif_npy,th_fit=500,im_raw=None,dic_psf=None,delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5,gpu=False):
    import torch
    dev = "cuda:0" if (torch.cuda.is_available() and gpu) else "cpu"
    im_dif = torch.from_numpy(im_dif_npy).to(dev)
    z,x,y = torch.where(im_dif>th_fit)
    zmax,xmax,ymax = im_dif.shape
    def get_ind(x,xmax):
        # modify x_ to be within image
        x_ = torch.clone(x)
        bad = x_>=xmax
        x_[bad]=xmax-x_[bad]-2
        bad = x_<0
        x_[bad]=-x_[bad]
        return x_
    #def get_ind(x,xmax):return x%xmax
    for d1 in range(-delta,delta+1):
        for d2 in range(-delta,delta+1):
            for d3 in range(-delta,delta+1):
                if (d1*d1+d2*d2+d3*d3)<=(delta*delta):
                    z_ = get_ind(z+d1,zmax)
                    x_ = get_ind(x+d2,xmax)
                    y_ = get_ind(y+d3,ymax)
                    keep = im_dif[z,x,y]>=im_dif[z_,x_,y_]
                    z,x,y = z[keep],x[keep],y[keep]
    h = im_dif[z,x,y]
    
    
    if len(x)==0:
        return []
    if delta_fit>0:
        d1,d2,d3 = np.indices([2*delta_fit+1]*3).reshape([3,-1])-delta_fit
        kp = (d1*d1+d2*d2+d3*d3)<=(delta_fit*delta_fit)
        d1,d2,d3 = d1[kp],d2[kp],d3[kp]
        d1 = torch.from_numpy(d1).to(dev)
        d2 = torch.from_numpy(d2).to(dev)
        d3 = torch.from_numpy(d3).to(dev)
        im_centers0 = (z.reshape(-1, 1)+d1).T
        im_centers1 = (x.reshape(-1, 1)+d2).T
        im_centers2 = (y.reshape(-1, 1)+d3).T
        z_ = get_ind(im_centers0,zmax)
        x_ = get_ind(im_centers1,xmax)
        y_ = get_ind(im_centers2,ymax)
        im_centers3 = im_dif[z_,x_,y_]
        if im_raw is not None:
            im_raw_ = torch.from_numpy(im_raw).to(dev)
            im_centers4 = im_raw_[z_,x_,y_]
            habs = im_raw_[z,x,y]
        else:
            im_centers4 = im_dif[z_,x_,y_]
            habs = x*0
            a = x*0
        Xft = torch.stack([d1,d2,d3]).T
    
        bk = torch.min(im_centers3,0).values
        im_centers3 = im_centers3-bk
        im_centers3 = im_centers3/torch.sum(im_centers3,0)
        if dic_psf is None:
            sigma = torch.tensor([sigmaZ,sigmaXY,sigmaXY],dtype=torch.float32,device=dev)#np.array([sigmaZ,sigmaXY,sigmaXY],dtype=np.flaot32)[np.newaxis]
            Xft_ = Xft/sigma
            norm_G = torch.exp(-torch.sum(Xft_*Xft_,-1)/2.)
            norm_G=(norm_G-torch.mean(norm_G))/torch.std(norm_G)
    
            hn = torch.mean(((im_centers3-im_centers3.mean(0))/im_centers3.std(0))*norm_G.reshape(-1,1),0)
            a = torch.mean(((im_centers4-im_centers4.mean(0))/im_centers4.std(0))*norm_G.reshape(-1,1),0)
            
        zc = torch.sum(im_centers0*im_centers3,0)
        xc = torch.sum(im_centers1*im_centers3,0)
        yc = torch.sum(im_centers2*im_centers3,0)
        Xh = torch.stack([zc,xc,yc,bk,a,habs,hn,h]).T.cpu().detach().numpy()
    else:
        Xh =  torch.stack([z,x,y,h]).T.cpu().detach().numpy()
        
def cell_segmentation(fl, save_seg_fl, ch_nuc=3, ch_mem=0, resc=4, rescz=2, cellprob_mem=-5, cellprobe_nuc=-5, 
                      cellprobe_thres=0, anisotropy=1.38, flag_plot=False):
    import nd2
    import tifffile
    import numpy as np
    import matplotlib.pylab as plt
    from cellpose import models, utils
    im = nd2.imread(fl)
    imf = np.swapaxes(im,0,1) #swap z and color dims
    imf = imf[:,::rescz]
    im_dapi,im_edges = imf[ch_nuc],imf[ch_mem]
    
    # resize, normalize and check the data
    im_dapi_ = norm_zstack(im_dapi,gb_sm=5,gb_big=200,resc=resc)
    im_edges_ = norm_zstack(im_edges,gb_sm=5,gb_big=200,resc=resc)
    im_edges__ = norm_perc(im_edges_,percm=1,percM=99.9)
    im_dapi__ = norm_perc(im_dapi_,percm=1,percM=99.9)
    imgf = np.rollaxis(np.array([im_edges__,im_edges__,im_dapi__]),0,4)
    
    
    # membrane segmentation
    imsm = imgf.copy()
    model = models.Cellpose(gpu=True, model_type='cyto2')
    chan = [2,3]
    masks, flows, styles, diams = model.eval(imsm, diameter=33, channels=chan,
                                        min_size=500,normalize=False,do_3D=True,anisotropy=anisotropy,cellprob_threshold=cellprob_mem)
    
    #from skimage.measure import regionprops, regionprops_table
    #props = regionprops_table(masks, flows[2], properties=['intensity_mean', 'label'])
    #inval = props['label'][props['intensity_mean']<0]

    masks_ = masks.copy()
    #for l_ in inval:
    #    masks_[masks_==l_] = 0
        
    # nuclear segmentation
    imsm = imgf.copy()
    model = models.Cellpose(gpu=True, model_type='nuclei')
    chan = [3, 0]
    masks_nuc, flows_nuc, styles_nuc, diams_nuc = model.eval(imsm, diameter=20, channels=chan,
                                        min_size=500,normalize=False,do_3D=True,anisotropy=anisotropy,cellprob_threshold=cellprobe_nuc)
        
    
    # combine masks
    from scipy import ndimage
    cells_ = np.unique(masks_nuc)[1:]
    cms_nuc = ndimage.center_of_mass(masks_nuc>0, masks_nuc, cells_)
    nuc_cell_id = np.array([masks_[l_[0], l_[1], l_[2]] for l_ in np.round(cms_nuc).astype(int)])
    cell_ids = np.unique(masks_)
    pset = set(cell_ids).difference(set(nuc_cell_id))
    
    masks_combined = masks_.copy()
    new_id = np.max(masks_) + 1
    for c_ in pset:
        masks_combined[masks_==c_] = 0

    dict_type = {0:'BG'}
    for c_ in np.unique(masks_combined)[1:]:
        dict_type[c_] = 'cell'
    for c_ in cells_[nuc_cell_id==0]:
        masks_combined[masks_nuc==c_] = new_id
        dict_type[new_id] = 'YSL'
        new_id+=1
    
    # export segmentation mask
    obj = nd2.ND2File(fl)
    shape = obj.metadata.channels[0].volume.voxelCount[::-1]
    masksf =  resize(masks_combined,shape)
    maskmem = resize(masks_,shape)
    masknuc = resize(masks_nuc, shape)
    np.savez_compressed(save_seg_fl, mask=masksf, maskmem=maskmem, masknuc=masknuc, dict_type=dict_type)
    
    if flag_plot==True:
        imsm = imgf.copy()
        imgedge = imsm[...,0].copy()
        edge = np.array([utils.masks_to_edges(msk) for msk in masks])
        imgedge[edge>0]=1
        imsm[...,0] = imgedge
        tifffile.imshow(imsm,interpolation='nearest',cmap='gray')
    
    return masks_combined, masks_, masks_nuc, dict_type
    
def nuclei_segmentation(fl, save_seg_fl, resc=4, rescz=2, cellprobe_nuc=-5, 
                      cellprobe_thres=0, flag_plot=False):

    import tifffile
    import numpy as np
    import matplotlib.pylab as plt
    from cellpose import models, utils
    im = tifffile.imread(fl)
    im_dapi = im[::rescz]
    
    # resize, normalize and check the data
    im_dapi_ = norm_zstack(im_dapi,gb_sm=5,gb_big=200,resc=4)
    im_dapi__ = norm_perc(im_dapi_,percm=1,percM=99.9)
        
    # nuclear segmentation
    model = models.Cellpose(gpu=True, model_type='nuclei')
    chan = [0, 0]
    masks, flows, styles, diams = model.eval(im_dapi__, diameter=20, channels=chan,
                                        min_size=500,normalize=False,do_3D=True,anisotropy=1.38,cellprob_threshold=cellprobe_nuc)
        
    
    # assign cell type
    masks_ = masks.copy()

    dict_type = {0:'BG'}
    for c_ in np.unique(masks_)[1:]:
        dict_type[c_] = 'nuclei'
    
    # export segmentation mask
    shape = im.shape
    masksf =  resize(masks_,shape)
    np.savez_compressed(save_seg_fl, mask=masksf, dict_type=dict_type)
    
    if flag_plot==True:
        imsm = im_dapi__.copy()
        imgedge = imsm[...].copy()
        edge = np.array([utils.masks_to_edges(msk) for msk in masks])
        imgedge[edge>0]=3
        imsm[...] = imgedge
        tifffile.imshow(imsm,interpolation='nearest')
        #show segmentation in xz view
        imsm_sw = np.swapaxes(imsm,0,2)
        tifffile.imshow(imsm_sw,interpolation='nearest')#,cmap='Blues')
    
    return masks, dict_type

def consolidate_seg_masks(fls, fl_tile_reg, point_list):
    from scipy import ndimage
    import shutil
    pairs = np.load(fl_tile_reg)['pairs']
    offsets = np.load(fl_tile_reg)['offsets']
    tileidx = np.load(fl_tile_reg)['tileidx']
    for ip in range(len(point_list)):
        save_seg_fl = fls[ip]
        dst_seg_fl = fls[ip].replace('_seg.npz','_seg_corr.npz')
        shutil.copyfile(save_seg_fl, dst_seg_fl)
    for i in range(len(pairs)):    
        print('Consolidating Point' + point_list[pairs[i][0]] +' and Point' + point_list[pairs[i][1]])
        save_seg_fl1 = fls[pairs[i][0]].replace('_seg.npz','_seg_corr.npz')
        mask1 = np.load(save_seg_fl1, mmap_mode='r')['mask']
        save_seg_fl2 = fls[pairs[i][1]].replace('_seg.npz','_seg_corr.npz')
        mask2 = np.load(save_seg_fl2, mmap_mode='r')['mask']
        
        szs = np.array(mask1.shape)
        txyz = offsets[i].copy()
        szf = np.array(mask1.shape)
        start1,end1 = txyz,szs+txyz
        start2,end2 = np.array([0,0,0]),szs
        start2[start1<0]-=start1[start1<0]
        start1[start1<0]=0
        end2[end1>szf]-=end1[end1>szf]-szs[end1>szf]
        end1[end1>szf]=szf[end1>szf]
        mask2_ol=mask2[start1[0]:end1[0],start1[1]:end1[1],start1[2]:end1[2]]
        mask1_ol=mask1[start2[0]:end2[0],start2[1]:end2[1],start2[2]:end2[2]]
        cells1 = np.unique(mask1_ol)[1:]
        cells2 = np.unique(mask2_ol)[1:]
        cms1 = ndimage.center_of_mass(mask1_ol>0, mask1_ol, cells1)
        cms2 = ndimage.center_of_mass(mask2_ol>0, mask2_ol, cells2)
        if len(cms1)>0 and len(cms2)>0 : 
            if   (tileidx[pairs[i][1], 0] - tileidx[pairs[i][0], 0] == 1):
                cells1_del = cells1[np.array(cms1)[:, 1]>(end1[1]-start1[1]+1)/2]
                cells2_del = cells2[np.array(cms2)[:, 1]<=(end1[1]-start1[1]+1)/2]
            elif (tileidx[pairs[i][1], 1] - tileidx[pairs[i][0], 1] == 1):
                cells1_del = cells1[np.array(cms1)[:, 2]>(end1[2]-start1[2]+1)/2]
                cells2_del = cells2[np.array(cms2)[:, 2]<=(end1[2]-start1[2]+1)/2]
            mask1[np.isin(mask1,cells1_del)] = 0
            mask2[np.isin(mask2,cells2_del)] = 0
            
        np.savez_compressed(save_seg_fl1,mask=mask1)
        np.savez_compressed(save_seg_fl2,mask=mask2)
    return
    
def expand_seg(im,npix=5,zinterp=2,xyinterp=4):
    A = im[::zinterp,::xyinterp,::xyinterp].copy()
    A_nonzero = A>0
    from scipy import ndimage as nd
    A_dil = nd.binary_dilation(A_nonzero,nd.generate_binary_structure(3, 1),iterations=npix)
    X = np.array(np.where(A_dil.astype(np.float32)-A_nonzero)).T
    Xincell = np.array(np.where(A_nonzero)).T
    from scipy.spatial import cKDTree as KDTree
    tree = KDTree(Xincell)
    dists,ielems = tree.query(X)
    A[tuple(X.T)] = A[tuple(Xincell[ielems].T)]
    return A    

def segment_EVL(wd, p, nhyb_mem, ch_dapi, ch_mem, nhyb_evl, ch_evl, thres_evl=20, b_th=250, corr_th=0.3, resc=4):
    import nd2
    raw_data_dir = wd + os.sep + 'rawdata'
    img = nd2.imread(raw_data_dir + os.sep + 'hyb' + str(nhyb_mem).zfill(3) + '_Point' + p + '.nd2', dask=True)
    im_dapi = np.array(img[:,ch_dapi],dtype=np.float32)
    im_edge = np.array(img[:,ch_mem],dtype=np.float32)
    
    print('construct image')
    # resize, normalize and check the data
    im_dapi_ = norm_zstack(im_dapi,gb_sm=5,gb_big=200,resc=resc)
    im_edge_ = norm_zstack(im_edge,gb_sm=5,gb_big=200,resc=resc)
    im_edge__ = norm_perc(im_edge_,percm=1,percM=99.9)
    im_dapi__ = norm_perc(im_dapi_,percm=1,percM=99.9)
    imgf = np.rollaxis(np.array([im_edge__,im_edge__,im_dapi__]),0,4)
    
    print('2D segmentation')
    # membrane segmentation
    from cellpose import models, utils
    imsm = imgf.copy()
    model = models.Cellpose(gpu=True, model_type='cyto2')
    chan = [2,3]
    masks2D, flows, styles, diams = model.eval(imsm, diameter=50, channels=chan,normalize=False,do_3D=False, cellprob_threshold=-30)
    
    # consolidate 2D masks to unique 3D id
    masks2D_ = masks2D.copy()
    offset = 0
    for z in range(masks2D.shape[0]):
        masks2D_[z] += offset
        offset += np.max(masks2D[z])
    masks2D_[masks2D==0] = 0
    
    print('3D segmentation')
    masks3D, flows, styles, diams = model.eval(imsm, diameter=50, channels=chan,normalize=False,do_3D=False, cellprob_threshold=-30, stitch_threshold=0.5)
    
    print('Keep EVL')
    dot_dir = wd + os.sep + r'Result\called_spots'
    fl_XH_evl = glob.glob(dot_dir + os.sep + 'hyb' + str(nhyb_evl).zfill(3) + '_Point' + p + '_ch' + str(ch_evl) + '*.npz')[0]
    XH_evl = np.load(fl_XH_evl)['arr_0']
    
    rtxyz = np.load(wd + os.sep + r'drift_correction\drift_correction_Point' + p + '.npz')['rtxyz']
    
    Xh = XH_evl[(XH_evl[:, -1]>b_th) & (XH_evl[:, -2]>corr_th), :3]
    txyzf = rtxyz[nhyb_evl-1] - rtxyz[nhyb_mem-1]
    
    Xh_evl = Xh.copy() - txyzf
    Xh_evl[:, 1:3] /= resc
    
    X = Xh_evl.astype(int)
    kp = np.all((X>=0)&(X<im_dapi__.shape),axis=-1)
    X_ = X[kp]
    cid = masks2D_[tuple(X_.T)]
    
    ucid, nkrt = np.unique(cid, return_counts=True)
    
    cid_evl = ucid[nkrt>thres_evl]
    masks2D_evl = masks2D_.copy()
    masks2D_evl[~np.isin(masks2D_, cid_evl)] = 0
    
    masks_evl = masks3D.copy()
    masks_evl[masks2D_evl==0] = 0
    
    masks_evl_ = expand_seg(masks_evl, npix=1,zinterp=1,xyinterp=1)
    
    save_seg_fl = wd + os.sep + 'Result' + os.sep + 'Point' + p +'_EVL.npz'
    
    masksf =  resize(masks_evl_,im_dapi.shape)
    masks2Dsf = resize(masks2D_, im_dapi.shape)
    np.savez_compressed(save_seg_fl, mask=masksf, mask2D=masks2Dsf)
    
    return masksf

def incorporate_evl(wd, p, nhyb_mem):
    mask_ori = np.load(wd + os.sep + 'Result' + os.sep + 'hyb'+ str(nhyb_mem).zfill(3) + '_Point'+p+'_seg.npz')['mask']
    type_ori = np.load(wd + os.sep + 'Result' + os.sep + 'hyb'+ str(nhyb_mem).zfill(3) + '_Point'+p+'_seg.npz', allow_pickle=True)['dict_type'].item()
    mask_evl = np.load(wd + os.sep + 'Result' + os.sep + 'Point' + p + '_EVL.npz')['mask']
    
    unique_labels, mask_evl_sorted = np.unique(mask_evl, return_inverse=True)
    mask_evl_sorted = np.reshape(mask_evl_sorted, mask_evl.shape).astype('uint16')
    
    lim_ori = np.max(mask_ori)
    mask_evl_sorted[mask_evl_sorted>0] += lim_ori
    
    mask_combined = np.maximum(mask_ori, mask_evl_sorted)
    
    # set regions above EVL as background
    z_evl = mask_evl.shape[0] - np.argmax((mask_evl>0)[::-1], axis=0)- 1
    mask_combined_ = mask_combined.copy()
    for x in range(mask_ori.shape[1]):
        for y in range(mask_ori.shape[2]):
                mask_combined_[z_evl[x, y]:, x, y] = 0
               
    ucells = np.unique(mask_combined_)
    
    dict_type = {0:'BG'}
    for c in ucells[1:]:
        if c<=lim_ori:
            dict_type[c] = type_ori[c]
        else:
            dict_type[c] = 'EVL'
    
    fl_out = wd + os.sep + 'Result' + os.sep + 'hyb'+ str(nhyb_mem).zfill(3) + '_Point'+p+'_evl_seg.npz'
    
    np.savez_compressed(fl_out, mask=mask_combined_, dict_type=dict_type)    
    
def extract_cellinfo(fls, fl_tile_reg, fl_dict, rescale=1):
    import numpy as np
    from tqdm import tqdm
    from scipy import ndimage
    tot_ = 0 # to mark the current max cell ID, add as an offset to the next mask
     #downsample the mask for volume calculation
    
    offsets_abs = np.load(fl_tile_reg)['offsets_abs']
    cellvols = {}
    cellcms = {}
    cellFOVs = {}
    celltypes = {}

    for ip in tqdm(range(len(fls))):
        seg_fl = fls[ip].replace('_seg.npz','_seg_corr.npz')
        dict_fl = fls[ip]
        dst_seg_fl = fls[ip].replace('_seg.npz', '_seg_corr_newID.npz')
        
        mask = np.load(seg_fl, mmap_mode='r')['mask']
        dict_type = np.load(dict_fl, mmap_mode='r', allow_pickle=True)['dict_type'].item()
        
        cells_resc = mask[::rescale,::rescale,::rescale]

        cells_,cell_vols = np.unique(cells_resc,return_counts=True)
        cell_vols = cell_vols*rescale*rescale*rescale
        cells_ = cells_[1:]
        cell_vols = cell_vols[1:]
        
        cms = ndimage.center_of_mass(cells_resc>0, cells_resc, cells_)
        cms_native = np.array(cms)*[rescale,rescale,rescale]
        cms = np.array(cms)*[rescale,rescale,rescale] + offsets_abs[ip]
        #print('max cell id: ' + str(np.max(cells_))+ ', current offset: ' + str(tot_))
        cells_ += tot_
        
        
        cellvols.update(dict(zip(cells_, cell_vols)))
        cellcms.update(dict(zip(cells_, cms))) 
        cellFOVs.update(dict(zip(cells_, np.repeat(ip, len(cells_)))))
                        
        mask[mask>0] += tot_
        
        for l_ in cells_:
            celltypes[l_]=dict_type[l_-tot_]      
        np.savez_compressed(dst_seg_fl, mask=mask, dict_type=dict_type, cells=cells_, cms=cms_native, cms_global=cms)     
        tot_ = np.max(cells_)
        
    np.savez(fl_dict,cellvols=cellvols, cellcms=cellcms, cellFOVs=cellFOVs, celltypes=celltypes)
    return
    
def calculate_unified_mask(fl_tile_reg, fls_seg, resc=4):
    import numpy as np
    offsets_abs = np.load(fl_tile_reg)['offsets_abs']
    mask0 = np.load(fls_seg[0])['mask']
    
    ind_range1 = [  0,    0,    0] + offsets_abs
    ind_range2 = [mask0.shape[0]-1, mask0.shape[1]-1, mask0.shape[2]-1] + offsets_abs
    offset_unif_mask = np.min(np.concatenate((ind_range1, ind_range2)), axis=0)
    sz_unif_mask = np.max(np.concatenate((ind_range1, ind_range2)), axis=0) - np.min(np.concatenate((ind_range1, ind_range2)), axis=0)
    sz_unif_mask = np.round(sz_unif_mask/resc).astype(int)
    
    unif_mask = np.zeros(sz_unif_mask, dtype=np.uint16)

    from tqdm import tqdm
    for ip in tqdm(range(len(fls_seg))):
        seg_fl = fls_seg[ip]
        mask = np.load(seg_fl, mmap_mode='r')['mask']
        mask = mask[::resc, ::resc, ::resc]
        
        start_unif = np.floor(([0, 0, 0] + offsets_abs[ip] - offset_unif_mask)/resc).astype(int)
        end_unif   = start_unif + mask.shape
        
        unif_mask[start_unif[0]:end_unif[0],start_unif[1]:end_unif[1], start_unif[2]:end_unif[2]] = np.maximum(unif_mask[start_unif[0]:end_unif[0],start_unif[1]:end_unif[1], start_unif[2]:end_unif[2]], mask)
    return unif_mask
    
def norm_slice_ref(im,ref_im,s=50):
    import cv2
    import numpy as np
    im_32=im.astype(np.float32)
    ref_im = ref_im.astype(np.float32)
    im_ = np.array([cv2.divide(zplane,ref_im) for zplane in im_32],dtype=np.float32)
    im_ = im_*np.max(ref_im)
    
    im_norm = np.array([im__-cv2.blur(im__,(s,s)) for im__ in im_],dtype=np.float32)
    
    return np.array(im_norm,dtype=np.float32)
    
def norm_slice_new_constant(im,s1=50,s2=300,c=50):
    import cv2
    import numpy as np
    im_32=im.astype(np.float32)
    im_ = np.array([cv2.divide(zplane,cv2.blur(zplane,(s2,s2))-c) for zplane in im_32],dtype=np.float32)
    im_ = im_*(np.mean(im_32)/np.mean(im_))
    im_norm = np.array([im__-cv2.blur(im__,(s1,s1)) for im__ in im_],dtype=np.float32)
    
    return np.array(im_norm,dtype=np.float32)
    
import os
class get_dapi_features:
    def __init__(self,fl,save_fl,set_='',gpu=False,im_med_fl = None,
                psf_fl = 'psf_cy5.npy',redo=False):
                
        """
        Given a file <fl> and a save folder <save_folder> this class will load the image fl, flat field correct it, it deconvolves it using <psf_fl> and then finds the local minimum and maximum.
        
        This saves data in: save_folder+os.sep+fov+'--'+htag+'--dapiFeatures.npz' which contains the local maxima: Xh_plus and local minima: Xh_min 
        """
      
        self.gpu=gpu
        
        self.fl,self.fl_ref='',''
        self.im_med=None
        self.im_med_fl=im_med_fl
        
        
        self.save_fl = save_fl#save_folder+os.sep+fov+'--'+htag+'--'+set_+'dapiFeatures.npz'
        self.fl = fl
        if not os.path.exists(self.save_fl) or redo:
            self.psf = np.load(psf_fl)
            if im_med_fl is not None:
                im_med = np.load(im_med_fl)['im']
                im_med = cv2.blur(im_med,(20,20))
                self.im_med=im_med
            self.load_im()
            self.get_X_plus_minus()
            np.savez(self.save_fl,Xh_plus = self.Xh_plus,Xh_minus = self.Xh_minus)
        else:
            dic = np.load(self.save_fl)
            self.Xh_minus,self.Xh_plus = dic['Xh_minus'],dic['Xh_plus']
    def load_im(self):
        """
        Load the image from file fl and apply: flat field, deconvolve, subtract local background and normalize by std
        """
        if type(self.fl) is str:
            im = np.array(read_im(self.fl)[-1],dtype=np.float32)
        else: 
            im = np.array(self.fl,dtype=np.float32)
        if self.im_med_fl is not None:
            im = im/self.im_med*np.median(self.im_med)
        imD = full_deconv(im,psf=self.psf,gpu=self.gpu)
        imDn = norm_slice(imD,s=30)
        imDn_ = imDn/np.std(imDn)
        self.im = imDn_
    def get_X_plus_minus(self):
        #load dapi
        im1 = self.im
        self.Xh_plus = get_local_maxfast_tensor(im1,th_fit=4.5,delta=5,delta_fit=5)
        self.Xh_minus = get_local_maxfast_tensor(-im1,th_fit=4.5,delta=5,delta_fit=5)



def get_im_from_Xh(Xh,resc=5):
    X = np.round(Xh[:,:3]/resc).astype(int)
    #X-=np.min(X,axis=0)
    sz = np.max(X,axis=0)
    imf = np.zeros(sz+1,dtype=np.float32)
    imf[tuple(X.T)]=1
    return imf
def get_Xtzxy(X,X_ref,tzxy0,resc,learn=1):
    tzxy = tzxy0
    for it_ in range(5):
        XT = X-tzxy
        ds,inds = cKDTree(X_ref).query(XT)
        keep = ds<resc*learn**it_
        X_ref_ = X_ref[inds[keep]]
        X_ = X[keep]
        if np.sum(keep)==0:
            return(np.array([np.nan, np.nan, np.nan]), 0)
        tzxy = np.mean(X_-X_ref_,axis=0)
        #print(tzxy)
    return tzxy,len(X_)
def get_best_translation_points(X,X_ref,resc=5):
    im = get_im_from_Xh(X,resc=resc)
    im_ref = get_im_from_Xh(X_ref,resc=resc)
    from scipy.signal import fftconvolve
    im_cor = fftconvolve(im,im_ref[::-1,::-1,::-1])
    tzxy = np.array(np.unravel_index(np.argmax(im_cor),im_cor.shape))-im_ref.shape+1
    tzxy = tzxy*resc
    tzxy,Npts = get_Xtzxy(X,X_ref,tzxy,resc)
    return tzxy,Npts
    
def get_txyz_dapi_features(obj, obj_ref, resc=5):
    cpX = obj.Xh_plus[:,:3]
    cpX_ref = obj_ref.Xh_plus[:,:3]
    tzxy_plus,Nplus = get_best_translation_points(cpX,cpX_ref,resc=resc)

    cpX = obj.Xh_minus[:,:3]
    cpX_ref = obj_ref.Xh_minus[:,:3]
    tzxy_minus,Nminus = get_best_translation_points(cpX,cpX_ref,resc=resc)
    cptzxy_plus = tzxy_plus
    cptzxy_minus = tzxy_minus
    tzxyf = -(cptzxy_plus+cptzxy_minus)/2
    return tzxyf, -cptzxy_plus, -cptzxy_minus, Nplus, Nminus
    return Xh