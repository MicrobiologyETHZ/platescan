import argparse,os,scipy
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from skimage import io,exposure
from skimage.draw import circle,ellipse

def trisect(image):
    # Use the mean value of each pixel row to perform auto-correlation and find the repeated plate edge
    print("Trisecting image")
    
    h,w = image.shape
    rowMeans = [np.mean(image[i,:]) for i in range(0,h)]

    # Treat the rowMeans as a signal
    # Use the fast fourier transform and auto-correlate to find repeats
    ft = np.fft.fft(rowMeans)
    cor = np.fft.ifft(ft*np.conjugate(ft)).real
    peakPositions = scipy.signal.argrelextrema(cor,np.greater)[0]
    peakValues = cor[np.array(peakPositions)]
    breakPositions = np.sort(peakPositions[np.argpartition(peakValues,-2)[-2:]])
    height = abs(breakPositions[1]-breakPositions[0])
    images = [image[0:height,0:w],image[breakPositions[0]:breakPositions[1],0:w],image[(h-height):h,0:w]]

    return(images)

def cropImage(image,blank):
    if blank is not None:
        # If a blank is provided, use that to crop the image to the plate
        xcor = crossCorrelate(image,blank)
        offset = np.where(xcor == xcor.max())
        h,w = blank.shape
        L = offset[1][0]
        R = offset[1][0]+w
        T = offset[0][0]
        B = offset[0][0]+h
    else:
        # Plate edge detection; performance highly dependent on lighting and plate type
        h,w = image.shape
        rowMeans = [np.mean(image[i,:]) for i in range(0,h)]
        colMeans = [np.mean(image[:,i]) for i in range(0,w)]
        
        # Smooth curves
        rowMeans = scipy.signal.savgol_filter(rowMeans,int(h/15)+1-int(h/15)%2,3)
        colMeans = scipy.signal.savgol_filter(colMeans,int(w/15)+1-int(w/15)%2,3)
        
        Lpos,Rpos = np.array_split(scipy.signal.argrelextrema(colMeans,np.greater)[0],2)
        L = np.sort(Lpos[np.argpartition(colMeans[Lpos],-1)[-1:]])[0]
        R = np.sort(Rpos[np.argpartition(colMeans[Rpos],-1)[-1:]])[0]
        Tpos,Bpos = np.array_split(scipy.signal.argrelextrema(rowMeans,np.greater)[0],2)
        T = np.sort(Tpos[np.argpartition(rowMeans[Tpos],-1)[-1:]])[0]
        B = np.sort(Bpos[np.argpartition(rowMeans[Bpos],-1)[-1:]])[0]
    
    print("Cropping plate:")
    print("{}:{} left to right".format(L,R))
    print("{}:{} top to bottom".format(T,B))
    
    return(image[T:B,L:R])
        
def crossCorrelate(a,b,boundary=None):
    #Perform a normalized cross-correlation of two arrays
    
    if boundary == 'wrap':
        a = wrapImage(a)
    aN = (a-np.mean(a))/np.std(a)
    bN = (b-np.mean(b))/np.std(b)
    xcor = scipy.signal.correlate(aN,bN,mode='valid',method='fft')
    
    return(xcor)
    
def findGrid(plate,nrows,ncols,layout,r,xGap,yGap,pad):
    #Make an array of spots to use as a mask, cross-correlate with the plate image and find the best-fitting grid points
    print("  Finding colonies")
    
    mask = -np.ones((round((2*pad)+((nrows-1)*yGap)),round((2*pad)+((ncols-1)*xGap))))
    for y in range(0,nrows):
        for x in range(0,ncols):
            if layout[y,x]:
                mask[circle(pad+round(y*yGap),pad+round(x*xGap),r)] = 1

    xcor = crossCorrelate(plate,mask)
    offset = np.where(xcor == xcor.max())
    offset = np.array((offset[0][0]+pad,offset[1][0]+pad))
    
    rowIndex = np.array([round(offset[0]+(y*yGap)) for y in range(0,nrows)],dtype=np.int32)
    colIndex = np.array([round(offset[1]+(x*xGap)) for x in range(0,ncols)],dtype=np.int32)
    
    masked = np.copy(plate)
    for y in range(0,nrows):
        for x in range(0,ncols):
            if layout[y,x]:
                masked[circle(offset[0]+round(y*yGap),offset[1]+round(x*xGap),r)] = 1

    return(masked,rowIndex,colIndex)

def correctLighting(cell):
    #Correct a lighting gradient across a cell by SVD (not used, not necessary?)
    h,w = cell.shape
    l = np.column_stack((np.repeat(0,h),range(0,h),cell[:,0]))
    r = np.column_stack((np.repeat(w-1,h),range(0,h),cell[:,-1]))
    t = np.column_stack((range(0,w),np.repeat(0,w),cell[0,:]))
    b = np.column_stack((range(0,w),np.repeat(h-1,w),cell[-1,:]))
    xyz = np.row_stack((l,r,t,b))
    xyz[:,0] -= np.mean(xyz[:,0])
    xyz[:,1] -= np.mean(xyz[:,1])
    xyz[:,2] -= np.mean(xyz[:,2])
    
    u,s,v = np.linalg.svd(xyz)
    dzdx = v[:,-1][0]
    dzdy = v[:,-1][1]
    
    correction = np.zeros((h,w))
    for y in range(0,h):
        for x in range(0,w):
            correction[y,x] = (x * dzdx) + (y * dzdy)
    corrected = cell + correction
    return(corrected)

def wrapImage(image):
    h,w = image.shape

    q1 = image[:int(h/2),:int(w/2)]
    q2 = image[:int(h/2),int(w/2):]
    q3 = image[int(h/2):,:int(w/2)]
    q4 = image[int(h/2):,int(w/2):]
    
    row1 = np.concatenate((q4,q3,q4,q3),axis=1)
    row2 = np.concatenate((q2,q1,q2,q1),axis=1)
    row3 = np.concatenate((q4,q3,q4,q3),axis=1)
    row4 = np.concatenate((q2,q1,q2,q1),axis=1)
    
    wrapped = np.concatenate((row1,row2,row3,row4))
    
    return(wrapped)

def findColony(cell,rmin,rmax):
    #Find a colony by cross-correlation of ideal circular templates of different radii to create inside and outside regions
    
    h,w = cell.shape
    
    offsets = []
    scores = []
    for r in np.arange(rmin,rmax,0.5):
        mask = -np.ones((h,w))
        mask[ellipse(int(h/2),int(w/2),r,r)] = 1
        xcor = crossCorrelate(cell,mask,boundary='wrap')
        offset = np.where(xcor==xcor[int(r):int(h)-int(r),int(r):int(w)-int(r)].max())
        offset = (offset[0][0],offset[1][0])
        offsets.append(offset)
        scores.append(xcor.max())

    best = scores.index(max(scores))
    r = np.arange(rmin,rmax,0.5)[best]
    offset = offsets[best]
    
    hili = np.copy(cell)
    outside = np.ones(cell.shape,np.bool)
    outside[ellipse(offset[0],offset[1],r,r)] = 0
    hili[outside] = hili[outside]/2
    return(offset[0],offset[1],r,max(scores),hili)

def scoreColony(cell,r,offset):
    #Score a colony by summing the pixel array intensities after subtracting the mean intensity of the outside region
    
    h,w = cell.shape
    outside = np.ones((h,w),np.bool)
    outside[ellipse(offset[0],offset[1],r,r)] = 0    
    
    bg = np.mean(cell[outside])
    fg = np.mean(cell[~outside])
    bgvar = np.var(cell[outside])
    fgvar = np.var(cell[~outside])
    return(bg,fg,bgvar,fgvar)

def scoreGrowth(plate,rowIndex,colIndex,layout,rmin,rmax,pad):
    #Find each cvarsolony within its cell, then score it according to brightness of pixels inside vs. outside the colony
    print("  Scoring growth")
    
    nrows = len(rowIndex)
    ncols = len(colIndex)
    
    h,w = plate.shape
    hilighted = np.copy(plate)
    roffsets = np.zeros((nrows,ncols))
    coffsets = np.zeros((nrows,ncols))
    radii = np.zeros((nrows,ncols))
    ccscores = np.zeros((nrows,ncols))
    bgs = np.zeros((nrows,ncols))
    fgs = np.zeros((nrows,ncols))
    bgvars = np.zeros((nrows,ncols))
    fgvars = np.zeros((nrows,ncols))
    for rid,row in enumerate(rowIndex):
        for cid,col in enumerate(colIndex):
            if layout[rid,cid]:
                print("    Working on cell "+str(rid)+"x"+str(cid))
                cell = plate[max(0,row-pad):min(h,row+pad),max(0,col-pad):min(w,col+pad)]
                roffsets[rid,cid],coffsets[rid,cid],radii[rid,cid],ccscores[rid,cid],hili = findColony(cell,rmin,rmax)
                bgs[rid,cid],fgs[rid,cid],bgvars[rid,cid],fgvars[rid,cid] = scoreColony(cell,radii[rid,cid],(roffsets[rid,cid],coffsets[rid,cid]))
                hilighted[max(0,row-pad):min(h,row+pad),max(0,col-pad):min(w,col+pad)] = hili
            
    return(hilighted,roffsets,coffsets,radii,ccscores,bgs,fgs,bgvars,fgvars)

def formatResults(rowIndex,colIndex,layout,roffsets,coffsets,radii,ccscores,bgs,fgs,bgvars,fgvars):
    results = np.empty(np.sum(layout),dtype=([('Strain','a3'),('Row','i8'),('Col','i8'),('PixelRow','i8'),('PixelCol','i8'),('Radius','f8'),('CCScore','f8'),('BgMean','f8'),('FgMean','f8'),('BgVar','f8'),('FgVar','f8')]))
    i = 0
    for row in range(0,len(rowIndex)):
        for col in range(0,len(colIndex)):
            if layout[row,col]:
                results[i]['Strain'] = chr(row+ord("A"))+str(col+1)
                results[i]['Row'] = row
                results[i]['Col'] = col
                results[i]['PixelRow'] = rowIndex[row]+roffsets[row,col]
                results[i]['PixelCol'] = colIndex[col]+coffsets[row,col]
                results[i]['Radius'] = radii[row,col]
                results[i]['CCScore'] = ccscores[row,col]
                results[i]['BgMean'] = bgs[row,col]
                results[i]['FgMean'] = fgs[row,col]
                results[i]['BgVar'] = bgvars[row,col]
                results[i]['FgVar'] = fgvars[row,col]
                i += 1
    return(results)
    
def outputResults(results,filename):
    fo = open(filename,'w')
    for result in results:
        for item in result:
            fo.write(str(item)+" ")
        fo.write("\n")
    fo.close()

### MAIN ###
def __main__():
    parser = argparse.ArgumentParser(description='Assessing colony growth on arrayed plates.')
    parser.add_argument('file',metavar='image_file',help='Image file of colonies arrayed on a plate(s)')
    parser.add_argument('-t','--three',action='store_true',help='Image contains three plates as repeats')
    parser.add_argument('-s','--scan',action='store_true',help='Image was scanned, not photographed')
    parser.add_argument('-b','--blank',metavar='blank_file',help='Image file of a blank plate')
    parser.add_argument('-l','--layout',metavar='layout_file',help='Layout of colonies on the plate')
    parser.add_argument('-r','--radius',metavar='colony_radius',help='Approximate radius of the colonies in pixels',type=int)
    parser.add_argument('--min_r',metavar='min_radius',help='Minimum radius of the colonies in pixels',type=int)
    parser.add_argument('--max_r',metavar='max_radius',help='Maximum radius of the colonies in pixels',type=int)
    parser.add_argument('-x','--xgap',metavar='xgap',help='Horizontal gap between colony centres in pixels',type=int)
    parser.add_argument('-y','--ygap',metavar='ygap',help='Vertical gap between colony centres in pixels',type=int)
    parser.add_argument('-p','--pad',metavar='pad',help='Area to search outside of colony centres in pixels',type=int)
    parser.add_argument('-o','--output',metavar='output_prefix',help='Prefix for output files')

    args = parser.parse_args()

    # Default arguments
    if args.blank is not None:
        blank = io.imread(args.blank,as_gray=True)
    else:
        blank = None
    if args.radius is None:
        args.radius = 30
    if args.xgap is None:
        args.xgap = 100
    if args.ygap is None:
        args.ygap = 100
    if args.pad is None:
        args.pad = args.xgap/2
    if args.min_r is None:
        args.min_r = args.radius/2
    if args.max_r is None:
        args.max_r = args.radius*2

    #Get image name and path for output files
    if args.output is None:
        args.output = os.path.splitext(args.file)[0]

    # Import layout file and determine the number of rows and columns
    if args.layout is not None:
        layout = np.loadtxt(args.layout,dtype='i')
    else:
        layout = np.ones((8,12))
    nrows,ncols = layout.shape

    #Flip, invert, convert to grayscale and normalise levels to between 1% and 99%
    image = io.imread(args.file,as_gray=True)
    if args.scan:
        image = np.fliplr(1-image)
    p1,p99 = np.percentile(image,(1,99))
    image = exposure.rescale_intensity(image,(p1,p99))
    h,w = image.shape

    if args.three:
        plates = trisect(image)
    else:
        plates = [image]

    for pid,plate in enumerate(plates):
        print("Plate "+str(pid+1)+":")
        cropped = cropImage(image,blank)
        io.imsave("{}_{}_cut.png".format(args.output,pid),cropped)
        masked,rowIndex,colIndex = findGrid(cropped,nrows,ncols,layout,r=args.radius,xGap=args.xgap,yGap=args.ygap,pad=args.pad)
        io.imsave("{}_{}_mask.png".format(args.output,pid),masked)
        hilighted,roffsets,coffsets,radii,ccscores,bgs,fgs,bgvars,fgvars = scoreGrowth(cropped,rowIndex,colIndex,layout,rmin=args.min_r,rmax=args.max_r,pad=args.pad)
        io.imsave("{}_{}_hili.png".format(args.output,pid),hilighted)
        results = formatResults(rowIndex,colIndex,layout,roffsets,coffsets,radii,ccscores,bgs,fgs,bgvars,fgvars)
        outputResults(results,"{}_{}.txt".format(args.output,pid))

    return()

if __name__ == "__main__":
    __main__()
