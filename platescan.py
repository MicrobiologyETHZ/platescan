import argparse,os,sys,shutil,scipy
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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


def cropImage(image):
    #Plate edge detection
    h,w = image.shape
    rowMeans = np.zeros(h)
    colMeans = np.zeros(w)
    for i in range(0,h):
        rowMeans[i] = np.mean(image[i,:])
    for i in range(0,w):
        colMeans[i] = np.mean(image[:,i])
    filt = scipy.signal.gaussian(21,2)
    rowMeans = np.pad(scipy.signal.convolve(rowMeans,filt,'valid'),10,'edge')
    colMeans = np.pad(scipy.signal.convolve(colMeans,filt,'valid'),10,'edge')
   
    Lpos,Rpos = np.array_split(scipy.signal.argrelextrema(colMeans,np.greater)[0],2)
    L = np.sort(Lpos[np.argpartition(colMeans[Lpos],-1)[-1:]])[0]
    R = np.sort(Rpos[np.argpartition(colMeans[Rpos],-1)[-1:]])[0]
    Tpos,Bpos = np.array_split(scipy.signal.argrelextrema(rowMeans,np.greater)[0],2)
    T = np.sort(Tpos[np.argpartition(rowMeans[Tpos],-1)[-1:]])[0]
    B = np.sort(Bpos[np.argpartition(rowMeans[Bpos],-1)[-1:]])[0]
    
    plt.figure("R")
    plt.plot(rowMeans,range(0,h))
    plt.plot(rowMeans[Tpos],Tpos,'o')
    plt.plot(rowMeans[Bpos],Bpos,'o')
    plt.savefig("row.png")

    plt.figure("C")
    plt.plot(colMeans)
    plt.plot(Lpos,colMeans[Lpos],'o')
    plt.plot(Rpos,colMeans[Rpos],'o')
    plt.savefig("col.png")

    print("Cropping plate")
    print("  L "+str(L))
    print("  R "+str(R))
    print("  T "+str(T))
    print("  B "+str(B))
   
    return(image[T:B,L:R])
        
def crossCorrelate(a,b,mode='full',boundary='fill'):
    #Perform a normalized cross-correlation of two arrays
    
    aN = (a-np.mean(a))/np.std(a)
    bN = (b-np.mean(b))/np.std(b)
    xcor = scipy.signal.correlate2d(aN,bN,mode=mode,boundary=boundary)
    return(xcor)

def findGrid(plate,grid=384):
    #Make an array of 384 spots to use as a mask, cross-correlate with the plate image and find the best-fitting grid points
    print("  Finding colonies")
    
    r = 5
    pad = 25
    xGap = 52.3
    yGap = 53.1
    
    mask = np.zeros((int((2*pad)+(2*r)+(15*yGap)),int((2*pad)+(2*r)+(23*xGap))))
    for y in range(0,16):
        for x in range(0,24):
            mask[circle(pad+r+round(y*yGap)-1,pad+r+round(x*xGap)-1,r)] = 1
    
    xcor = crossCorrelate(plate,mask,mode='valid')
    offset = np.where(xcor == xcor.max())
    offset = np.array((offset[0][0]+pad,offset[1][0]+pad))
    
    rows = np.array([round(offset[0]+(y*yGap)) for y in range(0,16)])
    cols = np.array([round(offset[1]+(x*xGap)) for x in range(0,24)])
    
    masked = np.copy(plate)
    for y in range(0,16):
        for x in range(0,24):
            masked[circle(offset[0]+r+round(y*yGap)-1,offset[1]+r+round(x*xGap)-1,r)] = 1

    return(masked,rows,cols)

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

def findColony(cell):
    #Find a colony by cross-correlation of ideal circular templates of different radii to create inside and outside regions
    
    h,w = cell.shape
    offsets = []
    scores = []
    for r in np.arange(5,15.5,0.5):
        mask = np.zeros((h,w))
        mask[ellipse(r-1,r-1,r,r)] = 1
        xcor = crossCorrelate(cell,mask,boundary='wrap')
        xcor = xcor[h-1:-(2*r),w-1:-(2*r)] #valid region
        offset = np.unravel_index(xcor.argmax(),xcor.shape)
        offsets.append(offset)
        scores.append(xcor.max())

    best = scores.index(max(scores))
    r = np.arange(5,15.5,0.5)[best]
    offset = offsets[best]
    
    hili = np.copy(cell)
    outside = np.ones((h,w),np.bool)
    outside[ellipse(offset[0]+r-1,offset[1]+r-1,r,r)] = 0
    hili[outside] = hili[outside]/2
    return(offset[0],offset[1],r,max(scores),hili)

def scoreColony(cell,r,offset):
    #Score a colony by summing the pixel array intensities after subtracting the mean intensity of the outside region
    
    h,w = cell.shape
    outside = np.ones((h,w),np.bool)
    outside[ellipse(offset[0]+r-1,offset[1]+r-1,r,r)] = 0    
    
    bg = np.mean(cell[outside])
    fg = np.mean(cell[~outside])
    bgvar = np.var(cell[outside])
    fgvar = np.var(cell[~outside])
    return(bg,fg,bgvar,fgvar)

def scoreGrowth(plate,rows,cols):
    #Find each cvarsolony within its cell, then score it according to brightness of pixels inside vs. outside the colony
    print("  Scoring growth")
    
    h,w = plate.shape
    hilighted = np.copy(plate)
    roffsets = np.zeros((16,24))
    coffsets = np.zeros((16,24))
    radii = np.zeros((16,24))
    ccscores = np.zeros((16,24))
    bgs = np.zeros((16,24))
    fgs = np.zeros((16,24))
    bgvars = np.zeros((16,24))
    fgvars = np.zeros((16,24))
    for rid,row in enumerate(rows):
        for cid,col in enumerate(cols):
            print("    Working on cell "+str(rid)+"x"+str(cid))
            cell = plate[max(0,row-25):min(h,row+25),max(0,col-25):min(w,col+25)]
            
            #A lighting correction may be needed, but it's unclear if that's true and how best to implement it
                        
            roffsets[rid,cid],coffsets[rid,cid],radii[rid,cid],ccscores[rid,cid],hili = findColony(cell)
            bgs[rid,cid],fgs[rid,cid],bgvars[rid,cid],fgvars[rid,cid] = scoreColony(cell,radii[rid,cid],(roffsets[rid,cid],coffsets[rid,cid]))
            hilighted[max(0,row-25):min(h,row+25),max(0,col-25):min(w,col+25)] = hili
            
    return(hilighted,roffsets,coffsets,radii,ccscores,bgs,fgs,bgvars,fgvars)

def formatResults(rows,cols,roffsets,coffsets,radii,ccscores,bgs,fgs,bgvars,fgvars):
    results = np.empty(384,dtype=([('Strain','a3'),('Row','i8'),('Col','i8'),('PixelRow','i8'),('PixelCol','i8'),('Radius','f8'),('CCScore','f8'),('BgMean','f8'),('FgMean','f8'),('BgVar','f8'),('FgVar','f8')]))
    for r in range(0,8):
        for c in range(0,12):
            for rpos,cpos in [(0,0),(0,1),(1,0),(1,1)]:
                n = (((r*12)+c)*4)+((rpos*2)+cpos)
                row,col = (2*r)+rpos,(2*c)+cpos
                results[n]['Strain'] = chr(r+ord("A"))+str(c+1)
                results[n]['Row'] = row
                results[n]['Col'] = col
                results[n]['PixelRow'] = int(max(0,rows[row]-25)+roffsets[row,col]+radii[row,col]-1)
                results[n]['PixelCol'] = int(max(0,cols[col]-25)+coffsets[row,col]+radii[row,col]-1)
                results[n]['Radius'] = radii[row,col]
                results[n]['CCScore'] = ccscores[row,col]
                results[n]['BgMean'] = bgs[row,col]
                results[n]['FgMean'] = fgs[row,col]
                results[n]['BgVar'] = bgvars[row,col]
                results[n]['FgVar'] = fgvars[row,col]
    return(results)
    
def plotResults(radii,deltas,varias,filename):
    #Unused, moved to R
    pdf = PdfPages(filename)

    radiiPlot = plt.figure()
    radiiPlot.suptitle("Colony Radius")
    plt.imshow(radii,interpolation='none')
    radiiPlot.savefig(pdf,format='pdf')
    
    deltasPlot = plt.figure()
    deltasPlot.suptitle("Mean FG - BG")
    plt.imshow(deltas,interpolation='none')
    deltasPlot.savefig(pdf,format='pdf')

    variasPlot = plt.figure()
    variasPlot.suptitle("Variance")
    plt.imshow(varias,interpolation='none')
    variasPlot.savefig(pdf,format='pdf')
    
    pdf.close()

def outputResults(results,filename):
    fo = open(filename,'w')
    for result in results:
        for item in result:
            fo.write(str(item)+" ")
        fo.write("\n")
    fo.close()

### MAIN ###
#def __main__():
if True:
    parser = argparse.ArgumentParser(description='Assessing colony growth on arrayed plates.')
    parser.add_argument('file',metavar='image_file',help='Image file of colonies arrayed on a plate(s)')
    parser.add_argument('-t','--three',action='store_true',help='Image contains three plates as repeats')
    parser.add_argument('-s','--scan',action='store_true',help='Image was scanned, not photographed')
    parser.add_argument('-o','--output',metavar='output_prefix',help='Prefix for output files')

    args = parser.parse_args()

    #Get image name and path for output files
    if args.output is None:
        args.output = os.path.splitext(args.file)[0]

    #Flip, invert, convert to grayscale and normalise levels to between 1% and 99%
    image = io.imread(args.file,as_gray=True)
    if args.scan:
        image = np.fliplr(1-image)
    p1,p99 = np.percentile(image,(1,99))
    image = exposure.rescale_intensity(image,(p1,p99))
    h,w = image.shape

    #Thresholds for growth
    deltaThreshold = 0.1
    zscoreThreshold = 2

    if args.three:
        plates = trisect(image,option)
    else:
        plates = [image]

    for pid,plate in enumerate(plates):
        print("Plate "+str(pid+1)+":")
#        plate = cropImage(plate)
        io.imsave("{}_{}_cut.png".format(args.output,pid),plate)
        interrupt
        masked,rows,cols = findGrid(plate)
        io.imsave("{}_{}_mask.png".format(args.output,pid),masked)
        hilighted,roffsets,coffsets,radii,ccscores,bgs,fgs,bgvars,fgvars = scoreGrowth(plate,rows,cols)
        io.imsave("{}_{}_hili.png".format(args.output,pid),hilighted)
        results = formatResults(rows,cols,roffsets,coffsets,radii,ccscores,bgs,fgs,bgvars,fgvars)
        plotResults(radii,deltas,varias,"{}_{}.pdf".format(args.output,pid))
        outputResults(results,imDir,"{}_{}.txt".format(args.output,pid))

#    return(0)

#if __name__ == "__main__":
#    __main__()

