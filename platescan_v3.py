import os,sys,shutil,numpy,scipy
import scipy.signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from skimage import io,exposure
from skimage.draw import circle,ellipse

def trisect(image,option):
    #Use the mean value of each pixel row to perform auto-correlation and find the repeated plate edge
    print "Trisecting image"
    
    h,w = image.shape
    
    #Trisect image by autocorrelation unless option == 'single'
    if(option=='single'):
        print "  Skipping"
        images = [image[0:1000,:],image[0:1000,:],image[0:1000,:]]
    else:
        rowMeans = numpy.zeros(h)
        for i in range(0,h):
            rowMeans[i] = numpy.mean(image[i,:])
        ft = numpy.fft.fft(rowMeans)
        cor = numpy.fft.ifft(ft*numpy.conjugate(ft)).real
        peakPositions = scipy.signal.argrelextrema(cor,numpy.greater)[0]
        peakValues = cor[numpy.array(peakPositions)]
        breakPositions = numpy.sort(peakPositions[numpy.argpartition(peakValues,-2)[-2:]])
        height = abs(breakPositions[1]-breakPositions[0])
        images = [image[0:height,0:w],image[breakPositions[0]:breakPositions[1],0:w],image[(h-height):h,0:w]]

    #Plate edge detection
    for imid,image in enumerate(images):
        h,w = image.shape
        rowMeans = numpy.zeros(h)
        colMeans = numpy.zeros(w)
        for i in range(0,h):
            rowMeans[i] = numpy.mean(image[i,:])
        for i in range(0,w):
            colMeans[i] = numpy.mean(image[:,i])
        filt = scipy.signal.gaussian(21,2)
        rowMeans = numpy.pad(scipy.signal.convolve(rowMeans,filt,'valid'),10,'edge')
        colMeans = numpy.pad(scipy.signal.convolve(colMeans,filt,'valid'),10,'edge')
        
        Lpos,Rpos = numpy.array_split(scipy.signal.argrelextrema(colMeans,numpy.greater)[0],2)
        L = numpy.sort(Lpos[numpy.argpartition(colMeans[Lpos],-1)[-1:]])
        R = numpy.sort(Rpos[numpy.argpartition(colMeans[Rpos],-1)[-1:]])
        Tpos,Bpos = numpy.array_split(scipy.signal.argrelextrema(rowMeans,numpy.greater)[0],2)
        T = numpy.sort(Tpos[numpy.argpartition(rowMeans[Tpos],-1)[-1:]])
        B = numpy.sort(Bpos[numpy.argpartition(rowMeans[Bpos],-1)[-1:]])
        print "Cropping plate "+str(imid+1)
        print "  L "+str(L)
        print "  R "+str(R)
        print "  T "+str(T)
        print "  B "+str(B)
        
        images[imid] = image[T:B,L:R]
        
    #Cut image into three based on the breakpoints and image height, crop the borders to remove the plate edge
    #border = [vb,hb] #vertical,horizontal
    #images = [image[(0+border[0]):(height-border[0]),(0+border[1]):(w-border[1])],
    #        image[(breakPositions[0]+border[0]):(breakPositions[1]-border[0]),(0+border[1]):(w-border[1])],
    #        image[(h-height+border[0]):(h-border[0]),(0+border[1]):(w-border[1])]]
    return(images)

def crossCorrelate(a,b,mode='full',boundary='fill'):
    #Perform a normalized cross-correlation of two arrays
    
    aN = (a-numpy.mean(a))/numpy.std(a)
    bN = (b-numpy.mean(b))/numpy.std(b)
    xcor = scipy.signal.correlate2d(aN,bN,mode=mode,boundary=boundary)
    return(xcor)

def findGrid(plate,option):
    #Make an array of 384 spots to use as a mask, cross-correlate with the plate image and find the best-fitting grid points
    print "  Finding colonies"
    
    r = 5
    if(option=='offcentre'):
        pad = 5
    else:
        pad = 25
    xGap = 52.3
    yGap = 53.1
    
    mask = numpy.zeros(((2*pad)+(2*r)+(15*yGap),(2*pad)+(2*r)+(23*xGap)))
    for y in range(0,16):
        for x in range(0,24):
            mask[circle(pad+r+round(y*yGap)-1,pad+r+round(x*xGap)-1,r)] = 1
    
    xcor = crossCorrelate(plate,mask,mode='valid')
    offset = numpy.where(xcor == xcor.max())
    offset = numpy.array((offset[0][0]+pad,offset[1][0]+pad))
    
    rows = numpy.array([round(offset[0]+(y*yGap)) for y in range(0,16)])
    cols = numpy.array([round(offset[1]+(x*xGap)) for x in range(0,24)])
    
    masked = numpy.copy(plate)
    for y in range(0,16):
        for x in range(0,24):
            masked[circle(offset[0]+r+round(y*yGap)-1,offset[1]+r+round(x*xGap)-1,r)] = 1

    return(masked,rows,cols)

def correctLighting(cell):
    #Correct a lighting gradient across a cell by SVD (not used, not necessary?)
    h,w = cell.shape
    l = numpy.column_stack((numpy.repeat(0,h),range(0,h),cell[:,0]))
    r = numpy.column_stack((numpy.repeat(w-1,h),range(0,h),cell[:,-1]))
    t = numpy.column_stack((range(0,w),numpy.repeat(0,w),cell[0,:]))
    b = numpy.column_stack((range(0,w),numpy.repeat(h-1,w),cell[-1,:]))
    xyz = numpy.row_stack((l,r,t,b))
    xyz[:,0] -= numpy.mean(xyz[:,0])
    xyz[:,1] -= numpy.mean(xyz[:,1])
    xyz[:,2] -= numpy.mean(xyz[:,2])
    
    u,s,v = numpy.linalg.svd(xyz)
    dzdx = v[:,-1][0]
    dzdy = v[:,-1][1]
    
    correction = numpy.zeros((h,w))
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
    for r in numpy.arange(5,15.5,0.5):
        mask = numpy.zeros((h,w))
        mask[ellipse(r-1,r-1,r,r)] = 1
        xcor = crossCorrelate(cell,mask,boundary='wrap')
        xcor = xcor[h-1:-(2*r),w-1:-(2*r)] #valid region
        offset = numpy.unravel_index(xcor.argmax(),xcor.shape)
        offsets.append(offset)
        scores.append(xcor.max())

    best = scores.index(max(scores))
    r = numpy.arange(5,15.5,0.5)[best]
    offset = offsets[best]
    
    hili = numpy.copy(cell)
    outside = numpy.ones((h,w),numpy.bool)
    outside[ellipse(offset[0]+r-1,offset[1]+r-1,r,r)] = 0
    hili[outside] = hili[outside]/2
    return(offset[0],offset[1],r,max(scores),hili)

def scoreColony(cell,r,offset):
    #Score a colony by summing the pixel array intensities after subtracting the mean intensity of the outside region
    
    h,w = cell.shape
    outside = numpy.ones((h,w),numpy.bool)
    outside[ellipse(offset[0]+r-1,offset[1]+r-1,r,r)] = 0    
    
    bg = numpy.mean(cell[outside])
    fg = numpy.mean(cell[~outside])
    bgvar = numpy.var(cell[outside])
    fgvar = numpy.var(cell[~outside])
    return(bg,fg,bgvar,fgvar)

def scoreGrowth(plate,rows,cols):
    #Find each cvarsolony within its cell, then score it according to brightness of pixels inside vs. outside the colony
    print "  Scoring growth"
    
    h,w = plate.shape
    hilighted = numpy.copy(plate)
    roffsets = numpy.zeros((16,24))
    coffsets = numpy.zeros((16,24))
    radii = numpy.zeros((16,24))
    ccscores = numpy.zeros((16,24))
    bgs = numpy.zeros((16,24))
    fgs = numpy.zeros((16,24))
    bgvars = numpy.zeros((16,24))
    fgvars = numpy.zeros((16,24))
    for rid,row in enumerate(rows):
        for cid,col in enumerate(cols):
            print "    Working on cell "+str(rid)+"x"+str(cid)
            cell = plate[max(0,row-25):min(h,row+25),max(0,col-25):min(w,col+25)]
            
            #A lighting correction may be needed, but it's unclear if that's true and how best to implement it
                        
            roffsets[rid,cid],coffsets[rid,cid],radii[rid,cid],ccscores[rid,cid],hili = findColony(cell)
            bgs[rid,cid],fgs[rid,cid],bgvars[rid,cid],fgvars[rid,cid] = scoreColony(cell,radii[rid,cid],(roffsets[rid,cid],coffsets[rid,cid]))
            hilighted[max(0,row-25):min(h,row+25),max(0,col-25):min(w,col+25)] = hili
            
    return(hilighted,roffsets,coffsets,radii,ccscores,bgs,fgs,bgvars,fgvars)

def formatResults(rows,cols,roffsets,coffsets,radii,ccscores,bgs,fgs,bgvars,fgvars):
    results = numpy.empty(384,dtype=([('Strain','a3'),('Row','i8'),('Col','i8'),('PixelRow','i8'),('PixelCol','i8'),('Radius','f8'),('CCScore','f8'),('BgMean','f8'),('FgMean','f8'),('BgVar','f8'),('FgVar','f8')]))
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
if len(sys.argv)<2:
    print 'Usage: python platescan.py image'
    sys.exit(0)
arg = sys.argv[1]
if len(sys.argv)>2:
    option = sys.argv[2] # 'single' if only one plate; 'offcentre' if colonies very close to bright plate edge
else:
    option = "normal"
print "Running as",option

#Get image name and path for output files
imPath = os.path.realpath(arg)
imName = os.path.basename(imPath[:-4])
imDir = os.path.dirname(imPath)

#Flip, invert, convert to grayscale and normalise levels to between 1% and 99%
image = io.imread(imPath,as_grey=True)
image = numpy.fliplr(1-image)
p1,p99 = numpy.percentile(image,(1,99))
image = exposure.rescale_intensity(image,(p1,p99))
h,w = image.shape

#Thresholds for growth
deltaThreshold = 0.1
zscoreThreshold = 2

plates = trisect(image,option)

for pid,plate in enumerate(plates):
    print "Plate "+str(pid+1)+":"
    io.imsave(os.path.join(imDir,imName+"_cut"+str(pid+1)+".png"),plate)
    masked,rows,cols = findGrid(plate,option)
    io.imsave(os.path.join(imDir,imName+"_grid"+str(pid+1)+".png"),masked)
    hilighted,roffsets,coffsets,radii,ccscores,bgs,fgs,bgvars,fgvars = scoreGrowth(plate,rows,cols)
    io.imsave(os.path.join(imDir,imName+"_hili"+str(pid+1)+".png"),hilighted)
    results = formatResults(rows,cols,roffsets,coffsets,radii,ccscores,bgs,fgs,bgvars,fgvars)
    #plotResults(radii,deltas,varias,os.path.join(imDir,imName+"_plots"+str(pid+1)+".pdf"))
    outputResults(results,os.path.join(imDir,imName+"_results"+str(pid+1)+".txt"))

