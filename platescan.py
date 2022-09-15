import argparse, os, scipy
import numpy as np
import scipy.signal
#import matplotlib.pyplot as plt
from skimage import io, exposure, color
from skimage.draw import disk, ellipse

def trisect(image):
    # Use the mean value of each pixel row to perform auto-correlation and find the repeated plate edge
    print("Trisecting image")
    image_gray = color.rgb2gray(image)
    
    h, w = image_gray.shape
    rowMeans = [np.mean(image_gray[i, :]) for i in range(0, h)]

    # Treat the rowMeans as a signal
    # Use the fast fourier transform and auto-correlate to find repeats
    ft = np.fft.fft(rowMeans)
    cor = np.fft.ifft(ft*np.conjugate(ft)).real
    peakPositions = scipy.signal.argrelextrema(cor, np.greater)[0]
    peakValues = cor[np.array(peakPositions)]
    breakPositions = np.sort(peakPositions[np.argpartition(peakValues, -2)[-2:]])
    height = abs(breakPositions[1]-breakPositions[0])
    images = [image[0:height, 0:w], image[breakPositions[0]:breakPositions[1], 0:w], image[(h-height):h, 0:w]]

    return(images)

def cropImage(image, blank):
    image_gray = color.rgb2gray(image)
    if blank is not None:
        # If a blank is provided, use that to crop the image to the plate
        xcor = crossCorrelate(image_gray, blank)
        offset = np.where(xcor == xcor.max())
        h, w = blank.shape
        L = offset[1][0]
        R = offset[1][0]+w
        T = offset[0][0]
        B = offset[0][0]+h
    else:
        # Plate edge detection; performance highly dependent on lighting and plate type
        h, w = image_gray.shape
        rowMeans = [np.mean(image_gray[i, :]) for i in range(0, h)]
        colMeans = [np.mean(image_gray[:, i]) for i in range(0, w)]
        
        # Smooth curves
        rowMeans = scipy.signal.savgol_filter(rowMeans, int(h/15)+1-int(h/15)%2, 3)
        colMeans = scipy.signal.savgol_filter(colMeans, int(w/15)+1-int(w/15)%2, 3)
        
        Lpos, Rpos = np.array_split(scipy.signal.argrelextrema(colMeans, np.greater)[0], 2)
        L = np.sort(Lpos[np.argpartition(colMeans[Lpos], -1)[-1:]])[0]
        R = np.sort(Rpos[np.argpartition(colMeans[Rpos], -1)[-1:]])[0]
        Tpos, Bpos = np.array_split(scipy.signal.argrelextrema(rowMeans, np.greater)[0], 2)
        T = np.sort(Tpos[np.argpartition(rowMeans[Tpos], -1)[-1:]])[0]
        B = np.sort(Bpos[np.argpartition(rowMeans[Bpos], -1)[-1:]])[0]
    
    print("Cropping plate:")
    print("{}:{} left to right".format(L, R))
    print("{}:{} top to bottom".format(T, B))
    
    return(image[T:B, L:R, :])
        
def crossCorrelate(a, b, boundary=None):
    #Perform a normalized cross-correlation of two arrays
    
    if boundary == 'wrap':
        a = wrapImage(a)
    aN = (a-np.mean(a))/np.std(a)
    bN = (b-np.mean(b))/np.std(b)
    xcor = scipy.signal.correlate(aN, bN, mode='valid', method='fft')
    
    return(xcor)
    
def findGrid(plate, nrows, ncols, layout, r, rmax, xGap, yGap, pad, edge):
    #Make an array of spots to use as a mask, cross-correlate with the plate image and find the best-fitting grid points
    print("  Finding colonies")
    
    plate_gray = color.rgb2gray(plate)
    dim = plate_gray.shape
    mask = -np.ones((int(np.ceil((2*(pad+rmax))+((nrows-1)*yGap))), int(np.ceil((2*(pad+rmax))+((ncols-1)*xGap)))))
    for y in range(0, nrows):
        for x in range(0, ncols):
            if layout[y, x]:
                mask[disk((pad+rmax+np.floor(y*yGap), pad+rmax+np.floor(x*xGap)), r)] = 1

    xcor = crossCorrelate(plate_gray[edge[0]:(dim[0]-edge[2]), edge[1]:(dim[1]-edge[3])], mask)
    offset = np.where(xcor == xcor.max())
    offset = [offset[0][0]+edge[0], offset[1][0]+edge[1]]
    offset = np.array((offset[0]+pad+rmax, offset[1]+pad+rmax))
    
    rowIndex = np.array([round(offset[0]+(y*yGap)) for y in range(0, nrows)], dtype=np.int32)
    colIndex = np.array([round(offset[1]+(x*xGap)) for x in range(0, ncols)], dtype=np.int32)
    
    masked = np.copy(plate)
    for y in range(0, nrows):
        for x in range(0, ncols):
            if layout[y, x]:
                masked[disk((offset[0]+round(y*yGap), offset[1]+round(x*xGap)), r)] = 1

    return(masked, rowIndex, colIndex)

def correctLighting(cell):
    #Correct a lighting gradient across a cell by SVD (not used, not necessary?)
    h, w = cell.shape
    l = np.column_stack((np.repeat(0, h), range(0, h), cell[:, 0]))
    r = np.column_stack((np.repeat(w-1, h), range(0, h), cell[:, -1]))
    t = np.column_stack((range(0, w), np.repeat(0, w), cell[0, :]))
    b = np.column_stack((range(0, w), np.repeat(h-1, w), cell[-1, :]))
    xyz = np.row_stack((l, r, t, b))
    xyz[:, 0] -= np.mean(xyz[:, 0])
    xyz[:, 1] -= np.mean(xyz[:, 1])
    xyz[:, 2] -= np.mean(xyz[:, 2])
    
    u, s, v = np.linalg.svd(xyz)
    dzdx = v[:, -1][0]
    dzdy = v[:, -1][1]
    
    correction = np.zeros((h, w))
    for y in range(0, h):
        for x in range(0, w):
            correction[y, x] = (x * dzdx) + (y * dzdy)
    corrected = cell + correction
    return(corrected)

def wrapImage(image):
    h, w = image.shape

    q1 = image[:int(h/2), :int(w/2)]
    q2 = image[:int(h/2), int(w/2):]
    q3 = image[int(h/2):, :int(w/2)]
    q4 = image[int(h/2):, int(w/2):]
    
    row1 = np.concatenate((q4, q3, q4, q3), axis=1)
    row2 = np.concatenate((q2, q1, q2, q1), axis=1)
    row3 = np.concatenate((q4, q3, q4, q3), axis=1)
    row4 = np.concatenate((q2, q1, q2, q1), axis=1)
    
    wrapped = np.concatenate((row1, row2, row3, row4))
    
    return(wrapped)

def findColony(cell, rmin, rmax, pad):
    #Find a colony by cross-correlation of ideal circular templates of different radii to create inside and outside regions
    
    cell_gray = color.rgb2gray(cell)
    h, w = cell_gray.shape
    
    offsets = []
    scores = []
    for r in np.arange(rmin, rmax, 0.5):
        mask = -np.ones((h, w))
        mask[ellipse(int(h/2), int(w/2), r, r)] = 1
        xcor = crossCorrelate(cell_gray, mask, boundary='wrap')
        offset = np.where(xcor==xcor[int((xcor.shape[0]/2)-pad):int((xcor.shape[0]/2)+pad), int((xcor.shape[1]/2)-pad):int((xcor.shape[1]/2)+pad)].max())
        offset = (offset[0][0], offset[1][0])
        offsets.append(offset)
        scores.append(xcor.max())

    best = scores.index(max(scores))
    r = np.arange(rmin, rmax, 0.5)[best]
    offset = offsets[best]
    
    hili = np.copy(cell)
    outside = np.ones(cell_gray.shape, np.bool)
    outside[ellipse(offset[0], offset[1], r, r)] = 0
    for channel in range(0, 3):
        hili[:, :, channel][outside] = hili[:, :, channel][outside]/2
    return(offset[0], offset[1], r, max(scores), hili)

def scoreColony(cell, channel, r, offset):
    #Score a colony by summing the pixel array intensities after subtracting the mean intensity of the outside region
    
    if channel == 3:
        cell_single = color.rgb2gray(cell)
    else:
        cell_single = cell[:, :, channel]
    h, w = cell_single.shape
    outside = np.ones((h, w), np.bool)
    outside[ellipse(offset[0], offset[1], r, r)] = 0    

    bg = np.mean(cell_single[outside])
    fg = np.mean(cell_single[~outside])
    bgvar = np.var(cell_single[outside])
    fgvar = np.var(cell_single[~outside])
    return(bg, fg, bgvar, fgvar)

def scoreGrowth(plate, rowIndex, colIndex, layout, rmin, rmax, pad):
    #Find each colony within its cell, then score it according to brightness of pixels inside vs. outside the colony
    print("  Scoring growth")
    
    nrows = len(rowIndex)
    ncols = len(colIndex)
    
    h, w, c = plate.shape
    hilighted = np.copy(plate)
    roffsets = np.zeros((nrows, ncols))
    coffsets = np.zeros((nrows, ncols))
    radii = np.zeros((nrows, ncols))
    ccscores = np.zeros((nrows, ncols))
    
    bgs = np.zeros((nrows, ncols, 4))
    fgs = np.zeros((nrows, ncols, 4))
    bgvars = np.zeros((nrows, ncols, 4))
    fgvars = np.zeros((nrows, ncols, 4))

    for rid, row in enumerate(rowIndex):
        for cid, col in enumerate(colIndex):
            if layout[rid, cid]:
                print("    Working on cell "+str(rid)+"x"+str(cid))
                cell = plate[max(0, row-pad-rmax):min(h, row+pad+rmax), max(0, col-pad-rmax):min(w, col+pad+rmax), :]
                roffsets[rid, cid], coffsets[rid, cid], radii[rid, cid], ccscores[rid, cid], hili = findColony(cell, rmin, rmax, pad)
                for channel in range(0, 4):
                    bgs[rid, cid, channel], fgs[rid, cid, channel], bgvars[rid, cid, channel], fgvars[rid, cid, channel] = scoreColony(cell, channel, radii[rid, cid], (roffsets[rid, cid], coffsets[rid, cid]))
                hilighted[max(0, row-pad-rmax):min(h, row+pad+rmax), max(0, col-pad-rmax):min(w, col+pad+rmax), :] = hili
            
    return(hilighted, roffsets, coffsets, radii, ccscores, bgs, fgs, bgvars, fgvars)

def formatResults(rowIndex, colIndex, layout, roffsets, coffsets, radii, ccscores, bgs, fgs, bgvars, fgvars):
    results = np.empty(np.sum(layout), dtype=([('Strain', 'a3'), ('Row', 'i8'), ('Col', 'i8'), ('PixelRow', 'i8'), ('PixelCol', 'i8'), ('Radius', 'f8'), ('CCScore', 'f8'), ('R_BgMean', 'f8'), ('R_FgMean', 'f8'), ('R_BgVar', 'f8'), ('R_FgVar', 'f8'), ('G_BgMean', 'f8'), ('G_FgMean', 'f8'), ('G_BgVar', 'f8'), ('G_FgVar', 'f8'), ('B_BgMean', 'f8'), ('B_FgMean', 'f8'), ('B_BgVar', 'f8'), ('B_FgVar', 'f8'), ('BgMean', 'f8'), ('FgMean', 'f8'), ('BgVar', 'f8'), ('FgVar', 'f8')]))
    i = 0
    for row in range(0, len(rowIndex)):
        for col in range(0, len(colIndex)):
            if layout[row, col]:
                results[i]['Strain'] = chr(row+ord("A"))+str(col+1)
                results[i]['Row'] = row
                results[i]['Col'] = col
                results[i]['PixelRow'] = rowIndex[row]+roffsets[row, col]
                results[i]['PixelCol'] = colIndex[col]+coffsets[row, col]
                results[i]['Radius'] = radii[row, col]
                results[i]['CCScore'] = ccscores[row, col]
                results[i]['R_BgMean'] = bgs[row, col, 0]
                results[i]['R_FgMean'] = fgs[row, col, 0]
                results[i]['R_BgVar'] = bgvars[row, col, 0]
                results[i]['R_FgVar'] = fgvars[row, col, 0]
                results[i]['B_BgMean'] = bgs[row, col, 1]
                results[i]['B_FgMean'] = fgs[row, col, 1]
                results[i]['B_BgVar'] = bgvars[row, col, 1]
                results[i]['B_FgVar'] = fgvars[row, col, 1]
                results[i]['G_BgMean'] = bgs[row, col, 2]
                results[i]['G_FgMean'] = fgs[row, col, 2]
                results[i]['G_BgVar'] = bgvars[row, col, 2]
                results[i]['G_FgVar'] = fgvars[row, col, 2]
                results[i]['BgMean'] = bgs[row, col, 3]
                results[i]['FgMean'] = fgs[row, col, 3]
                results[i]['BgVar'] = bgvars[row, col, 3]
                results[i]['FgVar'] = fgvars[row, col, 3]
                i += 1
    return(results)
    
def outputResults(results, filename):
    fo = open(filename, 'w')
    for result in results:
        for item in result:
            fo.write(str(item)+" ")
        fo.write("\n")
    fo.close()

### MAIN ###
if True:
#def __main__():
    parser = argparse.ArgumentParser(description='Assessing colony growth on arrayed plates.')

    parser.add_argument('file', metavar='image_file', help='Image file of colonies arrayed on a plate(s)')
    parser.add_argument('-t', '--three', action='store_true', help='Image contains three plates as repeats')
    parser.add_argument('-s', '--scan', action='store_true', help='Image was scanned, not photographed')
    parser.add_argument('-b', '--blank', metavar='blank_file', help='Image file of a blank plate')
    parser.add_argument('-l', '--layout', metavar='layout_file', help='Layout of colonies on the plate')
    parser.add_argument('-r', '--radius', default=30, metavar='colony_radius', help='Approximate radius of the colonies in pixels', type=int)
    parser.add_argument('--min_r', metavar='min_radius', help='Minimum radius of the colonies in pixels', type=int)
    parser.add_argument('--max_r', metavar='max_radius', help='Maximum radius of the colonies in pixels', type=int)
    parser.add_argument('-x', '--xgap', default=100, metavar='xgap', help='Horizontal gap between colony centres in pixels', type=int)
    parser.add_argument('-y', '--ygap', default=100, metavar='ygap', help='Vertical gap between colony centres in pixels', type=int)
    parser.add_argument('-p', '--pad', metavar='pad', help='Area to search outside of colony centres in pixels', type=int)
    parser.add_argument('-e', '--edge', default=[0, 0, 0, 0], metavar='edge', nargs=4, help='Plate edge in pixels to be avoided in search (bottom, left, top, right)', type=int)
    parser.add_argument('-o', '--output', metavar='output_prefix', help='Prefix for output files')

    args = parser.parse_args()

    # Default arguments
    if args.blank is not None:
        blank = io.imread(args.blank, as_gray=True)
    else:
        blank = None
    if args.pad is None:
        args.pad = args.xgap/2
    if args.min_r is None:
        args.min_r = args.radius/2
    if args.max_r is None:
        args.max_r = args.radius*2

    # Get image name and path for output files
    if args.output is None:
        args.output = os.path.splitext(args.file)[0]

    # Import layout file and determine the number of rows and columns
    if args.layout is not None:
        layout = np.loadtxt(args.layout, dtype='i')
    else:
        layout = np.ones((8, 12))
    nrows, ncols = layout.shape

    # Import image
    image = io.imread(args.file)

    if args.scan:
        image = np.fliplr(1-image)
    # Slight rescale to improve contrast
    p1, p99 = np.percentile(image, (1, 99))
    image = exposure.rescale_intensity(image, (p1, p99))
    h, w, c = image.shape

    if args.three:
        plates = trisect(image)
    else:
        plates = [image]

    for pid, plate in enumerate(plates):
        print("Plate "+str(pid+1)+":")
        cropped = cropImage(image, blank)
        io.imsave("{}_{}_cut.png".format(args.output, pid), cropped)
        masked, rowIndex, colIndex = findGrid(cropped, nrows, ncols, layout, r=args.radius, rmax=args.max_r, xGap=args.xgap, yGap=args.ygap, pad=args.pad, edge=args.edge)
        io.imsave("{}_{}_mask.png".format(args.output, pid), masked)
        hilighted, roffsets, coffsets, radii, ccscores, bgs, fgs, bgvars, fgvars = scoreGrowth(cropped, rowIndex, colIndex, layout, rmin=args.min_r, rmax=args.max_r, pad=args.pad)
        io.imsave("{}_{}_hili.png".format(args.output, pid), hilighted)
        results = formatResults(rowIndex, colIndex, layout, roffsets, coffsets, radii, ccscores, bgs, fgs, bgvars, fgvars)
        outputResults(results, "{}_{}.txt".format(args.output, pid))

    #return()

#if __name__ == "__main__":
#    __main__()
