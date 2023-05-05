# platescan
Assessing colony growth on arrayed plates.

## How to use

```
positional arguments:
  image_file            Image file of colonies arrayed on a plate(s)

optional arguments:
  -h, --help            show this help message and exit
  -t, --three           Image contains three plates as repeats
  -s, --scan            Image was scanned, not photographed
  -b blank_file, --blank blank_file
                        Image file of a blank plate
  -l layout_file, --layout layout_file
                        Layout of colonies on the plate
  -r colony_radius, --radius colony_radius
                        Approximate radius of the colonies in pixels
  --min_r min_radius    Minimum radius of the colonies in pixels
  --max_r max_radius    Maximum radius of the colonies in pixels
  -x xgap, --xgap xgap  Horizontal gap between colony centres in pixels
  -y ygap, --ygap ygap  Vertical gap between colony centres in pixels
  -p pad, --pad pad     Area to search outside of colony centres in pixels
  -e edge edge edge edge, --edge edge edge edge edge
                        Plate edge in pixels to be avoided in search (bottom,
                        left, top, right)
  -o output_prefix, --output output_prefix
                        Prefix for output files
```

## Requirements
Platescan was run with the following software versions (it may work for earlier or later versions):

| Python | 3.8.6 |
| numpy | 1.19.4 |
| scipy | 1.5.4 |

## Method
The pixel intensities of the plate image are first rescaled to be between the 1st and 99th percentile of all intensities to try to homogenize contrast between different images.

The image is then cropped as follows:
  - If a cropped blank plate image is provided it is positioned by cross-correlation and used to crop the image file
  - If no blank is provided, an attempt is made to find the plate edges based on row and column mean pixel intensities

The colony layout is then located on the image by cross-correlation with a grid of circles determined by xgap, ygap, r, rmax and the layout file if provided (the default layout is otherwise 8 rows x 12 columns).

Each cell in the layout is scanned for a potential colony by cross-correlation with circles of all possible radii and position within the cell.

The program reports for each expected colony:
  - Row
  - Column
  - Colony coordinates on the plate
  - Colony radius
  - Cross-correlation score
  - Foreground (inside the circle) and background (outside) pixel intensity mean and variance in Red, Blue, Green and Greyscale

## Example
Starting with this image:

<img src="examples/IMG_3413.JPG" alt="Image of bacterial colonies growing on a plate" width=48% />

The script was run:

`python platescan.py examples/IMG_3413.JPG -l examples/layout.txt -b examples/blank.jpg -r 20 -x 110 -y 109 -p 10 --min_r 15 --max_r 40 --edge 200 40 200 40`

The image is cut based on a blank:

<img src="examples/blank.jpg" alt="Image of a blank plate" width=48% /> <img src="examples/IMG_3413_0_cut.png" alt="Image of a cropped plate" width=48% />

The colony grid is then found:

<img src="examples/IMG_3413_0_mask.png" alt="Image of a grid of circles over a plate of colonies" width=48% />

Finally the best fit is found for each colony:

<img src="examples/IMG_3413_0_hili.png" alt="Image of a plate of colonies with each highlighted within a square cell" width=48% />
