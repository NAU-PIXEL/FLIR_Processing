import math

import numpy as np
import scipy.ndimage

# considerations for doing this processing.
# the math is straight foward, in Numpy it will be super fast.
## one large 2d convolution, then some array access and means, one big mean, and finish with matrix elementwise subtraction.
# 
# how to make it fast? What will be slow? 
# since this is a linear operation and the conversion to C is non-linear, we have to do this after getting individual temperatures. 
## have each thread that reads an image and does the C conversion on it return the matrix. 
## then stack those matrices, stick them in to the function, and get a stacked array back
# Assume Numpy automatically multithreads operations on different layers of a matrix
# This makes it very important to put in one big matrix and then do the math in it
# 
"""
for k = 1:length(FLIRFiles)
    FileName = fullfile(FLIRDir, FLIRFiles(k).name);
    destripe = [FileName(1:end-5),'_Destriped.tiff'];
    status = copyfile(FileName,destripe,'f');
    t = Tiff(destripe,'r+');
    RAW = single(read(t));
    DS = zeros(size(RAW)); 2d arr empty, used as destripe output target
    xavg = mean(RAW,2); mean along 2nd axis, column averages. 1D
    %The window size (in pixels) is important and should be at least a few times as large as the typical striping artifact.
    windowSize = 31; 
    conv = movmean(xavg,windowSize,'omitnan'); Moving mean convolution, uniform I think?
    for i = 1:size(RAW,1) over very column
        DS(i,:) = store into the first row
          RAW(i,:) original row
            - (xavg(i)-conv(i));  average at column minus convolution at pixel.
            when xavg is high (a stripe) removing the conv doesn't do a lot
            so subtracting the high value removes the stripe
%         %There has been an issue with this code adding stripes as a result of large temperature deviations that are real.
%         %One solution might be to ignore values above or below a certain threthold when calculating the row mean and moving mean
    end
    write(t,single(DS));
    close(t);
end
2 """


def destripe_rows(image_ndarr):
  row_avgs = np.mean(image_ndarr, axis=1)
  # find what the equivalent fill value is
  conv = scipy.ndimage.uniform_filter1d(row_avgs, 31,)
  # have to do this since we collapsed a middle dimension to get it back so we 
  # have broadcastable shapes again.
  conv = np.expand_dims(conv, axis=1)
  image_ndarr = image_ndarr - conv
  
  return image_ndarr

def static_flat_field_correct(image_ndarr):
  # filter each layer with box filter, in reference code it's 191 pix squared
  filtered = scipy.ndimage.uniform_filter(image_ndarr, size=191, mode="nearest", axes=(0,1))

  # get 10 pix squared section of each filtered image, then take mean of that subsection
  mid_x, mid_y = filtered.shape[0] // 2, filtered.shape[1] // 2
  center_temps = filtered[mid_x - 5:mid_x+5, mid_y - 5:mid_y+5, :]
  center_means = np.mean(center_temps, axis=(0,1))

  # get mean of each pixel across every input image
  flat = np.mean(filtered - center_means, axis=2)

  # since flat is 2D and image_ndarr is 3D we have to make them the same dimensions
  image_ndarr = image_ndarr - flat[..., np.newaxis]

  # this is modified in place but also return the reference for niceness of code.
  return image_ndarr