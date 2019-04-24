import cv2
import numpy as np
from  scipy.ndimage.measurements import center_of_mass


class Gaussian:
    def __init__(self, n=3):
        self.n = n
        self.gauss_kernels = []
        for i in range(self.n):
            span = 2+i
            self.gauss_kernels.append(self._makeGaussian(2*span, span))

    def __call__(self, i):
        if i < 0:
            return self.gauss_kernels[0]
        if i >= len(self.gauss_kernels):
            return self.gauss_kernels[-1]
        else:
            return self.gauss_kernels[i]

    def _makeGaussian(self, size, fwhm=3, center=None):
        """Make a square gaussian kernel.
         size is the length of a side of the square
         fwhm is full-width-at-half-maximum, which
         can be thought of as an effective radius.
         """
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        if center is None:
            x0 = y0 = size // 2
        else:
            x0 = center[0]
            y0 = center[1]
        return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

class MakeTarget:
    def __init__(self, n=5):
        self.gauss_k = Gaussian(n)
        kernel_shape = (5, 5)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_shape)
    
    def __call__(self, trg):
        '''
        Channels:
            [0: mask
             1: boundaries
             2: background
             3: distance transform
             4: 1-distance transform]
        '''
        

        mask = np.float32(trg > 0)
        unique = np.unique(trg)
        mask_op = np.pad(mask, ((1, 1), (1, 1)), mode='constant')
        boundaries = np.zeros_like(mask_op, dtype=np.uint8)  # Hope, 255 ships is the max on an image
        distance = np.zeros_like(mask_op, dtype=np.float32)
        # gauss = np.zeros_like(mask_op, dtype=np.float32)
        for i in unique[1:]:
            ship = np.uint8(trg == i)
            ship = np.pad(ship, ((1, 1), (1, 1)), mode='constant')
            # Boundaries
            dilation = cv2.dilate(ship, self.kernel, iterations=1)
            erosion = cv2.erode(ship, self.kernel, iterations=1)
            boundary = dilation - erosion
            boundaries += boundary.copy()
            
            # Distance transform
            dt = cv2.distanceTransform(ship, distanceType=cv2.DIST_L2, maskSize=5)
            cv2.normalize(dt, dt, 0, 1.0, cv2.NORM_MINMAX)
            distance += dt
            
            # Gaussian center
            # gauss_kernel = self.gauss_k(np.count_nonzero(ship)//50)
            # span = gauss_kernel.shape[0]//2
            # (y, x) = center_of_mass(ship)
            # x = int(x)
            # y = int(y)
            # x_max = min(mask_op.shape[1], x+span)
            # x_max_span = x_max - x
            # x_min = max(0, x-span)
            # x_min_span = x - x_min
    
            # y_max = min(mask_op.shape[0], y+span)
            # y_max_span = y_max - y
            # y_min = max(0, y-span)
            # y_min_span = y - y_min
            # gauss[y_min:y_max, x_min:x_max] +=\
                      # gauss_kernel[(span-y_min_span):(span+y_max_span),
                      # (span-x_min_span):(span+x_max_span)]
        boundaries = np.float32(boundaries > 1.5)
        if unique.shape[0] == 1:
            background = np.ones_like(mask_op, dtype=np.float32)
            inv_distance = np.ones_like(mask_op, dtype=np.float32)
        else:
            background = np.float32(np.ones_like(mask_op, dtype=np.uint8)\
                                    - boundaries - mask_op)
            inv_distance = 1-distance
        #gauss = np.clip(gauss, 0, 1)
        background = np.clip(background, 0, 1)
        boundaries = boundaries[1:-1, 1:-1]
        background = background[1:-1, 1:-1]
        distance = distance[1:-1, 1:-1]
        inv_distance = inv_distance[1:-1, 1:-1]
        #gauss = gauss[1:-1, 1:-1]
        #print(boundaries.min(), boundaries.max(), np.unique(mask), np.unique(background))
        return np.stack([mask, boundaries, background,
                         distance, inv_distance])
    
