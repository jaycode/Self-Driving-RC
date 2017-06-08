import cv2
import numpy as np
import glob

class ThresholdOperation(object):
    def __init__(self):
        pass
        
class Color(ThresholdOperation):
    def __init__(self, color_channels, gray=True, in_range=None):
        """
        in_range: Tuple with two elements, e.g.
                  ((20, 50, 150), (40, 255, 255))
                  or
                  (60, 255)
                  Can accept lambda too on either element e.g.
                  ((lambda img: int(np.percentile(img, p) - 30)), 255)
        """
        self.color_channels = color_channels
        self.in_range = in_range
        self.gray = gray
        
    def process(self, img):
        nb_ch = len(self.color_channels)
        if len([i for i in ['h', 'l', 's'] if i in self.color_channels]) > 0:
            hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        if len([i for i in ['y', 'u', 'v'] if i in self.color_channels]) > 0:
            yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            yuv = 255-yuv
        if len([i for i in ['h2', 's2', 'v2'] if i in self.color_channels]) > 0:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        ch = np.zeros((*img.shape[:-1], nb_ch))
        for i, color in enumerate(self.color_channels):
            if color == 'r':
                ch[:,:,i] = img[:,:,0]
            if color == 'g':
                ch[:,:,i] = img[:,:,1]
            if color == 'b':
                ch[:,:,i] = img[:,:,2]

            if color == 'h':
                ch[:,:,i] = hls[:,:,0]
            if color == 'l':
                ch[:,:,i] = hls[:,:,1]
            if color == 's':
                ch[:,:,i] = hls[:,:,2]

            if color == 'y':
                ch[:,:,i] = yuv[:,:,0]
            if color == 'u':
                ch[:,:,i] = yuv[:,:,1]
            if color == 'v':
                ch[:,:,i] = yuv[:,:,2]

            if color == 'h2':
                ch[:,:,i] = hsv[:,:,0]
            if color == 's2':
                ch[:,:,i] = hsv[:,:,1]
            if color == 'v2':
                ch[:,:,i] = hsv[:,:,2]
        
        if self.in_range is not None:
            p1 = self.in_range[0]
            p2 = self.in_range[1]
            # Checks if function
            if hasattr(self.in_range[0], '__call__'):
                p1 = self.in_range[0](ch)
            if hasattr(self.in_range[1], '__call__'):
                p2 = self.in_range[1](ch)
            ch = cv2.inRange(ch, p1, p2)
        if (self.in_range is None) and self.gray:
            ch = np.mean(ch, 2)
        return ch
    
class Sobel(ThresholdOperation):
    def __init__(self, axis='x', kernel=3, trange=(30, 130)):
        self.axis = axis
        self.kernel = kernel
        self.trange = trange
        
    def process(self, img):
        result = None
        if self.axis == 'y':
            result = cv2.Sobel(img, -1, 1, 0, ksize=self.kernel)
        else:
            result = cv2.Sobel(img, -1, 0, 1, ksize=self.kernel)
        return np.absolute(result)
            
class Magnitude(ThresholdOperation):
    def __init__(self):
        pass
    
    def process(self, imgs):
        mag = np.sqrt(np.sum(list(map(lambda img: img**2, imgs)), axis=(0)))
        scale_factor = np.max(mag) / 255
        mag = (mag/scale_factor).astype(np.uint8)
        return mag

class Direction(ThresholdOperation):
    def __init__(self, sobel_x_id=0, sobel_y_id=1):
        self.sobel_x_id = sobel_x_id
        self.sobel_y_id = sobel_y_id

    def process(self, imgs):
        if len(imgs) != 2:
            raise(ValueError('Must have two images before running a Direction operation'))
        absgraddir = np.arctan2(np.absolute(imgs[0]), np.absolute(imgs[1]))

        return absgraddir
    
        # Alternative method:
        # with np.errstate(divide='ignore', invalid='ignore'):
        #     abs_grad_dir = np.absolute(np.arctan(imgs[self.sobel_y_id] / imgs[self.sobel_x_id]))
        #     abs_grad_dir[np.isnan(abs_grad_dir)] = np.pi / 2

        # return abs_grad_dir.astype(np.float32)


## -----

TYPE_THRESHOLD = 0
TYPE_AGGREGATOR = 1
TYPE_LINEAR = 2

class Operation(object):
    def __init__(self):
        # 'threshold': Accepts a single image, outputs image to be combined in an array.
        # 'aggregator': Accepts list of images, outputs binary image.
        # 'linear': Accepts a single image, outputs a single image.
        self.type = None
        
        # References parent ImagePipeline object.
        self.ip = None
        
        # Needed so we keep track of what happened to the image after an operation.
        self.img = None
    
class Threshold(Operation):
    def __init__(self, trange=(30, 130),
                 crop_t=0, crop_l=0, crop_r=0, crop_b=0):
        self.type = TYPE_THRESHOLD
        self.operations = []
        self.trange = trange
        self.img = None
        self.crop_t = crop_t
        self.crop_l = crop_l
        self.crop_r = crop_r
        self.crop_b = crop_b
        
    def add(self, threshold_operation):
        self.operations.append(threshold_operation)
        
    def process(self, img):
        imgs = []
        result = None
        for op in self.operations:
            if type(op) == Color:
                res = np.zeros((img.shape[0], img.shape[1]))
                res[self.crop_t:(img.shape[0]-self.crop_b),
                    self.crop_l:(img.shape[1]-self.crop_r)] = op.process(
                        img[self.crop_t:(img.shape[0]-self.crop_b),
                            self.crop_l:(img.shape[1]-self.crop_r),:])
                img = res
            elif type(op) == Sobel:
                img[self.crop_t:(img.shape[0]-self.crop_b),
                    self.crop_l:(img.shape[1]-self.crop_r)] = op.process(
                        img[self.crop_t:(img.shape[0]-self.crop_b),
                            self.crop_l:(img.shape[1]-self.crop_r)])
                imgs.append(img)
            elif type(op) == Magnitude or type(op) == Direction:
                result = op.process(imgs)
        
        if result is None:
            if len(imgs) > 0:
                result = imgs[0]
            else:
                result = img
        
        binary_output =  np.zeros_like(result)
        binary_output[(result >= float(self.trange[0])) & (result <= float(self.trange[1]))] = 1
        return binary_output

class Combinator(Operation):
    def __init__(self, f=(lambda ths: np.where((ths[0] == 1)))):
        """
        combine_function example:
            lambda ths: np.where((ths[0] == 1) | (ths[1] == 1))
        """
        self.type = TYPE_AGGREGATOR
        self.f = f
    
    def process(self, imgs):
        combined = np.zeros_like(imgs[0])
        combined[self.f(imgs)] = 1
        return combined

def distance_between_lanes(leftx, rightx, xm_per_pix=1):
    return abs(rightx - leftx) * xm_per_pix

def calculate_curvature_radius(left_fit, right_fit, leftx, rightx, ploty, xm_per_pix, ym_per_pix):
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad_m = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad_m = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters

    return (left_curverad, right_curverad, left_curverad_m, right_curverad_m)

def lines_are_parallel(line1_fit, line2_fit, threshold=(0, 0)):
    """ Check if two lines are parallel by comparing their coefficients.
    """
    a_diff = np.abs(line1_fit[0] - line2_fit[0])
    b_diff = np.abs(line1_fit[1] - line2_fit[1])
    
    # Keep the print commands for debugging.
#     print("checking parallelism")
#     print("line 1 a: {}, b: {}".format(line1_fit[0], line1_fit[1]))
#     print("line 2 a: {}, b: {}".format(line2_fit[0], line2_fit[1]))
#     print("diff   a: {}, b: {}".format(a_diff, b_diff))
#     print("thres  a: {}, b: {}".format(threshold[0], threshold[1]))
#     print("parallel? {}".format((a_diff < threshold[0] and b_diff < threshold[1])))

    return(a_diff < threshold[0] and b_diff < threshold[1])

def lines_are_plausible(left, right, parallel_threshold=(0,0), dist_threshold=(0,0), minpix=3, xm_per_pix=1):
    """
    left: A tuple of left x pixel positions and polyfit tuple.
    right: A tuple of right x pixel positions and polyfit tuple.
    """
    left_fitx = left[0]
    left_fit = left[1]
    right_fitx = right[0]
    right_fit = right[1]
    is_parallel = lines_are_parallel(left_fit, right_fit, threshold=parallel_threshold)
    dist = distance_between_lanes(left_fitx[-1], right_fitx[-1], xm_per_pix=xm_per_pix)
    is_plausible_dist = dist_threshold[0] < dist < dist_threshold[1]
    
    # The following lines were used to decide on a good dst matrix when doing perspective warp.
#     print(left_fitx.shape)
#     print(left_fitx[-1])
#     print("distance in pixels: {}".format(right_fitx[-1] - left_fitx[-1]))
#     print("distance between lanes: {}".format(dist))
#     print("distance needs to be between: {} and {}".format(dist_threshold[0], dist_threshold[1]))
#     print("distance ok? {}".format(is_plausible_dist))

    return is_parallel & is_plausible_dist

def remove_outliers(x, y, q=5):
    """
    Removes horizontal outliers based on a given percentile.
    :param x: x coordinates of pixels
    :param y: y coordinates of pixels
    :param q: percentile
    :return: cleaned coordinates (x, y)
    """
    if len(x) == 0 or len(y) == 0:
        return x, y

    x = np.array(x)
    y = np.array(y)

    lower_bound = np.percentile(x, q)
    upper_bound = np.percentile(x, 100 - q)
    selection = (x >= lower_bound) & (x <= upper_bound)
    return x[selection], y[selection]

class FindLinesSlidingWindows(Operation):
    def __init__(self,
                 xm_per_pix=3.7/700, ym_per_pix=30/720,
                 parallel_threshold=(0.0003, 0.55),
                 # distance in meters
                 # https://en.wikipedia.org/wiki/Lane#Lane_width
                 # The widths of vehicle lanes typically vary from 9 to 15 feet (2.7 to 4.6 m).
                 dist_threshold=(2.7, 4.6),
                 alpha=1.,
                 nwindows=9,
                 window_minpix=50,
                 lane_minpix=3,
                 subsequent_search_margin=100, always_recalculate=False):
        self.type = TYPE_LINEAR
        self.reset()

        self.xm_per_pix = xm_per_pix
        self.ym_per_pix = ym_per_pix        
        
        # Initial search windows
        self.alpha = alpha
        self.nwindows = nwindows
        self.window_minpix = window_minpix
        
        # Subsequent search related
        self.subsequent_search_margin = subsequent_search_margin
        self.always_recalculate = always_recalculate
        
        # Check if lanes are reasonable
        self.lane_minpix = lane_minpix
        self.parallel_threshold = parallel_threshold
        self.dist_threshold = dist_threshold
    
    
    def _fitpoly(self, leftx, lefty, rightx, righty):
        # Fit a second order polynomial to each
        if len(lefty) > 0 and len(leftx) > 0:
            left_fit = np.polyfit(lefty, leftx, 2)
        else:
            left_fit = [0, 0, 0]
            
        if len(righty) > 0 and len(rightx) > 0:
            right_fit = np.polyfit(righty, rightx, 2)
        else:
            right_fit = [0, 0, 0]
            
        return (left_fit, right_fit)

    def _calculate_fits(self, binary_warped):
        left_lane_inds = []
        right_lane_inds = []
        margin = self.subsequent_search_margin
        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)

        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        # At this point, leftx_base and rightx_base should contain x position of each respective line.
        window_height = np.int(binary_warped.shape[0]/self.nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            self.windows.append({
                'win_y_low': win_y_low,
                'win_y_high': win_y_high,
                'win_xleft_low': win_xleft_low,
                'win_xleft_high': win_xleft_high,
                'win_xright_low': win_xright_low,
                'win_xright_high': win_xright_high
            })

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.window_minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.window_minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # binary_warped[nonzeroy[self.left_lane_inds], nonzerox[self.left_lane_inds]] would
        # select all non-zero points. Remember that binary_warped is one dimensional.

        left_fit, right_fit = self._fitpoly(leftx, lefty, rightx, righty)
        
        # An array of y value from 0 to (image height - 1)
        self.ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

        # Calculate x of each pixel y position
        left_fitx = left_fit[0]*self.ploty**2 + left_fit[1]*self.ploty + left_fit[2]
        right_fitx = right_fit[0]*self.ploty**2 + right_fit[1]*self.ploty + right_fit[2]
            
        return (left_fitx, left_fit, right_fitx, right_fit, left_lane_inds, right_lane_inds)
        
    def _reuse_fits(self, binary_warped):
        left_lane_inds = []
        right_lane_inds = []
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = self.subsequent_search_margin
        left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + \
                                            self.left_fit[1]*nonzeroy + self.left_fit[2] - margin)) & \
                               (nonzerox < (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + \
                                            self.left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + \
                                             self.right_fit[1]*nonzeroy + self.right_fit[2] - margin)) & \
                                (nonzerox < (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + \
                                             self.right_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # binary_warped[nonzeroy[self.left_lane_inds], nonzerox[self.left_lane_inds]] would
        # select all non-zero points. Remember that binary_warped is two dimensional.
        
        left_fit, right_fit = self._fitpoly(leftx, lefty, rightx, righty)
        
        # Generate x and y values for plotting
        self.ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*self.ploty**2 + left_fit[1]*self.ploty + left_fit[2]
        right_fitx = right_fit[0]*self.ploty**2 + right_fit[1]*self.ploty + right_fit[2]
        return (left_fitx, left_fit, right_fitx, right_fit, left_lane_inds, right_lane_inds)
    
    def _check_lines(self, left_fitx, left_fit, right_fitx, right_fit):
        detected = False

        # Compare left with right lines
        if lines_are_plausible((left_fitx, left_fit), (right_fitx, right_fit),
                               parallel_threshold=self.parallel_threshold, dist_threshold=self.dist_threshold,
                               minpix=self.lane_minpix, xm_per_pix=self.xm_per_pix):
            detected = True
        # Compare with previous line.
        elif self.left_fitx is not None and self.right_fitx is not None:
            if lines_are_parallel(left_fit, self.left_fit, threshold=self.parallel_threshold) and \
              lines_are_parallel(right_fit, self.right_fit, threshold=self.parallel_threshold):
                detected = True

        return detected

    def reset(self):
        """ Reset stored variables
        Gives the same effect as recalculating the lines.
        """
        self.left_lane_inds = []
        self.right_lane_inds = []
        self.ploty = None
        self.left_fit = None
        self.right_fit = None
        self.left_fitx = None
        self.right_fitx = None
        self.windows = []
        self.left_curverad = None
        self.right_curverad = None
        self.left_curverad_m = None
        self.right_curverad_m = None
        self.distance_to_center = None
        self.distance_to_center_m = None

    def process(self, binary_warped):
        self.windows = []
        
        if self.always_recalculate:
            self.left_fit = self.right_fit = self.left_fitx = self.right_fitx = None
        
        line_detected = False
        # First, fits from previous frame
        if self.left_fit is not None and self.right_fit is not None:
            left_fitx, left_fit, right_fitx, right_fit, left_lane_inds, right_lane_inds = self._reuse_fits(binary_warped)
            line_detected = self._check_lines(left_fitx, left_fit, right_fitx, right_fit)
        
        # If line not detected, recalculate
        if not line_detected:
            cur_left_fitx, left_fit, cur_right_fitx, right_fit, left_lane_inds, right_lane_inds = self._calculate_fits(binary_warped)
            if self.left_fitx is None or self.always_recalculate:
                left_fitx = cur_left_fitx
                right_fitx = cur_right_fitx
            else:
                prev_left_fitx = np.copy(self.left_fitx)
                prev_right_fitx = np.copy(self.right_fitx)
                left_fitx = prev_left_fitx * (1 - self.alpha) + cur_left_fitx * self.alpha
                right_fitx = prev_right_fitx * (1 - self.alpha) + cur_right_fitx * self.alpha
            line_detected = self._check_lines(left_fitx, left_fit, right_fitx, right_fit)
        
        # If after fitting with previous lanes and recalculating lanes we still cannot
        # detect lines, use results from the previous frame (unless if it is the first frame
        # in which case just use whatever calculation result was)
        self.left_lane_inds = left_lane_inds
        self.right_lane_inds = right_lane_inds
        if not line_detected and (self.left_fit is not None or self.right_fit is not None):
            self.windows = []
        else:
            self.left_fit = left_fit
            self.left_fitx = left_fitx
            self.right_fit = right_fit
            self.right_fitx = right_fitx
            
        self.left_curverad, self.right_curverad, self.left_curverad_m, self.right_curverad_m = \
          calculate_curvature_radius(self.left_fit, self.right_fit, self.left_fitx, self.right_fitx, self.ploty,
                                     self.xm_per_pix, self.ym_per_pix)
        
        # Find distance to center by calculating difference of bottom-most section of the lane lines and
        # then compare it to image center.
        length_of_center_of_lanes = (self.right_fitx[len(self.ploty)-1] - self.left_fitx[len(self.ploty)-1])/2
        distance_to_center_of_lanes = length_of_center_of_lanes + self.left_fitx[len(self.ploty)-1]
        self.distance_to_center = distance_to_center_of_lanes - (binary_warped.shape[1]/2)
        self.distance_to_center_m = self.distance_to_center * self.xm_per_pix        
        return binary_warped
    
class Annotate(Operation):
    def __init__(self, line_finder_obj):
        self.type = TYPE_LINEAR
        self.line_finder_obj = line_finder_obj
        self.map = None
    
    def process(self, img):
        warped = np.copy(self.ip.warped_img)
        undist = np.copy(self.ip.undist_img)
        f = self.line_finder_obj
        left_fitx = f.left_fitx
        right_fitx = f.right_fitx
        ploty = f.ploty
        Minv = self.ip.Minv
        margin = f.subsequent_search_margin

        # Create an image to draw the lines on
        color_warp = np.zeros_like(warped).astype(np.uint8)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))], np.int32)
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        cv2.polylines(color_warp, np.int_([pts_left]), False, (255, 0, 0), thickness=20)
        cv2.polylines(color_warp, np.int_([pts_right]), False, (0, 0, 255), thickness=20)
        
        text0 = "Curvature radius:"
        text = "left: {:.2f}m, right: {:.2f}m".format(f.left_curverad_m, f.right_curverad_m)
        text1 = "Distance from center: {:.2f}m".format(f.distance_to_center_m)
        cv2.putText(undist, text0, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=2)
        cv2.putText(undist, text, (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=2)
        cv2.putText(undist, text1, (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=2)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        
        # Add a small map section 
        minimap = f.img
        nonzero = minimap.nonzero()
        nonzeroy = nonzero[0]
        nonzerox = nonzero[1]
        left_lane_inds = f.left_lane_inds
        right_lane_inds = f.right_lane_inds
        
        # Color in left and right line pixels
        minimap = (np.dstack((minimap, minimap, minimap))*255).astype(np.uint8)
        if not(left_lane_inds.shape[0] == 0 or right_lane_inds.shape[0] == 0):
            minimap[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 255, 0]
            minimap[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 255, 255]
        
        if len(f.windows) > 0:
            for w in f.windows:
                cv2.rectangle(minimap,(w['win_xleft_low'],w['win_y_low']),(w['win_xleft_high'],w['win_y_high']),(0,255,0), 2) 
                cv2.rectangle(minimap,(w['win_xright_low'],w['win_y_low']),(w['win_xright_high'],w['win_y_high']),(0,255,0), 2)
        else:
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the minimap
            window_img = np.zeros_like(minimap)
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
            minimap = cv2.addWeighted(minimap, 1, window_img, 0.3, 0)

        cv2.polylines(minimap, np.int_([pts_left]), False, (255, 0, 0), thickness=10)
        cv2.polylines(minimap, np.int_([pts_right]), False, (0, 0, 255), thickness=10)
        
        self.map = np.copy(minimap)
        
        minimap = cv2.resize(minimap,(int(0.3*minimap.shape[1]), int(0.3*minimap.shape[0])), interpolation = cv2.INTER_CUBIC)
        
        x_offset=850
        y_offset=50
        result[y_offset:y_offset+minimap.shape[0], x_offset:x_offset+minimap.shape[1]] = minimap
        
        return result
    
class ImagePipeline(object):
    def __init__(self, input_color='bgr'):
        self.input_color = input_color
        self.operations = []
        self.warp_src = []
        self.warp_dst = []
        self.warped_img = None
        self.undist_img = None
        
        # Calibration variables
        self.mtx = None
        self.dist = None
        
        # Perspective transform variables
        self.M = None
        self.Minv = None
    
    def calibrate(self, image_paths, cols=9, rows=6):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((rows*cols,3), np.float32)
        objp[:,:2] = np.mgrid[0:cols,0:rows].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Make a list of calibration images
        images = image_paths

        # Step through the list and search for chessboard corners
        i = 0
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (cols,rows),None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (cols,rows), corners, ret)
                i+=1

        ret, self.mtx, self.dist, rvecs, tvecs = \
          cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    def set_perspective(self, src, dst):
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

    def add(self, operation):
        operation.ip = self
        self.operations.append(operation)
        
    def process(self, img):
        """ Process an image
        
        To pass in distorted image, make sure to call `calibrate()` function first.
        """
        imgs = []
        
        if self.input_color == 'bgr':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        self.undist_img = np.copy(img)
        
        if self.M is not None:
            img_size = (img.shape[1], img.shape[0])
            img = cv2.warpPerspective(img, self.M, img_size)
            self.warped_img = np.copy(img)

        for o in self.operations:
            if o.type == TYPE_THRESHOLD:
                res = o.process(img)
                imgs.append(res)
                o.img = np.copy(res)
            elif o.type == TYPE_AGGREGATOR:
                img = o.process(imgs)
                o.img = np.copy(img)
                res = img
            elif o.type == TYPE_LINEAR:
                img = o.process(img)
                o.img = np.copy(img)
                res = img

        return res

def example():
    ip = ImagePipeline(input_color='bgr')
    ip.calibrate(glob.glob('camera_cal/calibration*.jpg'))
    # If mtx and dist have been initialized, we can set them directly.
    # ip.mtx = mtx
    # ip.dist = dist

    src = np.float32([[  100.,   719.],
                      [  542.,   470.],
                      [  738.,   470.],
                      [ 1180.,   719.]])

    dst = np.float32([[ 200.,  720.],
                      [ 200.,    0.],
                      [ 1080.,    0.],
                      [ 1080.,  720.]])

    ip.set_perspective(src, dst)

    t1 = Threshold(trange=(0.3, 1.7))
    t1.add(Color(['l', 'r', 'g']))
    t1.add(Sobel('y', kernel=3))
    t1.add(Sobel('x', kernel=3))
    t1.add(Direction())

    t2 = Threshold(trange=(30, 130))
    t2.add(Color(['l', 'r', 'g']))
    t2.add(Sobel('y', kernel=3))
    t2.add(Magnitude())

    ip.add(t1)
    ip.add(t2)
    c = Combinator(f=(lambda ths: np.where((ths[0] == 1) | (ths[1] == 1))))
    ip.add(c)
    
    f = FindLinesSlidingWindows()
    ip.add(f)
    
    a = Annotate(f)
    ip.add(a)

    test_images = glob.glob('test_images/*.jpg')
    img = cv2.imread(test_images[0])
    
    fig = plt.figure(figsize=(18, 12))
    plt.imshow(ip.process(img))

# example()