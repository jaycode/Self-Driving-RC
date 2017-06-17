import os, sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

dir_path = os.path.dirname(os.path.realpath(__file__))

# Path to Computer root directory
LIB_DIR = os.path.realpath(os.path.join(dir_path))

sys.path.append(LIB_DIR)
from libraries.line_helpers import calc_error

class FindLinesSlidingWindows(object):
    def __init__(self,
                 debug=False, debug_dir=None, debug_show_lines=False,
                 debug_error_detail=False,
                 nwindows=20,
                 window_minpix=50,
                 error_top_percentile=70,
                 error_threshold=50000,
                 window_patience=2,
                 window_empty_px=4,
                 center_importance=2,
                 crop_top=0.4,
                 lr_start_slack=0.2,
                 subsequent_search_margin=30, always_recalculate=False):
        """ Initialize a Sliding Window line finder.
        
        When `debug` is True, add the following before running the `process()` method:
        >>> fig = plt.figure(figsize=(18, 48))
        >>> a = fig.add_subplot(1, 1, 1)

        Args:
            debug_dir: Set to 'v', 'h', or None. Combined with `debug==True` this will show the sliding windows.
            window_patience: Maximum number of windows with n pixels < window_min_threshold before
                             giving up.
            window_empty_px: Number of pixels inside a window must be larger than this to be considered
                             as non-empty.
            crop_top: When doing horizontal search, do not include a portion of the top section defined by this
                      parameter.
            error_top_percentile: Remove errors higher than this percentile.
                            This is used to remove outlier errors caused by missing lines and not fitting issue.
            lr_start_slack: Left and right lines must come from the edge + this slack * width
        """
        self.reset()     
        
        # Initial search windows
        self.nwindows = nwindows
        self.window_minpix = window_minpix
        
        # Subsequent search related
        self.subsequent_search_margin = subsequent_search_margin
        
        self.debug = debug
        self.debug_dir = debug_dir
        self.debug_show_lines = debug_show_lines
        self.debug_error_detail = debug_error_detail
        
        self.error_threshold = error_threshold    
        self.window_patience = window_patience
        self.window_empty_px = window_empty_px
        self.crop_top = crop_top
        self.lr_start_slack = lr_start_slack
        self.binary_warped = None
        self.error_top_percentile = error_top_percentile
        self.center_importance = center_importance
    def _polyfit(self, x, y):
        # Fit a second order polynomial
        if len(y) > 0 and len(x) > 0:
            fit, error, _, _, _ = np.polyfit(y, x, 2, full=True)
            if len(error) == 0:
                error = 0
            else:
                error = error[0]
            error /= len(y)
        else:
            return None
        if all( i==0 for i in fit ):
            return None
        return (fit, error)
    
    def _calculate_v_fits(self):
        """ Find lines that run vertically on the screen.
        """
        # binary_warped shape: (height, width)
        
        binary_warped = self.binary_warped

        left_lane_inds = []
        right_lane_inds = []
        
        margin = self.subsequent_search_margin

        # Pixels closer to the car are more important, so we apply weights the histogram
        importance = 1
        weights = np.array([range(binary_warped.shape[0])])*importance
        weighted = binary_warped * weights.T
#         weighted = weights.T ** binary_warped
#         weighted = binary_warped

        # For center weights, convert as follows:
        # 0 1 2 3 4 5 6 7 to 1 2 3 4 4 3 2 1
        # (changes: 1, 1, 1, 1, 0, -2, -4. -6)
        # In other words, points closer to the center have higher scores.

        cweights = np.array([range(binary_warped.shape[1])])
        hlen = int(cweights.shape[1]/2) # half-length
        adj = [1] * hlen # adjustments
        for i in range(hlen):
            v = -i*2
            adj.append(v)
        cweights += adj
        weighted *= cweights ** self.center_importance

        # Sums all weighted points in the bottom 50% section (remember that bigger numbers are at the bottom).
#         histogram = np.sum(weighted, axis=0)
        histogram = np.sum(weighted[int(weighted.shape[0]/2):,:], axis=0)
        midpoint = np.int(histogram.shape[0]/2)
        histogram_l = histogram[:(midpoint)]
        histogram_r = histogram[(midpoint+5):]
        
        # === SLIDING WINDOWS ===
        leftx_base = np.argmax(histogram_l) - margin
        rightx_base = np.argmax(histogram_r) + midpoint + margin
        # At this point, leftx_base and rightx_base should contain x position of each respective line.
        window_height = np.int(binary_warped.shape[0]/self.nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Only used for debugging.
        if self.debug:
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        
        left_patience_counter = 0
        right_patience_counter = 0
        # The window takes into account x position found in the previous
        # centroid. It finds a centroid closest to it in the next iteration of window.
        # If there is no centroid x, use window center.
        # Global coordinate is used here.
        left_window_prev_x = leftx_current
        right_window_prev_x = rightx_current
        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            # The higher win_y_low is, the closer to the top of the plot.
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
                        
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            
            # === Select better centroid (Left) ===
            # Find the centroids of pixels in the current window.
            connectivity = 4
            left_pixels_x = nonzerox[good_left_inds]-win_xleft_low
            left_pixels_y = nonzeroy[good_left_inds]-win_y_low
            pixels = np.zeros((window_height, (margin * 2)))

            best_centroid_x, best_pixels_pos = self._choose_best_centroid(
                pixels, leftx_current, win_xleft_low, left_pixels_x, left_pixels_y)

            if best_pixels_pos is not None:
                # Currently, best_pixels_pos contains the most relevant positions.
                # We just need to convert them into good_[left/right]_inds.
                best_pixels_pos += [win_y_low, win_xleft_low]
                bestx = best_pixels_pos[:, 1]
                besty = best_pixels_pos[:, 0]

                good_left_inds = np.intersect1d(np.argwhere(np.in1d(nonzerox,bestx)).flatten(), \
                    np.argwhere(np.in1d(nonzeroy,besty)).flatten()).tolist()
                
            # === END Select better centroid (Left)===

            # === Select better centroid (Right) ===
            # Find the centroids of pixels in the current window.
            connectivity = 4
            right_pixels_x = nonzerox[good_right_inds]-win_xright_low
            right_pixels_y = nonzeroy[good_right_inds]-win_y_low
            pixels = np.zeros((window_height, (margin * 2)))

            best_centroid_x, best_pixels_pos = self._choose_best_centroid(
                pixels, rightx_current, win_xright_low, right_pixels_x, right_pixels_y)

            if best_pixels_pos is not None:
                # Currently, best_pixels_pos contains the most relevant positions.
                # We just need to convert them into good_[left/right]_inds.
                best_pixels_pos += [win_y_low, win_xright_low]
                bestx = best_pixels_pos[:, 1]
                besty = best_pixels_pos[:, 0]

                good_right_inds = np.intersect1d(np.argwhere(np.in1d(nonzerox,bestx)).flatten(), \
                    np.argwhere(np.in1d(nonzeroy,besty)).flatten()).tolist()
                
            # === END Select better centroid (Right)===

#             right_previous_centroid_x, best_pixels_pos = self._choose_best_centroid(
#                 pixels, right_previous_centroid_x, right_pixels_x, right_pixels_y)

            # If any of the left lane pixels touches left section, stop.
            if np.any(nonzerox[good_left_inds] == 0):
                left_patience_counter = self.window_patience+1
            # If any of the right lane pixels touches right section, stop.
            if np.any(nonzerox[good_right_inds] == binary_warped.shape[1]):
                right_patience_counter = self.window_patience+1

            # If sliding windows do not find enough pixels for some iterations, give up.
            if left_patience_counter > self.window_patience:
                pass
            else:
                if len(good_left_inds) < self.window_empty_px:
                    left_patience_counter += 1
                else:
                    left_patience_counter = 0

                    # Append these indices to the lists
                    left_lane_inds.append(good_left_inds)

                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_left_inds) > self.window_minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

                # === DEBUGGING SLIDING WINDOWS ===
                if self.debug:
                    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(255,0,0), 2) 
                # === END DEBUGGING SLIDING WINDOWS ===


            if right_patience_counter > self.window_patience:
                pass
            else:
                if len(good_right_inds) < self.window_empty_px:
                    right_patience_counter += 1
                else:
                    right_patience_counter = 0

                    # Append these indices to the lists
                    right_lane_inds.append(good_right_inds)

                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_right_inds) > self.window_minpix:        
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
                    
                # === DEBUGGING SLIDING WINDOWS ===
                if self.debug:
                    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
                # === END DEBUGGING SLIDING WINDOWS ===

        # === END SLIDING WINDOWS ===
        fits = self._wrap_up_windows(left_lane_inds, right_lane_inds, nonzerox, nonzeroy, 'v')
        
        # === DEBUGGING ===
        if self.debug:
            if self.debug_dir == 'v':
                plt.imshow(out_img, cmap='gray')
            # Normalize histogram values so they don't go beyond image height.
            maxval = np.amax(histogram)
            hist_viz = np.copy(histogram)
            if maxval != 0:
                hist_viz = (hist_viz/maxval) * binary_warped.shape[0]

            # Subtract histogram values from max values so the histogram can be drawn
            # at the bottom of the plot.
            hist_viz = binary_warped.shape[0] - hist_viz
            a.plot(hist_viz, '-', c='#00FFFF', lw=2)

        # === END DEBUGGING ===

        return (fits, histogram_l, histogram_r)

    def _calculate_h_fits(self):
        """ Find lines that run horizontally on the screen.
        """
        
        binary_warped = self.binary_warped

        # binary_warped shape: (height, width)
        left_lane_inds = []
        right_lane_inds = []

        margin = self.subsequent_search_margin
        
        # Distance to bottom edges does not mean anything in horizontal fits.
        weighted = binary_warped

        # y position needs to be flipped for left windows.
        # This was found by visualizing the result and comparing the position of
        # windows vs. histograms.

        y_start_from = int(weighted.shape[0] * self.crop_top)
        histogram_l = np.flip(np.sum(weighted[y_start_from:,:int(weighted.shape[1]/2)], axis=1), axis=0)
        histogram_r = np.sum(weighted[y_start_from:,int(weighted.shape[1]/2):], axis=1)
        
        # === SLIDING WINDOWS ===
        lefty_base = weighted.shape[0] - np.argmax(histogram_l)
        righty_base = y_start_from + np.argmax(histogram_r)
        # At this point, leftx_base and rightx_base should contain y position of each respective line.
        window_height = np.int(binary_warped.shape[1]/self.nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        lefty_current = lefty_base
        righty_current = righty_base

        # Only used for debugging.
        if self.debug:
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

        left_patience_counter = 0
        right_patience_counter = 0
        # Step through the windows one by one
        for window in range(self.nwindows):
            # For right windows, low x means more right, and vice versa for left windows.
            lwin_x_low = (window)*window_height
            rwin_x_low = binary_warped.shape[1] - ((window)*window_height)
            
            # For right windows, high x means more left, and vice versa for left windows.
            lwin_x_high = (window+1)*window_height
            rwin_x_high = binary_warped.shape[1] - ((window+1)*window_height)
            
            # Bottom positions of the windows.
            lwin_y_low = lefty_current + margin
            rwin_y_low = righty_current + margin

            # Top positions of the windows.
            lwin_y_high = lefty_current - margin
            rwin_y_high = righty_current - margin
            
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy <= lwin_y_low) & (nonzeroy > lwin_y_high) & \
                              (nonzerox >= lwin_x_low) & (nonzerox < lwin_x_high)).nonzero()[0]
            good_right_inds = ((nonzeroy <= rwin_y_low) & (nonzeroy > rwin_y_high) & \
                               (nonzerox >= rwin_x_high) & (nonzerox < rwin_x_low)).nonzero()[0]
            
            # If any of the pixels touches top section, stop.
            if np.any(nonzeroy[good_right_inds] == 0):
                right_patience_counter = self.window_patience+1
            if np.any(nonzeroy[good_left_inds] == 0):
                left_patience_counter = self.window_patience+1
            
            # Debugging code. Keep for later.
#             print(nonzeroy, ">=", rwin_y_low, "&", nonzeroy, "<", rwin_y_high, "&",
#                   nonzerox, ">=", lwin_x_high, "&", nonzerox, "<", rwin_x_low)
#             print("(nonzeroy >= rwin_y_low):",np.sum(nonzeroy >= rwin_y_low))
#             print("(nonzeroy < rwin_y_high):",np.sum(nonzeroy < rwin_y_high))
#             print("(nonzerox >= rwin_x_high):",np.sum(nonzerox >= rwin_x_high))
#             print("(nonzerox < rwin_x_low):",np.sum(nonzerox < rwin_x_low))
#             print("rwin_y_low:",rwin_y_low)
#             print(len(good_right_inds))
#             (nonzeroy >= rwin_y_low).nonzero()
            
            # If sliding windows do not find enough pixels for some iterations, give up.
            if left_patience_counter > self.window_patience:
                pass
            else:
                if len(good_left_inds) < self.window_empty_px:
                    left_patience_counter += 1
                else:
                    left_patience_counter = 0

                    # Append these indices to the lists
                    left_lane_inds.append(good_left_inds)

                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_left_inds) > self.window_minpix:
                    lefty_current = np.int(np.mean(nonzeroy[good_left_inds]))
                
                # === DEBUGGING SLIDING WINDOWS ===
                if self.debug:
                    cv2.rectangle(out_img,(lwin_x_low,lwin_y_low),(lwin_x_high,lwin_y_high),(255,0,0), 2) 
                # === END DEBUGGING SLIDING WINDOWS ===

            if right_patience_counter > self.window_patience:
                pass
            else:
                if len(good_right_inds) < self.window_empty_px:
                    right_patience_counter += 1
                else:
                    right_patience_counter = 0

                    # Append these indices to the lists
                    right_lane_inds.append(good_right_inds)

                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_right_inds) > self.window_minpix:        
                    righty_current = np.int(np.mean(nonzeroy[good_right_inds]))

                # === DEBUGGING SLIDING WINDOWS ===
                if self.debug:
                    cv2.rectangle(out_img,(rwin_x_high,rwin_y_low),(rwin_x_low,rwin_y_high),(0,255,0), 2) 
                # === END DEBUGGING SLIDING WINDOWS ===
                    
        # === END SLIDING WINDOWS ===
        fits = self._wrap_up_windows(left_lane_inds, right_lane_inds, nonzerox, nonzeroy, 'h')
       
        # === DEBUGGING ===
        if self.debug:
            if self.debug_dir == 'h':
                plt.imshow(out_img, cmap='gray')

            # LEFT
            # Normalize histogram values so they don't go beyond half of image width.
            maxval = np.amax(histogram_l)
            hist_viz = np.copy(histogram_l)
            if maxval != 0:
                hist_viz = (hist_viz/maxval) * binary_warped.shape[1]/2

            a.plot(hist_viz, list(reversed(range(binary_warped.shape[0])[y_start_from:])), '-', c='#00FFFF', lw=2)

            # RIGHT
            # Normalize histogram values so they don't go beyond half of image width.
            maxval = np.amax(histogram_r)
            hist_viz = np.copy(histogram_r)
            if maxval != 0:
                hist_viz = (hist_viz/maxval) * binary_warped.shape[1]/2        

            # Subtract histogram values from max values so the histogram can be drawn
            # at the right side of the plot.
            hist_viz = (binary_warped.shape[1]) - hist_viz

            a.plot(hist_viz, list(range(binary_warped.shape[0])[y_start_from:]), '-', c='#00FFFF', lw=2)

        # === END DEBUGGING ===
        return (fits, histogram_l, histogram_r)
        
    def _calculate_fits(self):
        """ Try out both horizontal and vertical fits, then pick the best.
        """
        binary_warped = self.binary_warped
        
        if self.debug and self.debug_dir == None:
            plt.imshow(binary_warped, cmap='gray')
            
        # lfits[0]: vertical fits,
        # lfits[1]: horizontal_fits
        vfits, vhist_l, vhist_r = self._calculate_v_fits()
        hfits, hhist_l, hhist_r = self._calculate_h_fits()
        
        lfits = []
        rfits = []
        
        # Comment any of the following lines to debug.
        if 'left' in vfits:
            lfits.append(vfits['left'])
        if 'right' in vfits:
            rfits.append(vfits['right'])
        if 'left' in hfits:
            lfits.append(hfits['left'])
        if 'right' in hfits:
            rfits.append(hfits['right'])
        
        # Calculate mean squared error between lines and activated points.
        # Choose a line with least error for each side.
        
        midpoint = np.int(binary_warped.shape[1]/2)
        selected = []

        # Do the error calculation here.
        # Threshold value found by eyeballing min_error
        # For horizontal fits calculation, use y distance.

        nonzero = binary_warped.nonzero()
        
        best = None
        min_error = None
        for fits in lfits:
            if fits[0] is not None:
                error = calc_error(fits[0], nonzero, binary_warped.shape[1],
                                   top_percentile=self.error_top_percentile, debug=self.debug_error_detail)
                if min_error is None or error < min_error:
                    min_error = error
                    best = fits
        if self.debug:
            not_inc = ""
            if best is None or min_error >= self.error_threshold:
                not_inc = " (not included)"
            print("line 1 error: {}{}".format(min_error, not_inc))
            if best:
                print("line 1 coefs: {}".format(best[1]))
        if self.debug and self.debug_show_lines:
            for fits in lfits:
                if fits[0] is not None:
                    selected.append(fits)
        else:
            if best is not None and min_error < self.error_threshold:
                selected.append(best)
        line1_error = min_error
        
        best = None
        min_error = None
        for fits in rfits:
            if fits[0] is not None:
                error = calc_error(fits[0], nonzero, binary_warped.shape[1],
                                  top_percentile=self.error_top_percentile, debug=self.debug_error_detail)
                if min_error is None or error < min_error:
                    min_error = error
                    best = fits
        if self.debug:
            not_inc = ""
            if best is None or min_error >= self.error_threshold:
                not_inc = " (not included)"
            print("line 2 error: {}{}".format(min_error, not_inc))
            if best:
                print("line 2 coefs: {}".format(best[1]))
            if self.debug_show_lines:
                best = fits
                min_error = 0
        if self.debug and self.debug_show_lines:
            for fits in rfits:
                if fits[0] is not None:
                    selected.append(fits)
        else:
            if best is not None and min_error < self.error_threshold:
                selected.append(best)
        line2_error = min_error
        
        # === Remove converged line(s) ===
        if len(selected) == 2:
            conv = (selected[0][0]-selected[1][0])**2
#             print(conv)
#             print(np.any(conv < 1.0))
            if np.any(conv < 1.0):
                if line1_error > line2_error:
                    del selected[0]
                else:
                    del selected[1]
        # === END - Remove converged line(s) ===
        
        return (selected)
    
    def _wrap_up_windows(self, left_lane_inds, right_lane_inds, nonzerox, nonzeroy, orient):
        """
        orient: 'v' or 'h'
        """
        lane_inds_list = (('left', left_lane_inds), ('right', right_lane_inds))
        fits = {}
        for (key, lane_inds) in lane_inds_list:
            # Concatenate the arrays of indices
            if len(lane_inds) > 0:
                lane_inds = np.concatenate(lane_inds)
                # Extract left and right line pixel positions
                x = nonzerox[lane_inds]
                y = nonzeroy[lane_inds] 
                # binary_warped[nonzeroy[self.left_lane_inds], nonzerox[self.left_lane_inds]] would
                # select all non-zero points. Remember that binary_warped is one dimensional.
                fit, error = self._polyfit(x, y)
                
                # Calculate x of each pixel y position
                if fit is not None:
                    fitx = fit[0]*self.ploty**2 + fit[1]*self.ploty + fit[2]
                    # Checks if line starts at the right location.
                    if orient=='h' and key=='left' and np.min(fitx) > \
                      (0 + self.lr_start_slack * self.binary_warped.shape[1]):
                        fitx = None
                    elif orient=='h' and key=='right' and np.max(fitx) < \
                      (self.binary_warped.shape[1] - self.lr_start_slack * self.binary_warped.shape[1]):
                        fitx = None
                        
                else:
                    fitx = None
                fits[key] = (fitx, fit, lane_inds, error)
            else:
                lane_inds = None
                
        return fits
    
    def _choose_best_centroid(self, pixels, previous_center, global_left, pixels_x, pixels_y):
        """ Choose a centroid that is closest to the previous one, horizontally.
        
        Args:
            - previous_center: Global position of the center of previous window.
            - global_left: Global position of the left-most part of this window.
        
        Returns:
            Tuple of:
            - best_centroid_x: Best centroid's x position.
            - best_pixels_pos[y, x]: Positions of best pixels.
        """
        best_pixels_pos = None
        best_centroid_x = None
        if len(pixels_x) > 0:
            connectivity = 4
            pixels[pixels_y, pixels_x] = 1.0
            pixels = np.uint8(pixels)
            output = cv2.connectedComponentsWithStats(pixels, connectivity, cv2.CV_32S)
            # The first cell is the number of labels
            num_labels = output[0]
            # The second cell is the label matrix
            labels = output[1]
            # Third cell is the centroids
            centroids = output[3]

            min_mse = None
            chosen_label = 0
            for label in range(1, num_labels):
                # Compare the x position of the centroid with previous window's.
                centroid_x_g = (global_left + centroids[label][0])
                mse = (centroid_x_g - previous_center)**2
                if min_mse is None or mse < min_mse:
                    min_mse = mse
                    chosen_label = label
                    best_centroid_x = centroid_x_g
            if chosen_label != 0:
                best_pixels_pos = np.argwhere(labels==chosen_label)
        return (best_centroid_x, best_pixels_pos)
    
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

    def process(self, binary_warped):
        """ Calculate all x positions of lines.
        
        Add the following before running the debugging code:
        >>> fig = plt.figure(figsize=(18, 48))
        >>> a = fig.add_subplot(1, 1, 1)

        Args:
            binary_warped: Warped image.
        
        Returns:
            A list of fits that contains:
            - fit x positions
            - fit coefficients
            - indexes.
        """
        self.windows = []
        # An array of y value from 0 to (image height - 1)
        self.ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        self.binary_warped = binary_warped
        fits = self._calculate_fits()
        return fits