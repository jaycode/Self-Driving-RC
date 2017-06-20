import numpy as np
import cv2

def annotate_with_lines(background, lines, keep_background=True):
    """ Annotate a background with lines. """    
    ploty = [i for i in range(background.shape[0])]
    # Create an image to draw the lines on
    if keep_background:
        canvas = np.array(background, dtype=np.uint8)
    else:
        canvas = np.zeros_like(background).astype(np.uint8)

    canvas = np.stack((canvas, canvas, canvas), axis=2)
    canvas = (canvas * 255.0).astype(np.uint8)
    
    if 'left' in lines:
        for line in lines['left']:
            if line is not None:
                fitx = line['x']
                pts = np.array([np.transpose(np.vstack([fitx, ploty]))], np.int32)
                cv2.polylines(canvas, np.int_([pts]), False, (255, 255, 0), thickness=3)
            
    if 'right' in lines:
        for line in lines['right']:
            if line is not None:
                fitx = line['x']
                pts = np.array([np.transpose(np.vstack([fitx, ploty]))], np.int32)
                cv2.polylines(canvas, np.int_([pts]), False, (0, 255, 0), thickness=3)

    return canvas

def calc_error(fitx, nonzero, width_total, max_distance=170, top_percentile=75, debug=False):
    """ Calculate the similarity score of line positions and activated pixels.
    
    Error count is MSE.
    Args:
        fitx: The line's x value for each y position.
        nonzero: Non-zero pixels produced by np.nonzero() method.
        width: Image width. Do not count pixels outside the image.
    """
   # Identify the x and y positions of all nonzero pixels in the image
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    total_error = 0
    counter = 0
    if debug:
        print("start error calculation")
    errors =[]
    for y, x in enumerate(fitx):
        # Do not count pixels outside the image.
        if x < 0 or x >= width_total:
            continue
        min_error = None
        line_x = None
        if debug:
            print("y={}".format(y))
        # Indexes of active pixels where y == some value
        for [i] in np.argwhere(nonzeroy==y):
            # Compare the line's x with the pixel's.
            line_x = nonzerox[i]
            error = ((line_x - x)**2)**0.5
            if (min_error is None or error < min_error):
                min_error = error
                
        if min_error is not None:
            errors.append(min_error)
            
        if debug:
            print("line x = {}, point x = {}, error = {}".format(x, line_x, min_error))

    errors = np.array(errors)
    perc = np.percentile(errors, top_percentile)
    if debug:
        print("Remove errors higher than",perc)
    errors = errors[np.where(errors < perc)]
        
    return np.mean(errors)
