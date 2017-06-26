import numpy as np
import cv2

def annotate_with_lines(background, lines, keep_background=True):
    """ Annotate a background with lines. """    
    ploty = [i for i in range(background.shape[0])]
    plotx = [i for i in range(background.shape[1])]
    # Create an image to draw the lines on
    if keep_background:
        canvas = np.array(background, dtype=np.uint8)
    else:
        canvas = np.zeros_like(background).astype(np.uint8)

    # If grey image, convert to three channels
    if len(canvas.shape) == 2 or canvas.shape[2] == 1:
        canvas = np.stack((canvas, canvas, canvas), axis=2)
        canvas = (canvas * 255.0).astype(np.uint8)
    
    if 'left' in lines:
        for line in lines['left']:
            if line is not None:
                fitxy = line['xy']
                if line['orient'] == 'v':

                    pts = np.array([np.transpose(np.vstack([fitxy, ploty]))], np.int32)
                else:
                    pts = np.array([np.transpose(np.vstack([plotx, fitxy]))], np.int32)
                cv2.polylines(canvas, np.int_([pts]), False, (255, 0, 0), thickness=3)
            
    if 'right' in lines:
        for line in lines['right']:
            if line is not None:
                fitxy = line['xy']
                if line['orient'] == 'v':
                    pts = np.array([np.transpose(np.vstack([fitxy, ploty]))], np.int32)
                else:
                    pts = np.array([np.transpose(np.vstack([plotx, fitxy]))], np.int32)
                cv2.polylines(canvas, np.int_([pts]), False, (0, 255, 0), thickness=3)
        
    return canvas

def draw_poly(background, poly_coeffs, orient, degree=2, keep_background=True, color=(0, 255, 0), thickness=3):
    """ Draw (2-degree) polynomial line on an image.
    Args:
      - orient: 'v' or 'h'.
    """
    # Create an image to draw the lines on
    if keep_background:
        canvas = np.array(background, dtype=np.uint8)
    else:
        canvas = np.zeros_like(background).astype(np.uint8)
        
    # If grey image, convert to three channels
    if len(canvas.shape) == 2 or canvas.shape[2] == 1:
        canvas = np.stack((canvas, canvas, canvas), axis=2)
        canvas = (canvas * 255.0).astype(np.uint8)
        
    height = background.shape[0]
    width = background.shape[1]
    xlist = []
    ylist = []
    if orient == 'v':
        for yh in range(height):
            y = height-yh
            x = 0
            for i in range(degree+1):
                x += poly_coeffs[i]*y**(degree-i)
            xlist.append(x)
            ylist.append(yh)
    elif orient == 'h':
        for x in range(width):
            y = 0
            for i in range(degree+1):
                y += poly_coeffs[i]*x**(degree-i)
            ylist.append(y)
            xlist.append(x)
    else:
        raise ValueError("orient must be either 'v' or 'h'.")
    xlist = np.array(xlist, np.int32)
    ylist = np.array(ylist, np.int32)
    pts = np.array([np.transpose(np.vstack([xlist, ylist]))], np.int32)
    cv2.polylines(canvas, np.int_([pts]), False, color, thickness=thickness)
    return canvas