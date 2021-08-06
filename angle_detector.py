# %%
from cv2 import (
    imread, imwrite, imshow, waitKey,
    destroyAllWindows, cvtColor,
    inRange, findContours, drawContours,
    matchShapes, ellipse, fitEllipse,
    circle, line, putText,
    COLOR_BGR2GRAY, RETR_EXTERNAL,
    CHAIN_APPROX_NONE, CONTOURS_MATCH_I1,
    FONT_HERSHEY_SIMPLEX, LINE_AA
    )
from numpy import (
    delete, array, argmax, argmin, 
    zeros, shape, vstack, append)
from numpy import degrees, arctan, arctan2, savetxt
from numpy.linalg import norm
from math import sin, cos, radians, degrees
from os import listdir
from os.path import join
from re import findall

# Color space: (B, G, R)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
PURPLE = (255, 0, 255)


def get_midpoint(c1, c2):
    """
    Calculates the (x,y) indices of two given 2D points
    :param c1: coordinates of the first point; [int, int]
    :param c2: coordinates of the second point; [int, int]
    """
    return (int((c1[0]+c2[0])/2), int((c1[1]+c2[1])/2))


def crop_image(img, hor, vert):
    """
    Crops the image by given ratios (0.0~1.0)
    :param img: original cv image source
    :param hor: starting and ending ratios (left to right)
                of the desired horizontal crop;
                [float, float]
    :param hor: starting and ending ratios (top to bottom)
                of the desired vertical crop;
                [float, float]
    """
    height, width = img.shape[0:2]
    hor_left = int(width*hor[0])
    hor_right = int(width*hor[1])
    vert_top = int(height*vert[0])
    vert_bottom = int(height*vert[1])

    crop = img[vert_top:vert_bottom, hor_left:hor_right]
    return crop


def get_binary_brightness(img, bound):
    """
    Convert the pixels to either black or white.
    If a pixel's brightness is between
    the low and high thresholds,
    it converts to white. Otherwise, 
    it is converted to black.
    :param img: original cv image source
    :param bound: low and high thresholds
                  (0=darkest ~ 255=brightest)
    """
    gray = cvtColor(img, COLOR_BGR2GRAY)
    black_or_white = inRange(gray, bound[0], bound[1])
    return black_or_white


def get_contours(img_bin_brt, bounds_length):

    all_contours, _ = findContours(
        img_bin_brt,
        mode=RETR_EXTERNAL,
        method=CHAIN_APPROX_NONE)

    filtered_contours = filter_by_length(
        all_contours, bounds_length)

    return all_contours, filtered_contours


def filter_by_length(contours, length_treshold):
    filtered_contours = []
    thresh_low, thresh_high = length_treshold

    for contour in contours:
        if (thresh_low < len(contour) < thresh_high):
            filtered_contours.append(contour)

    return filtered_contours


def remove_non_eyes(eyes, diff_thresh, bDebug):
    """
    Removes contour(s) that are not likely eye(s)
    by comparing shapes (Hu moments) with each other.
    :param eyes: contours that may be eyes
    :param diff_thresh: Hu distance threshold.
                        If a contour's shape differences
                        to all other contours are bigger
                        than this threshold, that contour
                        is considered not an eye
    """
    if bDebug:
        print(f"{len(eyes)} contours are found as potential eyes...")
        print(f"Hu distance threshold: {diff_thresh}")
    remove_idx = array([], dtype=int)
    for i in range(len(eyes)):
        diff_list = []
        for j in range(len(eyes)):
            if i != j:
                shape_diff = matchShapes(
                    eyes[i], eyes[j], CONTOURS_MATCH_I1, 0)
                diff_list.append(shape_diff)
                if bDebug:
                    print(f"Hu distance of contours ({i, j}): {shape_diff}")
        if all(diff > diff_thresh for diff in diff_list):
            remove_idx = append(remove_idx, i)
            if bDebug:
                print(f"Eye(s) {remove_idx} removed.")

    eyes_filtered = delete(eyes, remove_idx, axis=0)

    return eyes_filtered, len(remove_idx)


def correct_angle(angle):
    """
    Converts counter-clock-wise minor-axis
    angle [degrees] to clock-wise 
    major-axis angle [degrees]
    """
    if angle > 90:
        maj_ax_angle = 180 - (angle - 90)
    else:
        maj_ax_angle = 90 - angle

    return maj_ax_angle


def inscribe_major_axis(result, xc, yc,
                        d1, d2, angle,
                        color, thickness):
    rmajor = max(d1, d2)/2
    xtop = xc - cos(radians(angle))*rmajor
    ytop = yc + sin(radians(angle))*rmajor
    xbot = xc + cos(radians(angle))*rmajor
    ybot = yc - sin(radians(angle))*rmajor
    line(result, (int(xtop), int(ytop)),
         (int(xbot), int(ybot)), color, thickness)


def inscribe_angle(result, angle, center,
                   pos_offset, fontsize,
                   color, thickness):
    xc, yc = center
    offset_x, offset_y = pos_offset
    text = f'{angle:.2f} deg'
    org = (int(xc+offset_x), int(yc+offset_y))  # bottom-left corner of text
    result = putText(result, text, org,
                     FONT_HERSHEY_SIMPLEX, fontsize,
                     color, thickness, LINE_AA)


def remove_false_bladder(bladders, eyes, TIMEBAR_YPOS_THRESH):
    """
    Removes falsely detected bladder by evaluating each
    candidate's distance to the eyes. If the distance is too short,
    it is likely to be an eye, and therefore is removed
    from the list of candidates. The timebar at the top is
    also removed, if falsely detected as a bladder.
    :param bladders: list of possible contours
                     that could be the bladder
    :param eyes: list of the two eyes that are already detected
    :param TIMEBAR_YPOS_THRESH: the y-coordinate position
                                of the timebar
    """
    eye_centers = []
    for eye in eyes:
        eye_center, _, _ = fitEllipse(eye)
        eye_centers.append(array(eye_center))
    eyes_midpoint = get_midpoint(eye_centers[0], eye_centers[1])

    bladder_to_eye_distances = []
    for bladder in bladders:
        bladder_center, _, _ = fitEllipse(bladder)
        dist = norm(
            array(bladder_center)-array(eyes_midpoint),
            ord=2)
        '''
        Excluding the timebar.
        Since the timebar is at the top-left corner,
        the y-coordinate of the bladder's center is evaluated.
        We assume that the bladder never goes higher than the timebar.
        '''
        if bladder_center[1] < TIMEBAR_YPOS_THRESH:
            dist = 0
        bladder_to_eye_distances.append(dist)

    argmax_idx = argmax(bladder_to_eye_distances)

    return bladders[argmax_idx]


def get_frame_no(filename: str) -> int:
    """
    Extracts number from a given filename.
    Error when the filename contains more than one numbers.
    """
    num = findall(r'\d+', filename)
    if len(num) == 1:
        return int(num[0])
    else:
        print("ERROR: Can't retrieve frame number ; \
              filename contains more than one number")
        return None


def alloc_result_space(nFrames):
    # bool, True by default
    out_bDetected = zeros(nFrames) < 1
    # int, from 0 to nFrames-1
    out_frame_no = zeros(nFrames)
    # float, left eye angle [degree]
    out_angle_L = zeros(nFrames)
    # float, right eye angle [degree]
    out_angle_R = zeros(nFrames)
    # float, body eye angle [degree]
    out_angle_B= zeros(nFrames)

    return (out_bDetected, out_frame_no,
            out_angle_L, out_angle_R,
            out_angle_B)

def get_angle2(ref_point, measure_point):
    # since y-axis is flipped in CV.
    yDiff = -(measure_point[1]-ref_point[1])
    xDiff = measure_point[0]-ref_point[0]
    angle = degrees(arctan2(yDiff, xDiff))
    if angle < 0:
        angle += 360
    return angle

def main(IMG_PATH: str,
         TIMEBAR_YPOS_THRESH: int,
         brt_bounds_eye,
         len_bounds_eye,
         brt_bounds_bladder,
         len_bounds_bladder,
         Hu_dist_thresh,
         inscription_pos_offset_eyeL,
         inscription_pos_offset_eyeR,
         inscription_pos_offset_bladder,
         img_input,
         img_output,
         bDebug,
         crop_ratio=-1):
    """
    Inscribes the eyes and body angles onto the pictures
    and also saves as a csv file
    :param IMG_PATH: the parent folder of the source
                     and destination image folders
    :param TIMEBAR_YPOS_THRESH: the y-coordinate position
                                of the timebar
    :param brt_bounds_eye: brightness bounds [low, high] of the eyes
    :param len_bounds_eye: length bounds [low, high] of the eyes
    :param brt_bounds_bladder: brightness bounds [low, high]
                               of the bladder
    :param len_bounds_bladder: length bounds [low, high]
                               of the bladder
    :param img_input: original image file name
    :param img_output: output image file name
    :param crop_ratio: starting and ending ratios
                        (left to right for horizontal,
                        top to bottom for vertical)
                        of the desired crop
    """
    bDetected = True
    img = imread(img_input)
    if crop_ratio != -1:
        if shape(crop_ratio) == (2, 2):
            crop_hor, crop_vert = crop_ratio
            img = crop_image(img, crop_hor, crop_vert)
        else:
            print("Error: Wrong input for crop_ratio."
                    "Refer to crop_image() function input description.")
    
    # Eyes
    img_bin_brt = get_binary_brightness(img, brt_bounds_eye)
    
    if bDebug:
        imshow(
            'Binary image <Eyes>',
            img_bin_brt)
        waitKey(0)
        destroyAllWindows()

    all_cnt_eyes, filtered_cnt_eyes = get_contours(
        img_bin_brt, len_bounds_eye)

    if bDebug:
        print("Length of all contours:")
        for contour in all_cnt_eyes:
            print(len(contour))
        tmp_img = img.copy()
        img_all_contours = drawContours(
            tmp_img, all_cnt_eyes, -1, YELLOW, 3)

        print("Filtering contours with length bounds of ",
              len_bounds_eye, "...")
        for contour in filtered_cnt_eyes:
            print("Length of filtered contours:",
                  len(contour))
        print(f"Total {len(filtered_cnt_eyes)} contours found.")
        img_len_fil_con = drawContours(
            img_all_contours, filtered_cnt_eyes, -1, RED, 2)
        imshow('Length-filtered contours <Eyes>', img_len_fil_con)
        waitKey(0)
        destroyAllWindows()
    
    '''
    When too many eyes are detected,
    remove one by one by comparing shapes
    '''
    while len(filtered_cnt_eyes) > 2:
        filtered_cnt_eyes, removal_count = remove_non_eyes(
            filtered_cnt_eyes, Hu_dist_thresh, bDebug)
        if removal_count==0:
            print("ERROR: failed at removing false eye(s)."
                  "Try adjusting the Hu distance threshold")
            bDetected = False
            return bDetected, 0, 0, 0
    '''
    When less than two eyes are found,
    it is likely that the brightness threshold
    'smudges' two eyes into one lump.
    Especially when the fish is in the middle of
    fast motions, the boundaries of the eyes are blurred.
    Hence, lowering the threshold helps separating
    the blurred lump. Therefore, we apply step-wise
    decrease to the upper brightness bound,
     for better separation of two eyes.
    '''
    tmp_brt_ub = brt_bounds_eye[1]
    while(len(filtered_cnt_eyes) < 2):
        tmp_brt_ub -= 5
        if tmp_brt_ub < 0:
            print(f"ERROR: insufficient eyes detected for {img_input}")
            bDetected = False
            break
        else:
            img_bin_brt = get_binary_brightness(
                img, [brt_bounds_eye[0], tmp_brt_ub])
            
            _, filtered_cnt_eyes = get_contours(img_bin_brt,
                                   len_bounds_eye)
    if not bDetected:
        print(f"ERROR: Failed at detecting 2 eyes for {img_input}")
        return bDetected, 0, 0, 0
    else:
        if bDebug:
            print(f"2 eyes successfully detected for {img_input}")
            tmp_img = img.copy()
            img_len_fil_con = drawContours(
                tmp_img, filtered_cnt_eyes, -1, GREEN, 2)
            imshow(f'Detected eyes for {img_input}', img_len_fil_con)
            waitKey(0)
            destroyAllWindows()

    # Bladder
    img_bin_brt = get_binary_brightness(img, brt_bounds_bladder)
    
    if bDebug:
        imshow(
            'Binary image <Bladder>',
            img_bin_brt)
        waitKey(0)
        destroyAllWindows()

    all_cnt_blad, filtered_cnt_blad = get_contours(
        img_bin_brt, len_bounds_bladder)

    if bDebug:
        for contour in all_cnt_blad:
            print("Length of all contours:", len(contour))
        tmp_img = img.copy()
        img_all_contours = drawContours(
            tmp_img, all_cnt_blad, -1, YELLOW, 3)

        print("Filtering contours with length bounds of ",
              len_bounds_bladder, "...")
        for contour in filtered_cnt_blad:
            print("Length of filtered contours:",
                  len(contour))
        print(f"Total {len(filtered_cnt_blad)} contours found.")
        img_len_fil_con = drawContours(
            img_all_contours, filtered_cnt_blad, -1, RED, 2)
        imshow('Length-filtered contours <Bladder>', img_len_fil_con)
        waitKey(0)
        destroyAllWindows()

    if len(filtered_cnt_blad) != 1:
        if len(filtered_cnt_blad) == 0:
            print(f"ERROR: No bladder detected for {img_input}.")
            bDetected = False
            return bDetected, 0, 0, 0

        elif (len(filtered_cnt_blad) > 1):
            if bDebug:
                print(f"{len(filtered_cnt_blad)} bladder candidates found.",
                "Removing false bladder(s)...")
            filtered_cnt_blad = remove_false_bladder(
                filtered_cnt_blad, filtered_cnt_eyes, TIMEBAR_YPOS_THRESH)
    else:
        filtered_cnt_blad = filtered_cnt_blad[0]
    if not bDetected:
        print(f"ERROR: Failed at detecting a bladder for {img_input}")
        return bDetected, 0, 0, 0
    else:
        if bDebug:
            print(f"Bladder successfully detected for {img_input}")
            tmp_img = img.copy()
            img_len_fil_con = drawContours(
                tmp_img, filtered_cnt_blad, -1, GREEN, 2)
            imshow(f'Detected bladder for {img_input}', img_len_fil_con)
            waitKey(0)
            destroyAllWindows()

    inscribed_img = img.copy()
    eye_centers = []
    eye_angles = []
    for eye in filtered_cnt_eyes:
        my_ellipse = fitEllipse(eye)
        [xc, yc], [d1, d2], min_ax_angle = my_ellipse
        angle = correct_angle(min_ax_angle)
        eye_centers.append([xc, yc])
        eye_angles.append(angle)

        ellipse(inscribed_img, my_ellipse, BLUE, 2)
        circle(inscribed_img, (int(xc), int(yc)),
               2, BLUE, 3)
        inscribe_major_axis(inscribed_img, xc, yc,
                            d1, d2, angle, BLUE, 1)
    
    [xc, yc], [d1, d2], _ = fitEllipse(filtered_cnt_blad)
    bladder_center = (int(xc), int(yc))
    circle(inscribed_img, bladder_center, 2, BLUE, 3)

    point_btwn_eyes = get_midpoint(eye_centers[0], eye_centers[1])
    circle(inscribed_img, point_btwn_eyes, 2, BLUE, 3)

    body_angle = get_angle2(ref_point=bladder_center,
                            measure_point=point_btwn_eyes)

    inscribe_angle(inscribed_img, body_angle, bladder_center,
                   inscription_pos_offset_bladder, 0.5, BLUE, 1)
    line(inscribed_img, point_btwn_eyes, bladder_center, BLUE, 1)

    angles_eye2blad = [0, 0]
    for i, eye_center in enumerate(eye_centers):
        angles_eye2blad[i] = get_angle2(
            ref_point=bladder_center,
            measure_point=eye_center)
        # print(f"angle_eye2blad[{i}]: {angles_eye2blad[i]}]")
    if max(angles_eye2blad) - min(angles_eye2blad) > 180:
        # Positive x-axis caught between the two eyes.
        print("positive x-axis caught between the two eyes.")
        left_eye_idx = argmin(angles_eye2blad)
        right_eye_idx = argmax(angles_eye2blad)
    else:
        left_eye_idx = argmax(angles_eye2blad)
        right_eye_idx = argmin(angles_eye2blad)
    eyeL_angle = eye_angles[left_eye_idx]
    eyeR_angle = eye_angles[right_eye_idx]
    inscribe_angle(inscribed_img, eyeL_angle,
                   eye_centers[left_eye_idx],
                   inscription_pos_offset_eyeL,
                   0.5, BLUE, 1)
    inscribe_angle(inscribed_img, eyeR_angle,
                   eye_centers[right_eye_idx],
                   inscription_pos_offset_eyeR,
                   0.5, BLUE, 1)

    if bDebug:
        imshow("fish_eyes", inscribed_img)
        waitKey(0)
        destroyAllWindows()

    imwrite(img_output, inscribed_img)
    return bDetected, body_angle, eyeL_angle, eyeR_angle

    

if __name__ == "__main__":
    print("WARNING: this is not the main module.")
