# %%
from cv2 import (
    imread, imwrite, imshow, waitKey,
    destroyAllWindows, cvtColor,
    inRange, findContours, drawContours,
    matchShapes, ellipse, fitEllipse,
    circle, line, putText, contourArea,
    COLOR_BGR2GRAY, RETR_EXTERNAL,
    CHAIN_APPROX_NONE, CONTOURS_MATCH_I1,
    FONT_HERSHEY_SIMPLEX, LINE_AA
    )
from numpy import (
    delete, array, argmax, argmin, 
    zeros, shape, append)
from numpy import degrees, arctan2
from numpy.linalg import norm
from math import sin, cos, radians, degrees
from re import findall

# Color space: (B, G, R)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
PURPLE = (255, 0, 255)


def alloc_result_space(nFrames):
    # bool, True by default
    out_bDetected = zeros(nFrames) < 1
    # int, from 0 to nFrames-1
    out_frame_no = zeros(nFrames)
    # float, body eye angle [degree]
    out_angle_B= zeros(nFrames)
    # float, left eye angle [degree]
    out_angle_L = zeros(nFrames)
    # float, right eye angle [degree]
    out_angle_R = zeros(nFrames)
    # float, left eye area
    out_area_L = zeros(nFrames)
    # float, right eye area
    out_area_R= zeros(nFrames)
    # float, left eye minor axis length
    out_ax_min_L = zeros(nFrames)
    # float, left eye major axis length
    out_ax_maj_L = zeros(nFrames)
    # float, right eye minor axis length
    out_ax_min_R = zeros(nFrames)
    # float, right eye major axis length
    out_ax_maj_R = zeros(nFrames)

    return (out_bDetected, out_frame_no,
            out_angle_B,
            out_angle_L, out_angle_R,
            out_area_L, out_area_R,
            out_ax_min_L, out_ax_maj_L,
            out_ax_min_R, out_ax_maj_R)


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


def get_midpoint(c1, c2):
    """
    Calculates (x,y) indices of Euclidean mean
    of two given 2D points
    :param c1: coordinate of the first point; [int, int]
    :param c2: coordinate of the second point; [int, int]
    """
    return (int((c1[0]+c2[0])/2), int((c1[1]+c2[1])/2))


def crop_image(img, hor, vert):
    """
    Crops image by given ratios (0.0~1.0)
    :param img: original openCV image source
    :param hor: starting and ending horizontal ratios
                (left to right) [float, float]
    :param hor: starting and ending vertical ratios
                (top to bottom) [float, float]
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
    Converts all pixels to either black or white.
    If a pixel's brightness is between
    the given bound, it converts to white.
    Otherwise, it is converted to black.
    :param img: original cv image source
    :param bound: low and high thresholds
                  (0=darkest, 255=brightest)
                  [int, int]
    """
    gray = cvtColor(img, COLOR_BGR2GRAY)
    black_or_white = inRange(gray, bound[0], bound[1])
    return black_or_white


def get_contours(img_bin_brt):
    """
    Get all contours within the length bound.
    :param img_bin_brt: black-and-white image 
    """
    all_contours, _ = findContours(
        img_bin_brt,
        mode=RETR_EXTERNAL,
        method=CHAIN_APPROX_NONE)

    return all_contours


def filter_by_length(contours, length_treshold):
    filtered_contours = []
    thresh_low, thresh_high = length_treshold

    for contour in contours:
        if (thresh_low < len(contour) < thresh_high):
            filtered_contours.append(contour)

    return filtered_contours


def remove_non_eyes(eyes, diff_thresh, debug):
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
    if debug != None:
        print(f"{len(eyes)} contours are found as potential eyes...")
    remove_idx = array([], dtype=int)
    for i in range(len(eyes)):
        diff_list = []
        for j in range(len(eyes)):
            if i != j:
                shape_diff = matchShapes(
                    eyes[i], eyes[j], CONTOURS_MATCH_I1, 0)
                diff_list.append(shape_diff)
                if debug != None:
                    print(f"Hu distance of contours ({i, j}): {shape_diff}")
        if all(diff > diff_thresh for diff in diff_list):
            remove_idx = append(remove_idx, i)
            if debug != None:
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


def inscribe_text(result, text, center,
             pos_offset, fontsize,
             color, thickness):
    xc, yc = center
    offset_x, offset_y = pos_offset
    org = (int(xc+offset_x), int(yc+offset_y))  # bottom-left corner of text
    result = putText(result, text, org,
                     FONT_HERSHEY_SIMPLEX, fontsize,
                     color, thickness, LINE_AA)


def remove_false_bladder(bladders, eyes):
    """
    Removes falsely detected bladder by evaluating each
    candidate's distance to the eyes. If the distance is too short,
    it is likely to be an eye, and therefore is removed
    from the list of candidates. The timebar at the top is
    also removed, if falsely detected as a bladder.
    :param bladders: list of possible contours
                     that could be the bladder
    :param eyes: list of the two eyes that are already detected
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
        bladder_to_eye_distances.append(dist)

    argmax_idx = argmax(bladder_to_eye_distances)

    return bladders[argmax_idx]


def get_angle(ref_point, measure_point):
    # since y-axis is flipped in CV.
    yDiff = -(measure_point[1]-ref_point[1])
    xDiff = measure_point[0]-ref_point[0]
    angle = degrees(arctan2(yDiff, xDiff))
    if angle < 0:
        angle += 360
    return angle


def main(crop_ratio,
         bBladderSkip,
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
         debug):
    """
    Detects and inscribes angles onto the image(s).
    """
    bDetected = True
    font_size = 0.5
    font_thickness = 1
    img = imread(img_input)

    crop_hor, crop_vert = crop_ratio
    img = crop_image(img, crop_hor, crop_vert)
    if debug == "crop":
        # imshow('Cropped image', img)
        # waitKey(0)
        # destroyAllWindows()
        savename = f"{img_input[:-4]}" + "_cropped.png"
        imwrite(savename, img)
        return savename
    
    # Eyes
    img_bin_brt = get_binary_brightness(img, brt_bounds_eye)
    
    if debug == "eye_brt":
        savename = f"{img_input[:-4]}" + "_eyebrt.png"
        imwrite(savename, img_bin_brt)
        return savename
        # imshow(
        #     'Binary image <Eyes>',
        #     img_bin_brt)
        # waitKey(0)
        # destroyAllWindows()

    all_cnt_eyes = get_contours(img_bin_brt)
    filtered_cnt_eyes = filter_by_length(
        all_cnt_eyes, len_bounds_eye)
    if debug == "eye_cnt":
        print("Lengths of all contours:")
        lens = []
        for contour in all_cnt_eyes:
            lens.append(len(contour))
        print(lens)
        tmp_img = img.copy()
        img_all_contours = drawContours(
            tmp_img, all_cnt_eyes, -1, YELLOW, 3)

        print("Filtering contours with length bounds of ",
              len_bounds_eye, "...")
        print("Lengths of filtered contours:")
        lens = []
        for contour in filtered_cnt_eyes:
            lens.append(len(contour))
        print(lens)
        img_len_fil_con = drawContours(
            img_all_contours, filtered_cnt_eyes, -1, RED, 2)
        # imshow('Length-filtered contours <Eyes>', img_len_fil_con)
        # waitKey(0)
        # destroyAllWindows()
        savename = f"{img_input[:-4]}" + "_eyecnt.png"
        imwrite(savename, img_len_fil_con)
        return savename
    
    '''
    When too many eyes are detected,
    remove one by one by comparing shapes
    '''
    while len(filtered_cnt_eyes) > 2:
        filtered_cnt_eyes, removal_count = remove_non_eyes(
            filtered_cnt_eyes, Hu_dist_thresh, debug)
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
        print(f"Insufficient eyes detected for {img_input}. Lowering the brightness upper bound to {tmp_brt_ub}...")
        if tmp_brt_ub < 0:
            print(f"ERROR: Can't detect two eyes for {img_input}")
            bDetected = False
            break
        else:
            img_bin_brt = get_binary_brightness(
                img, [brt_bounds_eye[0], tmp_brt_ub])
            filtered_cnt_eyes = filter_by_length(all_cnt_eyes, len_bounds_eye)
    if not bDetected:
        print(f"ERROR: Failed at detecting 2 eyes for {img_input}")
        return bDetected, 0, 0, 0
    else:
        if debug == "eye_hu":
            print(f"2 eyes successfully detected for {img_input}")
            tmp_img = img.copy()
            img_len_fil_con = drawContours(
                tmp_img, filtered_cnt_eyes, -1, GREEN, 2)
            # imshow(f'Detected eyes for {img_input}', img_len_fil_con)
            # waitKey(0)
            # destroyAllWindows()
            savename = f"{img_input[:-4]}" + "_eyeresult.png"
            imwrite(savename, img_len_fil_con)
            return savename

    # Bladder
    if not bBladderSkip:
        img_bin_brt = get_binary_brightness(
            img, brt_bounds_bladder)
    
        if debug == "blad_brt":
            # imshow(
            #     'Binary image <Bladder>',
            #     img_bin_brt)
            # waitKey(0)
            # destroyAllWindows()
            savename = f"{img_input[:-4]}" + "_bladbrt.png"
            imwrite(savename, img_bin_brt)
            return savename

        all_cnt_blad = get_contours(img_bin_brt)
        filtered_cnt_blad = filter_by_length(
            all_cnt_blad, len_bounds_bladder)

        if debug == "blad_cnt":
            print("Lengths of all contours:")
            lens = []
            for contour in all_cnt_blad:
                lens.append(len(contour))
            print(lens)
            tmp_img = img.copy()
            img_all_contours = drawContours(
                tmp_img, all_cnt_blad, -1, YELLOW, 3)

            print("Filtering contours with length bounds of ",
                len_bounds_bladder, "...")
            print("Lengths of filtered contours:")
            lens = []
            for contour in filtered_cnt_blad:
                lens.append(len(contour))
            print(lens)
            img_len_fil_con = drawContours(
                img_all_contours, filtered_cnt_blad, -1, RED, 2)
            # imshow('Length-filtered contours <Bladder>', img_len_fil_con)
            # waitKey(0)
            # destroyAllWindows()
            savename = f"{img_input[:-4]}" + "_bladcnt.png"
            imwrite(savename, img_len_fil_con)
            return savename

        if len(filtered_cnt_blad) != 1:
            if len(filtered_cnt_blad) == 0:
                print(f"ERROR: No bladder detected for {img_input}.")
                bDetected = False
                return bDetected, 0, 0, 0

            elif (len(filtered_cnt_blad) > 1):
                if debug == "blad_cnt":
                    print(f"{len(filtered_cnt_blad)} bladder candidates found.",
                    "Removing false bladder(s)...")
                filtered_cnt_blad = remove_false_bladder(
                    filtered_cnt_blad, filtered_cnt_eyes)
        else:
            filtered_cnt_blad = filtered_cnt_blad[0]
        if not bDetected:
            print(f"ERROR: Failed at detecting a bladder for {img_input}")
            return bDetected, 0, 0, 0
        else:
            if debug == "blad_cnt":
                print(f"Bladder successfully detected for {img_input}")
                tmp_img = img.copy()
                img_len_fil_con = drawContours(
                    tmp_img, filtered_cnt_blad, -1, GREEN, 2)
                imshow(f'Detected bladder for {img_input}', img_len_fil_con)
                waitKey(0)
                destroyAllWindows()
    else:
        if debug == "blad_brt" or debug == "blad_cnt":
            print("ERROR: Bladder detection disabled by configuration.")
            return img_input

    
    inscribed_img = img.copy()

    eye_centers = []
    eye_angles = []
    eye_areas = []
    eye_ax_mins = []
    eye_ax_majs = []
    for eye in filtered_cnt_eyes:
        my_ellipse = fitEllipse(eye)
        [xc, yc], [d1, d2], min_ax_angle = my_ellipse
        angle = correct_angle(min_ax_angle)
        eye_centers.append([xc, yc])
        eye_angles.append(angle)
        eye_areas.append(contourArea(eye))
        eye_ax_mins.append(d1)
        eye_ax_majs.append(d2)
        ellipse(inscribed_img, my_ellipse, BLUE, 2)
        circle(inscribed_img, (int(xc), int(yc)),
            2, BLUE, 3)
        inscribe_major_axis(inscribed_img, xc, yc,
                            d1, d2, angle, BLUE, 1)
    
    if not bBladderSkip:
        [xc, yc], [d1, d2], _ = fitEllipse(filtered_cnt_blad)
        bladder_center = (int(xc), int(yc))
        circle(inscribed_img, bladder_center, 2, BLUE, 3)

        point_btwn_eyes = get_midpoint(eye_centers[0], eye_centers[1])
        circle(inscribed_img, point_btwn_eyes, 2, BLUE, 3)

        body_angle = get_angle(ref_point=bladder_center,
                                measure_point=point_btwn_eyes)

        inscribe_text(
            inscribed_img,
            f'{body_angle:.2f} deg',
            bladder_center,
            inscription_pos_offset_bladder,
            font_size, BLUE, font_thickness)
        line(inscribed_img, point_btwn_eyes, bladder_center, BLUE, 1)

        angles_eye2blad = [0, 0]
        for i, eye_center in enumerate(eye_centers):
            angles_eye2blad[i] = get_angle(
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
    else:
        # Skipping bladder detection
        body_angle = 0.0
        left_eye_idx = 0
        right_eye_idx = 1
    
    eyeL_angle = eye_angles[left_eye_idx]
    eyeR_angle = eye_angles[right_eye_idx]
    eyeL_area = eye_areas[left_eye_idx]
    eyeR_area = eye_areas[right_eye_idx]
    eyeL_ax_min = eye_ax_mins[left_eye_idx]
    eyeR_ax_min = eye_ax_mins[right_eye_idx]
    eyeL_ax_maj = eye_ax_majs[left_eye_idx]
    eyeR_ax_maj = eye_ax_majs[right_eye_idx]
    if bBladderSkip:
        angle_inscription_L = eyeL_angle
        angle_inscription_R = eyeR_angle
    else:
        angle_inscription_L = -(eyeL_angle-body_angle)
        angle_inscription_R = -(eyeR_angle-body_angle)
    inscribe_text(
        inscribed_img,
        f'{angle_inscription_L:.2f} deg',
        eye_centers[left_eye_idx],
        inscription_pos_offset_eyeL,
        font_size, BLUE, font_thickness)
    inscribe_text(
        inscribed_img,
        f'{angle_inscription_R:.2f} deg',
        eye_centers[right_eye_idx],
        inscription_pos_offset_eyeR,
        font_size, BLUE, font_thickness)
    areaOffsetL = inscription_pos_offset_eyeL.copy()
    areaOffsetL[1] += 20
    inscribe_text(
        inscribed_img,
        f'{eyeL_area:.2f}',
        eye_centers[left_eye_idx],
        areaOffsetL,
        font_size, GREEN, font_thickness)
    areaOffsetR = inscription_pos_offset_eyeR.copy()
    areaOffsetR[1] += 20
    inscribe_text(
        inscribed_img,
        f'{eyeR_area:.2f}',
        eye_centers[right_eye_idx],
        areaOffsetR,
        font_size, GREEN, font_thickness)

    if debug == "all":
        # imshow("fish_eyes", inscribed_img)
        # waitKey(0)
        # destroyAllWindows()
        imwrite(img_output, inscribed_img)
        return img_output

    imwrite(img_output, inscribed_img)
    return (bDetected, body_angle,
            eyeL_angle, eyeR_angle,
            eyeL_area, eyeR_area,
            eyeL_ax_min, eyeL_ax_maj,
            eyeR_ax_min, eyeR_ax_maj)

    

if __name__ == "__main__":
    print("WARNING: this is not the main module.")
