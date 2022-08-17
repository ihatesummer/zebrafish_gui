import cv2
import numpy as np
import math

# Color space: (B, G, R)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
PURPLE = (255, 0, 255)


def get_midpoint(point1, point2):
    """
    Calculates (x,y) indices of Euclidean mean
    of two given 2D points.
    - c1: coordinate of the first point. [int, int]
    - c2: coordinate of the second point. [int, int]
    """
    midpoint_x = int((point1[0]+point2[0])/2)
    midpoint_y = int((point1[1]+point2[1])/2)
    return (midpoint_x, midpoint_y)


def crop_image(original_image, bounds_horizontal, bounds_vertical):
    """
    Crops image by given ratios.
    - original_image: original openCV image source
    - bounds_horizontal: horizontal crop-bound ratios.
        left(0.0) to right(1.0). [float, float]
    - bounds_vertical: starting and ending vertical ratios.
        top(0.0) to bottom(1.0). [float, float]
    """
    original_height, original_width = original_image.shape[0:2]

    bound_left = int(original_width*bounds_horizontal[0])
    bound_right = int(original_width*bounds_horizontal[1])
    bound_top = int(original_height*bounds_vertical[0])
    bound_bottom = int(original_height*bounds_vertical[1])

    cropped_image = original_image[bound_top:bound_bottom,
                    bound_left:bound_right]

    return cropped_image


def convert_to_black_or_white(original_image, thresholds):
    """
    Converts all pixels to either black or white.
    If a pixel's brightness is between
    the given thresholds, it converts to white.
    Otherwise, it is converted to black.
    - original_image: original cv image source
    - thresholds: low and high thresholds
                  (0=darkest, 255=brightest)
                  [int, int]
    """
    image_grayscale = cv2.cvtColor(original_image,
                                   cv2.COLOR_BGR2GRAY)
    image_black_or_white = cv2.inRange(image_grayscale,
                                       thresholds[0],
                                       thresholds[1])
    return image_black_or_white


def get_contours(image_black_or_white):
    """
    Get all contours from the given image
    - image_black_or_white: black-or-white image 
    """
    all_contours, _ = cv2.findContours(
        image_black_or_white,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_NONE)
    return all_contours


def filter_contours_by_length(contours, thresholds):
    """
    Filter the contours that fit within the given length bounds
    - contours: list of all contours
    - thresholds: low and high length thresholds [int, int]
    """
    filtered_contours = []
    for contour in contours:
        if (thresholds[0] < len(contour) < thresholds[1]):
            filtered_contours.append(contour)
    return filtered_contours


def remove_non_eyes(
        potential_eyes, difference_threshold):
    """
    Removes contour(s) that are not likely eye(s)
    by comparing shapes with each other.
    - potential_eyes: contours that may be eyes
    - difference_threshold: Hu moment threshold.
        If a contour's shape differences to all 
        other contours are bigger than the threshold,
        that contour is considered not an eye
    """
    # print(f"{len(potential_eyes)} contours are found as potential eyes...")
    n_eyes = len(potential_eyes)
    remove_idx = np.array([], dtype=int)
    for i in range(n_eyes):
        differences = []
        for j in range(n_eyes):
            if i == j:
                continue
            shape_diff = cv2.matchShapes(
                potential_eyes[i], potential_eyes[j],
                cv2.CONTOURS_MATCH_I1, 0)
            differences.append(shape_diff)
            # print(f"Hu distance of contours ({i, j}): {shape_diff}")
        if all((difference > difference_threshold) for
                difference in differences):
            remove_idx = np.append(remove_idx, i)
            # print(f"Eye(s) {remove_idx} removed.")
    eyes_filtered = np.delete(potential_eyes,
                              remove_idx, axis=0)
    return eyes_filtered, len(remove_idx)


def correct_angle(angle):
    """
    Converts counter-clock-wise minor-axis
    angle [degrees] to clock-wise 
    major-axis angle [degrees].
    - angle: original uncorrected angle
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
    xtop = xc - math.cos(np.radians(angle))*rmajor
    ytop = yc + math.sin(np.radians(angle))*rmajor
    xbot = xc + math.cos(np.radians(angle))*rmajor
    ybot = yc - math.sin(np.radians(angle))*rmajor
    cv2.line(result, (int(xtop), int(ytop)),
             (int(xbot), int(ybot)), color, thickness)


def inscribe_text(result, text, center,
                  pos_offset, fontsize,
                  color, thickness):
    xc, yc = center
    offset_x, offset_y = pos_offset
    org = (int(xc+offset_x), int(yc+offset_y))  # bottom-left corner of text
    result = cv2.putText(result, text, org,
                     cv2.FONT_HERSHEY_SIMPLEX, fontsize,
                     color, thickness, cv2.LINE_AA)


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
        eye_center, _, _ = cv2.fitEllipse(eye)
        eye_centers.append(np.array(eye_center))
    eyes_midpoint = get_midpoint(eye_centers[0], eye_centers[1])

    bladder_to_eye_distances = []
    for bladder in bladders:
        bladder_center, _, _ = cv2.fitEllipse(bladder)
        dist = np.linalg.norm(
            np.array(bladder_center)-np.array(eyes_midpoint),
            ord=2)
        bladder_to_eye_distances.append(dist)

    argmax_idx = np.argmax(bladder_to_eye_distances)

    return bladders[argmax_idx]


def get_angle(ref_point, measure_point):
    # since y-axis is flipped in CV.
    yDiff = -(measure_point[1]-ref_point[1])
    xDiff = measure_point[0]-ref_point[0]
    angle = np.degrees(np.arctan2(yDiff, xDiff))
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
    font_size = 0.3
    font_thickness = 1
    img = cv2.imread(img_input)

    crop_hor, crop_vert = crop_ratio
    img = crop_image(img, crop_hor, crop_vert)
    if debug == "crop":
        # cv2.imshow('Cropped image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        savename = f"{img_input[:-4]}" + "_cropped.png"
        cv2.imwrite(savename, img)
        return savename
    
    # Eyes
    img_bin_brt = convert_to_black_or_white(img, brt_bounds_eye)
    
    if debug == "eye_brt":
        savename = f"{img_input[:-4]}" + "_eyebrt.png"
        cv2.imwrite(savename, img_bin_brt)
        return savename
        # cv2.imshow(
        #     'Binary image <Eyes>',
        #     img_bin_brt)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    all_cnt_eyes = get_contours(img_bin_brt)
    filtered_cnt_eyes = filter_contours_by_length(
        all_cnt_eyes, len_bounds_eye)
    if debug == "eye_cnt":
        print("Lengths of all contours:")
        lens = []
        for contour in all_cnt_eyes:
            lens.append(len(contour))
        print(lens)
        tmp_img = img.copy()
        img_all_contours = cv2.drawContours(
            tmp_img, all_cnt_eyes, -1, YELLOW, 3)

        print("Filtering contours with length bounds of ",
              len_bounds_eye, "...")
        print("Lengths of filtered contours:")
        lens = []
        for contour in filtered_cnt_eyes:
            lens.append(len(contour))
        print(lens)
        img_len_fil_con = cv2.drawContours(
            img_all_contours, filtered_cnt_eyes, -1, RED, 2)
        # cv2.imshow('Length-filtered contours <Eyes>', img_len_fil_con)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        savename = f"{img_input[:-4]}" + "_eyecnt.png"
        cv2.imwrite(savename, img_len_fil_con)
        return savename
    
    '''
    When too many eyes are detected,
    remove one by one by comparing shapes
    '''
    while len(filtered_cnt_eyes) > 2:
        filtered_cnt_eyes, removal_count = remove_non_eyes(
            filtered_cnt_eyes, Hu_dist_thresh)
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
            img_bin_brt = convert_to_black_or_white(
                img, [brt_bounds_eye[0], tmp_brt_ub])
            filtered_cnt_eyes = filter_contours_by_length(all_cnt_eyes, len_bounds_eye)
    if not bDetected:
        print(f"ERROR: Failed at detecting 2 eyes for {img_input}")
        return bDetected, 0, 0, 0
    else:
        if debug == "eye_hu":
            print(f"2 eyes successfully detected for {img_input}")
            tmp_img = img.copy()
            img_len_fil_con = cv2.drawContours(
                tmp_img, filtered_cnt_eyes, -1, GREEN, 2)
            # cv2.imshow(f'Detected eyes for {img_input}', img_len_fil_con)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            savename = f"{img_input[:-4]}" + "_eyeresult.png"
            cv2.imwrite(savename, img_len_fil_con)
            return savename

    # Bladder
    if not bBladderSkip:
        img_bin_brt = convert_to_black_or_white(
            img, brt_bounds_bladder)
    
        if debug == "blad_brt":
            # cv2.imshow(
            #     'Binary image <Bladder>',
            #     img_bin_brt)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            savename = f"{img_input[:-4]}" + "_bladbrt.png"
            cv2.imwrite(savename, img_bin_brt)
            return savename

        all_cnt_blad = get_contours(img_bin_brt)
        filtered_cnt_blad = filter_contours_by_length(
            all_cnt_blad, len_bounds_bladder)

        if debug == "blad_cnt":
            print("Lengths of all contours:")
            lens = []
            for contour in all_cnt_blad:
                lens.append(len(contour))
            print(lens)
            tmp_img = img.copy()
            img_all_contours = cv2.drawContours(
                tmp_img, all_cnt_blad, -1, YELLOW, 3)

            print("Filtering contours with length bounds of ",
                len_bounds_bladder, "...")
            lens = []
            for contour in filtered_cnt_blad:
                lens.append(len(contour))
            print(f"Lengths of filtered contours: {lens}")
            img_len_fil_con = cv2.drawContours(
                img_all_contours, filtered_cnt_blad, -1, RED, 2)
            # cv2.imshow('Length-filtered contours <Bladder>', img_len_fil_con)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            savename = f"{img_input[:-4]}" + "_bladcnt.png"
            cv2.imwrite(savename, img_len_fil_con)
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
                img_len_fil_con = cv2.drawContours(
                    tmp_img, filtered_cnt_blad, -1, GREEN, 2)
                # cv2.imshow(f'Detected bladder for {img_input}', img_len_fil_con)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
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
        my_ellipse = cv2.fitEllipse(eye)
        [xc, yc], [d1, d2], min_ax_angle = my_ellipse
        angle = correct_angle(min_ax_angle)
        eye_centers.append([xc, yc])
        eye_angles.append(angle)
        eye_areas.append(cv2.contourArea(eye))
        eye_ax_mins.append(d1)
        eye_ax_majs.append(d2)
        cv2.ellipse(inscribed_img, my_ellipse, BLUE, 2)
        cv2.circle(inscribed_img, (int(xc), int(yc)),
            2, BLUE, 3)
        inscribe_major_axis(inscribed_img, xc, yc,
                            d1, d2, angle, BLUE, 1)
    
    if not bBladderSkip:
        [xc, yc], [d1, d2], _ = cv2.fitEllipse(filtered_cnt_blad)
        bladder_center = (int(xc), int(yc))
        cv2.circle(inscribed_img, bladder_center, 2, BLUE, 3)

        point_btwn_eyes = get_midpoint(eye_centers[0], eye_centers[1])
        cv2.circle(inscribed_img, point_btwn_eyes, 2, BLUE, 3)

        body_angle = get_angle(ref_point=bladder_center,
                                measure_point=point_btwn_eyes)

        inscribe_text(
            inscribed_img,
            f'{body_angle:.2f} deg',
            bladder_center,
            inscription_pos_offset_bladder,
            font_size, BLUE, font_thickness)
        cv2.line(inscribed_img, point_btwn_eyes, bladder_center, BLUE, 1)

        angles_eye2blad = [0, 0]
        for i, eye_center in enumerate(eye_centers):
            angles_eye2blad[i] = get_angle(
                ref_point=bladder_center,
                measure_point=eye_center)
            # print(f"angle_eye2blad[{i}]: {angles_eye2blad[i]}]")
        if max(angles_eye2blad) - min(angles_eye2blad) > 180:
            # Positive x-axis caught between the two eyes.
            print("positive x-axis caught between the two eyes.")
            left_eye_idx = np.argmin(angles_eye2blad)
            right_eye_idx = np.argmax(angles_eye2blad)
        else:
            left_eye_idx = np.argmax(angles_eye2blad)
            right_eye_idx = np.argmin(angles_eye2blad)
    else:
        # Skipping bladder detection
        body_angle = 0.0
        if eye_centers[0][0] < eye_centers[1][0]:
            left_eye_idx = 0
            right_eye_idx = 1
        else:
            left_eye_idx = 1
            right_eye_idx = 0
    
    eyeL_angle = eye_angles[left_eye_idx]
    eyeL_angle = 180 - eyeL_angle # symmetric angle
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
        # cv2.imshow("fish_eyes", inscribed_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(img_output, inscribed_img)
        return img_output

    cv2.imwrite(img_output, inscribed_img)
    return (bDetected, body_angle,
            eyeL_angle, eyeR_angle,
            eyeL_area, eyeR_area,
            eyeL_ax_min, eyeL_ax_maj,
            eyeR_ax_min, eyeR_ax_maj)
    

if __name__ == "__main__":
    print("WARNING: this is not the main module.")
