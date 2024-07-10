import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import time

# Number of matches to retain (can be adjusted as needed)
NUM_MATCHES = 50

def resize_image(img, new_width):
    """Resizes the image to a new width while maintaining aspect ratio.
    
    Args:
        img (np.ndarray): The input image.
        new_width (int): The desired width of the resized image.
    
    Returns:
        np.ndarray: The resized image.
    """
    ratio = float(new_width) / img.shape[1]
    new_height = int(img.shape[0] * ratio)
    return cv2.resize(img, (new_width, new_height))

def harris_corner_detection(img):
    """Performs Harris Corner Detection on an image.
    
    Args:
        img (np.ndarray): The input image.
    
    Returns:
        tuple: A tuple containing:
            - harris_img (np.ndarray): The image with detected corners highlighted.
            - keypoints (list of cv2.KeyPoint): The list of keypoints detected in the image.
    """
    harris_img = np.copy(img)
    gray = cv2.cvtColor(harris_img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 9, 0.06)
    
    dst = cv2.dilate(dst, None)
    harris_img[dst > 0.15 * dst.max()] = [0, 0, 255]
    
    rows, cols, depths = np.where(harris_img==[0, 0, 255])
    keypoints = [cv2.KeyPoint(np.float32(cols[i]), np.float32(rows[i]), 5) for i in range(len(rows))]

    print(len(keypoints))
    
    return harris_img, keypoints

def initialise_sift():
    """Initializes and returns a SIFT detector with custom parameters.
    
    Returns:
        cv2.SIFT: The SIFT detector object.
    """
    sift = cv2.SIFT_create()
    sift.setContrastThreshold(0.125)
    sift.setEdgeThreshold(10)
    return sift
    
def sift_feature_detection(img):
    """Detects features in an image using SIFT.
    
    Args:
        img (np.ndarray): The input image.
    
    Returns:
        list of cv2.KeyPoint: The list of keypoints detected in the image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = initialise_sift()
    start_time = time.time()
    keypoints = sift.detect(gray, None)
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken for SIFT detection: {time_taken} seconds")
    return keypoints

def sift_descriptor(img, keypoints):
    """Computes SIFT descriptors for the given keypoints in an image.
    
    Args:
        img (np.ndarray): The input image.
        keypoints (list of cv2.KeyPoint): The list of keypoints for which descriptors are to be computed.
    
    Returns:
        tuple: A tuple containing:
            - keypoints (list of cv2.KeyPoint): The list of keypoints after descriptor computation.
            - desc (np.ndarray): The array of computed descriptors.
    """
    start_time = time.time()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create a SIFT detector object
    sift = initialise_sift()
    
    # Compute the descriptors for the detected keypoints
    keypoints, desc = sift.compute(gray, keypoints)
    
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Execution time for SIFT descriptor: {time_taken} seconds")
    
    # Print the descriptors
    #print(desc)
    
    return keypoints, desc

def binary_descriptor(img, keypoints):
    """Computes binary descriptors (BRIEF) for the given keypoints in an image.
    
    Args:
        img (np.ndarray): The input image.
        keypoints (list of cv2.KeyPoint): The list of keypoints for which descriptors are to be computed.
    
    Returns:
        tuple: A tuple containing:
            - keypoints (list of cv2.KeyPoint): The list of keypoints after descriptor computation.
            - desc (np.ndarray): The array of computed descriptors.
    """
    start_time = time.time()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    keypoints, desc = brief.compute(gray, keypoints)

    end_time = time.time()
    execution_time = end_time - start_time
    print("Binary Descriptor Execution Time:", execution_time, "seconds")
    #print(desc)

    return keypoints, desc

def get_matches_ssd(des1, des2):
    """Finds matches between two sets of descriptors using Sum of Squared Differences (SSD).
    
    Args:
        des1 (np.ndarray): The first set of descriptors.
        des2 (np.ndarray): The second set of descriptors.
    
    Returns:
        list of cv2.DMatch: The list of matched features.
    """
    matches = []
    for i in range(len(des1)):
        feature1 = np.float32(des1[i])
        distances = []
        for j in range(len(des2)):
            feature2 = np.float32(des2[j])
            ssd = sum((feature2 - feature1) ** 2)
            distances.append(ssd)
        bestMatchIdx = np.argmin(distances)
        dMatch = cv2.DMatch(i, bestMatchIdx, distances[bestMatchIdx])
        matches.append(dMatch)
    return matches

def get_matches_ratio(des1, des2):
    """Finds matches between two sets of descriptors using the ratio test

    Args:
        des1 (np.ndarray): The first set of descriptors
        des2 (np.ndarray): The second set of descriptors

    Returns:
        list of cv2.DMatch: The list of matched features passing the ratio test
    """
    matches = []
    for i in range(len(des1)):
        feature1 = np.float32(des1[i])
        distances = []
        for j in range(len(des2)):
            feature2 = np.float32(des2[j])
            ssd = sum((feature2 - feature1) ** 2)
            distances.append(ssd)
        bestMatchIdxs = np.argpartition(distances, 2)
        best_distance = distances[bestMatchIdxs[0]]
        second_best_distance = distances[bestMatchIdxs[1]]
        if best_distance / second_best_distance < 0.7:
            dMatch = cv2.DMatch(i, bestMatchIdxs[0], distances[bestMatchIdxs[0]])
            matches.append(dMatch)
    return matches

def ssd_matching(descriptors1, descriptors2):
    """Finds and sorts SSD matches between two sets of descriptors.
    
    Args:
        descriptors1 (np.ndarray): The first set of descriptors.
        descriptors2 (np.ndarray): The second set of descriptors.
    
    Returns:
        list of cv2.DMatch: The sorted list of matched features.
    """
    global NUM_MATCHES
    if descriptors1 is None or descriptors2 is None:
        print("[INFO] No descriptors found in one of the images.")
        update_status("")
        return None
    matches = get_matches_ssd(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)[:NUM_MATCHES]
    return matches

def ratio_matching(descriptors1, descriptors2):
    """Finds and sorts ratio matches between two sets of descriptors.
    
    Args:
        descriptors1 (np.ndarray): The first set of descriptors.
        descriptors2 (np.ndarray): The second set of descriptors.
    
    Returns:
        list of cv2.DMatch: The sorted list of matched features passing the ratio test.
    """
    global NUM_MATCHES
    if descriptors1 is None or descriptors2 is None:
        print("[INFO] No descriptors found in one of the images.")
        update_status("")
        return
    matches = get_matches_ratio(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)[:NUM_MATCHES]
    return matches

def image_stitch(images, kpsA, kpsB, matches, reprojThresh=4.0):
    """Stitches two images together using the matched features.
    
    Args:
        images (list of np.ndarray): The list of images to be stitched (must contain exactly two images).
        kpsA (list of cv2.KeyPoint): Keypoints from the first image.
        kpsB (list of cv2.KeyPoint): Keypoints from the second image.
        matches (list of cv2.DMatch): The list of matched features.
        reprojThresh (float): The RANSAC reprojection threshold.
    
    Returns:
        np.ndarray: The stitched image.
    """
    if len(images) != 2:
        print("[INFO] exactly two images are required for stitching")
        update_status("")
        return None

    if len(matches) > 4:
        ptsA = np.float32([kpsA[m.queryIdx].pt for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx].pt for m in matches])

        print("[INFO] computing homography...")
        (H, status) = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, reprojThresh)

        if H is None:
            print("[INFO] homography computation failed")
            update_status("")
            return None

        print("[INFO] warping images...")
        width = images[0].shape[1] + images[1].shape[1]
        height = max(images[0].shape[0], images[1].shape[0])
        result = cv2.warpPerspective(images[1], H, (width, height))

        result[0:images[0].shape[0], 0:images[0].shape[1]] = images[0]

        display_width = 800
        aspect_ratio = display_width / result.shape[1]
        display_height = int(result.shape[0] * aspect_ratio)
        resized_result = cv2.resize(result, (display_width, display_height))

        cv2.imshow("Stitched", resized_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        update_status("")
    else:
        print("[INFO] not enough matches found")
        update_status("")

def on_button_click(button_name):
    global images_to_stitch, concatenated_img

    if button_name == "Image Pair One":
        image_paths = ["left.png", "right.png"]
    elif button_name == "Image Pair Three":
        image_paths = ["left7.png", "right7.png"]
    elif button_name == "Image Pair Four":
        image_paths = ["left9.png", "right9.png"]
    elif button_name == "Image Pair Five":
        image_paths = ["left10.png", "right10.png"]
    elif button_name == "Image Pair Two":
        image_paths = ["left8.png", "right8.png"]

    images_to_stitch = [cv2.imread(image_path) for image_path in image_paths]
    resized_images = [resize_image(img, new_width=400) for img in images_to_stitch]
    concatenated_img = cv2.hconcat(resized_images)
    cv2.imshow(f"Images Side by Side for {button_name}", concatenated_img)


def execute_detection():
    update_status("Executing Selected Methods...")
    feature_method = feature_var.get()
    global images_to_stitch
    
    if feature_method == "Harris Feature Detection":
        update_status("Executing Harris Corner Detection...")
        img1, keypoints1 = harris_corner_detection(images_to_stitch[0])
        img2, keypoints2 = harris_corner_detection(images_to_stitch[1])
        
        resized_images = [resize_image(img1, new_width=400), resize_image(img2, new_width=400)]
        concatenated_img = cv2.hconcat(resized_images)
        
        cv2.imshow("Harris Corner Detection", concatenated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        update_status("")
    
    elif feature_method == "SIFT Feature Detection":
        update_status("Executing SIFT Feature Detection...")
        keypoints1 = sift_feature_detection(images_to_stitch[0])
        keypoints2 = sift_feature_detection(images_to_stitch[1])
        
        img_with_keypoints1 = cv2.drawKeypoints(images_to_stitch[0], keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img_with_keypoints2 = cv2.drawKeypoints(images_to_stitch[1], keypoints2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        resized_images = [resize_image(img_with_keypoints1, new_width=400), resize_image(img_with_keypoints2, new_width=400)]
        concatenated_img = cv2.hconcat(resized_images)
        
        cv2.imshow("SIFT Feature Detection", concatenated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        update_status("")
        
    update_status("")

def execute_matching():
    update_status("Executing Selected Methods...")
    global images_to_stitch

    feature_method = feature_var.get()
    descriptor_method = descriptor_var.get()
    matching_method = matching_var.get()
    
    keypoints1 = []
    keypoints2 = []
    if feature_method == "Harris Feature Detection":
        update_status("Executing Harris Corner Detection...")
        harris_img, keypoints1 = harris_corner_detection(images_to_stitch[0])
        harris_img, keypoints2 = harris_corner_detection(images_to_stitch[1])
        update_status("")
    elif feature_method == "SIFT Feature Detection":
        update_status("Executing SIFT Feature Detection...")
        keypoints1 = sift_feature_detection(images_to_stitch[0])
        keypoints2 = sift_feature_detection(images_to_stitch[1])
        update_status("")

    descriptors1 = []
    descriptors2 = []
    if descriptor_method == "SIFT":
        update_status("Executing SIFT Feature Description...")
        keypoints1, descriptors1 = sift_descriptor(images_to_stitch[0], keypoints1)
        keypoints2, descriptors2 = sift_descriptor(images_to_stitch[1], keypoints2)
        update_status("")
    elif descriptor_method == "BRIEF Binary descriptor":
        update_status("Executing BRIEF Binary Feature Description...")
        keypoints1, descriptors1 = binary_descriptor(images_to_stitch[0], keypoints1)
        keypoints2, descriptors2 = binary_descriptor(images_to_stitch[1], keypoints2)
        update_status("")

    print(len(descriptors1), len(descriptors2))
    matches = []
    if matching_method == "SSD matching":
        update_status("Executing SSD Matching...")
        matches = ssd_matching(descriptors1, descriptors2)
        img_matches = cv2.drawMatches(images_to_stitch[0], keypoints1, images_to_stitch[1], keypoints2, 
                                      matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        resized_image = resize_image(img_matches, new_width=800)
        cv2.imshow("SSD Matching", resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        update_status("")
    elif matching_method == "Ratio matching":
        update_status("Executing Ratio Matching...")
        matches = ratio_matching(descriptors1, descriptors2)
        img_matches = cv2.drawMatches(images_to_stitch[0], keypoints1, images_to_stitch[1], keypoints2, 
                                      matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        resized_image = resize_image(img_matches, new_width=800)
        cv2.imshow("Ratio Matching", resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        update_status("")
        
    update_status("")

def execute_stitching():
    update_status("Executing Selected Methods...")
    global images_to_stitch

    feature_method = feature_var.get()
    descriptor_method = descriptor_var.get()
    matching_method = matching_var.get()
    
    keypoints1 = []
    keypoints2 = []
    if feature_method == "Harris Feature Detection":
        update_status("Executing Harris Corner Detection...")
        harris_img, keypoints1 = harris_corner_detection(images_to_stitch[0])
        harris_img, keypoints2 = harris_corner_detection(images_to_stitch[1])
        update_status("")
    elif feature_method == "SIFT Feature Detection":
        update_status("Executing SIFT Feature Detection...")
        keypoints1 = sift_feature_detection(images_to_stitch[0])
        keypoints2 = sift_feature_detection(images_to_stitch[1])
        update_status("")

    descriptors1 = []
    descriptors2 = []
    if descriptor_method == "SIFT":
        update_status("Executing SIFT Feature Description...")
        keypoints1, descriptors1 = sift_descriptor(images_to_stitch[0], keypoints1)
        keypoints2, descriptors2 = sift_descriptor(images_to_stitch[1], keypoints2)
        update_status("")
    elif descriptor_method == "BRIEF Binary descriptor":
        update_status("Executing BRIEF Binary Feature Description...")
        keypoints1, descriptors1 = binary_descriptor(images_to_stitch[0], keypoints1)
        keypoints2, descriptors2 = binary_descriptor(images_to_stitch[1], keypoints2)
        update_status("")

    matches = []
    if matching_method == "SSD matching":
        update_status("Executing SSD Matching...")
        matches = ssd_matching(descriptors1, descriptors2)
        update_status("")
    elif matching_method == "Ratio matching":
        update_status("Executing Ratio Matching...")
        matches = ratio_matching(descriptors1, descriptors2)
        update_status("")

    update_status("Executing Image Stitching...")
    image_stitch(images_to_stitch, keypoints1, keypoints2, matches)

    update_status("")

def update_status(message):
    status_var.set(message)
    status_label.update_idletasks()

image_paths = ["left.png", "right.png"]
images_to_stitch = [cv2.imread(image_path) for image_path in image_paths]
concatenated_img = None

win = tk.Tk()
win.geometry('1200x800')

label = tk.Label(win, text="Choose an Image Pair button to compare a pair of images and then select an action based on the options provided to perform on those images:", bg= "white", fg="purple")
label.pack(pady=10)

button_names = ["Image Pair One", "Image Pair Two", "Image Pair Three", "Image Pair Four", "Image Pair Five"]

button_frame = tk.Frame(win)
button_frame.pack()

for name in button_names:
    button = tk.Button(button_frame, text=name, command=lambda btn_name=name: on_button_click(btn_name))
    button.pack(side=tk.LEFT, padx=5)

dropdown_frame = tk.Frame(win)
dropdown_frame.pack(pady=10)

# Feature Detection
feature_label = tk.Label(dropdown_frame, text="Feature Detection:")
feature_label.grid(row=0, column=0, padx=5)
feature_options = ["Harris Feature Detection", "SIFT Feature Detection"]
feature_var = tk.StringVar()
feature_var.set(feature_options[0])
feature_menu = ttk.Combobox(dropdown_frame, textvariable=feature_var, values=feature_options, state='readonly')
feature_menu.grid(row=0, column=1, padx=5)

# Descriptor
descriptor_label = tk.Label(dropdown_frame, text="Descriptor:")
descriptor_label.grid(row=0, column=2, padx=5)
descriptor_options = ["SIFT", "BRIEF Binary descriptor"]
descriptor_var = tk.StringVar()
descriptor_var.set(descriptor_options[0])
descriptor_menu = ttk.Combobox(dropdown_frame, textvariable=descriptor_var, values=descriptor_options, state='readonly')
descriptor_menu.grid(row=0, column=3, padx=5)

# Matching Method
matching_label = tk.Label(dropdown_frame, text="Matching Method:")
matching_label.grid(row=0, column=4, padx=5)
matching_options = ["SSD matching", "Ratio matching"]
matching_var = tk.StringVar()
matching_var.set(matching_options[0])
matching_menu = ttk.Combobox(dropdown_frame, textvariable=matching_var, values=matching_options, state='readonly')
matching_menu.grid(row=0, column=5, padx=5)

# Execute Buttons

execute_button_frame = tk.Frame(win)
execute_button_frame.pack(pady=10)

execute_detection_button = tk.Button(execute_button_frame, text="Execute Detection", command=execute_detection)
execute_detection_button.pack(side=tk.LEFT, padx=20)

execute_matching_button = tk.Button(execute_button_frame, text="Execute Matching", command=execute_matching)
execute_matching_button.pack(side=tk.LEFT, padx=20)

execute_stitching_button = tk.Button(execute_button_frame, text="Execute Stitching", command=execute_stitching)
execute_stitching_button.pack(side=tk.LEFT, padx=20)

# Status Label
status_var = tk.StringVar()
status_label = tk.Label(win, textvariable=status_var, fg="blue")
status_label.pack()

win.mainloop()
