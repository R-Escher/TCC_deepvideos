# src: https://nanonets.com/blog/optical-flow/

import cv2 as cv

def sparse_optical_flow(frames, first_frame = None):
    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(maxCorners = 300, qualityLevel = 0.2, minDistance = 2, blockSize = 7)
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    if (not first_frame):
        first_frame = frames[0]

    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    # Finds the strongest corners in the first frame by Shi-Tomasi method - we will track the optical flow for these corners
    # https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack
    prev = cv.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)

    total_frames = len(frames)
    first_features = len(prev)
    last_features = 0
    counter = total_frames
    
    features = [first_features]
    for frame in frames:
        counter -= 1
        # Converts each frame to grayscale - we previously only converted the first frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Calculates sparse optical flow by Lucas-Kanade method
        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
        next, status, error = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
        try:
            # Selects good feature points for previous position
            good_old = prev[status == 1]                
            # Selects good feature points for next position
            good_new = next[status == 1]
            last_features = len(good_new)
            
            features.append(last_features)

        except:
            print('Features perdidas')
            break

    #return first_features + last_features + (total_frames + counter) 
    return features

if __name__ == '__main__':
    print(sparse_optical_flow([]))