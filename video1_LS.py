import cv2
import numpy as np
import time
from KalmanFilter import KalmanFilter
import imutils
import xlsxwriter
def start_tracking():
    # Create a workbook and add a worksheet.
    workbook = xlsxwriter.Workbook('C:\\Users\\adrie\PycharmProjects\\pythonProject\\datafiles.xlsx')
    ws1 = workbook.add_worksheet()
    ws1.title = "Markers positions"


    #yellow colour range
    yellowLower = (16, 180, 110)
    yellowUpper = (28, 255, 255)

    # A function to display the coordinates of  the points clicked on the image
    def click_event(event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            # print the coordinates
            print('mouse_click = ({}, {})'.format(x,y))

            # displaying the coordinates on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, str(x) + ',' +
                        str(y), (x - 50, y - 15), font,
                        0.5, (255, 0, 0), 2)
            cv2.imshow("Frame", frame)

        # checking for right mouse clicks
        if event == cv2.EVENT_RBUTTONDOWN:
            # displaying the coordinates on the Shell
            print('mouse_click = ({}, {})'.format(x,y))

            # displaying the coordinates on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            b = frame[y, x, 0]
            g = frame[y, x, 1]
            r = frame[y, x, 2]
            cv2.putText(frame, str(b) + ',' +
                        str(g) + ',' + str(r),
                        (x, y), font, 1,
                        (255, 255, 0), 2)
            cv2.imshow("Frame", frame)

    # ---- Variables to define before the while loop ----

    debugMode = 1
    scaling_factor = 0.5
    num_frames_to_track = 5

    # Initialise variables related to tracking paths and frame index:
    tracking_paths = []
    est_paths = []
    update_paths = []
    cov_e_paths = []  # Estimate of the covariance
    cov_u_paths = []  # Update of the covariance
    frame_index = 0
    # Define the tracking parameters:
    tracking_params = dict(winSize=(13, 13), maxLevel=4, criteria=
    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03))
    # Create KalmanFilter objects KF and KF1
    # KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
    KF = KalmanFilter(0.1, 0, 0, 1, 0.001, 0.001)
    KF1 = KalmanFilter(0.1, 0, 0, 0.01, 1, 1)

    j = 0   # the row number in the workbook (= the frame number)

    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture('videos/exp_LS_1.mov')

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # (*'MP42') *'avc1'
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1800, 1012))

    fps = cap.get(cv2.CAP_PROP_FPS)
    print("frames per seconds = {}".format(fps))

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video file")
    # allow the camera or video file to warm up
    time.sleep(2.0)

    # Iterate indefinitely until the user presses the Esc key.
    while (cap.isOpened()):
        ret,frame = cap.read()
        if ret == True:
            k = 1  # the marker number
            i = 1  # the column number in the workbook
            # grab the current frame
            # resize the frame and convert to HSV scale
            frame = imutils.resize(frame, width=1800)
            frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            ws1.write(j, i, "frame {}".format(j))
            i += 2

            if len(tracking_paths) > 0:
                prev_img, current_img = prev_HSV, frame_HSV
                # reshape(-1, 1, 2) reshapes the tracking paths in x arrays of size (1,2)
                # take all the points tp to track in feature_points_0
                feature_points_0 = np.float32([tp[-1] for tp in tracking_paths]).reshape(-1, 1, 2)
                # Use the Lucas - Kanade method to track points on a video, feature_points_1 is output vector of 2D points
                # containing the calculated new positions of input features in the second image
                feature_points_1, _, _ = cv2.calcOpticalFlowPyrLK(prev_img, current_img, feature_points_0, None,
                                                                  **tracking_params)
                print('feature_points_1 = {}'.format(feature_points_1))
                feature_points_0_rev, _, _ = cv2.calcOpticalFlowPyrLK(current_img, prev_img, feature_points_1, None,
                                                                      **tracking_params)
                print('feature_points_0_rev = {}'.format(feature_points_0_rev))
                diff_feature_points = abs(feature_points_0 - feature_points_0_rev).reshape(-1, 2).max(-1)
                good_points = diff_feature_points < 2
                print('good point: {}'.format(good_points))
                new_tracking_paths = []
                new_est_paths = []
                new_update_paths = []
                new_cov_e_paths = []
                new_cov_u_paths = []

                # CREATE NEW TRACKING PATHS AND APPLY THE KALMAN FILTER
                for tp, ep, up, (x, y), cep, cup, good_points_flag in zip(tracking_paths, est_paths, update_paths,
                                                                          feature_points_1.reshape(-1, 2), cov_e_paths,
                                                                          cov_u_paths, good_points):
                    if not good_points_flag:  # Check for an occlusion

                        if (j == 18) and k == 2:  # marker on the ASIS in frame 18
                             print('Path of marker 2: {}'.format(up))

                             # ----update----
                             (xst_u, Pst_u) = KF1.update(np.block([[x], [y]]), np.reshape(ep[-1], (4, 1)),
                                                         np.reshape(cep[-1], (4, 4)))
                             for (x_u, y_u, xdot_u, ydot_u) in np.float32(xst_u).reshape(-1, 4):
                                 up.append((x_u, y_u, xdot_u, ydot_u))
                                 tp.append((x_u, y_u))
                                 print("occl = {},{}".format(x_u,y_u))
                             cup.append((Pst_u))
                             if len(up) > num_frames_to_track:
                                 del up[0]
                                 del cup[0]
                                 del tp[0]
                             new_update_paths.append(up)
                             new_cov_u_paths.append(cup)
                             new_tracking_paths.append(tp)

                             # ----predict----
                             (xst_e, Pst_e) = KF1.predict(xst_u, Pst_u)
                             for (x_e, y_e, xdot_e, ydot_e) in np.float32(xst_e).reshape(-1, 4):
                                 ep.append((x_e, y_e, xdot_e, ydot_e))
                             cep.append((Pst_e))
                             if len(ep) > num_frames_to_track:
                                 del ep[0]
                                 del cep[0]
                             new_est_paths.append(ep)
                             new_cov_e_paths.append(cep)

                             # Draw on the outup_image
                             cv2.circle(frame, (int(x_u), int(y_u)), 4, (150, 50, 50), -1)
                             cv2.putText(frame, "Occl: {}".format(k), (int(x_u + 2), int(y_u + 2)), 0, 0.5,
                                         (0, 191, 255), 2)

                             k += 1

                        # continue goes directly to the next iteration of the code and finish the current iteration
                        continue
                    tp.append((x, y))  # Append the tracking paths to tp if these are good points
                    if len(tp) > num_frames_to_track:
                        del tp[0]  # delete the first tracking paths
                    new_tracking_paths.append(tp)

                    # ---- Update ----

                    (xst_u, Pst_u) = KF.update(np.block([[x], [y]]), np.reshape(ep[-1], (4, 1)), np.reshape(cep[-1], (
                    4, 4)))  # xst_u is the state variable with the positions and velocities, Pst_u is the covariance
                    for (x_u, y_u, xdot_u, ydot_u) in np.float32(xst_u).reshape(-1, 4):
                        up.append((x_u, y_u, xdot_u, ydot_u))
                    cup.append((Pst_u))
                    if len(up) > num_frames_to_track:
                        del up[0]
                        del cup[0]
                    new_update_paths.append(up)
                    new_cov_u_paths.append(cup)

                    # ---- Predict ----

                    (xst_e, Pst_e) = KF.predict(xst_u, Pst_u)
                    for (x_e, y_e, xdot_e, ydot_e) in np.float32(xst_e).reshape(-1, 4):
                        ep.append((x_e, y_e, xdot_e, ydot_e))
                    cep.append((Pst_e))
                    if len(ep) > num_frames_to_track:
                        del ep[0]
                        del cep[0]
                    new_est_paths.append(ep)
                    new_cov_e_paths.append(cep)

                    # Draw on output image

                    # feature_points_1
                    cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), 1)  # blue circle in the RGB color space
                    # Updates points
                    cv2.circle(frame, (int(x_u), int(y_u)), 2, (0, 255, 0), -1)  # green center in the RGB color space
                    cv2.putText(frame, "{}".format(k), (int(x_u + 4), int(y_u + 4)), 0, 0.5, (0, 191, 255), 2)

                    # write the position of the marker in the workbook
                    ws1.write(j, i, "marker {}".format(k))
                    i+=1
                    ws1.write(j, i,x_u)
                    i += 1
                    ws1.write(j, i, y_u)
                    i += 2
                    k+=1  # the marker number

                # Positions update for the next frame
                tracking_paths = new_tracking_paths
                update_paths = new_update_paths
                est_paths = new_est_paths
                cov_u_paths = new_cov_u_paths
                cov_e_paths = new_cov_e_paths

                cv2.polylines(frame, [np.int32(tp) for tp in tracking_paths], False,
                              (0, 150, 0))  # Draw polygonal curves

            # ---- Colour detection ----

            mask = cv2.inRange(frame_HSV, yellowLower, yellowUpper)
            # cv2.imshow("Mask", mask)
            mask = cv2.erode(mask, None, iterations=1)
            mask = cv2.dilate(mask, None, iterations=1)

            # Black circles where markers are already detected
            for x, y in [np.int32(tp[-1]) for tp in tracking_paths]:
                cv2.circle(mask, (x, y), 15, 0,-1)
            cv2.rectangle(mask,(76,367),(116,434),0,-1)
            cv2.circle(mask, (194, 561), 10, 0, -1)

            # Find contours
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Set the accepted minimum & maximum radius of a detected object
            min_radius_thresh = 1
            max_radius_thresh = 20
            new_centers = []
            new_centers_u = []
            new_centers_e = []
            new_cov_u = []
            new_cov_e = []
            x_state = np.block([[0], [0], [0], [0]])
            P_state = np.eye(4)
            for c in contours:
                (x, y), radius = cv2.minEnclosingCircle(c)
                radius = int(radius)
                # Take only the valid circles
                if (max_radius_thresh > radius > min_radius_thresh):
                    # Image Moment is a particular weighted average of image pixel intensities.
                    M = cv2.moments(c)
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    cv2.circle(frame, (int(center[0]), int(center[1])), 3, (0, 0, 255), -1)  # red
                    new_centers.append(np.array([[center[0]], [center[1]]]))
                    # Update
                    x_u, P_u = KF.update(np.block([[center[0]], [center[1]]]), x_state, P_state)  # put an array in the update function
                    new_centers_u.append(np.array(x_u))
                    new_cov_u.append(np.array(P_u))

                    # Predict the next measurement
                    x_e, P_e = KF.predict(x_u, P_u)
                    new_centers_e.append(np.array(x_e))
                    new_cov_e.append(np.array(P_e))
            feature_points = new_centers

            # Add new markers in tracking paths list
            if feature_points is not None:
                for (x, y), (x_e, y_e, xdot_e, ydot_e), (x_u, y_u, xdot_u, ydot_u) in zip(
                        np.float32(feature_points).reshape(-1, 2), np.float32(new_centers_u).reshape(-1, 4),
                        np.float32(new_centers_e).reshape(-1, 4)):
                    tracking_paths.append([(x, y)])
                    est_paths.append([(x_e, y_e, xdot_e, ydot_e)])
                    update_paths.append([(x_u, y_u, xdot_u, ydot_u)])
                    ws1.write(j, i, "marker {}".format(k))
                    i += 1
                    ws1.write(j, i, x_u)
                    i += 1
                    ws1.write(j, i, y_u)
                    i += 2
                    k += 1
                for (P_e, P_u) in zip(new_cov_e, new_cov_u):
                    cov_e_paths.append([(P_e)])
                    cov_u_paths.append([(P_u)])
            cv2.putText(frame, "Frame : " + str(int(frame_index)), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (50, 170, 50), 2)
            cv2.imshow("Frame", frame)

            cv2.setMouseCallback("Frame", click_event)

            # ---- Update for the next iterationq ----
            frame_index += 1
            j += 1
            print('Frame number {}'.format(frame_index))
            output_img = frame.copy()
            out.write(output_img)
            prev_HSV = frame_HSV
            key = cv2.waitKey(0)
            
            while key not in [ord('q'), ord('k')]:
                key = cv2.waitKey(0)
            # Quit when 'q' is pressed
            if key == ord('q'):
                break



        else:
            break
            cap.release()
            out.release()
if __name__ == "__main__":
    # Start the tracker
    start_tracking()
    # Close all the windows
    cv2.destroyAllWindows()
