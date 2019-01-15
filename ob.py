#!/usr/bin/python

# 2.12 Lab 4 object detection: a node for detecting objects
# Peter Yu Oct 2016
# Revised by Michael Su Oct 2018

import rospy
import numpy as np
import cv2  # OpenCV module

from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose, Twist, Vector3, Quaternion
from std_msgs.msg import ColorRGBA

from cv_bridge import CvBridge, CvBridgeError
import message_filters
import math

rospy.init_node('object_detection', anonymous=True)

# Publisher for publishing pyramid marker in rviz
vis_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)

# Publisher for publishing images in rviz
mask_eroded_pub = rospy.Publisher('/object_detection/mask_eroded', Image, queue_size=10)
mask_ero_dil_pub = rospy.Publisher('/object_detection/mask_eroded_dilated', Image, queue_size=10)
img_result_pub = rospy.Publisher('/object_detection/img_result', Image, queue_size=10)

# Bridge to convert ROS Image type to OpenCV Image type
cv_bridge = CvBridge()

# Get the camera calibration parameter for the rectified image
msg = rospy.wait_for_message('/camera/rgb/camera_info', CameraInfo, timeout=None)
#     [fx'  0  cx' Tx]
# P = [ 0  fy' cy' Ty]
#     [ 0   0   1   0]

fx = msg.P[0]
fy = msg.P[5]
cx = msg.P[2]
cy = msg.P[6]

##########################3
cap = cv2.VideoCapture(0)
############################

def main():
    useHSV   = True
    useDepth = True
    if not useHSV:
        # Subscribe to RGB image
        rospy.Subscriber('/camera/rgb/image_rect_color', Image, rosImageVizCallback)
    else:
        if not useDepth:
            #    Subscribe to RGB images
            rospy.Subscriber('/camera/rgb/image_rect_color', Image, rosHSVProcessCallBack)
        else:
            #    Subscribe to both RGB and Depth images with a Synchronizer
            image_sub = message_filters.Subscriber("/camera/rgb/image_rect_color", Image)
            depth_sub = message_filters.Subscriber("/camera/depth_registered/sw_registered/image_rect", Image)
            ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 10, 0.5)
            ts.registerCallback(rosRGBDCallBack)
##################################################################

    # Capture Video from Camera

####################################################################
    rospy.spin()


def rosImageVizCallback(msg):
    # 1. convert ROS image to opencv format
    try:
        cv_image = cv_bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        print(e)
    # 2. visualize it in a cv window
    cv2.imshow("OpenCV_View", cv_image)
    # set callback func for mouse hover event
    cv2.setMouseCallback("OpenCV_View", cvWindowMouseCallBackFunc)
    cv2.waitKey(1)  # milliseconds


def cvWindowMouseCallBackFunc(event, xp, yp, flags, param):
    print 'In cvWindowMouseCallBackFunc: (xp, yp)=', xp, yp  # xp, yp is the mouse location in the window
    # 1. Set the object to 2 meters away from camera
    zc = 2.0
    # 2. Visualize the pyramid
    showPyramid(xp, yp, zc, 10, 10)


def rosHSVProcessCallBack(msg):
    try:
        cv_image = cv_bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        print(e)
    contours, mask_image = HSVObjectDetection(cv_image)
    res = cv2.bitwise_and(cv_image, cv_image, mask = mask_image)
    img_result_pub.publish(cv_bridge.cv2_to_imgmsg(res, encoding="passthrough"))

def HSVObjectDetection(cv_image, toPrint = True):
    # convert image to HSV color space
    hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    # define range of red color in HSV
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])

    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv_image, lower_red, upper_red)

    mask_eroded = cv2.erode(mask, None, iterations = 3)
    mask_eroded_dilated = cv2.dilate(mask_eroded, None, iterations = 3)

    mask_eroded_pub.publish(cv_bridge.cv2_to_imgmsg(mask_eroded,
        encoding="passthrough"))
    mask_ero_dil_pub.publish(cv_bridge.cv2_to_imgmsg(mask_eroded_dilated,
        encoding="passthrough"))
    image, contours, hierarchy = cv2.findContours(
        mask_eroded_dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    return contours, mask_eroded_dilated

def rosRGBDCallBack(rgb_data, depth_data):
    draw_contours = False
    detect_shape = False
    calculate_size = False
    try:
        cv_image = cv_bridge.imgmsg_to_cv2(rgb_data, "bgr8")
	print("cv_image@@@@@@@@",type(cv_image))
        cv_depthimage = cv_bridge.imgmsg_to_cv2(depth_data, "32FC1")
        cv_depthimage2 = np.array(cv_depthimage, dtype=np.float32)
    except CvBridgeError as e:
        print(e)

    contours, mask_image = HSVObjectDetection(cv_image, toPrint = False)

    for cnt in contours:
        if not draw_contours:
            xp,yp,w,h = cv2.boundingRect(cnt)

            cv2.rectangle(cv_image,(xp,yp),(xp+w,yp+h),[0,255,255],2)
            cv2.circle(cv_image,(int(xp+w/2),int(yp+h/2)),5,(55,255,155),4)
            if not math.isnan(cv_depthimage2[int(yp+h/2)][int(xp+w/2)]) :
                zc = cv_depthimage2[int(yp+h/2)][int(xp+w/2)]
                X1 = getXYZ(xp+w/2, yp+h/2, zc, fx, fy, cx, cy)
                print "x:",X1[0],"y:",X1[1],"z:",X1[2]
            else:
                continue
        else:
            #################Draw contours#####################################
            # In task1, you need to call the function "cv2.drawContours" to show
            # object contour in RVIZ
            #
            #
            cv2.drawContours(cv_image, contours, -1, (0,255,0), 3)
	    # print type(cnt)
	    # print len(cnt)
            ###################################################################

            if detect_shape:
                ##############Detect object shape##############################
                # In task2, you need to detect object shape for each object
                # contour. After you find the shape, you have to assign the
                # variable "obj_shape" corresponding value according to the
                # object shape.
                #

		approx = cv2.approxPolyDP(cnt,15,True)
		print len(approx)
		print "@@@@@@@@@@@@@@@@@@@@@@@@"
		
		if len(approx) == 3:
			obj_shape = "triangle"
		elif len(approx) == 4:
                        obj_shape = "square"
		else:
                        #obj_shape = "test"
			obj_shape = "circle"
		
                #obj_shape = "triangle" #options: "triangle", "square", "circle"
                #
                #
                #
                # Hint: Approximate the contour as polygon (cv2.approxPolyDP)
                ###############################################################

                # Get the first vertex of the object
                if len(cnt) < 1:
                    continue
                x_str, y_str = cnt[0][0][:]
                font = cv2.FONT_HERSHEY_SIMPLEX
                if obj_shape == "triangle":
                    cv2.putText(cv_image, "Triangle", (x_str, y_str), font, 1,
                        (0, 255, 255), 2, cv2.LINE_AA)
                elif obj_shape == "square":
                    cv2.putText(cv_image, "Square", (x_str, y_str), font, 1,
                        (0, 255, 255), 2, cv2.LINE_AA)
                elif obj_shape == "circle":
                    cv2.putText(cv_image, "Circle", (x_str, y_str), font, 1,
                        (0, 255, 255), 2, cv2.LINE_AA)

            if calculate_size:
                ##################################################
                # In task3, we offer the methods to calculate area.
                # If you want to use the method TA offers, you will need to
                # assign vertex list of single object to "vtx_list" and finish
                # functions "get_side_length", "get_radius".
                # You can also write your own code to calculate
                # area. if so, please ignore and comment the line 193 - 218,
                # and write your code there.
                #
                #
                vtx_list = cnt #hint: vtx_list is similar to cnt
                ##################################################
                if obj_shape == "triangle":
                    tri_side_len = get_side_length(vtx_list, cv_depthimage2,obj_shape)
                    if tri_side_len is None:
                        continue
                    tri_area = tri_side_len**2 * math.sqrt(3) / 4
                    string = "%.3e" % tri_area
                    cv2.putText(cv_image, string, (x_str, y_str+30), font,
                        1, (0, 255, 255), 2, cv2.LINE_AA)

                elif obj_shape == "square":
                    squ_side_len = get_side_length(vtx_list, cv_depthimage2,obj_shape)
                    if squ_side_len is None:
                        continue
                    squ_area = squ_side_len**2
                    string = "%.3e" % squ_area
                    cv2.putText(cv_image, string, (x_str, y_str+30), font,
                        1, (0, 255, 255), 2, cv2.LINE_AA)

                elif obj_shape == "circle":
                    circle_radius = get_side_length(vtx_list, cv_depthimage2,obj_shape)
                    if circle_radius is None:
                        continue
                    circle_area = circle_radius**2 * math.pi
                    string = "%.3e" % circle_area
                    cv2.putText(cv_image, string, (x_str, y_str+30), font,
                        1, (0, 255, 255), 2, cv2.LINE_AA)
    print "666666666666666666666666666666666666666666666"
#########################################################################

 # Capture Video from Camera
    #_, frame = cap.read()
    frame = cv_image
    print("cv_image", type(cv_image))
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # define range of green color in HSV 
    lower_green = np.array([45,100,50])
    upper_green = np.array([75,255,255])

    # define range of red color in HSV 
    lower_red = np.array([0,70,50])
    upper_red = np.array([10,255,255])

    # Threshold the HSV image to get only blue colors
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    # Threshold the HSV image to get only green colors
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    # Threshold the HSV image to get only red colors   
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    # Bitwise-AND mask and original image

    res_blue = cv2.bitwise_and(frame,frame, mask= mask_blue)
    res_green = cv2.bitwise_and(frame,frame, mask= mask_green)
    res_red = cv2.bitwise_and(frame,frame, mask= mask_red)
    cv2.imshow('frame',frame)
    cv2.imshow('mask_blue)',mask_blue)
    cv2.imshow('mask_green',mask_green)
    cv2.imshow('mask_red',mask_red)

    cv2.imshow('res_blue',res_blue)
    cv2.imshow('res_green',res_green)
    cv2.imshow('res_red',res_red)
    k = cv2.waitKey(5) & 0xFF


#######################################################################3
    img_result_pub.publish(cv_bridge.cv2_to_imgmsg(cv_image,
        encoding="passthrough"))


def get_side_length(cnt_vtx_list, depthimage,obj_shape):
    """Get side length of square or equilateral triangle

    From real world coordinate of object vertexes, we can calculate the distance
    between all vertexes. Thus,side length of square and equilateral triangle can
    be calculated.

    Args:
        cnt_vtx_list: contour vertext list
        depthimage: captured depth image which is registered with RGB image

    Returns:
        side length
    """
    vtx_3d_list = []
    # Get the real world coordinate of the vertex
    for i in range(0, len(cnt_vtx_list)):
        x, y = cnt_vtx_list[i][0][:]
        if not math.isnan(depthimage[int(y)][int(x)]):
            zc = depthimage[int(y)][int(x)]
            v1 = np.array(getXYZ(x, y, zc, fx, fy, cx, cy))
            vtx_3d_list.append(v1)
        else:
            vtx_3d_list.append(None)

    ##print type(vtx_3d_list)
    ##print len(vtx_3d_list)
    ##print vtx_3d_list[20]
    side_len_list = [] #A list which stores side lengths of a polygon
    ########################Finish side_len_list###############################
    # Hint: You can access all 3D coordinates of object vertexes from variable
    # "vtx_3d_list". However, some coordinates is None because the
    # depth is invalid. You need to  use "vtx_3d_list" and design algorithm to
    # get side length of this object.
    #
    # Step1. Check value in vtx_3d_list is not "None"
    # Step2. Calculate the distance of all the 2 neighboring vertexes
    #
    # Example for calculating distance:
    # vector = vtx_3d_list[i] - vtx_3d_list[i+1]
    # side_len = math/sqrt(i**2 for i in vector)
    #
    side_len = 0
    vtx_3d_list_nonzr = []
    for i in range(0,len(vtx_3d_list)):
	if not vtx_3d_list[i] == None:
		vtx_3d_list_nonzr.append(vtx_3d_list[i])
	else:
		continue
    vtx_length = len(vtx_3d_list_nonzr)


    for i in range(0,len(vtx_3d_list_nonzr)-1):
	vector = vtx_3d_list_nonzr[i] - vtx_3d_list_nonzr[i+1]
	vector = vector.tolist()
	##print vector
	##print type(vector)
	##print len(vector)
	side_len += math.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)
    
    ##vector = vtx_3d_list_nonzr[vtx_length-1] - vtx_3d_list_nonzr[0]
    ##side_len += math.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)

    print side_len
    print "111111111111111111111111111111111"

    if obj_shape == "triangle":
	return side_len/3
    elif obj_shape == "square":
	return side_len/4
    else:
	return side_len/2/3.14

    #
    #
    #
    #
    #
    #side_len_list = []
    ###########################################################################
   # if len(side_len_list) < 1:
   #     return None
   # else:
   #     print ("Side length: \n" +  str(side_len_list))
    ##    return sum(side_len_list)/len(side_len_list)


def get_radius(cnt_vtx_list, depthimage):
    """Get max distance of vertexes in a circle

    We approximate cricle as a polygin, and we can get many vertexes. In this
    function, we will calculate the maximum distance between these vertexes and
    regard it as the diameter of circle.

    Args:
        cnt_vtx_list: contour vertext list
        depthimage: captured depth image which is registered with RGB image

    Returns:
        circle radius
    """
    vtx_3d_list = []
    # Get the real world coordinate of the vertex
    for i in range(0, len(cnt_vtx_list)):
        x, y = cnt_vtx_list[i][0][:]
        if not math.isnan(depthimage[int(y)][int(x)]):
            zc = depthimage[int(y)][int(x)]
            v1 = np.array(getXYZ(x, y, zc, fx, fy, cx, cy))
            vtx_3d_list.append(v1)
    if len(vtx_3d_list) < 2:
        return None
    vtx_dis = [] #A list which stores distance between vertexes
    ########################Finish vtx_dis#####################################
    # Hint: You can access all 3D coordinates of object vertexes from variable
    # "vtx_3d_list". However, some of "vtx_3d_list" are None because the
    # depth is invalid. You need to use "vtx_3d_list" and design algorithm to
    # get all possible distance between two vertexes of this object.
    #
    # Step1. Check value in vtx_3d_list is not "None"
    # Step2. Make pairs of 2 vertexes in a polygon and calculate the distance.
    #
    # Example for calculating distance:
    # vector = vtx_3d_list[i] - vtx_3d_list[i+1]
    # side_len = math/sqrt(i**2 for i in vector)
    #
    #
    #
    #
    #
    #
    #
    # vtx_dis = ??
    ###########################################################################
    return max(vtx_dis)/2

def getXYZ(xp, yp, zc, fx,fy,cx,cy):
    #### Definition:
    # cx, cy : image center(pixel)
    # fx, fy : focal length
    # xp, yp: index of the depth image
    # zc: depth
    inv_fx = 1.0/fx
    inv_fy = 1.0/fy
    x = (xp-cx) *  zc * inv_fx
    y = (yp-cy) *  zc * inv_fy
    z = zc
    return (x,y,z)



# Create a pyramid using 4 triangles
def showPyramid(xp, yp, zc, w, h):
    # X1-X4 are the 4 corner points of the base of the pyramid
    X1 = getXYZ(xp-w/2, yp-h/2, zc, fx, fy, cx, cy)
    X2 = getXYZ(xp-w/2, yp+h/2, zc, fx, fy, cx, cy)
    X3 = getXYZ(xp+w/2, yp+h/2, zc, fx, fy, cx, cy)
    X4 = getXYZ(xp+w/2, yp-h/2, zc, fx, fy, cx, cy)
    vis_pub.publish(createTriangleListMarker(1, [X1, X2, X3, X4], rgba = [1,0,0,1], frame_id = "/camera_rgb_optical_frame"))

# Create a list of Triangle markers for visualizationi
def createTriangleListMarker(marker_id, points, rgba, frame_id = "/camera_rgb_optical_frame"):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.type = marker.TRIANGLE_LIST
    marker.scale = Vector3(1,1,1)
    marker.id = marker_id

    n = len(points)

    if rgba is not None:
        marker.color = ColorRGBA(*rgba)

    o = Point(0,0,0)
    for i in xrange(n):
        p = Point(*points[i])
        marker.points.append(p)
        p = Point(*points[(i+1)%4])
        marker.points.append(p)
        marker.points.append(o)

    marker.pose = poselist2pose([0,0,0,0,0,0,1])
    return marker


def poselist2pose(poselist):
    return Pose(Point(*poselist[0:3]), Quaternion(*poselist[3:7]))


if __name__=='__main__':
    main()

