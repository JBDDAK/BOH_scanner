import sys
import dlib
import cv2
import openface


# You can download the required pre-trained face detection model here:
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2


# Take the image file name from the command line
def save_img(imgname, num):
    predictor_model = "../shape_predictor_68_face_landmarks.dat"
    file_name = imgname

    # Create a HOG face detector using the built-in dlib class
    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    face_aligner = openface.AlignDlib(predictor_model)

    # Take the image file name from the command line

    # Load the image
    image = cv2.imread(file_name)

    # Run the HOG face detector on the image data
    detected_faces = face_detector(image, 3)

    print("Found {} faces in the image file {}".format(len(detected_faces), file_name))

    # Loop through each face we found in the image
    for i, face_rect in enumerate(detected_faces):
        # Detected faces are returned as an object with the coordinates
        # of the top, left, right and bottom edges
        print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(),
                                                                                 face_rect.right(), face_rect.bottom()))

        # Get the the face's pose
        pose_landmarks = face_pose_predictor(image, face_rect)

        # Use openface to calculate and perform the face alignment
        # 512 * 512
        alignedFace = face_aligner.align(256, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

        # Save the aligned image to a file
        cv2.imwrite("../data/QWER.jpg".format(num), alignedFace)
        # cv2.imwrite("images/QWER.jpg".format(num), alignedFace)
