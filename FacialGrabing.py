mport face_recognition
import numpy as np
import cv2
import glob

# ######################################################################
# face capture
print('program start to create data matrix')
print('process 0%')
# Create an empty matrix to be our data set
face_Arr_rs_Matrix_list = []

percentage_counter = 0
for filename in glob.glob('faces/*.pgm'):    # loop through all images in a file
    percentage_counter += 1
    print('converting {}th image to learning pool'.format(percentage_counter))
    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(filename)
    # image = cv2.cvtColor(image_pre, cv2.COLOR_BGR2GRAY)

    # Find all the faces in the image using the default HOG-based model.
    # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
    # See also: find_faces_in_picture_cnn.py
    face_locations = face_recognition.face_locations(image)
    # print("I found {} face(s) in this photograph.".format(len(face_locations)))

    for face_location in face_locations:

        # Print the location of each face in this image
        top, right, bottom, left = face_location
        # print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top,
        #                                                                                             left,
        #                                                                                             bottom,
        #                                                                                             right))

        # You can access the actual face itself like this:
        face_Arr = image[top:bottom, left:right]
        face_Arr_rs = cv2.resize(face_Arr, (90, 90))

        # print(face_Arr.shape, face_Arr_rs.shape)

        # uncomment two lines below to  display recognized faces
        # pil_image = Image.fromarray(face_Arr_rs)
        # pil_image.show()

        face_Arr_record = face_Arr_rs.reshape(90*90*3)
        # face_Arr_record = face_Arr_rs.reshape(90 * 90)
        # print(face_Arr_record.shape)
        face_Arr_rs_Matrix_list.append(face_Arr_record)

face_Arr_rs_Matrix = np.array(face_Arr_rs_Matrix_list)
print('process 100%')
print('finished creating data matrix!')
print('data matrix shape', face_Arr_rs_Matrix.shape, 'with total {} records'.format(face_Arr_rs_Matrix.shape[0]))
print('')
print(face_Arr_rs_Matrix)

# ###########################################################################
# face classification (Unfinished)
averageFace = (1 / face_Arr_rs_Matrix.shape[0]) * np.sum(face_Arr_rs_Matrix, axis=0)
# print(averageFace.shape)  # uncomment these code to check if avgFace make sense
# averageFace_image = cv2.imwrite('avg.jpg', averageFace.reshape(90, 90, 3))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

AVGExcludedFaceMatrix = face_Arr_rs_Matrix - averageFace
print('this matrix is ', AVGExcludedFaceMatrix.shape)

DReducedCov = np.dot(AVGExcludedFaceMatrix, AVGExcludedFaceMatrix.T)
# DReducedCov = AVGExcludedFaceMatrix.T.dot(AVGExcludedFaceMatrix)
print('reduced dimensional Cov', DReducedCov.shape)
# notice the difference between python matrix ATA and real ATA

# Eigen Decomposition (Need to Prove)
eig_vals_DRcov, eig_vecs_DRcov = np.linalg.eig(DReducedCov)

# for i in range(len(eig_vals_DRcov)):
#
#     eigvec_DRcov = eig_vecs_DRcov[:, i].reshape(1, 122).T
#
#     print('Eigenvector {}: \n{}'.format(i+1, eigvec_DRcov))
#     print('Eigenvalue {} from covariance matrix: {}'.format(i+1, eig_vals_DRcov[i]))
#     print(40 * '-')

# print(eig_vals_DRcov)
# print(eig_vecs_DRcov.shape)
#
np.save('eigen-vector.npy', eig_vecs_DRcov)
np.save('eigen-value.npy', eig_vals_DRcov)

np.save('AVGExcluded.npy', AVGExcludedFaceMatrix)
np.save('DimentionReduced.npy', DReducedCov)
np.save('avgFac.npy', averageFace)

