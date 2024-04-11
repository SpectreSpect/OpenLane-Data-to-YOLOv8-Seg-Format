import cv2
import numpy as np
from skimage.morphology import skeletonize
from timeit import default_timer as timer
import math


def draw_line_from_points(image, points, color=(0, 255, 0), thickness=2):
    # Convert the points to numpy array format
    points = np.array(points)

    # Convert points to integer coordinates
    points = points.astype(int)

    # Draw the line connecting the points
    for i in range(len(points) - 1):
        t = i / len(points)
        color = (0, int(255.0 * (1.0 - t)), int(255.0 * t))
        cv2.line(image, tuple(points[i]), tuple(points[i + 1]), color, thickness)

    return image



# def test_func(image):
#     # Read an image

#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Threshold the image to get a binary image
#     ret, thresh = cv2.threshold(gray, 127, 255, 0)

#     # Find contours in the binary image
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Assuming there's only one contour in this example
#     contour = contours[0]

#     # Create a mask from the contour
#     mask = np.zeros_like(gray)
#     cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

#     # Use skeletonization to find the centerline of the contour
#     skeleton = skeletonize(mask / 255).astype(np.uint8) * 255

#     # Find contours in the skeleton image
#     skeleton_contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#     # Extract the coordinates of the contour points
#     centerline_points = []
#     for contour in skeleton_contours:
#         for point in contour:
#             centerline_points.append(tuple(point[0]))



#     # Filter out points that are too close to each other
#     min_distance = 30  # Adjust as needed
#     filtered_points = [centerline_points[0]]
#     for point in centerline_points:
#         if all(cv2.norm(np.array(point) - np.array(p)) > min_distance for p in filtered_points):
#             filtered_points.append(point)

#     # for idx, point in enumerate(filtered_points):
#     #     filtered_points[0]


#     # # Make sure the first point corresponds to the starting point of the curve
#     # # Find the point closest to the starting point of the contour
#     # start_point = contour[contour[:, :, 0].argmin()][0]
#     # start_index = min(range(len(filtered_points)), key=lambda i: cv2.norm(np.array(filtered_points[i]) - start_point))
#     # centerline_points = filtered_points[start_index:] + filtered_points[:start_index]


#     draw_line_from_points(image, filtered_points, (0, 0, 255))
#     # Display the centerline points
#     for point in filtered_points:
#         cv2.circle(image, point, 2, (255, 0, 0), -1)  # Draw a green circle at each point
#     return image



# image = cv2.imread('image2.jpg')
# mean_fps = 0
# count = 100
# for i in range(count):
#     start = timer()
#     test_func(image)
#     end = timer()
#     mean_fps += 1 / (end - start)

# mean_fps /= count

# print(mean_fps)


def is_endpoint(image, point):
    x, y = point
    # Check if the current pixel is an endpoint by examining its neighborhood
    # For simplicity, let's check if the pixel has fewer than 2 nonzero neighbors

    sum = 0
    for xi in range(-5, 6):
        for yi in range(-5, 6):
            sum += 1 if image[y+yi, x+xi][0] > 0 else 0

    num_nonzero_neighbors = np.count_nonzero(image[y-1:y+2, x-1:x+2])
    return sum < 100


# Read an image
image = cv2.imread('image2.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold the image to get a binary image
ret, thresh = cv2.threshold(gray, 127, 255, 0)

# Find contours in the binary image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assuming there's only one contour in this example
contour = contours[0]

# Create a mask from the contour
mask = np.zeros_like(gray)
cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

# Use skeletonization to find the centerline of the contour
skeleton = skeletonize(mask / 255).astype(np.uint8) * 255

# Find contours in the skeleton image
skeleton_contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Extract the coordinates of the contour points
centerline_points = []
for contour in skeleton_contours:
    for point in contour:
        centerline_points.append(tuple(point[0]))


def find_neighbors(image, point, radius=10):
    
    math.sin()



# Filter out points that are too close to each other
# min_distance = 10  # Adjust as needed
# filtered_points = [centerline_points[0]]
# for point in centerline_points:
#     if all(cv2.norm(np.array(point) - np.array(p)) > min_distance for p in filtered_points):
#         filtered_points.append(point)

# for idx, point in enumerate(filtered_points):
#     filtered_points[0]


# # Make sure the first point corresponds to the starting point of the curve
# # Find the point closest to the starting point of the contour
# start_point = contour[contour[:, :, 0].argmin()][0]
# start_index = min(range(len(filtered_points)), key=lambda i: cv2.norm(np.array(filtered_points[i]) - start_point))
# centerline_points = filtered_points[start_index:] + filtered_points[:start_index]


# draw_line_from_points(image, centerline_points, (0, 0, 255))
# # Display the centerline points

nonzero_indices = np.nonzero(skeleton)

x = nonzero_indices[1][0]
y = nonzero_indices[0][0]

cv2.circle(skeleton, (x, y), 10, (255, 0, 0), -1)  # Draw a green circle at each point

# for point in centerline_points:
#     if is_endpoint(image, point):
#         cv2.circle(skeleton, point, 10, (255, 0, 0), -1)  # Draw a green circle at each point


# Display the result
cv2.imshow('Centerline Points', skeleton)
cv2.waitKey(0)
cv2.destroyAllWindows()