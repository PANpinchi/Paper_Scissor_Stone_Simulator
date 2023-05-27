import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
from time import sleep


def initialize_map(map_size):
    game_map = [[(200, 200, 200)] * map_size for _ in range(map_size)]
    return np.array(game_map)


def initialize_object(map_size, object_number, boundary):
    paper = []
    scissor = []
    stone = []

    for _ in range(object_number):
        random_x, random_y = random.randint(boundary, map_size - boundary), random.randint(boundary, map_size - boundary)
        paper.append((random_x, random_y))
        random_x, random_y = random.randint(boundary, map_size - boundary), random.randint(boundary, map_size - boundary)
        scissor.append((random_x, random_y))
        random_x, random_y = random.randint(boundary, map_size - boundary), random.randint(boundary, map_size - boundary)
        stone.append((random_x, random_y))

    return np.array(paper), np.array(scissor), np.array(stone)


def draw_point(img, point, val=(0, 0, 255), w=10):
    row, col = point[0], point[1]
    img[row - w // 2:row + w // 2 + 1, col - w // 2:col + w // 2 + 1, :] = val
    return img


def draw_img(img, point, object_img):
    row, col = point[0], point[1]
    w = object_img.shape[0]
    object_mask = (~(object_img != 0)).astype(np.int32)

    img[row - w // 2:row + w // 2, col - w // 2:col + w // 2, :] *= object_mask
    img[row - w // 2:row + w // 2, col - w // 2:col + w // 2, :] += object_img
    return img


def draw_object(map, paper, scissor, stone, paper_image, scissor_image, stone_image, val=255):
    draw_map = map.copy()
    for i in range(len(paper)):
        # draw_map = draw_point(draw_map, paper[i], (val, 0, 0))
        draw_map = draw_img(draw_map, paper[i], paper_image)
    for i in range(len(scissor)):
        # draw_map = draw_point(draw_map, scissor[i], (val, val, 0))
        draw_map = draw_img(draw_map, scissor[i], scissor_image)
    for i in range(len(stone)):
        # draw_map = draw_point(draw_map, stone[i], (0, 0, val))
        draw_map = draw_img(draw_map, stone[i], stone_image)

    return draw_map


def find_the_target(point, target):
    # Calculate Euclidean distance to find the nearest target point
    # ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) ^ 0.5
    r, c = point[0], point[1]
    min_distance = np.inf
    min_target_point = point

    for i, target_point in enumerate(target):
        target_r, target_c = target_point[0], target_point[1]
        distance = np.sqrt(np.square(r - target_r) + np.square(c - target_c))
        if distance < min_distance and r != target_r and c != target_r:
            min_distance = distance
            min_target_point = target_point

    return min_target_point


def move_to_target(point, object, target_point):
    r, c = point[0], point[1]
    target_r, target_c = target_point[0], target_point[1]
    min_distance = np.inf

    for i in range(-1, 2):
        for j in range(-1, 2):
            # Calculate the Euclidean distance to find the closest moving position to the target point
            # ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) ^ 0.5
            target_distance = np.sqrt(np.square((r + i) - target_r) + np.square((c + j) - target_c))

            # find the nearest object
            nearest_object_point = find_the_target(((r + i), (c + j)), object)

            # Calculate the Euclidean distance to find the farthest moving position to the object point
            object_distance = np.sqrt(np.square((r + i) - nearest_object_point[0]) + np.square((c + j) - nearest_object_point[1]))

            distance = target_distance + object_distance * -0.04

            # avoid close proximity of identical objects
            if distance < min_distance:
                min_distance = distance
                next_point = [r + i, c + j]

    return next_point


def check_bound(next_point, map_size, image_size):
    image_size = image_size // 2
    next_point[0] = image_size if next_point[0] <= image_size else next_point[0]
    next_point[0] = map_size - image_size if next_point[0] >= map_size - image_size else next_point[0]
    next_point[1] = image_size if next_point[1] <= image_size else next_point[1]
    next_point[1] = map_size - image_size if next_point[1] >= map_size - image_size else next_point[1]
    return next_point


def move_object(object, target, map_size, image_size):
    for i, point in enumerate(object):
        # find the nearest target
        target_point = find_the_target(point, target)
        # move to nearest target
        next_point = move_to_target(point, object, target_point)
        # check if the next point is out of bounds
        next_point = check_bound(next_point, map_size, image_size)
        # update point
        object[i] = next_point
    return object


def touch_detection(object, target, threshold):
    # Calculate the Euclidean distance to implement touch detection
    # ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) ^ 0.5
    for i, object_point in enumerate(object):
        delete_index = []
        object_r, object_c = object_point[0], object_point[1]
        for j, target_point in enumerate(target):
            target_r, target_c = target_point[0], target_point[1]
            distance = np.sqrt(np.square(object_r - target_r) + np.square(object_c - target_c))

            if distance <= threshold:
                object = np.append(object, np.array([target_point]), axis=0)
                delete_index.append(j)

        if len(delete_index) != 0:
            target = np.delete(target, delete_index, axis=0)

    return object, target


def save_as_video(frame_list, size):
    result_name = 'output.mp4'

    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(result_name, fourcc, fps, size)

    for idx, frame in enumerate(frame_list):
        current_frame = idx + 1
        total_frame_count = len(frame_list)
        percentage = int(current_frame * 30 / (total_frame_count + 1))
        print("\rSave Video Process: [{}{}] {:06d} / {:06d}".format("#" * percentage, "." * (30 - 1 - percentage),
                                                                    current_frame, total_frame_count), end='')
        out.write(frame)

    out.release()


def simulate_game(map_size=500, object_number=20):
    paper_image = cv2.imread('./img/Paper.png')
    scissor_image = cv2.imread('./img/Scissor.png')
    stone_image = cv2.imread('./img/Stone.png')

    paper_image = cv2.cvtColor(paper_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    scissor_image = cv2.cvtColor(scissor_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    stone_image = cv2.cvtColor(stone_image.astype(np.uint8), cv2.COLOR_BGR2RGB)

    image_size = 30
    threshold = image_size ^ 2
    paper_image = cv2.resize(paper_image, (image_size, image_size))
    scissor_image = cv2.resize(scissor_image, (image_size, image_size))
    stone_image = cv2.resize(stone_image, (image_size, image_size))

    map = initialize_map(map_size=map_size)
    paper, scissor, stone = initialize_object(map_size=map_size, object_number=object_number, boundary=image_size)

    frame_list = []
    breakon = False
    while True:
        draw_map = map.copy()
        draw_map = draw_object(draw_map, paper, scissor, stone, paper_image, scissor_image, stone_image)

        # move object
        next_paper = move_object(paper, stone, map_size, image_size) if stone.shape[0] != 0 else move_object(paper, scissor, map_size, image_size)
        next_scissor = move_object(scissor, paper, map_size, image_size) if paper.shape[0] != 0 else move_object(scissor, stone, map_size, image_size)
        next_stone = move_object(stone, scissor, map_size, image_size) if scissor.shape[0] != 0 else move_object(stone, paper, map_size, image_size)

        paper = next_paper
        scissor = next_scissor
        stone = next_stone

        # touch detection
        paper, stone = touch_detection(paper, stone, threshold)
        scissor, paper = touch_detection(scissor, paper, threshold)
        stone, scissor = touch_detection(stone, scissor, threshold)

        draw_map = cv2.cvtColor(draw_map.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imshow('Paper Scissor Stone Simulator', draw_map)
        frame_list.append(draw_map)

        sleep(0.01)

        k = cv2.waitKey(33)
        if k == 27:  # Esc key to stop
            breakon = True

        if breakon:
            break

    # Save as video
    save_as_video(frame_list, (map_size, map_size))


if __name__ == '__main__':
    simulate_game(map_size=500, object_number=10)

