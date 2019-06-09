import cv2
import os
import glob
from os.path import isfile
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, log_loss, precision_recall_curve
import pandas as pd
from matplotlib import pyplot as plt
from inspect import signature
from random import sample


def get_best_canny_params_for_dataset(train_images, human_edges_train_images):
    resolution = 16
    best_score = 0
    best_low_threshold, best_high_threshold = 0, 0
    best_aperture_size = 3
    for low_threshold in range(resolution - 1, 255, resolution):
        for high_threshold in range(low_threshold + resolution, 255, resolution):
            for aperture_size in (3, 5, 7):
                score = 0
                for image, human_edges in zip(train_images, human_edges_train_images):
                    _, human_edges_thresholded = cv2.threshold(human_edges, 127, 255, cv2.THRESH_BINARY)
                    canny_edges = cv2.Canny(image, low_threshold, high_threshold,
                                            apertureSize=aperture_size, L2gradient=True)
                    score += f1_score(human_edges_thresholded.flatten(),
                                      canny_edges.flatten(),
                                      average='macro')
                score /= len(train_images)
                if score >= best_score:
                    best_score = score
                    best_low_threshold, best_high_threshold = low_threshold, high_threshold
                    best_aperture_size = aperture_size

    print("Best edge detection score:", best_score)
    print("Optimal thresholds:", best_low_threshold, best_high_threshold)
    print("Optimal aperture size:", best_aperture_size)
    return best_low_threshold, best_high_threshold, best_aperture_size


class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0
        self.blobs = blobs
        self.params = params

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]


cv2.dnn_registerLayer('Crop', CropLayer)


def get_hed_edges(model, image):
    width, height, _ = image.shape
    inp = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(width, height),
                                mean=(104.00698793, 116.66876762, 122.67891434),
                                swapRB=False, crop=False)
    model.setInput(inp)
    out = model.forward()
    out = out[0, 0]
    out = cv2.resize(out, (height, width))
    out = 255 * out
    out = out.astype(np.uint8)
    return out


def get_classification_based_evaluation_for_edges(edges_image, human_edges):
    _, edges_thresholded = cv2.threshold(edges_image, 127, 255, cv2.THRESH_BINARY)
    _, human_edges_thresholded = cv2.threshold(human_edges, 127, 255, cv2.THRESH_BINARY)

    predicted_edge_probs = np.array(edges_image / 255, dtype="float").flatten()
    predicted_edges = np.array(edges_thresholded / 255, dtype="int").flatten()
    true_edges = np.array(human_edges_thresholded / 255, dtype="int").flatten()

    scores = [accuracy_score(y_true=true_edges,
                             y_pred=predicted_edges),
              precision_score(y_true=true_edges,
                              y_pred=predicted_edges,
                              average='macro'),
              recall_score(y_true=true_edges,
                           y_pred=predicted_edges,
                           average='macro'),
              f1_score(y_true=true_edges,
                       y_pred=predicted_edges,
                       average='macro'),
              roc_auc_score(y_true=true_edges,
                            y_score=predicted_edge_probs,
                            average='macro'),
              average_precision_score(y_true=true_edges,
                                      y_score=predicted_edge_probs,
                                      average='macro'),
              log_loss(y_true=true_edges,
                       y_pred=predicted_edge_probs)]
    return scores


def draw_precision_recall_curve_for_edge_detector(algorithm, dataset, edges_image, human_edges_image, plot_path):
    human_edges_image_thresholded = cv2.threshold(human_edges_image, 127, 255, cv2.THRESH_BINARY)[1]

    predicted_edges_probs = np.array(edges_image / 255, dtype="float").flatten()
    true_edges = np.array(human_edges_image_thresholded / 255, dtype="int").flatten()

    precision, recall = precision_recall_curve(true_edges, predicted_edges_probs)[:2]
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})

    plt.figure(figsize=(16, 9), dpi=1280 // 16)
    plt.title("Precision-recall curve for the " + algorithm + " edge detector evaluated on the " + dataset + " dataset"
              + '\n' + "Average precision: " + str(average_precision_score(true_edges, predicted_edges_probs,
                                                                           average="weighted")) + '\n', fontsize=24)
    plt.step(recall, precision, color='orange', alpha=0.75, where='post')
    plt.fill_between(recall, precision, alpha=0.75, color='orange', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.savefig(fname=plot_path, dpi="figure", format="png")


def generate_random_collage(dataset, images, canny_edges_images, hed_edges_images, human_edges_images, plot_path):
    indexes = sample(range(len(images)), 5)
    plt.figure(figsize=(5 * 16, 4 * 9), dpi=1920 // (5 * 16))
    for k, i in enumerate(indexes):
        collage = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        banner = [collage, canny_edges_images[i], hed_edges_images[i], human_edges_images[i]]
        banner = [cv2.resize(image, (480, 360)) for image in banner]
        collage = np.hstack(tuple(banner))
        plt.subplot(5, 1, k + 1)
        plt.imshow(collage, cmap="gray")
        plt.xticks([])
        plt.yticks([])
        if k == 0:
            plt.title("Edge detection on the " + dataset + " dataset\n", fontsize=96)
    plt.savefig(fname=plot_path, dpi="figure", format="png")


if __name__ == "__main__":

    if not os.path.exists('/canny_edges/bsds300') or not os.path.exists('/canny_edges/bsds500'):
        os.makedirs('/canny_edges/bsds300')
        os.makedirs('/canny_edges/bsds500')

    if not os.path.exists('/hed_edges/bsds300') or not os.path.exists('/hed_edges/bsds500'):
        os.makedirs('/hed_edges/bsds300')
        os.makedirs('/hed_edges/bsds500')

    if not os.path.exists('/plots'):
        os.makedirs('/plots')

    if not os.path.exists('/scores'):
        os.makedirs('/scores')

    bsds300_dataset_path = "../BSDS300"

    bsds300_train_images_files = glob.glob(bsds300_dataset_path + "/images/train/*.jpg")
    bsds300_train_images = [cv2.imread(image_file, cv2.IMREAD_COLOR) for image_file in bsds300_train_images_files]
    print("Number of training images in the BSDS300 dataset:", len(bsds300_train_images))
    bsds300_test_images_files = glob.glob(bsds300_dataset_path + "/images/test/*.jpg")
    bsds300_test_images = [cv2.imread(image_file, cv2.IMREAD_COLOR) for image_file in bsds300_test_images_files]
    print("Number of testing images in the BSDS300 dataset:", len(bsds300_test_images))
    print()

    bsds300_human_edges_files = glob.glob(bsds300_dataset_path + "/humanpb/color/human/*.bmp")
    bsds300_human_edges_images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in bsds300_human_edges_files]

    bsds300_canny_edges_files = glob.glob("canny_edges/bsds300/*.bmp")
    if len(bsds300_canny_edges_files) != len(bsds300_test_images):
        print("Finding best Canny parameters from train images...")
        best_canny_params = get_best_canny_params_for_dataset(bsds300_test_images[:70], bsds300_human_edges_images[:70])
        print("Performing Canny edge detection on the BSDS300 dataset...")
        bsds300_canny_edges = []
        for index, (bsds300_test_image, bsds300_human_edges_image) in \
                enumerate(zip(bsds300_test_images, bsds300_human_edges_images)):
            if (index + 1) % 10 == 0:
                print("Processing image #" + str(index + 1) + "...")
            edges = cv2.Canny(bsds300_test_image, threshold1=best_canny_params[0], threshold2=best_canny_params[1],
                              apertureSize=best_canny_params[2], L2gradient=True)
            bsds300_canny_edges.append(edges)
            cv2.imwrite("canny_edges/bsds300/" + "{:03d}".format(index + 1) + ".bmp", edges)
    else:
        bsds300_canny_edges = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in bsds300_canny_edges_files]
    print()

    hed_model = cv2.dnn.readNetFromCaffe(prototxt="hed_model.prototxt", caffeModel="hed_model.caffemodel")
    print("Types of layes used in the HED network:")
    print('\n'.join(hed_model.getLayerTypes()))
    print()
    print("Complete architecture of the HED network:")
    print('\n'.join(hed_model.getLayerNames()))
    print()

    bsds300_hed_edges_files = glob.glob("hed_edges/bsds300/*.bmp")
    if len(bsds300_hed_edges_files) != len(bsds300_test_images):
        print("Performing HED on the BSDS300 dataset...")
        bsds300_hed_edges = []
        for index, bsds300_test_image in enumerate(bsds300_test_images):
            if (index + 1) % 10 == 0:
                print("Processing image #" + str(index + 1) + "...")
            edges = get_hed_edges(hed_model, bsds300_test_image)
            bsds300_hed_edges.append(edges)
            cv2.imwrite("hed_edges/bsds300/" + "{:03d}".format(index + 1) + ".bmp", edges)
    else:
        bsds300_hed_edges = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in bsds300_hed_edges_files]
    print()

    generate_random_collage("BSDS300", bsds300_test_images, bsds300_canny_edges, bsds300_hed_edges,
                            bsds300_human_edges_images, "plots/bsds300_collage.png")

    algorithms = ['Canny', 'HED']
    classification_evaluation_metrics = ['Accuracy', 'Precision', 'Recall', 'F1',
                                         'ROC AUC', 'Average precision', 'Cross-entropy loss']
    score_table_features = []
    for metric in classification_evaluation_metrics:
        score_table_features.append(metric + " MEAN")
        score_table_features.append(metric + " VARIANCE")

    bsds300_canny_scores = [
        get_classification_based_evaluation_for_edges(bsds300_canny_edges_image, bsds300_human_edges_image)
        for bsds300_canny_edges_image, bsds300_human_edges_image in
        zip(bsds300_canny_edges[70:], bsds300_human_edges_images[70:])]

    bsds300_hed_scores = [
        get_classification_based_evaluation_for_edges(bsds300_hed_edges_image, bsds300_human_edges_image)
        for bsds300_hed_edges_image, bsds300_human_edges_image in
        zip(bsds300_hed_edges[70:], bsds300_human_edges_images[70:])]

    if not isfile("scores/bsds300_classification_scores.csv"):
        bsds300_mean_canny_scores = np.mean(bsds300_canny_scores, axis=0)
        bsds300_variance_canny_scores = np.var(bsds300_canny_scores, axis=0)
        bsds300_compound_canny_scores = []
        for mean_score, variance_score in zip(bsds300_mean_canny_scores, bsds300_variance_canny_scores):
            bsds300_compound_canny_scores.append(round(mean_score, 4))
            bsds300_compound_canny_scores.append(round(variance_score, 4))
        bsds300_compound_canny_scores = np.array(bsds300_compound_canny_scores, dtype="float")

        bsds300_mean_hed_scores = np.mean(bsds300_hed_scores, axis=0)
        bsds300_variance_hed_scores = np.var(bsds300_hed_scores, axis=0)
        bsds300_compound_hed_scores = []
        for mean_score, variance_score in zip(bsds300_mean_hed_scores, bsds300_variance_hed_scores):
            bsds300_compound_hed_scores.append(round(mean_score, 4))
            bsds300_compound_hed_scores.append(round(variance_score, 4))
        bsds300_compound_hed_scores = np.array(bsds300_compound_hed_scores, dtype="float")

        bsds300_scores = np.vstack((bsds300_compound_canny_scores, bsds300_compound_hed_scores))
        bsds300_scores = pd.DataFrame(bsds300_scores, algorithms, score_table_features, dtype="float")
        bsds300_scores.to_csv("scores/bsds300_classification_scores.csv", encoding="utf-8")
    else:
        bsds300_scores = pd.read_csv("scores/bsds300_classification_scores.csv", header=0, index_col=0)
    print("Classification-based scores on BSDS300:\n", bsds300_scores)
    print()

    bsds300_best_canny_result_index = np.argmax(bsds300_canny_scores, axis=0)[5]
    bsds300_best_hed_result_index = np.argmax(bsds300_hed_scores, axis=0)[5]

    draw_precision_recall_curve_for_edge_detector("Canny", "BSDS300",
                                                  bsds300_canny_edges[bsds300_best_canny_result_index],
                                                  bsds300_human_edges_images[bsds300_best_canny_result_index],
                                                  "plots/bsds300_canny_pr_curve.png")
    draw_precision_recall_curve_for_edge_detector("HED", "BSDS300",
                                                  bsds300_hed_edges[bsds300_best_hed_result_index],
                                                  bsds300_human_edges_images[bsds300_best_hed_result_index],
                                                  "plots/bsds300_hed_pr_curve.png")

    bsds500_dataset_path = "../BSR/BSDS500/data/"
    bsds500_train_images_files = glob.glob(bsds500_dataset_path + "/images/train/*.jpg")
    bsds500_train_images = [cv2.imread(image_file, cv2.IMREAD_COLOR) for image_file in bsds500_train_images_files]
    print("Number of training images in the BSDS500 dataset:", len(bsds500_train_images))
    bsds500_validation_images_files = glob.glob(bsds500_dataset_path + "/images/val/*.jpg")
    bsds500_validation_images = [cv2.imread(image_file, cv2.IMREAD_COLOR)
                                 for image_file in bsds500_validation_images_files]
    print("Number of validation images in the BSDS500 dataset:", len(bsds500_validation_images))
    bsds500_test_images_files = glob.glob(bsds500_dataset_path + "/images/test/*.jpg")
    bsds500_test_images = [cv2.imread(image_file, cv2.IMREAD_COLOR) for image_file in bsds500_test_images_files]
    print("Number of testing images in the BSDS500 dataset:", len(bsds500_test_images))
    print()

    bsds500_human_train_edges_files = glob.glob(bsds500_dataset_path + "/groundTruth/train/*.png")
    bsds500_human_train_edges_images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                                        for file in bsds500_human_train_edges_files]
    bsds500_human_validation_edges_files = glob.glob(bsds500_dataset_path + "/groundTruth/val/*.png")
    bsds500_human_validation_edges_images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                                             for file in bsds500_human_validation_edges_files]
    bsds500_human_test_edges_files = glob.glob(bsds500_dataset_path + "/groundTruth/test/*.png")
    bsds500_human_test_edges_images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                                       for file in bsds500_human_test_edges_files]

    bsds500_canny_edges_files = glob.glob("canny_edges/bsds500/*.bmp")
    if len(bsds500_canny_edges_files) != len(bsds500_test_images):
        print("Finding best Canny parameters from validation images...")
        best_canny_params = get_best_canny_params_for_dataset(bsds500_validation_images,
                                                              bsds500_human_validation_edges_images)
        print("Performing Canny edge detection on the BSDS500 dataset...")
        bsds500_canny_edges = []
        for index, (bsds500_test_image, bsds500_human_edges_image) in \
                enumerate(zip(bsds500_test_images, bsds500_human_test_edges_images)):
            if (index + 1) % 10 == 0:
                print("Processing image #" + str(index + 1) + "...")
            edges = cv2.Canny(bsds500_test_image, threshold1=best_canny_params[0], threshold2=best_canny_params[1],
                              apertureSize=best_canny_params[2], L2gradient=True)
            bsds500_canny_edges.append(edges)
            cv2.imwrite("canny_edges/bsds500/" + "{:03d}".format(index + 1) + ".bmp", edges)
    else:
        bsds500_canny_edges = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in bsds500_canny_edges_files]
    print()

    bsds500_hed_edges_files = glob.glob("hed_edges/bsds500/*.bmp")
    if len(bsds500_hed_edges_files) != len(bsds500_test_images):
        print("Performing HED on the BSDS500 dataset...")
        bsds500_hed_edges = []
        for index, bsds500_test_image in enumerate(bsds500_test_images):
            if (index + 1) % 10 == 0:
                print("Processing image #" + str(index + 1) + "...")
            edges = get_hed_edges(hed_model, bsds500_test_image)
            bsds500_hed_edges.append(edges)
            cv2.imwrite("hed_edges/bsds500/" + "{:03d}".format(index + 1) + ".bmp", edges)
    else:
        bsds500_hed_edges = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in bsds500_hed_edges_files]
    print()

    generate_random_collage("BSDS500", bsds500_test_images, bsds500_canny_edges, bsds500_hed_edges,
                            bsds500_human_test_edges_images, "plots/bsds500_collage.png")

    bsds500_canny_scores = [
        get_classification_based_evaluation_for_edges(bsds500_canny_edges_image, bsds500_human_edges_image)
        for bsds500_canny_edges_image, bsds500_human_edges_image in
        zip(bsds500_canny_edges, bsds500_human_test_edges_images)]
    bsds500_hed_scores = [
        get_classification_based_evaluation_for_edges(bsds500_hed_edges_image, bsds500_human_edges_image)
        for bsds500_hed_edges_image, bsds500_human_edges_image in
        zip(bsds500_hed_edges, bsds500_human_test_edges_images)]

    if not isfile("scores/bsds500_classification_scores.csv"):
        bsds500_mean_canny_scores = np.mean(bsds500_canny_scores, axis=0)
        bsds500_variance_canny_scores = np.var(bsds500_canny_scores, axis=0)
        bsds500_compound_canny_scores = []
        for mean_score, variance_score in zip(bsds500_mean_canny_scores, bsds500_variance_canny_scores):
            bsds500_compound_canny_scores.append(round(mean_score, 4))
            bsds500_compound_canny_scores.append(round(variance_score, 4))
        bsds500_compound_canny_scores = np.array(bsds500_compound_canny_scores, dtype="float")

        bsds500_mean_hed_scores = np.mean(bsds500_hed_scores, axis=0)
        bsds500_variance_hed_scores = np.var(bsds500_hed_scores, axis=0)
        bsds500_compound_hed_scores = []
        for mean_score, variance_score in zip(bsds500_mean_hed_scores, bsds500_variance_hed_scores):
            bsds500_compound_hed_scores.append(round(mean_score, 4))
            bsds500_compound_hed_scores.append(round(variance_score, 4))
        bsds500_compound_hed_scores = np.array(bsds500_compound_hed_scores, dtype="float")

        bsds500_scores = np.vstack((bsds500_compound_canny_scores, bsds500_compound_hed_scores))
        bsds500_scores = pd.DataFrame(bsds500_scores, algorithms, score_table_features, dtype="float")
        bsds500_scores.to_csv("scores/bsds500_classification_scores.csv", encoding="utf-8")
    else:
        bsds500_scores = pd.read_csv("scores/bsds500_classification_scores.csv", header=0, index_col=0)
    print("Classification-based scores on BSDS500:\n", bsds500_scores)

    bsds500_best_canny_result_index = np.argmax(bsds500_canny_scores, axis=0)[5]
    bsds500_best_hed_result_index = np.argmax(bsds500_hed_scores, axis=0)[5]

    draw_precision_recall_curve_for_edge_detector("Canny", "BSDS500",
                                                  bsds500_canny_edges[bsds500_best_canny_result_index],
                                                  bsds500_human_test_edges_images[bsds500_best_canny_result_index],
                                                  "plots/bsds500_canny_pr_curve.png")
    draw_precision_recall_curve_for_edge_detector("HED", "BSDS500",
                                                  bsds500_hed_edges[bsds500_best_hed_result_index],
                                                  bsds500_human_test_edges_images[bsds500_best_hed_result_index],
                                                  "plots/bsds500_hed_pr_curve.png")
