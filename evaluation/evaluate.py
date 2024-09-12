"""
The following is a simple example evaluation method.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the evaluation, reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./save.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
import json
from pathlib import Path
from pprint import pformat, pprint
from monai.metrics import compute_froc_score, compute_froc_curve_data
from scipy.spatial import distance
from sklearn.metrics import auc
import numpy as np

from helpers import run_prediction_processing

# for local debugging
import os
INPUT_DIRECTORY = Path(f"{os.getcwd()}/test/input")
OUTPUT_DIRECTORY = Path(f"{os.getcwd()}/test/output")
GROUND_TRUTH_DIRECTORY = Path(f"{os.getcwd()}/ground_truth")

# for docker building
# INPUT_DIRECTORY = Path("/input")
# OUTPUT_DIRECTORY = Path("/output")
# GROUND_TRUTH_DIRECTORY = Path("/opt/ml/input/data/ground_truth")

SPACING_LEVEL0 = 0.24199951445730394

def process(job):
    """Processes a single algorithm job, looking at the outputs"""
    report = "Processing:\n"
    report += pformat(job)
    report += "\n"

    # Firstly, find the location of the results
    location_detected_lymphocytes = get_file_location(
        job_pk=job["pk"],
        values=job["outputs"],
        slug="detected-lymphocytes",
    )
    location_detected_monocytes = get_file_location(
        job_pk=job["pk"],
        values=job["outputs"],
        slug="detected-monocytes",
    )
    location_detected_inflammatory_cells = get_file_location(
        job_pk=job["pk"],
        values=job["outputs"],
        slug="detected-inflammatory-cells",
    )

    # Secondly, read the results
    result_detected_lymphocytes = load_json_file(
        location=location_detected_lymphocytes,
    )
    result_detected_lymphocytes = convert_mm_to_pixel(result_detected_lymphocytes)

    result_detected_monocytes = load_json_file(
        location=location_detected_monocytes,
    )
    result_detected_monocytes = convert_mm_to_pixel(result_detected_monocytes)

    result_detected_inflammatory_cells = load_json_file(
        location=location_detected_inflammatory_cells,
    )
    result_detected_inflammatory_cells = convert_mm_to_pixel(result_detected_inflammatory_cells)

    # Thirdly, retrieve the input image name to match it with an image in your ground truth
    file_id = get_image_name(
        values=job["inputs"],
        slug="kidney-transplant-biopsy",
    )
    file_id = file_id.split("_PAS")[0]
    # Fourthly, load your ground truth
    # Include it in your evaluation container by placing it in ground_truth/
    gt_lymphocytes = load_json_file(location=GROUND_TRUTH_DIRECTORY / f"{file_id}_lymphocytes.json")
    gt_monocytes = load_json_file(location=GROUND_TRUTH_DIRECTORY / f"{file_id}_monocytes.json")
    gt_inf_cells = load_json_file(location=GROUND_TRUTH_DIRECTORY / f"{file_id}_inflammatory-cells.json")

    # compare the results to your ground truth and compute some metrics
    lymphocytes_froc = get_froc_vals(gt_lymphocytes, result_detected_lymphocytes,
                                     radius=int(4 / SPACING_LEVEL0))  # margin for lymphocytes is 4um at spacing 0.25 um / pixel
    monocytes_froc = get_froc_vals(gt_monocytes, result_detected_monocytes,
                                   radius=int(10 / 0.25))  # margin for monocytes is 10um at spacing 0.25 um / pixel
    inflamm_froc = get_froc_vals(gt_inf_cells, result_detected_inflammatory_cells, radius=int(
        7.5 / SPACING_LEVEL0))  # margin for inflammatory cells is 7.5um at spacing 0.24 um / pixel

    report += "Lymphocytes FROC:\n" + pformat({k: v for k, v in lymphocytes_froc.items() if type(v) is not list}) + "\n"
    report += "Monocytes FROC:\n" + pformat({k: v for k, v in monocytes_froc.items() if type(v) is not list}) + "\n"
    report += "Inflammatory cells FROC:\n" + pformat({k: v for k, v in inflamm_froc.items() if type(v) is not list}) + "\n"

    print(report)

    # Finally, calculate by comparing the ground truth to the actual results
    return (file_id, {
        'lymphocytes': lymphocytes_froc,
        'monocytes': monocytes_froc,
        'inflammatory-cells': inflamm_froc
    })


def get_froc_vals(gt_dict, result_dict, radius: int):
    """
    Computes the Free-Response Receiver Operating Characteristic (FROC) values for given ground truth and result data.
    Using https://docs.monai.io/en/0.5.0/_modules/monai/metrics/froc.html
    Args:
        gt_dict (dict): Ground truth data containing points and regions of interest (ROIs).
        result_dict (dict): Result data containing detected points and their probabilities.
        radius (int): The maximum distance in pixels for considering a detection as a true positive.

    Returns:
        dict: A dictionary containing FROC metrics such as sensitivity, false positives per mm²,
              true positive probabilities, false positive probabilities, total positives,
              area in mm², and FROC score.
    """
    # create a mask from the gt coordinates with circles of given radius
    if len(gt_dict['points']) == 0:
        return {'sensitivity_slide': [0], 'fp_per_mm2_slide': [0], 'fp_probs_slide': [0],
                'tp_probs_slide': [0], 'total_pos_slide': 0, 'area_mm2_slide': 0, 'froc_score_slide': 0}
    gt_coords = [i['point'] for i in gt_dict['points']]
    gt_rois = [i['polygon'] for i in gt_dict['rois']]
    # compute the area of the polygon in roi
    area_mm2 = SPACING_LEVEL0 * SPACING_LEVEL0 * gt_dict["area_rois"] / 1000000
    # result_prob = [i['probability'] for i in result_dict['points']]
    result_prob = [i['probability'] for i in result_dict['points']]
    # make some dummy values between 0 and 1 for the result prob
    # result_prob = [np.random.rand() for i in range(len(result_dict['points']))]
    result_coords = [[i['point'][0], i['point'][1]] for i in result_dict['points']]

    # prepare the data for the FROC curve computation with monai
    true_positives, false_negatives, false_positives, tp_probs, fp_probs = match_coordinates(gt_coords, result_coords,
                                                                                             result_prob, radius)
    total_pos = len(gt_coords)
    # the metric is implemented to normalize by the number of images, we however want to have it by mm2, so we set
    # num_images = ROI area in mm2
    fp_per_mm2_slide, sensitivity = compute_froc_curve_data(fp_probs, tp_probs, total_pos, area_mm2)
    if len(fp_per_mm2_slide) > 1 and len(sensitivity) > 1:
        area_under_froc = auc(fp_per_mm2_slide, sensitivity)
        froc_score = compute_froc_score(fp_per_mm2_slide, sensitivity, eval_thresholds=(10, 20, 50, 100, 200, 300))
    else:
        area_under_froc = 0
        froc_score = 0

    return {'sensitivity_slide': list(sensitivity), 'fp_per_mm2_slide': list(fp_per_mm2_slide),
            'fp_probs_slide': list(fp_probs), 'tp_probs_slide': list(tp_probs), 'total_pos_slide': total_pos,
            'area_mm2_slide': area_mm2, 'froc_score_slide': float(froc_score)}


def match_coordinates(ground_truth, predictions, pred_prob, margin):
    """
    Matches predicted coordinates to ground truth coordinates within a certain distance margin
    and computes the associated probabilities for true positives and false positives.

    Args:
        ground_truth (list of tuples): List of ground truth coordinates as (x, y).
        predictions (list of tuples): List of predicted coordinates as (x, y).
        pred_prob (list of floats): List of probabilities associated with each predicted coordinate.
        margin (float): The maximum distance for considering a prediction as a true positive.

    Returns:
        true_positives (int): Number of correctly matched predictions.
        false_negatives (int): Number of ground truth coordinates not matched by any prediction.
        false_positives (int): Number of predicted coordinates not matched by any ground truth.
        tp_probs (list of floats): Probabilities of the true positive predictions.
        fp_probs (list of floats): Probabilities of the false positive predictions.
    """
    if len(ground_truth) == 0 or len(predictions) == 0:
        return 0, 0, 0, np.array([]), np.array([])
        # return true_positives, false_negatives, false_positives, np.array(tp_probs), np.array(fp_probs)
    # Convert lists to numpy arrays for easier distance calculations
    gt_array = np.array(ground_truth)
    pred_array = np.array(predictions)
    pred_prob_array = np.array(pred_prob)

    # Distance matrix between ground truth and predictions
    dist_matrix = distance.cdist(gt_array, pred_array)

    # Initialize sets for matched indices
    matched_gt = set()
    matched_pred = set()

    # Iterate over the distance matrix to find the closest matches
    for gt_idx in range(len(ground_truth)):
        closest_pred_idx = np.argmin(dist_matrix[gt_idx])
        if dist_matrix[gt_idx, closest_pred_idx] <= margin:
            matched_gt.add(gt_idx)
            matched_pred.add(closest_pred_idx)

    # Calculate true positives, false negatives, and false positives
    true_positives = len(matched_gt)
    false_negatives = len(ground_truth) - true_positives
    false_positives = len(predictions) - true_positives

    # Compute probabilities for true positives and false positives
    tp_probs = [pred_prob[i] for i in matched_pred]
    fp_probs = [pred_prob[i] for i in range(len(predictions)) if i not in matched_pred]

    return true_positives, false_negatives, false_positives, np.array(tp_probs), np.array(fp_probs)


def main():
    print_inputs()
    predictions = read_predictions()
    metrics = {}

    # We now process each algorithm job for this submission
    # Note that the jobs are not in any order!
    # We work that out from predictions.json

    # Use concurrent workers to process the predictions more efficiently
    results = run_prediction_processing(fn=process, predictions=predictions)
    file_ids = [r[0] for r in results]
    metrics_per_slide = [r[1] for r in results]
    metrics['per_slide'] = {file_id: metrics_per_slide[i] for i, file_id in enumerate(file_ids)}

    # We have the results per prediction, we can aggregate over the results and
    # generate an overall score(s) for this submission
    lymphocytes_metrics = format_metrics_for_aggr(metrics_per_slide, 'lymphocytes')
    monocytes_metrics = format_metrics_for_aggr(metrics_per_slide, 'monocytes')
    inflammatory_cells_metrics = format_metrics_for_aggr(metrics_per_slide, 'inflammatory-cells')
    aggregated_metrics = {
        'lymphocytes': get_aggr_froc(lymphocytes_metrics),
        'monocytes': get_aggr_froc(monocytes_metrics),
        'inflammatory-cells': get_aggr_froc(inflammatory_cells_metrics)
    }

    # clean up the per file metrics
    for file_id, file_metrics in metrics['per_slide'].items():
        for cell_type in ['lymphocytes', 'monocytes', 'inflammatory-cells']:
            for i in ['sensitivity_slide', 'fp_per_slide', 'fp_probs_slide', 'tp_probs_slide',
                      'total_pos_slide']:
                if i in file_metrics[cell_type]:
                    del file_metrics[cell_type][i]

    # Aggregate the metrics_per_slide
    metrics["aggregates"] = aggregated_metrics

    # Make sure to save the metrics
    write_metrics(metrics=metrics)

    return 0


def get_aggr_froc(metrics_dict):
    # https://docs.monai.io/en/0.5.0/_modules/monai/metrics/froc.html
    fp_probs = np.array([item for sublist in metrics_dict['fp_probs_slide'] for item in sublist])
    tp_probs = np.array([item for sublist in metrics_dict['tp_probs_slide'] for item in sublist])
    total_pos = sum(metrics_dict['total_pos_slide'])
    area_mm2 = sum(metrics_dict['area_mm2_slide'])

    # sensitivity, fp_overall = compute_froc_curve_data(fp_probs, tp_probs, total_pos, area_mm2)
    fp_overall, sensitivity_overall, = compute_froc_curve_data(fp_probs, tp_probs, total_pos, area_mm2)
    if len(fp_overall) > 1 and len(sensitivity_overall) > 1:
        area_under_froc = auc(fp_overall, sensitivity_overall)
        froc_score = compute_froc_score(fp_overall, sensitivity_overall, eval_thresholds=(10, 20, 50, 100, 200, 300))
    else:
        area_under_froc = 0
        froc_score = 0

    # return {'sensitivity_aggr': list(sensitivity_overall), 'fp_aggr': list(fp_overall),
    #         'fp_probs_aggr': list(fp_probs),
    #         'tp_probs_aggr': list(tp_probs), 'total_pos_aggr': total_pos, 'area_mm2_aggr': area_mm2,
    #         'froc_score_aggr': float(froc_score)}

    return {'area_mm2_aggr': area_mm2,
            'froc_score_aggr': float(froc_score)}


def format_metrics_for_aggr(metrics_list, cell_type):
    """
    Formats the metrics dictionary to be used in the aggregation function.
    """
    aggr = {}
    for d in [i[cell_type] for i in metrics_list]:
        # Iterate over each key-value pair in the dictionary
        for key, value in d.items():
            # If the key is not already in the collapsed_dict, initialize it with an empty list
            if key not in aggr:
                aggr[key] = []
            # Append the value to the list corresponding to the key
            aggr[key].append(value)

    return aggr


def print_inputs():
    # Just for convenience, in the logs you can then see what files you have to work with
    input_files = [str(x) for x in Path(INPUT_DIRECTORY).rglob("*") if x.is_file()]

    print("Input Files:")
    pprint(input_files)
    print("")


def read_predictions():
    # The prediction file tells us the location of the users' predictions
    print(INPUT_DIRECTORY)
    with open(INPUT_DIRECTORY / "predictions.json") as f:
        return json.loads(f.read())


def get_image_name(*, values, slug):
    # This tells us the user-provided name of the input or output image
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["image"]["name"]

    raise RuntimeError(f"Image with interface {slug} not found!")


def get_interface_relative_path(*, values, slug):
    # Gets the location of the interface relative to the input or output
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["interface"]["relative_path"]

    raise RuntimeError(f"Value with interface {slug} not found!")


def get_file_location(*, job_pk, values, slug):
    # Where a job's output file will be located in the evaluation container
    relative_path = get_interface_relative_path(values=values, slug=slug)
    return INPUT_DIRECTORY / job_pk / "output" / relative_path


def load_json_file(*, location):
    # Reads a json file
    with open(location) as f:
        return json.loads(f.read())


def convert_mm_to_pixel(data_dict, spacing=SPACING_LEVEL0):
    # Converts a distance in mm to pixels: coord in mm * 1000 * spacing
    points_pixels = []
    for d in data_dict['points']:
        if len(d['point']) == 2:
            d['point'] = [mm_to_pixel(d['point'][0]), mm_to_pixel(d['point'][1]), 0]
        else:
            d['point'] = [mm_to_pixel(d['point'][0]), mm_to_pixel(d['point'][1]), mm_to_pixel(d['point'][2])]
        points_pixels.append(d)
    data_dict['points'] = points_pixels
    return data_dict


def mm_to_pixel(dist, spacing=SPACING_LEVEL0):
    spacing = spacing / 1000
    dist_px = int(round(dist / spacing))
    return dist_px


def write_metrics(*, metrics):
    # Write a json document used for ranking results on the leaderboard
    with open(OUTPUT_DIRECTORY / "metrics.json", "w") as f:
        f.write(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    raise SystemExit(main())
