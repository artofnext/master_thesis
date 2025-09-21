import json

from torch.utils.data import DataLoader
from tqdm import tqdm

import inference
import classificator
import torch

from data_loader import SRDataset, ClassifierDataset


def generate_classify_and_calculate(test_image_dir, sr_model_path, classificator_model_path, save_path=None, batch_size=1,
                                    num_steps=500, crop_size=128, scale_factor=2):
    #  Dataset loader (LR images and ground true classes)
    test_dataset = ClassifierDataset(image_dir=test_image_dir, crop_size=crop_size, downscale=scale_factor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    num_classes = test_dataset.num_classes
    null_class_idx = test_dataset.NULL_CLASS[1]

    #  m1 - classification of ground true image; m2 - classification of generated image
    #  a: m1 & m2 correct; b: m1 correct, m2 wrong; c: m1 wrong m2 correct; d: m1 wrong, m2 correct
    mcnemar_stats = {"a": 0, "b": 0, "c": 0, "d": 0, "mcnemar": 0.0}

    image_classes = []
    ground_truth_predictions = []
    generated_predictions = []

    # this data for two sets statistical comparison
    # lists of stats data that represents pairs of classifications of the ground true and generated images as True/False
    # True if classification is right and False if classification is wrong with respect to the original image class in dataset
    is_correct_ground_true_predictions = []
    is_correct_generated_predictions = []

    #  Initialize SR diffusion model
    sr_model = inference.get_sr_model_state(sr_model_path, num_classes=num_classes + 1)  # +1  due to cfg null class
    scheduler = inference.get_scheduler(num_steps)
    #  Initialize Classificator
    classifier_model = classificator.get_model_state(classificator_model_path, num_classes=num_classes)

    # TODO delete stopper
    # stopper = 0
    for hr, lr, cls in tqdm(test_loader, desc="Processing test dataset"):

        # stopper += 1

        class_index = cls[0].item()

        # add image class to the list
        image_classes.append(class_index)

        #  generate image with class
        # true_image = inference.generate_image(crop_size, lr, cls, sr_model, scheduler, num_steps)

        #  get high resolution ground true image and classify it
        true_image = hr
        true_image_cls = classificator.classify_image(classifier_model, true_image)

        ground_truth_predictions.append(true_image_cls)

        is_correct_ground_true_predictions.append(True if true_image_cls != true_image_cls else False)

        # with null class
        generated_image = inference.generate_image(crop_size, lr, torch.tensor([null_class_idx]), sr_model, scheduler,
                                                   num_steps)
        generated_image_cls = classificator.classify_image(classifier_model, generated_image)

        generated_predictions.append(generated_image_cls)

        is_correct_generated_predictions.append(True if generated_image_cls != true_image_cls else False)

        #  collect mcnemar stats
        if true_image_cls == class_index and generated_image_cls == class_index:
            mcnemar_stats["a"] += 1
        elif true_image_cls != class_index and not generated_image_cls != class_index:
            mcnemar_stats["b"] += 1
        elif not true_image_cls != class_index and generated_image_cls != class_index:
            mcnemar_stats["c"] += 1
        else:
            mcnemar_stats["d"] += 1

        # if stopper >= 50:
        #     break
    assert (mcnemar_stats["b"] + mcnemar_stats["c"]) != 0, "Can't calculate Mcnemar stats, b + c = 0, division by 0"

    mcnemar_stats["mcnemar"] = (mcnemar_stats["b"] - mcnemar_stats["c"]) ** 2 / (
                mcnemar_stats["b"] + mcnemar_stats["c"])

    base_stats_data = {
        'image_classes': image_classes,
        'ground_truth_predictions': ground_truth_predictions,
        'generated_predictions': generated_predictions,
    }

    is_correct_predictions = {
        'is_correct_ground_true_predictions': is_correct_ground_true_predictions,
        'is_correct_generated_predictions': is_correct_generated_predictions,
    }

    data = {
        'mcnemar_stats': mcnemar_stats,
        'base_stats_data': base_stats_data,
        'is_correct_predictions': is_correct_predictions
    }
    if save_path is None:
        save_path = "stats/stat_data.json"
    # Saving dictionaries to JSON files
    with open(save_path, "w") as stat_data_file:
        json.dump(data, stat_data_file)

    return data
