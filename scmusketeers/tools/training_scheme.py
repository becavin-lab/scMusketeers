import logging

logger = logging.getLogger("Sc-Musketeers")

def get_training_scheme(training_scheme_id, run_file):
    if training_scheme_id == "training_scheme_1":
        training_scheme = [
            ("warmup_dann", run_file.warmup_epoch, False),
            ("full_model", run_file.fullmodel_epoch, True),
        ]  # This will end with a callback
    if training_scheme_id == "training_scheme_2":
        training_scheme = [
            ("warmup_dann_no_rec", run_file.warmup_epoch, False),
            ("full_model", run_file.fullmodel_epoch, True),
        ]  # This will end with a callback
    if training_scheme_id == "training_scheme_3":
        training_scheme = [
            ("warmup_dann", run_file.warmup_epoch, False),
            ("full_model", run_file.fullmodel_epoch, True),
            ("classifier_branch", run_file.classifier_epoch, False),
        ]  # This will end with a callback
    if training_scheme_id == "training_scheme_4":
        training_scheme = [
            ("warmup_dann", run_file.warmup_epoch, False),
            (
                "permutation_only",
                100,
                True,
            ),  # This will end with a callback
            ("classifier_branch", run_file.classifier_epoch, False),
        ]  # This will end with a callback
    if training_scheme_id == "training_scheme_5":
        training_scheme = [
            ("warmup_dann", run_file.warmup_epoch, False),
            ("full_model", run_file.fullmodel_epoch, False),
        ]  # This will end with a callback, NO PERMUTATION HERE
    if training_scheme_id == "training_scheme_6":
        training_scheme = [
            (
                "warmup_dann",
                run_file.warmup_epoch,
                True,
            ),  # Permutating with pseudo labels during warmup
            ("full_model", run_file.fullmodel_epoch, True),
        ]

    if training_scheme_id == "training_scheme_7":
        training_scheme = [
            (
                "warmup_dann",
                run_file.warmup_epoch,
                True,
            ),  # Permutating with pseudo labels during warmup
            ("full_model", run_file.fullmodel_epoch, False),
        ]

    if training_scheme_id == "training_scheme_8":
        training_scheme = [
            (
                "warmup_dann",
                run_file.warmup_epoch,
                False,
            ),  # No Permutating with pseudo labels during warmup
            ("full_model", run_file.fullmodel_epoch, False),
            ("classifier_branch", run_file.classifier_epoch, False),
        ]  # This will end with a callback]

    if training_scheme_id == "training_scheme_9":
        training_scheme = [
            ("warmup_dann", run_file.warmup_epoch, False),
            ("full_model", run_file.fullmodel_epoch, True),
            ("classifier_branch", run_file.classifier_epoch, False),
        ]  # This will end with a callback]

    if training_scheme_id == "training_scheme_10":
        training_scheme = [
            (
                "warmup_dann",
                run_file.warmup_epoch,
                True,
            ),  # Permutating with pseudo labels during warmup
            ("full_model", run_file.fullmodel_epoch, False),
            ("classifier_branch", run_file.classifier_epoch, False),
            (
                "warmup_dann_pseudolabels",
                run_file.warmup_epoch,
                True,
            ),  # Permutating with pseudo labels from the current model state
            ("full_model", run_file.fullmodel_epoch, False),
            ("classifier_branch", run_file.classifier_epoch, False),
        ]  # This will end with a callback

    if training_scheme_id == "training_scheme_11":
        training_scheme = [
            (
                "warmup_dann",
                run_file.warmup_epoch,
                True,
            ),  # Permutating with pseudo labels during warmup
            ("full_model", run_file.fullmodel_epoch, False),
            ("classifier_branch", run_file.classifier_epoch, False),
            (
                "full_model_pseudolabels",
                run_file.fullmodel_epoch,
                True,
            ),  # using permutations on plabels for full training
            ("classifier_branch", run_file.classifier_epoch, False),
        ]  # This will end with a callback

    if training_scheme_id == "training_scheme_12":
        training_scheme = [
            (
                "permutation_only",
                run_file.warmup_epoch,
                True,
            ),  # Permutating with pseudo labels during warmup
            ("classifier_branch", run_file.classifier_epoch, False),
        ]

    if training_scheme_id == "training_scheme_13":
        training_scheme = [
            ("full_model", run_file.fullmodel_epoch, True),
            ("classifier_branch", run_file.classifier_epoch, False),
        ]

    if training_scheme_id == "training_scheme_14":
        training_scheme = [
            ("full_model", run_file.fullmodel_epoch, False),
            ("classifier_branch", run_file.classifier_epoch, False),
        ]

    if training_scheme_id == "training_scheme_15":
        training_scheme = [
            ("warmup_dann_train", run_file.warmup_epoch, True),
            ("full_model", run_file.fullmodel_epoch, False),
            ("classifier_branch", run_file.classifier_epoch, False),
        ]

    if training_scheme_id == "training_scheme_16":
        training_scheme = [
            ("warmup_dann", run_file.warmup_epoch, True),
            ("full_model", run_file.fullmodel_epoch, True),
            ("classifier_branch", run_file.classifier_epoch, False),
        ]

    if training_scheme_id == "training_scheme_17":
        training_scheme = [
            ("no_dann", run_file.fullmodel_epoch, True),
            ("classifier_branch", run_file.classifier_epoch, False),
        ]

    if training_scheme_id == "training_scheme_18":
        training_scheme = [
            ("no_dann", run_file.fullmodel_epoch, False),
            ("classifier_branch", run_file.classifier_epoch, False),
        ]

    if training_scheme_id == "training_scheme_19":
        training_scheme = [
            (
                "warmup_dann",
                run_file.warmup_epoch,
                False,
            ),  # Permutating with pseudo labels during warmup
            ("full_model", run_file.fullmodel_epoch, False),
            ("classifier_branch", run_file.classifier_epoch, False),
        ]

    if training_scheme_id == "training_scheme_20":
        training_scheme = [
            (
                "warmup_dann_semisup",
                run_file.warmup_epoch,
                True,
            ),  # Permutating in semisup fashion ie unknown cells reconstruc themselves
            ("full_model", run_file.fullmodel_epoch, False),
            ("classifier_branch", run_file.classifier_epoch, False),
        ]

    if training_scheme_id == "training_scheme_21":
        training_scheme = [
            ("warmup_dann", run_file.warmup_epoch, False),
            ("no_dann", run_file.fullmodel_epoch, False),
            ("classifier_branch", run_file.classifier_epoch, False),
        ]

    if training_scheme_id == "training_scheme_22":
        training_scheme = [
            ("permutation_only", run_file.warmup_epoch, True),
            ("warmup_dann", run_file.warmup_epoch, True),
            ("full_model", run_file.fullmodel_epoch, False),
            ("classifier_branch", run_file.classifier_epoch, False),
        ]

    if training_scheme_id == "training_scheme_23":
        training_scheme = [("full_model", run_file.fullmodel_epoch, True)]

    if training_scheme_id == "training_scheme_24":
        training_scheme = [
            ("full_model", run_file.fullmodel_epoch, False),
        ]

    if training_scheme_id == "training_scheme_25":
        training_scheme = [
            ("no_decoder", run_file.fullmodel_epoch, False),
        ]

    if training_scheme_id == "training_scheme_26":
        training_scheme = [
            (
                "warmup_dann",
                run_file.warmup_epoch,
                False,
            ),  # Permutating with pseudo labels during warmup
            ("full_model", run_file.fullmodel_epoch, False),
            ("classifier_branch", run_file.classifier_epoch, False),
            (
                "full_model_pseudolabels",
                run_file.fullmodel_epoch,
                False,
            ),  # using permutations on plabels for full training
            ("classifier_branch", run_file.classifier_epoch, False),
        ]

    if training_scheme_id == "training_scheme_debug_1":
        training_scheme = [
            (
                "full_model_pseudolabels",
                run_file.warmup_epoch,
                False,
            ),  # Permutating with pseudo labels during warmup
            #("full_model", run_file.fullmodel_epoch, False),
            #("full_model_pseudolabels", run_file.fullmodel_epoch, True),
        ]

    if training_scheme_id == "training_scheme_debug_2":
        training_scheme = [
            ("encoder_classifier", run_file.classifier_epoch, False),
        ]
    logger.info(f"Setting up training scheme: {training_scheme_id} - {training_scheme}")
    return training_scheme