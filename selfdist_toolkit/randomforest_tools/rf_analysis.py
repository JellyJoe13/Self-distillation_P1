import typing


def convert_acc_dict(
        acc_dict: typing.Dict[str, typing.List[float]]
) -> typing.Dict[str, float]:
    """
    Function that computes the mean accuracy score of the measured attempts.

    Parameters
    ----------
    acc_dict : typing.Dict[str, typing.List[float]]
        Dictionary containing lists of measured accuracy scores

    Returns
    -------
    typing.Dict[str, float]
        Dictionary containing the mean accuracy scores computed with this function
    """

    # create placeholder dict
    output = {}

    # iterate over old dict
    for key in acc_dict:
        # calculate average
        output[key] = sum(acc_dict[key]) / len(acc_dict[key])

    # return the dict
    return output


def compare_accuracy_dict(
        normal: typing.Dict[str, float],
        sd: typing.Dict[str, float]
) -> typing.Dict[str, float]:
    """
    Function that computes the difference in accuracy between the two input dicts.

    Parameters
    ----------
    normal : typing.Dict[str, float]
        Dictionary containing the accuracy scores of the normal test
    sd : typing.Dict[str, float]
        Dictionary containing the accuracy scores of the self distillation test

    Returns
    -------
    dict_diff : typing.Dict[str, float]
        Dictionary containing the difference in accuracy scores between the input dictionaries
    """

    # generate comparison dict
    comparison_dict = {}

    # go over individual dicts
    for key in normal:
        comparison_dict[key] = (sd[key] - normal[key])

    return comparison_dict
