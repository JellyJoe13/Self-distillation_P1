import typing


def convert_acc_dict(
        acc_dict: typing.Dict[str, typing.List[float]]
) -> typing.Dict[str, float]:
    # create placeholder dict
    output = {}

    # iterate over old dict
    for key in acc_dict:
        # calculate average
        output[key] = sum(acc_dict[key]) / len(acc_dict[key])

    # return the dict
    return output


def compare_accuracy_dict(normal, sd):
    # generate comparison dict
    comparison_dict = {}

    # go over individual dicts
    for key in normal:
        comparison_dict[key] = (sd[key] - normal[key])

    return comparison_dict
