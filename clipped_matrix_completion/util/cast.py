import numpy as np

# def cast_all_to_rating(values, rating_options):
#     return [[cast_to_rating(val, rating_options) for val in values_row]
#             for values_row in values]

# def _cast_to_rating(val, rating_options):
#     return rating_options[(np.abs(rating_options - val)).argmin()]

# cast_to_rating = np.vectorize(_cast_to_rating, excluded=[1])


def cast_to_rating(vals, rating_options):
    """
    Assumption: rating_options are equally spaced
    """
    rating_options = np.sort(rating_options)
    interval = (rating_options[1] - rating_options[0])
    ratings = np.around(
        (vals - rating_options[0]) / interval) * interval + rating_options[0]
    ratings[ratings > rating_options[-1]] = rating_options[-1]
    ratings[ratings < rating_options[0]] = rating_options[0]
    return ratings


class RatingCaster():
    def __init__(self, rating_options):
        self.rating_options = rating_options

    def __call__(self, vals):
        return cast_to_rating(vals, self.rating_options)


def cast_to_natural_numbers(vals):
    """
    Assumption: rating_options are equally spaced
    """
    ratings = np.around(vals)
    ratings[ratings < 1] = 1
    return ratings


def cast_to_integer(vals):
    """
    Assumption: rating_options are equally spaced
    """
    ratings = np.around(vals)
    return ratings
