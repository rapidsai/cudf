# Copyright (c) 2018-2019, NVIDIA CORPORATION.

import nvstrings
from utils import assert_eq


def test_from_csv():
    tweets = nvstrings.from_csv("tests/data/tweets.csv", 7)
    got = tweets[:5]
    expected = [
        "@Bill_Porter nice to know that your site is back :-)",  # noqa E501
        "@sudhamshu after trying out various tools to take notes and I found that paper is the best to take notes and to maintain todo lists.",  # noqa E501
        "@neetashankar Yeah, I got the connection. I am getting 20 mbps for a 15 mbps connection. Customer service is also good.",  # noqa E501
        '@Bill_Porter All posts from your website http://t.co/NUWn5HUFsK seems to have been deleted. I am getting a ""Not Found"" page even in homepage',  # noqa E501
        'Today is ""bring your kids"" day at office and the entire office is taken over by cute little creatures ;)',  # noqa E501
    ]

    assert_eq(got, expected)


def test_free():
    # TODO: Check that GPU memory has been freed.
    data = nvstrings.to_device(["a", "b", "c", "d"])
    nvstrings.free(data)
