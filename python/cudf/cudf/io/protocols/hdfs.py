import urllib
from io import BytesIO

from pyarrow import hdfs


def get_filepath_or_buffer(path_or_data, compression=None):
    parsed_url = urllib.parse.urlsplit(path_or_data)

    host = parsed_url.hostname if parsed_url.hostname else "default"
    port = parsed_url.port if parsed_url.port else 0
    username = parsed_url.username

    fs = hdfs.connect(host=host, port=port, user=username)
    with fs.open(parsed_url.path, mode="rb") as f:
        filepath_or_buffer = BytesIO(f.read())

    return filepath_or_buffer, compression
