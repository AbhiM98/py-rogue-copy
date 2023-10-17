"""GPS utilities for extracting GoPro gps data stream."""
import logging
import struct
from collections import namedtuple

import ffmpeg
import numpy

"""
The majority of this code is a modified version of the code found here:
https://github.com/alexis-mignon/pygpmf
"""

GPSData = namedtuple(
    "GPSData",
    [
        "description",
        "timestamp",  # yyyy-mm-dd HH:MM:SS.FFF
        "microseconds",  # microseconds since beginning of recording
        "samples_delivered",  # number of samples delivered since beginning of recording
        "precision",  # gps precision (under 500 is good)
        "fix",  # indicates if the go pro orientation is fixed
        "latitude",  # latitude [deg]
        "longitude",  # longitude [deg]
        "altitude",  # altitude (z) [m]
        "speed_2d",  # speed in 2d (x, y) [m/s]
        "speed_3d",  # speed in 3d (x, y, z) [m/s]
        "units",  # units of various quantities
        "npoints",  # number of gps points
    ],
)

num_types = {
    "d": ("float64", "d"),
    "f": ("float32", "f"),
    "b": ("int8", "b"),
    "B": ("uint8", "B"),
    "s": ("int16", "h"),
    "S": ("uint16", "H"),
    "l": ("int32", "i"),
    "L": ("uint32", "I"),
    "j": ("int64", "q"),
    "J": ("uint64", "Q"),
}

FIX_TYPE = {0: "none", 2: "2d", 3: "3d"}

KLVItem = namedtuple("KLVItem", ["key", "length", "value"])
KLVLength = namedtuple("KLVLength", ["type", "size", "repeat"])

logger = logging.getLogger(__name__)


# Basic GPS data parsing
def extract_gps_blocks(stream):
    """Extract GPS data blocks from binary stream.

    This is a generator on lists `KVLItem` objects. In
    the GPMF stream, GPS data comes into blocks of several
    different data items. For each of these blocks we return a list.

    Parameters:
    stream: bytes. The raw GPMF binary stream

    Returns:
    gps_items_generator: generator. Generator of lists of `KVLItem` objects
    """
    for s in filter_klv(stream, "STRM"):
        content = []
        is_gps = False
        for elt in s.value:
            content.append(elt)
            if elt.key == "GPS5":
                is_gps = True
        if is_gps:
            # print(content)
            yield content


def parse_gps_block(gps_block):
    """Turn GPS data blocks into `GPSData` objects.

    Parameters:
    gps_block: list of KVLItem. A list of KVLItem corresponding to a GPS data block.

    Returns:
    gps_data: GPSData. A GPSData object holding the GPS information of a block.
    """
    block_dict = {s.key: s for s in gps_block}

    try:
        gps_data = block_dict["GPS5"].value * 1.0 / block_dict["SCAL"].value
    except ZeroDivisionError:
        gps_data = block_dict["GPS5"].value
    except KeyError:
        raise RuntimeError(
            "Could not find GPS data in stream, missing key 'GPS5' or 'SCAL'"
        )
    except ValueError:
        print("problem with gps data, block is probably corrupt")
        print("GPS5:", block_dict["GPS5"].value)
        print("SCAL:", block_dict["SCAL"].value)
        print("continuing without scaling")
        gps_data = block_dict["GPS5"].value

    try:
        latitude, longitude, altitude, speed_2d, speed_3d = gps_data.T
    except ValueError:
        print("problem with gps data, block is probably corrupt")
        print("GPS5:", block_dict["GPS5"].value)
        print("SCAL:", block_dict["SCAL"].value)
        return None

    return GPSData(
        description=block_dict["STNM"].value,
        timestamp=block_dict["GPSU"].value,
        microseconds=block_dict["STMP"].value,
        samples_delivered=block_dict["TSMP"].value,
        precision=block_dict["GPSP"].value / 100.0,
        fix=block_dict["GPSF"].value,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        speed_2d=speed_2d,
        speed_3d=speed_3d,
        units=block_dict["UNIT"].value,
        npoints=len(gps_data),
    )


# GPS io management
try:

    def find_gpmf_stream(fname):
        """Find the reference to the GPMF Stream in the video file.

        Parameters:
        fname: str. The input file

        Returns:
        stream_info: dict. The GPMF Stream info.

        Raises:
        RuntimeError: If no stream found.
        """
        probe = ffmpeg.probe(fname)

        for s in probe["streams"]:
            if s["codec_tag_string"] == "gpmd":
                return s

        raise RuntimeError("Could not find GPS stream")

    def extract_gpmf_stream(fname, verbose=False):
        """Extract GPMF binary data from video files.

        Parameters:
        fname: str. The input file
        verbose: bool, optional (default=False). If True, display ffmpeg messages.

        Returns:
        gpmf_data: bytes. The raw GPMF binary stream.
        """
        stream_info = find_gpmf_stream(fname)
        stream_index = stream_info["index"]
        return (
            ffmpeg.input(fname)
            .output("pipe:", format="rawvideo", map="0:%i" % stream_index, codec="copy")
            .run(capture_stdout=True, capture_stderr=not verbose)[0]
        )

except ImportError:
    logger.info(
        "The 'ffmpeg' module could not be loaded. The function 'find_gpmf_stream' will not be available."
    )


# GPS and KLV parsing
def ceil4(x):
    """Find the closest greater or equal multiple of 4.

    Parameters:
    x: int.The size

    Returns:
    x_ceil: int. The closest greater integer which is a multiple of 4.
    """
    return (((x - 1) >> 2) + 1) << 2


def parse_payload(x, fourcc, type_str, size, repeat):
    """Parse the payload.

    Parameters:
    x: byte. The byte array corresponding to the payload.
    fourcc: str. The fourcc code.
    type_str: str. The type of the value.
    size: int. The size of the value.
    repeat: int. The number of times the value is repeated.

    Returns:
    payload: object. The parsed payload. the actual type depends on the type_str and the size and repeat values.
    """
    if type_str == "\x00":
        return iter_klv(x)
    else:
        x = x[: size * repeat]
        if type_str == "c":
            if fourcc == "UNIT":
                x = list(numpy.frombuffer(x, dtype="S%i" % size))
                return [s.decode("latin1") for s in x]
            else:
                return x.decode("latin1")

        elif type_str in num_types:
            dtype, stype = num_types[type_str]
            dtype = numpy.dtype(">" + stype)
            a = numpy.frombuffer(x, dtype=dtype)
            type_size = dtype.itemsize
            dim1 = size // type_size

            if a.size == 1:
                a = a[0]
            elif dim1 > 1 and repeat > 1:
                a = a.reshape(repeat, dim1)
            return a
        elif type_str == "U":
            x = x.decode()
            year = "20" + x[:2]
            month = x[2:4]
            day = x[4:6]
            hours = x[6:8]
            mins = x[8:10]
            seconds = x[10:]
            return "%s-%s-%s %s:%s:%s" % (year, month, day, hours, mins, seconds)
        else:
            return x


def iter_klv(x):
    """Iterate on KLV items.

    Parameters:
    x: byte. The byte array corresponding to the stream.

    Returns:
    klv_gen: generator. A generator of (fourcc, (type_str, size, repeat), payload) tuples.
    """
    start = 0

    while start < len(x):
        head = struct.unpack(">cccccBH", x[start : start + 8])
        fourcc = (b"".join(head[:4])).decode()
        type_str, size, repeat = head[4:]
        type_str = type_str.decode()
        start += 8
        payload_size = ceil4(size * repeat)
        payload = parse_payload(
            x[start : start + payload_size], fourcc, type_str, size, repeat
        )
        start += payload_size

        yield KLVItem(fourcc, KLVLength(type_str, size, repeat), payload)


def filter_klv(x, filter_fourcc):
    """Filter only KLV items with chosen fourcc code.

    Parameters:
    x: byte. The input stream.
    filter_fourcc: list of str. A list of FourCC codes.

    Returns:
    klv_gen: generator. De-nested generator of (fourcc, (type_str, size, repeat), payload) with only chosen fourcc
    """
    generators = [iter(iter_klv(x))]

    while len(generators) > 0:
        it = generators[-1]
        try:
            (fourcc, (type_str, size, repeat), payload) = next(it)
            if fourcc in filter_fourcc:
                yield KLVItem(fourcc, KLVLength(type_str, size, repeat), payload)
            if type_str == "\x00":
                generators.append(iter(payload))
        except StopIteration:
            generators = generators[:-1]
