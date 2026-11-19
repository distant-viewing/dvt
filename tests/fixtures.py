"""ffprobe reports used by more than one test, written as ffprobe writes them."""

FILM: dict[str, object] = {
    "streams": [
        {
            "codec_type": "video", "codec_name": "h264", "width": 1440, "height": 1080,
            "sample_aspect_ratio": "4:3", "display_aspect_ratio": "16:9",
            "field_order": "progressive", "avg_frame_rate": "24000/1001",
            "r_frame_rate": "24000/1001", "bit_rate": "4500000", "duration": "5400.250000",
            "nb_frames": "129476",
        },
        {
            "codec_type": "audio", "codec_name": "ac3", "channels": 2,
            "channel_layout": "stereo", "sample_rate": "48000", "bit_rate": "192000",
            "start_time": "0.032000", "duration": "5400.100000", "tags": {"language": "fre"},
        },
        {
            "codec_type": "audio", "codec_name": "dts", "channels": 6,
            "channel_layout": "5.1(side)", "sample_rate": "48000", "bit_rate": "1536000",
            "profile": "DTS", "start_time": "0.000000", "tags": {"language": "fra"},
        },
        {
            "codec_type": "audio", "codec_name": "aac", "channels": 2, "bit_rate": "128000",
            "tags": {"language": "eng"},
        },
        {
            "codec_type": "subtitle", "codec_name": "subrip", "nb_frames": "1200",
            "disposition": {"default": 1, "forced": 0}, "tags": {"language": "eng"},
        },
        {"codec_type": "subtitle", "codec_name": "hdmv_pgs_subtitle",
         "disposition": {"forced": 1}},
    ],
    "format": {"duration": "5400.300000", "bit_rate": "6000000"},
    "chapters": [
        {"start_time": "0.000000"},
        {"start": 90090000, "time_base": "1/1000000"},
        {"start": 5, "time_base": "0/1"},
        {"start_time": "N/A"},
    ],
}  # fmt: skip

SILENT: dict[str, object] = {
    "streams": [
        {
            "codec_type": "video", "codec_name": "mpeg2video", "width": 720, "height": 480,
            "sample_aspect_ratio": "0:1", "display_aspect_ratio": "0:1",
            "avg_frame_rate": "0/0", "r_frame_rate": "30000/1001", "nb_frames": "300",
        },
    ],
    "format": {"duration": "10.010000", "bit_rate": "8000000"},
}  # fmt: skip

EMPTY: dict[str, object] = {}
