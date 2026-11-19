"""Measures one film or episode: the video and audio annotators, each writing a parquet table.

Nothing here knows where a corpus is kept. A Source says where one file is
and where its outputs go, config names the model directory, and run.annotate
does the work.
"""
