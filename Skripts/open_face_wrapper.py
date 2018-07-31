"""Wrapper to execute the openFace commands with options and stores the output.
The wrapper is based on the commandline usage: https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments"""

import os
import subprocess as sp
# import environment_laptop as paths
# import environment_techfak as paths
import environment as paths


class OpenFace():
    """Wrapper for OpenFace"""
    def __init__(self, path):
        # self.path_open_face = paths.OPEN_FACE_PATH
        self.path_open_face = path
        os.chdir(self.path_open_face)

    def FeatureExtraction(self, args):
        """ Executes FeatureExtraction.
            args: dict of options and values
                options and values should both be strings.
                They will be concatinated to one command string and then executed."""
        command = "./FeatureExtraction "
        for op, value in args.items():
            command += op + " " + value + " "
        sp.run(command, shell=True)

    def FaceLandmarkVidMulti(self, args):
        """ Executes FeatureExtraction.
            args: dict of options and values
                options and values should both be strings.
                They will be concatinated to one command string and then executed."""
        command = "./FaceLandmarkVidMulti "
        for op, value in args.items():
            command += op + " " + value + " "
        sp.run(command, shell=True)

    def wrapper(self, command, args):
        """ Executes Any given command with args in the OpenFace Directory.
            args: dict of options and values
                options and values should both be strings.
                They will be concatinated to one command string and then executed."""
        command += " "
        for op, value in args.items():
            command += op + " " + value + " "
        sp.run(command, shell=True)
