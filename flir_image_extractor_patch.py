# The flirimageextractor library is great, but it really wants to load the 
# thermal image metadata from the source file, while we have better data
# that comes from a different file. 
# Instead of doing file level shenanigans or subclassing, we monkey patch in a 
# different implementation of extract_thermal_image that handles the metadata how we want.
# If this gets any more complicated we're going to just use a subclass.
import io
import json
import subprocess
import types
from PIL import Image

import numpy as np
from flirimageextractor import FlirImageExtractor

def monkey_patch_flir(flir_obj, new_meta):
    flir_obj.extract_thermal_image = types.MethodType(extract_thermal_image, flir_obj)
    setattr(flir_obj, 'meta_override', new_meta)

    return flir_obj


def extract_thermal_image(self):
    """
    extracts the thermal image as 2D numpy array with temperatures in oC

    :return: Numpy Array of thermal values
    """

    # read image metadata needed for conversion of the raw sensor values
    # E=1,SD=1,RTemp=20,ATemp=RTemp,IRWTemp=RTemp,IRT=1,RH=50,PR1=21106.77,PB=1501,PF=1,PO=-7340,PR2=0.012545258
    if self.flir_img_filename:
        meta_json = subprocess.check_output(
            [
                self.exiftool_path,
                self.flir_img_filename,
                "-Emissivity",
                "-SubjectDistance",
                "-AtmosphericTemperature",
                "-ReflectedApparentTemperature",
                "-IRWindowTemperature",
                "-IRWindowTransmission",
                "-RelativeHumidity",
                "-PlanckR1",
                "-PlanckB",
                "-PlanckF",
                "-PlanckO",
                "-PlanckR2",
                "-j",
            ]
        )
    else:
        self.flir_img_bytes.seek(0)
        args = [
            "exiftool",
            "-Emissivity",
            "-SubjectDistance",
            "-AtmosphericTemperature",
            "-ReflectedApparentTemperature",
            "-IRWindowTemperature",
            "-IRWindowTransmission",
            "-RelativeHumidity",
            "-PlanckR1",
            "-PlanckB",
            "-PlanckF",
            "-PlanckO",
            "-PlanckR2",
            "-j",
            "-",
        ]
        p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        meta_json, err = p.communicate(input=self.flir_img_bytes.read())

    meta = json.loads(meta_json.decode())[0]
    ## START OF CHANGED CODE
    if hasattr(self, 'meta_override'):
        for key, val in self.meta_override.items():
            meta[key] = val
    ## END OF CHANGED CODE

    # use exiftool to extract the thermal images
    if self.flir_img_filename:
        thermal_img_bytes = subprocess.check_output(
            [self.exiftool_path, "-RawThermalImage", "-b", self.flir_img_filename]
        )
    else:
        self.flir_img_bytes.seek(0)
        args = ["exiftool", "-RawThermalImage", "-b", "-"]
        p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        thermal_img_bytes, err = p.communicate(input=self.flir_img_bytes.read())

    thermal_img_stream = io.BytesIO(thermal_img_bytes)
    thermal_img_stream.seek(0)

    thermal_img = Image.open(thermal_img_stream)
    thermal_np = np.array(thermal_img)

    # raw values -> temperature
    subject_distance = self.default_distance
    if "SubjectDistance" in meta:
        subject_distance = FlirImageExtractor.extract_float(meta["SubjectDistance"])

    if self.fix_endian:
        # fix endianness, the bytes in the embedded png are in the wrong order
        thermal_np = np.right_shift(thermal_np, 8) + np.left_shift(
            np.bitwise_and(thermal_np, 0x00FF), 8
        )

    # run the thermal data numpy array through the raw2temp conversion
    return FlirImageExtractor.raw2temp(
        thermal_np,
        E=meta["Emissivity"],
        OD=subject_distance,
        RTemp=FlirImageExtractor.extract_float(
            meta["ReflectedApparentTemperature"]
        ),
        ATemp=FlirImageExtractor.extract_float(meta["AtmosphericTemperature"]),
        IRWTemp=FlirImageExtractor.extract_float(meta["IRWindowTemperature"]),
        IRT=meta["IRWindowTransmission"],
        RH=FlirImageExtractor.extract_float(meta["RelativeHumidity"]),
        PR1=meta["PlanckR1"],
        PB=meta["PlanckB"],
        PF=meta["PlanckF"],
        PO=meta["PlanckO"],
        PR2=meta["PlanckR2"],
    )