from enum import Enum


class Sentinel2Bands(Enum):
    B01 = "B1"
    B02 = "B2"
    B03 = "B3"
    B04 = "B4"
    B05 = "B5"
    B06 = "B6"
    B07 = "B7"
    B08 = "B8"
    B8A = "B8A"
    B09 = "B9"
    B11 = "B11"
    B12 = "B12"

    @classmethod
    def get_band_names(cls):
        return [band.value for band in cls]


class ICESAT2Properties(Enum):
    Time = "Time (sec)"
    DeltaTime = "Delta Time (sec)"
    SegmentID = "Segment ID"
    GTNum = "GT Num"
    BeamNum = "Beam Num"
    BeamType = "Beam Type"
    Latitude = "Latitude (deg)"
    Longitude = "Longitude (deg)"
    UTM_Easting = "UTM Easting (m)"
    UTM_Northing = "UTM Northing (m)"
    UTM_Zone = "UTM Zone"
    UTM_Hemisphere = "UTM Hemisphere"
    CrossTrack = "Cross-Track (m)"
    AlongTrack = "Along-Track (m)"
    Height_HAE = "Height (m HAE)"
    Height_MSL = "Height (m MSL)"
    Classification = "Classification"
    SignalConfidence = "Signal Confidence"
    SolarElevation = "Solar Elevation (deg)"

    @classmethod
    def get_property_names(cls):
        return [prop.value for prop in cls]
