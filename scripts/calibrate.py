"""Standalone PTQ calibration and conversion entry point."""
from common import load_config
from ptq import run
if __name__=='__main__': run(load_config())
