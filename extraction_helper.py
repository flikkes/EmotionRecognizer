import sys
from WavFileHelper import WavFileHelper

helper = WavFileHelper()
helper.save_mfcc(sys.argv[1], sys.argv[2])