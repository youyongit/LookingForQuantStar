import sys
import os
import glob
import imp
import inspect

ALL_FACTORS = []

def append_path(Paths):
    for path in Paths:
        fileName  = path.split("/")[-1]
        modelName = fileName.split(".")[0]
        ALL_FACTORS.append(modelName)
        imp.load_source('Factors',path)

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
		
alphaPath = base_path + '/factors/bar_alpha_factors/'
alphaPath = glob.glob(alphaPath+r'*.py')
append_path(alphaPath)

CDLPath = base_path + '/factors/bar_CDL_factors/'
CDLPath = glob.glob(CDLPath + r'*.py')
append_path(CDLPath)

momPath = base_path + '/factors/bar_mom_factors/'
momPath = glob.glob(momPath + r'*.py')
append_path(momPath)

oscPath = base_path + '/factors/bar_osc_factors/'
oscPath = glob.glob(oscPath + r'*.py')
append_path(oscPath)

volPath = base_path + '/factors/bar_vol_factors/'
volPath = glob.glob(volPath + r'*.py')
append_path(volPath)

volumePath = base_path + '/factors/bar_volume_factors/'
volumePath = glob.glob(volumePath + r'*.py')
append_path(volumePath)

testPath = base_path + '/factors/bar_test_factors/'
testPath = glob.glob(testPath + r'*.py')

# labelPath = os.getcwd() + '/strategy/mlutils/factors/bar_label_factors/'
# labelPath = glob.glob(labelPath + r'*.py')
# append_path(labelPath)


print('[Factors Load]: Load {} Factors\' Functions'.format(len(ALL_FACTORS)))

if __name__ == "__main__":

    pass