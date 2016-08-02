from xml.etree import ElementTree
import struct
import sys
import re

try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:
    import Tkinter as tk
    import tkFileDialog as filedialog


def readFmt(fd, fmt):
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, fd.read(size))[0]

def readHeader(fd):

    FILE_MAGIC_STRING = b'OMAS_BF\n'
    MAGIC_NUMBER = 0xFFFF

    fmt = ["H", "I", "Q"]

    res = []
    res.append(fd.read(len(FILE_MAGIC_STRING)))

    for f in fmt:
        res.append(readFmt(fd, f))

    # skip 4 bytes
    fd.read(4);

    # read metadata offset
    if (res[2] > 1):
        res.append(readFmt(fd, "Q"))

    # skip 5 bytes
    fd.read(5)

    # desclen
    desclen = readFmt(fd, "H")
    desc = fd.read(desclen)

    res.append(desclen)
    res.append(desc)

    if (res[0] == FILE_MAGIC_STRING and res[1] == MAGIC_NUMBER):
        return res
    else:
        return None


def readStack(fd, pos, version):
    STACK_MAGIC_STRING = b"OMAS_BF_STACK\n"
    MAGIC_NUMBER = 0xFFFF
    STACK_VERSION = 5
    MAXIMAL_NUMBER_OF_DIMENSIONS = 15

    fd.seek(pos)

    fmt = ["H", "i"]

    res = []
    res.append(fd.read(len(STACK_MAGIC_STRING)))



    for f in fmt:
        res.append(readFmt(fd, f))

#    print(res)

    if not (res[0] == STACK_MAGIC_STRING and res[1] == MAGIC_NUMBER and res[2] <= STACK_VERSION):
        print("ERROR")
        return None

    numberOfDimensions = readFmt(fd, "i")
#    print("numberofdims: " + str(numberOfDimensions))

    dims = []
    for i in range(MAXIMAL_NUMBER_OF_DIMENSIONS):
        d = readFmt(fd, "i")
        dims.append(d if i < numberOfDimensions else 1)
#    print dims

    lengths = []
    for i in range(MAXIMAL_NUMBER_OF_DIMENSIONS):
        d = readFmt(fd, "d")
        lengths.append(d if i < numberOfDimensions else 0.0)
#    print lengths

    offsets = []
    for i in range(MAXIMAL_NUMBER_OF_DIMENSIONS):
        d = readFmt(fd, "d")
        offsets.append(d if i < numberOfDimensions else 0.0)
#    print offsets

    pType = readFmt(fd, "i")
#    print(pType)

    compr = readFmt(fd, "i")
#    print(compr)

    fd.read(4)

    lengthOfName = readFmt(fd, "i")
    lengthOfDescription = readFmt(fd, "i")
    #print("namelen;" + str(lengthOfName))
#    print("desclen;" + str(lengthOfDescription))


    fd.read(8)

    lengthOfData = readFmt(fd, "q")
    #print("datalen;" + str(lengthOfData))
    nextPos =  readFmt(fd, "q")
    #print("next;" + str(nextPos))

    name = fd.read(lengthOfName)
#    print(name)

# TODO this doesn't seem to do anything
    description = fd.read(lengthOfDescription)
#    print(description)

    fd.seek(lengthOfData, 1)
    footerStart = fd.tell()

    footerSize = readFmt(fd, "I")
    fd.read(4*MAXIMAL_NUMBER_OF_DIMENSIONS)
    fd.read(4*MAXIMAL_NUMBER_OF_DIMENSIONS)
    firstMetaLen = readFmt(fd, "I")

    #print(firstMetaLen)

    fd.seek(footerStart)
    fd.seek(footerSize, 1)

    for i in range(numberOfDimensions):
        nameLen = readFmt(fd, "I")
        fd.seek(nameLen,1)

    # skip first xml
    fd.seek(firstMetaLen, 1)


    ## seek until we find '<root>'
    accum = b''
    while not accum.endswith(b'<root'):
        accum += fd.read(1)

    fd.seek(-9, 1)

    sndMetaLen = readFmt(fd, "I")
    #print(sndMetaLen)
    StackMeta = fd.read(sndMetaLen)

    #print(StackMeta)

    return((name, nextPos, StackMeta))


def get_parameters_from_xml(xml, keys):
    res = []
    myET = ElementTree.fromstring(xml)

    for k in keys:
        res.append(str(myET.find(k).text))

    return res

def parse_msr(path):
    fd = open(path, "rb")
    hd = readHeader(fd)
    # print hd
    res = []
    offset = hd[3]

    while offset != 0:
        resI = readStack(fd, offset, hd[2])
        res.append(resI)
        offset = resI[1]

    return (hd, res)


def printParameters(params):
    print(params[0])
    myET = ElementTree.fromstring(params[2])

    print ("dwelltime:\t" + str(myET.find(".doc/ExpControl/scan/dwelltime").text))
    print ("line accu:\t" + str(myET.find(".doc/ExpControl/scan/range/line_accu").text))
    print ("STED power:\t" + str(myET.find(".doc/STED775/power").text))
    print ("3D power:\t" + str(myET.find(".doc/ExpControl/three_d/modules/item/mrs/position/value/calibrated").text))
    print ("pinhole size:\t" + str(myET.find(".doc/Pinhole/pinhole_size").text))

    chans = str(myET.find(".doc/ExpControl/gating/linesteps/chans_enabled").text)
    chans = chans.split(" ")
    chansIdx = []
    for i in range(len(chans)):
        if chans[i] == b"1":
            chansIdx.append(i)
    print(chansIdx)

    laser_ena = str(myET.find(".doc/ExpControl/gating/linesteps/laser_enabled").text)
    laser_ena = laser_ena.split(" ")
    laserEnaIdx = []
    for i in range(len(laser_ena)):
        if laser_ena[i] == b"1":
            laserEnaIdx.append(i)
    print(laserEnaIdx)

    chans_on = str(myET.find(".doc/ExpControl/gating/linesteps/chans_on").text)
    laser_on = str(myET.find(".doc/ExpControl/gating/linesteps/laser_on").text)

    # print detector for all enabled channels
    chanDets = myET.findall(".doc/ExpControl/scan/detsel/detsel/item")
    for i in chansIdx:
        print ( "channel " + str(i+1) + " detector:\t" + str(chanDets[i].text))

    # print power for all enabled lasers
    laserPwrs = myET.findall(".doc/ExpControl/lasers/power_calibrated/item/value/calibrated")
    for i in laserEnaIdx:
        print ( "laser " + str(i+1) + " power:\t" + str(laserPwrs[i].text))



def main():

    #askopenfilenames()
    #root = tk.Tk()
    #root.withdraw()
    #file_path = filedialog.askopenfilename()
    file_path = '/Users/david/Desktop/AutomatedAcquisitions/GM_81C_150s/overviews/3ce7893ba6275ab5988c1395aec5251e_field1.msr'
    fname = file_path

    fd = open(fname, "rb")
    hd = readHeader(fd)
    #print hd
    res = []
    offset = hd[3]

    while offset != 0:
        resI = readStack(fd, offset, hd[2])
        res.append(resI)
        offset = resI[1]

    print (len(res))

    print(res[0][2])

    for r in res:
        printParameters(r)
        print("")


    #print (str(myET.find(".doc/ExpControl/scan/dwelltime").text))
    #print (str(myET.find(".doc/ExpControl/scan/range/line_accu").text))

    #for e in myET:
    #    print (e.tag, e.attrib)



if __name__ == '__main__':
    main()
