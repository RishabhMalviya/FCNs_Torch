require 'nn'
local helper =  dofile('../helperFunctions.lua')

local net = nn.Sequential()

-- Keeping track of output dimensions
local W = 500
local H = 500
local inC = 3
local outC

-- Keeping track of convolution parameters
local kw --kernel width
local kh --kernel height
local dw --stride width
local dh --stride height
local pw --padding width
local ph --padding height
local group --grouping


kw = 11
kh = 11
dw = 4
dh = 4
pw = 100
ph = 100
group = 1
outC = 96
net:add(nn.SpatialConvolution(inC, outC, kw, kh, dw, dh, pw, ph))
net:add(nn.ReLU(true))
W, H = helper.computeOutputDims(W, H, kw, kh, dw, dh, pw, ph)
inC = outC
print('outW: ' .. W)
print('outH: ' .. H)
print('outC: ' .. outC .. '\n')

kw = 3 
kh = 3
dw = 2
dh = 2
pw = 0
ph = 0
group = 1
outC = 96
net:add(nn.SpatialMaxPooling(kw, kh, dw, dh, pw, ph):ceil())
net:add(nn.SpatialCrossMapLRN(5, 0.000100, 0.7500, 1.000000))
W, H = helper.computeOutputDims(W, H, kw, kh, dw, dh, pw, ph)
inC = outC
print('outW: ' .. W)
print('outH: ' .. H)
print('outC: ' .. outC .. '\n')

kw = 5
kh = 5
dw = 1
dh = 1
pw = 2
ph = 2
group = 2
outC = 256
net:add(helper.groupedConvolution(inC, outC, W, H, kw, kh, dw, dh, pw, ph, group))
net:add(nn.ReLU(true))
W, H = helper.computeOutputDims(W, H, kw, kh, dw, dh, pw, ph)
inC = outC
print('outW: ' .. W)
print('outH: ' .. H)
print('outC: ' .. outC .. '\n')

kw = 3 
kh = 3
dw = 2
dh = 2
pw = 0
ph = 0
group = 1
outC = 256
net:add(nn.SpatialMaxPooling(kw, kh, dw, dh, pw, ph):ceil())
net:add(nn.SpatialCrossMapLRN(5, 0.000100, 0.7500, 1.000000))
W, H = helper.computeOutputDims(W, H, kw, kh, dw, dh, pw, ph)
inC = outC
print('outW: ' .. W)
print('outH: ' .. H)
print('outC: ' .. outC .. '\n')

kw = 3 
kh = 3
dw = 1
dh = 1
pw = 1
ph = 1
group = 1
outC = 384
net:add(nn.SpatialConvolution(inC, outC, kw, kh, dw, dh, pw, ph))
net:add(nn.ReLU(true))
W, H = helper.computeOutputDims(W, H, kw, kh, dw, dh, pw, ph)
inC = outC
print('outW: ' .. W)
print('outH: ' .. H)
print('outC: ' .. outC .. '\n')

kw = 3 
kh = 3
dw = 1
dh = 1
pw = 1
ph = 1
group = 2
outC = 384
net:add(helper.groupedConvolution(inC, outC, W, H, kw, kh, dw, dh, pw, ph, group))
net:add(nn.ReLU(true))
W, H = helper.computeOutputDims(W, H, kw, kh, dw, dh, pw, ph)
inC = outC
print('outW: ' .. W)
print('outH: ' .. H)
print('outC: ' .. outC .. '\n')

kw = 3 
kh = 3
dw = 1
dh = 1
pw = 1
ph = 1
group = 2
outC = 256
net:add(helper.groupedConvolution(inC, outC, W, H, kw, kh, dw, dh, pw, ph, group))
net:add(nn.ReLU(true))
W, H = helper.computeOutputDims(W, H, kw, kh, dw, dh, pw, ph)
inC = outC
print('outW: ' .. W)
print('outH: ' .. H)
print('outC: ' .. outC .. '\n')

kw = 3 
kh = 3
dw = 2
dh = 2
pw = 0
ph = 0
group = 1
outC = 256
net:add(nn.SpatialMaxPooling(kw, kh, dw, dh, pw, ph):ceil())
W, H = helper.computeOutputDims(W, H, kw, kh, dw, dh, pw, ph)
inC = outC
print('outW: ' .. W)
print('outH: ' .. H)
print('outC: ' .. outC .. '\n')

kw = 6
kh = 6
dw = 1
dh = 1
pw = 0
ph = 0
group = 1
outC = 4096
net:add(nn.SpatialConvolution(inC, outC, kw, kh, dw, dh, pw, ph))
net:add(nn.ReLU(true))
net:add(nn.Dropout(0.500000))
W, H = helper.computeOutputDims(W, H, kw, kh, dw, dh, pw, ph)
inC = outC
print('outW: ' .. W)
print('outH: ' .. H)
print('outC: ' .. outC .. '\n')

kw = 1
kh = 1
dw = 1
dh = 1
pw = 0
ph = 0
group = 1
outC = 4096
net:add(nn.SpatialConvolution(inC, outC, kw, kh, dw, dh, pw, ph))
net:add(nn.ReLU(true))
net:add(nn.Dropout(0.500000))
W, H = helper.computeOutputDims(W, H, kw, kh, dw, dh, pw, ph)
inC = outC
print('outW: ' .. W)
print('outH: ' .. H)
print('outC: ' .. outC .. '\n')

kw = 1
kh = 1
dw = 1
dh = 1
pw = 0
ph = 0
group = 1
outC = 21
net:add(nn.SpatialConvolution(inC, outC, kw, kh, dw, dh, pw, ph))
W, H = helper.computeOutputDims(W, H, kw, kh, dw, dh, pw, ph)
inC = outC
print('outW: ' .. W)
print('outH: ' .. H)
print('outC: ' .. outC .. '\n')