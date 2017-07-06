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

net:add(nn.SpatialConvolution(3, 64, 3, 3, 1, 1, 100, 100))
net:add(nn.ReLU(true))

net:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU(true))

net:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())

net:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU(true))

net:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU(true))

net:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())

net:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU(true))

net:add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU(true))

net:add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU(true))

net:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())

net:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU(true))

net:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU(true))

net:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU(true))

net:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())

net:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU(true))

net:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU(true))

net:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU(true))

net:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())

net:add(nn.SpatialConvolution(512, 4096, 7, 7, 1, 1, 0, 0))
net:add(nn.ReLU(true))
net:add(nn.Dropout(0.500000))

net:add(nn.SpatialConvolution(4096, 4096, 1, 1, 1, 1, 0, 0))
net:add(nn.ReLU(true))
net:add(nn.Dropout(0.500000))

net:add(nn.SpatialConvolution(4096, 21, 1, 1, 1, 1, 0, 0))

net:add(nn.SpatialConvolution(512, 21, 1, 1, 1, 1, 0, 0))

net:add(nn.SpatialConvolution(256, 21, 1, 1, 1, 1, 0, 0))
