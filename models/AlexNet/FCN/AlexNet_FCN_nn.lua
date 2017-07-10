require 'nn'
local helper =  dofile('../helperFunctions.lua')

local net = nn.Sequential()


net:add(nn.SpatialConvolution(3, 96, 11, 11, 4, 4, 100, 100))
net:add(nn.ReLU(true))

net:add(nn.SpatialMaxPooling(3, 3, 2, 2, 0, 0):ceil())
net:add(nn.SpatialCrossMapLRN(5, 0.000100, 0.7500, 1.000000))

net:add(helper.groupedConvolution(96, 256, 86, 86, 5, 5, 1, 1, 2, 2, 2))
net:add(nn.ReLU(true))

net:add(nn.SpatialMaxPooling(4, 4, 2, 2, 0, 0):ceil())
net:add(nn.SpatialCrossMapLRN(5, 0.000100, 0.7500, 1.000000))

net:add(nn.SpatialConvolution(256, 384, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU(true))

net:add(helper.groupedConvolution(384, 384, 42, 42, 3, 3, 1, 1, 1, 1, 2))
net:add(nn.ReLU(true))

net:add(helper.groupedConvolution(384, 256, 42, 42, 3, 3, 1, 1, 1, 1, 2))
net:add(nn.ReLU(true))

net:add(nn.SpatialMaxPooling(3, 3, 2, 2, 0, 0):ceil())

net:add(nn.SpatialConvolution(256, 4096, 6, 6, 1, 1, 0, 0))
net:add(nn.ReLU(true))
net:add(nn.Dropout(0.500000))

net:add(nn.SpatialConvolution(4096, 4096, 1, 1, 1, 1, 0, 0))
net:add(nn.ReLU(true))
net:add(nn.Dropout(0.500000))

scoreLayer = nn.SpatialConvolution(4096, 21, 1, 1, 1, 1, 0, 0)
scoreLayer.weight:fill(0)
scoreLayer.bias:fill(0)
net:add(scoreLayer)

net:add(nn.SpatialFullConvolution(21, 21, 64, 64, 32, 32, 22, 22))

net:add(nn.SpatialLogSoftMax())


return net
