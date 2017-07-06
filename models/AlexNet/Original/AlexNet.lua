require 'nn'
require 'cudnn'

local net = nn.Sequential()

-- Keeping track of output dimensions
local outW = 500
local outH = 500

-- Keeping track of convolution parameters

net:add(cudnn.SpatialConvolution(3, 96, 11, 11, 4, 4, 0, 0, 1))
net:add(cudnn.ReLU(true))
net:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 0, 0):ceil())
net:add(cudnn.SpatialCrossMapLRN(5, 0.000100, 0.7500, 1.000000))
net:add(cudnn.SpatialConvolution(96, 256, 5, 5, 1, 1, 2, 2, 2))
net:add(cudnn.ReLU(true))
net:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 0, 0):ceil())
net:add(cudnn.SpatialCrossMapLRN(5, 0.000100, 0.7500, 1.000000))
net:add(cudnn.SpatialConvolution(256, 384, 3, 3, 1, 1, 1, 1, 1))
net:add(cudnn.ReLU(true))
net:add(cudnn.SpatialConvolution(384, 384, 3, 3, 1, 1, 1, 1, 2))
net:add(cudnn.ReLU(true))
net:add(cudnn.SpatialConvolution(384, 256, 3, 3, 1, 1, 1, 1, 2))
net:add(cudnn.ReLU(true))
net:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 0, 0):ceil())
net:add(nn.View(-1):setNumInputDims(3))
net:add(nn.Linear(9216, 4096))
net:add(cudnn.ReLU(true))
net:add(nn.Dropout(0.500000))
net:add(nn.Linear(4096, 4096))
net:add(cudnn.ReLU(true))
net:add(nn.Dropout(0.500000))
net:add(nn.Linear(4096, 1000))
net:add(cudnn.LogSoftMax())

return net