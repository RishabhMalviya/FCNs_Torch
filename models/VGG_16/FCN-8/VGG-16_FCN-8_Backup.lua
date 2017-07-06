require 'nn'
local model = {}
-- warning: module 'data [type Python]' not found
-- warning: module 'data_data_0_split [type Split]' not found
table.insert(model, {'conv1_1', nn.SpatialConvolution(3, 64, 3, 3, 1, 1, 100, 100)})
table.insert(model, {'relu1_1', nn.ReLU(true)})
table.insert(model, {'conv1_2', nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu1_2', nn.ReLU(true)})
table.insert(model, {'pool1', nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()})
table.insert(model, {'conv2_1', nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu2_1', nn.ReLU(true)})
table.insert(model, {'conv2_2', nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu2_2', nn.ReLU(true)})
table.insert(model, {'pool2', nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()})
table.insert(model, {'conv3_1', nn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu3_1', nn.ReLU(true)})
table.insert(model, {'conv3_2', nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu3_2', nn.ReLU(true)})
table.insert(model, {'conv3_3', nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu3_3', nn.ReLU(true)})
table.insert(model, {'pool3', nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()})
-- warning: module 'pool3_pool3_0_split [type Split]' not found
table.insert(model, {'conv4_1', nn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu4_1', nn.ReLU(true)})
table.insert(model, {'conv4_2', nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu4_2', nn.ReLU(true)})
table.insert(model, {'conv4_3', nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu4_3', nn.ReLU(true)})
table.insert(model, {'pool4', nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()})
-- warning: module 'pool4_pool4_0_split [type Split]' not found
table.insert(model, {'conv5_1', nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu5_1', nn.ReLU(true)})
table.insert(model, {'conv5_2', nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu5_2', nn.ReLU(true)})
table.insert(model, {'conv5_3', nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu5_3', nn.ReLU(true)})
table.insert(model, {'pool5', nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil()})
table.insert(model, {'fc6', nn.SpatialConvolution(512, 4096, 7, 7, 1, 1, 0, 0)})
table.insert(model, {'relu6', nn.ReLU(true)})
table.insert(model, {'drop6', nn.Dropout(0.500000)})
table.insert(model, {'fc7', nn.SpatialConvolution(4096, 4096, 1, 1, 1, 1, 0, 0)})
table.insert(model, {'relu7', nn.ReLU(true)})
table.insert(model, {'drop7', nn.Dropout(0.500000)})
table.insert(model, {'score_fr', nn.SpatialConvolution(4096, 21, 1, 1, 1, 1, 0, 0)})
-- warning: module 'upscore2 [type Deconvolution]' not found
-- warning: module 'upscore2_upscore2_0_split [type Split]' not found
table.insert(model, {'score_pool4', nn.SpatialConvolution(512, 21, 1, 1, 1, 1, 0, 0)})
-- warning: module 'score_pool4c [type Crop]' not found
-- warning: module 'fuse_pool4 [type Eltwise]' not found
-- warning: module 'upscore_pool4 [type Deconvolution]' not found
-- warning: module 'upscore_pool4_upscore_pool4_0_split [type Split]' not found
table.insert(model, {'score_pool3', nn.SpatialConvolution(256, 21, 1, 1, 1, 1, 0, 0)})
-- warning: module 'score_pool3c [type Crop]' not found
-- warning: module 'fuse_pool3 [type Eltwise]' not found
-- warning: module 'upscore8 [type Deconvolution]' not found
-- warning: module 'score [type Crop]' not found
return model