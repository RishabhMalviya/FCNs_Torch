require 'image'


labelColorMap = torch.load('labelColorMap.t7')

for i = 0,9 do
    trainData = {data = {}, labels = {}}

    -- Load training images
        local imageSet = io.open("./VOCdevkit/VOC2012/ImageSets/Segmentation/codeTesting.txt")
        imageNumber = 1
        if imageSet then
            for line in imageSet:lines() do
                imageFile = './VOCdevkit/VOC2012/JPEGImages/' .. line .. '.jpg'
                trainData.data[imageNumber] = image.load(imageFile)
                imageNumber = imageNumber + 1
            end
        end
        print('Done loading input images from slice ' .. i)
        -- Pad each so that they're all 500x500
        finalData = torch.Tensor(#trainData.data,3,500,500):fill(0)
        for i = 1,#trainData.data do
            width = trainData.data[i]:size(2)
            height = trainData.data[i]:size(3)

            finalData[i][{ {}, {1,width}, {1,height} }] = trainData.data[i]
            trainData.data[i] = nil
        end
        collectgarbage()
        trainData.data = finalData
        print('Done padding input images from slice ' .. i)


    -- Load label images
        local imageSet = io.open("./VOCdevkit/VOC2012/ImageSets/Segmentation/codeTesting.txt")
        imageNumber = 1
        if imageSet then
            for line in imageSet:lines() do
                imageFile = './VOCdevkit/VOC2012/SegmentationClass/' .. line .. '.png'
                trainData.labels[imageNumber] = image.load(imageFile)
                imageNumber = imageNumber + 1
            end
        end
        print('Done loading label images from slice ' .. i)
        -- Convert label images to GT tensors and pad
        finalLabels = torch.Tensor(#trainData.labels,1,500,500):fill(22):type('torch.ByteTensor') -- 22 is ignoreIndex
        for i = 1,#trainData.labels do
            width = trainData.labels[i]:size(2)
            height = trainData.labels[i]:size(3)

            trainData.labels[i] = trainData.labels[i]*10
            trainData.labels[i] = trainData.labels[i] + 1
            trainData.labels[i] = trainData.labels[i]:type('torch.IntTensor')

            for x = 1,width do
                for y = 1,height do
                    finalLabels[i][1][x][y] = labelColorMap[trainData.labels[i][1][x][y]][trainData.labels[i][2][x][y]][trainData.labels[i][3][x][y]]
                end
            end
            trainData.labels[i] = nil
        end
        collectgarbage()
        trainData.labels = finalLabels
        print('Done converting and padding label images from slice ' .. i .. '\n')

    
    -- Save trainData tensors
        torch.save('./trainvalData/trainvalData' .. i .. '.t7',trainData)
        print('Done saving data from slice ' .. i .. '\n')
end


