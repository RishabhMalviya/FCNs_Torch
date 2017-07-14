labelColorMap = torch.load('labelColorMap_directAccess.t7')

for i = 0,9 do

    trainData = torch.load('./trainvalData/trainvalDataTemp' .. i .. '.t7')

    print('Done loading slice: ' .. i)

    for i = 1,#trainData.labels do
        width = trainData.labels[i]:size(2)
        height = trainData.labels[i]:size(3)

        tempCopy = trainData.labels[i]
        tempCopy = tempCopy*10
        tempCopy = tempCopy + 1
        tempCopy = tempCopy:type('torch.IntTensor')

        trainData.labels[i] = torch.Tensor(1,500,500):fill(22):type('torch.ByteTensor')

        for x = 1,width do
            for y = 1,height do
                trainData.labels[i][1][x][y] = labelColorMap[tempCopy[1][x][y]][tempCopy[2][x][y]][tempCopy[3][x][y]]
            end
        end
    end

    print('Done parsing slice:  ' .. i)

    torch.save('./trainvalFinal/trainvalDataFinal' .. i .. '.t7', trainData)

    print('Done saving slice: ' .. i .. '\n')
end
