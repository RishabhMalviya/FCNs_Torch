for i = 0,9 do

    trainData = torch.load('./trainvalParsed/trainvalDataParsed' .. i .. '.t7')

    print('Done loading slice: ' .. i)

    for i = 1,#trainData.labels do
        width = trainData.labels[i]:size(2)
        height = trainData.labels[i]:size(3)

        tempCopy = trainData.labels[i]
        trainData.labels[i] = torch.Tensor(1,500,500):fill(22):type('torch.ByteTensor')

        trainData.labels[i][{ {}, {1,width}, {1,height} }] = tempCopy
    end

    print('Done parsing slice:  ' .. i)

    torch.save('./trainvalFinal/trainvalDataFinal' .. i .. '.t7', trainData)

    print('Done saving slice: ' .. i .. '\n')
end
