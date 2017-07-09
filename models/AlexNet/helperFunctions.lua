require 'nn'

helper = {}


function helper.groupedConvolution(inC, outC, W, H, kw, kh, dw, dh, pw, ph, group)
    local net = nn.Sequential()
    
    net:add(nn.View(group, inC/group, W, H))
    net:add(nn.SplitTable(1))
    
    groupedConv = nn.MapTable(nn.SpatialConvolution(inC/group, outC, kw, kh, dw, dh, pw, ph))
    groupedConv:resize(group)
    net:add(groupedConv)
    
    --net:add(nn.View(group, inC/group, W, H))
    --net:add(nn.SplitTable(1))
    net:add(nn.CAddTable())
    
    return net
end

function helper.computeOutputDims(W,H,kw,kh,dw,dh,pw,ph,ceil)
    ceil = ceil or nil
    
    outW = (W + 2*pw - kw)/dw + 1
    outH = (H + 2*ph - kh)/dh + 1

    if((W + 2*pw - kw)%dw > pw or ceil) then
        outW = math.ceil(outW)
    else
        outW = math.floor(outW)
    end

    if((H + 2*ph - kh)%dh > ph or ceil) then
        outH = math.ceil(outH)
    else
        outH = math.floor(outH)
    end

    return outW, outH
end

function helper.computeOutputDims_Deconvolution(W,H,kw,kh,dw,dh,pw,ph)
    outW = (W - 1)*dw - 2*pw + kw
    outH = (H - 1)*dh - 2*ph + kh
    
    return outW, outH
end


return helper