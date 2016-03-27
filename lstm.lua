dataH, dataW = 120, 240
seqLen = 10
batchSizw = 16
epoches = 3

data = require 'data'

dir = './data/'

print('Loading data..')
X,Y = data.loadXY(dir)
print('Converting targets..')
data.convert(Y, seqLen)

function repl(N)
    local net = nn.ConcatTable()
    for i=1,N do 
        net:add(nn.Identity())
    end
    return net
end

print('Creating model..')
require 'nn'
require 'rnn'
net = nn.Sequential()
net:add(nn.Reshape(1,dataH,dataW))
net:add(nn.SpatialConvolution(1,64,3,3,1,1,1,1))
net:add(nn.SpatialBatchNormalization(64,1e-3))
net:add(nn.ReLU(true))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
net:add(nn.SpatialBatchNormalization(64,1e-3))
net:add(nn.ReLU(true))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
net:add(nn.SpatialBatchNormalization(64,1e-3))
net:add(nn.ReLU(true))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
net:add(nn.SpatialBatchNormalization(64,1e-3))
net:add(nn.ReLU(true))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
net:add(nn.SpatialBatchNormalization(64,1e-3))
net:add(nn.ReLU(true))
net:add(nn.SpatialMaxPooling(2,2,2,2))
--net:add(nn.View(64*1*6))
net:add(nn.Reshape(8,168))
net:add(nn.SplitTable(2,3))
enc = nn.Sequential()
enc:add(net)
enc:add(nn.Sequencer(nn.LSTM(168, 168)))
enc:add(nn.SelectTable(-1))
dec = nn.Sequential()
dec:add(enc)
dec:add(repl(10))

mlp = nn.Sequential()
       :add(nn.LSTM(168, 168))
       :add(nn.Linear(168, 36))
       :add(nn.LogSoftMax())

dec:add(nn.Sequencer(mlp))
--net:add(repl(10))

require 'cunn';
require 'nn';
require 'rnn';

--[[
print('Loading dec model..')
dec = torch.load('lstm_dec.t7')
]]--
dec = dec:cuda()

criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
criterion = criterion:cuda()

rutil = require 'rutil'

Xt,Yt,Xv,Yv = data.split(X,Y,50)

tnet = nn.SplitTable(2,2):cuda()

--[[
a = rutil.valid(dec,Xv,Yv,512,tnet)
print(a)
]]--


function rutil.kfacc(outputs,targets)
    local Y,y = nil,nil;
    local N = outputs[1]:size(1)
    local C = outputs[1]:size(2)
    for k=1,#outputs do 
        Y = Y and torch.cat(Y,outputs[k]:reshape(N,1,C),2) or outputs[k]:reshape(N,1,C)
        y = y and torch.cat(y,targets[k]:reshape(N,1),2) or targets[k]:reshape(N,1)
    end
    local t,idx = Y:max(3)
    return idx:squeeze():eq(y):sum(2):eq(#outputs):sum()
end

function rutil.kvalid(rnn,Xv,Yv,batchSize,tnet)
    local batchSize = batchSize or 16
    local acc = 0
    local acci = {}
    local Nv = Xv:size(1)
    rnn:evaluate()
    for i=1,Nv,batchSize do 
        xlua.progress(i/batchSize, Nv/batchSize)
        local j = math.min(Nv,i+batchSize-1)
        local Xb = Xv[{{i,j}}]:cuda()            
        local Yb = Yv[{{i,j}}]:cuda()
        local inputs = Xb
        local targets = tnet:forward(Yb)
        local outputs = rnn:forward(inputs)
        local aa,ai = rutil.kfacc2(outputs,targets)
        acc = acc + aa
        rnn:forget()
    end
    return (acc*100)/Nv,acci
end


for epoch=1, epoches do
 print('Epoch: ' .. epoch)
 rutil.train(dec,criterion,Xt,Yt,Xv,Yv,1,batchSize,tnet,0.01)
 torch.save('lstm_dec.t7',dec)
 torch.save('lstm_crit.t7',criterion)
 torch.save('tnet.t7',tnet)
end
