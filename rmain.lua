data = require 'data'
dir = './data/'--'/home/arun/simple/'

dataH, dataW = 120,240

--print('creating data.t7..')
--X,Y = data.storeXY(dir,dataH,dataW,'captchaImage.')

print('Loading images...')
X,Y = data.loadXY(dir)
Xt,Yt,Xv,Yv = data.split(X,Y,50)

rutil = require 'rutil'
net = rutil.model(dataH,dataW)
rnn,ct = rutil.getNetCt(net)
rnn = rnn:cuda()
if arg[1] then rnn = torch.load(arg[1]) end
ct = ct:cuda()
tnet = nn.SplitTable(2,2):cuda()
batchSize = 24

epoches=10
for epoch=1,epoches do
  local lrate = 0.1/(10*epoch)
  print('Epoch:' .. epoch .. '; Learning rate:' .. lrate)
  rutil.train(rnn,ct,Xt,Yt,Xv,Yv,1,batchSize,tnet,lrate)
  print('saving model..')
  torch.save('trained.t7', rnn)
  torch.save('tnet.t7', tnet)
end
