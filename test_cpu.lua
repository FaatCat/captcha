function output_to_label(output)
   local len = #output
   --print(output)
   local label = ""
   for i=1, len do
      local char = output[i]
      if char<=10 then
         label = label .. string.char(char - 1 + string.byte('0'))
      else
         label = label .. string.char(char - 10 - 1 + string.byte('A'))
      end
   end
   return label
end

require 'MultiCrossEntropyCriterion'
--require 'cutorch'
--require 'cunn'
--require 'cudnn'
require 'nn'
require 'rnn'

model_filename = arg[2] or 'trained_cpu.t7'
model = torch.load(model_filename)
--ct = torch.load('ct.t7')

model:evaluate()
--model:cuda()

require 'image'


local batch = image.load(arg[1]):float()--:cuda()

out = model:forward(batch)

local outseq = {}

for i=1, #out do
   confidence, index = out[i]:max(2)
   outseq[i] = index:double()[1][1]
end

outseq = output_to_label(outseq)
--print('Output: ')
print(outseq)
