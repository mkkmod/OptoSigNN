import torch
import torch.nn as nn
import onnx
import onnxruntime as rt


class MinimalModel(nn.Module):
    def __init__(self, inp_size, out_size):
        super(MinimalModel, self).__init__()
         
        self.linear = nn.Linear(inp_size, out_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # .double() to ZŁO
        #outp = self.linear(x).double()
        outp = self.linear(x)
        return self.sig(outp)

        

torch_model = MinimalModel(9, 1)
dummy_input = torch.randn(9)

torch_model(dummy_input)

savname_onnx = "minimalModel.onnx"

torch.onnx.export(torch_model, dummy_input, savname_onnx, opset_version=11)
 
sess = rt.InferenceSession(savname_onnx)



print("done")