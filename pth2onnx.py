import torch
import torchvision
# import densenet as dn
import torch.nn as nn
import models.densenet as dn
import torch.nn.functional as F
class densenet(nn.Module):
    def __init__(self, depth, num_classes, growth_rate=12,
                 reduction=0.5, bottleneck=True, dropRate=0.0,small_inputs = False):
        super(densenet, self).__init__()

        self.avgpool_size=2
        model = dn.DenseNet3(depth, num_classes, growth_rate, reduction,
                            bottleneck, dropRate, small_inputs)
        checkpoint = torch.load("/home/adt/project_model/densenet-pytorch-master/runs2/DenseNet_Unet_fs/D_20200323_model_best.pth")
        model.load_state_dict(checkpoint['state_dict'])
        self.in_planes = model.in_planes
        self.extractor=nn.Sequential(model.conv1,model.block1,model.trans1,model.block2,model.trans2,model.block21,model.trans21,model.block3,model.bn1,model.relu)
        self.fc=nn.Sequential(model.fc)
    def forward(self, x):

        out = self.extractor(x)
        out = F.avg_pool2d(out, kernel_size = self.avgpool_size)
        # out = F.avg_pool2d(out, kernel_size = self.avgpool_size)
        #pdb.set_trace()
        out = out.view(-1, self.in_planes)
        out = self.fc(out)
        return out
model = dn.DenseNet3(32, 7, 16, reduction=0.5,
                      bottleneck=True, dropRate=0.2, small_inputs=False)
# checkpoint = torch.load("/home/adt/project_model/densenet-pytorch-master/runs8/DenseNet_Unet_fs/D_20200323_model_best.pth")
# print("checkpoint:::",checkpoint['state_dict'])
# model.load_state_dict(checkpoint['state_dict'])
#model=densenet(32, 7, 16, reduction=0.5,
#                         bottleneck=True, dropRate=0.2, small_inputs = True)

example = torch.ones(1, 3, 64, 64).cuda()

model = model.eval()
model = model.cuda()
# traced_script_module = torch.jit.trace(model, example)
# output = traced_script_module(example)
# print(traced_script_module)
# 导出trace后的模型
torch.onnx.export(model, example,'/home/adt/project_model/densenet-pytorch-master/runs2/DenseNet_Unet_fs/model_onnx.onnx', verbose=True, opset_version=11)








# # 读入之前训练好的.pth模型
# print('Torch Version: ', torch.__version__)
# model = torch.load('E:\\DLNetwork\\densenet_pytorch_master\\D_20200323_CP0.pth')
# # model.load_state_dict(state['model'], strict=True)

# # example = torch.rand(1, 3, 128, 128).cuda()
# # model.to(device)

# # torch_out = torch.onnx.export(model,
# #                               example,
# #                               "new-mobilenetv2-128_S.onnx",
# #                               verbose=True,
# #                               export_params=True
# #                               )

# example = torch.ones(1, 3, 200, 200).cuda()

# model = model.eval()
# model = model.cuda()
# # traced_script_module = torch.jit.trace(model, example)
# # output = traced_script_module(example)
# # print(traced_script_module)
# # 导出trace后的模型
# torch.onnx.export(model, example,'E:\\DLNetwork\\densenet_pytorch_master\\pth2pt\\model_onnx.onnx', verbose=True)
# # traced_script_module.save("E:\\DLNetwork\\densenet_pytorch_master\\pth2pt\\model_D_class_20190916.pt")





# net = UNet(n_channels=3, n_classes=1)

# net = net.cuda()
# net.load_state_dict(torch.load("C:\\Users\\fs\\Desktop\\Pytorch-UNet-master\\CP174.pth"),strict=False)
# net = net.eval()
# example = torch.rand(1, 3, 1100, 1100).cuda()


# traced_script_module = torch.jit.trace(net, example)
# output = traced_script_module(example)
# print(traced_script_module)
# # 导出trace后的模型
# traced_script_module.save("C:\\Users\\fs\\Desktop\\test2\\model.pt")