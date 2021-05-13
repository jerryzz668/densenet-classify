import torch
import torchvision
import densenet as dn

# 读入之前训练好的.pth模型
# model = dn.DenseNet3(20, 2, 16, reduction=0.5,
#                          bottleneck=True, dropRate=0.2, small_inputs = False)
model = torch.load('D:\\net\\densenet-pytorch-master\\0018_class_20200730.pth')
# model.load_state_dict(checkpoint['state_dict'])
# model.load_state_dict(state['model'], strict=True)

# example = torch.rand(1, 3, 128, 128).cuda()
# model.to(device)

# torch_out = torch.onnx.export(model,
#                               example,
#                               "new-mobilenetv2-128_S.onnx",
#                               verbose=True,
#                               export_params=True
#                               )

example = torch.ones(1, 3, 224, 224).cuda()

model = model.eval()
model = model.cuda()
traced_script_module = torch.jit.trace(model, example)
output = traced_script_module(example)
print(traced_script_module)
# 导出trace后的模型
traced_script_module.save("D:\\net\\densenet-pytorch-master\\pth2pt\\0018_class_20200730.pt")





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