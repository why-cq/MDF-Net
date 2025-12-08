class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        # 这里是我自己模型的输入,需要修改
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


class ImageCaptionActivationsAndGradients:
    """ 图像描述模型提取目标层激活和配准梯度 """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        # self.reshape_transform = reshape_transform
        self.handles = []
        # target_layers表示需要进行提取的层,我们的任务中提取多头注意力中的层,注册hook函数
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        # if self.reshape_transform is not None:
        #     activation = self.reshape_transform(activation)
        # self.activations.append(activation.cpu().detach())
        if output.shape[1] > 1:
            activation = self.reshape_transform(activation)
        else:
            activation = self.reshape_transform_decoder(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        # def _store_grad(grad):
        #     if self.reshape_transform is not None:
        #         grad = self.reshape_transform(grad)
        #     self.gradients = [grad.cpu().detach()] + self.gradients
        def _store_grad(grad):
            if grad.shape[1] > 1:
                grad = self.reshape_transform(grad)
            else:
                grad = self.reshape_transform_decoder(grad)

            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, **kwargs):
        self.gradients = []
        self.activations = []
        seq, logit = self.model.module.decode(**kwargs)
        return seq,logit

    def release(self):
        for handle in self.handles:
            handle.remove()

    def reshape_transform(self,x, height=16, width=16):
        result = x[:, 1:, :].reshape(x.size(0),
                                          height, width, x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    # todo: 这里的transform是针对于解码器而言,解码输出的feature是正对于每一个token进行的注意力
    # 输入x为(B,1,512)应该将其转换为(B,512,1,1)
    def reshape_transform_decoder(self,x,height=1, width=1):
        result = x[:, :, :].reshape(x.size(0),
                                     height, width, x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        return result
