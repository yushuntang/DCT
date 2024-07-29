from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import math
import numpy as np
import timm
# from models.vision_transformer import _load_weights
# from helpers import adapt_input_conv

class ViTAttentionGet:
    """ViTAttentionGet: get the attention map

    Args:
        model (nn.Module): the model
        attention_layer_name (str, optional): the name of the attention layer. Defaults to 'attn_drop'.
        discard_ratio (float, optional): the ratio of the lowest attentions to be discarded. Defaults to 0.9.
    """
    def __init__(self, model, attention_layer_name='attn_drop'):
        self.model = model
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)

        self.attentions = []

    def get_attention(self, module, input, output):
        self.attentions.append(output)


    def __call__(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)

        return self.attentions

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss



class Attention(timm.models.vision_transformer.Attention):
    def __init__(self, dim=768, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__(dim=768)

        print('Define Attention...')
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.local_attention_scale = nn.Parameter(torch.zeros(num_heads, 1, 1), requires_grad=True)
        # nn.init.constant_(self.local_attention_scale, 1.)
        distance_matrix = compute_distance_matrix(16, 196, 14)
        mask = 1-torch.tensor(distance_matrix)/distance_matrix.max()
        rows, cols = mask.shape
        mask = torch.cat((torch.zeros(rows, 1), mask), dim=1)
        self.mask = torch.cat((torch.zeros(1, cols+1), mask), dim=0).cuda()

    def forward(self, x):
        # x = x[0]
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        batch_mask = (self.local_attention_scale*self.mask).unsqueeze(0).expand(64, -1, -1, -1)
        # print('batch_mask', batch_mask)
        # attn += batch_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class One_shot_train(nn.Module):
    def __init__(self, one_shot_model=None, oneshot_val_loader=None, optimizer=None, args=None):
        super().__init__()
        self.model = one_shot_model
        self.dataloader  = oneshot_val_loader
        self.optimizer = optimizer
        self.args = args
        # self.criterion = nn.CrossEntropyLoss()
        # label smoothing
        if args.use_label_smoothing:
            self.criterion = CrossEntropyLabelSmooth(num_classes=1000, epsilon=0.1)
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.num_epochs = args.oneshot_epochs
        self.attention_weights = None

        # self.custom_attention = Attention().cuda()
        # self.model.blocks[0].attn = self.custom_attention
        # self.model.blocks[0].attn.register_forward_hook(lambda m, inp, out: self.custom_attention(inp))
        # print(self.model)
        # load_weights(self.model, checkpoint_path='/home/nkd/TYS/SAR_nips/models/vit_base_patch16_224.npz')

    def collect_params(self):
        """Collect the affine scale + shift parameters from norm layers.
        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        # for np, p in self.custom_attention.named_parameters():
        #     p.requires_grad_(True)
        #     params.append(p)
        #     names.append(f"{np}")

        for nm, m in self.model.named_modules():
            # print(nm)
            # if isinstance(m, timm.models.vision_transformer.Block):
            #     for np, p in m.named_parameters():
            #             p.requires_grad_(True)
            #             params.append(p)
            #             names.append(f"{nm}.{np}")
            # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
            # if 'layer4' in nm:
            #     continue
            # if 'blocks.9' in nm:
            #     continue
            # if 'blocks.10' in nm:
            #     continue
            # if 'blocks.11' in nm:
            #     continue
            # if 'norm.' in nm:
            #     continue
            # if nm in ['norm']:
            #     continue
            # if 'blocks.0' in nm:
            if self.args.add_LN:
                if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                    for np, p in m.named_parameters():
                        if np in ['weight', 'bias']:  # weight is scale, bias is shift
                            # params.append(p)
                            params += [{'params': p, 'lr': self.args.LN_lr}]
                            names.append(f"{nm}.{np}")
            if self.args.add_prompt_token:
                if isinstance(m, timm.models.vision_transformer.Attention):
                    for np, p in m.named_parameters():
                        if 'prompt_embeddings' in np or 'noise_generator' in np:
                            # print('add prompt_embeddings to optimizer:',f"{np}: {p.shape}")
                            p.requires_grad_(True)
                            params += [{'params': p, 'lr': self.args.prompt_token_lr}]
                            names.append(f"{nm}.{np}")
            if self.args.add_head:
                for np, p in m.named_parameters():
                    if 'head' in np:
                        print('add head to optimizer:',f"{np}: {p.shape}")
                        p.requires_grad_(True)
                        # params.append(p)
                        params += [{'params': p, 'lr': self.args.cls_head_lr}]
                        names.append(f"{nm}.{np}")
            # if 'blocks.0' in nm and isinstance(m, timm.models.vision_transformer.Attention):
            #     for np, p in m.named_parameters():
            #         p.requires_grad_(True)
            #         params.append(p)
            #         names.append(f"{nm}.{np}")
            # if isinstance(m, timm.models.vision_transformer.Attention):
            #     for np, p in m.named_parameters():
            #         # print(np)
            #         if 'local_attention_scale' in np:
            #             p.requires_grad_(True)
            #             # params.append(p)
            #             params += [{'params': p, 'lr': 0.1}]
            #             names.append(f"{nm}.{np}")
            # for np, p in m.named_parameters():
            #     if 'local_attention_scale' in np:
            #         p.requires_grad_(True)
            #         params.append(p)
            #         names.append(f"{nm}.{np}")

            
            
            # for np, p in m.named_parameters():
            #     if 'cls_token' in np:
            #         print('finding token:',f"{np}: {p.shape}")
            #         p.requires_grad_(True)
            #         params.append(p)
            #         # params += [{'params': p, 'lr': args.cls_token_lr}]
            #         names.append(f"{nm}.{np}")
            # if nm == 'patch_embed.proj':
            #     m.train()
            #     m.requires_grad_(True)
            #     for np, p in m.named_parameters():
            #         # params.append(p)
            #         params += [{'params': p, 'lr': 1e-6}]
            #         names.append(f"{nm}.{np}")

        return params, names
    
    def train(self, args):
        self.model.train()
        # self.model.head.bias.data.zero_()
        # 注册钩子函数
        # get_attention_weights_hook = self.model.blocks[0].attn.attn_drop.register_forward_hook(self.get_attention_weights)

        for epoch in range(self.num_epochs):

            for inputs, labels in self.dataloader:
                inputs = inputs.cuda()
                labels = labels.cuda()

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                # outputs, noise = self.model(inputs)
                # print('self.local_attention_scale parameter values:', self.custom_attention.local_attention_scale.grad)
                # mean_distances = compute_mean_attention_dist(16, self.attention_weights.mean(0), num_cls_tokens=1)
                # print('mean_distances', mean_distances)
                # print('mean_distances', mean_distances.min())
                loss = self.criterion(outputs, labels)
                # loss = self.criterion(outputs, labels) - args.lambda_noise_entropy*softmax_entropy(noise).mean()
                # loss = self.criterion(outputs, labels) - 0.1*mean_distances.min().cuda()
                loss.backward()
                # loss.backward(retain_graph=True)
                self.optimizer.step()

                # running_loss += loss.item()
                # _, predicted = outputs.max(1)
                # total += labels.size(0)
                # correct += predicted.eq(labels).sum().item()

            # epoch_loss = running_loss / len(self.dataloader)
            # epoch_acc = correct / total

            # print(f'Epoch [{epoch+1}/{self.num_epochs}]\tLoss: {epoch_loss:.4f}\tAccuracy: {epoch_acc:.4f}')

        print('One-shot Training complete!')
        try:
            local_attention_scale = []
            for i in range(len(self.model.blocks)):
                local_attention_scale.append(self.model.blocks[i].attn.local_attention_scale)
            print('local_attention_scale parameter values:', local_attention_scale)
            # print('model.blocks[0].local_attention_scale parameter values:', self.model.blocks[0].attn.local_attention_scale)
            # print('model.blocks[-1].local_attention_scale parameter values:', self.model.blocks[-1].attn.local_attention_scale)
        except:
            pass
        # get_attention_weights_hook.remove()

        return self.model

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)
        print(f'Model saved to {save_path}')

    def get_attention_weights(self, module, input, output):
        # global self.attention_weights
        self.attention_weights = output

    def get_features_matrix(self):
        features_matrix = self.model.head.weight.data.clone().detach()
        self.model.eval()
        with torch.no_grad():
            all_features = None
            for inputs, labels in self.dataloader:
                inputs = inputs.cuda()
                labels = labels.cuda()

                feature = self.model.forward_features(inputs)[:, 0]
                output = self.model.head(feature)
                if all_features is None:
                    all_features = feature
                    all_labels = labels
                    all_outputs = output
                else:
                    all_features = torch.cat((all_features, feature), 0)
                    all_labels = torch.cat((all_labels, labels), 0)
                    all_outputs = torch.cat((all_outputs, output), 0)

            all_predictions = all_outputs.argmax(1)
            sorted_all_predictions = all_predictions[torch.argsort(all_labels)]
            # print('sorted_all_predictions', sorted_all_predictions)
            # print('accuracy', (sorted_all_predictions == torch.sort(all_labels)[0]).sum().item() / len(all_labels))
            # print('accuracy', (all_predictions == all_labels).sum().item() / len(all_labels))
            all_features = all_features[torch.argsort(all_labels)]
            correct_features = all_features[sorted_all_predictions == torch.sort(all_labels)[0]]
            # features_matrix[sorted_all_predictions == torch.sort(all_labels)[0]] = correct_features

        return features_matrix, all_labels


def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data


class SAR(nn.Module):
    """SAR online adapts a model by Sharpness-Aware and Reliable entropy minimization during testing.
    Once SARed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, margin_e0=0.4*math.log(1000), reset_constant_em=0.2):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "SAR requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.margin_e0 = margin_e0  # margin E_0 for reliable entropy minimization, Eqn. (2)
        self.reset_constant_em = reset_constant_em  # threshold e_m for model recovery scheme
        self.ema = None  # to record the moving average of model output entropy, as model recovery criteria

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs, ema, reset_flag = forward_and_adapt_sar(x, self.model, self.optimizer, self.margin_e0, self.reset_constant_em, self.ema)
            if reset_flag:
                self.reset()
            self.ema = ema  # update moving average value of loss

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.ema = None


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_sar(x, model, optimizer, margin, reset_constant, ema):
    """Forward and adapt model input data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    optimizer.zero_grad()
    # forward
    outputs = model(x)
    # adapt
    # filtering reliable samples/gradients for further adaptation; first time forward
    entropys = softmax_entropy(outputs)
    filter_ids_1 = torch.where(entropys < margin)
    entropys = entropys[filter_ids_1]
    loss = entropys.mean(0)
    loss.backward()

    optimizer.first_step(zero_grad=True) # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
    entropys2 = softmax_entropy(model(x))
    entropys2 = entropys2[filter_ids_1]  # second time forward  
    loss_second_value = entropys2.clone().detach().mean(0)
    filter_ids_2 = torch.where(entropys2 < margin)  # here filtering reliable samples again, since model weights have been changed to \Theta+\hat{\epsilon(\Theta)}
    loss_second = entropys2[filter_ids_2].mean(0)
    if not np.isnan(loss_second.item()):
        ema = update_ema(ema, loss_second.item())  # record moving average loss values for model recovery

    # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
    loss_second.backward()
    optimizer.second_step(zero_grad=True)

    # perform model recovery
    reset_flag = False
    if ema is not None:
        if ema < 0.2:
            print("ema < 0.2, now reset the model")
            reset_flag = True

    return outputs, ema, reset_flag


def SAR_collect_params(args, model):
    """Collect the affine scale + shift parameters from norm layers.
    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
        # if 'layer4' in nm:
        #     continue
        # if 'blocks.9' in nm:
        #     continue
        # if 'blocks.10' in nm:
        #     continue
        # if 'blocks.11' in nm:
        #     continue
        # if 'norm.' in nm:
        #     continue
        # if nm in ['norm']:
        #     continue

        # if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
        #     for np, p in m.named_parameters():
        #         if np in ['weight', 'bias']:  # weight is scale, bias is shift
        #             params.append(p)
        #             names.append(f"{nm}.{np}")
        # if args.add_LN:
        #     if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
        #         for np, p in m.named_parameters():
        #             if np in ['weight', 'bias']:  # weight is scale, bias is shift
        #                 # params.append(p)
        #                 params += [{'params': p, 'lr': 0.001}]
        #                 names.append(f"{nm}.{np}")
        if args.add_prompt_token:
            if isinstance(m, timm.models.vision_transformer.Attention):
                for np, p in m.named_parameters():
                    if 'prompt_embeddings' in np or 'noise_generator' in np:
                        # print('add prompt_embeddings to optimizer:',f"{np}: {p.shape}")
                        p.requires_grad_(True)
                        params += [{'params': p, 'lr': 0.001}]
                        names.append(f"{nm}.{np}")

    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with SAR."""
    # train mode, because SAR optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what SAR updates
    model.requires_grad_(False)
    # configure norm for SAR updates: enable grad + force batch statisics (this only for BN models)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with SAR."""
    is_training = model.training
    assert is_training, "SAR needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "SAR needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "SAR should not update all params: " \
                               "check which require grad"
    has_norm = any([isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)) for m in model.modules()])
    assert has_norm, "SAR needs normalization layer parameters for its optimization"



def compute_distance_matrix(patch_size, num_patches, length):
    """compute_distance_matrix: Computes the distance matrix for the patches in the image

    Args:
        patch_size (int): the size of the patch
        num_patches (int): the number of patches in the image
        length (int): the length of the image

    Returns:
        distance_matrix (np.ndarray): The distance matrix for the patches in the image
    """
    distance_matrix = np.zeros((num_patches, num_patches))
    for i in range(num_patches):
        for j in range(num_patches):
            if i == j:  # zero distance
                continue

            xi, yi = (int(i / length)), (i % length)
            xj, yj = (int(j / length)), (j % length)
            distance_matrix[i, j] = patch_size * np.linalg.norm([xi - xj, yi - yj])

    return distance_matrix


def compute_mean_attention_dist(patch_size, attention_weights, num_cls_tokens=1):
    """compute_mean_attention_dist: Computes the mean attention distance for the image

    Args:
        patch_size (int): the size of the patch
        attention_weights (np.ndarray): The attention weights for the image
        num_cls_tokens (int, optional): The number of class tokens. Defaults to 1.

    Returns:
        mean_distances (np.ndarray): The mean attention distance for the image
    """
    # The attention_weights shape = (batch, num_heads, num_patches, num_patches)
    attention_weights = attention_weights[
        ..., num_cls_tokens:, num_cls_tokens:
    ]  # Removing the CLS token
    num_patches = attention_weights.shape[-1]
    length = int(np.sqrt(num_patches))
    assert length ** 2 == num_patches, "Num patches is not perfect square"

    distance_matrix = compute_distance_matrix(patch_size, num_patches, length)
    h, w = distance_matrix.shape

    distance_matrix = torch.tensor(distance_matrix.reshape((1, 1, h, w))).cuda()
    # The attention_weights along the last axis adds to 1
    # this is due to the fact that they are softmax of the raw logits
    # summation of the (attention_weights * distance_matrix)
    # should result in an average distance per token
    mean_distances = attention_weights * distance_matrix
    mean_distances = torch.sum(
        mean_distances, axis=-1
    )  # sum along last axis to get average distance per token
    mean_distances = torch.mean(
        mean_distances, axis=-1
    )  # now average across all the tokens

    return mean_distances

