import torch
import random
import numpy as np
# from projection import *
import math
import os
import model.mae as mae
import torch.nn.functional as F
import math

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def distance(model_a,model_b):
    model_a = model_a.cpu()
    model_b = model_b.cpu()
    param_list_a = [param.detach() for param in model_a.parameters()]
    param_list_b = [param.detach() for param in model_b.parameters()]
    d_a = tensorlist_2_tensor(param_list_a)
    d_b = tensorlist_2_tensor(param_list_b)
    return torch.dist(d_a,d_b).item()

def model_2_tensor(model:torch.nn.Module)->torch.Tensor:
    param_list = [param for param in model.parameters()]
    return tensorlist_2_tensor(param_list)

def tensor_2_tensorlist(tensor:torch.Tensor, model:torch.nn.Module) -> list[torch.Tensor]:
    params = get_weights(model)
    shapes = [p.shape for p in params]

    # 根据 params 的形状拆分原始 Tensor
    tensor_slices = torch.split(tensor, [torch.prod(torch.tensor(s)).detach().clone() for s in shapes])

    # 将拆分后的 Tensor 放入列表
    tensor_list = [slice.view(shape) for slice, shape in zip(tensor_slices, shapes)]

    return tensor_list

def tensorlist_2_tensor(tensor_list:list[torch.Tensor])->torch.Tensor:
    """ Concatnate a list of tensors into one tensor.

        Args:
            weights: a list of parameter tensors, e.g. net_plotter.get_weights(net).

        Returns:
            concatnated 1D tensor
    """
    return torch.cat([w.view(w.numel()).cpu() if w.dim() > 1 else torch.FloatTensor(w.cpu()) for w in tensor_list])

def rescale_direction(direction:list[torch.Tensor],scale:float) -> list[torch.Tensor]:
    direction_norm = math.sqrt(dot_list_tensor(direction,direction))
    rescaled_direction = [d * (scale/direction_norm) for d in direction]
    return rescaled_direction


def get_weights(net:torch.nn.Module)->list[torch.Tensor]:
    return [p.data for p in net.parameters() if p.requires_grad ]

def rescale(importance:list[torch.Tensor],scale:float)->list[torch.Tensor]:
    # norm = torch.norm(tensorlist_2_tensor(importance))

    sum_value = sum([torch.sum(d) for d in importance])
    # return rescaled importance
    return [d * (scale/sum_value) for d in importance]

def sum_list_tensor(list_tensor:list[torch.Tensor])->float:
    return sum([torch.sum(t) for t in list_tensor])

def set_importance_2_zero(importance:list[torch.Tensor]):
    return [torch.zeros_like(imp) for imp in importance]

def rescale_importance_by_contribution(importance:list[torch.Tensor], loss_contribution:float, path_contribution:float, tau_1:float, tau_2:float)->list[torch.Tensor]:
    importance = rescale(importance, loss_contribution)
    scaling_factor = abs(1 / torch.mean(torch.cat([imp.flatten() for imp in importance])))
    scaled_loss_importance = [(imp * scaling_factor).clamp(min = -1.0, max = 3.0) for imp in importance]
    rescaled_importance = [torch.pow((1 + tau_1 * path_contribution), (1 + imp).clamp(min = 0.0)).clamp(max = 100.0) for imp in scaled_loss_importance]
    return rescaled_importance

#TODO all different
def compute_stochastic_gradient(model:torch.nn.Module,inputs,targets,criterion)->list[torch.Tensor]:
    """
    Computes the stochastic gradient of the model with respect to the given inputs and targets.

    Args:
        model (nn.Module): The PyTorch model to compute the gradient for.
        inputs (torch.Tensor): The input data of shape (batch_size, *input_shape).
        targets (torch.Tensor): The target data of shape (batch_size, *target_shape).

    Returns:
        list[torch.Tensor]: A list of gradients for each parameter in the model.
    """
    # Ensure the model is in training mode
    model.train()
    # Zero out any existing gradients
    model.zero_grad()
    # Forward pass
    outputs = model(inputs)
    # Compute the loss
    loss = torch.nn.functional.cross_entropy(outputs, targets)
    # Backward pass (compute gradients)
    loss.backward()
    # Collect gradients
    gradients = [param.grad.clone().detach() for param in model.parameters() if param.grad is not None]
    
    return gradients
def compute_stochastic_gradient_moco(model:torch.nn.Module,inputs,r_targets,criterion)->list[torch.Tensor]:
    """
    Computes the stochastic gradient of the model with respect to the given inputs and targets.

    Args:
        model (nn.Module): The PyTorch model to compute the gradient for.
        inputs (torch.Tensor): The input data of shape (batch_size, *input_shape).
        targets (torch.Tensor): The target data of shape (batch_size, *target_shape).

    Returns:
        list[torch.Tensor]: A list of gradients for each parameter in the model.
    """
    # Ensure the model is in training mode
    model.train()
    # Zero out any existing gradients
    model.zero_grad()
    # Forward pass
    outputs,targets = model(im_q=inputs[0], im_k=inputs[1])
    # Compute the loss
    loss = criterion(outputs, targets)
    # Backward pass (compute gradients)
    loss.backward()
    # Collect gradients
    gradients = [param.grad.clone().detach() for param in model.parameters() if param.grad is not None]
    
    return gradients

def compute_stochastic_gradient_byol(model:torch.nn.Module,inputs)->list[torch.Tensor]:
    """
    Computes the stochastic gradient of the model with respect to the given inputs and targets.

    Args:
        model (nn.Module): The PyTorch model to compute the gradient for.
        inputs (torch.Tensor): The input data of shape (batch_size, *input_shape).
        targets (torch.Tensor): The target data of shape (batch_size, *target_shape).

    Returns:
        list[torch.Tensor]: A list of gradients for each parameter in the model.
    """
    # Ensure the model is in training mode
    model.train()
    # Zero out any existing gradients
    model.zero_grad()
    # Forward pass
    loss= model(inputs[0], inputs[1])
    # Compute the loss

    # Backward pass (compute gradients)
    loss.backward()
    # Collect gradients
    gradients = [param.grad.clone().detach() for param in model.parameters() if param.grad is not None]
    
    return gradients
def compute_stochastic_gradient_simclr(model:torch.nn.Module,inputs,args,criterion)->list[torch.Tensor]:
    """
    Computes the stochastic gradient of the model with respect to the given inputs and targets.

    Args:
        model (nn.Module): The PyTorch model to compute the gradient for.
        inputs (torch.Tensor): The input data of shape (batch_size, *input_shape).
        targets (torch.Tensor): The target data of shape (batch_size, *target_shape).

    Returns:
        list[torch.Tensor]: A list of gradients for each parameter in the model.
    """
    # Ensure the model is in training mode
    model.train()
    # Zero out any existing gradients
    model.zero_grad()
    # Forward pass
    features = model(inputs)
    logits, labels = info_nce_loss(features,args)
    loss = criterion(logits, labels)
    # Compute the loss

    # Backward pass (compute gradients)
    loss.backward()
    # Collect gradients
    gradients = [param.grad.clone().detach() for param in model.parameters() if param.grad is not None]
    
    return gradients

def add_delta_param(model:torch.nn.Module, delta_param: list[torch.Tensor], alpha:float) -> torch.nn.Module:
    '''
    add a direction update on the model network
    Args:
        model(nn.Module): The PyTorch model.
        delta_param(list[torch.Tensor]):  list of tensors representing the direction to update the network weights
        alpha: scale for the update
    Returns(nn.Module): the updated network
    '''
    # Ensure that the number of tensors in delta_param matches the model's parameters
    if len(delta_param) != len([p for p in model.parameters() if p.requires_grad]):
        raise ValueError("The length of delta_param does not match the number of parameters in the model.")
    
    # Iterate over the model's parameters and the provided delta_param list and update each parameter
    with torch.no_grad():  # Disable gradient computation
        i = 0
        for param in model.parameters():
            if not param.requires_grad:
                continue
            else:
                delta = delta_param[i]
                if param.data.shape != delta.shape:
                    raise ValueError("The shape of the delta_param does not match the shape of the model's parameters.")
                # Add the scaled delta to the parameter
                param.data.add_(alpha * delta)
                i += 1
        # for param, delta in zip(model.parameters(), delta_param):
        #     if param is not None:
        #         # Verify if the shapes match
        #         if param.data.shape != delta.shape:
        #             raise ValueError("The shape of the delta_param does not match the shape of the model's parameters.")
        #         # Add the scaled delta to the parameter
        #         param.data.add_(alpha * delta)

    return model

def add_list_tensor(list_a:list[torch.Tensor],list_b:list[torch.Tensor],alpha = 1)->list[torch.Tensor]:
    """
    This function takes two lists of PyTorch tensors of the same length and returns a new list
    where each element is the sum of the corresponding elements in the input lists.
    Args:
    list_a: The first list of PyTorch tensors.
    list_b: The second list of PyTorch tensors.
    Returns:
    A new list of PyTorch tensors where each element is the sum of the corresponding 
    elements in the input lists.

    Raises:
    ValueError: If the input lists are not the same length.
    """
    if len(list_a) != len(list_b):
        raise ValueError("Input lists must have the same length.")
    return [(a + alpha * b).detach() for a, b in zip(list_a, list_b)]

def mul_list_tensor(list_a:list[torch.Tensor],list_b:list[torch.Tensor],alpha:float)->list[torch.Tensor]:
    
    if len(list_a) != len(list_b):
        print(len(list_a),len(list_b))
        raise ValueError("Input lists must have the same length.")
    return [(a * b * alpha).detach() for a, b in zip(list_a, list_b)]

def cons_mul_list_tensor(list:list[torch.Tensor],cons:float)->list[torch.Tensor]:
    return [(cons * d).detach() for d in list]

def dot_list_tensor(list_a:list[torch.Tensor],list_b:list[torch.Tensor])->float:
    assert equal_shape_list(list_a,list_b)
    dot_products = []
    for a, b in zip(list_a, list_b):
        if a.dim() == 1 and b.dim() == 1:
            dot_products.append(torch.dot(a, b).item())
        else:
            dot_products.append(torch.sum(a * b).item())
    return sum(dot_products)
    
def equal_shape_list(list_a:list[torch.Tensor],list_b:list[torch.Tensor])->bool:
    """
    Returns True if all tensors in list_a have the same shape as the corresponding
    tensor in list_b, and False otherwise.
    """
    if len(list_a) != len(list_b):
        return False
    for tensor_a,tensor_b in zip(list_a,list_b):
        if tensor_a.shape!= tensor_b.shape:
            return False
    return True

def update_ema_direction(ema_direction:list[torch.Tensor], delta_param:list[torch.Tensor], ema_factor:float)->list[torch.Tensor]:
    return add_list_tensor(cons_mul_list_tensor(ema_direction,ema_factor), cons_mul_list_tensor(delta_param,(1 - ema_factor)))

def get_projection_length(d:list[torch.Tensor],d_i:list[torch.Tensor])->float:
    '''
    calculate projection length on d_i

    Args:
    d(list[torch.Tensor]): direction vector(s)
    d_i(list[torch.Tensor]): information direction vector(s)
    
    Return:
    projection length
    '''
    # dot product
    dot_product = dot_list_tensor(d,d_i)
    norm_d_i = math.sqrt(dot_list_tensor(d_i,d_i))

    if norm_d_i == 0:
        return 0
    projection_length = dot_product / norm_d_i
    return projection_length

def compute_loss(args, model:torch.nn.Module, batch, criterion) -> float:
    '''
    compute loss value of model on a specific data

    Args:
    model : given model
    batch: specific data
    criterion: criterion to compute loss

    Return:
    loss value
    '''
    inputs, targets = batch
    inputs, targets = inputs.to(args.device), targets.to(args.device)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    return loss.item()

def comp_loss_batch(args, net, inputs, targets, criterion):
    '''
    compute the loss of model
    '''
    # net.eval()
    with torch.no_grad():
        outputs = net(inputs)
        criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
        loss = criterion(outputs,targets)
    return loss.item()
    # train_loss = 0
    # total = 0 
    # for batch_idx,(inputs, targets) in enumerate(train_loader):
    #     batch_size = inputs.shape[0]
    #     total += batch_size
    #     inputs, targets = inputs.to(args.device),targets.to(args.device)
    #     outputs = net(inputs)
    #     loss = criterion(outputs,targets)
    #     train_loss += loss.item() * batch_size
    # return train_loss/total
def comp_loss_batch_moco(args, net, inputs, criterion):
    '''
    compute the loss of model
    '''
    # net.eval()
    if args.gpu is not None:
        inputs[0] = inputs[0].cuda(args.gpu, non_blocking=True)
        inputs[1] = inputs[1].cuda(args.gpu, non_blocking=True)
    with torch.no_grad():
        output, target = net(im_q=inputs[0], im_k=inputs[1])
        loss = criterion(output, target)
    return loss.item()

def comp_loss_batch_byol(args, net, inputs):
    '''
    compute the loss of model
    '''
    # net.eval()
    if args.gpu is not None:
        inputs[0] = inputs[0].cuda(args.gpu, non_blocking=True)
        inputs[1] = inputs[1].cuda(args.gpu, non_blocking=True)
    with torch.no_grad():
        loss = net(inputs[0], inputs[1])

    return loss.item()
def comp_loss_batch_simclr(args, net, inputs,criterion):
    '''
    compute the loss of model
    '''
    # net.eval()
    inputs = torch.cat(inputs, dim=0)
    if args.gpu is not None:
        inputs = inputs.cuda(args.gpu, non_blocking=True)

    with torch.no_grad():
        features = net(inputs)
        logits, labels = info_nce_loss_sub(features,args)
        loss = criterion(logits, labels)

    return loss.item()


def info_nce_loss(features,args):

        labels = torch.cat([torch.arange(args.batch_size) for i in range(args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(features.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

        logits = logits / args.temperature
        return logits, labels

def info_nce_loss_sub(features,args):

        labels = torch.cat([torch.arange(args.sub_batch_size) for i in range(args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(features.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

        logits = logits / args.temperature
        return logits, labels
def save_importance(accumulated_importance: list[torch.Tensor], epoch: int, path: str):
    '''
    Save accumulated_importance as a file.

    Parameters:
    accumulated_importance (list[torch.Tensor]): List of importance tensors to save.
    epoch (int): The current epoch number, used to name the file.
    path (str): The directory path where the file will be saved.
    '''
    # Ensure the directory exists
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Construct the filename using the epoch number
    filename = f"{path}/importance_epoch_{epoch}.pt"
    
    # Save the accumulated importance list as a file
    torch.save(accumulated_importance, filename)
    
    print(f"Saved accumulated importance to {filename}")

def compute_importance(model:torch.nn.Module, gradients:list[torch.tensor], delta_param:list[torch.Tensor], inputs, targets, criterion) -> list[torch.Tensor]:
    
    # compute gradient at theta 0
    # gradients = compute_stochastic_gradient(model,inputs,targets,criterion)
    importance = mul_list_tensor(gradients, delta_param, -1/6)

    # with torch.no_grad():
    #     loss_before = criterion(model(inputs),targets)

    # compute gradient at theta 0.5
    model = add_delta_param(model,delta_param,alpha=0.5)
    gradients = compute_stochastic_gradient(model,inputs,targets,criterion)
    importance = add_list_tensor(importance,mul_list_tensor(gradients, delta_param, -4/6))

    # compute gradient at theta 1
    model = add_delta_param(model,delta_param,alpha=0.5)
    gradients = compute_stochastic_gradient(model,inputs,targets,criterion)
    importance = add_list_tensor(importance,mul_list_tensor(gradients, delta_param, -1/6))

    # with torch.no_grad():
    #     loss_after = criterion(model(inputs),targets)
        
    # print("delta loss: ", loss_before - loss_after, sum([torch.sum(imp) for imp in importance]))
    return importance
def compute_importance_moco(model:torch.nn.Module, gradients:list[torch.tensor], delta_param:list[torch.Tensor], inputs, targets, criterion) -> list[torch.Tensor]:
    
    # compute gradient at theta 0
    # gradients = compute_stochastic_gradient(model,inputs,targets,criterion)
    importance = mul_list_tensor(gradients, delta_param, -1/6)

    # with torch.no_grad():
    #     loss_before = criterion(model(inputs),targets)

    # compute gradient at theta 0.5
    model = add_delta_param(model,delta_param,alpha=0.5)
    gradients = compute_stochastic_gradient_moco(model,inputs,targets,criterion)
    importance = add_list_tensor(importance,mul_list_tensor(gradients, delta_param, -4/6))

    # compute gradient at theta 1
    model = add_delta_param(model,delta_param,alpha=0.5)
    gradients = compute_stochastic_gradient_moco(model,inputs,targets,criterion)
    importance = add_list_tensor(importance,mul_list_tensor(gradients, delta_param, -1/6))

    # with torch.no_grad():
    #     loss_after = criterion(model(inputs),targets)
        
    # print("delta loss: ", loss_before - loss_after, sum([torch.sum(imp) for imp in importance]))
    return importance


def compute_importance_byol(model:torch.nn.Module, gradients:list[torch.tensor], delta_param:list[torch.Tensor], inputs) -> list[torch.Tensor]:
    
    # compute gradient at theta 0
    # gradients = compute_stochastic_gradient(model,inputs,targets,criterion)
    importance = mul_list_tensor(gradients, delta_param, -1/6)

    # with torch.no_grad():
    #     loss_before = criterion(model(inputs),targets)

    # compute gradient at theta 0.5
    model = add_delta_param(model,delta_param,alpha=0.5)
    gradients = compute_stochastic_gradient_byol(model,inputs)
    importance = add_list_tensor(importance,mul_list_tensor(gradients, delta_param, -4/6))

    # compute gradient at theta 1
    model = add_delta_param(model,delta_param,alpha=0.5)
    gradients = compute_stochastic_gradient_byol(model,inputs)
    importance = add_list_tensor(importance,mul_list_tensor(gradients, delta_param, -1/6))

    # with torch.no_grad():
    #     loss_after = criterion(model(inputs),targets)
        
    # print("delta loss: ", loss_before - loss_after, sum([torch.sum(imp) for imp in importance]))
    return importance
def compute_importance_simclr(model:torch.nn.Module, gradients:list[torch.tensor], delta_param:list[torch.Tensor], inputs,args,criterion) -> list[torch.Tensor]:
    
    # compute gradient at theta 0
    # gradients = compute_stochastic_gradient(model,inputs,targets,criterion)
    importance = mul_list_tensor(gradients, delta_param, -1/6)

    # with torch.no_grad():
    #     loss_before = criterion(model(inputs),targets)

    # compute gradient at theta 0.5
    model = add_delta_param(model,delta_param,alpha=0.5)
    gradients = compute_stochastic_gradient_simclr(model,inputs,args,criterion)
    importance = add_list_tensor(importance,mul_list_tensor(gradients, delta_param, -4/6))

    # compute gradient at theta 1
    model = add_delta_param(model,delta_param,alpha=0.5)
    gradients = compute_stochastic_gradient_simclr(model,inputs,args,criterion)
    importance = add_list_tensor(importance,mul_list_tensor(gradients, delta_param, -1/6))

    # with torch.no_grad():
    #     loss_after = criterion(model(inputs),targets)
        
    # print("delta loss: ", loss_before - loss_after, sum([torch.sum(imp) for imp in importance]))
    return importance


def compute_stochastic_gradient_mae(args, model:mae.MaskedAutoencoderViT, inputs, ids_shuffle)->list[torch.Tensor]:
    model.train()
    model.zero_grad()
    loss,_,_,_ = model(inputs, mask_ratio = args.mask_ratio, ids_shuffle = ids_shuffle)
    # Backward pass (compute gradients)
    loss.backward()
    # Collect gradients
    gradients = [param.grad.clone().detach() for param in model.parameters() if param.grad is not None]
    
    return gradients
def compute_importance_mae(args, model:torch.nn.Module, gradients:list[torch.Tensor], delta_param:list[torch.Tensor], inputs, ids_shuffle)->list[torch.Tensor]:
    # compute gradient at theta 0
    importance = mul_list_tensor(gradients, delta_param, -1/6)

    model = add_delta_param(model,delta_param,alpha=0.5)
    gradients = compute_stochastic_gradient_mae(args, model,inputs,ids_shuffle)
    importance = add_list_tensor(importance,mul_list_tensor(gradients, delta_param, -4/6))

    # compute gradient at theta 1
    model = add_delta_param(model,delta_param,alpha=0.5)
    gradients = compute_stochastic_gradient_mae(args, model,inputs,ids_shuffle)
    importance = add_list_tensor(importance,mul_list_tensor(gradients, delta_param, -1/6))

    return importance

def compute_importance_dino(args, student, teacher, dino_loss, student_gradient, student_delta_param, images,epoch):
    # compute gradient at theta 0
    importance = mul_list_tensor(student_gradient, student_delta_param, -1/6)
    
    # compute gradient at theta 0.5
    student = add_delta_param(student, student_delta_param, alpha=0.5)
    student_gradient = compute_stochastic_gradient_dino(args,student,teacher,dino_loss,images, epoch)
    importance = add_list_tensor(importance, mul_list_tensor(student_gradient, student_delta_param, -4/6))
    
    # compute gradient at theta 1
    student = add_delta_param(student, student_delta_param, alpha=0.5)
    student_gradient = compute_stochastic_gradient_dino(args,student,teacher,dino_loss,images,epoch)
    importance = add_list_tensor(importance, mul_list_tensor(student_gradient, student_delta_param, -1/6))
    
    return importance
    

def compute_stochastic_gradient_dino(args, student, teacher, dino_loss, images,epoch):
    student.train()
    student.zero_grad()
    # teacher and student forward passes + compute dino loss
    teacher_output = teacher(images[:2])
    student_output = student(images)
    loss = dino_loss(student_output, teacher_output, epoch)
    
    loss.backward()
    # Collect gradients
    gradients = [param.grad.clone().detach() for param in student.parameters() if param.grad is not None]
    
    return gradients


def compute_stochastic_gradient_ibot(args, student_copy, teacher, ibot_loss, images, masks, epoch):
    student_copy.train()
    student_copy.zero_grad()
    
    # teacher and student forward passes + compute dino loss
    teacher_output = teacher(images[:args.global_crops_number])
    student_output = student_copy(images[:args.global_crops_number], mask=masks[:args.global_crops_number])
    
    student_copy.backbone.masked_im_modeling = False
    student_local_cls = student_copy(images[args.global_crops_number:])[0] if len(images) > args.global_crops_number else None
    student_copy.backbone.masked_im_modeling = args.use_masked_im_modeling
    
    all_loss = ibot_loss(student_output, teacher_output, student_local_cls, masks, epoch)
    loss = all_loss.pop('loss')
    loss.backward()
    # Collect gradients
    gradients = [param.grad.clone().detach() for param in student_copy.parameters() if param.grad is not None]
    return gradients


def compute_importance_ibot(args, student_copy, teacher, ibot_loss, student_gradient, student_delta_param, images, masks, epoch):
    # compute gradient at theta 0
    importance = mul_list_tensor(student_gradient, student_delta_param, -1/6)
    
    # compute gradient at theta 0.5
    student_copy = add_delta_param(student_copy, student_delta_param, alpha=0.5)
    student_gradient = compute_stochastic_gradient_ibot(args,student_copy,teacher,ibot_loss,images, masks, epoch)
    importance = add_list_tensor(importance, mul_list_tensor(student_gradient, student_delta_param, -4/6))
    
    # compute gradient at theta 1
    student_copy = add_delta_param(student_copy, student_delta_param, alpha=0.5)
    student_gradient = compute_stochastic_gradient_ibot(args,student_copy,teacher,ibot_loss,images, masks, epoch)
    importance = add_list_tensor(importance, mul_list_tensor(student_gradient, student_delta_param, -1/6))
    
    return importance

def checknan(importance: list[torch.Tensor]) -> list[torch.Tensor]:
    # 创建一个新列表，检查每个张量是否有 NaN
    result = []
    for tensor in importance:
        if torch.isnan(tensor).any():  # 如果张量中有 NaN
            result.append(torch.zeros_like(tensor))  # 用全零张量替换
        else:
            result.append(tensor)  # 没有 NaN，保留原张量
    return result